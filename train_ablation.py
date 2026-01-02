"""
train_ablation.py

Purpose
-------
Short ablation runs (default: 500 steps) on the SmolLM2-135M reverse-engineered architecture,
logging throughput, memory, and (optionally) *proof* of SDPA backend selection via:

  (2) torch.backends.cuda.sdp_kernel flags (flash / mem_efficient / math)
  (3) torch.profiler traces + top CUDA ops table

Outputs
-------
For each variant:
  - {out_dir}/logs/<variant>.log                    (human readable log)
  - {out_dir}/profiles/<variant>/trace_*.json       (optional Chrome trace, if --profile_sdpa)
  - {out_dir}/profiles/<variant>/top_ops.txt        (optional top CUDA ops table)

Aggregates:
  - {out_dir}/ablations.jsonl                       (one JSON per run)
  - {out_dir}/ablations.csv                         (tabular summary)

Notes
-----
- This file is for ablation studies only. It does NOT affect your main assignment training run (train.py).
- SDPA backend selection is dynamic in PyTorch; enabling flags does not guarantee Flash kernels will be used,
  but profiler evidence + backend flags provide strong documentation.
"""

from __future__ import annotations

import os
import json
import time
import math
import argparse
import datetime
from contextlib import nullcontext

import torch
import torch.optim as optim
from transformers import AutoTokenizer

from smollm2_config import SmolLM2Config
from smollm2_model import SmolLM2ForCausalLM


# -------------------------
# Utils: logging + env
# -------------------------
def ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_line(path: str, s: str, also_console: bool = True):
    if also_console:
        print(s, flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")



# def sdp_backend_status() -> dict:
#     """Return SDPA backend enablement flags (CUDA only)."""
#     status = {"flash": None, "mem_efficient": None, "math": None}
#     if torch.cuda.is_available():
#         try:
#             from torch.backends.cuda import sdp_kernel
#             status["flash"] = bool(sdp_kernel.is_flash_enabled())
#             status["mem_efficient"] = bool(sdp_kernel.is_mem_efficient_enabled())
#             status["math"] = bool(sdp_kernel.is_math_enabled())
#         except Exception:
#             pass
#     return status

def sdp_backend_status() -> dict:
    """
    Return SDPA backend flags (CUDA only).

    Meanings:
      - True/False : explicitly readable enabled state from PyTorch
      - "auto"     : not forced/overridden; PyTorch will auto-dispatch at runtime
      - "unknown"  : API not available in this build / could not be queried
    """
    status = {"flash": "auto", "mem_efficient": "auto", "math": "auto"}

    if not torch.cuda.is_available():
        return status

    try:
        from torch.backends.cuda import sdp_kernel
        status["flash"] = bool(sdp_kernel.is_flash_enabled())
        status["mem_efficient"] = bool(sdp_kernel.is_mem_efficient_enabled())
        status["math"] = bool(sdp_kernel.is_math_enabled())
    except Exception:
        # Keep explicit string instead of null
        status = {"flash": "unknown", "mem_efficient": "unknown", "math": "unknown"}

    return status



def log_env(log_path: str):
    """Log environment + SDPA backend flags for reproducibility."""
    write_line(log_path, f"{ts()} | PyTorch={torch.__version__}")
    if torch.cuda.is_available():
        write_line(log_path, f"{ts()} | CUDA={torch.version.cuda} | GPU={torch.cuda.get_device_name(0)}")
    write_line(log_path, f"{ts()} | SDPA backends enabled: {json.dumps(sdp_backend_status())}")


# -------------------------
# Determinism / device
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# Data stream
# -------------------------
class PackedTextStream:
    """Deterministic token stream over one text file."""
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.tokens = token_ids  # (L,) on CPU
        self.seq_len = seq_len
        self.L = int(token_ids.numel())
        if self.L < (seq_len + 2):
            raise ValueError(f"Not enough tokens ({self.L}) for seq_len={seq_len}. Add more text or reduce seq_len.")

    def batch(self, global_step: int, batch_size: int, device: torch.device):
        stride = batch_size * self.seq_len
        start = (global_step * stride) % (self.L - (self.seq_len + 1))

        x = torch.empty((batch_size, self.seq_len), dtype=torch.long)
        y = torch.empty((batch_size, self.seq_len), dtype=torch.long)

        for i in range(batch_size):
            s = (start + i * self.seq_len) % (self.L - (self.seq_len + 1))
            chunk = self.tokens[s : s + self.seq_len + 1]
            x[i] = chunk[:-1]
            y[i] = chunk[1:]

        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# -------------------------
# LR schedule + AMP
# -------------------------
def cosine_lr(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / (warmup + 1)
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + coeff * (max_lr - min_lr)


def get_amp_ctx(device: torch.device, use_autocast: bool):
    """
    Returns (context_manager, dtype_name).
    Fix: use contextlib.nullcontext (torch.nullcontext does not exist).
    """
    if not use_autocast:
        return nullcontext(), "fp32"

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), "bf16"
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16), "fp16"

    return nullcontext(), "fp32"


# -------------------------
# Optional profiling window (SDPA proof)
# -------------------------
def maybe_profile_one_window(
    args,
    variant: str,
    model,
    stream: PackedTextStream,
    device: torch.device,
    amp_ctx,
    log_path: str,
) -> dict:
    """Optional profiler window for SDPA proof (CUDA only)."""
    if (not args.profile_sdpa) or (device.type != "cuda"):
        return {}

    import torch.profiler

    prof_dir = os.path.join(args.out_dir, "profiles", variant)
    os.makedirs(prof_dir, exist_ok=True)
    trace_path = os.path.join(prof_dir, f"trace_{variant}.json")

    write_line(log_path, f"{ts()} | PROFILER: capturing {args.profile_steps} steps -> {trace_path}")

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # no-op optimizer for profiler window

    # small warmup
    for w in range(min(2, args.profile_steps)):
        x, y = stream.batch(global_step=w, batch_size=args.batch_size, device=device)
        with amp_ctx:
            out = model(x, labels=y)
            loss = out["loss"]
        loss.backward()
        opt.zero_grad(set_to_none=True)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for pstep in range(args.profile_steps):
            x, y = stream.batch(global_step=pstep, batch_size=args.batch_size, device=device)
            with amp_ctx:
                out = model(x, labels=y)
                loss = out["loss"]
            loss.backward()
            opt.zero_grad(set_to_none=True)

    try:
        prof.export_chrome_trace(trace_path)
        write_line(log_path, f"{ts()} | PROFILER: export_chrome_trace OK")
    except Exception as e:
        write_line(log_path, f"{ts()} | PROFILER: export_chrome_trace failed: {e}")

    top_ops_path = os.path.join(prof_dir, "top_ops.txt")
    try:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        with open(top_ops_path, "w", encoding="utf-8") as f:
            f.write(table)
        write_line(log_path, f"{ts()} | PROFILER: wrote top ops -> {top_ops_path}")
    except Exception as e:
        write_line(log_path, f"{ts()} | PROFILER: top ops table failed: {e}")

    return {
        "profile_enabled": True,
        "profile_steps": int(args.profile_steps),
        "trace_path": trace_path,
        "top_ops_path": top_ops_path,
    }


# -------------------------
# Variant specs
# -------------------------
def variant_spec(name: str):
    name = name.lower()
    if name == "baseline":
        return dict(autocast=False, compile=False, attention_impl="manual", tie_word_embeddings=True)
    if name == "autocast":
        return dict(autocast=True, compile=False, attention_impl="manual", tie_word_embeddings=True)
    if name == "autocast_comp":
        return dict(autocast=True, compile=True, attention_impl="manual", tie_word_embeddings=True)
    if name == "sdpa_on":
        return dict(autocast=True, compile=False, attention_impl="sdpa", tie_word_embeddings=True)
    if name == "sdpa_off":
        return dict(autocast=True, compile=False, attention_impl="manual", tie_word_embeddings=True)
    if name == "sdpa_with_tie_on":
        return dict(autocast=True, compile=False, attention_impl="sdpa", tie_word_embeddings=True)
    if name == "sdpa_with_tie_off":
        return dict(autocast=True, compile=False, attention_impl="sdpa", tie_word_embeddings=False)
    raise ValueError(f"Unknown variant: {name}")


# -------------------------
# Run one ablation
# -------------------------
def run_one(args, variant: str, tokenizer, stream: PackedTextStream, device: torch.device):
    spec = variant_spec(variant)
    log_path = os.path.join(args.out_dir, "logs", f"{variant}.log")
    write_line(log_path, f"{ts()} | ===============================================")
    write_line(log_path, f"{ts()} | RUNNING VARIANT: {variant} | max_steps={args.max_steps}")
    write_line(log_path, f"{ts()} | START variant={variant}")
    write_line(log_path, f"{ts()} | spec={json.dumps(spec)}")
    log_env(log_path)

    cfg = SmolLM2Config(vocab_size=int(tokenizer.vocab_size))
    cfg.tie_word_embeddings = bool(spec["tie_word_embeddings"])
    cfg.attention_impl = spec["attention_impl"]  # used by smollm2_model.py

    model = SmolLM2ForCausalLM(cfg).to(device)

    if spec["compile"] and hasattr(torch, "compile"):
        if os.name == "nt":  # Windows
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model)  # Inductor (Triton) on Linux


    params = sum(p.numel() for p in model.parameters())

    # optimizer groups (like train.py)
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    fused_ok = (device.type == "cuda") and ("fused" in optim.AdamW.__init__.__code__.co_varnames)
    optimizer = optim.AdamW(
        optim_groups,
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True if fused_ok else False,
    )

    amp_ctx, amp_name = get_amp_ctx(device, spec["autocast"])
    prof_info = maybe_profile_one_window(args, variant, model, stream, device, amp_ctx, log_path)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    global_step = 0
    running_loss = 0.0
    running_tokens = 0

    warmup_ignore = min(args.ignore_first_steps, max(0, args.max_steps - 1))
    timed_steps = 0
    timed_tokens = 0
    timed_elapsed = 0.0

    win_t0 = time.perf_counter()

    while global_step < args.max_steps:
        lr = cosine_lr(global_step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        step_t0 = time.perf_counter()

        for _ in range(args.grad_accum):
            x, y = stream.batch(global_step=global_step, batch_size=args.batch_size, device=device)
            with amp_ctx:
                out = model(x, labels=y)
                loss = out["loss"] / args.grad_accum
            loss.backward()
            accum_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_t1 = time.perf_counter()

        tokens_this_step = args.batch_size * args.seq_len * args.grad_accum
        running_loss += accum_loss
        running_tokens += tokens_this_step

        global_step += 1

        if global_step > warmup_ignore:
            timed_steps += 1
            timed_tokens += tokens_this_step
            timed_elapsed += (step_t1 - step_t0)

        if (global_step % args.log_every == 0) or (global_step in (1, args.max_steps)):
            elapsed = time.perf_counter() - win_t0
            tok_s = running_tokens / max(elapsed, 1e-9)
            denom = args.log_every if (global_step % args.log_every == 0) else max(1, global_step % args.log_every)
            avg_loss = running_loss / denom
            write_line(log_path, f"{ts()} | step {global_step:4d}/{args.max_steps} | lr {lr:.3e} | loss {avg_loss:.4f} | tok/s {tok_s:.1f}")
            running_loss = 0.0
            running_tokens = 0
            win_t0 = time.perf_counter()

    avg_tok_s_steady = (timed_tokens / max(timed_elapsed, 1e-9)) if timed_steps > 0 else 0.0
    peak_mem_bytes = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0

    result = {
        "timestamp": ts(),
        "variant": variant,
        "steps": args.max_steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "device": str(device),
        "autocast": bool(spec["autocast"]),
        "amp_dtype": amp_name,
        "compile": bool(spec["compile"]),
        "attention_impl": spec["attention_impl"],
        "tie_word_embeddings": bool(spec["tie_word_embeddings"]),
        "params": int(params),
        "warmup_ignored_steps": int(warmup_ignore),
        "avg_tok_s_steady": float(avg_tok_s_steady),
        "peak_mem_bytes": int(peak_mem_bytes),
        "sdp_backends_enabled": sdp_backend_status(),
        "profiler": prof_info,
    }

    write_line(log_path, f"{ts()} | END variant={variant}")
    return result


# -------------------------
# Aggregation: jsonl -> csv
# -------------------------
def append_results(out_dir: str, result: dict):
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "ablations.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    csv_path = os.path.join(out_dir, "ablations.csv")
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    cols = [
        "timestamp", "variant", "steps", "seq_len", "batch_size", "grad_accum", "seed", "device",
        "autocast", "amp_dtype", "compile", "attention_impl", "tie_word_embeddings",
        "params", "warmup_ignored_steps", "avg_tok_s_steady", "peak_mem_bytes"
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="input.txt")
    parser.add_argument("--out_dir", type=str, default="out_ablation_study")
    parser.add_argument("--tokenizer_id", type=str, default="HuggingFaceTB/SmolLM2-135M")

    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=500)

    # Convenience alias so "--steps 10" works
    parser.add_argument("--steps", type=int, default=None, help="Alias for --max_steps")

    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ignore_first_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "autocast", "autocast_comp", "sdpa_on", "sdpa_off", "sdpa_with_tie_on", "sdpa_with_tie_off"],
    )
    parser.add_argument("--run_all", action="store_true")

    # SDPA proof toggles
    parser.add_argument(
        "--profile_sdpa",
        action="store_true",
        help="Enable torch.profiler window for SDPA/Flash proof (CUDA only). OFF by default.",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=20,
        help="Number of steps to capture in the profiler window when --profile_sdpa is enabled.",
    )

    args = parser.parse_args()

    # alias handling
    if args.steps is not None:
        args.max_steps = int(args.steps)

    device = get_device()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # suppress long-seq tokenizer warning (we slice ourselves)
    tokenizer.model_max_length = 10**9
    tokenizer.init_kwargs["model_max_length"] = tokenizer.model_max_length

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    token_ids = torch.tensor(ids, dtype=torch.long)
    stream = PackedTextStream(token_ids, seq_len=args.seq_len)

    variants = (
        ["baseline", "autocast", "autocast_comp", "sdpa_on", "sdpa_off", "sdpa_with_tie_on", "sdpa_with_tie_off"]
        if args.run_all
        else [args.variant]
    )

    for v in variants:
        set_seed(args.seed)  # same init each time for fair comparison
        result = run_one(args, v, tokenizer, stream, device)
        append_results(args.out_dir, result)

    print(f"Done. See: {os.path.join(args.out_dir, 'ablations.csv')}")


if __name__ == "__main__":
    main()
