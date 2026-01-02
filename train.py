"""
train.py

Assignment requirements implemented
-----------------------------------
- Train SmolLM2-135M *from scratch* (random initialization)
- Run for 5000 steps, then continue for 50 more steps (steps 5001..5050)
- Log/print every 500 steps
- Checkpoint frequently (default every 250 steps) and also at step 5000
- Resume correctly and show evidence: "RESUME: starting from global_step=XXXX"

Reverse-engineering reference
------------------------------
Architecture values used here match HuggingFaceTB/SmolLM2-135M public config.json. 
Tokenizer is used ONLY for text->token IDs; embeddings are learned inside the model.

Usage
-----
python train.py --input_file input.txt
"""

from __future__ import annotations

import os
import time
import math
import json
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from smollm2_config import SmolLM2Config
from smollm2_model import SmolLM2ForCausalLM


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


class PackedTextStream:
    """
    Deterministic token stream over one text file.

    We tokenize the entire file once. For each global_step, we deterministically choose
    where to cut the next batch. This makes checkpoint resumption clear and verifiable.
    """
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


def cosine_lr(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / (warmup + 1)
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, global_step: int, cfg: SmolLM2Config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "timestamp": time.time(),
    }
    torch.save(ckpt, path)


# def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> int:
#     ckpt = torch.load(path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     optimizer.load_state_dict(ckpt["optimizer_state"])
#     torch.set_rng_state(ckpt["rng_state"])
#     if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
#         torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
#     return int(ckpt["global_step"])

def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    # --- RNG restoration (robust across platforms / torch versions) ---
    rng_state = ckpt.get("rng_state", None)
    if rng_state is not None:
        # torch.set_rng_state expects a ByteTensor (uint8)
        if not isinstance(rng_state, torch.Tensor):
            # if stored as bytes or list-like
            rng_state = torch.tensor(list(rng_state), dtype=torch.uint8)
        else:
            rng_state = rng_state.to(dtype=torch.uint8, device="cpu")
        torch.set_rng_state(rng_state)

    cuda_rng_state = ckpt.get("cuda_rng_state", None)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        fixed_states = []
        for s in cuda_rng_state:
            if s is None:
                fixed_states.append(None)
                continue
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(list(s), dtype=torch.uint8)
            else:
                s = s.to(dtype=torch.uint8, device="cpu")
            fixed_states.append(s)
        torch.cuda.set_rng_state_all(fixed_states)

    return int(ckpt["global_step"])


@torch.no_grad()
def sample_generation(model: SmolLM2ForCausalLM, tokenizer, device, prompt: str, max_new_tokens: int = 60):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=50)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="input.txt")
    parser.add_argument("--out_dir", type=str, default="out_smollm2_scratch")
    parser.add_argument("--tokenizer_id", type=str, default="HuggingFaceTB/SmolLM2-135M",
                        help="Tokenizer source (text->IDs only). No model weights are loaded.")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=5050)     # 5000 + 50
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", type=int, default=1, help="1 to resume from latest checkpoint if present.")
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    workdir = os.path.abspath(args.out_dir)
    ckpt_dir = os.path.join(workdir, "checkpoints")
    latest_ckpt = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    ckpt_5000 = os.path.join(ckpt_dir, "checkpoint_step_5000.pt")
    log_path = os.path.join(workdir, "train.log")

    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    def log_line(s: str):
        print(s, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    log_line(f"Device: {device} | CUDA: {torch.cuda.is_available()} | "
             f"bf16_supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
    log_line(f"Args: {json.dumps(vars(args), indent=2)}")

    # Tokenizer does ONLY text->token IDs; embeddings are learned inside the model.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Suppress tokenizer warning for long files (we will slice into seq_len windows anyway)
    tokenizer.model_max_length = 10**9
    tokenizer.init_kwargs["model_max_length"] = tokenizer.model_max_length


    # Model from scratch (NO pretrained weights).
    cfg = SmolLM2Config(vocab_size=int(tokenizer.vocab_size))
    model = SmolLM2ForCausalLM(cfg).to(device)
    log_line(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # AdamW with decoupled weight decay: decay only for 2D params (weights)
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

    # Mixed precision
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if device.type != "cpu" else torch.nullcontext()

    # Tokenize dataset once
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    #ids = tokenizer(text, return_tensors=None, add_special_tokens=True)["input_ids"]
    ids = tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    token_ids = torch.tensor(ids, dtype=torch.long)  # CPU
    stream = PackedTextStream(token_ids, seq_len=args.seq_len)

    # Resume if checkpoint exists
    global_step = 0
    if args.resume and os.path.exists(latest_ckpt):
        global_step = load_checkpoint(latest_ckpt, model, optimizer, device=device)
        log_line(f"RESUME: loaded {latest_ckpt}")
        log_line(f"RESUME: starting from global_step={global_step}")
    else:
        log_line("Starting from scratch (random init).")

    model.train()
    t0 = time.time()
    running_loss = 0.0
    running_tokens = 0

    while global_step < args.max_steps:
        # Mandatory checkpoint + marker at 5000 to prove continuation into 5001
        if global_step == 5000:
            save_checkpoint(ckpt_5000, model, optimizer, global_step, cfg)
            save_checkpoint(latest_ckpt, model, optimizer, global_step, cfg)
            log_line("=== REACHED STEP 5000. Starting continuation phase: steps 5001..5050 ===")

        lr = cosine_lr(global_step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            x, y = stream.batch(global_step=global_step, batch_size=args.batch_size, device=device)
            with autocast_ctx:
                out = model(x, labels=y)
                loss = out["loss"] / args.grad_accum
            loss.backward()
            accum_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_this_step = args.batch_size * args.seq_len * args.grad_accum
        running_loss += accum_loss
        running_tokens += tokens_this_step

        global_step += 1

        
        # Frequent checkpointing
        if global_step % args.save_every == 0:
            # Always keep a "latest" checkpoint for resume
            save_checkpoint(latest_ckpt, model, optimizer, global_step, cfg)

            # Also keep a step-numbered checkpoint for clarity/evidence
            step_ckpt = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}.pt")
            save_checkpoint(step_ckpt, model, optimizer, global_step, cfg)
            log_line(f"CHECKPOINT: saved {os.path.basename(step_ckpt)} and checkpoint_latest.pt")


        # Required logging (every 50 steps) + explicit visibility around 5000/5001
        if global_step % args.log_every == 0 or global_step in (1, 5000, 5001, args.max_steps):
            elapsed = time.time() - t0
            tok_per_s = running_tokens / max(elapsed, 1e-6)
            steps_in_window = args.log_every if (global_step % args.log_every == 0) else max(1, global_step % args.log_every)
            avg_loss = running_loss / steps_in_window

            log_line(f"step {global_step:5d}/{args.max_steps} | lr {lr:.3e} | loss {avg_loss:.4f} | tok/s {tok_per_s:.1f}")

            # Optional qualitative sanity (kept small)
            if (global_step % args.sample_every == 0) or (global_step in (1, 5000, 5001, args.max_steps)):
                try:
                    prompt = "JULIET:"
                    sample = sample_generation(model, tokenizer, device, prompt=prompt, max_new_tokens=50)
                    log_line(f"[sample @ step {global_step}] {sample}")
                except Exception as e:
                    log_line(f"[sample @ step {global_step}] generation failed: {e}")

            running_loss = 0.0
            running_tokens = 0
            t0 = time.time()

    # Final save
    final_path = os.path.join(workdir, "final_checkpoint.pt")
    save_checkpoint(final_path, model, optimizer, global_step, cfg)
    save_checkpoint(latest_ckpt, model, optimizer, global_step, cfg)
    log_line(f"TRAINING COMPLETE. Saved final_checkpoint.pt at step {global_step}.")


if __name__ == "__main__":
    main()
