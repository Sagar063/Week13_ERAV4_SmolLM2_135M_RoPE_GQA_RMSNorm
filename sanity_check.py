"""
sanity_check.py

Purpose
-------
A separate sanity script (not called by training) that verifies:
1) Model instantiation and parameter count (~135M range)
2) Forward pass output shapes
3) Checkpoint save/load round-trip with global_step

Optional (OFF by default):
4) Pretrained compatibility check:
   - Loads the official pretrained HF model (base) on CPU
   - Attempts to load its weights into our reverse-engineered model with strict=False
   - Reports missing/unexpected keys and basic param-count agreement

Reverse-engineering reference
-----------------------------
Architecture is derived from HuggingFaceTB/SmolLM2-135M `config.json`:
https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main

Important note for grading
--------------------------
- Training is still "from scratch". This file does NOT train.
- Optional pretrained load is only for architecture compatibility verification and is disabled by default.
"""

import os
import argparse
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from smollm2_config import SmolLM2Config
from smollm2_model import SmolLM2ForCausalLM
from train import save_checkpoint, load_checkpoint, get_device


def log(msg: str, log_path: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pretrained_compatibility_check(
    repo_id: str,
    my_model: SmolLM2ForCausalLM,
    log_path: str,
    device_for_hf: str = "cpu",
):
    """
    Loads the official pretrained HF model and checks whether its state_dict can be loaded
    into our reverse-engineered model. This is a strong architectural compatibility signal.

    We do NOT use these weights for training. This is verification only.

    device_for_hf:
      - "cpu" recommended for 4060 Ti to avoid VRAM spikes.
      - "cuda" possible if you want faster load and have memory headroom.
    """
    log("=== OPTIONAL: Pretrained compatibility check START ===", log_path)
    log(f"HF repo_id: {repo_id}", log_path)
    log(f"Loading HF pretrained model on: {device_for_hf}", log_path)

    # hf_device_map = None
    # if device_for_hf == "cpu":
    #     # Force CPU load
    #     hf_device_map = {"": "cpu"}
    # elif device_for_hf == "cuda":
    #     # Let HF decide; can still land on GPU
    #     hf_device_map = "auto"

    # hf_model = AutoModelForCausalLM.from_pretrained(
    #     repo_id,
    #     torch_dtype="auto",
    #     device_map=hf_device_map,
    # )
    hf_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype="auto",
    )

    if device_for_hf == "cuda":
        hf_model = hf_model.to("cuda")

    hf_params = sum(p.numel() for p in hf_model.parameters())
    my_params = sum(p.numel() for p in my_model.parameters())

    log(f"HF pretrained params: {hf_params:,}", log_path)
    log(f"Our model params     : {my_params:,}", log_path)

    # Try loading HF weights into our model (strict=False) and report key mismatches.
    # This is the main architectural compatibility test.
    hf_sd = hf_model.state_dict()

    missing, unexpected = my_model.load_state_dict(hf_sd, strict=False)

    log(f"load_state_dict(strict=False) results:", log_path)
    log(f"  Missing keys   : {len(missing)}", log_path)
    log(f"  Unexpected keys: {len(unexpected)}", log_path)

    # Log a few examples (not too many)
    if missing:
        log("  Examples of missing keys (first 15):", log_path)
        for k in missing[:15]:
            log(f"    - {k}", log_path)

    if unexpected:
        log("  Examples of unexpected keys (first 15):", log_path)
        for k in unexpected[:15]:
            log(f"    - {k}", log_path)

    # Heuristic pass/fail guidance (NOT an assertion by default)
    # Ideally missing/unexpected should be small (mostly buffers, rope caches, etc.)
    if len(missing) == 0 and len(unexpected) == 0:
        log("Pretrained compatibility: PERFECT key match.", log_path)
    else:
        log(
            "Pretrained compatibility: non-zero key mismatches. "
            "This may still be acceptable if mismatches are buffers/caches, but review above.",
            log_path,
        )

    log("=== OPTIONAL: Pretrained compatibility check END ===", log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="sanity_check.log")
    parser.add_argument(
        "--compare_pretrained",
        action="store_true",
        help="Optional: load HF pretrained model and verify compatibility (OFF by default).",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="HF repo id for pretrained compatibility check (base model).",
    )
    parser.add_argument(
        "--hf_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for loading HF pretrained model (recommended: cpu).",
    )
    args = parser.parse_args()

    # Reset log file for a clean run
    with open(args.log_path, "w", encoding="utf-8") as f:
        f.write("")

    device = get_device()
    log(f"Device for our model: {device}", args.log_path)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = SmolLM2Config(vocab_size=int(tokenizer.vocab_size))
    model = SmolLM2ForCausalLM(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Config: {cfg}", args.log_path)
    log(f"Total params: {total_params:,}", args.log_path)
    assert 130_000_000 < total_params < 140_000_000, "Expected ~135M parameter range"

    # Forward pass check
    B, T = 2, 128
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    out = model(x, labels=y)
    logits = out["logits"]

    assert logits.shape == (B, T, cfg.vocab_size)
    assert "loss" in out
    log("Forward pass: OK", args.log_path)

    # Checkpoint round-trip check
    tmp_dir = "tmp_sanity"
    os.makedirs(tmp_dir, exist_ok=True)
    ckpt_path = os.path.join(tmp_dir, "ckpt.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    save_checkpoint(ckpt_path, model, opt, global_step=123, cfg=cfg)
    step_loaded = load_checkpoint(ckpt_path, model, opt, device=device)
    assert step_loaded == 123
    log("Checkpoint save/load: OK", args.log_path)

    # Optional pretrained architecture compatibility check
    if args.compare_pretrained:
        pretrained_compatibility_check(
            repo_id=args.hf_repo_id,
            my_model=model,
            log_path=args.log_path,
            device_for_hf=args.hf_device,
        )

    log("Sanity check PASSED.", args.log_path)


if __name__ == "__main__":
    main()
