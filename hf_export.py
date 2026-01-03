import argparse
import json
from pathlib import Path

import torch


def _dtype_from_arg(dtype_str: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported --dtype {dtype_str}. Choose from fp32/bf16/fp16.")
    return mapping[dtype_str]


def _torch_dtype_str(dtype_str: str) -> str:
    return {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[dtype_str]


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_cfg() -> dict:
    """
    Fallback model config (used only if checkpoint does not contain 'config').
    Matches your Week13 SmolLM2-135M training setup.
    """
    return {
        "vocab_size": 49152,
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100000.0,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }


def _build_hf_llama_config_dict(cfg: dict, torch_dtype: str) -> dict:
    """
    Build a Hugging Face compatible LLaMA-family config for SmolLM2-135M.
    """
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": int(cfg["vocab_size"]),
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
        "rope_theta": float(cfg["rope_theta"]),
        "rms_norm_eps": float(cfg["rms_norm_eps"]),
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": float(cfg.get("attention_dropout", 0.0)),
        "initializer_range": float(cfg.get("initializer_range", 0.041666666666666664)),
        "use_cache": True,
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", True)),
        "bos_token_id": int(cfg.get("bos_token_id", 0)),
        "eos_token_id": int(cfg.get("eos_token_id", 0)),
        "rope_scaling": None,
        "torch_dtype": torch_dtype,
    }


def _strip_compile_prefixes(state_dict: dict) -> dict:
    """
    Removes common prefixes introduced by torch.compile / DDP wrappers.
    """
    prefixes = ("_orig_mod.", "module.")
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        cleaned[nk] = v
    return cleaned


def _make_safetensors_state_dict_safe(state_dict: dict) -> dict:
    """
    safetensors cannot store shared-storage tensors.
    If lm_head.weight is tied to model.embed_tokens.weight, clone lm_head.weight.
    This preserves values; HF will re-tie weights at load when tie_word_embeddings=True.
    """
    emb_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"

    if emb_key in state_dict and head_key in state_dict:
        emb = state_dict[emb_key]
        head = state_dict[head_key]
        try:
            shared = emb.storage().data_ptr() == head.storage().data_ptr()
        except Exception:
            shared = False

        if shared:
            new_sd = dict(state_dict)
            new_sd[head_key] = emb.clone()
            return new_sd

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Export ERA-V4 SmolLM2 checkpoint to HF model repo format.")
    parser.add_argument("--ckpt_path", type=str, default=r"out_smollm2_scratch\\final_checkpoint.pt")
    parser.add_argument("--out_dir", type=str, default="hf_export")
    parser.add_argument("--tokenizer_id", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dtype = _dtype_from_arg(args.dtype)
    torch_dtype_str = _torch_dtype_str(args.dtype)

    # 1) Load checkpoint
    print(f"[1/6] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError("Unrecognized checkpoint format: expected a dict.")

    if "model_state" in ckpt:
        model_state = ckpt["model_state"]
    else:
        # Sometimes a checkpoint is saved as a raw state_dict
        # (dict[str, Tensor]) without wrapping keys.
        model_state = ckpt

    if not isinstance(model_state, dict):
        raise ValueError("model_state is not a dict.")

    model_state = _strip_compile_prefixes(model_state)

    cfg = ckpt.get("config", None)
    if cfg is None:
        print("[1/6] WARNING: checkpoint has no 'config'. Using default SmolLM2-135M config.")
        cfg = _default_cfg()

    # 2) Build HF config + HF model
    print("[2/6] Building HF LlamaConfig + LlamaForCausalLM ...")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg_dict = _build_hf_llama_config_dict(cfg, torch_dtype=torch_dtype_str)
    _write_json(out_dir / "config.json", cfg_dict)

    hf_config = LlamaConfig.from_dict(cfg_dict)
    hf_model = LlamaForCausalLM(hf_config)

    # 3) Load weights (strict)
    print("[3/6] Loading state_dict into HF model (strict=True) ...")
    load_res = hf_model.load_state_dict(model_state, strict=True)
    if len(load_res.missing_keys) or len(load_res.unexpected_keys):
        raise RuntimeError(
            f"HF strict load failed: missing={len(load_res.missing_keys)}, "
            f"unexpected={len(load_res.unexpected_keys)}"
        )

    # Enforce tying
    if hf_config.tie_word_embeddings:
        hf_model.tie_weights()

    # Cast for saving
    hf_model = hf_model.to(dtype=save_dtype)
    hf_model.eval()

    # 4) Save model weights (prefer safetensors if available)
    print("[4/6] Saving model weights ...")
    sd_to_save = hf_model.state_dict()

    try:
        sd_safe = _make_safetensors_state_dict_safe(sd_to_save)
        hf_model.save_pretrained(
            str(out_dir),
            state_dict=sd_safe,
            safe_serialization=True,
        )
        print("  Saved with safe_serialization=True (model.safetensors if safetensors is installed).")
    except Exception as e:
        print(f"  safetensors save failed ({repr(e)}). Falling back to pytorch_model.bin ...")
        hf_model.save_pretrained(
            str(out_dir),
            state_dict=sd_to_save,
            safe_serialization=False,
        )
        print("  Saved with safe_serialization=False (pytorch_model.bin).")

    # 5) Save tokenizer
    print("[5/6] Saving tokenizer files ...")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    tok.save_pretrained(str(out_dir))

    # 6) generation_config + README
    print("[6/6] Writing generation_config.json and README.md ...")
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
        eos_token_id=hf_config.eos_token_id,
        bos_token_id=hf_config.bos_token_id,
        pad_token_id=getattr(tok, "pad_token_id", None),
    )
    gen_cfg.save_pretrained(str(out_dir))

    readme = f"""---
license: apache-2.0
tags:
- text-generation
- llama
---

# SmolLM2-135M (ERA V4 Week13) — trained from scratch

This repository contains a SmolLM2-135M compatible checkpoint trained from scratch as part of ERA V4 Week13.

## Architecture (exported)
- vocab_size={cfg_dict["vocab_size"]}
- hidden_size={cfg_dict["hidden_size"]}
- intermediate_size={cfg_dict["intermediate_size"]}
- num_layers={cfg_dict["num_hidden_layers"]}
- n_heads={cfg_dict["num_attention_heads"]}
- n_kv_heads={cfg_dict["num_key_value_heads"]}
- RoPE theta={cfg_dict["rope_theta"]}
- RMSNorm eps={cfg_dict["rms_norm_eps"]}
- max_position_embeddings={cfg_dict["max_position_embeddings"]}
- tie_word_embeddings={cfg_dict["tie_word_embeddings"]}

## Quick usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "<YOUR_HF_USERNAME/YOUR_MODEL_REPO>"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tok("BIANCA:", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=80)
print(tok.decode(out[0], skip_special_tokens=True))
```
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("\nDone.")
    print(f"Exported HF folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
