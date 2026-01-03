# Week13_ERAV4_SmolLM2_135M_RoPE_GQA_RMSNorm

## From-Scratch Training of SmolLM2-135M (Base) — Reverse-Engineered Architecture (RoPE + GQA + RMSNorm)

This repository demonstrates **reverse-engineering and from-scratch training** of the **base** `SmolLM2-135M` architecture (not the instruct model), using only **public Hugging Face artifacts** for configuration/tokenizer, and **training with random initialization**.

Reference (base model):  
https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main

---

## 1. What the Assignment Requires (Grading-Focused)

- Build the **SmolLM2-135M base** architecture (`model.py` equivalent) from public artifacts
- Train **from scratch** (no pretrained weights)
- Run:
  - **5000 steps**
  - Resume from the saved checkpoint and continue **5001 → 5050**
- Provide evidence of:
  - Frequent checkpointing
  - Clear “RESUME: starting from global_step=5000” style logs
  - Training logs at fixed intervals (I used `--log_every 50`)

> Loss target is not graded; continuity and correctness are.

---

## 2. Base vs Instruct (Important)

- **SmolLM2-135M (base):** pretrained on broad data for next-token prediction.
- **SmolLM2-135M-Instruct:** the base model further **instruction-tuned** (e.g., supervised fine-tuning / alignment) to follow prompts better.

This repo uses **base only** for architecture reference and tokenizer. Training is **from scratch** (random init).

---

## 3. What Was Reverse-Engineered (from `config.json`)

From the reference repo’s `config.json`, we reconstructed:

- Decoder-only Transformer (`LlamaForCausalLM`-style)
- `hidden_size = 576`
- `num_hidden_layers = 30`
- `num_attention_heads = 9`
- `num_key_value_heads = 3` (**GQA**)
- MLP `intermediate_size = 1536` with **SwiGLU/SiLU**
- **RMSNorm** (`rms_norm_eps = 1e-5`)
- **RoPE** (`rope_theta = 100000`)
- `max_position_embeddings = 8192`
- `vocab_size = 49152`
- **Weight tying** (`tie_word_embeddings = true`)

Sanity checks confirm parameter count matches the reference: **134,515,008 params (~134.5M)**.

---

## 4. Tokenization vs Embeddings (Clarification)

- The tokenizer converts raw text → **integer token IDs** (e.g., `[123, 987, ...]`).
- The model then maps token IDs → **learned embeddings** using `nn.Embedding`.
- We use Hugging Face tokenizer **only** for token IDs (text→IDs).  
  **All embeddings and weights are learned from scratch** during training.

---

## 5. Repository Structure

```
Week13_ERAV4_SmolLM2_135M_RoPE_GQA_RMSNorm/
│
├── smollm2_config.py              # Config dataclass mirroring HF config.json
├── smollm2_model.py               # Reverse-engineered SmolLM2-135M model
├── sanity_check.py                # Architecture + checkpoint roundtrip + optional pretrained-compat check
├── train.py                       # From-scratch training with resume
├── train_ablation.py              # Ablation runner (speed/throughput/memory comparisons)
├── requirements.txt
├── README.md
├── update_readme.py               # Auto-inserts logs + ablation CSV into README
├── input.txt                      # Training corpus (single file for this assignment)
│
├── out_smollm2_scratch/
│   ├── train.log                  # Main training log (steps + samples)
│   └── checkpoints/               # Frequent checkpoints (local only; ignored by git)
│
└── ablation_results_500steps_profiles/
    ├── ablation_results.csv       # Aggregated results table
    └── logs/                      # Per-variant logs
```

---

## 6. Why `sanity_check.py` Exists (What It Proves)

`sanity_check.py` is intentionally **not part of training**. It provides fast verification that:

1. Model can be instantiated and has ~135M parameters
2. Forward pass produces correct logits shape `[B, T, vocab]`
3. Checkpoint save/load round-trip works (including `global_step`)
4. (Optional) **Pretrained compatibility check**: loads HF base model and checks that the state dict keys match ours.

### Run sanity checks

```bash
python sanity_check.py
```

Optional pretrained-compatibility verification (for architecture confidence only):

```bash
python sanity_check.py --compare_pretrained --hf_device cpu
```

> This does **not** violate “train from scratch”. It only validates architecture parity.

---

## 7. Training: How I Ran It (CLI Commands)

### Step A — Train from scratch to 5000 steps

```bash
python train.py --input_file input.txt --seq_len 256 --batch_size 1 --grad_accum 4 --max_steps 5000 --resume 0 --log_every 50 --save_every 250 --sample_every 500
```

### Step B — Resume from step 5000 and continue to 5050

```bash
python train.py --input_file input.txt --seq_len 256 --batch_size 1 --grad_accum 4 --max_steps 5050 --resume 1 --log_every 50 --save_every 250 --sample_every 500
```

#### Resume semantics (important)
- When resuming, logs show: `RESUME: starting from global_step=5000`
- The loop then increments and the **first new training step executed is 5001**.
- Step 5000 is **not recomputed**; it’s the checkpoint boundary.

---

## 8. Training Logs (Auto-Embedded)

The full training log is generated at:

- `out_smollm2_scratch/train.log`

To embed it into this README (collapsible), run:

```bash
python update_readme.py
```

After running, this section will contain the actual log:

<!-- TRAIN_LOG_BEGIN -->
<details>
  <summary><b>Click to expand: out_smollm2_scratch/train.log</b></summary>

  _Run `python update_readme.py` to auto-insert the latest training log here._

</details>
<!-- TRAIN_LOG_END -->

---

## 9. Ablation Study (Speed/Memory/Config Comparisons)

Goal: compare training throughput and memory usage for different settings (500 steps each).

### Variants tested

| Variant | Autocast | Compile | Attention | Weight tying |
|---|---:|---:|---|---:|
| `baseline` | OFF | OFF | manual | ON |
| `autocast` | ON (bf16) | OFF | manual | ON |
| `autocast_comp` | ON (bf16) | ON | manual | ON |
| `sdpa_on` | ON (bf16) | OFF | SDPA | ON |
| `sdpa_off` | ON (bf16) | OFF | manual | ON |
| `sdpa_with_tie_on` | ON (bf16) | OFF | SDPA | ON |
| `sdpa_with_tie_off` | ON (bf16) | OFF | SDPA | OFF |

### Commands

Run all variants for 500 steps:

```bash
python train_ablation.py --input_file input.txt --seq_len 256 --batch_size 1 --grad_accum 4 --steps 500 --ignore_first_steps 50 --log_every 50 --run_all --out_dir ablation_results_500steps_profiles
```

Run a single variant:

```bash
python train_ablation.py --input_file input.txt --seq_len 256 --batch_size 1 --grad_accum 4 --steps 500 --ignore_first_steps 50 --log_every 50 --variant sdpa_on --out_dir ablation_results_500steps_profiles
```

### Ablation results (Auto-Embedded)

Ablation CSV is expected at:

- `ablation_results_500steps_profiles/ablation_results.csv`

Run:

```bash
python update_readme.py
```

to auto-render it below as a Markdown table.

<!-- ABLATION_TABLE_BEGIN -->
_Run `python update_readme.py` to auto-insert ablation_results.csv here._
<!-- ABLATION_TABLE_END -->

---

## 10. Key Findings (From My Runs)

- **SDPA ON** improved throughput and reduced peak memory compared to manual attention on RTX 4060 Ti.
- `torch.compile` was slower in my setup (Windows + missing Triton), so it is not recommended for the graded run.
- Turning **weight tying OFF** increases parameter count substantially (different parameterization) and increases memory.

---

## 11. Hugging Face Upload (Placeholder)

I will upload the trained model in Hugging Face format later:

- Model repo: **TBD**
- Space: **TBD**

---

## 12. Repro Notes

Hardware used:
- RTX 4060 Ti (16GB VRAM)
- Intel i7-13700K
- 32GB RAM

---

## Appendix: Automating README Inserts

`update_readme.py` auto-inserts:
- `out_smollm2_scratch/train.log` into the collapsible block
- `ablation_results_500steps_profiles/ablation_results.csv` as a Markdown table

Run:

```bash
python update_readme.py
```
