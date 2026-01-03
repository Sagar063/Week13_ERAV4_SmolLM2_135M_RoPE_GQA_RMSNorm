---
title: ERAV4 Week13 SmolLLM2 135m
emoji: 📌
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# SmolLM2-135M — ERA V4 Week13 (CPU Inference)

This Space provides a Gradio UI for inference on a SmolLM2-135M compatible model trained from scratch (RoPE + GQA + RMSNorm).

- Model repo: `Sunny063/ERAV4-Week13-SmolLLM2-135m`
- Runs on: CPU (Hugging Face free tier)

## Notes
CPU inference is slower than GPU. Keep `Max new tokens` around 64–128 for responsive generation.
