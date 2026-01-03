import os
from functools import lru_cache

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# HF Model repo id (NOT the Space repo id)
MODEL_ID = os.getenv("MODEL_ID", "Sunny063/ERAV4-Week13-SmolLLM2-135m")


@lru_cache(maxsize=1)
def load_model():
    """
    Load tokenizer + model once per Space container.
    lru_cache ensures we don't reload on every request.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # CPU tier: safest is float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32,
    )
    model.eval()

    # Many causal LM tokenizers have no pad token; use eos as pad for generation.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def generate_text(prompt, max_new_tokens, temperature, top_p, top_k):
    tokenizer, model = load_model()

    prompt = (prompt or "").strip()
    if not prompt:
        return "Please enter a prompt."

    inputs = tokenizer(prompt, return_tensors="pt")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


with gr.Blocks(title="SmolLM2-135M (ERA V4 Week13)") as demo:
    gr.Markdown(
        """
# SmolLM2-135M — ERA V4 Week13 (From Scratch)

This Space runs inference on a SmolLM2-135M compatible decoder-only model trained from scratch (RoPE + GQA + RMSNorm).  
It loads the model from the Hugging Face **Model Repository** and runs generation on **CPU (free tier)**.
"""
    )

    prompt = gr.Textbox(
        label="Prompt",
        value="BIANCA:",
        lines=6,
        placeholder="Type your prompt here..."
    )

    with gr.Row():
        max_new_tokens = gr.Slider(1, 256, value=96, step=1, label="Max new tokens")
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")

    with gr.Row():
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
        top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")

    btn = gr.Button("Generate")
    output = gr.Textbox(label="Output", lines=14)

    btn.click(
        fn=generate_text,
        inputs=[prompt, max_new_tokens, temperature, top_p, top_k],
        outputs=output
    )

demo.launch()
