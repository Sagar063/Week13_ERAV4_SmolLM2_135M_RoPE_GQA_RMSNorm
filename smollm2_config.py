"""
smollm2_config.py

Purpose
-------
Defines the SmolLM2-135M model hyperparameters as a small, explicit dataclass.

Reverse-engineering reference
-----------------------------
These values are taken from the public Hugging Face repository for the base model:
    https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main

Specifically, they match the published `config.json` in that repo (architectures = LlamaForCausalLM),
including:
- hidden_size=576
- num_hidden_layers=30
- num_attention_heads=9
- num_key_value_heads=3 (GQA)
- intermediate_size=1536 (SwiGLU MLP)
- max_position_embeddings=8192 (context length)
- rope_theta=100000
- vocab_size=49152
- tie_word_embeddings=true

Note: This file contains *only* architecture metadata. No pretrained weights are used anywhere in
this project; training starts from random initialization.
"""

from dataclasses import dataclass


@dataclass
class SmolLM2Config:
    # --- Token / vocabulary ---
    vocab_size: int = 49152

    # --- Model width / depth ---
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30

    # --- Attention ---
    num_attention_heads: int = 9
    num_key_value_heads: int = 3  # Grouped-Query Attention (GQA)
    attention_dropout: float = 0.0

    # --- Positional encoding ---
    max_position_embeddings: int = 8192
    rope_theta: float = 100000.0

    # --- Normalization / misc ---
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    # --- Special token IDs (per config.json) ---
    bos_token_id: int = 0
    eos_token_id: int = 0

    # --- Initialization (per config.json) ---
    initializer_range: float = 0.041666666666666664
