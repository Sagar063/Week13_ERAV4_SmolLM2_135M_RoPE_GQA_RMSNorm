"""
smollm2_model.py

Purpose
-------
A minimal, self-contained implementation of the SmolLM2-135M *architecture* (decoder-only Transformer)
derived from public configuration files in:
    https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main

What we reverse-engineered from Hugging Face artifacts
------------------------------------------------------
From `config.json` (base model repo):
- LLaMA-style decoder-only Transformer ("model_type": "llama", "architectures": ["LlamaForCausalLM"])
- RoPE rotary positional embeddings (rope_theta=100000, max_position_embeddings=8192)
- RMSNorm (rms_norm_eps=1e-5)
- SwiGLU MLP (hidden_act=silu, intermediate_size=1536)
- Grouped-Query Attention (num_attention_heads=9, num_key_value_heads=3)
- Weight tying (tie_word_embeddings=true)
- vocab_size=49152, hidden_size=576, num_hidden_layers=30

Tokenizer note
--------------
Tokenization is *not* part of the neural network. This model expects `input_ids` (integers) as input.
A tokenizer (e.g., `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")`) converts raw text
into token IDs. The model then learns embeddings internally via a trainable embedding matrix.

Design choices
--------------
- This implementation supports two attention backends for coursework ablations:
    (A) "sdpa": PyTorch `scaled_dot_product_attention(is_causal=True)` (may use FlashAttention kernels when available)
    (B) "manual": explicit QK^T / softmax / V with a causal mask
  The backend is selected via `cfg.attention_impl` if present, otherwise defaults to "sdpa".
- This file contains NO pretrained weights and NO downloads.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollm2_config import SmolLM2Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (.., D) -> split into even/odd and rotate
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    out = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)
    return out


class RotaryEmbedding(nn.Module):
    """
    RoPE cache builder.

    Reverse-engineered parameters:
    - base (= theta) from config.json: rope_theta=100000
    - max_position_embeddings from config.json: 8192
    """
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even; got {head_dim}")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_pos = max_position_embeddings

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        self._device_cached = None
        self._dtype_cached = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))   # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)                        # (T, D)
        self._cos_cached = emb.cos().to(dtype=dtype)
        self._sin_cached = emb.sin().to(dtype=dtype)
        self._seq_len_cached = seq_len
        self._device_cached = device
        self._dtype_cached = dtype

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or seq_len > self._seq_len_cached
            or device != self._device_cached
            or dtype != self._dtype_cached
        ):
            self._build_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (B, H, T, D)
    cos, sin: (T, D) -> broadcast to (1,1,T,D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class SmolLM2MLP(nn.Module):
    """
    SwiGLU MLP:
        down( silu(gate(x)) * up(x) )
    """
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SmolLM2Attention(nn.Module):
    """
    Grouped-Query Attention (GQA):
    - Q heads = num_attention_heads
    - K/V heads = num_key_value_heads
    - K/V repeated to match Q heads
    """
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            base=cfg.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)     # (B, Hq, T, D)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, T, D)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, T, D)

        cos, sin = self.rope(seq_len=seq_len, device=device, dtype=dtype)
        q, k = apply_rope(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)  # (B, Hq, T, D)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

                # --- Attention backend switch (for ablation studies) ---
        # Default is "sdpa" to match your original training behavior (train.py).
        attn_impl = getattr(self.cfg, "attention_impl", "sdpa")
        attn_impl = (attn_impl or "sdpa").lower()

        if attn_impl == "sdpa":
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.cfg.attention_dropout if (self.training and self.cfg.attention_dropout > 0) else 0.0,
                is_causal=True,
            )  # (B, H, T, D)

        elif attn_impl == "manual":
            d = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)  # (B, H, T, T)

            causal = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            if self.training and self.cfg.attention_dropout > 0:
                attn = torch.dropout(attn, p=self.cfg.attention_dropout, train=True)

            out = torch.matmul(attn, v)  # (B, H, T, D)

        else:
            raise ValueError(f"Unknown attention_impl: {attn_impl}. Use 'sdpa' or 'manual'.")

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class SmolLM2Block(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = SmolLM2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SmolLM2MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        x = x + self.self_attn(h)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class SmolLM2Model(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([SmolLM2Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)  # (B, T, C)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.model = SmolLM2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights(std=cfg.initializer_range)

    def _init_weights(self, std: float):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        hidden = self.model(input_ids)
        logits = self.lm_head(hidden)

        out = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ):
        self.eval()
        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_position_embeddings:]
            logits = self(idx_cond)["logits"][:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits = torch.where(
                    logits < v[:, [-1]],
                    torch.full_like(logits, float("-inf")),
                    logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx
