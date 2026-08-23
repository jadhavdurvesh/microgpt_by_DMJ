"""
Saudade v4 model — the "modern block" from the roadmap:

    RMSNorm -> RoPE attention (SDPA) -> residual -> RMSNorm -> SwiGLU -> residual

replacing v3's:

    LayerNorm -> learned-position attention (manual) -> residual -> LayerNorm -> GELU FFN -> residual

Everything else about the shape of the model — pre-norm residual blocks,
causal masking, tied input/output embeddings, careful init — carries over
from v3 unchanged. Still deliberately NOT doing: GQA, MoE, KV-cache. Those
stay out of scope for v4 too (this is "Option A": modern block, small model).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm: like LayerNorm but no mean-subtraction and no bias — cheaper, and what
    every modern small-Transformer recipe (LLaMA, Mistral, etc.) uses instead of LayerNorm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim, max_seq_len, theta=10000.0):
    """Precompute cos/sin tables for rotary position embeddings, LLaMA-style
    (each frequency repeated across both halves of head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)      # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: (B, heads, T, head_dim). cos/sin: (T, head_dim)."""
    cos = cos[None, None, :, :].to(dtype=x.dtype, device=x.device)
    sin = sin[None, None, :, :].to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    """Causal self-attention with RoPE, using PyTorch's fused
    scaled_dot_product_attention kernel instead of a hand-written
    QK^T-softmax-V — the "faster attention" item from the roadmap. This
    picks the best available kernel (flash/mem-efficient/math) automatically
    per-device; it is not a hand-rolled Flash Attention implementation."""

    def __init__(self, embed_size, heads, block_size, dropout, rope_theta):
        super().__init__()
        assert embed_size % heads == 0

        self.heads = heads
        self.head_dim = embed_size // heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.proj = nn.Linear(embed_size, embed_size, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

        cos, sin = precompute_rope(self.head_dim, block_size, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, T, head_dim)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class SwiGLU(nn.Module):
    """Gated FFN: down(silu(gate(x)) * up(x)) instead of v3's down(GELU(up(x)))."""

    def __init__(self, embed_size, hidden, dropout):
        super().__init__()
        self.gate = nn.Linear(embed_size, hidden, bias=False)
        self.up = nn.Linear(embed_size, hidden, bias=False)
        self.down = nn.Linear(hidden, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def swiglu_hidden_size(embed_size):
    """LLaMA-style sizing: 2/3 of a 4x expansion, rounded up to a multiple of 32
    so matmuls stay well-aligned. Keeps SwiGLU's compute roughly comparable to
    a plain 4x GELU FFN despite the extra gate projection."""
    hidden = int(2 * (embed_size * 4) / 3)
    return ((hidden + 31) // 32) * 32


class Block(nn.Module):
    """Pre-norm block: x = x + attn(rmsnorm(x)); x = x + swiglu(rmsnorm(x))."""

    def __init__(self, embed_size, heads, block_size, dropout, rope_theta, ffn_hidden):
        super().__init__()
        self.norm1 = RMSNorm(embed_size)
        self.attn = Attention(embed_size, heads, block_size, dropout, rope_theta)
        self.norm2 = RMSNorm(embed_size)
        self.ffn = SwiGLU(embed_size, ffn_hidden, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SaudadeV4(nn.Module):
    """Saudade v4: token embedding -> N modern blocks (RoPE/RMSNorm/SwiGLU) -> head.

    No learned position embedding table — RoPE encodes position inside attention
    instead, so there's nothing added to the token embeddings up front.
    """

    def __init__(self, config):
        super().__init__()

        vocab_size = config["vocab_size"]
        embed_size = config["embed_size"]
        block_size = config["block_size"]
        heads = config["heads"]
        layers = config["layers"]
        dropout = config["dropout"]
        rope_theta = config.get("rope_theta", 10000.0)
        ffn_hidden = config.get("ffn_hidden") or swiglu_hidden_size(embed_size)

        self.block_size = block_size
        self.gradient_checkpointing = config.get("gradient_checkpointing", False)

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_size, heads, block_size, dropout, rope_theta, ffn_hidden)
            for _ in range(layers)
        ])

        self.norm = RMSNorm(embed_size)

        self.head = nn.Linear(embed_size, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight  # tied, same as v3

        self.apply(self._init_weights)

        # GPT-2-style scaled init on the two projections that feed directly
        # back into the residual stream (attention out-proj, SwiGLU down-proj) —
        # same trick as v3, needed even more here since v4 is 8 layers deep.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("ffn.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        h = self.emb_dropout(self.token_embedding(x))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        h = self.norm(h)
        logits = self.head(h)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
