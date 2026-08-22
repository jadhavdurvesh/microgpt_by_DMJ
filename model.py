"""
Saudade v3 model.

This is still the same idea as the original MiniGPT — a small, from-scratch,
decoder-only Transformer — just properly engineered:

  * token + position embeddings (the original had no positional info at all)
  * pre-norm residual blocks (kept from the original, it was already right)
  * GELU instead of ReLU in the feed-forward network
  * dropout in attention, feed-forward, and embeddings
  * tied input/output embedding weights
  * causal masking (kept from the original, it was already right)

Deliberately NOT included, on purpose, for v3: RoPE, GQA, MoE, a real
Flash Attention kernel, KV-cache, SwiGLU, or any other "modern" pieces.
Those are v4/v5 experiments — v3's job is to see how far the classic
architecture, properly engineered, can go.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)

        return weights @ v


class MultiHead(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, num_heads, head_size, embed_size, block_size, dropout):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(embed_size, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network, 4x expansion, GELU."""

    def __init__(self, embed_size, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block: x = x + attn(ln(x)); x = x + ff(ln(x))."""

    def __init__(self, embed_size, heads, block_size, dropout):
        super().__init__()

        head_size = embed_size // heads

        self.attn = MultiHead(heads, head_size, embed_size, block_size, dropout)
        self.ff = FeedForward(embed_size, dropout)

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Saudade v3: token+position embeddings -> N transformer blocks -> head."""

    def __init__(self, config):
        super().__init__()

        vocab_size = config["vocab_size"]
        embed_size = config["embed_size"]
        block_size = config["block_size"]
        heads = config["heads"]
        layers = config["layers"]
        dropout = config["dropout"]

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            Block(embed_size, heads, block_size, dropout) for _ in range(layers)
        ])

        self.ln = nn.LayerNorm(embed_size)

        # output projection — weights tied to the token embedding
        self.head = nn.Linear(embed_size, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # GPT-2 style scaled init: the two projections that feed directly
        # back into the residual stream (attention out-proj, and the second
        # linear in each feed-forward) get their std scaled down by
        # 1/sqrt(2 * layers). Without this, the residual stream's variance
        # grows with depth and deeper stacks (we're at 6 layers here) get
        # harder to train. Purely a training-stability trick — doesn't
        # change the architecture.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape

        positions = torch.arange(T, device=x.device)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        h = self.emb_dropout(tok_emb + pos_emb)
        h = self.blocks(h)
        h = self.ln(h)

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
