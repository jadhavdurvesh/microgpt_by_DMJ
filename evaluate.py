"""
Saudade v4 — evaluation suite.

The roadmap's "proper evaluation" item: instead of eyeballing one sample,
run a fixed set of category prompts through the model and report both the
loss-based number (val loss / perplexity) and generation-quality numbers
(repetition rate, unique-token ratio) so different checkpoints — including
future v5 checkpoints — can be compared on the same yardstick.

Produces evaluation.json:
    {
      "checkpoint": ...,
      "val_loss": ...,
      "perplexity": ...,
      "params": ...,
      "categories": {
        "STORY": {"prompt": ..., "generation": ..., "repetition_rate": ..., "unique_token_ratio": ...},
        ...
      }
    }

Usage:
    python evaluate.py --checkpoint saudade_v4.pt
    python evaluate.py --checkpoint checkpoints/checkpoint_best.pt --val-bin val.bin
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

from model import SaudadeV4
from config import EVAL_CONFIG

CATEGORY_PROMPTS = {
    "STORY": "Once, in a small village by the sea, there lived",
    "DIALOGUE": "\"I don't think you understand,\" she said. \"",
    "DESCRIPTION": "The old library was filled with",
    "CONTINUATION": "He opened the letter slowly and read the first line:",
    "LONG_CONTEXT": (
        "The house had stood at the edge of the forest for nearly a hundred years. "
        "Its windows were dark, its garden overgrown, and yet every winter a single "
        "light appeared in the topmost room. No one in the village could explain it, "
        "and so"
    ),
    "RARE_WORDS": "The archaeologist examined the peculiar, weather-beaten artifact and",
    "REPETITION": "Again and again, the same thought returned to him:",
}


def top_k_filter(logits, top_k):
    top_k = min(top_k, logits.size(-1))
    kth_value = torch.topk(logits, top_k)[0][..., -1]
    return torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)


@torch.no_grad()
def generate_sample(model, sp, prompt, block_size, device, max_new_tokens, temperature,
                     top_k, repetition_penalty):
    idx = torch.tensor(sp.encode(prompt), dtype=torch.long)

    for _ in range(max_new_tokens):
        idx_cond = idx[-block_size:].unsqueeze(0).to(device)
        logits, _ = model(idx_cond)
        logits = logits[0, -1]

        if repetition_penalty != 1.0 and len(idx) > 0:
            seen = torch.unique(idx)
            seen_logits = logits[seen]
            logits = logits.clone()
            logits[seen] = torch.where(seen_logits > 0, seen_logits / repetition_penalty,
                                        seen_logits * repetition_penalty)

        logits = logits / max(temperature, 1e-6)
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        idx = torch.cat([idx, torch.tensor([next_id])])

    return idx


def repetition_rate(token_ids, n):
    """Fraction of n-grams in the generation that have appeared before."""
    ids = token_ids.tolist()
    if len(ids) < n + 1:
        return 0.0
    ngrams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    seen = set()
    repeats = 0
    for g in ngrams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(ngrams)


def unique_token_ratio(token_ids):
    ids = token_ids.tolist()
    if not ids:
        return 0.0
    return len(set(ids)) / len(ids)


@torch.no_grad()
def compute_val_loss(model, val_bin, block_size, batch_size, device, vocab_size, iters=50):
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    data = np.memmap(val_bin, dtype=dtype, mode="r")

    losses = []
    for _ in range(iters):
        ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
        x = torch.from_numpy(np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[i + 1:i + block_size + 1] for i in ix]).astype(np.int64)).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="saudade_v4.pt")
    parser.add_argument("--val-bin", type=str, default="val.bin")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model")
    parser.add_argument("--out", type=str, default="evaluation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    sp = spm.SentencePieceProcessor()
    sp.load(checkpoint.get("tokenizer", args.tokenizer))

    model = SaudadeV4(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 60)
    print("SAUDADE V4 EVALUATION")
    print("=" * 60)
    print()

    results = {
        "checkpoint": args.checkpoint,
        "step": checkpoint.get("step"),
        "tokens_seen": checkpoint.get("tokens_seen"),
        "params": model.num_params(),
        "config": config,
    }

    try:
        val_loss = compute_val_loss(model, args.val_bin, config["block_size"],
                                     batch_size=16, device=device, vocab_size=config["vocab_size"])
        results["val_loss"] = val_loss
        results["perplexity"] = float(np.exp(val_loss))
        print(f"Val loss    : {val_loss:.4f}")
        print(f"Perplexity  : {results['perplexity']:.2f}")
    except FileNotFoundError:
        print(f"({args.val_bin} not found — skipping val loss/perplexity)")
        results["val_loss"] = None
        results["perplexity"] = None

    print(f"Parameters  : {model.num_params() / 1e6:.2f} M")
    print()

    categories = {}
    for name, prompt in CATEGORY_PROMPTS.items():
        out_ids = generate_sample(
            model, sp, prompt, config["block_size"], device,
            max_new_tokens=EVAL_CONFIG["max_new_tokens"],
            temperature=EVAL_CONFIG["temperature"],
            top_k=EVAL_CONFIG["top_k"],
            repetition_penalty=EVAL_CONFIG["repetition_penalty"],
        )
        prompt_len = len(sp.encode(prompt))
        gen_only = out_ids[prompt_len:]

        rep_rate = repetition_rate(gen_only, EVAL_CONFIG["repetition_window"])
        utr = unique_token_ratio(gen_only)
        text = sp.decode(out_ids.tolist())

        categories[name] = {
            "prompt": prompt,
            "generation": text,
            "repetition_rate": rep_rate,
            "unique_token_ratio": utr,
        }

        print(f"=== {name} ===")
        print(text)
        print(f"  repetition_rate={rep_rate:.3f}  unique_token_ratio={utr:.3f}")
        print()

    results["categories"] = categories

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
