"""
Saudade v3 — generation script.

Loads the architecture config from the checkpoint itself (saved by train.py),
so this can never mismatch the model it's generating from. The prompt is
defined in-script below instead of via input(), to match how the Kaggle
notebook cell is meant to run.

Also supports a repetition penalty (small models tend to loop without one)
and an optional greedy mode for deterministic output.
"""

import torch
import torch.nn.functional as F
import sentencepiece as spm

from model import MiniGPT
from config import GEN_CONFIG

# ----------------------------
# Edit these for each run
# ----------------------------
CHECKPOINT_PATH = "saudade_v3.pt"
PROMPT = "The night was quiet and"

TEMPERATURE = GEN_CONFIG["temperature"]
TOP_K = GEN_CONFIG["top_k"]
TOP_P = GEN_CONFIG["top_p"]
MAX_NEW_TOKENS = GEN_CONFIG["max_new_tokens"]
REPETITION_PENALTY = GEN_CONFIG["repetition_penalty"]
GREEDY = GEN_CONFIG["greedy"]


def top_k_top_p_filter(logits, top_k=0, top_p=0.0):
    """Filter a 1D logits tensor down to the top-k / top-p candidates."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k)[0][..., -1]
        logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)

    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_mask = cum_probs > top_p
        # always keep at least the top token
        sorted_mask[..., 0] = False

        idx_to_remove = sorted_idx[sorted_mask]
        logits = logits.clone()
        logits[idx_to_remove] = float("-inf")

    return logits


def apply_repetition_penalty(logits, generated_ids, penalty):
    """
    Divide the logit of any token already in `generated_ids` by `penalty`
    (for positive logits) or multiply (for negative), pushing recently-used
    tokens down. penalty == 1.0 is a no-op. Small models repeat a LOT
    without this.
    """
    if penalty == 1.0 or len(generated_ids) == 0:
        return logits

    logits = logits.clone()
    seen = torch.unique(generated_ids)
    seen_logits = logits[seen]
    logits[seen] = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
    return logits


@torch.no_grad()
def generate(model, idx, block_size, max_new_tokens, temperature, top_k, top_p,
             repetition_penalty, greedy, device):
    for _ in range(max_new_tokens):
        idx_cond = idx[-block_size:].unsqueeze(0).to(device)

        logits, _ = model(idx_cond)
        logits = logits[0, -1]

        logits = apply_repetition_penalty(logits, idx, repetition_penalty)

        if greedy:
            next_id = torch.argmax(logits).item()
        else:
            logits = logits / max(temperature, 1e-6)
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        idx = torch.cat([idx, torch.tensor([next_id])])

    return idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint["config"]  # architecture travels with the checkpoint

    sp = spm.SentencePieceProcessor()
    sp.load(checkpoint.get("tokenizer", "tokenizer.model"))

    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if "dataset_tokens" in checkpoint:
        print("Checkpoint trained on:")
        print(f"  {checkpoint['dataset_tokens']:,} tokens "
              f"({checkpoint.get('train_tokens', 0):,} train / "
              f"{checkpoint.get('val_tokens', 0):,} val)")
        print(f"  vocab={config['vocab_size']:,}  context={config['block_size']}  "
              f"embed={config['embed_size']}  heads={config['heads']}  layers={config['layers']}")
        if "best_val_loss" in checkpoint:
            print(f"  best val loss: {checkpoint['best_val_loss']:.4f}")
        print(f"  step: {checkpoint.get('step', '?')}")
        print()

    idx = torch.tensor(sp.encode(PROMPT), dtype=torch.long)

    out = generate(
        model, idx,
        block_size=config["block_size"],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        greedy=GREEDY,
        device=device,
    )

    print("Prompt:")
    print(PROMPT)
    print()
    mode = "greedy" if GREEDY else f"temp={TEMPERATURE} top-k={TOP_K} top-p={TOP_P}"
    print(f"Mode: {mode}  repetition_penalty={REPETITION_PENALTY}  tokens={MAX_NEW_TOKENS}")
    print()
    print("Generated text:")
    print()
    print(sp.decode(out.tolist()))


if __name__ == "__main__":
    main()
