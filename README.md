# Saudade v4

The "Option A" upgrade from the roadmap: a modern block (RoPE, RMSNorm,
SwiGLU, PyTorch's fused attention kernel) on a deliberately still-small
model (~20-30M params), trained on a bigger, quality-checked corpus. The
question v4 answers is *"does the modern block actually help, at roughly
the same scale?"* — not "how big can we make it."

**v3 is frozen as the baseline.** This lives in its own `saudade_v4/`
directory; nothing here touches `saudade_v3/`. v3's final numbers
(8.9M params, 98.55M tokens, best val loss **3.7963**) are what v4 needs to
beat before anything from here graduates past "experiment."

## Full history: v1 → v2 → v3 → v4

v1 and v2 predate this repo's checkpoint-metadata/logging conventions, so
their numbers below come from reading their actual training scripts
directly (`microgpt_by_DMJ-master.zip` for v1, the v2 Kaggle notebook for
v2) rather than from a saved run report. Two honest gaps: **v2's Kaggle
notebook has no saved cell outputs**, so there's no recorded final loss or
measured corpus token count for it — only its architecture, which is fully
determined by the code. And v2's notebook only overrode `train.py`; it left
`train_tokenizer.py` as whatever was in the repo at the time, which is the
2,000-vocab tokenizer also used by v1 — noted below as an assumption, not
a confirmed fact.

| | v1 (original) | v2 | v3 | v4 |
|---|---|---|---|---|
| corpus | small sample corpus (~108K words) | "100-book corpus" (per notebook; not measured) | MicroCorpus v2: 1,033 books, 98.55M actual tokens | 10,000 books, quality-ranked |
| tokenizer | 2,000 vocab BPE | 2,000 vocab BPE *(assumed — not overridden in the v2 notebook)* | 16,000 vocab BPE | benchmarked 16K/32K/50K |
| context | 64 | 256 | 256 | 512 |
| embedding | 64 | 512 | 256 | 384 |
| heads | 4 | 8 | 8 | 8 |
| layers | 2 | 8 | 6 | 8 |
| parameters | ~0.36M | ~27.3M *(bigger than v3, on a much smaller tokenizer)* | ~8.9M | ~26.5M |
| position encoding | none | none | learned embeddings | RoPE |
| normalization | LayerNorm | LayerNorm | LayerNorm | RMSNorm |
| FFN | ReLU, 4x | ReLU, 4x | GELU, 4x | SwiGLU, ~2.67x |
| attention | hand-written | hand-written | hand-written | `scaled_dot_product_attention` |
| optimizer | Adam, fixed lr=0.001 | Adam, fixed lr=0.001 | AdamW, warmup+cosine | AdamW, warmup+cosine |
| dropout | none | none | 0.1 | 0.1 |
| weight tying | no | no | yes | yes |
| train/val split | none (trains on 100%) | none | 90/10 (v3 notebook fix: ~99/1) | 98/2, memory-mapped |
| dataset loading | full corpus into RAM | full corpus into RAM | full corpus into RAM (until the streaming-tokenization fix) | streaming → memory-mapped from the start |
| multi-GPU | no | yes (`DataParallel`) | yes (`DataParallel`) | yes (`DataParallel`) |
| checkpointing | final only | final only | periodic + best-val-loss, resumable | periodic + best-val-loss, resumable |
| progress tracking | steps, printed every 500 | steps, printed every 500 | steps | steps + tokens seen |
| training steps | 10,000 (fixed) | 10,000 (fixed) | 24,000 (~3 passes) | measured from actual token count |
| generation | temperature only, blocking `input()` | *(not rewritten — inherited v1's `generate.py`, and would have hit the size-mismatch bug against v2's bigger `train.py`)* | temp + top-k + top-p + repetition penalty + greedy | same as v3 |
| final / best val loss | not tracked (no val split) | not recorded (no saved outputs) | best 3.7963 | TBD |

The v2 row is the clearest illustration of why v3 introduced a single
shared `config.py`: v2 quietly became a **27M-parameter model on a
2,000-word vocabulary** — bigger than v3 ended up being — with no
positional embeddings, no val loss, and a `generate.py` that was never
updated to match, so it would have failed with a state-dict size mismatch
the moment anyone tried to actually generate from it.

## What changed vs. v3

| | v3 | v4 |
|---|---|---|
| corpus | 1,033 books, ~98.55M tokens | 10,000 books (quality-ranked, per MicroCorpus-Builder v3) |
| tokenizer | 16K, chosen without comparison | benchmarked across 16K/32K/50K (`tokenizer_benchmark.py`) |
| context | 256 | 512 |
| position encoding | learned embeddings | RoPE |
| normalization | LayerNorm | RMSNorm |
| FFN | GELU, 4x expansion | SwiGLU, ~2.67x expansion (LLaMA-style sizing) |
| attention | hand-written QK^T/softmax/V | `torch.nn.functional.scaled_dot_product_attention` |
| model size | ~8.9M params (256 embed / 8 heads / 6 layers) | ~20-30M params, "Option A": 384 embed / 8 heads / 8 layers |
| dataset loading | full corpus tokenized into RAM (the thing you had to fix) | streaming tokenization straight to memory-mapped `train.bin`/`val.bin` — corpus never fully in RAM |
| progress tracking | steps only | steps **and** tokens seen |
| checkpoint metadata | config, tokenizer path, step, losses | + git commit, dataset fingerprint, tokens seen, full train config |
| hyperparameter changes | edit `config.py` | CLI flags (`--batch-size`, `--steps`, etc.) override `TRAIN_CONFIG` without touching the file |
| evaluation | loss + eyeballing one sample | `evaluate.py`: perplexity + 7 category prompts + repetition-rate / unique-token-ratio per category |
| experiment tracking | none | `--experiment-name` snapshots config + log + metrics into `experiments/<name>/` |

## Deliberately NOT in v4

Same discipline as v3: no GQA, no MoE, no KV-cache, no instruction tuning.
Those (and RoPE/SwiGLU's bigger sibling ideas, if v4 justifies them) are v5
material. Instruction tuning specifically is its own later training stage
on top of a good v4 base — not something to fold into pretraining.

## Files

- `config.py` — architecture (`CONFIG`), training (`TRAIN_CONFIG`), generation
  (`GEN_CONFIG`), and evaluation (`EVAL_CONFIG`) settings. `vocab_size`
  defaults to a placeholder — run `tokenizer_benchmark.py` on the real v4
  corpus and set it deliberately before training for real.
- `model.py` — `SaudadeV4`: RoPE + RMSNorm + SwiGLU + SDPA attention,
  optional gradient checkpointing, tied embeddings, scaled residual-branch
  init (same discipline as v3, matters more at 8 layers).
- `tokenizer_benchmark.py` — trains throwaway tokenizers at several vocab
  sizes and reports compression stats, so 32K (or whatever) is a measured
  choice, not a guess.
- `train_tokenizer.py` — trains the real tokenizer at `CONFIG["vocab_size"]`.
- `prepare_data.py` — streams the corpus straight into memory-mapped
  `train.bin`/`val.bin` (uint16, since vocab ≤ 65536) — this is your v3 RAM
  fix, built in from the start instead of patched in afterward.
- `train.py` — the training loop. Memory-maps `train.bin`/`val.bin`
  (`np.memmap`, never loads them into RAM), tracks tokens seen alongside
  steps, writes richer checkpoint metadata, and accepts CLI overrides for
  the training knobs.
- `generate.py` — same generation feature set as v3 (temperature, top-k,
  top-p, repetition penalty, greedy mode); loads architecture from the
  checkpoint, same as v3.
- `evaluate.py` — runs 7 fixed category prompts (STORY, DIALOGUE,
  DESCRIPTION, CONTINUATION, LONG_CONTEXT, RARE_WORDS, REPETITION),
  reports perplexity plus per-category repetition-rate and
  unique-token-ratio, writes `evaluation.json`.
- `utils.py` — git commit lookup, a cheap large-file fingerprint (hashes a
  bounded sample, not the whole multi-hundred-MB corpus, on every
  checkpoint), and the experiment-snapshot helper.

## Running it

```bash
pip install -r requirements.txt   # same as v3: torch, sentencepiece, numpy

# 1. Put your 10,000-book corpus in data.txt

# 2. Benchmark tokenizer vocab sizes, then set CONFIG["vocab_size"] in config.py
python tokenizer_benchmark.py --vocab-sizes 16000 32000 50000

# 3. Train the real tokenizer at the chosen vocab size
python train_tokenizer.py

# 4. Tokenize the corpus into memory-mapped train.bin/val.bin
python prepare_data.py

# 5. Train (config.py's steps/batch_size are placeholders — check train.py's
#    startup banner, which recomputes steps/pass and estimated passes from
#    the ACTUAL token count in train.bin, same idea as v3's sanity check)
python train.py

#    override a knob without editing config.py:
python train.py --batch-size 16 --steps 50000

#    resume after a disconnect:
python train.py --resume checkpoints/checkpoint_20000.pt

#    snapshot this run into experiments/:
python train.py --experiment-name v4_baseline

# 6. Generate
python generate.py

# 7. Evaluate against the fixed prompt suite
python evaluate.py --checkpoint checkpoints/checkpoint_best.pt
```

## Notes

- **Gradient checkpointing**: `CONFIG["gradient_checkpointing"]` is off by
  default. The 8-layer/384-dim model is small enough that a 2xT4 should
  handle a reasonable batch size without it — flip it on only if you hit
  OOM, since it trades ~30% speed for lower activation memory.
- **`val_split`** defaults to 2% (v3's notebook fix used 1%, which is thin
  for computing a stable val loss every eval — 2% is a bit safer while
  still leaving nearly all of a 10,000-book corpus for training).
- **Corpus quality**: this repo doesn't include the MicroCorpus-Builder v3
  pipeline (dedup, boilerplate/OCR-noise detection, quality scoring) —
  that's upstream of `data.txt`. `prepare_data.py` and `train.py` just
  assume you're handing them a corpus you already trust.
