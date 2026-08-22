# Saudade v3

A properly-engineered classic Transformer, evolved from the original
MiniGPT prototype in this repo. Still a small, from-scratch, decoder-only
model — no RoPE, GQA, MoE, Flash Attention, KV-cache, or SwiGLU. Those are
left for v4/v5. v3's question is: *how much better does the original
architecture get once it's engineered properly and trained on the real
MicroCorpus v2 (98,574,591 actual tokens, measured with the 16K tokenizer —
not the earlier word-count estimate)?*

With `batch_size=32`, `block_size=256` (8,192 tokens/step) and
`TRAIN_CONFIG["steps"]=36000`, that's ~12,033 steps per full pass over the
90% training split and ~3 passes total — `train.py` prints this breakdown
(and the real token count it's based on) at the start of every run, computed
from the actual corpus rather than hard-coded.

## What changed vs. the original

| | v1 (original) | v3 |
|---|---|---|
| context (`block_size`) | 64 | 256 |
| embedding (`embed_size`) | 64 | 256 |
| attention heads | 4 | 8 |
| layers | 2 | 6 |
| positional info | none | learned position embeddings |
| activation | ReLU | GELU |
| dropout | none | 0.1 |
| optimizer | Adam | AdamW + weight decay |
| LR schedule | fixed | warmup → cosine decay |
| gradient clipping | none | max norm 1.0 |
| precision | FP32 | mixed precision (bf16 autocast) |
| train/val split | none (trains on 100%) | 90/10 |
| checkpoints | final only | periodic, resumable |
| output/embedding weights | separate | tied |
| generation | temperature only, blocking `input()` | temperature + top-k + top-p, prompt set in-script |
| config | scattered constants | single `config.py`, saved into every checkpoint |
| residual-branch init | default | GPT-2-style scaled init (std / sqrt(2·layers)) |
| weight decay | none | AdamW, decay only on matrices (not biases/LayerNorm) |
| effective batch size | fixed at `batch_size` | gradient accumulation (`grad_accum_steps`) |
| checkpoint selection | last only | periodic + a tracked best-val-loss checkpoint |
| training visibility | loss numbers only | loss numbers + a qualitative text sample every N steps |
| logging | console only | console + `train_log.txt` |
| generation repetition | none | repetition penalty + optional greedy mode |

Files:

- `config.py` — the one place the architecture and training hyperparameters
  live. `train.py` and `generate.py` both import it, and it's saved inside
  every checkpoint, so a checkpoint can never end up paired with the wrong
  architecture.
- `model.py` — the `MiniGPT` model itself.
- `train_tokenizer.py` — trains a 16K-vocab BPE tokenizer on `data.txt`.
- `train.py` — training loop with everything above.
- `generate.py` — generation with temperature / top-k / top-p.

## Running it

```bash
pip install -r requirements.txt

# 1. Put your corpus in data.txt (MicroCorpus v2, or whatever you're using)
#    A small placeholder data.txt is included so the scripts run out of the box.

# 2. Train the tokenizer (16K BPE, must match config.py's vocab_size)
python train_tokenizer.py

# 3. Train the model
python train.py
#    resume a run:
python train.py --resume checkpoints/checkpoint_5000.pt

# 4. Generate
#    edit PROMPT / TEMPERATURE / TOP_K / TOP_P at the top of generate.py, then:
python generate.py
```

## Notes for Kaggle (2x T4)

- `train.py` auto-detects available GPUs and wraps the model in
  `nn.DataParallel` when more than one is visible — no changes needed.
- Mixed precision is on by default when CUDA is available.
- Checkpoints save every `TRAIN_CONFIG["checkpoint_interval"]` steps to
  `checkpoints/`, each with model, optimizer, and scaler state plus the
  step number — so a Kaggle disconnect only costs you back to the last
  checkpoint, not the whole run. Resume with `--resume`.
- Before committing to a full v3 training run, run the tokenizer once on
  the real MicroCorpus v2 and check the actual token count (Section 4 of
  the Kaggle notebook does this) — `config.py`'s `steps` is already set from
  the measured 98,574,591-token count (36,000 steps ≈ 3 passes), but if your
  corpus size changes, recompute and update `steps` accordingly.
- `train.py`'s startup banner reports dataset stats, model config, the
  training-step math (tokens/step, steps/pass, approx. passes), and
  optimizer settings all in one place — check it before committing to a
  many-hour run. Every checkpoint also stores `dataset_tokens`,
  `train_tokens`, and `val_tokens`, so `generate.py` (and anything else
  loading `saudade_v3.pt` later) can report what a model was actually
  trained on without needing to re-derive it.

## What each new knob does

- **`grad_accum_steps`** (config.py, `TRAIN_CONFIG`) — accumulates gradients
  over N micro-batches before stepping the optimizer, so effective batch
  size is `batch_size * grad_accum_steps` without needing more GPU memory.
- **`checkpoint_best.pt`** — always the checkpoint with the lowest val loss
  seen so far, updated every eval. Use this one for generation unless you
  have a specific reason to want a later (possibly slightly overfit) step.
- **`sample_interval` / `sample_prompt`** — every N steps, `train.py` prints
  a short generation from the current weights so you can watch it improve
  qualitatively, not just watch the loss number drop.
- **`repetition_penalty`** (generate.py) — small models loop a lot without
  this; 1.2 is a reasonable default, 1.0 disables it.
- **`greedy`** (generate.py) — ignores temperature/top-k/top-p and always
  picks the highest-probability token; deterministic, useful for debugging
  but produces the most repetitive output of any mode (as you'll see if you
  try it on an undertrained checkpoint).

## Deliberately not in v3

RoPE, GQA, MoE, a real Flash Attention kernel, KV-cache, SwiGLU, other
modern positional schemes, distributed training, a much larger model,
instruction tuning, RLHF. All candidates for v4/v5, once v3's numbers are in.
