"""
Saudade v4 — training script.

Carries forward everything from v3 (AdamW + warmup/cosine, grad clipping,
mixed precision, grad accumulation, weight-decay param grouping, periodic +
best-val-loss checkpoints, qualitative samples, file logging) and adds the
v4-specific roadmap items:

  - memory-mapped train.bin/val.bin (never loads the corpus into RAM —
    the fix you made to v3, now built in from the start)
  - token-based tracking (tokens seen, not just steps) alongside step-based
  - richer checkpoint metadata: git commit, dataset fingerprint, tokenizer
    info, tokens seen — the "make experiments reproducible" item
  - CLI overrides for the training knobs (--batch-size, --steps, etc.) so
    you don't have to hand-edit config.py from a Kaggle cell anymore
  - optional gradient checkpointing (config.py: CONFIG["gradient_checkpointing"])
  - optional experiment snapshot at the end (--experiment-name)

Usage:
    python train.py
    python train.py --batch-size 16 --steps 40000
    python train.py --resume checkpoints/checkpoint_5000.pt
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from config import CONFIG, TRAIN_CONFIG
from model import SaudadeV4
from utils import get_git_commit, dataset_fingerprint, save_experiment_snapshot


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def log(self, msg=""):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


class BinaryTokenDataset:
    """Memory-mapped token array — only the requested batch indices are ever
    actually read into RAM, the file itself stays on disk."""

    def __init__(self, path, dtype):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run prepare_data.py first."
            )
        self.data = np.memmap(path, dtype=dtype, mode="r")

    def __len__(self):
        return len(self.data)

    def get_batch(self, block_size, batch_size, device):
        ix = np.random.randint(0, len(self.data) - block_size - 1, size=batch_size)
        x = np.stack([self.data[i:i + block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + block_size + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x).to(device, non_blocking=True)
        y = torch.from_numpy(y).to(device, non_blocking=True)
        return x, y


def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def make_optimizer(model, weight_decay, lr):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


@torch.no_grad()
def estimate_loss(model, train_ds, val_ds, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for name, ds in (("train", train_ds), ("val", val_ds)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = ds.get_batch(block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.mean().item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_text(model, sp, prompt, block_size, device, max_new_tokens=60, temperature=0.8, top_k=50):
    model.eval()
    idx = torch.tensor(sp.encode(prompt), dtype=torch.long)

    for _ in range(max_new_tokens):
        idx_cond = idx[-block_size:].unsqueeze(0).to(device)
        logits, _ = model(idx_cond)
        logits = logits[0, -1] / max(temperature, 1e-6)

        top_k_eff = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k_eff)[0][..., -1]
        logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        idx = torch.cat([idx, torch.tensor([next_id])])

    model.train()
    return sp.decode(idx.tolist())


def build_checkpoint(raw_model, optimizer, scaler, step, tokens_seen, loss,
                      best_val_loss, args, dataset_meta, git_commit):
    return {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "best_val_loss": best_val_loss,
        "config": CONFIG,
        "train_config": TRAIN_CONFIG,
        "tokenizer": args.tokenizer,
        "dataset_manifest": dataset_meta,
        "git_commit": git_commit,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Saudade v4")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--train-bin", type=str, default=TRAIN_CONFIG["train_bin"])
    parser.add_argument("--val-bin", type=str, default=TRAIN_CONFIG["val_bin"])
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--experiment-name", type=str, default=TRAIN_CONFIG["experiment_name"])

    # CLI overrides for TRAIN_CONFIG — no more hand-editing config.py from a
    # Kaggle cell just to change batch size or step count.
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--sample-interval", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)

    args = parser.parse_args()

    # apply CLI overrides on top of config.py's TRAIN_CONFIG
    overrides = {
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "sample_interval": args.sample_interval,
        "warmup_steps": args.warmup_steps,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
    }
    for key, value in overrides.items():
        if value is not None:
            TRAIN_CONFIG[key] = value

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.manual_seed(TRAIN_CONFIG["seed"])

    logger = Logger(TRAIN_CONFIG["log_file"])
    log = logger.log

    overridden = {k: v for k, v in overrides.items() if v is not None}
    if overridden:
        log(f"CLI overrides applied: {overridden}")
        log()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.get_piece_size()
    assert vocab_size == CONFIG["vocab_size"], (
        f"tokenizer vocab_size={vocab_size} != CONFIG['vocab_size']={CONFIG['vocab_size']}"
    )

    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    train_ds = BinaryTokenDataset(args.train_bin, dtype)
    val_ds = BinaryTokenDataset(args.val_bin, dtype)

    dataset_meta = {
        "train_tokens": len(train_ds),
        "val_tokens": len(val_ds),
        "train_fingerprint": dataset_fingerprint(args.train_bin),
        "val_fingerprint": dataset_fingerprint(args.val_bin),
    }
    git_commit = get_git_commit()

    model = SaudadeV4(CONFIG).to(device)
    raw_model = model

    if n_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = make_optimizer(raw_model, TRAIN_CONFIG["weight_decay"], TRAIN_CONFIG["max_lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        tokens_seen = ckpt.get("tokens_seen", 0)
        start_step = ckpt["step"] + 1
        log(f"Resumed from {args.resume} at step {start_step} ({tokens_seen:,} tokens seen)")

    total_steps = TRAIN_CONFIG["steps"]
    block_size = CONFIG["block_size"]
    batch_size = TRAIN_CONFIG["batch_size"]
    grad_accum_steps = max(1, TRAIN_CONFIG["grad_accum_steps"])

    tokens_per_step = batch_size * grad_accum_steps * block_size
    steps_per_epoch = dataset_meta["train_tokens"] / tokens_per_step
    estimated_passes = (total_steps * tokens_per_step) / dataset_meta["train_tokens"]

    log("=" * 60)
    log("SAUDADE V4".center(60))
    log("=" * 60)
    log()
    log("DATASET")
    log(f"Train tokens       : {dataset_meta['train_tokens']:,}")
    log(f"Val tokens         : {dataset_meta['val_tokens']:,}")
    log(f"Vocabulary         : {vocab_size:,}")
    log()
    log("MODEL")
    log(f"Embedding          : {CONFIG['embed_size']}")
    log(f"Context            : {CONFIG['block_size']}")
    log(f"Attention heads    : {CONFIG['heads']}")
    log(f"Transformer layers : {CONFIG['layers']}")
    log(f"Position encoding  : RoPE")
    log(f"Normalization      : RMSNorm")
    log(f"FFN                : SwiGLU")
    log(f"Attention kernel   : scaled_dot_product_attention")
    log(f"Gradient checkpoint: {CONFIG.get('gradient_checkpointing', False)}")
    log(f"Parameters         : {raw_model.num_params() / 1e6:.2f} M")
    log()
    log("TRAINING")
    log(f"Batch size         : {batch_size}")
    log(f"Grad accum steps   : {grad_accum_steps}")
    log(f"Tokens / step      : {tokens_per_step:,}")
    log(f"Steps / pass       : ~{steps_per_epoch:,.0f}")
    log(f"Total steps        : {total_steps:,}")
    log(f"Approx. passes     : ~{estimated_passes:.2f}")
    log()
    log("REPRODUCIBILITY")
    log(f"Git commit         : {git_commit or 'n/a'}")
    log(f"Train fingerprint  : {dataset_meta['train_fingerprint']}")
    log()
    log(f"Device             : {device}   GPUs: {n_gpus}")
    log()
    log("=" * 60)
    log()

    model.train()
    t0 = time.time()

    for step in range(start_step, total_steps):
        lr = get_lr(step, TRAIN_CONFIG["warmup_steps"], TRAIN_CONFIG["max_lr"],
                    TRAIN_CONFIG["min_lr"], total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            x, y = train_ds.get_batch(block_size, batch_size, device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss.mean() / grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), TRAIN_CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % TRAIN_CONFIG["eval_interval"] == 0 or step == total_steps - 1:
            losses = estimate_loss(model, train_ds, val_ds, block_size, batch_size,
                                    TRAIN_CONFIG["eval_iters"], device)
            dt = time.time() - t0
            log(f"Step {step} / {total_steps}   (tokens seen: {tokens_seen:,})")
            log(f"Train Loss : {losses['train']:.4f}")
            log(f"Val Loss   : {losses['val']:.4f}")
            log(f"LR         : {lr:.6f}")
            log(f"Elapsed    : {dt:.1f}s")
            log()

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pt")
                torch.save(build_checkpoint(raw_model, optimizer, scaler, step, tokens_seen,
                                             accum_loss, best_val_loss, args, dataset_meta, git_commit),
                           best_path)
                log(f"New best val loss {best_val_loss:.4f} — saved {best_path}")
                log()

        if step > 0 and step % TRAIN_CONFIG["sample_interval"] == 0:
            sample = sample_text(raw_model, sp, TRAIN_CONFIG["sample_prompt"], block_size, device)
            log("--- sample ---")
            log(sample)
            log("--------------")
            log()

        if step > 0 and (step % TRAIN_CONFIG["checkpoint_interval"] == 0 or step == total_steps - 1):
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save(build_checkpoint(raw_model, optimizer, scaler, step, tokens_seen,
                                         accum_loss, best_val_loss, args, dataset_meta, git_commit),
                       ckpt_path)
            log(f"Saved checkpoint: {ckpt_path}")
            log()

    final_ckpt = {
        "model_state": raw_model.state_dict(),
        "config": CONFIG,
        "tokenizer": args.tokenizer,
        "step": total_steps - 1,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "dataset_manifest": dataset_meta,
        "git_commit": git_commit,
    }
    torch.save(final_ckpt, "saudade_v4.pt")

    log("Training complete. Final model saved to saudade_v4.pt")
    log(f"Best val loss seen: {best_val_loss:.4f}  |  tokens seen: {tokens_seen:,}")

    if args.experiment_name:
        exp_dir = save_experiment_snapshot(
            args.experiment_name,
            base_dir=".",
            files_to_copy=["config.py", TRAIN_CONFIG["log_file"], "data_manifest.json"],
            metrics={
                "best_val_loss": best_val_loss,
                "final_step": total_steps - 1,
                "tokens_seen": tokens_seen,
                "params": raw_model.num_params(),
                "git_commit": git_commit,
            },
        )
        log(f"Experiment snapshot saved to {exp_dir}")


if __name__ == "__main__":
    main()
