"""
Saudade v3 — training script.

Upgrades over the original train.py (see project notes for the full list):
  train/val split, AdamW + warmup/cosine LR schedule, gradient clipping,
  mixed precision, periodic checkpoints (resumable), proper logging,
  and a single CONFIG shared with generate.py so the two can never drift
  out of sync.

Further training-quality upgrades (still no architecture changes):
  weight decay only applied to matrices (not biases/LayerNorm), gradient
  accumulation for a larger effective batch size, a tracked best-val-loss
  checkpoint alongside the periodic ones, a qualitative text sample printed
  every so often so you can watch generations improve (not just the loss
  number), and logging duplicated to a file.

Usage:
    python train.py                  # start from scratch
    python train.py --resume checkpoints/checkpoint_5000.pt
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from config import CONFIG, TRAIN_CONFIG
from model import MiniGPT


class Logger:
    """Prints to stdout and appends the same line to a log file."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def log(self, msg=""):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    """Linear warmup, then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def make_optimizer(model, weight_decay, lr):
    """
    Only decay 2D+ parameters (matrices — embeddings, linear weights).
    Biases and LayerNorm gains/offsets are 1D and shouldn't be decayed;
    decaying them gives a small but consistent regression in practice.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.mean().item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_text(model, sp, prompt, block_size, device, max_new_tokens=60, temperature=0.8, top_k=50):
    """Quick qualitative sample during training — greedy top-k, no repetition penalty."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint to resume from")
    parser.add_argument("--data", type=str, default="data.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    torch.manual_seed(TRAIN_CONFIG["seed"])

    logger = Logger(TRAIN_CONFIG["log_file"])
    log = logger.log

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0

    # ----------------------------
    # Tokenizer + data
    # ----------------------------
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    vocab_size = sp.get_piece_size()
    assert vocab_size == CONFIG["vocab_size"], (
        f"tokenizer vocab_size={vocab_size} does not match "
        f"CONFIG['vocab_size']={CONFIG['vocab_size']}. "
        f"Retrain the tokenizer or update config.py."
    )

    text = open(args.data, encoding="utf-8").read()
    tokens = sp.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)

    split = int((1 - TRAIN_CONFIG["val_split"]) * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # ----------------------------
    # Model
    # ----------------------------
    model = MiniGPT(CONFIG).to(device)
    raw_model = model  # keep an un-wrapped handle for state_dict / checkpointing

    if n_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = make_optimizer(raw_model, TRAIN_CONFIG["weight_decay"], TRAIN_CONFIG["max_lr"])

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_step = ckpt["step"] + 1
        log(f"Resumed from {args.resume} at step {start_step}")

    total_steps = TRAIN_CONFIG["steps"]
    block_size = CONFIG["block_size"]
    batch_size = TRAIN_CONFIG["batch_size"]
    grad_accum_steps = max(1, TRAIN_CONFIG["grad_accum_steps"])

    dataset_tokens = len(data)
    train_tokens = len(train_data)
    val_tokens = len(val_data)
    tokens_per_step = batch_size * grad_accum_steps * block_size
    steps_per_epoch = train_tokens / tokens_per_step
    estimated_passes = (total_steps * tokens_per_step) / train_tokens

    words = len(text.split())
    characters = len(text)
    val_pct = TRAIN_CONFIG["val_split"] * 100
    train_pct = 100 - val_pct
    mixed_precision = device == "cuda"

    log("=" * 60)
    log("SAUDADE V3".center(60))
    log("=" * 60)
    log()
    log("DATASET")
    log(f"Words              : {words:,}")
    log(f"Characters         : {characters:,}")
    log(f"Actual tokens      : {dataset_tokens:,}")
    log(f"  train tokens     : {train_tokens:,}")
    log(f"  val tokens       : {val_tokens:,}")
    log(f"Vocabulary         : {vocab_size:,}")
    log()
    log("MODEL")
    log(f"Embedding          : {CONFIG['embed_size']}")
    log(f"Context            : {CONFIG['block_size']}")
    log(f"Attention heads    : {CONFIG['heads']}")
    log(f"Transformer layers : {CONFIG['layers']}")
    log(f"FFN size           : {CONFIG['embed_size'] * 4}")
    log(f"Dropout            : {CONFIG['dropout']}")
    log(f"Position embedding : Yes")
    log(f"Weight tying       : Yes")
    log(f"Parameters         : {raw_model.num_params() / 1e6:.2f} M")
    log()
    log("TRAINING")
    log(f"Train split        : {train_pct:.0f}%")
    log(f"Validation split   : {val_pct:.0f}%")
    log(f"Batch size         : {batch_size}")
    log(f"Grad accum steps   : {grad_accum_steps}")
    log(f"Tokens / step      : {tokens_per_step:,}")
    log(f"Steps / pass       : ~{steps_per_epoch:,.0f}")
    log(f"Total steps        : {total_steps:,}")
    log(f"Approx. passes     : ~{estimated_passes:.2f}")
    log()
    log("OPTIMIZER")
    log(f"AdamW")
    log(f"Weight decay       : {TRAIN_CONFIG['weight_decay']}")
    log(f"Learning rate      : {TRAIN_CONFIG['max_lr']:.0e}")
    log(f"Warmup             : Yes ({TRAIN_CONFIG['warmup_steps']} steps)")
    log(f"Cosine decay       : Yes")
    log(f"Gradient clipping  : {TRAIN_CONFIG['grad_clip']}")
    log(f"Mixed precision    : {'Yes' if mixed_precision else 'No (CPU)'}")
    log()
    log("CHECKPOINTS")
    log(f"checkpoint_*.pt, checkpoint_best.pt -> {args.checkpoint_dir}/")
    log(f"saudade_v3.pt (final)")
    log()
    log("DEVICE")
    log(f"Device             : {device}")
    log(f"GPUs               : {n_gpus}")
    log()
    log("=" * 60)
    log()

    model.train()
    t0 = time.time()

    for step in range(start_step, total_steps):

        lr = get_lr(
            step,
            TRAIN_CONFIG["warmup_steps"],
            TRAIN_CONFIG["max_lr"],
            TRAIN_CONFIG["min_lr"],
            total_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = get_batch(train_data, block_size, batch_size, device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss.mean() / grad_accum_steps  # mean() reduces DataParallel's per-GPU losses

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), TRAIN_CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % TRAIN_CONFIG["eval_interval"] == 0 or step == total_steps - 1:
            losses = estimate_loss(
                model, train_data, val_data, block_size,
                batch_size, TRAIN_CONFIG["eval_iters"], device,
            )
            dt = time.time() - t0
            log(f"Step {step} / {total_steps}")
            log(f"Train Loss : {losses['train']:.4f}")
            log(f"Val Loss   : {losses['val']:.4f}")
            log(f"LR         : {lr:.6f}")
            log(f"Elapsed    : {dt:.1f}s")
            log()

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pt")
                torch.save({
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "step": step,
                    "loss": accum_loss,
                    "best_val_loss": best_val_loss,
                    "config": CONFIG,
                    "tokenizer": args.tokenizer,
                    "dataset_tokens": dataset_tokens,
                    "train_tokens": train_tokens,
                    "val_tokens": val_tokens,
                }, best_path)
                log(f"New best val loss {best_val_loss:.4f} — saved {best_path}")
                log()

        if step > 0 and step % TRAIN_CONFIG["sample_interval"] == 0:
            sample = sample_text(
                raw_model, sp, TRAIN_CONFIG["sample_prompt"], block_size, device,
            )
            log("--- sample ---")
            log(sample)
            log("--------------")
            log()

        if step > 0 and (step % TRAIN_CONFIG["checkpoint_interval"] == 0 or step == total_steps - 1):
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save({
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "step": step,
                "loss": accum_loss,
                "best_val_loss": best_val_loss,
                "config": CONFIG,
                "tokenizer": args.tokenizer,
                "dataset_tokens": dataset_tokens,
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
            }, ckpt_path)
            log(f"Saved checkpoint: {ckpt_path}")
            log()

    # final save, in the same format generate.py expects
    torch.save({
        "model_state": raw_model.state_dict(),
        "config": CONFIG,
        "tokenizer": args.tokenizer,
        "step": total_steps - 1,
        "dataset_tokens": dataset_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "best_val_loss": best_val_loss,
    }, "saudade_v3.pt")

    log("Training complete. Final model saved to saudade_v3.pt")
    log(f"Best val loss seen: {best_val_loss:.4f} "
        f"(checkpoints/checkpoint_best.pt)")


if __name__ == "__main__":
    main()
