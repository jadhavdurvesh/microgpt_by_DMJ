"""
Saudade v4 — streaming tokenization to memory-mapped binary files.

This is the fix you already made to the v3 notebook, formalized as a real
script instead of a notebook cell: the corpus is read and tokenized in
chunks and written straight to train.bin/val.bin, so the full corpus is
never held in RAM at once — needed even more for v4's larger (10,000-book)
corpus than it was for v3's.

Output:
    train.bin, val.bin   — uint16 (vocab <= 65536) or uint32 token arrays
    data_manifest.json   — token counts, split sizes, dataset fingerprint

train.py memory-maps these files directly (np.memmap) rather than loading
them into RAM either.
"""

import argparse
import json
import os

import numpy as np
import sentencepiece as spm

from config import CONFIG, TRAIN_CONFIG
from utils import dataset_fingerprint

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB of text at a time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model")
    parser.add_argument("--train-out", type=str, default=TRAIN_CONFIG["train_bin"])
    parser.add_argument("--val-out", type=str, default=TRAIN_CONFIG["val_bin"])
    parser.add_argument("--val-split", type=float, default=TRAIN_CONFIG["val_split"])
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()

    if vocab_size != CONFIG["vocab_size"]:
        raise ValueError(
            f"tokenizer vocab_size={vocab_size} does not match "
            f"CONFIG['vocab_size']={CONFIG['vocab_size']}. "
            f"Retrain the tokenizer or update config.py."
        )

    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    data_size = os.path.getsize(args.data)

    print("=" * 70)
    print("SAUDADE V4 — STREAMING DATASET TOKENIZATION")
    print("=" * 70)
    print(f"Dataset   : {args.data}  ({data_size / (1024**2):.2f} MB)")
    print(f"Tokenizer : {args.tokenizer}  (vocab={vocab_size:,}, dtype={dtype.__name__})")
    print()
    print("Streaming — the corpus is never fully loaded into RAM.")
    print()

    tmp_path = args.train_out + ".tmp"
    total_tokens = 0
    total_chars = 0
    chunks = 0

    with open(args.data, "r", encoding="utf-8", errors="replace") as src, \
         open(tmp_path, "wb") as dst:

        while True:
            text = src.read(CHUNK_SIZE)
            if not text:
                break

            chunks += 1
            total_chars += len(text)

            ids = sp.encode(text, out_type=int)
            arr = np.asarray(ids, dtype=dtype)
            arr.tofile(dst)
            total_tokens += len(arr)

            if chunks % 10 == 0:
                print(f"Chunks: {chunks:6d} | Characters: {total_chars:12,d} | "
                      f"Tokens: {total_tokens:12,d}")

    print()
    print(f"Tokenization complete: {total_tokens:,} tokens from {total_chars:,} characters")
    print()

    # split the flat token file into train/val without loading it into RAM
    all_tokens = np.memmap(tmp_path, dtype=dtype, mode="r")
    split_idx = int(len(all_tokens) * (1 - args.val_split))

    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    train_tokens.tofile(args.train_out)
    val_tokens.tofile(args.val_out)

    os.remove(tmp_path)

    manifest = {
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "total_tokens": int(total_tokens),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "val_split": args.val_split,
        "source_data": os.path.abspath(args.data),
        "tokenizer": os.path.abspath(args.tokenizer),
        "dataset_fingerprint": dataset_fingerprint(args.data),
    }
    with open("data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Train file : {args.train_out}  ({len(train_tokens):,} tokens, "
          f"{os.path.getsize(args.train_out) / (1024**2):.2f} MB)")
    print(f"Val file   : {args.val_out}  ({len(val_tokens):,} tokens, "
          f"{os.path.getsize(args.val_out) / (1024**2):.2f} MB)")
    print(f"Manifest   : data_manifest.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
