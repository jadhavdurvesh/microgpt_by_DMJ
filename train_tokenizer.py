"""
Saudade v4 — trains the final tokenizer at CONFIG["vocab_size"].

Run tokenizer_benchmark.py first to pick a vocab size (16K/32K/50K), set it
in config.py, then run this to produce the tokenizer train.py/prepare_data.py
actually use.
"""

import os
import sys
import sentencepiece as spm

from config import CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.txt")
MODEL_PREFIX = os.path.join(BASE_DIR, "tokenizer")

VOCAB_SIZE = int(CONFIG["vocab_size"])
CHARACTER_COVERAGE = 0.9995
# Sampling sentences rather than reading the whole corpus keeps this
# RAM-safe on a large (10,000-book) v4 corpus, same reasoning as v3's fix.
INPUT_SENTENCE_SIZE = 2_000_000
NUM_THREADS = min(8, os.cpu_count() or 1)

print("=" * 70)
print("SAUDADE V4 TOKENIZER TRAINING")
print("=" * 70)
print()
print(f"Dataset          : {DATA_FILE}")
print(f"Vocabulary size  : {VOCAB_SIZE:,}")
print(f"Character cover  : {CHARACTER_COVERAGE}")
print(f"Sentence sample  : {INPUT_SENTENCE_SIZE:,}")
print(f"CPU threads      : {NUM_THREADS}")
print()

if not os.path.isfile(DATA_FILE):
    print("ERROR: data.txt was not found.")
    sys.exit(1)

spm.SentencePieceTrainer.train(
    input=DATA_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=CHARACTER_COVERAGE,
    input_sentence_size=INPUT_SENTENCE_SIZE,
    shuffle_input_sentence=True,
    num_threads=NUM_THREADS,
)

sp = spm.SentencePieceProcessor()
sp.load(MODEL_PREFIX + ".model")

print()
print(f"Tokenizer trained: {MODEL_PREFIX}.model / {MODEL_PREFIX}.vocab")
print(f"Configured vocab : {VOCAB_SIZE:,}")
print(f"Actual vocab     : {sp.get_piece_size():,}")
