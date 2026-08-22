"""
Trains the Saudade v3 tokenizer: SentencePiece BPE, 16K vocab.

Run this on MicroCorpus v2 (data.txt) before training the model — the
resulting tokenizer.model/.vocab are what train.py and generate.py load,
and CONFIG["vocab_size"] must match the vocab_size used here.
"""

import sentencepiece as spm
from config import CONFIG

spm.SentencePieceTrainer.train(
    input="data.txt",
    model_prefix="tokenizer",
    vocab_size=CONFIG["vocab_size"],
    model_type="bpe",
    character_coverage=0.9995,
    input_sentence_size=5_000_000,
    shuffle_input_sentence=True,
)

print("Tokenizer trained: tokenizer.model / tokenizer.vocab")
print(f"Vocab size: {CONFIG['vocab_size']}")
