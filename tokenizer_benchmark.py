"""
Saudade v4 — tokenizer benchmark.

Trains a throwaway tokenizer at each candidate vocab size and reports the
numbers you'd actually want before picking one — compression (tokens/word,
tokens/char), and how well each one round-trips a held-out sample. This is
the "benchmark, don't just guess bigger is better" step from the roadmap.

Doesn't decide for you. Prints a table; you set CONFIG["vocab_size"] in
config.py yourself based on it, then run train_tokenizer.py for real.

Usage:
    python tokenizer_benchmark.py                    # benchmarks 16000,32000,50000
    python tokenizer_benchmark.py --vocab-sizes 16000 24000 32000
"""

import argparse
import os
import tempfile

import sentencepiece as spm

HOLDOUT_SAMPLE = (
    "The night was quiet, and the old house stood alone beneath the moon. "
    "She whispered, \"I don't understand why he never wrote back,\" and the "
    "candlelight flickered against the peculiar, weather-beaten door. "
    "Somewhere in the distance, a dog barked twice, then silence returned."
)


def benchmark_one(data_file, vocab_size, sentence_sample):
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "bench")
        spm.SentencePieceTrainer.train(
            input=data_file,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            input_sentence_size=sentence_sample,
            shuffle_input_sentence=True,
        )

        sp = spm.SentencePieceProcessor()
        sp.load(prefix + ".model")

        # compression on the holdout sample
        ids = sp.encode(HOLDOUT_SAMPLE, out_type=int)
        words = len(HOLDOUT_SAMPLE.split())
        chars = len(HOLDOUT_SAMPLE)

        # round-trip check
        decoded = sp.decode(ids)
        roundtrip_ok = decoded.strip() == HOLDOUT_SAMPLE.strip()

        return {
            "vocab_size": sp.get_piece_size(),
            "holdout_tokens": len(ids),
            "tokens_per_word": len(ids) / max(words, 1),
            "tokens_per_char": len(ids) / max(chars, 1),
            "roundtrip_ok": roundtrip_ok,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.txt")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[16000, 32000, 50000])
    parser.add_argument("--sentence-sample", type=int, default=500_000,
                         help="sentences sampled per candidate — kept small since this runs "
                              "N times and is only for comparison, not the final tokenizer")
    args = parser.parse_args()

    print("=" * 70)
    print("SAUDADE V4 TOKENIZER BENCHMARK")
    print("=" * 70)
    print()
    print(f"Dataset: {args.data}")
    print(f"Candidates: {args.vocab_sizes}")
    print()

    results = []
    for vs in args.vocab_sizes:
        print(f"Training candidate tokenizer: vocab_size={vs:,} ...")
        r = benchmark_one(args.data, vs, args.sentence_sample)
        results.append(r)
        print(f"  done.")
        print()

    print("=" * 70)
    print(f"{'Vocab':>8} | {'Tokens/word':>12} | {'Tokens/char':>12} | {'Roundtrip':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['vocab_size']:>8,} | {r['tokens_per_word']:>12.3f} | "
              f"{r['tokens_per_char']:>12.3f} | {str(r['roundtrip_ok']):>9}")
    print("=" * 70)
    print()
    print("Lower tokens/word and tokens/char = better compression (fewer tokens per")
    print("sequence, cheaper training/inference at the same context length) but a")
    print("larger vocabulary also means a bigger embedding/output matrix and more")
    print("params spent on rare pieces. Pick the smallest vocab whose compression")
    print("is 'good enough' rather than defaulting to the largest.")


if __name__ == "__main__":
    main()
