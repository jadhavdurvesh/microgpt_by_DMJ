"""
Saudade v4 — model configuration.

Single source of truth for architecture, same principle as v3: train.py,
generate.py, and evaluate.py all import CONFIG from here, and it's saved
into every checkpoint so a checkpoint can never be loaded with the wrong
architecture.

v4 is "Option A" from the roadmap — the conservative upgrade, not the
ambitious one. Bigger corpus and a properly modern block (RoPE, RMSNorm,
SwiGLU, PyTorch's fused attention kernel), but the model itself stays
deliberately small (~20-30M params) until we know this architecture is
actually better than v3, not just different.

v3 stays frozen as the baseline — this file lives in a separate saudade_v4/
directory and never touches saudade_v3/.
"""

CONFIG = {
    # --- picked after running tokenizer_benchmark.py on the v4 corpus ---
    # v3 used 16K without benchmarking alternatives. Benchmark 16K/32K/50K
    # on the actual 10,000-book corpus before committing to a number —
    # this default is a placeholder, not a decision.
    "vocab_size": 32000,

    "block_size": 512,       # was 256 in v3
    "embed_size": 384,       # was 256 in v3          -- Option A
    "heads": 8,
    "layers": 8,              # was 6 in v3            -- Option A
    "dropout": 0.1,

    # SwiGLU hidden dim. None = auto-compute as round_up(2/3 * 4*embed_size, 32)
    # (the standard LLaMA-style SwiGLU sizing, which keeps compute roughly
    # comparable to a 4x ReLU/GELU FFN despite the extra gate matrix).
    "ffn_hidden": None,

    "rope_theta": 10000.0,

    # Trades ~30% training speed for a big activation-memory reduction —
    # worth it if the 8-layer/384-dim model doesn't fit T4 memory at a
    # reasonable batch size. Off by default; flip on if you hit OOM.
    "gradient_checkpointing": False,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "grad_accum_steps": 1,

    # Set this from the ACTUAL token count of your v4 corpus (prepare_data.py
    # prints it, and train.py's startup banner recomputes passes/steps from
    # whatever train.bin actually contains — this default is not load-bearing,
    # just a starting point before you've measured the 10,000-book corpus).
    "steps": 60000,

    "eval_interval": 250,
    "eval_iters": 50,

    "checkpoint_interval": 1000,
    "sample_interval": 1000,
    "sample_prompt": "The night was quiet and",

    "warmup_steps": 1000,
    "max_lr": 3e-4,
    "min_lr": 3e-5,
    "weight_decay": 0.1,
    "grad_clip": 1.0,

    "val_split": 0.02,       # v3's notebook fix used 1% — 2% is a bit safer at this corpus size
    "seed": 1337,
    "log_file": "train_log.txt",

    # Data files produced by prepare_data.py (memory-mapped — the corpus is
    # NOT loaded into RAM, same fix you already made for v3).
    "train_bin": "train.bin",
    "val_bin": "val.bin",

    # Optional: name a subfolder under experiments/ to snapshot config +
    # logs + final metrics into after training, for the experiment-tracking
    # workflow from the roadmap (experiments/v4_rope/, experiments/v4_swiglu/, ...).
    "experiment_name": None,
}

GEN_CONFIG = {
    "max_new_tokens": 500,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "greedy": False,
}

EVAL_CONFIG = {
    "max_new_tokens": 80,
    "temperature": 0.8,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "repetition_window": 4,   # n-gram size used for the repetition-rate metric
}
