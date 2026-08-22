"""
Saudade v3 — model configuration.

This is the ONLY place the architecture is defined. train.py and generate.py
both import CONFIG from here, and the trained checkpoint stores a copy of
this dict alongside its weights — so generation can never end up mismatched
with the model that produced it (the bug that hit the old generate.py).
"""

CONFIG = {
    "vocab_size": 16000,     # must match train_tokenizer.py's vocab_size
    "block_size": 256,       # context length
    "embed_size": 256,       # embedding / residual stream dimension
    "heads": 8,              # attention heads (256 / 8 = 32 dims/head)
    "layers": 6,             # transformer blocks
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "grad_accum_steps": 1,     # >1 simulates a larger batch (batch_size * grad_accum_steps)
    # 98,574,591 actual tokens (MicroCorpus v2, 16K tokenizer) / (batch_size * block_size=8,192
    # tokens/step) ≈ 12,033 steps/pass. 36,000 steps ≈ 3 passes over the corpus — see the
    # "estimated passes" line train.py prints at startup, computed from the real token count
    # rather than hard-coded.
    "steps": 36000,
    "eval_interval": 250,      # how often to compute train/val loss
    "eval_iters": 50,          # batches averaged per eval
    "checkpoint_interval": 1000,
    "sample_interval": 1000,   # how often to print a qualitative sample during training
    "sample_prompt": "The night was quiet and",
    "warmup_steps": 500,       # scaled up for the longer 36k-step run (was 200 for 10k steps)
    "max_lr": 3e-4,
    "min_lr": 3e-5,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "val_split": 0.1,
    "seed": 1337,
    "log_file": "train_log.txt",
}

GEN_CONFIG = {
    "max_new_tokens": 500,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,  # 1.0 = off; >1.0 discourages repeating recent tokens
    "greedy": False,            # if True, ignore temperature/top-k/top-p and always take argmax
}
