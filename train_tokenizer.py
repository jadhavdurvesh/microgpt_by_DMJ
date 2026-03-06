import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='data.txt',
    model_prefix='tokenizer',
    vocab_size=2000,
    model_type='bpe'
)