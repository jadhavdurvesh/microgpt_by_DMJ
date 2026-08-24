# Contributing to Saudade

Thank you for your interest in contributing to Saudade.

Saudade is a small-scale language-model research and engineering project maintained by DMJ Labs. Contributions are welcome in code, documentation, testing, evaluation, benchmarking, tooling, and research discussion.

## Before You Start

Please read:

- `README.md`
- `LICENSE`
- `CODE_OF_CONDUCT.md`

For significant architectural changes, open an issue or discussion before doing substantial implementation work.

## What You Can Contribute

Useful contributions include:

- Bug fixes
- Performance improvements
- Training improvements
- Generation improvements
- Evaluation tooling
- Benchmarking
- Documentation
- Tests
- Dataset tooling
- Reproducibility improvements
- Experiment tracking
- Research ideas
- Developer tooling

## Pull Requests

A good pull request should:

- Have a clear purpose.
- Explain what changed and why.
- Keep unrelated changes out of the PR.
- Include tests or validation when appropriate.
- Update documentation when behavior changes.
- Preserve applicable attribution and licensing information.

## Development

Typical setup:

```bash
git clone https://github.com/jadhavdurvesh/Saudade.git
cd Saudade

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Run relevant tests or validation before opening a PR.

Do not commit API keys, passwords, access tokens, private data, local environments, or unnecessary temporary artifacts.

## Model and Training Changes

When changing model architecture or training behavior, document the configuration, dataset/tokenizer version, training steps or tokens seen, important hyperparameters, hardware when relevant, evaluation results, and known limitations.

Do not present an experimental result as a final benchmark without sufficient context.

## Datasets and Third-Party Material

Only contribute datasets, files, or other material that you have the right to distribute.

Third-party licenses remain applicable to their respective materials.

## Commits

Use clear commit messages, for example:

```text
Add generation benchmark
Fix tokenizer loading
Improve checkpoint metadata
Document v4.5 experiments
```

## License and Contributions

By submitting a contribution, you agree to the contribution terms in the project's `LICENSE` file.

Contributors must have the right to submit the work they contribute.

## Questions

For questions that are not appropriate for an issue:

jadhavdurvesh65@gmail.com
