# Contributing

Thank you for helping improve Multimodal Flow.

## Local checks

Create an isolated environment and install the quality tools:

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src
python -m ruff check src
python -m build
```

Behavioral validation belongs in the project's private engineering workspace;
the public repository intentionally contains only the implementation and
documentation. Keep datasets, model weights, generated checkpoints,
credentials, and machine-specific output directories outside the repository.

## Pull requests

Describe the user-visible or research behavior being changed, the checks you
ran, and any hardware or asset requirements. Do not include private data,
access tokens, or undistributable model files.
