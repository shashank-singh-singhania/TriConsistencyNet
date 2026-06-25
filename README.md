# TriConsistencyNet

Confidence-Aware Multi-Representation Learning for Generalizable and Explainable Deepfake Detection.

## Setup

This project targets Python 3.12.
Use Python 3.12 for dependency installation; `facenet-pytorch` does not install cleanly in the current Python 3.13 environment.

Install dependencies with either of these commands from the repository root:

```bash
uv sync
```

or

```bash
python -m pip install -e .
```

For development extras:

```bash
python -m pip install -e ".[dev]"
```

## Status

🚧 Under Active Research

## Features

- Spatial Representation Learning
- Frequency Representation Learning
- Semantic Transformer Branch
- Representation Consistency Module
- Confidence-Aware Fusion
- Explainability Guided Learning

## Repository Structure

See docs/PROJECT_STRUCTURE.md

## License

MIT