# Classification of Sequence / Categorical Feature Pairs

Training simple models (Random Forest, Fully Connected, BERT + Fully Connected) on paired sequence and categorical features. The data is private medical data and not included in this repository.

## Models

- **Random Forest** — scikit-learn baseline
- **SimpleClassifier** — fully connected PyTorch network with one-hot encoded sequences
- **TinyBERTClassifier** — Intel's TinyBERT followed by a fully connected head

## Project Structure

- `modules/` — core library: datasets, models, preprocessing, analysis utilities
- `experiments_notebook/` — Jupyter notebooks for running experiments
- `tests/` — unit tests (42 tests)

## Getting Started

### Install

```bash
uv sync
```

### Run experiments

Open and run the notebooks in `experiments_notebook/`.

### Tests

```bash
uv run pytest
```

## Authors

Mathieu Charbonnel

## Acknowledgments

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.

Hugging Face Transformers Library: https://github.com/huggingface/transformers
