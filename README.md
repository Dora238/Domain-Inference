# Domain-Inference

This project aims to discover knowledge domain features corresponding to each label in language classification models. It works by obtaining embeddings containing label-specific knowledge, guiding language generation models to produce texts corresponding to the target label, and using large language models for concept extraction.

## 📁 Project Structure

```bash
domain-inference/
├── scripts/                      # Script entry points
│   ├── optimize_embedding.py     # Embedding optimization 
│   ├── prefix_tuning_T5.py       # T5 model prefix tuning 
│   └── run.sh                    # Main execution script
├── src/                          # Source code
│   └── domain_infer/             # Main application logic
│       ├── __init__.py
│       ├── classifier.py           # Blackbox classifier evaluation
│       ├── generater.py          # Text generator
│       ├── optimizer.py          # Soft prompt optimizer
│       ├── wordnet_init.py       # WordNet conditioner
├── data/                         # Data directory
│   └── init_from_wordnet
│       ├── init_from_bert 
└── models/                       # Pretrained or fine-tuned models
│   └── bert
├── README.md                     # Project overview
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Locked dependencies (used by uv)
```

## Installation
We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
uv sync
uv pip install -e .
```

# Usage