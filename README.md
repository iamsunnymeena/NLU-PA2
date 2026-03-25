# B23CM1008 Assignment 2

This repository contains complete code and outputs for:
1. Problem 1: Learning Word Embeddings from IIT Jodhpur data
2. Problem 2: Character-level Name Generation using recurrent models

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything

```bash
bash scripts/run_all.sh
```

## Run individually

```bash
.venv/bin/python scripts/problem1_pipeline.py
.venv/bin/python scripts/problem2_pipeline.py
```

## Key outputs

- `corpus.txt`
- `TrainingNames.txt`
- `outputs/reports/p1_dataset_stats.json`
- `outputs/reports/p1_form_answers.json`
- `outputs/reports/p1_scratch_experiments.csv`
- `outputs/reports/p1_gensim_experiments.csv`
- `outputs/reports/p2_metrics.json`
- `outputs/reports/p2_form_answers.json`
- `outputs/plots/p1_wordcloud.png`
- `outputs/plots/p1_pca_scratch_sgns.png`
- `outputs/plots/p1_tsne_scratch_sgns.png`

## Submission packaging

Create folder `B23CM1008-A2` with only:
- `report.pdf`
- `corpus.txt`

Then zip it as `B23CM1008-A2.zip`.
