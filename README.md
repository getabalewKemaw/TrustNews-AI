# TrustNewsAI — Fake News Detection 

TrustNewsAI is a compact, reproducible pipeline for detecting likely fake news from article title + text. It demonstrates practical skills in data cleaning, NLP feature engineering, model training, evaluation, and deployment (simple web demo). This repository is structured so recruiters and reviewers can quickly run the demo and evaluate results.

## Table of Contents
- Overview
- Highlights for recruiters
- Quick demo (run locally)
- Installation
- Usage
- Project structure
- Data & model artifacts
- How to train and evaluate
- Contributing & contact
- License

## Overview

TrustNewsAI loads publicly available news data, cleans and combines title and body text, extracts features (TF‑IDF baseline), and trains a lightweight classifier to predict whether an article is REAL or FAKE. The project focuses on clarity, reproducibility, and teaching best practices for an ML screening pipeline.

## Problem statement

Online misinformation spreads quickly; manual fact-checking cannot scale. The goal of this project is to build a reproducible, explainable pipeline that flags likely fake news from an article's title and body text so downstream reviewers can prioritize content for human verification.

## Why I solved this problem (motivation)

- Real-world impact: reducing the time needed to triage suspicious articles helps platforms and fact-checkers focus resources.
- Learning goals: demonstrate end-to-end NLP workflow, trade-offs between lightweight baselines and larger models, and deployment-best-practices for demos.
- Recruiter / interview value: shows data cleaning, feature engineering, evaluation, and product-aware model thinking.

## My approach (what I implemented)

- Data cleaning: canonicalize text, remove duplicates and very short articles, combine title + body into a single `content` field.
- Feature engineering: TF‑IDF vectorization of cleaned content as a strong, interpretable baseline.
- Model: Logistic Regression (interpretable, fast, robust baseline) wrapped as a scikit-learn pipeline with the TF‑IDF vectorizer.
- Evaluation: hold-out test set with precision, recall, F1; sample inspection of high‑confidence false positives/negatives for qualitative analysis.
- Deployment: simple Gradio demo and a CLI script to run quick checks.

## Key results (example outputs)

- Baseline metrics (example): precision ~0.85, recall ~0.78, F1 ~0.81 (your run may vary depending on preprocessing and random seed).
- Saved pipeline: `ml_services/artifacts/fake_news_pipeline.joblib` for quick inference.

## What I learned & next steps

- Lessons: careful preprocessing and feature selection materially affect small-data baselines; TF‑IDF + Logistic Regression is a competitive baseline for short text when compute is limited.
- Next improvements: evaluate transformer-based embeddings, address dataset bias, try class-weighted or sampling strategies, and add more robust unit/integration tests.

## Highlights for recruiters
- **Project type:** NLP / binary classification, end-to-end pipeline (data → model → demo)
- **Key skills:** Python, scikit-learn, data cleaning, feature engineering (TF‑IDF), model evaluation, lightweight deployment with Gradio
- **Deliverables:** Cleaned datasets ([ml_services/data/processed](ml_services/data/processed)), trained artifact ([ml_services/artifacts/fake_news_pipeline.joblib](ml_services/artifacts/fake_news_pipeline.joblib)), scripts for training, inference, and a Gradio demo
- **Why hire for this:** Clear project scope, reproducible steps, demonstrable metrics and a runnable demo — ideal to discuss model choices, trade-offs, and evaluation in interviews

## Quick demo 
1. Create and activate a Python virtual environment (from repo root):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r ml_services/requirements.txt
```

2. Run the web demo (Gradio):

```powershell
python ml_services/scripts/gradio_app.py
```

3. Or run a quick CLI prediction:

```powershell
python ml_services/scripts/predict.py --title "Example title" --text "Example article text"
```

If you see a running local Gradio URL (e.g., http://127.0.0.1:7860), open it to interact with the model.

## Installation
- Python 3.8+ recommended
- Install dependencies:

```powershell
pip install -r ml_services/requirements.txt
```

## Usage
- Train a baseline model (saves artifact to `ml_services/artifacts`):

```powershell
python ml_services/scripts/train_model.py
```

- Run unit-style check for inference:

```powershell
python ml_services/scripts/test_predict.py
```

## Project structure (important files)
- `ml_services/scripts/` — scripts for explore, prepare, train, predict, and Gradio demo
- `ml_services/src/trustnews_ml/preprocessing.py` — text cleaning and helper functions
- `ml_services/data/processed/` — cleaned and split datasets (train/test)
- `ml_services/artifacts/fake_news_pipeline.joblib` — example trained pipeline (model + vectorizer)

## Data & model artifacts
- Processed data: `ml_services/data/processed/train.csv` and `test.csv`
- Trained model artifact: `ml_services/artifacts/fake_news_pipeline.joblib`

## How to train and evaluate
1. Ensure processed data exists (run `prepare_dataset.py` if needed):

```powershell
python ml_services/scripts/prepare_dataset.py
```

2. Train and produce evaluation metrics:

```powershell
python ml_services/scripts/train_model.py
```

The training script prints precision/recall/F1 and saves the pipeline to `ml_services/artifacts`.

## For  — concise candidate summary
- **Role demonstrated:** ML Engineer / NLP Engineer
- **Responsibilities :** dataset cleaning, feature engineering, baseline model selection, evaluation and simple deployment


## Contributing & contact
If you'd like to contribute or see enhancements, please open an issue or PR. Replace the contact placeholder below with your GitHub profile or email before pushing.

- Maintainer: Replace with your name and contact (e.g., GitHub: your-username)

## License
This project is provided for demonstration and hiring purposes. Add a license file if you intend to publish publicly (for example, `MIT`).

---



Built by Getabalew Kemaw-ML Engineer

