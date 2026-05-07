# TrustNewsAI Progress Notes

## What We Decided

We decided to build TrustNewsAI as a real-world AI project with separate parts:

```text
frontend/      Next.js user interface
backend/       FastAPI backend API
ml_services/   Jupyter-based ML workflow
docs/          project planning and guide files
```

The user prefers the ML work to be done in Jupyter notebooks instead of normal
Python `.py` files for now.

## What Was Changed

### Project guide

Updated:

```text
docs/guide.md
```

It now explains:

- the product flow
- the separated architecture
- the 20-day roadmap
- dataset plan
- baseline model plan
- notebook workflow

### ML service folder

Created a proper ML workspace:

```text
ml_services/
  scripts/         Python scripts
    explore_dataset.py
    prepare_dataset.py
  src/             Reusable Python modules
    trustnews_ml/
      __init__.py
      preprocessing.py
  artifacts/
  data/
    raw/
    processed/
  README.md
  requirements.txt
```

### Python Scripts

Created Python scripts for the ML workflow:

**Runnable scripts (in `scripts/`):**
```text
ml_services/scripts/explore_dataset.py
ml_services/scripts/prepare_dataset.py
```

**Reusable modules (in `src/`):**
```text
ml_services/src/trustnews_ml/preprocessing.py
ml_services/src/trustnews_ml/__init__.py
```

Purpose:
- `explore_dataset.py` - Understand the Kaggle dataset (includes visualization)
- `prepare_dataset.py` - Clean and save `news_clean.csv`
- `preprocessing.py` - Reusable text cleaning functions

The expected raw files are:

```text
ml_services/data/raw/True.csv
ml_services/data/raw/Fake.csv
```

The processed output will be:

```text
ml_services/data/processed/news_clean.csv
```

### Requirements

Updated:

```text
ml_services/requirements.txt
```

It includes packages for:

- pandas
- scikit-learn
- joblib
- jupyter
- ipykernel
- matplotlib
- seaborn

### Git ignore rules

Updated:

```text
.gitignore
```

It now ignores:

- raw dataset files
- processed dataset files
- model artifact files
- Jupyter checkpoint files

This keeps large or generated ML files out of Git.

## Workflow Change

Switched from Jupyter notebooks to Python scripts for easier execution.

Old notebook workflow → New Python script workflow:
- `notebooks/01_dataset_exploration.ipynb` → `scripts/explore_dataset.py`
- `notebooks/02_prepare_dataset.ipynb` → `scripts/prepare_dataset.py`

## Next Step

1. Download the Kaggle Fake and Real News Dataset and place:

```text
True.csv
Fake.csv
```

inside:

```text
ml_services/data/raw/
```

2. Setup and run the Python scripts:

```powershell
cd ml_services
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/explore_dataset.py
python scripts/prepare_dataset.py
```

