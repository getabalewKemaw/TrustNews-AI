# TrustNewsAI ML Service

This folder owns the machine learning workflow:

- raw dataset storage
- data cleaning
- model training
- model artifacts
- future ML inference API

The first milestone is to turn Kaggle raw CSV files into one clean dataset.

## Setup

Create and activate a virtual environment from inside `ml_services/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Use the Kaggle Fake and Real News Dataset. Download it from Kaggle and place the
raw files here:

```text
ml_services/data/raw/True.csv
ml_services/data/raw/Fake.csv
```

## Python Script Workflow

Run the scripts in order:

### 1. Explore the dataset

```powershell
python scripts/explore_dataset.py
```

This shows:
- Data shape and columns
- Label distribution
- Missing values
- Duplicate count
- Content statistics
- Creates visualization chart (`label_distribution.png`)

### 2. Prepare the clean dataset

```powershell
python scripts/prepare_dataset.py
```

This creates:
- `ml_services/data/processed/news_clean.csv`

## Cleaning Process
- Removes URLs and emails
- Normalizes whitespace
- Combines title and text into content
- Drops rows with missing/short content
- Removes duplicates
- Shuffles the data

## Project Structure

```text
ml_services/
  scripts/           Runnable Python scripts
    explore_dataset.py
    prepare_dataset.py
  src/               Reusable Python modules
    trustnews_ml/
      __init__.py
      preprocessing.py
  data/
    raw/             Raw dataset files (not in Git)
    processed/       Cleaned dataset (not in Git)
  artifacts/         Trained models (not in Git)
```

## Current Data Contract

The processed CSV must contain:

- `title`
- `text`
- `content`
- `label`

Allowed labels:

- `REAL`
- `FAKE`
