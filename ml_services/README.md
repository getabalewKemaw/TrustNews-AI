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
python -m ipykernel install --user --name trustnews-ml --display-name "Python (TrustNewsAI ML)"
```

When opening notebooks, select the `Python (TrustNewsAI ML)` kernel.

## Dataset

Use the Kaggle Fake and Real News Dataset. Download it from Kaggle and place the
raw files here:

```text
ml_services/data/raw/True.csv
ml_services/data/raw/Fake.csv
```

The cleaning script also supports a single CSV if it has a label column and text
columns.

## Clean The Data In Jupyter

Start Jupyter from inside `ml_services/`:

```powershell
jupyter notebook
```

Open and run:

```text
notebooks/01_dataset_exploration.ipynb
notebooks/02_prepare_dataset.ipynb
```

Output:

```text
ml_services/data/processed/news_clean.csv
```

## Notebook Workflow

Use notebooks for learning, exploration, charts, cleaning, and first model
training.

Recommended order:

1. `notebooks/01_dataset_exploration.ipynb`
2. `notebooks/02_prepare_dataset.ipynb`
3. future training notebook

Later, when we build the API service, FastAPI will still need normal `.py`
files. For now, the ML learning and dataset work stays in notebooks.

## Current Data Contract

The processed CSV must contain:

- `title`
- `text`
- `content`
- `label`

Allowed labels:

- `REAL`
- `FAKE`
