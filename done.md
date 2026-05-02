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
  artifacts/
  data/
    raw/
    processed/
  notebooks/
  README.md
  requirements.txt
```

### Jupyter notebooks

Created:

```text
ml_services/notebooks/01_dataset_exploration.ipynb
ml_services/notebooks/02_prepare_dataset.ipynb
```

Purpose:

- `01_dataset_exploration.ipynb` is for understanding the Kaggle dataset.
- `02_prepare_dataset.ipynb` is for cleaning the raw dataset and saving a clean CSV.

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

## Important Correction

The first version had ML helper code in `.py` files:

```text
ml_services/scripts/prepare_dataset.py
ml_services/src/trustnews_ml/preprocessing.py
```

Because the user requested notebook-based ML work, those files were removed and
the logic was moved into:

```text
ml_services/notebooks/02_prepare_dataset.ipynb
```

FastAPI backend files will still use `.py` later because APIs need Python
modules to run. But dataset exploration, cleaning, and first model training will
use notebooks for now.

## Next Step

Next, download the Kaggle Fake and Real News Dataset and place:

```text
True.csv
Fake.csv
```

inside:

```text
ml_services/data/raw/
```

Then open Jupyter from `ml_services/`:

```powershell
cd ml_services
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name trustnews-ml --display-name "Python (TrustNewsAI ML)"
jupyter notebook
```

Run the notebooks in this order:

```text
01_dataset_exploration.ipynb
02_prepare_dataset.ipynb
```

