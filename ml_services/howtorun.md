# TrustNewsAI ML Service - How to Run

This guide shows you how to run the complete ML workflow for fake news detection.

---

## Prerequisites

- Python 3.12+ installed
- Kaggle Fake and Real News Dataset downloaded

---

## Step 1: Setup Virtual Environment

```powershell
cd ml_services
python -m venv .venv
.venv\Scripts\activate
```

**To deactivate later:**
```powershell
deactivate
```

---

## Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- pandas, scikit-learn, joblib (ML libraries)
- matplotlib, seaborn (visualization)
- gradio (web interface)

---

## Step 3: Download Dataset

1. Go to: https://www.kaggle.com/datasets/clmentbischoff/fake-and-real-news-dataset
2. Download `True.csv` and `Fake.csv`
3. Place them in: `ml_services/data/raw/`

---

## Step 4: Explore the Dataset

```powershell
.venv\Scripts\python.exe scripts/explore_dataset.py
```

**What it does:**
- Loads raw data
- Shows statistics (records, columns, label distribution)
- Checks for missing values and duplicates
- Detects outliers
- Creates visualization chart (`label_distribution.png`)

**Output:** Console statistics + chart in `scripts/` folder

---

## Step 5: Prepare Clean Dataset

```powershell
.venv\Scripts\python.exe scripts/prepare_dataset.py
```

**What it does:**
- Cleans text (removes URLs, emails, normalizes whitespace)
- Removes bad rows (too short, duplicates)
- Splits into train/test sets (80/20)
- Checks for class imbalance
- Saves to `data/processed/`:
  - `train.csv` (31,152 samples)
  - `test.csv` (7,789 samples)
  - `news_clean.csv` (full cleaned dataset)

**Output:** 3 CSV files in `data/processed/`

---

## Step 6: Train the Model

```powershell
.venv\Scripts\python.exe scripts/train_model.py
```

**What it does:**
- Loads train/test data
- Uses TF-IDF for feature extraction
- Trains Logistic Regression with class weights
- Evaluates on test set
- Saves model to `artifacts/fake_news_pipeline.joblib`

**Output:**
- Model saved to `artifacts/`
- Test accuracy (should be ~98.84%)
- Classification report (precision, recall, F1)

---

## Step 7: Use the Model

### Option A: Command-line Interface

```powershell
.venv\Scripts\python.exe scripts/predict.py
```

**How to use:**
1. Enter article title
2. Enter article text
3. See prediction (FAKE/REAL) with confidence

**To exit:** Press Enter without entering text

---

### Option B: Web Interface (Gradio)

```powershell
.venv\Scripts\python.exe scripts/gradio_app.py
```

**How to use:**
1. Wait for "Running on local URL" message
2. Open browser to: http://127.0.0.1:7860
3. Enter article title and text
4. Click "Analyze Article"
5. See prediction with confidence

**To stop:**
- Press `Ctrl+C` in the terminal
- Or close the terminal window

---

## How to Stop Running Processes

### Stop Gradio Web Interface

**Method 1:** Press `Ctrl+C` in the terminal where it's running

**Method 2:** Close the terminal window

### Stop Any Python Script

Press `Ctrl+C` in the terminal

---

## File Locations

```
ml_services/
├── data/
│   ├── raw/              # Place True.csv and Fake.csv here
│   └── processed/        # train.csv, test.csv, news_clean.csv created here
├── artifacts/            # fake_news_pipeline.joblib created here
├── scripts/
│   ├── explore_dataset.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── predict.py
│   └── gradio_app.py
└── src/
    └── trustnews_ml/
        ├── __init__.py
        └── preprocessing.py
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Install dependencies
```powershell
pip install -r requirements.txt
```

### Error: "Dataset files not found"

**Solution:** Download and place `True.csv` and `Fake.csv` in `data/raw/`

### Error: "Model not found"

**Solution:** Run `train_model.py` first to train and save the model

### Error: "Port 7860 is already in use"

**Solution:** Either:
1. Stop the existing Gradio app (Ctrl+C)
2. Or change the port in `gradio_app.py` line 118

---

## Quick Reference

| Task | Command |
|------|---------|
| Setup venv | `python -m venv .venv` |
| Install deps | `pip install -r requirements.txt` |
| Explore data | `python scripts/explore_dataset.py` |
| Prepare data | `python scripts/prepare_dataset.py` |
| Train model | `python scripts/train_model.py` |
| Predict (CLI) | `python scripts/predict.py` |
| Predict (Web) | `python scripts/gradio_app.py` |
| Stop process | `Ctrl+C` |

---

## Model Performance

- **Accuracy:** 98.84%
- **FAKE detection:** 99% precision, 98% recall
- **REAL detection:** 99% precision, 99% recall
- **Features:** TF-IDF (10,000 features, unigrams + bigrams)
- **Class weights:** Balanced (handles 0.84 FAKE/REAL ratio)

---

## Next Steps

After training and testing, you can:
1. Use the web interface for predictions
2. Integrate with backend API (if needed)
3. Deploy to production
4. Improve model with more data or different algorithms
