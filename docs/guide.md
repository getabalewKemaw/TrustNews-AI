TrustNewsAI project guide

TrustNewsAI is an AI-powered news credibility analyzer. The first version uses a
separate frontend, backend API, and ML service from the start.

## Product flow

1. User pastes a news title and article text in the web app.
2. Frontend sends the text to the backend API.
3. Backend forwards the analysis request to the ML service.
4. ML service cleans the text, runs the model, and returns prediction data.
5. Backend returns a user-friendly result to the frontend.
6. Frontend shows prediction, confidence, and explanation signals.

## Architecture

```text
frontend/       Next.js user interface
backend/        FastAPI application API
ml_services/    Data preparation, model training, and ML inference service
docs/           Project planning and implementation notes
```

Target runtime architecture:

```text
Browser
  -> Next.js frontend
  -> FastAPI backend
  -> ML service
  -> trained model artifacts
```

## 20-day roadmap

### Phase 1: Data foundation

Day 1:
- Confirm folder structure.
- Add ML service requirements.
- Add dataset loading and cleaning Python scripts.

Day 2:
- Download the Kaggle Fake and Real News Dataset.
- Place raw CSV files in `ml_services/data/raw/`.
- Run the exploration script (includes visualization).
- Run the preparation script to create clean dataset.

Day 3:
- Inspect processed data.
- Check label balance, duplicates, null values, and content length.

Day 4:
- Freeze the first processed dataset format.
- Document dataset assumptions and known weaknesses.

### Phase 2: Baseline model

Day 5:
- Build TF-IDF feature extraction.

Day 6:
- Train Logistic Regression baseline.

Day 7:
- Run predictions on the test split.

Day 8:
- Evaluate with accuracy, precision, recall, and F1 score.

Day 9:
- Save model and vectorizer artifacts.

Day 10:
- Test manual examples and document failures.

### Phase 3: System integration

Day 11:
- Create ML service prediction endpoint.

Day 12:
- Create backend API route that calls the ML service.

Day 13:
- Add confidence score and response schema.

Day 14:
- Add simple explanation signals.

Day 15:
- Build Next.js analysis form.

Day 16:
- Connect frontend to backend.

### Phase 4: Portfolio polish

Day 17:
- Improve loading, errors, and empty states.

Day 18:
- Add clean result UI and prediction history if time allows.

Day 19:
- Prepare deployment configuration.

Day 20:
- Final README, screenshots, demo script, and GitHub cleanup.

## Dataset plan

Initial dataset:

- Kaggle Fake and Real News Dataset
- Expected raw files: `True.csv` and `Fake.csv`
- Labels normalized to `REAL` and `FAKE`

Processed dataset output:

```text
ml_services/data/processed/news_clean.csv
```

Python script workflow:

```text
ml_services/scripts/explore_dataset.py
ml_services/scripts/prepare_dataset.py
```

Required processed columns:

- `title`
- `text`
- `content`
- `label`

## Model plan

Baseline:

- TF-IDF vectorizer
- Logistic Regression classifier

Reason:

- Easy to understand
- Fast to train
- Good baseline before using BERT or DistilBERT

Later upgrade:

- DistilBERT for better context understanding
- Explanation signals using feature weights, keywords, or LIME/SHAP
