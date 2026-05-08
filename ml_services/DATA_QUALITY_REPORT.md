# TrustNewsAI Data Quality Report

## Business Context

**Problem:** Fake news detection for news articles
**Goal:** Classify articles as REAL or FAKE based on title and text content
**Domain:** News media, journalism, misinformation detection

**Key Business Insights:**
- Fake news often has sensationalist headlines
- Real news typically follows journalistic standards
- Content length patterns may differ between real/fake
- Text patterns (word choice, sentence structure) are key indicators

---

## Dataset Overview

**Source:** Kaggle Fake and Real News Dataset
**Total Records:** 44,898
**Columns:** title, text, subject, date, label
**Target Variable:** label (REAL/FAKE)

---

## Data Quality Assessment

### Missing Values
- Check for null values in key columns
- Identify columns with high missing rates

### Duplicates
- Duplicate articles (same title + text)
- Need to remove to prevent data leakage

### Content Quality
- Very short articles (< 80 chars) - low quality
- Empty titles or text bodies
- HTML entities and special characters

### Label Balance
- Check if REAL vs FAKE is balanced
- Imbalance may require oversampling/undersampling

---

## Feature Analysis

### Input Features
- `title`: Article headline (text)
- `text`: Article body (text)
- `content`: Combined title + text (text)

### Target Feature
- `label`: REAL or FAKE (categorical)

### Useless Features
- `subject`: News category (not predictive of authenticity)
- `date`: Publication date (not relevant for classification)

---

## Data Preparation Strategy

### 1. Text Cleaning
- Remove URLs and email addresses
- Remove HTML entities
- Normalize whitespace
- Handle special characters

### 2. Feature Engineering
- Combine title and text into content
- Calculate content length
- Create word count features

### 3. Train/Test Split
- 80% train, 20% test
- Stratified split to maintain label balance
- Random seed for reproducibility

### 4. Handling Imbalance
- Check label distribution
- If imbalanced, consider:
  - Class weighting in model
  - Oversampling minority class
  - Undersampling majority class

### 5. Outlier Detection
- Content length outliers
- Very long or very short articles
- Investigate and handle appropriately

---

## Assumptions and Limitations

### Assumptions
- Dataset is representative of real news distribution
- Text patterns are consistent across time periods
- Title + text content is sufficient for classification

### Limitations
- No author information
- No source/website information
- May not generalize to news from different sources
- Temporal shifts in fake news patterns not captured

---

## Next Steps

1. Run EDA to validate assumptions
2. Clean and preprocess data
3. Split into train/test sets
4. Train baseline model (TF-IDF + Logistic Regression)
5. Evaluate and iterate
