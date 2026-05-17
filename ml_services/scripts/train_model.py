"""Train a baseline fake news detection model using TF-IDF and Logistic Regression.

This script handles class imbalance using class weights and evaluates the model
on a held-out test set.
"""

from pathlib import Path
import sys
import joblib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_data():
    """Load train and test datasets."""
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    train_path = processed_dir / 'train.csv'
    test_path = processed_dir / 'test.csv'
    
    if not train_path.exists() or not test_path.exists():
        print('Error: Train/test files not found!')
        print(f'Expected: {train_path} and {test_path}')
        print('\nRun prepare_dataset.py first to create these files.')
        return None, None, None, None
    print('Loading data...')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df['content']
    y_train = train_df['label']
    X_test = test_df['content']
    y_test = test_df['label']
    
    print(f'Train set: {len(X_train)} samples')
    print(f'Test set: {len(X_test)} samples')
    
    print(f'\nTrain label distribution:')
    print(y_train.value_counts())
    print(f'\nTest label distribution:')
    print(y_test.value_counts())
    
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """Train a Logistic Regression model with TF-IDF features and class weights."""
    print('\n' + '=' * 60)
    print('TRAINING MODEL')
    print('=' * 60)
    
    # Create pipeline: TF-IDF -> Logistic Regression
    # TF-IDF parameters:
    # - max_features=10000: Limit vocabulary to top 10,000 words
    # - ngram_range=(1, 2): Use unigrams and bigrams
    # - stop_words='english': Remove common English words
    # - min_df=5: Ignore words appearing in fewer than 5 documents
    # - max_df=0.7: Ignore words appearing in more than 70% of documents
    
    # Logistic Regression parameters:
    # - class_weight='balanced': Automatically adjust weights inversely proportional to class frequencies
    # - max_iter=1000: Maximum iterations for convergence
    # - random_state=42: Reproducibility
    # - C=1.0: Regularization strength (smaller = stronger regularization)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=5,
            max_df=0.7
        )),
        ('classifier', LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            random_state=42,
            C=1.0,
            n_jobs=-1  # Use all CPU cores
        ))
    ])
    
    print('Training Logistic Regression with class_weight="balanced"...')
    print('This automatically adjusts weights to handle class imbalance.')
    
    pipeline.fit(X_train, y_train)
    
    print('✓ Training complete!')
    
    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""
    print('\n' + '=' * 60)
    print('MODEL EVALUATION')
    print('=' * 60)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
    
    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print confusion matrix interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f'\nTrue Negatives (FAKE correctly identified): {tn}')
    print(f'False Positives (FAKE misclassified as REAL): {fp}')
    print(f'False Negatives (REAL misclassified as FAKE): {fn}')
    print(f'True Positives (REAL correctly identified): {tp}')
    
    return accuracy, cm


def save_model(model, artifacts_dir):
    """Save the trained model and vectorizer."""
    print('\n' + '=' * 60)
    print('SAVING MODEL')
    print('=' * 60)
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the entire pipeline (includes vectorizer and classifier)
    model_path = artifacts_dir / 'fake_news_pipeline.joblib'
    joblib.dump(model, model_path)
    print(f'✓ Model pipeline saved to: {model_path}')
    
    return model_path


def main():
    """Main execution function."""
    print('\n' + '=' * 60)
    print('TRUSTNEWSAI MODEL TRAINING')
    print('=' * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None:
        return
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, cm = evaluate_model(model, X_test, y_test)
    
    # Save model
    artifacts_dir = Path(__file__).parent.parent / 'artifacts'
    model_path = save_model(model, artifacts_dir)
    
    # Summary
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Model saved to: {model_path}')
    print(f'\nNext step: Use the model for inference on new articles')


if __name__ == '__main__':
    main()
