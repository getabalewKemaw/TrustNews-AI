"""Prepare and clean the dataset for training.

Run this after explore_dataset.py to create news_clean.csv
Includes train/test split with stratification.
"""

from pathlib import Path

import pandas as pd
import sys
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trustnews_ml.preprocessing import clean_text, build_content


def main():
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'

    true_path = raw_dir / 'True.csv'
    fake_path = raw_dir / 'Fake.csv'

    if not true_path.exists() or not fake_path.exists():
        print('Error: Dataset files not found!')
        print(f'Expected: {true_path} and {fake_path}')
        return

    print('Loading raw data...')
    real_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    real_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'

    raw_df = pd.concat([real_df, fake_df], ignore_index=True)
    print(f'Loaded {len(raw_df)} records')

    print('Cleaning data...')
    df = pd.DataFrame()
    df['title'] = raw_df['title'].map(clean_text)
    df['text'] = raw_df['text'].map(clean_text)
    df['content'] = [build_content(t, txt) for t, txt in zip(df['title'], df['text'])]
    df['label'] = raw_df['label']

    before_rows = len(df)

    # Filter bad rows
    df = df.dropna(subset=['content', 'label'])
    df = df[df['label'].isin(['REAL', 'FAKE'])]
    df = df[df['content'].str.len() >= 80]
    df = df.drop_duplicates(subset=['content'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    after_rows = len(df)

    print(f'Removed {before_rows - after_rows} bad/duplicate rows')
    print(f'Final dataset: {after_rows} records')
    print(f'\nLabel distribution:')
    label_counts = df['label'].value_counts()
    print(label_counts)
    print(f'\nLabel percentages:')
    print(label_counts / len(df) * 100)

    # Check for class imbalance
    print('\n' + '=' * 60)
    print('CLASS IMBALANCE CHECK')
    print('=' * 60)
    imbalance_ratio = label_counts['FAKE'] / label_counts['REAL']
    print(f'FAKE/REAL ratio: {imbalance_ratio:.2f}')
    if imbalance_ratio < 0.9 or imbalance_ratio > 1.1:
        print(' WARNING: Classes are imbalanced!')
        print('   Consider using class weights or resampling.')
    else:
        print('✓ Classes are reasonably balanced')

    # Train/Test Split with stratification
    print('\n' + '=' * 60)
    print('TRAIN/TEST SPLIT')
    print('=' * 60)
    
    # Feature/Target separation
    X = df['content']  # Features: text content
    y = df['label']    # Target: REAL/FAKE labels
    
    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')
    
    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain label balance in both sets
    )
    
    print(f'\nTrain set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)')
    print(f'Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)')
    
    print(f'\nTrain label distribution:')
    print(y_train.value_counts())
    print(f'\nTest label distribution:')
    print(y_test.value_counts())

    # Save splits
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train set
    train_df = pd.DataFrame({'content': X_train, 'label': y_train})
    train_path = processed_dir / 'train.csv'
    train_df.to_csv(train_path, index=False)
    print(f'\nSaved train set to: {train_path}')
    
    # Save test set
    test_df = pd.DataFrame({'content': X_test, 'label': y_test})
    test_path = processed_dir / 'test.csv'
    test_df.to_csv(test_path, index=False)
    print(f'Saved test set to: {test_path}')
    
    # Save full cleaned dataset (for reference)
    full_path = processed_dir / 'news_clean.csv'
    df.to_csv(full_path, index=False)
    print(f'Saved full cleaned dataset to: {full_path}')
    
    print('\n' + '=' * 60)
    print('PREPARATION COMPLETE')
    print('=' * 60)
    print(f'\nNext step: Run feature extraction (TF-IDF) and model training')


if __name__ == '__main__':
    main()
