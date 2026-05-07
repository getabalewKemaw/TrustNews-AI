"""Prepare and clean the dataset for training.

Run this after explore_dataset.py to create news_clean.csv
"""

from pathlib import Path

import pandas as pd
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trustnews_ml.preprocessing import clean_text, build_content


def main():
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'news_clean.csv'

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
    print(df['label'].value_counts())

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
