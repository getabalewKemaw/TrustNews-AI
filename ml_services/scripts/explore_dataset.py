"""
TrustNewsAI Dataset Exploration
===============================

Goal: understand the Kaggle Fake and Real News Dataset before training any model.

Expected raw files:
    - ../data/raw/True.csv
    - ../data/raw/Fake.csv

Run this script from the ml_services directory:
    python notebooks/01_dataset_exploration.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define paths to raw data directories
RAW_DIR = Path(__file__).parent.parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

# File paths for true and fake news
TRUE_PATH = RAW_DIR / 'True.csv'
FAKE_PATH = RAW_DIR / 'Fake.csv'


def check_files_exist():
    """Check if the required CSV files exist."""
    print("=" * 60)
    print("FILE CHECK")
    print("=" * 60)
    
    true_exists = TRUE_PATH.exists()
    fake_exists = FAKE_PATH.exists()
    
    print(f"True.csv exists: {true_exists}")
    print(f"Fake.csv exists: {fake_exists}")
    # .venv\Scripts\python.exe notebooks/01_dataset_exploration.py
    if not true_exists or not fake_exists:
        print("\nERROR: Missing dataset files!")
        print(f"Expected: {TRUE_PATH}")
        print(f"Expected: {FAKE_PATH}")
        print("\nDownload from:")
        print("https://www.kaggle.com/datasets/clmentbischoff/fake-and-real-news-dataset")
        return False
    
    return True


def load_and_label_data():
    """
    Load the raw CSV files and add labels.
    
    Returns:
        pd.DataFrame: Combined dataframe with REAL and FAKE labels
    """
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load the real news dataset
    print(f"Loading {TRUE_PATH}...")
    real_df = pd.read_csv(TRUE_PATH)
    
    # Load the fake news dataset
    print(f"Loading {FAKE_PATH}...")
    fake_df = pd.read_csv(FAKE_PATH)
    
    # Add labels to distinguish real from fake
    real_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'
    
    # Combine both datasets into one
    raw_df = pd.concat([real_df, fake_df], ignore_index=True)
    
    print(f"\nTotal records loaded: {len(raw_df)}")
    print(f"Columns: {list(raw_df.columns)}")
    
    return raw_df


def display_basic_info(df):
    """Display basic information about the dataset."""
    print("\n" + "=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    
    # Show column types and non-null counts
    print("\n--- DataFrame Info ---")
    df.info()
    
    # Show distribution of labels (REAL vs FAKE)
    print("\n--- Label Distribution ---")
    print(df['label'].value_counts())
    print(f"\nPercentages:")
    print(df['label'].value_counts(normalize=True) * 100)


def check_data_quality(df):
    """Check for missing values and duplicates."""
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    
    # Check for missing values in each column
    print("\n--- Missing Values ---")
    missing = df.isna().sum()
    print(missing)
    print(f"\nTotal missing: {missing.sum()}")
    
    # Check for duplicate articles (same title and text)
    print("\n--- Duplicates ---")
    duplicates = df.duplicated(subset=['title', 'text']).sum()
    print(f"Duplicate articles (title + text): {duplicates}")


def analyze_content_length(df):
    """Analyze the length of article content."""
    print("\n" + "=" * 60)
    print("CONTENT LENGTH ANALYSIS")
    print("=" * 60)
    
    # Combine title and text to get full content
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Calculate content length for each article
    df['content_length'] = df['content'].str.len()
    
    # Show statistics grouped by label
    print("\n--- Content Length by Label ---")
    stats = df.groupby('label')['content_length'].describe()
    print(stats)
    
    # Show very short articles (potential quality issues)
    short_articles = df[df['content_length'] < 100]
    print(f"\n--- Very Short Articles (< 100 chars): {len(short_articles)} ---")
    if len(short_articles) > 0:
        print(short_articles[['title', 'content_length', 'label']].head())


def plot_label_distribution(df):
    """Create a simple bar chart of label distribution."""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='label', palette=['green', 'red'])
    plt.title('Label Distribution: Real vs Fake News')
    plt.xlabel('Label')
    plt.ylabel('Count')
    
    # Save the plot
    output_path = Path(__file__).parent / 'label_distribution.png'
    plt.savefig(output_path)
    print(f"Chart saved to: {output_path}")
    plt.close()


def show_samples(df):
    """Display sample records from each category."""
    print("\n" + "=" * 60)
    print("SAMPLE RECORDS")
    print("=" * 60)
    
    print("\n--- Real News Samples ---")
    real_samples = df[df['label'] == 'REAL'][['title', 'label']].head(3)
    for idx, row in real_samples.iterrows():
        print(f"\n{idx + 1}. {row['title'][:80]}...")
    
    print("\n--- Fake News Samples ---")
    fake_samples = df[df['label'] == 'FAKE'][['title', 'label']].head(3)
    for idx, row in fake_samples.iterrows():
        print(f"\n{idx + 1}. {row['title'][:80]}...")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("TRUSTNEWSAI DATASET EXPLORATION")
    print("=" * 60)
    
    # Step 1: Check if files exist
    if not check_files_exist():
        return
    
    # Step 2: Load and combine the data
    df = load_and_label_data()
    
    # Step 3: Display basic information
    display_basic_info(df)
    
    # Step 4: Check data quality
    check_data_quality(df)
    
    # Step 5: Analyze content lengths
    analyze_content_length(df)
    
    # Step 6: Create visualization
    try:
        plot_label_distribution(df)
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    # Step 7: Show sample records
    show_samples(df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal records: {len(df)}")
    print(f"Next step: Run prepare_dataset.py to clean the data")


if __name__ == '__main__':
    main()
