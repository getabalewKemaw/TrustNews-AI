"""
TrustNewsAI Dataset Exploration
===============================

Goal: understand the Kaggle Fake and Real News Dataset before training any model.

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
    
    if not true_exists or not fake_exists:
        print("\nERROR: Missing dataset files!")
        print(f"Expected: {TRUE_PATH}")
        print(f"Expected: {FAKE_PATH}")
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
    real_df = pd.read_csv(TRUE_PATH)
    fake_df = pd.read_csv(FAKE_PATH)
    
    # Add labels to distinguish real from fake
    # becuase the model learn from lables
    real_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'
    

    # Combine both datasets into one
    # why not train separetlly ? b/c classification models expect x,y=all inputs,and all outputs respectivelly plus to  shuffle the data
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
    # show the percentage using the normalize =True methods
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
    # The model may memorize instead of learning real patterns if the data we give hs duplicates
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
    stats = df.groupby('label')['content_length'].describe()# these gives the summary statics like the metrics like count mean std min
    print(stats)
    
    # Show very short articles (potential quality issues)
    short_articles = df[df['content_length'] < 100]
    print(f"\n--- Very Short Articles (< 100 chars): {len(short_articles)} ---")
    if len(short_articles) > 0:
        print(short_articles[['title', 'content_length', 'label']].head())



"""Are there articles whose length is unusually too small or too large compared to the rest?"""
def detect_outliers(df):
    """Detect outliers in content length using IQR method."""
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    
    # Calculate content length if not already done
    if 'content_length' not in df.columns:
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['content_length'] = df['content'].str.len()
    
    # Calculate IQR for content length
    Q1 = df['content_length'].quantile(0.25)
    Q3 = df['content_length'].quantile(0.75)
    IQR = Q3 - Q1 # interquartile range measures the normal spread of data
    
    # Define outlier bounds (<0 and >150)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nContent Length Statistics:")
    print(f"  Q1 (25th percentile): {Q1:.0f}")
    print(f"  Q3 (75th percentile): {Q3:.0f}")
    print(f"  IQR: {IQR:.0f}")
    print(f"  Lower bound (outlier): {lower_bound:.0f}")
    print(f"  Upper bound (outlier): {upper_bound:.0f}")
    
    # Identify outliers -select rows that are too short or too long
    outliers = df[(df['content_length'] < lower_bound) | (df['content_length'] > upper_bound)]
    print(f"\n--- Outliers Detected: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%) ---")
    
    if len(outliers) > 0:
        print(f"  Short outliers (< {lower_bound:.0f}): {len(outliers[outliers['content_length'] < lower_bound])}")
        print(f"  Long outliers (> {upper_bound:.0f}): {len(outliers[outliers['content_length'] > upper_bound])}")
        
        # Show sample outliers
        print("\n--- Sample Outliers ---")
        print(outliers[['title', 'content_length', 'label']].head())
        
        # Check if outliers are balanced by label
        print("\n--- Outlier Distribution by Label ---")
        print(outliers['label'].value_counts())
    
    return len(outliers)


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
    
    # Step 6: Detect outliers
    outlier_count = detect_outliers(df)
    
    # Step 7: Create visualization
    try:
        plot_label_distribution(df)
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    # Step 8: Show sample records
    show_samples(df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal records: {len(df)}")
    print(f"Outliers detected: {outlier_count}")
    print(f"Next step: Run prepare_dataset.py to clean and split the data")


if __name__ == '__main__':
    main()
