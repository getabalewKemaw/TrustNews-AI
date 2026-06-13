"""Use the trained fake news detection model to predict on new articles.

This script loads the trained model pipeline and makes predictions on new articles.
"""

from pathlib import Path
import sys
import joblib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trustnews_ml.preprocessing import clean_text, build_content

def load_model():
    """Load the trained model pipeline."""
    artifacts_dir = Path(__file__).parent.parent / 'artifacts'
    model_path = artifacts_dir / 'fake_news_pipeline.joblib'
    
    if not model_path.exists():
        print('Error: Model not found!')
        print(f'Expected: {model_path}')
        print('\nRun train_model.py first to train and save the model.')
        return None
    
    print('Loading trained model...')
    model = joblib.load(model_path)
    print('✓ Model loaded successfully')
    
    return model


def predict_article(model, title: str, text: str):
    """Predict whether an article is FAKE or REAL.
    
    Args:
        model: Trained model pipeline
        title: Article title
        text: Article body text
        
    Returns:
        tuple: (prediction, confidence)
    """
    # Clean and combine text
    content = build_content(title, text)
    
    # Make prediction
    prediction = model.predict([content])[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba([content])[0]
    
    # Get confidence for the predicted class
    if prediction == 'FAKE':
        confidence = probabilities[0]
    else:
        confidence = probabilities[1]
    
    return prediction, confidence


def main():
    """Main execution function."""
    print('\n' + '=' * 60)
    print('TRUSTNEWSAI FAKE NEWS DETECTION')
    print('=' * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    print('\n' + '=' * 60)
    print('ENTER ARTICLE DETAILS')
    print('=' * 60)
    
    # Get article title
    print('\nEnter article title (or press Enter to exit):')
    title = input('> ').strip()
    
    if not title:
        print('No title provided. Exiting.')
        return
    
    # Get article text
    print('\nEnter article text (or press Enter to exit):')
    text = input('> ').strip()
    
    if not text:
        print('No text provided. Exiting.')
        return
    
    # Make prediction
    print('\n' + '=' * 60)
    print('ANALYZING ARTICLE...')
    print('=' * 60)
    
    prediction, confidence = predict_article(model, title, text)
    
    # Display result
    print('\n' + '=' * 60)
    print('PREDICTION RESULT')
    print('=' * 60)
    
    if prediction == 'REAL':
        print(f'\n✓ Prediction: REAL NEWS')
        print(f'  Confidence: {confidence * 100:.2f}%')
    else:
        print(f'\n⚠ Prediction: FAKE NEWS')
        print(f'  Confidence: {confidence * 100:.2f}%')
    
    print('\n' + '=' * 60)
    print('ANALYSIS COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()


