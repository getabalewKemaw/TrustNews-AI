"""Test the prediction script with sample articles."""

from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from predict import load_model, predict_article


def main():
    """Test prediction with sample articles."""
    print('\n' + '=' * 60)
    print('TESTING PREDICTION SCRIPT')
    print('=' * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Test with a fake news sample
    print('\n' + '=' * 60)
    print('TEST 1: FAKE NEWS SAMPLE')
    print('=' * 60)
    
    fake_title = "BREAKING: Scientists Discover That Coffee Causes Cancer"
    fake_text = "A new study from an unknown research facility claims that drinking coffee will give you cancer within 24 hours. The study was conducted on exactly 3 people and has not been peer-reviewed. Mainstream media is hiding this information from you!"
    
    prediction, confidence = predict_article(model, fake_title, fake_text)
    print(f'\nTitle: {fake_title}')
    print(f'Text: {fake_text[:100]}...')
    print(f'\nPrediction: {prediction}')
    print(f'Confidence: {confidence * 100:.2f}%')
    
    # Test with a real news sample
    print('\n' + '=' * 60)
    print('TEST 2: REAL NEWS SAMPLE')
    print('=' * 60)
    
    real_title = "Senate Passes Infrastructure Bill After Months of Debate"
    real_text = "The Senate voted 69-30 on Tuesday to pass a $1.2 trillion infrastructure bill that will fund roads, bridges, and broadband internet across the United States. The bill now moves to the House of Representatives for final approval."
    
    prediction, confidence = predict_article(model, real_title, real_text)
    print(f'\nTitle: {real_title}')
    print(f'Text: {real_text[:100]}...')
    print(f'\nPrediction: {prediction}')
    print(f'Confidence: {confidence * 100:.2f}%')
    
    print('\n' + '=' * 60)
    print('TEST COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
