"""Gradio interface for fake news detection.

This provides a simple web UI to use the trained model for predictions.
"""

from pathlib import Path
import sys
import joblib
import gradio as gr

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trustnews_ml.preprocessing import clean_text, build_content


def load_model():
    """Load the trained model pipeline."""
    artifacts_dir = Path(__file__).parent.parent / 'artifacts'
    model_path = artifacts_dir / 'fake_news_pipeline.joblib'
    
    if not model_path.exists():
        return None
    
    return joblib.load(model_path)

def predict(title: str, text: str):
    """Predict whether an article is FAKE or REAL.
    
    Args:
        title: Article title
        text: Article body text
        
    Returns:
        tuple: (label, confidence, message)
    """
    if not title or not text:
        return "ERROR", "0%", "Please provide both title and text."
    
    # Load model
    model = load_model()
    if model is None:
        return "ERROR", "0%", "Model not found. Run train_model.py first."
    
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
    
    # Format result
    if prediction == 'REAL':
        label = " REAL NEWS"
        message = "This article appears to be legitimate news."
    else:
        label = " FAKE NEWS"
        message = "This article appears to be fake or misleading."
    
    return label, f"{confidence * 100:.2f}%", message


def main():
    """Launch the Gradio interface."""
    print('Loading model...')
    model = load_model()
    
    if model is None:
        print('Error: Model not found!')
        print('Run train_model.py first to train and save the model.')
        return
    
    print('✓ Model loaded successfully')
    print('Starting Gradio interface...')
    
    # Create Gradio interface
    with gr.Blocks(title="TrustNewsAI - Fake News Detection") as app:
        gr.Markdown("#  TrustNewsAI - Fake News Detection")
        gr.Markdown("Enter an article title and text to check if it's real or fake news.")
        
        with gr.Row():
            with gr.Column():
                title_input = gr.Textbox(
                    label="Article Title",
                    placeholder="Enter the article headline...",
                    lines=2
                )
                text_input = gr.Textbox(
                    label="Article Text",
                    placeholder="Enter the article body...",
                    lines=10
                )
                predict_btn = gr.Button("Analyze Article", variant="primary")
            
            with gr.Column():
                label_output = gr.Textbox(label="Prediction", interactive=False)
                confidence_output = gr.Textbox(label="Confidence", interactive=False)
                message_output = gr.Textbox(label="Analysis", interactive=False)
        
        # Examples
        gr.Examples(
            examples=[
                [
                    "Senate Passes Infrastructure Bill After Months of Debate",
                    "The Senate voted 69-30 on Tuesday to pass a $1.2 trillion infrastructure bill that will fund roads, bridges, and broadband internet across the United States. The bill now moves to the House of Representatives for final approval."
                ],
                [
                    "BREAKING: Scientists Discover That Coffee Causes Cancer",
                    "A new study from an unknown research facility claims that drinking coffee will give you cancer within 24 hours. The study was conducted on exactly 3 people and has not been peer-reviewed. Mainstream media is hiding this information from you!"
                ]
            ],
            inputs=[title_input, text_input]
        )
        
        # Button click handler
        predict_btn.click(
            fn=predict,
            inputs=[title_input, text_input],
            outputs=[label_output, confidence_output, message_output]
        )
    
    # Launch the app
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == '__main__':
    main()
