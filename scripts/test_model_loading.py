"""
Test script to verify trained models can be loaded by Person 3-5.

This script tests that:
1. Saved models can be loaded from disk
2. Tokenizer works correctly
3. Model can make predictions on sample text

Usage:
    python scripts/test_model_loading.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path


def test_model_loading(model_path, sample_texts):
    """
    Test loading a saved DistilBERT model and running inference.
    
    Args:
        model_path (str): Path to saved model directory
        sample_texts (list): List of sample texts to test
    """
    print(f"\n{'='*70}")
    print(f"Testing Model: {model_path}")
    print('='*70)
    
    # Check if model exists
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"Model not found at {model_path}")
        print("   Train the model first using: python src/train.py")
        return False
    
    try:
        # Load tokenizer and model
        print("\n Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        
        print(" Model loaded successfully!")
        print(f"   Model type: {model.__class__.__name__}")
        print(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test on sample texts
        print(f"\n Testing on {len(sample_texts)} sample texts...")
        print("="*70)
        
        for i, text in enumerate(sample_texts, 1):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = probabilities.argmax().item()
                confidence = probabilities.max().item()
            
            # Display result
            label = "FAKE" if predicted_class == 0 else "REAL"
            print(f"\nSample {i}:")
            print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Probabilities: Fake={probabilities[0][0]:.2%}, Real={probabilities[0][1]:.2%}")
        
        print("\n" + "="*70)
        print(" ALL TESTS PASSED!")
        print("   Person 3-5 can use this model for explainability.")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("   Model loading failed. Check the model files.")
        return False


def main():
    """Run model loading tests."""
    print("\n" + "="*70)
    print(" "*15 + "MODEL LOADING TEST SCRIPT")
    print(" "*12 + "Person 2 (Zaid) - CP322 Project")
    print("="*70)
    
    # Sample texts for testing
    sample_texts = [
        "Breaking news: Scientists discover cure for all diseases using artificial intelligence!",
        "The Federal Reserve announced a modest interest rate adjustment today following economic indicators.",
        "You won't believe what this celebrity did! Doctors hate this one weird trick!",
        "Climate change continues to pose significant challenges according to latest research findings.",
    ]
    
    # Test Kaggle model
    kaggle_model_path = "artifacts/distilbert/kaggle/final_model"
    kaggle_success = test_model_loading(kaggle_model_path, sample_texts[:2])
    
    # Test LIAR model
    liar_model_path = "artifacts/distilbert/liar/final_model"
    liar_success = test_model_loading(liar_model_path, sample_texts[2:])
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Kaggle Model: {' PASS' if kaggle_success else ' FAIL'}")
    print(f"LIAR Model:   {' PASS' if liar_success else ' FAIL'}")
    
    if kaggle_success and liar_success:
        print("\n Both models are ready for Person 3-5!")
    elif kaggle_success or liar_success:
        print("\n Some models need to be trained.")
    else:
        print("\n No models found. Run training first:")
        print("   python src/train.py --dataset both")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
