"""
File: models.py

Responsibilities:
- Define and construct baseline models (e.g., TF-IDF + Logistic Regression, SVM, Random Forest).
- Load and configure DistilBERT-based sequence classification model from Hugging Face.
- Provide utility functions to initialize tokenizers and models in a consistent way.

Contributors:
- Anton Nemchinski (DistilBERT setup)
- Prasa Pirabagaran (Person 1 - Baseline models)

Key functions to implement:
- build_baseline_model(cfg) -> sklearn model
- get_tokenizer(cfg) -> PreTrainedTokenizerFast
- get_distilbert_model(cfg) -> PreTrainedModel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pickle
import json

# ============================================================================
# DISTILBERT MODEL UTILITIES (Anton)
# ============================================================================

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    def get_tokenizer(cfg):
        """Load DistilBERT tokenizer from HuggingFace."""
        return AutoTokenizer.from_pretrained(cfg["model"]["name"])
    
    def get_distilbert_model(cfg):
        """Load DistilBERT model for sequence classification."""
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model"]["name"],
            num_labels=cfg["model"]["num_labels"]
        )
except ImportError:
    print("⚠️ transformers not installed - DistilBERT functions unavailable")
    def get_tokenizer(cfg):
        raise ImportError("transformers package required for DistilBERT")
    def get_distilbert_model(cfg):
        raise ImportError("transformers package required for DistilBERT")


# ============================================================================
# BASELINE MODEL TRAINING 
# ============================================================================

def train_tfidf_logreg(train_df, val_df, max_features=5000):
    """
    Train TF-IDF + Logistic Regression baseline.
    
    Args:
        train_df (pd.DataFrame): Training data with 'text' and 'label'
        val_df (pd.DataFrame): Validation data
        max_features (int): Max TF-IDF features
        
    Returns:
        tuple: (model, vectorizer, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training TF-IDF + Logistic Regression")
    print("="*60)
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    
    print(f"TF-IDF shape: {X_train.shape}")
    
    # Train
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, train_df['label'])
    
    # Evaluate
    val_preds = model.predict(X_val)
    metrics = compute_metrics(val_df['label'], val_preds)
    
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1 Score: {metrics['f1']:.4f}")
    
    return model, vectorizer, metrics


def train_tfidf_svm(train_df, val_df, max_features=5000):
    """
    Train TF-IDF + Linear SVM baseline.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        max_features (int): Max TF-IDF features
        
    Returns:
        tuple: (model, vectorizer, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training TF-IDF + Linear SVM")
    print("="*60)
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    
    print(f"TF-IDF shape: {X_train.shape}")
    
    # Train
    model = LinearSVC(
        max_iter=2000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, train_df['label'])
    
    # Evaluate
    val_preds = model.predict(X_val)
    metrics = compute_metrics(val_df['label'], val_preds)
    
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1 Score: {metrics['f1']:.4f}")
    
    return model, vectorizer, metrics


def train_tfidf_random_forest(train_df, val_df, max_features=3000):
    """
    Train TF-IDF + Random Forest baseline.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        max_features (int): Max TF-IDF features (lower for RF)
        
    Returns:
        tuple: (model, vectorizer, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training TF-IDF + Random Forest")
    print("="*60)
    
    # Vectorize (fewer features for RF to avoid memory issues)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    
    print(f"TF-IDF shape: {X_train.shape}")
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, train_df['label'])
    
    # Evaluate
    val_preds = model.predict(X_val)
    metrics = compute_metrics(val_df['label'], val_preds)
    
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1 Score: {metrics['f1']:.4f}")
    
    return model, vectorizer, metrics


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary')
    }


def evaluate_model(model, vectorizer, test_df, dataset_name, model_name):
    """
    Evaluate a trained model on test set.
    
    Args:
        model: Trained sklearn model
        vectorizer: Fitted TF-IDF vectorizer
        test_df (pd.DataFrame): Test data
        dataset_name (str): 'kaggle' or 'liar'
        model_name (str): Model name for logging
        
    Returns:
        dict: Test metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {dataset_name} test set")
    print('='*60)
    
    X_test = vectorizer.transform(test_df['text'])
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(test_df['label'], y_pred)
    
    print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test F1 Score:  {metrics['f1']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall:    {metrics['recall']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_df['label'], y_pred, 
                                target_names=['Fake', 'Real']))
    
    return metrics


# ============================================================================
# MODEL SAVING/LOADING
# ============================================================================

def save_model(model, vectorizer, metrics, dataset_name, model_name, 
               output_dir="artifacts/models"):
    """
    Save trained model, vectorizer, and metrics.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        metrics (dict): Performance metrics
        dataset_name (str): 'kaggle' or 'liar'
        model_name (str): Model identifier
        output_dir (str): Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{dataset_name}_{model_name}"
    
    # Save model
    with open(output_path / f"{prefix}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save vectorizer
    with open(output_path / f"{prefix}_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save metrics
    with open(output_path / f"{prefix}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved {prefix} to {output_dir}/")


def load_baseline_model(dataset_name, model_name, models_dir="artifacts/models"):
    """
    Load a saved baseline model.
    
    Args:
        dataset_name (str): 'kaggle' or 'liar'
        model_name (str): 'logreg', 'svm', or 'rf'
        models_dir (str): Directory containing saved models
        
    Returns:
        tuple: (model, vectorizer, metrics)
    """
    models_path = Path(models_dir)
    prefix = f"{dataset_name}_{model_name}"
    
    with open(models_path / f"{prefix}_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open(models_path / f"{prefix}_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(models_path / f"{prefix}_metrics.json", 'r') as f:
        metrics = json.load(f)
    
    return model, vectorizer, metrics


# ============================================================================
# MAIN EXECUTION (Person 1 - Baseline Training)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "BASELINE MODEL TRAINING")
    print(" "*10 + "- CP322 Group Project")
    print("="*70 + "\n")
    
    results = []
    
    # ========== KAGGLE DATASET ==========
    print("\n### TRAINING ON KAGGLE DATASET ###\n")
    
    kaggle_train = pd.read_csv("data/processed/kaggle_train.csv")
    kaggle_val = pd.read_csv("data/processed/kaggle_val.csv")
    kaggle_test = pd.read_csv("data/processed/kaggle_test.csv")
    
    print(f"Kaggle - Train: {len(kaggle_train)}, Val: {len(kaggle_val)}, Test: {len(kaggle_test)}")
    
    # Logistic Regression
    lr_model, lr_vec, lr_val_metrics = train_tfidf_logreg(kaggle_train, kaggle_val)
    lr_test_metrics = evaluate_model(lr_model, lr_vec, kaggle_test, "kaggle", "logreg")
    save_model(lr_model, lr_vec, lr_test_metrics, "kaggle", "logreg")
    results.append({"dataset": "Kaggle", "model": "TF-IDF + LogReg", **lr_test_metrics})
    
    # SVM
    svm_model, svm_vec, svm_val_metrics = train_tfidf_svm(kaggle_train, kaggle_val)
    svm_test_metrics = evaluate_model(svm_model, svm_vec, kaggle_test, "kaggle", "svm")
    save_model(svm_model, svm_vec, svm_test_metrics, "kaggle", "svm")
    results.append({"dataset": "Kaggle", "model": "TF-IDF + SVM", **svm_test_metrics})
    
    # Random Forest
    rf_model, rf_vec, rf_val_metrics = train_tfidf_random_forest(kaggle_train, kaggle_val)
    rf_test_metrics = evaluate_model(rf_model, rf_vec, kaggle_test, "kaggle", "rf")
    save_model(rf_model, rf_vec, rf_test_metrics, "kaggle", "rf")
    results.append({"dataset": "Kaggle", "model": "TF-IDF + RF", **rf_test_metrics})
    
    # ========== LIAR DATASET ==========
    print("\n\n### TRAINING ON LIAR DATASET ###\n")
    
    liar_train = pd.read_csv("data/processed/liar_train.csv")
    liar_val = pd.read_csv("data/processed/liar_val.csv")
    liar_test = pd.read_csv("data/processed/liar_test.csv")
    
    print(f"LIAR - Train: {len(liar_train)}, Val: {len(liar_val)}, Test: {len(liar_test)}")
    
    # Logistic Regression
    lr_model, lr_vec, lr_val_metrics = train_tfidf_logreg(liar_train, liar_val)
    lr_test_metrics = evaluate_model(lr_model, lr_vec, liar_test, "liar", "logreg")
    save_model(lr_model, lr_vec, lr_test_metrics, "liar", "logreg")
    results.append({"dataset": "LIAR", "model": "TF-IDF + LogReg", **lr_test_metrics})
    
    # SVM
    svm_model, svm_vec, svm_val_metrics = train_tfidf_svm(liar_train, liar_val)
    svm_test_metrics = evaluate_model(svm_model, svm_vec, liar_test, "liar", "svm")
    save_model(svm_model, svm_vec, svm_test_metrics, "liar", "svm")
    results.append({"dataset": "LIAR", "model": "TF-IDF + SVM", **svm_test_metrics})
    
    # Random Forest
    rf_model, rf_vec, rf_val_metrics = train_tfidf_random_forest(liar_train, liar_val)
    rf_test_metrics = evaluate_model(rf_model, rf_vec, liar_test, "liar", "rf")
    save_model(rf_model, rf_vec, rf_test_metrics, "liar", "rf")
    results.append({"dataset": "LIAR", "model": "TF-IDF + RF", **rf_test_metrics})
    
    # ========== SUMMARY TABLE ==========
    print("\n" + "="*70)
    print(" "*20 + "✓ BASELINE RESULTS SUMMARY")
    print("="*70 + "\n")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("artifacts/metrics/baseline_results.csv", index=False)
    print(f"\n✓ Results saved to artifacts/metrics/baseline_results.csv")
    
    print("\n" + "="*70)
    print(" "*15 + "BASELINE TRAINING COMPLETE!")
    print("="*70 + "\n")