"""
File: train.py

Responsibilities:
- Entry point for training models (mainly DistilBERT) using configuration files.
- Set random seeds and prepare datasets/tokenizer/model.
- Configure Hugging Face Trainer (or custom loop), run training, and save best checkpoints and training logs.

Contributors:
- Anton Nemchinski
- Zaid (Person 2 - DistilBERT Training & Tuning)

Key functions to implement:
- main() -> None
- setup_training(cfg) -> (model, tokenizer, train_dataset, val_dataset, training_args)
- run_training(model, tokenizer, train_dataset, val_dataset, training_args, cfg) -> None
"""

import argparse
import json
import numpy as np
from pathlib import Path
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.utils import load_config, set_seed, get_device, get_run_id, ensure_dir
from src.models import get_tokenizer, get_distilbert_model
from src.data import load_processed_dataset


def compute_metrics(eval_pred):
    """
    Calculate classification metrics during training.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        dict: Dictionary of metric names and values
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='binary'),
        'precision': precision_score(labels, predictions, average='binary', zero_division=0),
        'recall': recall_score(labels, predictions, average='binary', zero_division=0)
    }


def train_distilbert(config_file='config/kaggle.yaml'):
    """
    Train DistilBERT on specified dataset configuration.
    
    Args:
        config_file (str): Path to YAML config file
        
    Returns:
        dict: Test set results
    """
    # Load configuration
    cfg = load_config(config_file)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    
    print("\n" + "="*70)
    print(f"{'DISTILBERT TRAINING PIPELINE':^70}")
    print(f"{'Person 2 (Zaid) - CP322 Project':^70}")
    print("="*70)
    print(f"\nConfig: {config_file}")
    print(f"Dataset: {cfg['data']['dataset']}")
    print(f"Model: {cfg['model']['name']}")
    print(f"Device: {device}")
    print(f"Max Length: {cfg['model']['max_length']} tokens")
    print(f"Batch Size: {cfg['train']['batch_train']}")
    print(f"Learning Rate: {cfg['train']['learning_rate']}")
    print(f"Epochs: {cfg['train']['num_epochs']}")
    print("="*70)
    
    # Load dataset
    print("\n Loading dataset...")
    if cfg['data']['dataset'] == 'kaggle_fake_news':
        ds = load_processed_dataset('kaggle')
    elif cfg['data']['dataset'] == 'liar':
        ds = load_processed_dataset('liar')
    else:
        raise ValueError(f"Unknown dataset: {cfg['data']['dataset']}")
    
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  Val:   {len(ds['validation'])} samples")
    print(f"  Test:  {len(ds['test'])} samples")
    
    # Load tokenizer and model
    print("\n Loading DistilBERT...")
    tokenizer = get_tokenizer(cfg)
    model = get_distilbert_model(cfg)
    
    # Tokenize datasets
    print("\n Tokenizing text...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=cfg['model']['max_length']
        )
    
    tokenized_ds = ds.map(tokenize_function, batched=True, desc="Tokenizing")
    
    # Remove text column, keep only model inputs
    tokenized_ds = tokenized_ds.remove_columns(['text'])
    tokenized_ds.set_format('torch')
    
    # Training arguments
    print("\n Setting up training...")
    output_dir = Path(cfg['train']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg['train']['num_epochs'],
        per_device_train_batch_size=cfg['train']['batch_train'],
        per_device_eval_batch_size=cfg['train']['batch_eval'],
        learning_rate=cfg['train']['learning_rate'],
        weight_decay=cfg['train']['weight_decay'],
        warmup_steps=cfg['train']['warmup_steps'],
        logging_dir=str(output_dir / "logs"),
        logging_steps=cfg['train']['logging_steps'],
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=cfg['train']['eval_steps'],
        save_strategy="steps",
        save_steps=cfg['train']['save_steps'],
        save_total_limit=cfg['train']['save_total_limit'],
        load_best_model_at_end=cfg['train']['load_best_model_at_end'],
        metric_for_best_model=cfg['train']['metric_for_best_model'],
        greater_is_better=True,
        use_cpu=(device.type == "cpu"),  # Changed from fp16
        report_to="none",  # Changed from list to string
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        compute_metrics=compute_metrics,
    )
    
    # Train!
    print("\n" + "="*70)
    print(" STARTING TRAINING...")
    print("="*70)
    
    train_result = trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print(" EVALUATING ON TEST SET...")
    print("="*70)
    
    test_results = trainer.evaluate(tokenized_ds['test'])
    
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall:    {test_results['eval_recall']:.4f}")
    print(f"Loss:      {test_results['eval_loss']:.4f}")
    print("="*70)
    
    # Save final model
    print("\n💾 Saving final model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    # Save metrics
    metrics_path = Path("artifacts/metrics")
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'dataset': cfg['data']['dataset'],
        'model': cfg['model']['name'],
        'test_accuracy': float(test_results['eval_accuracy']),
        'test_f1': float(test_results['eval_f1']),
        'test_precision': float(test_results['eval_precision']),
        'test_recall': float(test_results['eval_recall']),
        'test_loss': float(test_results['eval_loss']),
        'hyperparameters': {
            'learning_rate': cfg['train']['learning_rate'],
            'batch_size': cfg['train']['batch_train'],
            'epochs': cfg['train']['num_epochs'],
            'max_length': cfg['model']['max_length'],
            'weight_decay': cfg['train']['weight_decay'],
            'warmup_steps': cfg['train']['warmup_steps'],
        }
    }
    
    results_file = metrics_path / f"distilbert_{cfg['data']['dataset']}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Model saved to: {final_model_path}")
    print(f"✅ Metrics saved to: {results_file}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")
    
    return test_results


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train DistilBERT for fake news detection')
    parser.add_argument(
        '--config',
        type=str,
        default='config/kaggle.yaml',
        help='Path to config YAML file (default: config/kaggle.yaml)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['kaggle', 'liar', 'both'],
        default=None,
        help='Which dataset to train on (overrides config file)'
    )
    
    args = parser.parse_args()
    
    # Determine which config(s) to use
    if args.dataset == 'both':
        configs = ['config/kaggle.yaml', 'config/liar.yaml']
    elif args.dataset == 'kaggle':
        configs = ['config/kaggle.yaml']
    elif args.dataset == 'liar':
        configs = ['config/liar.yaml']
    else:
        configs = [args.config]
    
    # Train on each dataset
    results = {}
    for config_file in configs:
        test_results = train_distilbert(config_file)
        dataset_name = 'kaggle' if 'kaggle' in config_file else 'liar'
        results[dataset_name] = test_results
    
    # Print summary if training on both
    if len(configs) > 1:
        print("\n" + "="*70)
        print("TRAINING SUMMARY - ALL DATASETS")
        print("="*70)
        for dataset_name, test_results in results.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
            print(f"  F1 Score: {test_results['eval_f1']:.4f}")
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

