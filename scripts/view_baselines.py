"""
View baseline results from Person 1 to know what scores to beat.

Usage:
    python scripts/view_baselines.py
"""

import pandas as pd
from pathlib import Path


def view_baselines():
    """Display baseline results from Person 1."""
    print("\n" + "="*70)
    print(" "*15 + "BASELINE RESULTS (Person 1)")
    print(" "*12 + "Scores to Beat with DistilBERT")
    print("="*70 + "\n")
    
    # Check if baseline results exist
    baseline_file = Path("artifacts/metrics/baseline_results.csv")
    
    if not baseline_file.exists():
        print(" Baseline results not found!")
        print(f"   Expected file: {baseline_file}")
        print("\n   Person 1 needs to run their baseline models first.")
        print("   Or the file might be in a different location.")
        return
    
    # Load and display results
    try:
        df = pd.read_csv(baseline_file)
        
        # Group by dataset
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            
            print(f"{'─'*70}")
            print(f"{dataset.upper()} DATASET")
            print(f"{'─'*70}")
            
            # Display each model's results
            for _, row in dataset_df.iterrows():
                print(f"\nModel: {row['model']}")
                if 'accuracy' in df.columns:
                    print(f"  Accuracy:  {row['accuracy']:.4f}")
                if 'f1' in df.columns:
                    print(f"  F1 Score:  {row['f1']:.4f}")
                if 'precision' in df.columns:
                    print(f"  Precision: {row['precision']:.4f}")
                if 'recall' in df.columns:
                    print(f"  Recall:    {row['recall']:.4f}")
            
            # Find best F1 score
            best_f1 = dataset_df['f1'].max() if 'f1' in df.columns else dataset_df['accuracy'].max()
            best_model = dataset_df.loc[dataset_df['f1'].idxmax(), 'model'] if 'f1' in df.columns else dataset_df.loc[dataset_df['accuracy'].idxmax(), 'model']
            
            print(f"\n{'─'*70}")
            print(f" TARGET TO BEAT:")
            print(f"   Best Model: {best_model}")
            print(f"   Best F1: {best_f1:.4f}")
            print(f"   Your DistilBERT should achieve F1 > {best_f1:.4f}")
            print(f"{'─'*70}\n")
        
        print("\n" + "="*70)
        print("NEXT STEP:")
        print("  Train DistilBERT: python src/train.py --dataset both")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f" Error reading baseline file: {e}")
        print("   File might be corrupted or in wrong format.")


if __name__ == "__main__":
    view_baselines()
