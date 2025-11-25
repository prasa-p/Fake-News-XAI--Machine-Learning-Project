"""
File: data.py

Responsibilities:
- Load raw datasets (Kaggle Fake News, LIAR) from data/raw.
- Clean and preprocess text (e.g., remove boilerplate, handle missing values).
- Create train/validation/test splits and save processed versions.
- Provide helper functions to return dataset objects or DataLoaders for training and evaluation.

Contributors:
- Anton Nemchinski
- Prasa (Person 1 - Data Preparation & Baselines)

Key functions to implement:
- load_raw_datasets(cfg) -> dict
- preprocess_examples(examples, cfg) -> dict
- build_splits(cfg) -> (train_dataset, val_dataset, test_dataset)
- get_hf_datasets(cfg) -> DatasetDict
"""

import pandas as pd
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


# ============================================================================
# LOAD FROM PROCESSED CSVs (For Training - Person 2)
# ============================================================================

def load_processed_dataset(dataset_name='kaggle'):
    """
    Load already-processed train/val/test splits from CSV files.
    This is faster than reprocessing from raw data.
    
    Args:
        dataset_name (str): 'kaggle' or 'liar'
        
    Returns:
        DatasetDict: HuggingFace DatasetDict with train/validation/test splits
    """
    processed_dir = Path("data/processed")
    
    # Load the three splits
    train_df = pd.read_csv(processed_dir / f"{dataset_name}_train.csv")
    val_df = pd.read_csv(processed_dir / f"{dataset_name}_val.csv")
    test_df = pd.read_csv(processed_dir / f"{dataset_name}_test.csv")
    
    print(f"Loaded {dataset_name} dataset from processed CSVs:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df[['text', 'label']], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[['text', 'label']], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[['text', 'label']], preserve_index=False)
    
    # Encode labels as ClassLabel
    train_ds = train_ds.class_encode_column("label")
    val_ds = val_ds.class_encode_column("label")
    test_ds = test_ds.class_encode_column("label")
    
    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })


# ============================================================================
# TEXT CLEANING UTILITIES (Person 1 - Prasa)
# ============================================================================

def clean_text(text):
    """
    Clean text by removing URLs, special characters, and extra whitespace.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text in lowercase
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep periods and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text


def preprocess_text_column(df, text_col='text', min_words=10):
    """
    Apply text cleaning to a DataFrame column and filter short/empty texts.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_col (str): Name of the text column to clean
        min_words (int): Minimum number of words required
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text
    """
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    
    # Remove empty texts
    df = df[df[text_col].str.len() > 0]
    
    # Remove very short texts (< min_words)
    df = df[df[text_col].str.split().str.len() >= min_words]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=text_col)
    
    return df.reset_index(drop=True)


# ============================================================================
# KAGGLE FAKE NEWS DATASET LOADER (Anton + Prasa enhancements)
# ============================================================================

def load_fake_news_dataset(cfg):
    """
    Load the Kaggle "Fake and Real News Dataset" for HuggingFace training.
    
    This expects:
        data/raw/kaggle/Fake.csv
        data/raw/kaggle/True.csv
    
    Each CSV is treated as one class:
      - Fake.csv  -> label = 0
      - True.csv  -> label = 1

    Returns a DatasetDict with "train", "validation" and "test" splits.
    Each item has:
        - 'text'  (input string)
        - 'label' (integer class id: 0=fake, 1=real)
    
    Args:
        cfg (dict): Configuration with seed and data split parameters
        
    Returns:
        DatasetDict: HuggingFace DatasetDict with train/validation/test splits
    """
    data_dir = Path("data/raw/kaggle")

    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = pd.read_csv(data_dir / "True.csv")

    # Assign labels: 0 = fake, 1 = real/true
    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = 0
    true_df["label"] = 1

    # Build a single 'text' column by combining title + text
    for df in (fake_df, true_df):
        if "text" in df.columns and "title" in df.columns:
            df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        elif "text" in df.columns:
            df["text"] = df["text"].fillna("")
        elif "title" in df.columns:
            df["text"] = df["title"].fillna("")
        else:
            raise ValueError("Expected at least a 'text' or 'title' column in the CSV files.")

    # Keep only the fields we need
    fake_df = fake_df[["text", "label"]]
    true_df = true_df[["text", "label"]]

    # Combine and shuffle
    full_df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Enhanced: Apply text cleaning (Person 1 addition)
    print(f"Before cleaning: {len(full_df)} samples")
    full_df = preprocess_text_column(full_df, text_col='text', min_words=10)
    print(f"After cleaning: {len(full_df)} samples")
    
    full_df = full_df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)

    # Build HuggingFace Dataset and create splits
    ds = Dataset.from_pandas(full_df, preserve_index=False)
    ds = ds.class_encode_column("label")

    ds_split = ds.train_test_split(test_size=cfg["data"]["test_size"], seed=cfg["seed"])
    train_val = ds_split["train"].train_test_split(
        test_size=cfg["data"]["val_size"] / (1 - cfg["data"]["test_size"]), 
        seed=cfg["seed"]
    )

    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": ds_split["test"],
        }
    )


# ============================================================================
# LIAR DATASET LOADER (Person 1 - Prasa)
# ============================================================================

def load_liar_dataset(cfg):
    """
    Load the LIAR dataset for HuggingFace training.
    
    This expects:
        data/raw/liar/train.tsv
        data/raw/liar/valid.tsv
        data/raw/liar/test.tsv
    
    Original LIAR has 6 truthfulness labels. We convert to binary:
        - true, mostly-true, half-true -> label = 1 (real)
        - barely-true, false, pants-fire -> label = 0 (fake)
    
    Args:
        cfg (dict): Configuration with seed and data split parameters
        
    Returns:
        DatasetDict: HuggingFace DatasetDict with train/validation/test splits
    """
    data_dir = Path("data/raw/liar")
    
    print("=" * 60)
    print("Loading LIAR Dataset...")
    print("=" * 60)
    
    # Column names for LIAR TSV files
    cols = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job', 
        'state', 'party', 'barely_true_ct', 'false_ct', 'half_true_ct',
        'mostly_true_ct', 'pants_on_fire_ct', 'context'
    ]
    
    # Load all splits
    train_df = pd.read_csv(data_dir / "train.tsv", sep='\t', names=cols, header=None)
    val_df = pd.read_csv(data_dir / "valid.tsv", sep='\t', names=cols, header=None)
    test_df = pd.read_csv(data_dir / "test.tsv", sep='\t', names=cols, header=None)
    
    print(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test statements")
    
    # Combine all splits
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Convert to binary labels
    true_labels = ['true', 'mostly-true', 'half-true']
    full_df['label'] = full_df['label'].apply(
        lambda x: 1 if x in true_labels else 0
    )
    
    # Use statement as text
    full_df['text'] = full_df['statement'].fillna("")
    full_df = full_df[['text', 'label']]
    
    # Clean text
    print(f"Before cleaning: {len(full_df)} samples")
    full_df = preprocess_text_column(full_df, text_col='text', min_words=5)  # LIAR has shorter texts
    print(f"After cleaning: {len(full_df)} samples")
    print(f"Label distribution:\n{full_df['label'].value_counts()}")
    print("=" * 60)
    
    # Shuffle
    full_df = full_df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    
    # Convert to HuggingFace Dataset
    ds = Dataset.from_pandas(full_df, preserve_index=False)
    ds = ds.class_encode_column("label")
    
    # Create splits
    ds_split = ds.train_test_split(test_size=cfg["data"]["test_size"], seed=cfg["seed"])
    train_val = ds_split["train"].train_test_split(
        test_size=cfg["data"]["val_size"] / (1 - cfg["data"]["test_size"]),
        seed=cfg["seed"]
    )
    
    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": ds_split["test"],
        }
    )


# ============================================================================
# PANDAS-BASED LOADERS FOR BASELINE MODELS (Person 1 - Prasa)
# ============================================================================

def load_kaggle_pandas(raw_data_dir="data/raw"):
    """
    Load Kaggle dataset as pandas DataFrame for baseline model training.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['text', 'label']
    """
    data_dir = Path(raw_data_dir) / "kaggle"
    
    print("Loading Kaggle dataset for baselines...")
    
    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = pd.read_csv(data_dir / "True.csv")
    
    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = 0
    true_df["label"] = 1
    
    # Combine title + text
    for df in (fake_df, true_df):
        if "text" in df.columns and "title" in df.columns:
            df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        elif "text" in df.columns:
            df["text"] = df["text"].fillna("")
        elif "title" in df.columns:
            df["text"] = df["title"].fillna("")
    
    fake_df = fake_df[["text", "label"]]
    true_df = true_df[["text", "label"]]
    
    full_df = pd.concat([fake_df, true_df], ignore_index=True)
    full_df = preprocess_text_column(full_df, text_col='text')
    
    print(f"Loaded {len(full_df)} Kaggle samples")
    return full_df


def load_liar_pandas(raw_data_dir="data/raw"):
    """
    Load LIAR dataset as pandas DataFrame for baseline model training.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['text', 'label']
    """
    data_dir = Path(raw_data_dir) / "liar"
    
    print("Loading LIAR dataset for baselines...")
    
    cols = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job', 
        'state', 'party', 'barely_true_ct', 'false_ct', 'half_true_ct',
        'mostly_true_ct', 'pants_on_fire_ct', 'context'
    ]
    
    train_df = pd.read_csv(data_dir / "train.tsv", sep='\t', names=cols, header=None)
    val_df = pd.read_csv(data_dir / "valid.tsv", sep='\t', names=cols, header=None)
    test_df = pd.read_csv(data_dir / "test.tsv", sep='\t', names=cols, header=None)
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    true_labels = ['true', 'mostly-true', 'half-true']
    full_df['label'] = full_df['label'].apply(lambda x: 1 if x in true_labels else 0)
    
    full_df['text'] = full_df['statement'].fillna("")
    full_df = full_df[['text', 'label']]
    full_df = preprocess_text_column(full_df, text_col='text', min_words=5)
    
    print(f"Loaded {len(full_df)} LIAR samples")
    return full_df


# ============================================================================
# TRAIN/VAL/TEST SPLITTING (Person 1 - Prasa)
# ============================================================================

def create_splits(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create stratified train/validation/test splits.
    
    Default: 80% train, 10% val, 10% test
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'label' column
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, 
        test_size=val_ratio, 
        random_state=random_state, 
        stratify=train_val['label']
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train):5d} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val):5d} ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test):5d} ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test


def save_splits(train, val, test, dataset_name, output_dir="data/processed"):
    """
    Save train/val/test splits to CSV files.
    
    Args:
        train, val, test (pd.DataFrame): Split DataFrames
        dataset_name (str): Name prefix (e.g., 'kaggle', 'liar')
        output_dir (str): Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(output_path / f"{dataset_name}_train.csv", index=False)
    val.to_csv(output_path / f"{dataset_name}_val.csv", index=False)
    test.to_csv(output_path / f"{dataset_name}_test.csv", index=False)
    
    print(f"✓ Saved {dataset_name} splits to {output_dir}/\n")


# ============================================================================
# MAIN EXECUTION (Person 1 - Prasa)
# ============================================================================

if __name__ == "__main__":
    """
    Run this script to generate processed train/val/test CSVs for baselines.
    
    Usage:
        python src/data.py
    """
    print("\n" + "="*70)
    print(" "*15 + "DATA PREPROCESSING PIPELINE")
    print(" "*10 + "Person 1 (Prasa) - CP322 Group Project")
    print("="*70 + "\n")
    
    # Process Kaggle dataset
    print("### PROCESSING KAGGLE DATASET ###\n")
    kaggle_df = load_kaggle_pandas()
    kaggle_train, kaggle_val, kaggle_test = create_splits(kaggle_df, random_state=42)
    save_splits(kaggle_train, kaggle_val, kaggle_test, "kaggle")
    
    # Process LIAR dataset
    print("### PROCESSING LIAR DATASET ###\n")
    liar_df = load_liar_pandas()
    liar_train, liar_val, liar_test = create_splits(liar_df, random_state=42)
    save_splits(liar_train, liar_val, liar_test, "liar")
    
    print("="*70)
    print(" "*20 + "✓ ALL DATASETS PROCESSED")
    print("="*70 + "\n")