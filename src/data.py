"""
File: data.py

Responsibilities:
- Load raw datasets (Kaggle Fake News, LIAR) from data/raw.
- Clean and preprocess text (e.g., remove boilerplate, handle missing values).
- Create train/validation/test splits and save processed versions.
- Provide helper functions to return dataset objects or DataLoaders for training and evaluation.

Contributors:
- Anton Nemchinski
- <Name 2>
- <Name 3>

Key functions to implement:
- load_raw_datasets(cfg) -> dict
- preprocess_examples(examples, cfg) -> dict
- build_splits(cfg) -> (train_dataset, val_dataset, test_dataset)
- get_hf_datasets(cfg) -> DatasetDict
"""

import pandas as pd
from datasets import Dataset, DatasetDict

import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict

def load_fake_news_dataset(cfg):
    """
    TEMPORARY simple loader for a single fake-news dataset so we can wire up
    the pipeline early.

    Right now this expects the Kaggle "Fake and Real News Dataset":
      https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

    and assumes you have:
        data/raw/Fake.csv
        data/raw/True.csv

    Each CSV is treated as one class:
      - Fake.csv  -> label = 0
      - True.csv  -> label = 1

    We build a single 'text' column by combining title + text when available,
    and keep only ['text', 'label'] for the model.

    Prasa: you are totally free to change this later:
      - choose different columns,
      - add more cleaning,
      - or even swap in a different dataset entirely.
    The only contract the rest of the code relies on is that this function
    returns a DatasetDict with "train", "validation" and "test" splits, and
    each item has at least:
        - 'text'  (input string)
        - 'label' (integer class id)
    """

    data_dir = Path("data/raw")

    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = pd.read_csv(data_dir / "True.csv")

    # Assign labels: 0 = fake, 1 = real/true
    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = 0
    true_df["label"] = 1

    # Build a single 'text' column.
    # If both 'title' and 'text' exist, concatenate them; otherwise fall back.
    for df in (fake_df, true_df):
        if "text" in df.columns and "title" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "text" in df.columns:
            df["text"] = df["text"].fillna("")
        elif "title" in df.columns:
            df["text"] = df["title"].fillna("")
        else:
            raise ValueError("Expected at least a 'text' or 'title' column in the CSV files.")

    # Keep only the fields we actually need
    fake_df = fake_df[["text", "label"]]
    true_df = true_df[["text", "label"]]

    # Combine and shuffle
    full_df = pd.concat([fake_df, true_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)

    # Build HuggingFace Dataset and create splits
    ds = Dataset.from_pandas(full_df, preserve_index=False)

    # Optional: this encodes label as a ClassLabel, but 0/1 already works
    ds = ds.class_encode_column("label")

    ds_split = ds.train_test_split(test_size=cfg["data"]["test_size"], seed=cfg["seed"])
    train_val = ds_split["train"].train_test_split(
        test_size=cfg["data"]["val_size"], seed=cfg["seed"]
    )

    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": ds_split["test"],
        }
    )

