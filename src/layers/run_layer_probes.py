# src/layers/run_layer_probes.py
"""
CLI script to run layer-wise probes for DistilBERT fake-news classifier.

Usage (from repo root, with venv active):

  PYTHONPATH=. python -m src.layers.run_layer_probes --dataset kaggle
  PYTHONPATH=. python -m src.layers.run_layer_probes --dataset liar

This will:
- Load the fine-tuned DistilBERT model from artifacts/distilbert/<dataset>/final_model
- Load the processed test split from data/processed/<dataset>_test.csv
- Extract layer representations and train a logistic probe per layer
- Save metrics JSON to artifacts/layers/<dataset>_layer_probes.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import load_config, get_device, ensure_dir
from src.layers.probe import (
    build_probe_config,
    extract_layer_representations,
    train_layer_probes,
)


def load_test_split(dataset_name: str) -> Dict[str, np.ndarray]:
    """
    Load processed test split for a given dataset.

    Expects:
      data/processed/kaggle_test.csv
      data/processed/liar_test.csv

    with columns:
      - text
      - label (0 = fake, 1 = real)
    """
    test_path = os.path.join("data", "processed", f"{dataset_name}_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find test split at: {test_path}")

    df = pd.read_csv(test_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{test_path} must contain 'text' and 'label' columns. "
            f"Found columns: {list(df.columns)}"
        )

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return {"texts": np.array(texts), "labels": np.array(labels)}


def load_finetuned_model_and_tokenizer(
    dataset_name: str,
    device: torch.device,
):
    """
    Load the fine-tuned DistilBERT model + tokenizer for a given dataset.

    Expects:
      artifacts/distilbert/<dataset>/final_model/
        - config.json
        - pytorch_model.bin / safetensors
        - tokenizer files
    """
    ckpt_dir = os.path.join("artifacts", "distilbert", dataset_name, "final_model")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kaggle", "liar"],
        required=True,
        help="Which dataset to run probes on",
    )
    args = parser.parse_args()

    # Load config + device
    cfg = load_config(args.config)
    device = get_device(cfg)
    data_cfg = cfg.get("data", {})
    max_length = data_cfg.get("max_length", 256)

    print(f"[layers] Device: {device}")
    print(f"[layers] Dataset: {args.dataset}")

    # Build probe config
    probe_cfg = build_probe_config(cfg)
    print(
        f"[layers] ProbeConfig: max_samples={probe_cfg.max_samples}, "
        f"batch_size={probe_cfg.batch_size}, test_size={probe_cfg.test_size}"
    )

    # Load model + tokenizer
    model, tokenizer = load_finetuned_model_and_tokenizer(args.dataset, device)

    # Load test split
    split = load_test_split(args.dataset)
    texts, labels = split["texts"], split["labels"]
    print(f"[layers] Loaded {len(texts)} test examples")

    # Extract layer representations
    X, y = extract_layer_representations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        labels=labels,
        device=device,
        probe_cfg=probe_cfg,
        max_length=max_length,
    )

    # Train probes
    metrics = train_layer_probes(X, y, probe_cfg)

    # Print a small summary
    print("\n[layer probes] Results per layer:")
    for layer, acc, f1 in zip(
        metrics["layers"], metrics["accuracy"], metrics["f1"]
    ):
        print(f"  Layer {layer:2d}: acc={acc:.3f}, f1={f1:.3f}")

    # Save to JSON
    out_dir = os.path.join("artifacts", "layers")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{args.dataset}_layer_probes.json")

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[layer probes] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
