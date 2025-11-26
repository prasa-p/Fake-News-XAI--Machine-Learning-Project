"""
File: layers/probe.py

Responsibilities:
- Extracts hidden representations from each DistilBERT layer for a subset
  of the test set.
- Trains a simple LogisticRegression probe per layer to predict fake vs real.
- Returns a metrics dict (accuracy, F1 per layer) and optionally the features.

Contributors:
- Anton Nemchinski

Key functions to implement:
- extract_layer_representations(model, tokenizer, dataset, cfg) -> dict[layer_index -> np.array]
- train_probes(layer_reps: dict, labels: np.array, cfg) -> dict[layer_index -> float]
- save_probe_results(results: dict, out_path: str) -> None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class ProbeConfig:
    max_samples: int = 2000     # how many test samples to use at most
    batch_size: int = 32        # batch size for forward passes
    test_size: float = 0.2      # fraction of data for probe evaluation
    random_state: int = 42      # for reproducibility


def build_probe_config(cfg: Dict[str, Any]) -> ProbeConfig:
    """
    Build a ProbeConfig from the global config dict.

    Optional config section:

    layers:
      probe_max_samples: 2000
      probe_batch_size: 32
      probe_test_size: 0.2
    """
    layers_cfg = cfg.get("layers", {})

    return ProbeConfig(
        max_samples=layers_cfg.get("probe_max_samples", 2000),
        batch_size=layers_cfg.get("probe_batch_size", 32),
        test_size=layers_cfg.get("probe_test_size", 0.2),
        random_state=cfg.get("seed", 42),
    )


def extract_layer_representations(
    model: torch.nn.Module,
    tokenizer,
    texts,
    labels,
    device: torch.device,
    probe_cfg: ProbeConfig,
    max_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract [CLS]-pooled representations from each layer of DistilBERT.

    Args:
        model: fine-tuned DistilBertForSequenceClassification (or similar)
        tokenizer: matching tokenizer
        texts: list/array of strings
        labels: list/array of ints (0/1)
        device: torch.device
        probe_cfg: ProbeConfig
        max_length: max token length (should match training, e.g. 256)

    Returns:
        X: np.ndarray of shape [N, L, H]  (N=samples, L=layers, H=hidden_dim)
        y: np.ndarray of shape [N]
    """
    model.eval()
    model.to(device)

    # Limit number of samples for speed
    n_total = len(texts)
    n_use = min(n_total, probe_cfg.max_samples)

    texts = np.array(texts)[:n_use]
    labels = np.array(labels)[:n_use]

    # Tokenize once up front
    encodings = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels_t = torch.tensor(labels, dtype=torch.long)

    ds = TensorDataset(input_ids, attention_mask, labels_t)
    dl = DataLoader(ds, batch_size=probe_cfg.batch_size, shuffle=False)

    all_reprs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Layer probe: extracting reps"):
            batch_input_ids, batch_attention_mask, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states  # tuple length = n_layers + 1
            # For DistilBERT: 7 entries (embeddings + 6 transformer layers)

            # hidden_states[i]: [batch_size, seq_len, hidden_dim]
            # We'll use the [CLS] token (position 0) as the pooled representation
            cls_per_layer = [h[:, 0, :] for h in hidden_states]  # list of [B, H]
            # Stack across layers -> [B, L, H]
            cls_per_layer = torch.stack(cls_per_layer, dim=1)    # [B, L, H]

            all_reprs.append(cls_per_layer.cpu())
            all_labels.append(batch_labels.cpu())

    X = torch.cat(all_reprs, dim=0).numpy()   # [N, L, H]
    y = torch.cat(all_labels, dim=0).numpy()  # [N]

    return X, y


def train_layer_probes(
    X: np.ndarray,
    y: np.ndarray,
    probe_cfg: ProbeConfig,
) -> Dict[str, Any]:
    """
    Train a simple LogisticRegression probe for each layer.

    Args:
        X: [N, L, H] numpy array (from extract_layer_representations)
        y: [N] labels
        probe_cfg: ProbeConfig

    Returns:
        metrics: dict with lists of acc/F1 per layer, plus raw layer indices.
    """
    n_samples, n_layers, hidden_dim = X.shape

    metrics = {
        "n_samples": int(n_samples),
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "layers": [],
        "accuracy": [],
        "f1": [],
    }

    for layer_idx in range(n_layers):
        X_layer = X[:, layer_idx, :]  # [N, H]

        X_train, X_test, y_train, y_test = train_test_split(
            X_layer,
            y,
            test_size=probe_cfg.test_size,
            random_state=probe_cfg.random_state,
            stratify=y,
        )

        clf = LogisticRegression(
            penalty="l2",
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")

        metrics["layers"].append(int(layer_idx))
        metrics["accuracy"].append(float(acc))
        metrics["f1"].append(float(f1))

    return metrics
