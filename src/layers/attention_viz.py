# src/layers/attention_viz.py
"""
Attention visualization utilities for DistilBERT fake news classifier.

Responsibilities:
- Generate and save attention visualizations from DistilBERT for selected examples.
- Provide simple, static matplotlib plots (CLS -> token attention).
- Help illustrate how early vs. late layers attend to tokens in fake vs. real news.

Usage examples:
    # 1 fake + 1 real high-confidence sample (default), all layers, Kaggle
    PYTHONPATH=. python -m src.layers.attention_viz --dataset kaggle

    # 3 fake + 3 real high-confidence samples, last layer only
    PYTHONPATH=. python -m src.layers.attention_viz --dataset kaggle --n 3 --layers last

    # Specific layers 0 and 5
    PYTHONPATH=. python -m src.layers.attention_viz --dataset liar --layers 0,5 --n 2

    # Custom text file instead of dataset-based selection (ignores --n)
    PYTHONPATH=. python -m src.layers.attention_viz --dataset kaggle --sample_path my_article.txt --layers last

Figures are written under:
    artifacts/layers/
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import List, Dict, Any, Optional, Sequence, Tuple

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import load_config


# ------------------------------------------------------------------------
# Low-level attention plotting
# ------------------------------------------------------------------------


def _get_layer_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper: returns the 'layers' sub-config (or defaults).

    Optional fields in config/default.yaml:

    layers:
      attention_layer_index: -1        # default layer if none specified via CLI
      attention_max_tokens: 64         # truncate very long sequences in plots
    """
    layers_cfg = cfg.get("layers", {})
    return {
        "layer_index": layers_cfg.get("attention_layer_index", -1),
        "max_tokens": layers_cfg.get("attention_max_tokens", 64),
    }


def plot_cls_attention(
    tokens: List[str],
    cls_attn: torch.Tensor,
    layer_idx: int,
    meta: Dict[str, Any],
    out_path: str,
    top_k: int = 10,
    raw_text: Optional[str] = None,
) -> None:
    """
    Plot CLS->token attention for a single layer/sample.

    Args:
        tokens: list of wordpiece tokens (including specials).
        cls_attn: 1D tensor of attention weights for CLS over all tokens (len = T).
        layer_idx: which transformer layer (0-based).
        meta: dict with at least:
              - "dataset_name": str, e.g. "Kaggle"
              - "sample_idx": int or str
              - "true_label": int or "N/A"
              - "pred_label": int or "N/A"
              - "pred_conf": float in [0,1] or "N/A"
        out_path: where to save the figure (PNG).
        top_k: number of top tokens to highlight.
        raw_text: original article/statement text for the caption.
    """
    # ---- Convert to numpy ----
    if isinstance(cls_attn, torch.Tensor):
        cls_attn = cls_attn.detach().cpu().numpy() # type: ignore
    cls_attn = np.asarray(cls_attn, dtype=float) # type: ignore

    # ---- Drop special tokens completely (no bar, no label) ----
    specials = {"[CLS]", "[SEP]", "[PAD]"}
    filtered_tokens: List[str] = []
    filtered_scores: List[float] = []
    for tok, score in zip(tokens, cls_attn):
        if tok in specials:
            continue
        filtered_tokens.append(tok)
        filtered_scores.append(score) # type: ignore

    if not filtered_tokens:
        # Nothing to plot
        return

    tokens = filtered_tokens
    cls_attn = np.asarray(filtered_scores, dtype=float) # type: ignore

    # Renormalize after removing specials
    total = cls_attn.sum()
    if total > 0:
        cls_attn = cls_attn / total

    T = len(tokens)
    x = np.arange(T)

    # ---- Determine top-k tokens ----
    k = min(top_k, T)
    if k > 0:
        order = np.argsort(cls_attn)[::-1][:k]
        top_indices = set(order.tolist())
    else:
        top_indices = set()

    # Colors: blue for top-k tokens, grey for others
    main_color = "#1f77b4"
    grey_color = "#d3d3d3"
    bar_colors = [
        main_color if i in top_indices else grey_color for i in range(T)
    ]

    # ---- Figure + axes ----
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.bar(
        x,
        cls_attn,
        width=0.8,
        color=bar_colors,
        alpha=0.9,
        align="center",
    )

    # Legend (dummy handles)
    from matplotlib.patches import Patch

    handles = [
        Patch(color=main_color, label=f"Top {min(top_k, len(top_indices))} tokens"),
        Patch(color=grey_color, label="Other tokens"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10)

    # ---- X tick labels: show all (non-special) tokens ----
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=10)
    ax.set_ylabel("CLS attention weight")

    # ---- Title with full context ----
    dataset_name = meta.get("dataset_name", "Dataset")
    sample_idx = meta.get("sample_idx", "?")

    true_raw = meta.get("true_label", None)
    pred_raw = meta.get("pred_label", None)
    conf = meta.get("pred_conf", None)

    label_names = meta.get("class_names", {0: "fake", 1: "real"})

    def fmt_label(raw):
        if isinstance(raw, (int, np.integer)):
            name = label_names.get(int(raw), str(int(raw)))
            return f"{name} ({int(raw)})"
        if raw is None or str(raw).upper() == "N/A":
            return "N/A"
        return str(raw)

    true_str = fmt_label(true_raw)
    pred_str = fmt_label(pred_raw)
    if conf is None or str(conf).upper() == "N/A":
        conf_str = "?"
    else:
        conf_str = f"{float(conf):.2f}"

    title = (
        f"{dataset_name} sample #{sample_idx} — "
        f"true_label={true_str}, "
        f"pred={pred_str}, "
        f"conf={conf_str} — "
        f"Layer {layer_idx} CLS→Token Attention"
    )
    ax.set_title(title)

    # Leave room at the bottom for caption + top-token summary
    plt.subplots_adjust(bottom=0.22)
    """
    # ------------------------------------------------------------------
    # Text summary under the plot
    # ------------------------------------------------------------------
    # 1) Top tokens line (sorted by attention, highest first)
    if top_indices:
        sorted_top = sorted(top_indices, key=lambda i: cls_attn[i], reverse=True)
        top_token_strings = []
        for idx in sorted_top:
            tok = tokens[idx]
            clean_tok = tok.replace("##", "")
            top_token_strings.append(clean_tok)

        top_line = "Top tokens: " + ", ".join(top_token_strings)
        top_text = fig.text(
            0.5,
            0.16,
            top_line,
            ha="center",
            va="bottom",
            fontsize=11,
            color=main_color,
        )
        top_text.set_wrap(True)

    # 2) Original sentence/statement as caption
    if raw_text:
        wrapped = textwrap.fill(raw_text, width=120)
        caption = fig.text(
            0.5,
            0.06,
            wrapped,
            ha="center",
            va="bottom",
            fontsize=10,
        )
        caption.set_wrap(True)
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def visualize_attention_for_sample(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    cfg: Dict[str, Any],
    out_path: str,
    layer_index: int,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Visualize CLS -> token attention for a single text example at a single layer.
    """
    model.eval()
    device = next(model.parameters()).device

    layer_cfg = _get_layer_cfg(cfg)
    max_length = cfg.get("data", {}).get("max_length", 256)
    max_tokens_plot = layer_cfg["max_tokens"]

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(
            **enc,
            output_attentions=True,
        )

    attentions = outputs.attentions
    n_layers = len(attentions)
    if layer_index < 0 or layer_index >= n_layers:
        raise ValueError(f"Invalid layer_index={layer_index}, n_layers={n_layers}")

    # [1, heads, seq_len, seq_len] -> [heads, seq_len, seq_len]
    att_layer = attentions[layer_index][0]
    att_mean = att_layer.mean(dim=0)  # [seq_len, seq_len]
    cls_to_tokens = att_mean[0]       # [seq_len]

    input_ids = enc["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    seq_len = len(tokens)
    if seq_len > max_tokens_plot:
        tokens = tokens[:max_tokens_plot]
        cls_to_tokens = cls_to_tokens[:max_tokens_plot]

    scores = cls_to_tokens.detach().cpu().numpy()
    total = scores.sum()
    if total > 0:
        scores = scores / total

    base_meta = {
        "dataset_name": cfg.get("data", {}).get("name", "Dataset"),
        "class_names": cfg.get("data", {}).get("class_names", {0: "fake", 1: "real"}),
    }
    if meta is not None:
        base_meta.update(meta)
    meta = base_meta

    # Decide top-k based on dataset: 5 for LIAR, 10 otherwise
    dataset_name = str(meta.get("dataset_name", "")).lower()
    if dataset_name.startswith("liar"):
        top_k = 5
    else:
        top_k = 10

    plot_cls_attention(
        tokens=tokens,
        cls_attn=scores,
        layer_idx=layer_index,
        meta=meta,
        out_path=out_path,
        top_k=top_k,
        raw_text=text,
    )


# ------------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------------


def _load_test_split(dataset: str) -> Tuple[List[str], List[int]]:
    """
    Load test split from data/processed for a given dataset.

    Expects:
        data/processed/kaggle_test.csv  with columns ['text', 'label']
        data/processed/liar_test.csv    with columns ['text', 'label']
    """
    if dataset not in {"kaggle", "liar"}:
        raise ValueError(f"Unknown dataset: {dataset}")

    path = os.path.join("data", "processed", f"{dataset}_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed test file: {path}")

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected 'text' and 'label' columns in {path}")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def _predict_probs(
    texts: Sequence[str],
    model: torch.nn.Module,
    tokenizer,
    max_length: int,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Run the model over a list of texts and return class probabilities.

    Returns:
        probs: np.ndarray of shape (n_samples, n_classes)
    """
    device = next(model.parameters()).device
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        enc = tokenizer(
            list(batch_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def _select_high_confidence_indices(
    texts: Sequence[str],
    labels: Sequence[int],
    probs: np.ndarray,
    n_per_class: int,
) -> List[int]:
    """
    Select up to n_per_class correctly classified, high-confidence samples
    for each class (0 and 1).
    """
    labels = np.array(labels) # type: ignore
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    selected_indices: List[int] = []
    for cls in (0, 1):
        mask = (labels == cls) & (preds == cls)
        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            continue

        order = np.argsort(-confs[cand_idx])
        top = cand_idx[order][:n_per_class]
        selected_indices.extend(top.tolist())

    return sorted(set(selected_indices))


# ------------------------------------------------------------------------
# Layer selection parsing
# ------------------------------------------------------------------------


def _parse_layers_arg(layers_str: str, num_layers: int) -> List[int]:
    """
    Parse the --layers argument.

    Allowed forms:
      - "all"     -> [0, 1, ..., num_layers-1]
      - "last"    -> [num_layers-1]
      - "0,3,5"   -> [0, 3, 5]
    """
    s = layers_str.strip().lower()
    if s == "all":
        return list(range(num_layers))
    if s == "last":
        return [num_layers - 1]

    indices: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {num_layers - 1}]")
        indices.append(idx)

    if not indices:
        raise ValueError(f"Could not parse --layers from: {layers_str}")
    return indices


# ------------------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLS→token attention visualizations for DistilBERT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config (used mainly for max_length).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kaggle",
        choices=["kaggle", "liar"],
        help="Which dataset's fine-tuned model + test split to use.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help=(
            "Number of high-confidence samples PER CLASS to visualize if no "
            "--sample_path is provided. For binary labels 0/1, n=1 gives up "
            "to 2 samples (one fake, one real)."
        ),
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Which layers to visualize: 'all', 'last', or comma-separated indices.",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default=None,
        help=(
            "Optional path to a text file containing a single article/statement. "
            "If provided, ignores --n and uses only this sample."
        ),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="artifacts/layers",
        help="Root directory to save figures into.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    max_length = data_cfg.get("max_length", 256)

    ckpt_dir = os.path.join("artifacts", "distilbert", args.dataset, "final_model")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Missing checkpoint directory: {ckpt_dir}")

    print(f"[attention_viz] Using checkpoint: {ckpt_dir}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_layers = model.config.num_hidden_layers
    layer_indices = _parse_layers_arg(args.layers, num_layers)
    print(f"[attention_viz] Will visualize layers: {layer_indices}")

    os.makedirs(args.out_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Case 1: custom sample from text file
    # ------------------------------------------------------------------
    if args.sample_path is not None:
        if not os.path.exists(args.sample_path):
            raise FileNotFoundError(f"sample_path does not exist: {args.sample_path}")
        with open(args.sample_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError(f"sample_path {args.sample_path} is empty.")

        probs = _predict_probs([text], model, tokenizer, max_length=max_length)
        pred = int(probs[0].argmax())
        conf = float(probs[0].max())
        print(f"[attention_viz] Custom sample prediction: label={pred}, conf={conf:.3f}")

        meta = {
            "dataset_name": args.dataset.capitalize(),
            "sample_idx": "custom",
            "true_label": "N/A",
            "pred_label": pred,
            "pred_conf": conf,
            "class_names": {0: "fake", 1: "real"},
        }

        for L in layer_indices:
            out_name = f"{args.dataset}_custom_layer{L}.png"
            out_path = os.path.join(args.out_root, out_name)
            visualize_attention_for_sample(
                text=text,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                out_path=out_path,
                layer_index=L,
                meta=meta,
            )
            print(f"[attention_viz] Saved {out_path}")
        return

    # ------------------------------------------------------------------
    # Case 2: high-confidence samples from test split
    # ------------------------------------------------------------------
    texts, labels = _load_test_split(args.dataset)
    print(f"[attention_viz] Loaded {len(texts)} test samples from {args.dataset}.")

    probs = _predict_probs(texts, model, tokenizer, max_length=max_length, batch_size=32)
    indices = _select_high_confidence_indices(
        texts=texts,
        labels=labels,
        probs=probs,
        n_per_class=args.n,
    )

    if not indices:
        raise RuntimeError(
            "No high-confidence correctly classified samples found; "
            "check model and labels."
        )

    print(
        f"[attention_viz] Selected {len(indices)} high-confidence samples "
        f"(up to {args.n} per class)."
    )

    for idx in indices:
        text = texts[idx]
        label = labels[idx]
        pred = int(probs[idx].argmax())
        conf = float(probs[idx].max())

        meta = {
            "dataset_name": args.dataset.capitalize(),
            "sample_idx": idx,
            "true_label": label,
            "pred_label": pred,
            "pred_conf": conf,
            "class_names": {0: "fake", 1: "real"},
        }

        for L in layer_indices:
            out_name = (
                f"{args.dataset}_idx{idx}_true{label}_pred{pred}_"
                f"conf{conf:.2f}_L{L}.png"
            )
            out_path = os.path.join(args.out_root, out_name)
            try:
                visualize_attention_for_sample(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    out_path=out_path,
                    layer_index=L,
                    meta=meta,
                )
                print(f"[attention_viz] Saved {out_path}")
            except Exception as e:
                print(f"[attention_viz] Error on idx={idx}, layer={L}: {e}")


if __name__ == "__main__":
    main()
