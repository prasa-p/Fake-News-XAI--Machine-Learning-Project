#!/usr/bin/env python
"""
File: src/explain/eval_xai.py

Evaluate explanation methods (IG, LIME, SHAP) along three axes:

- Faithfulness: how much the model's confidence drops when we delete
  the top-k most important tokens from the input (per method).
- Stability: how consistent top-k tokens remain under small input
  perturbations, using *_perturbed.jsonl files.
- Plausibility: placeholder only (human ratings not yet available).

Expected explanation file schema (one JSON per line):

{
  "sample_id": int,
  "text": str,
  "tokens": [str, ...],
  "importances": [float, ...],
  "pred_label": int,
  "true_label": int,
  "dataset": "kaggle" | "liar",
  "method": "ig" | "lime" | "shap",
  // optionally:
  "prob_pred": float
}

Perturbed files are expected at:
  artifacts/explanations/{dataset}_{method}_perturbed.jsonl

Metrics are saved to:
  artifacts/metrics/xai_metrics.json

Usage:
  PYTHONPATH=. python src/explain/eval_xai.py \
      --datasets kaggle liar \
      --methods ig lime shap \
      --top-k 10 \
      --max-samples 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EXPL_DIR = ARTIFACTS_DIR / "explanations"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
DISTILBERT_DIR = ARTIFACTS_DIR / "distilbert"

DEFAULT_TOP_K = 10
DEFAULT_MAX_SAMPLES = 100
DEFAULT_MAX_LEN = 256


# ---------------------------------------------------------------------------
# Model loading & prediction helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(dataset: str, device: Optional[str] = None):
    """
    Load the fine-tuned DistilBERT model and tokenizer for a given dataset.

    Expects checkpoint at:
      artifacts/distilbert/{dataset}/final_model
    """
    ckpt_dir = DISTILBERT_DIR / dataset / "final_model"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    label2id = model.config.label2id

    return model, tokenizer, device, id2label, label2id


def predict_prob_for_label(
    text: str,
    model,
    tokenizer,
    device: str,
    max_length: int,
    label_id: int,
) -> float:
    """Return P(y = label_id | text) under the model."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits  # (1, num_labels)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return float(probs[label_id])


# ---------------------------------------------------------------------------
# Explanation IO helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Read a JSONL file into a list of dicts."""
    records: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Explanation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def safe_get_pred_label(rec: Dict) -> Optional[int]:
    """Best-effort extraction of the predicted label id from explanation record."""
    if "pred_label" in rec:
        return int(rec["pred_label"])
    if "predicted_label" in rec:
        return int(rec["predicted_label"])
    return None


# ---------------------------------------------------------------------------
# Faithfulness (deletion test)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def _top_indices(importances: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k |importance| values, restricted by mask."""
    if mask.sum() == 0:
        return np.array([], dtype=int)

    masked_scores = np.abs(importances.copy())
    masked_scores[~mask] = -np.inf
    k = min(k, int(mask.sum()))
    if k <= 0:
        return np.array([], dtype=int)

    top_idx = np.argpartition(-masked_scores, k - 1)[:k]
    # sort by magnitude descending
    top_idx = top_idx[np.argsort(-masked_scores[top_idx])]
    return top_idx


def deletion_text_lime(
    rec: Dict,
    model,
    tokenizer,
    device: str,
    max_length: int,
    top_k: int,
) -> Optional[float]:
    """
    Faithfulness for LIME-style explanations.

    Strategy: remove (string-replace) the top-k tokens from the original
    text and measure the drop in model confidence for the original
    predicted class.
    """
    text = rec["text"]
    tokens = rec["tokens"]
    importances = np.asarray(rec["importances"], dtype=float)

    label_id = safe_get_pred_label(rec)
    if label_id is None:
        # fallback: recompute prediction & use argmax
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        label_id = int(np.argmax(probs))

    # mask out empty tokens
    tokens_arr = np.array(tokens, dtype=object)
    valid_mask = np.array(
        [bool(t.strip()) for t in tokens_arr],
        dtype=bool,
    )
    if valid_mask.sum() == 0:
        return None

    top_idx = _top_indices(importances, valid_mask, top_k)
    if top_idx.size == 0:
        return None

    # Original probability
    p_orig = predict_prob_for_label(text, model, tokenizer, device, max_length, label_id)

    # Remove top-k tokens by naive string replacement
    pert_text = text
    for idx in top_idx:
        tok = str(tokens_arr[idx])
        if not tok.strip():
            continue
        # simple, over-aggressive replacement is OK for this approximate metric
        pert_text = pert_text.replace(tok, " ")

    p_pert = predict_prob_for_label(pert_text, model, tokenizer, device, max_length, label_id)
    return p_orig - p_pert


def deletion_tokens_bert(
    rec: Dict,
    model,
    tokenizer,
    device: str,
    max_length: int,
    top_k: int,
) -> Optional[float]:
    """
    Faithfulness for IG / SHAP explanations with BERT-style tokens.

    Strategy: mask the top-k token *positions* in the input_ids (using
    [MASK] if available) and measure probability drop.
    """
    text = rec["text"]
    tokens = rec["tokens"]
    importances = np.asarray(rec["importances"], dtype=float)

    label_id = safe_get_pred_label(rec)
    if label_id is None:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        label_id = int(np.argmax(probs))

    # Encode text exactly as during training
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = enc["input_ids"][0]  # (L,)
    if len(tokens) != int(input_ids.shape[0]):
        # Tokenization mismatch; skip this record
        return None

    # build mask of positions that are eligible for deletion
    bert_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    valid_mask = np.array(
        [t not in SPECIAL_TOKENS for t in bert_tokens],
        dtype=bool,
    )
    if valid_mask.sum() == 0:
        return None

    top_idx = _top_indices(importances, valid_mask, top_k)
    if top_idx.size == 0:
        return None

    # Original probability
    p_orig = predict_prob_for_label(text, model, tokenizer, device, max_length, label_id)

    # Mask top-k positions
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        # fall back to [UNK] if [MASK] missing
        mask_id = tokenizer.unk_token_id

    pert_input_ids = input_ids.clone()
    for idx in top_idx:
        pert_input_ids[idx] = mask_id

    pert_enc = {
        "input_ids": pert_input_ids.unsqueeze(0).to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }

    with torch.no_grad():
        logits = model(**pert_enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    p_pert = float(probs[label_id])

    return p_orig - p_pert


def evaluate_faithfulness_for_file(
    dataset: str,
    method: str,
    model,
    tokenizer,
    device: str,
    top_k: int,
    max_samples: int,
    max_length: int,
) -> Optional[Dict]:
    """Compute average deletion drop for a dataset/method."""
    path = EXPL_DIR / f"{dataset}_{method}.jsonl"
    if not path.exists():
        print(f"[faithfulness] {path} not found, skipping.")
        return None

    records = read_jsonl(path, max_samples=max_samples)

    drops: List[float] = []
    n_errors = 0
    n_skipped_none = 0
    n_skipped_nan = 0

    for rec in records:
        sample_id = rec.get("sample_id")
        drop = None

        try:
            if method == "lime":
                drop = deletion_text_lime(
                    rec, model, tokenizer, device, max_length, top_k
                )
            else:  # ig / shap
                drop = deletion_tokens_bert(
                    rec, model, tokenizer, device, max_length, top_k
                )

        except Exception as e:
            n_errors += 1
            print(
                f"[faithfulness] error on {dataset}/{method}, "
                f"sample_id={sample_id}: {e}"
            )
            # For SHAP, show a bit more detail + stack trace
            if method == "shap":
                print(
                    f"[DEBUG faithfulness shap] exception details for "
                    f"{dataset}/{method}, sample_id={sample_id}"
                )
                print(f"  rec keys: {list(rec.keys())}")
                print(f"  len(tokens)={len(rec.get('tokens', []))}, "
                      f"len(importances)={len(rec.get('importances', []))}")
                traceback.print_exc() # type: ignore
            drop = None  # ensure we skip below

        # Skip invalid drops; log why for SHAP
        if drop is None:
            n_skipped_none += 1
            if method == "shap":
                print(
                    f"[DEBUG faithfulness shap] drop is None for {dataset}/{method}, "
                    f"sample_id={sample_id}. "
                    f"len(tokens)={len(rec.get('tokens', []))}, "
                    f"len(importances)={len(rec.get('importances', []))}"
                )
            continue

        # np.isnan on non-float raises, so be safe
        try:
            if np.isnan(drop):
                n_skipped_nan += 1
                if method == "shap":
                    print(
                        f"[DEBUG faithfulness shap] drop is NaN for {dataset}/{method}, "
                        f"sample_id={sample_id}, drop={drop}"
                    )
                continue
        except TypeError:
            # If drop is not a scalar, this is also a bug – log it
            n_skipped_nan += 1
            if method == "shap":
                print(
                    f"[DEBUG faithfulness shap] drop has unexpected type "
                    f"({type(drop)}) for {dataset}/{method}, "
                    f"sample_id={sample_id}, value={drop}"
                )
            continue

        drops.append(float(drop))

    print(
        f"[DEBUG faithfulness] Finished {dataset}/{method}, "
        f"valid={len(drops)}, skipped_none={n_skipped_none}, "
        f"skipped_nan={n_skipped_nan}, errors={n_errors}."
    )

    if not drops:
        print(f"[faithfulness] No valid samples for {dataset}/{method}.")
        return None

    drops_arr = np.array(drops, dtype=float)
    metrics = {
        "dataset": dataset,
        "method": method,
        "n": int(len(drops_arr)),
        "avg_drop": float(drops_arr.mean()),
        "avg_abs_drop": float(np.abs(drops_arr).mean()),
        "prop_positive_drop": float((drops_arr > 0).mean()),
    }
    print(f"[faithfulness] {dataset}_{method}: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Stability (top-k Jaccard using *_perturbed.jsonl)
# ---------------------------------------------------------------------------

def top_token_set(rec: Dict, k: int) -> set:
    """
    Return a set of top-k token strings for a record, using |importance|.

    We normalize some BERT quirks (strip '##', lower-case) and ignore
    obvious special tokens / empty strings.
    """
    tokens = rec["tokens"]
    importances = np.asarray(rec["importances"], dtype=float)

    pairs = []
    for tok, imp in zip(tokens, importances):
        t = str(tok).strip()
        if not t:
            continue
        if t in SPECIAL_TOKENS:
            continue
        # normalize subwords like "##ing" -> "ing"
        if t.startswith("##"):
            t = t[2:]
        pairs.append((t.lower(), abs(float(imp))))

    if not pairs:
        return set()

    pairs.sort(key=lambda x: -x[1])
    k = min(k, len(pairs))
    top = [w for (w, _) in pairs[:k]]
    return set(top)


def evaluate_stability_for_file(
    dataset: str,
    method: str,
    top_k: int,
    max_samples: int,
) -> Optional[Dict]:
    """
    Stability metric: Jaccard similarity between top-k token sets for
    original vs perturbed explanations.

    Requires both:
      artifacts/explanations/{dataset}_{method}.jsonl
      artifacts/explanations/{dataset}_{method}_perturbed.jsonl
    """
    orig_path = EXPL_DIR / f"{dataset}_{method}.jsonl"
    pert_path = EXPL_DIR / f"{dataset}_{method}_perturbed.jsonl"

    if not orig_path.exists() or not pert_path.exists():
        print(f"[stability] Skipping {dataset}_{method} (need {orig_path} AND {pert_path})")
        return None

    orig_records = read_jsonl(orig_path, max_samples=max_samples)
    pert_records = read_jsonl(pert_path, max_samples=max_samples)

    n = min(len(orig_records), len(pert_records))
    if n == 0:
        print(f"[stability] No paired samples for {dataset}/{method}.")
        return None

    jaccards: List[float] = []

    for i in range(n):
        r0 = orig_records[i]
        r1 = pert_records[i]

        # Optionally enforce same sample_id
        if r0.get("sample_id") != r1.get("sample_id"):
            # Still proceed, but warning once
            pass

        s0 = top_token_set(r0, top_k)
        s1 = top_token_set(r1, top_k)

        union = s0.union(s1)
        if not union:
            continue
        inter = s0.intersection(s1)
        j = len(inter) / len(union)
        jaccards.append(j)

    if not jaccards:
        print(f"[stability] No valid overlaps for {dataset}/{method}.")
        return None

    j_arr = np.array(jaccards, dtype=float)
    metrics = {
        "dataset": dataset,
        "method": method,
        "n": int(len(j_arr)),
        "avg_jaccard": float(j_arr.mean()),
        "std_jaccard": float(j_arr.std()),
    }
    print(f"[stability] {dataset}_{method}: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Plausibility (placeholder)
# ---------------------------------------------------------------------------

def evaluate_plausibility_placeholder():
    """
    Placeholder for plausibility metric based on human ratings.

    Expected future schema for ratings file:
      artifacts/metrics/human_ratings.jsonl, with fields like:
        {
          "sample_id": int,
          "dataset": str,
          "method": str,
          "tokens": [...],
          "importances": [...],
          "human_score": float   # e.g. 1-5 scale
        }

    Currently, we just log that plausibility is skipped.
    """
    ratings_path = METRICS_DIR / "human_ratings.jsonl"
    if not ratings_path.exists():
        print("[plausibility] No human_ratings.jsonl found, skipping plausibility.")
        return None

    # If you do get ratings later, implement aggregation here.
    print("[plausibility] human_ratings.jsonl found, but plausibility metric "
          "is not yet implemented.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["kaggle", "liar"],
        help="Datasets to evaluate (e.g., kaggle liar)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lime", "shap", "ig"],
        help="Explanation methods to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k tokens to delete or compare.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Max samples per dataset/method to evaluate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Max sequence length (should match training/explanation config).",
    )
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "faithfulness": {},
        "stability": {},
        "plausibility": None,
    }

    # Faithfulness & stability
    for dataset in args.datasets:
        # Load fine-tuned model for this dataset once
        try:
            model, tokenizer, device, id2label, label2id = load_model_and_tokenizer(dataset)
        except Exception as e:
            print(f"[eval_xai] Could not load model for dataset '{dataset}': {e}")
            continue

        for method in args.methods:
            # Faithfulness
            faith = evaluate_faithfulness_for_file(
                dataset=dataset,
                method=method,
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_k=args.top_k,
                max_samples=args.max_samples,
                max_length=args.max_length,
            )
            if faith is not None:
                all_metrics["faithfulness"].setdefault(dataset, {})[method] = faith

            # Stability
            stab = evaluate_stability_for_file(
                dataset=dataset,
                method=method,
                top_k=args.top_k,
                max_samples=args.max_samples,
            )
            if stab is not None:
                all_metrics["stability"].setdefault(dataset, {})[method] = stab

    # Plausibility (placeholder)
    all_metrics["plausibility"] = evaluate_plausibility_placeholder()

    out_path = METRICS_DIR / "xai_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved XAI metrics to {out_path}")


if __name__ == "__main__":
    main()
