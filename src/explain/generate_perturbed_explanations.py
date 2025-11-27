#!/usr/bin/env python
"""
File: src/explain/generate_perturbed_explanations.py

Generate "perturbed" explanation files to support stability evaluation.

For each dataset/method, this script:
  - Loads the fine-tuned DistilBERT classifier.
  - Reads original explanations from:
        artifacts/explanations/{dataset}_{method}.jsonl
  - For up to --max-samples records:
        * Perturbs the input text by randomly deleting ~15% of words.
        * Recomputes the explanation with the same method
          (IG / LIME / SHAP) on the perturbed text.
        * Saves a new JSONL file:
              artifacts/explanations/{dataset}_{method}_perturbed.jsonl

Explanation schema matches eval_xai expectations (see eval_xai.py).
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from captum.attr import IntegratedGradients
from lime.lime_text import LimeTextExplainer
import shap


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EXPL_DIR = ARTIFACTS_DIR / "explanations"
DISTILBERT_DIR = ARTIFACTS_DIR / "distilbert"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(dataset: str, device: Optional[str] = None):
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


def read_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def perturb_text_drop_words(text: str, drop_frac: float = 0.15) -> str:
    """
    Simple perturbation: randomly drop ~drop_frac of tokens from the text.
    Tries to keep at least a few words.
    """
    words = text.split()
    n = len(words)
    if n <= 5:
        # Too short, do a no-op perturbation
        return text

    k = max(1, int(drop_frac * n))
    drop_indices = set(random.sample(range(n), k))
    kept = [w for i, w in enumerate(words) if i not in drop_indices]
    if not kept:
        kept = words  # safety
    return " ".join(kept)


# ---------------------------------------------------------------------------
# IG per-sample explanation
# ---------------------------------------------------------------------------

from src.explain.ig_utils import explain_sample_ig
from src.utils import load_config
import numpy as np

def explain_with_ig(
    text: str,
    model,
    tokenizer,
    cfg: dict,
    sample_id: int | None = None,
    true_label: int | None = None,
):
    """
    Thin wrapper around src.explain.ig_utils.explain_sample_ig.

    Returns:
        tokens: list[str]
        scores: np.ndarray of shape (seq_len,)
        pred_label: int
        raw: full dict from explain_sample_ig
    """
    raw = explain_sample_ig(
        text=text,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        sample_id=sample_id,
        true_label=true_label,
    )

    tokens = raw["tokens"]
    scores = np.array(raw["importances"], dtype=float)
    pred_label = int(raw["pred_label"])

    return tokens, scores, pred_label, raw


def explain_ig_single(
    text: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 256,
    n_steps: int = 32,
) -> Tuple[List[str], List[float], int, float]:
    """
    Compute Integrated Gradients attributions for a single text.

    Returns:
        tokens: list of BERT tokens (length L)
        importances: per-token IG scores (length L)
        pred_label: int
        prob_pred: float (probability of predicted class)
    """
    # Load config (or you can pass this in instead of loading every time)
    cfg = load_config("config/default.yaml")

    # Use the shared wrapper that calls explain_sample_ig
    tokens, scores, pred_label, _ = explain_with_ig(
        text=text,
        model=model,
        tokenizer=tokenizer,   # <-- use tokenizer, NOT enc
        cfg=cfg,
        sample_id=None,
        true_label=None,
    )

    # Get predicted probability for the predicted class
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        prob_pred = float(probs[pred_label].item())

    # scores is a numpy array already; convert to list if your caller expects list
    return tokens, scores.tolist(), pred_label, prob_pred


# ---------------------------------------------------------------------------
# LIME per-sample explanation
# ---------------------------------------------------------------------------

def build_lime_explainer(class_names: List[str]) -> LimeTextExplainer:
    return LimeTextExplainer(
        class_names=class_names,
        bow=True,
        split_expression=r"\W+",
    )


def make_lime_predict_fn(model, tokenizer, device: str, max_length: int = 256):
    def predict(texts: List[str]) -> np.ndarray:
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
    return predict


def explain_lime_single(
    text: str,
    model,
    tokenizer,
    device: str,
    lime_explainer: LimeTextExplainer,
    max_length: int = 256,
    num_features: int = 15,
    num_samples: int = 500,
) -> Tuple[List[str], List[float], int, float]:
    predict_fn = make_lime_predict_fn(model, tokenizer, device, max_length)
    # First get prediction to know which label to explain
    probs = predict_fn([text])[0]
    pred_label = int(np.argmax(probs))
    prob_pred = float(probs[pred_label])

    explanation = lime_explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(pred_label,),
    )

    # LIME returns top features only
    feature_list = explanation.as_list(label=pred_label)  # [(token, weight), ...]
    tokens = [t for (t, _) in feature_list]
    scores = [float(w) for (_, w) in feature_list]

    return tokens, scores, pred_label, prob_pred


# ---------------------------------------------------------------------------
# SHAP per-sample explanation
# ---------------------------------------------------------------------------

def build_shap_explainer(
    model,
    tokenizer,
    device: str,
    max_length: int = 256,
):
    """
    Build a SHAP Text masker + Explainer for the current model.
    """

    def shap_predict(texts: List[str]) -> np.ndarray:
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    masker = shap.maskers.Text(tokenizer) # type: ignore
    explainer = shap.Explainer(shap_predict, masker)
    return explainer


def explain_shap_single(
    text: str,
    shap_explainer,
    tokenizer,
    device: str,
    max_length: int = 256,
) -> Tuple[List[str], List[float], int, float]:
    """
    Compute SHAP values for a single text.

    Returns:
        tokens: list[str]
        importances: list[float]
        pred_label: int
        prob_pred: float
    """
    # We need prediction & label
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # Use CPU probs to get label
    with torch.no_grad():
        logits = shap_explainer.model(
            tokenizer.decode(enc["input_ids"][0].tolist(), skip_special_tokens=False)
        )
    # However, this usage of shap.Model is messy; more robust is to
    # call tokenizer+model directly:
    # Recompute directly:
    from transformers import PreTrainedTokenizerBase  # noqa: F401  (doc hint)
    from transformers import PreTrainedModel  # noqa: F401

    # NOTE: we can't easily reach shap_explainer.model internals here
    # in a type-safe way, so instead we call explainer on [text],
    # then use the resulting .values and .data fields.
    shap_values = shap_explainer([text])
    # shap_values.values: (batch, seq_len, num_classes)
    values = shap_values.values[0]  # (seq_len, num_classes)
    data_tokens = shap_values.data[0]  # list of tokens as strings

    # Derive predicted label from aggregated SHAP or re-run model
    # (safer: re-run model here)
    # shap_explainer has .model; we call it with [text]
    probs = shap_explainer.model([text])
    # shap.Model returns numpy array
    probs = np.array(probs)[0]
    pred_label = int(np.argmax(probs))
    prob_pred = float(probs[pred_label])

    # Use SHAP values for that label
    token_scores = values[:, pred_label]

    return list(data_tokens), token_scores.tolist(), pred_label, prob_pred


# ---------------------------------------------------------------------------
# Main driver to generate perturbed explanations
# ---------------------------------------------------------------------------

def generate_for_dataset_method(
    dataset: str,
    method: str,
    max_samples: int,
    max_length: int,
    drop_frac: float,
) -> None:
    orig_path = EXPL_DIR / f"{dataset}_{method}.jsonl"
    if not orig_path.exists():
        print(f"[perturbed] {orig_path} not found, skipping {dataset}/{method}.")
        return

    print(f"[perturbed] Loading model/tokenizer for dataset '{dataset}'...")
    model, tokenizer, device, id2label, label2id = load_model_and_tokenizer(dataset)

    records = read_jsonl(orig_path, max_samples=max_samples)
    perturbed_records: List[Dict] = []

    lime_explainer = None
    shap_explainer = None

    if method == "lime":
        class_names = [id2label.get(i, str(i)) for i in range(len(id2label))]
        lime_explainer = build_lime_explainer(class_names)
    elif method == "shap":
        shap_explainer = build_shap_explainer(model, tokenizer, device, max_length)

    for rec in records:
        text = rec["text"]
        true_label = rec.get("true_label")

        # Make a lightly perturbed version of the text
        pert_text = perturb_text_drop_words(text, drop_frac=drop_frac)

        try:
            if method == "ig":
                tokens, imps, pred_label, prob_pred = explain_ig_single(
                    pert_text, model, tokenizer, device, max_length=max_length
                )
            elif method == "lime":
                tokens, imps, pred_label, prob_pred = explain_lime_single(
                    pert_text,
                    model,
                    tokenizer,
                    device,
                    lime_explainer, # type: ignore
                    max_length=max_length,
                    num_features=15,
                    num_samples=500,
                )
            elif method == "shap":
                tokens, imps, pred_label, prob_pred = explain_shap_single(
                    pert_text,
                    shap_explainer,
                    tokenizer,
                    device,
                    max_length=max_length,
                )
            else:
                print(f"[perturbed] Unknown method '{method}', skipping.")
                return
        except Exception as e:
            print(f"[perturbed] Error explaining {dataset}/{method}, "
                  f"sample_id={rec.get('sample_id')}: {e}")
            continue

        perturbed_rec = {
            "sample_id": rec.get("sample_id"),
            "text": pert_text,
            "tokens": tokens,
            "importances": imps,
            "pred_label": int(pred_label),
            "true_label": int(true_label) if true_label is not None else None,
            "dataset": dataset,
            "method": method,
            "prob_pred": float(prob_pred),
            "perturbation": f"drop_words_{drop_frac:.2f}",
        }
        perturbed_records.append(perturbed_rec)

    out_path = EXPL_DIR / f"{dataset}_{method}_perturbed.jsonl"
    write_jsonl(out_path, perturbed_records)
    print(f"[perturbed] Wrote {len(perturbed_records)} records to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["kaggle", "liar"],
        help="Datasets to process (e.g., kaggle liar).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ig", "lime", "shap"],
        help="Explanation methods to process.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max explanations to perturb per dataset/method.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length for tokenizer.",
    )
    parser.add_argument(
        "--drop-frac",
        type=float,
        default=0.15,
        help="Fraction of words to randomly drop in perturbation.",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for dataset in args.datasets:
        for method in args.methods:
            generate_for_dataset_method(
                dataset=dataset,
                method=method,
                max_samples=args.max_samples,
                max_length=args.max_length,
                drop_frac=args.drop_frac,
            )


if __name__ == "__main__":
    main()
