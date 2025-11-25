"""
File: explain/ig_utils.py

Responsibilities:
- Implement Integrated Gradients (IG) for DistilBERT using Captum.
- Optionally provide LayerIntegratedGradients for layer-aware attributions.
- Convert raw attributions to human-readable token importance scores.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- compute_ig_attributions(text_batch: list[str], model, tokenizer, cfg) -> list[dict]
- compute_layer_ig_attributions(text_batch: list[str], model, tokenizer, layer, cfg) -> list[dict]
- format_attributions(tokens: list[str], attributions) -> dict

Schema for each explanation:
{
    "sample_id": int,
    "text": str,
    "tokens": [str],
    "importances": [float],
    "pred_label": int,
    "true_label": int,
    "method": "ig",
    "dataset": "kaggle" or "liar"
}

These will be saved as JSON lines in artifacts/explanations/<method>_<run-id>.jsonl
"""
