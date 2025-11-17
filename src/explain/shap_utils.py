"""
File: explain/shap_utils.py

Responsibilities:
- Wrap SHAP (KernelSHAP or other variants) for generating explanations on text inputs.
- Provide utilities to compute token- or word-level SHAP values for a subset of samples.
- Serialize SHAP outputs for later metric calculations and visualization.

Contributors:
- <Name 1>
- <Name 2>
- <Name 3>

Key functions to implement:
- make_shap_explainer(model, tokenizer, cfg)
- explain_sample_shap(text: str, explainer, cfg) -> dict
- run_shap_batch(dataset, model, tokenizer, cfg) -> list[dict]
"""



"""
All explanation methods should output a list of dicts with fields:

{
  "sample_id": int or str,
  "text": str,
  "tokens": [str, ...],
  "importances": [float, ...],  # same length as tokens
  "pred_label": int,
  "true_label": int
}

These will be saved as JSON lines in artifacts/explanations/<method>_<run-id>.jsonl
"""
