"""
File: explain/shap_utils.py

Responsibilities:
- Wrap SHAP (KernelSHAP or other variants) for generating explanations on text inputs.
- Provide utilities to compute token- or word-level SHAP values for a subset of samples.
- Serialize SHAP outputs for later metric calculations and visualization.

Contributors:
- Ryan Wilson
- <Name 2>
- <Name 3>

Key functions to implement:
- make_shap_explainer(model, tokenizer, cfg)
- explain_sample_shap(text: str, explainer, cfg) -> dict
- run_shap_batch(dataset, model, tokenizer, cfg) -> list[dict]
"""
import numpy as np
from typing import List, Dict, Any
import shap
import torch
from tqdm import tqdm

def make_shap_explainer(model, tokenizer, cfg):
    """
    Build a SHAP explainer using SHAP's transformer text masker, which
    understands token boundaries and BERT tokenization

    Args:
        model: HuggingFace transformer model
        tokenizer: HuggingFace tokenizer
        cfg: config dictionary

    Returns:
        shap.Explainer instance
    """
    shap_cfg = cfg.get("shap", {})  
    device = next(model.parameters()).device
    model.eval()
  
    def predict(texts):
        """
        Prediction function for SHAP.
        Takes a list/array of text strings and returns class probabilities.
        
        Args:
            texts: List or array of text strings
        
        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        #Filter empty strings
        texts = [t if len(t.strip()) > 0 else "[MASK]" for t in texts]

        #Tokenize inputs
        inputs = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=shap_cfg.get("max_seq_length", 512)
        ) 
        #Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        #Get predictions
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        return probs
    
    #Build SHAP explainer
    explainer = shap.Explainer(
        model=predict,
        masker=shap.maskers.Text(tokenizer),
        output_names=cfg.get("class_names", ["fake", "real"])
    )

    return explainer

def explain_sample_shap(
    text: str,
    explainer,
    cfg: Dict[str, Any],
    sample_id: Any = None,
    true_label: int = None
):
    """
    Generate SHAP explanation for a single text sample
    Returns a dict formatted according to project standards

    Args:
        text: input string
        explainer: SHAP Transformer explainer
        cfg: config dictionary
        sample_id: numeric/string ID
        true_label: optional class label

    Returns:
        dict containing SHAP explanations in standard format
    """
    shap_cfg = cfg.get("shap", {})  

    #Compute SHAP values
    shap_values = explainer([text], max_evals=shap_cfg.get("max_evals", 5000))

    #Get predicted label
    pred_label = int(np.argmax(shap_values.values[0].mean(axis=0)))

    #Get tokens and importance weights
    tokens = shap_values.data[0]
    importances = shap_values.values[0][:, pred_label].tolist()

    result_dict = {
        "sample_id": sample_id if sample_id is not None else 0,
        "text": text,
        "tokens": tokens,
        "importances": importances,
        "pred_label": pred_label,
        "true_label": true_label
    }

    return result_dict
    
def run_shap_batch(dataset, model, tokenizer, cfg):
    """
    Run SHAP explanations over a subset of dataset

    Args:
        dataset: object with __getitem__ and storing { "text", "label" }
        model: DistilBERT model
        tokenizer: HuggingFace tokenizer
        cfg: config dictionary

    Returns:
        List of result dicts in project explanation format
    """
    shap_cfg = cfg.get("shap", {})  

    # Check for empty dataset
    if len(dataset) == 0:
      print("Warning: Empty dataset provided")
      return []
    
    num_samples = shap_cfg.get("num_explain_samples", min(50, len(dataset))) #Using less samples as SHAP is slower
    explainer = make_shap_explainer(model, tokenizer, cfg)
    results: List[Dict[str, Any]] = []

    print(f"Running SHAP explanations on {num_samples} samples...")

    #Run through sample batch
    for i in tqdm(range(num_samples), desc="SHAP Batch"):
        try:
          sample = dataset[i]
          text = sample["text"]
          true_label = int(sample.get("label", -1))

          explanation = explain_sample_shap(
              text=text,
              explainer=explainer,
              cfg=cfg,
              sample_id=i,
              true_label=true_label
          )
          results.append(explanation)

        except Exception as e:
          print(f"Error accessing dataset at index {i}: {e}")
          continue
    
    return results
