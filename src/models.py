"""
File: models.py

Responsibilities:
- Define and construct baseline models (e.g., TF-IDF + Logistic Regression).
- Load and configure DistilBERT-based sequence classification model from Hugging Face.
- Provide utility functions to initialize tokenizers and models in a consistent way.

Contributors:
- Anton Nemchinski
- <Name 2>
- <Name 3>

Key functions to implement:
- build_baseline_model(cfg) -> sklearn model
- get_tokenizer(cfg) -> PreTrainedTokenizerFast
- get_distilbert_model(cfg) -> PreTrainedModel
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg["model"]["name"])

def get_distilbert_model(cfg):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"]
    )

