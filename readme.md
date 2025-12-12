# CP322 – Fake News Detection with DistilBERT & XAI

This project trains and analyzes a DistilBERT-based fake-news detector on two very different datasets: the Kaggle Fake News dataset (long, full-text news articles) and the LIAR dataset (short, headline-style political claims). We first establish TF-IDF baselines (LogReg, SVM, RF), then fine-tune DistilBERT separately on each dataset and compare performance, showing that the transformer architecture is especially strong on long-form news.

Beyond raw accuracy, the focus is on explainability. We compute token-level attributions with Integrated Gradients (IG), LIME, and SHAP, and analyze model attention maps and layer-wise probing to see where in the network fake vs. real signals emerge. Finally, we quantify explanation quality with faithfulness (probability drop when important tokens are removed) and stability (how consistent explanations are under small input perturbations), to evaluate not just how well the model predicts fake news, but how trustworthy and robust its explanations are across the two datasets.

---

## How to run the code (quick start)

1. **Clone the repo**

    git clone <repo-url>  
    cd CP322_FAKE_NEWS_XAI

2. **Create and activate a virtual environment (recommended)**

    python -m venv .venv  
    source .venv/bin/activate      # Linux/macOS  
    .venv\Scripts\activate         # Windows

3. **Install minimal dependencies**

    pip install -r requirements_minimal.txt

4. **Open the main notebook (and select appropriate kernel)**

   - Launch Jupyter / VS Code and open `project_overview.ipynb`.
   - Run the cells from top to bottom.

**Notes**

- Only **test splits** (`kaggle_test.csv`, `liar_test.csv`) and **final trained
  DistilBERT checkpoints** are included in the repo to keep the size manageable.
- The notebook **does not retrain** the model; it loads the saved checkpoints,
  runs evaluation on the provided test sets, and visualizes pre-computed XAI and
  probing results.

---

## Directory overview

### Top level

- `project_overview.ipynb` – Main entry point: loads trained models, runs test predictions, and summarizes results / visualizations.
- `readme.md` – Project description and usage.
- `requirements.txt` – Minimal Python dependencies for the overview notebook.
- `.gitignore`, `.gitattributes` – Git configuration.
- `data_download_instructions.md` – How full training data were originally obtained (not required for running this submission).

### Data & artifacts

- `data/`
  - `processed/kaggle_test.csv` – Preprocessed Kaggle test set (long articles).
  - `processed/liar_test.csv` – Preprocessed LIAR test set (short claims).
  - `raw/figures/eda/*.png` – Saved EDA plots (length distributions, word clouds, top markers, etc.).
  - `raw/figures/attention_vis/*.png` – Saved CLS→token attention plots for selected examples.

- `artifacts/`
  - `distilbert/`
    - `kaggle/final_model/` – Final DistilBERT checkpoint for Kaggle (HF config, tokenizer, weights).
    - `liar/final_model/` – Final DistilBERT checkpoint for LIAR.
  - `explanations/`
    - `kaggle_ig.jsonl`, `kaggle_lime.jsonl`, `kaggle_shap.jsonl` – Token-level explanations on Kaggle test set.
    - `kaggle_*_perturbed.jsonl` – Explanations on perturbed Kaggle texts (for stability).
    - `liar_ig.jsonl`, `liar_lime.jsonl`, `liar_shap.jsonl` – Explanations on LIAR test set.
    - `liar_*_perturbed.jsonl` – Explanations on perturbed LIAR texts (for stability).
  - `layers/`
    - `kaggle_layer_probes.json` – Layer-wise probe results for Kaggle (accuracy/F1 per layer).
    - `liar_layer_probes.json` – Layer-wise probe results for LIAR.
  - `metrics/`
    - `baseline_results.csv` – Scores for non-transformer baselines.
    - `distilbert_kaggle_fake_news_results.json` – Final DistilBERT metrics on Kaggle.
    - `distilbert_liar_results.json` – Final DistilBERT metrics on LIAR.
    - `xai_metrics.json` – Faithfulness and stability metrics for all XAI methods / datasets.

- `config/`
  - `default.yaml` – Shared hyperparameters and paths.
  - `kaggle.yaml` – Training / data config for Kaggle.
  - `liar.yaml` – Training / data config for LIAR.
  - `paths.yaml` – Centralized filesystem paths used by scripts.

### Source code

- `src/`
  - `data.py` – Dataset loaders and preprocessing utilities for Kaggle and LIAR.
  - `models.py` – DistilBERT classifier definition and model helpers.
  - `train.py` – End-to-end training loop for DistilBERT (used offline to produce final checkpoints).
  - `utils.py` – Shared helpers (logging, seeding, config handling, etc.).

  - `layers/`
    - `probe.py` – Implementation of linear probing on [CLS] representations per layer.
    - `run_layer_probes.py` – Script to train and evaluate layer probes and save results.
    - `attention_viz.py` – Functions to extract and plot CLS→token attention weights.

  - `explain/`
    - `ig_utils.py` – Integrated Gradients implementation for token-level attributions.
    - `lime_utils.py` – LIME wrapper for text explanations.
    - `shap_utils.py` – SHAP wrapper for text explanations.
    - `run_xai_batch.py` – Batch generation of IG/LIME/SHAP explanations over test sets.
    - `generate_perturbed_explanations.py` – Produces explanations on perturbed texts for stability analysis.
    - `eval_xai.py` – Computes XAI metrics (faithfulness via deletion, stability via Jaccard) from saved explanations.

### Notebooks & scripts

- `notebooks/`
  - `00_eda.ipynb` – Original exploratory data analysis; plots are saved under `data/raw/figures/eda`.
  - `01_live_demo.ipynb` – Live demo notebook used during presentations (interactive predictions + explanations).

- `scripts/`
  - `run_train.sh` – Helper to launch full training jobs (not needed for grading).
  - `run_layers.sh` – Runs the layer probing pipeline end-to-end.
  - `run_xai.sh` – Runs XAI explanation generation end-to-end.
  - `test_model_loading.py` – Sanity check that DistilBERT checkpoints can be loaded and used for inference.
  - `view_baselines.py` – Loads and prints baseline metrics from `artifacts/metrics`.

---
