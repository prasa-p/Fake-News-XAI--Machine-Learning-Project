#!/usr/bin/env bash
# File: run_xai.sh
#
# Responsibilities:
# - Orchestrate running LIME, SHAP, and IG on a subset of the test data.
# - Store explanations and XAI metrics under artifacts/explanations and artifacts/metrics.
#
# Contributors:
# - Anton Nemchinski
# - <Name 2>
# - <Name 3>

#!/usr/bin/env bash
set -e

# Example usage:
#   ./scripts/run_xai.sh

PYTHONPATH=. python -m src.explain.run_xai_batch --dataset kaggle --methods ig lime shap
PYTHONPATH=. python -m src.explain.run_xai_batch --dataset liar   --methods ig lime shap

PYTHONPATH=. python -m src.explain.eval_xai


