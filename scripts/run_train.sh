#!/usr/bin/env bash
# File: run_train.sh
#
# Responsibilities:
# - Thin wrapper to launch model training with the desired config.
# - Keeps CLI usage simple for teammates and the instructor.
#
# Contributors:
# - Anton Nemchinski


python -m src.train --config config/default.yaml
