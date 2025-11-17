"""
File: utils.py

Responsibilities:
- Provide shared utility functions for seeding, logging, paths, and config handling.
- Keep small helpers here instead of duplicating them across files.

Contributors:
- Anton Nemchinski
- <Name 2>
- <Name 3>

Key functions to implement:
- load_config(config_path: str) -> dict
- set_seed(seed: int) -> None
- get_run_id() -> str
- ensure_dir(path: str) -> None
"""

import os, yaml, random, numpy as np, torch
from datetime import datetime

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(cfg: dict) -> torch.device:
    if cfg["hardware"]["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
