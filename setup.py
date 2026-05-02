
import os
import sys

import numpy as np
from omegaconf import DictConfig
import hydra
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_PATH = "/kaggle/working/cdvae"
DATA_PATH    = os.path.join(PROJECT_PATH, "src")
os.makedirs(DATA_PATH, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

print("Environment ready.")