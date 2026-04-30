"""Pipeline configuration.

Only static configuration constants live here. Cached cross-seed
metrics (used to render figures and tables) are loaded by
:mod:`pipeline.evaluation` from ``results/evaluation_summary.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
for _p in (DATA_DIR, FIG_DIR, RESULTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datasets used in the paper
# ---------------------------------------------------------------------------
DATASETS: List[str] = [
    "UCI Adult",
    "COMPAS",
    "Diabetes 130-US",
    "MEPS",
    "Student Performance",
    "Titanic",
    "CivilComments",
]

# ---------------------------------------------------------------------------
# Pipeline hyper-parameters
# ---------------------------------------------------------------------------
RANDOM_SEEDS: List[int] = list(range(42, 52))     # 10-run stability sweep
TEST_SIZE: float = 0.20
THRESHOLD: float = 0.5                             # binarisation threshold

# Fairness decision rule (Section IV-C)
DPD_PASS: float = 0.10
DIR_LO, DIR_HI = 0.80, 1.25

# Synthetic-data hyper-parameters
SMOTE_K: int = 5
CTGAN_EPOCHS: int = 100   # used when sdv is available; falls back to copula otherwise
COUNTERFACTUAL_PER_GROUP: int = 1000

# Explainability hyper-parameters
SHAP_BG: int = 200        # KernelSHAP background sample
SHAP_EXP: int = 200
LIME_NUM: int = 200       # LIME perturbation samples
