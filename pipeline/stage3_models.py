"""Stage 3 - Model training and inference (Algorithm 3).

Trains scikit-learn classifiers on the (optionally augmented) training
set and produces probabilities / predictions on the **original**
held-out test set.  Synthetic data is therefore used purely for training
augmentation, never for test evaluation -- this is the leakage-control
design described in Section IV-B of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from . import config
from .stage1_preprocess import ProcessedDataset
from .stage2_synthetic import SyntheticBatch
from .utils import log


@dataclass
class TrainResult:
    name: str            # model id (e.g. "rf"/"lr")
    augmentation: str    # "baseline" | "smote" | "counterfactual" | "gan" | ...
    model: object
    proba_test: np.ndarray
    pred_test: np.ndarray
    accuracy: float
    f1: float
    precision: float
    recall: float


def make_model(kind: str, seed: int = 42):
    if kind == "lr":
        return LogisticRegression(max_iter=200, C=1.0, solver="lbfgs",
                                  n_jobs=None, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=8,
                                      n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown model kind {kind}")


def train_eval(ds: ProcessedDataset,
               kind: str = "rf",
               augmentation: str = "baseline",
               batch: Optional[SyntheticBatch] = None,
               seed: int = 42) -> TrainResult:
    if batch is None:
        X_tr, y_tr = ds.X_train, ds.y_train
    else:
        X_tr, y_tr = batch.X, batch.y
    model = make_model(kind, seed=seed)
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(ds.X_test)[:, 1]
    else:
        proba = model.decision_function(ds.X_test)
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
    pred = (proba >= config.THRESHOLD).astype(int)
    return TrainResult(
        name=kind, augmentation=augmentation, model=model,
        proba_test=proba, pred_test=pred,
        accuracy=accuracy_score(ds.y_test, pred),
        f1=f1_score(ds.y_test, pred, zero_division=0),
        precision=precision_score(ds.y_test, pred, zero_division=0),
        recall=recall_score(ds.y_test, pred, zero_division=0),
    )


def train_all_augmentations(ds: ProcessedDataset, kind: str = "rf",
                            seed: int = 42) -> Dict[str, TrainResult]:
    """Run baseline + every augmentation defined in stage2."""
    from .stage2_synthetic import (generate_smote, generate_counterfactual,
                                   generate_gan, generate_smote_gan)
    out: Dict[str, TrainResult] = {}
    out["baseline"] = train_eval(ds, kind=kind, augmentation="baseline", seed=seed)
    out["smote"] = train_eval(ds, kind=kind, augmentation="smote",
                              batch=generate_smote(ds, seed=seed), seed=seed)
    out["counterfactual"] = train_eval(ds, kind=kind, augmentation="counterfactual",
                                       batch=generate_counterfactual(ds, seed=seed),
                                       seed=seed)
    out["gan"] = train_eval(ds, kind=kind, augmentation="gan",
                            batch=generate_gan(ds, seed=seed), seed=seed)
    out["smote_gan"] = train_eval(ds, kind=kind, augmentation="smote_gan",
                                  batch=generate_smote_gan(ds, seed=seed), seed=seed)
    log("stage3", f"{ds.name:<20} "
        + " ".join(f"{k}={v.accuracy*100:5.2f}" for k, v in out.items()))
    return out
