"""Stage 4 - Fairness assessment (Algorithm 4).

Computes Demographic Parity Difference (DPD), Disparate Impact Ratio
(DIR) and Equalized Odds Difference (EOD) per dataset and per
sensitive attribute, with a two-proportion z-test for statistical
significance.  The "any-fail" decision rule from Section IV-C of the
paper is also implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from fairlearn.metrics import (demographic_parity_difference,
                                   equalized_odds_difference)
    _HAS_FAIRLEARN = True
except Exception:  # pragma: no cover
    _HAS_FAIRLEARN = False

from . import config


@dataclass
class FairnessResult:
    dataset: str
    sensitive: str
    dpd: float
    dir: float
    eod: float
    p_value: float
    verdict: str        # "PASS" | "FAIL"
    bias_score: float   # composite (used for "bias-score reduction" %)


def _privileged_unprivileged(s: np.ndarray, y_pred: np.ndarray):
    """Return (priv_mask, unpriv_mask) using the group with higher
    favorable-outcome rate as 'privileged'."""
    groups = np.unique(s)
    rates = {g: y_pred[s == g].mean() for g in groups}
    if len(rates) < 2:
        return None, None
    priv = max(rates, key=rates.get)
    unpriv = min(rates, key=rates.get)
    return s == priv, s == unpriv


def dpd_dir_eod(y_true: np.ndarray, y_pred: np.ndarray,
                sensitive: np.ndarray) -> Dict[str, float]:
    sensitive = np.asarray(sensitive)
    priv, unpriv = _privileged_unprivileged(sensitive, y_pred)
    if priv is None:
        return {"dpd": 0.0, "dir": 1.0, "eod": 0.0, "p": 1.0}

    p_priv = y_pred[priv].mean() if priv.any() else 0.0
    p_unpriv = y_pred[unpriv].mean() if unpriv.any() else 0.0
    dpd = abs(p_priv - p_unpriv)
    dir_val = (p_unpriv / p_priv) if p_priv > 0 else 1.0

    # equalised odds via TPR/FPR
    def _rates(mask):
        yt = y_true[mask]; yp = y_pred[mask]
        tpr = (yp[yt == 1].mean() if (yt == 1).any() else 0.0)
        fpr = (yp[yt == 0].mean() if (yt == 0).any() else 0.0)
        return tpr, fpr
    tpr_p, fpr_p = _rates(priv)
    tpr_u, fpr_u = _rates(unpriv)
    eod = max(abs(tpr_p - tpr_u), abs(fpr_p - fpr_u))

    # two-proportion z-test on demographic parity
    n_p = int(priv.sum()); n_u = int(unpriv.sum())
    succ_p = int(y_pred[priv].sum()); succ_u = int(y_pred[unpriv].sum())
    p_pool = (succ_p + succ_u) / max(1, (n_p + n_u))
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / max(1, n_p) + 1.0 / max(1, n_u)))
    z = 0.0 if se == 0 else (p_priv - p_unpriv) / se
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return {"dpd": float(dpd), "dir": float(dir_val), "eod": float(eod),
            "p": float(p_value)}


def _verdict(dpd: float, dir_val: float, eod: float, p: float) -> str:
    if dir_val < config.DIR_LO or dir_val > config.DIR_HI:
        return "FAIL"
    if dpd > config.DPD_PASS:
        return "FAIL"
    if eod > 0.10 and p < 0.05:
        return "FAIL"
    return "PASS"


def evaluate(dataset_name: str, y_true, y_pred, s_frame: pd.DataFrame
             ) -> List[FairnessResult]:
    out: List[FairnessResult] = []
    for col in s_frame.columns:
        m = dpd_dir_eod(np.asarray(y_true), np.asarray(y_pred),
                        np.asarray(s_frame[col]))
        bias_score = (m["dpd"] + max(0, 1 - m["dir"]) + m["eod"]) / 3.0
        out.append(FairnessResult(
            dataset=dataset_name, sensitive=col,
            dpd=m["dpd"], dir=m["dir"], eod=m["eod"],
            p_value=m["p"], bias_score=bias_score,
            verdict=_verdict(m["dpd"], m["dir"], m["eod"], m["p"]),
        ))
    return out


def aggregate(results: List[FairnessResult]) -> Dict[str, float]:
    return {
        "dpd_avg": float(np.mean([r.dpd for r in results])),
        "dir_avg": float(np.mean([r.dir for r in results])),
        "eod_avg": float(np.mean([r.eod for r in results])),
        "bias_score": float(np.mean([r.bias_score for r in results])),
        "fail_count": int(sum(1 for r in results if r.verdict == "FAIL")),
    }
