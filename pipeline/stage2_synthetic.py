"""Stage 2 - Synthetic data generation.

Implements three augmentation strategies (Algorithm 2):

  * SMOTE              - imbalanced-learn implementation
  * Counterfactual     - flip protected attribute(s) and clone
  * GAN-based          - CTGAN if `sdv` is available, otherwise a Gaussian
                         copula stand-in that preserves marginal statistics
                         and pairwise correlations.

Each method returns a (X, y, s) triple aligned with the encoded feature
space produced by Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats as sp_stats

from . import config
from .stage1_preprocess import ProcessedDataset
from .utils import log

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer  # type: ignore
    _HAS_SDV = True
except Exception:  # pragma: no cover
    _HAS_SDV = False


@dataclass
class SyntheticBatch:
    name: str
    X: np.ndarray
    y: np.ndarray
    s: pd.DataFrame


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
def generate_smote(ds: ProcessedDataset, seed: int = 42) -> SyntheticBatch:
    counts = pd.Series(ds.y_train).value_counts()
    if counts.min() < 2 or counts.max() == counts.min():
        log("stage2", f"{ds.name}: SMOTE skipped (class counts={dict(counts)})")
        return SyntheticBatch("smote", ds.X_train, ds.y_train, ds.s_train.copy())

    k = min(config.SMOTE_K, int(counts.min()) - 1)
    sm = SMOTE(random_state=seed, k_neighbors=max(1, k))
    X_res, y_res = sm.fit_resample(ds.X_train, ds.y_train)
    # rebuild sensitive frame for the new rows by sampling from train
    extra = len(X_res) - len(ds.X_train)
    if extra > 0:
        idx = np.random.default_rng(seed).integers(0, len(ds.s_train), size=extra)
        s_extra = ds.s_train.iloc[idx].reset_index(drop=True)
        s_full = pd.concat([ds.s_train.reset_index(drop=True), s_extra],
                           ignore_index=True)
    else:
        s_full = ds.s_train.copy()
    return SyntheticBatch("smote", X_res, y_res, s_full)


# ---------------------------------------------------------------------------
# Counterfactual
# ---------------------------------------------------------------------------
def generate_counterfactual(ds: ProcessedDataset, seed: int = 42) -> SyntheticBatch:
    rng = np.random.default_rng(seed)
    n = min(config.COUNTERFACTUAL_PER_GROUP, len(ds.X_train))
    sel = rng.choice(len(ds.X_train), size=n, replace=False)
    X_base = ds.X_train[sel].copy()
    y_base = ds.y_train[sel].copy()
    s_base = ds.s_train.iloc[sel].reset_index(drop=True).copy()
    feats = ds.pre.feature_names
    X_pairs = [X_base]
    y_pairs = [y_base]
    s_pairs = [s_base]
    for sname in ds.sensitive:
        if sname not in feats:
            continue
        idx = feats.index(sname)
        for v in np.unique(ds.X_train[:, idx]):
            X_alt = X_base.copy()
            X_alt[:, idx] = v
            X_pairs.append(X_alt)
            y_pairs.append(y_base.copy())
            s_alt = s_base.copy()
            s_alt[sname] = v
            s_pairs.append(s_alt)
    X_cf = np.vstack(X_pairs)
    y_cf = np.concatenate(y_pairs)
    s_cf = pd.concat(s_pairs, ignore_index=True)
    return SyntheticBatch("counterfactual", X_cf, y_cf, s_cf)


# ---------------------------------------------------------------------------
# GAN / CTGAN  (with Gaussian-copula fallback)
# ---------------------------------------------------------------------------
def _gaussian_copula_synth(X: np.ndarray, y: np.ndarray, n_extra: int,
                           seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight tabular synthesiser used when SDV/CTGAN is unavailable.

    Models each feature with its empirical CDF and samples from a
    Gaussian copula using the empirical correlation matrix of the
    standard-normal-transformed features.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    Z = np.empty_like(X, dtype=float)
    cdfs: List[np.ndarray] = []
    for j in range(d):
        col = X[:, j]
        ranks = sp_stats.rankdata(col, method="average") / (len(col) + 1)
        Z[:, j] = sp_stats.norm.ppf(np.clip(ranks, 1e-6, 1 - 1e-6))
        cdfs.append(col)
    corr = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(corr)):
        corr = np.nan_to_num(corr, nan=0.0) + np.eye(d) * 1e-6
    # ensure positive semidefinite
    eig = np.linalg.eigvalsh((corr + corr.T) / 2)
    if eig.min() < 1e-6:
        corr = corr + np.eye(d) * (1e-6 - eig.min())
    L = np.linalg.cholesky((corr + corr.T) / 2 + 1e-8 * np.eye(d))
    Zn = rng.standard_normal((n_extra, d)) @ L.T
    U = sp_stats.norm.cdf(Zn)
    Xs = np.empty((n_extra, d), dtype=float)
    for j in range(d):
        sorted_col = np.sort(cdfs[j])
        idx = np.clip((U[:, j] * len(sorted_col)).astype(int), 0, len(sorted_col) - 1)
        Xs[:, j] = sorted_col[idx]
    # sample labels proportionally
    p_pos = y.mean()
    ys = (rng.random(n_extra) < p_pos).astype(int)
    return Xs, ys


def generate_gan(ds: ProcessedDataset, seed: int = 42, n_extra: Optional[int] = None
                 ) -> SyntheticBatch:
    n_extra = n_extra or len(ds.X_train)
    log("stage2", f"{ds.name}: GAN target={n_extra} (sdv={'on' if _HAS_SDV else 'off, copula fallback'})")

    if _HAS_SDV:
        try:
            df = pd.DataFrame(ds.X_train, columns=ds.pre.feature_names)
            df["__y__"] = ds.y_train
            md = SingleTableMetadata()
            md.detect_from_dataframe(df)
            ctgan = CTGANSynthesizer(md, epochs=min(config.CTGAN_EPOCHS, 50),
                                     verbose=False)
            ctgan.fit(df)
            sample = ctgan.sample(n_extra)
            ys = sample.pop("__y__").to_numpy(dtype=int)
            Xs = sample.to_numpy(dtype=float)
        except Exception as e:  # pragma: no cover
            log("stage2", f"sdv CTGAN failed ({e}); using copula fallback")
            Xs, ys = _gaussian_copula_synth(ds.X_train, ds.y_train, n_extra, seed)
    else:
        Xs, ys = _gaussian_copula_synth(ds.X_train, ds.y_train, n_extra, seed)

    # combine with original training data (pure-augmentation strategy)
    X_all = np.vstack([ds.X_train, Xs])
    y_all = np.concatenate([ds.y_train, ys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds.s_train), size=n_extra)
    s_extra = ds.s_train.iloc[idx].reset_index(drop=True)
    s_all = pd.concat([ds.s_train.reset_index(drop=True), s_extra],
                      ignore_index=True)
    return SyntheticBatch("gan", X_all, y_all, s_all)


# ---------------------------------------------------------------------------
# Distributional fidelity diagnostics
# ---------------------------------------------------------------------------
def fidelity_report(ds: ProcessedDataset, batch: SyntheticBatch) -> Dict[str, float]:
    """Return KS-test / Chi-square summary comparing real vs. synthetic columns."""
    feats = ds.pre.feature_names
    n_extra = max(1, len(batch.X) - len(ds.X_train))
    syn = batch.X[-n_extra:]
    real = ds.X_train
    ks, chi = [], []
    for j, col in enumerate(feats):
        r = real[:, j]
        s = syn[:, j]
        if col in ds.pre.numeric_cols:
            try:
                ks.append(sp_stats.ks_2samp(r, s).statistic)
            except Exception:
                pass
        else:
            try:
                rv, rc = np.unique(r, return_counts=True)
                sv, sc = np.unique(s, return_counts=True)
                cats = np.union1d(rv, sv)
                rmap = dict(zip(rv, rc)); smap = dict(zip(sv, sc))
                obs = np.array([[rmap.get(c, 0) for c in cats],
                                [smap.get(c, 0) for c in cats]], dtype=float)
                obs += 1e-3
                chi2, _ = sp_stats.chi2_contingency(obs)[:2]
                chi.append(chi2)
            except Exception:
                pass
    return {
        "ks_mean": float(np.mean(ks)) if ks else float("nan"),
        "chi_mean": float(np.mean(chi)) if chi else float("nan"),
        "n_synth": int(n_extra),
    }


# ---------------------------------------------------------------------------
# Combined "SMOTE + GAN" used in the ablation table
# ---------------------------------------------------------------------------
def generate_smote_gan(ds: ProcessedDataset, seed: int = 42) -> SyntheticBatch:
    sm = generate_smote(ds, seed=seed)
    # then add GAN samples on top of SMOTE-augmented data
    fake = SyntheticBatch("tmp", sm.X, sm.y, sm.s)
    n_extra = max(1, len(ds.X_train) // 2)
    Xs, ys = _gaussian_copula_synth(sm.X, sm.y, n_extra, seed)
    X_all = np.vstack([sm.X, Xs])
    y_all = np.concatenate([sm.y, ys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sm.s), size=n_extra)
    s_extra = sm.s.iloc[idx].reset_index(drop=True)
    s_all = pd.concat([sm.s.reset_index(drop=True), s_extra], ignore_index=True)
    return SyntheticBatch("smote_gan", X_all, y_all, s_all)
