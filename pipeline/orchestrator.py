"""End-to-end pipeline orchestrator.

Iterates Stages 1-6 across the seven benchmark datasets, collects
per-dataset measurements, and writes raw per-run measurements together
with publication-ready figures and tables.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, figures, tables
from .datasets import LOADERS, RawDataset
from .stage1_preprocess import ProcessedDataset, fit_preprocess
from .stage2_synthetic import (fidelity_report, generate_counterfactual,
                               generate_gan, generate_smote,
                               generate_smote_gan)
from .stage3_models import train_all_augmentations, train_eval
from .stage4_fairness import aggregate, evaluate
from .stage5_diagnostics import (LimeReport, ShapReport, consistency,
                                 lime_report, shap_report)
from .stage6_reporting import build_record, save_audit
from .utils import log, timer


@dataclass
class DatasetResult:
    name: str
    n_train: int
    n_test: int
    feature_count: int
    sensitive: List[str]
    accuracy_baseline: float
    accuracy_smote: float
    accuracy_cf: float
    accuracy_gan: float
    accuracy_full: float
    fairness_before: Dict[str, float]
    fairness_after: Dict[str, float]
    shap_topk: List[str]
    shap_flagged: List[str]
    fidelity: Dict[str, Any]
    runtime_sec: float


def run_one(raw: RawDataset, model_kind: str = "rf",
            seed: int = 42, run_xai: bool = True) -> DatasetResult:
    t0 = time.perf_counter()
    with timer("stage1", f"preprocess {raw.name}"):
        ds = fit_preprocess(raw, seed=seed)

    with timer("stage2", f"synthetic data for {raw.name}"):
        sm_batch = generate_smote(ds, seed=seed)
        cf_batch = generate_counterfactual(ds, seed=seed)
        gan_batch = generate_gan(ds, seed=seed)
        smote_gan = generate_smote_gan(ds, seed=seed)
        # full = SMOTE + Counterfactual + GAN (concatenated)
        X_full = np.vstack([smote_gan.X, cf_batch.X])
        y_full = np.concatenate([smote_gan.y, cf_batch.y])
        s_full = pd.concat([smote_gan.s, cf_batch.s], ignore_index=True)
        from .stage2_synthetic import SyntheticBatch as _SB
        full_batch = _SB("full", X_full, y_full, s_full)
        fid = fidelity_report(ds, gan_batch)

    with timer("stage3", f"train models for {raw.name}"):
        base = train_eval(ds, model_kind, "baseline", seed=seed)
        sm = train_eval(ds, model_kind, "smote", batch=sm_batch, seed=seed)
        cf = train_eval(ds, model_kind, "counterfactual", batch=cf_batch, seed=seed)
        gan = train_eval(ds, model_kind, "gan", batch=gan_batch, seed=seed)
        full = train_eval(ds, model_kind, "full", batch=full_batch, seed=seed)

    with timer("stage4", f"fairness for {raw.name}"):
        fr_before = evaluate(raw.name, ds.y_test, base.pred_test, ds.s_test)
        fr_after = evaluate(raw.name, ds.y_test, full.pred_test, ds.s_test)

    shap_top: List[str] = []
    shap_flagged: List[str] = []
    if run_xai:
        with timer("stage5", f"SHAP/LIME for {raw.name}"):
            try:
                sr = shap_report(ds, full.model, seed=seed)
                shap_top = sorted(sr.feature_importance,
                                  key=lambda k: -sr.feature_importance[k])[:5]
                shap_flagged = sr.flagged
                _ = lime_report(ds, full.model, seed=seed, n_explain=15)
            except Exception as e:  # pragma: no cover
                log("stage5", f"WARN xai failed for {raw.name}: {e}")

    elapsed = time.perf_counter() - t0
    return DatasetResult(
        name=raw.name, n_train=len(ds.y_train), n_test=len(ds.y_test),
        feature_count=len(ds.pre.feature_names),
        sensitive=list(ds.sensitive),
        accuracy_baseline=base.accuracy,
        accuracy_smote=sm.accuracy, accuracy_cf=cf.accuracy,
        accuracy_gan=gan.accuracy, accuracy_full=full.accuracy,
        fairness_before=aggregate(fr_before),
        fairness_after=aggregate(fr_after),
        shap_topk=shap_top, shap_flagged=shap_flagged,
        fidelity=fid, runtime_sec=elapsed,
    )


def run_pipeline(model_kind: str = "rf",
                 datasets: Optional[List[str]] = None,
                 seed: int = 42,
                 run_xai: bool = True) -> List[DatasetResult]:
    chosen = datasets or config.DATASETS
    out: List[DatasetResult] = []
    for name in chosen:
        if name not in LOADERS:
            log("orchestr", f"skip unknown dataset {name}")
            continue
        try:
            raw = LOADERS[name]()
        except Exception as e:
            log("orchestr", f"WARN cannot load {name}: {e}")
            continue
        out.append(run_one(raw, model_kind=model_kind, seed=seed,
                           run_xai=run_xai))
    return out


def save_raw(results: List[DatasetResult]) -> Path:
    rows = []
    for r in results:
        d = asdict(r)
        d["fairness_before"] = json.dumps(d["fairness_before"])
        d["fairness_after"] = json.dumps(d["fairness_after"])
        d["shap_topk"] = ", ".join(d["shap_topk"])
        d["shap_flagged"] = ", ".join(d["shap_flagged"])
        d["fidelity"] = json.dumps(d["fidelity"])
        d["sensitive"] = ", ".join(d["sensitive"])
        rows.append(d)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "raw_pipeline_measurements.csv"
    df.to_csv(out, index=False)
    log("orchestr", f"raw measurements saved to {out}")
    return out


def save_audit_records(results: List[DatasetResult]) -> Path:
    audit = []
    for r in results:
        # build minimal FairnessResult-like dicts
        fr_after_obj = []
        before = r.fairness_before
        after = r.fairness_after
        # construct synthetic AuditRecord using only summarised numbers
        audit.append({
            "dataset": r.name,
            "accuracy_baseline": round(r.accuracy_baseline * 100, 2),
            "accuracy_full": round(r.accuracy_full * 100, 2),
            "dpd_before": round(before["dpd_avg"], 3),
            "dpd_after": round(after["dpd_avg"], 3),
            "dir_before": round(before["dir_avg"], 3),
            "dir_after": round(after["dir_avg"], 3),
            "eod_before": round(before["eod_avg"], 3),
            "eod_after": round(after["eod_avg"], 3),
            "shap_flagged": r.shap_flagged,
            "verdict_before": "FAIL" if before["fail_count"] > 0 else "PASS",
            "verdict_after": "FAIL" if after["fail_count"] > 0 else "PASS",
            "runtime_sec": round(r.runtime_sec, 2),
        })
    out = config.RESULTS_DIR / "audit_summary.json"
    out.write_text(json.dumps(audit, indent=2))
    log("orchestr", f"audit summary saved to {out}")
    return out
