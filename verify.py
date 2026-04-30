"""End-to-end component verification script.

Runs every pipeline stage on a small dataset (Titanic) and prints
concrete evidence that each component is wired up correctly:

    Stage 1 - preprocessing                  -> shape + dtypes
    Stage 2 - SMOTE / Counterfactual / GAN   -> row counts + fidelity
    Stage 3 - Logistic Regression / Random Forest
    Stage 4 - DPD / DIR / EOD + significance
    Stage 5 - SHAP top features + LIME attributions
    Stage 6 - audit JSON + Markdown summary

Usage:
    python verify.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.datasets import LOADERS
from pipeline.stage1_preprocess import fit_preprocess
from pipeline.stage2_synthetic import (fidelity_report,
                                       generate_counterfactual,
                                       generate_gan, generate_smote,
                                       generate_smote_gan)
from pipeline.stage3_models import train_eval
from pipeline.stage4_fairness import evaluate
from pipeline.stage5_diagnostics import lime_report, shap_report
from pipeline.stage6_reporting import build_record, save_audit


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main(dataset_name: str = "Titanic") -> int:
    t0 = time.perf_counter()

    section(f"VERIFICATION ON DATASET: {dataset_name}")
    raw = LOADERS[dataset_name]()
    print(f"raw shape           : {raw.df.shape}")
    print(f"label column        : {raw.label!r}")
    print(f"sensitive attrs     : {raw.sensitive}")
    print(f"label distribution  : {raw.df[raw.label].value_counts().to_dict()}")

    # ------------------------------------------------------------------ Stage 1
    section("STAGE 1 - Preprocessing & Validation")
    ds = fit_preprocess(raw, seed=42)
    print(f"X_train shape       : {ds.X_train.shape}")
    print(f"X_test shape        : {ds.X_test.shape}")
    print(f"feature names       : {ds.pre.feature_names}")
    print(f"numeric columns     : {ds.pre.numeric_cols}")
    print(f"categorical columns : {ds.pre.categorical_cols}")
    print(f"label encoders      : {list(ds.pre.label_encoders)}")
    print(f"NaN in X_train      : {np.isnan(ds.X_train).sum()}")
    print(f"NaN in X_test       : {np.isnan(ds.X_test).sum()}")
    print(f"sensitive frame head:\n{ds.s_test.head().to_string(index=False)}")

    # ------------------------------------------------------------------ Stage 2
    section("STAGE 2 - Synthetic Data Generation")
    smote_b = generate_smote(ds, seed=42)
    cf_b    = generate_counterfactual(ds, seed=42)
    gan_b   = generate_gan(ds, seed=42)
    sg_b    = generate_smote_gan(ds, seed=42)

    print(f"SMOTE         rows: {len(smote_b.X):>5}  (orig {len(ds.X_train)})")
    print(f"Counterfactual rows: {len(cf_b.X):>5}")
    print(f"GAN/CTGAN     rows: {len(gan_b.X):>5}")
    print(f"SMOTE+GAN     rows: {len(sg_b.X):>5}")

    fid = fidelity_report(ds, gan_b)
    print(f"GAN fidelity (KS, Chi2, n) : {fid}")

    # ------------------------------------------------------------------ Stage 3
    section("STAGE 3 - Model Training & Inference")
    base_lr = train_eval(ds, "lr", "baseline", seed=42)
    base_rf = train_eval(ds, "rf", "baseline", seed=42)
    full_rf = train_eval(ds, "rf", "smote_gan_cf",
                         batch=sg_b, seed=42)
    print(f"Logistic Regression baseline acc: {base_lr.accuracy*100:5.2f}%  F1={base_lr.f1:.3f}")
    print(f"Random Forest       baseline acc: {base_rf.accuracy*100:5.2f}%  F1={base_rf.f1:.3f}")
    print(f"Random Forest (SMOTE+GAN)   acc : {full_rf.accuracy*100:5.2f}%  F1={full_rf.f1:.3f}")

    # ------------------------------------------------------------------ Stage 4
    section("STAGE 4 - Fairness Assessment (DPD / DIR / EOD)")
    fr_before = evaluate(dataset_name, ds.y_test, base_rf.pred_test, ds.s_test)
    fr_after  = evaluate(dataset_name, ds.y_test, full_rf.pred_test, ds.s_test)
    print(f"BEFORE mitigation:")
    for r in fr_before:
        print(f"  sensitive={r.sensitive:>6}  DPD={r.dpd:.3f}  DIR={r.dir:.3f}  "
              f"EOD={r.eod:.3f}  p={r.p_value:.3f}  verdict={r.verdict}")
    print(f"AFTER  mitigation:")
    for r in fr_after:
        print(f"  sensitive={r.sensitive:>6}  DPD={r.dpd:.3f}  DIR={r.dir:.3f}  "
              f"EOD={r.eod:.3f}  p={r.p_value:.3f}  verdict={r.verdict}")

    # ------------------------------------------------------------------ Stage 5
    section("STAGE 5 - Explainability Diagnostics (SHAP / LIME)")
    sr = shap_report(ds, full_rf.model, seed=42)
    top10 = sorted(sr.feature_importance, key=lambda k: -sr.feature_importance[k])[:10]
    print("Top-10 SHAP features:")
    for f in top10:
        print(f"  {f:<22}  mean|SHAP|={sr.feature_importance[f]:.4f}")
    if sr.flagged:
        print(f"SHAP-flagged (>20% group disparity): {sr.flagged}")
    else:
        print("SHAP-flagged: (no feature exceeded the 20% disparity threshold)")

    lr = lime_report(ds, full_rf.model, seed=42, n_explain=10)
    top5_lime = sorted(lr.avg_attribution, key=lambda k: -lr.avg_attribution[k])[:5]
    print(f"LIME explained instances: {lr.n_explained}")
    print("Top-5 LIME features:")
    for f in top5_lime:
        print(f"  {f:<22}  avg|attribution|={lr.avg_attribution[f]:.4f}")

    # ------------------------------------------------------------------ Stage 6
    section("STAGE 6 - Decision Logic & Audit Reporting")
    record = build_record(dataset_name, full_rf.accuracy, fr_after, sr.flagged)
    save_audit([record], filename="verify_audit_report")
    audit_path = config.RESULTS_DIR / "verify_audit_report.json"
    md_path = config.RESULTS_DIR / "verify_audit_report.md"
    print(f"audit JSON written : {audit_path}")
    print(f"audit Markdown     : {md_path}")
    print(f"plain summary      : {record.plain_summary}")
    print(f"overall verdict    : {record.overall_verdict}")

    section("VERIFICATION COMPLETE")
    print(f"total verification time: {time.perf_counter()-t0:.2f}s")
    print(f"all six pipeline stages executed end-to-end with real data.")
    return 0


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Titanic"
    sys.exit(main(name))
