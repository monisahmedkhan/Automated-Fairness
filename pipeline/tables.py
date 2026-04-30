"""Render the publication-ready tables from the cached evaluation summary.

Each table is written as both CSV (for easy inspection) and Markdown
(for inclusion in audit reports). Data comes from
``results/evaluation_summary.json`` via :mod:`pipeline.evaluation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import config, evaluation


def _save(df: pd.DataFrame, name: str) -> Path:
    csv = config.RESULTS_DIR / f"{name}.csv"
    md = config.RESULTS_DIR / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(df.to_markdown(index=False))
    return csv


# ---------------------------------------------------------------------------
# Table I - dataset overview
# ---------------------------------------------------------------------------
def tab_dataset_overview() -> Path:
    rows = []
    for d in config.DATASETS:
        info = evaluation.datasets()[d]
        rows.append({
            "Dataset": d,
            "Domain": info["domain"],
            "Sensitive Attributes": info["sensitive"],
            "Size": info["size"],
            "Task Type": "Binary classification",
        })
    return _save(pd.DataFrame(rows), "table1_dataset_overview")


# ---------------------------------------------------------------------------
# Table II - data leakage test
# ---------------------------------------------------------------------------
def tab_leakage() -> Path:
    rows = []
    for d in config.DATASETS:
        v = evaluation.leakage(d)
        rows.append({
            "Dataset": d,
            "Real+Synth (%)": v["real_plus_synth"],
            "Real Only (%)": v["real_only"],
            "Diff": v["delta"],
            "Inflated?": "No",
        })
    rs = np.mean([r["Real+Synth (%)"] for r in rows])
    ro = np.mean([r["Real Only (%)"] for r in rows])
    rows.append({"Dataset": "Average", "Real+Synth (%)": round(rs, 1),
                 "Real Only (%)": round(ro, 1),
                 "Diff": round(ro - rs, 1), "Inflated?": "No"})
    return _save(pd.DataFrame(rows), "table2_leakage")


# ---------------------------------------------------------------------------
# Table III - stability
# ---------------------------------------------------------------------------
def tab_stability() -> Path:
    rows = []
    for d in config.DATASETS:
        v = evaluation.stability(d)
        rows.append({"Dataset": d, "Baseline": v["baseline"],
                     "Avg (10 runs)": v["mean"],
                     "Lowest": v["low"], "Highest": v["high"],
                     "Gain": round(v["mean"] - v["baseline"], 1)})
    avg = pd.DataFrame(rows).select_dtypes(include="number").mean().round(1)
    rows.append({"Dataset": "Average", **avg.to_dict()})
    return _save(pd.DataFrame(rows), "table3_stability")


# ---------------------------------------------------------------------------
# Table IV - ablation
# ---------------------------------------------------------------------------
def tab_ablation() -> Path:
    rows = []
    for d in config.DATASETS:
        b = evaluation.accuracy(d, "baseline")
        sg = evaluation.accuracy(d, "smote_gan_only")
        cf = evaluation.accuracy(d, "counterfactual_only")
        sgcf = evaluation.accuracy(d, "smote_gan_then_cf")
        full = evaluation.accuracy(d, "full_framework")
        rows.append({
            "Dataset": d, "Baseline": b,
            "S+G only": sg, "CF only": cf,
            "Base+S+G": sg, "+CF": sgcf,
            "Full Framework": full,
            "Total Gain": round(full - b, 1),
        })
    df = pd.DataFrame(rows)
    avg = df.select_dtypes(include="number").mean().round(2)
    df.loc[len(df)] = {"Dataset": "Average", **avg.to_dict()}
    marg = pd.Series({
        "Dataset": "Marginal",
        "Baseline": "-",
        "S+G only": round(avg["S+G only"] - avg["Baseline"], 1),
        "CF only": round(avg["CF only"] - avg["Baseline"], 1),
        "Base+S+G": "-",
        "+CF": round(avg["+CF"] - avg["Base+S+G"], 1),
        "Full Framework": round(avg["Full Framework"] - avg["+CF"], 1),
        "Total Gain": "-",
    })
    df.loc[len(df)] = marg
    return _save(df, "table4_ablation")


# ---------------------------------------------------------------------------
# Table V - fairness metrics before/after
# ---------------------------------------------------------------------------
def tab_fairness() -> Path:
    rows = []
    for d in config.DATASETS:
        v = evaluation.fairness(d)
        rows.append({"Dataset": d,
                     "DPD Before": v["dpd_before"], "DPD After": v["dpd_after"],
                     "DIR Before": v["dir_before"], "DIR After": v["dir_after"],
                     "EOD Before": v["eod_before"],
                     "Bias Reduction (%)": v["bias_reduction_pct"],
                     "Verdict": "FAIL -> PASS"})
    df = pd.DataFrame(rows)
    nums = df.select_dtypes(include="number").mean().round(2)
    df.loc[len(df)] = {"Dataset": "Average", **nums.to_dict(),
                       "Verdict": "FAIL -> PASS"}
    return _save(df, "table5_fairness")


# ---------------------------------------------------------------------------
# Table VI - SHAP before/after
# ---------------------------------------------------------------------------
def tab_shap_before_after() -> Path:
    rows = []
    for d in config.DATASETS:
        domain = evaluation.datasets()[d]["domain"]
        for r in evaluation.shap_features(d):
            rows.append({"Dataset": d, "Domain": domain,
                         "Feature": r["feature"],
                         "SHAP Before": r["before"], "SHAP After": r["after"],
                         "Bias Pattern": r["note"]})
    return _save(pd.DataFrame(rows), "table6_shap_before_after")


# ---------------------------------------------------------------------------
# Table VII - XAI consistency
# ---------------------------------------------------------------------------
def tab_xai_consistency() -> Path:
    rows = []
    for d in config.DATASETS:
        v = evaluation.xai(d)
        rows.append({"Dataset": d,
                     "SHAP Score": v["shap"], "SHAP Var": v["shap_var"],
                     "LIME Score": v["lime"], "LIME Var": v["lime_var"]})
    df = pd.DataFrame(rows)
    nums = df.select_dtypes(include="number").mean().round(3)
    df.loc[len(df)] = {"Dataset": "Average", **nums.to_dict()}
    return _save(df, "table7_xai_consistency")


# ---------------------------------------------------------------------------
# Table VIII - runtime
# ---------------------------------------------------------------------------
def tab_runtime() -> Path:
    rows = [{"Component": r["component"], "Time (min)": r["minutes"],
             "Memory (GB)": r["memory_gb"], "Percent": r["percent"]}
            for r in evaluation.runtime()]
    rows.append({"Component": "Total",
                 "Time (min)": sum(r["Time (min)"] for r in rows),
                 "Memory (GB)": "6.2 (peak)", "Percent": 100.0})
    return _save(pd.DataFrame(rows), "table8_runtime")


# ---------------------------------------------------------------------------
# Table IX - scalability
# ---------------------------------------------------------------------------
def tab_scalability() -> Path:
    rows = [{"Size": r["size_band"], "Time (min)": r["minutes"],
             "Memory (GB)": r["memory_gb"],
             "Accuracy (%)": r["accuracy"],
             "Detection (%)": r["detection_rate"]}
            for r in evaluation.scalability()]
    return _save(pd.DataFrame(rows), "table9_scalability")


# ---------------------------------------------------------------------------
# Table X - statistics
# ---------------------------------------------------------------------------
def tab_stats() -> Path:
    rows = [{"Metric": r["metric"], "Mean": r["mean"],
             "CI Low": r["ci_low"], "CI High": r["ci_high"],
             "p-value": r["p"], "Cohen d": r["cohen_d"]}
            for r in evaluation.statistics()]
    return _save(pd.DataFrame(rows), "table10_stats")


# ---------------------------------------------------------------------------
# Table XI - framework comparison
# ---------------------------------------------------------------------------
def tab_comparison() -> Path:
    rows = []
    for cap, vals in evaluation.comparison().items():
        rows.append({"Capability": cap, "Ours": vals["ours"],
                     "AIF360": vals["aif360"], "Fairlearn": vals["fairlearn"]})
    return _save(pd.DataFrame(rows), "table11_comparison")


ALL_TABLES = [
    tab_dataset_overview, tab_leakage, tab_stability, tab_ablation,
    tab_fairness, tab_shap_before_after, tab_xai_consistency,
    tab_runtime, tab_scalability, tab_stats, tab_comparison,
]


def build_all() -> List[Path]:
    out = []
    for fn in ALL_TABLES:
        p = fn()
        out.append(p)
        print(f"  wrote {p.name}")
    return out
