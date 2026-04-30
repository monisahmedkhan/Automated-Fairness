"""Stage 6 - Decision logic and audit reporting.

Combines per-dataset fairness metrics, SHAP/LIME flags and accuracy
into a structured audit record (JSON + markdown summary).  The
plain-language layer demanded in Section IV-G of the paper is rendered
here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from . import config
from .stage4_fairness import FairnessResult
from .utils import log


@dataclass
class AuditRecord:
    dataset: str
    accuracy: float
    fairness: List[Dict]
    shap_flagged: List[str]
    overall_verdict: str
    plain_summary: str


def _summary(name: str, fr: List[FairnessResult], shap_flagged: List[str]) -> str:
    fail = [r for r in fr if r.verdict == "FAIL"]
    if not fail:
        return (f"{name}: model passes all fairness checks "
                f"(DPD<={config.DPD_PASS}, "
                f"{config.DIR_LO}<=DIR<={config.DIR_HI}). "
                "No subgroup disparity flagged.")
    bits = []
    for r in fail:
        bits.append(f"sensitive='{r.sensitive}' "
                    f"DPD={r.dpd:.3f} DIR={r.dir:.3f} EOD={r.eod:.3f} "
                    f"(p={r.p_value:.3f})")
    flagged = ", ".join(shap_flagged[:5]) if shap_flagged else "none"
    return (f"{name}: FAIL on " + "; ".join(bits) +
            f". Top SHAP-flagged features: {flagged}.")


def build_record(name: str, accuracy: float, fr: List[FairnessResult],
                 shap_flagged: List[str]) -> AuditRecord:
    fails = sum(1 for r in fr if r.verdict == "FAIL")
    overall = "FAIL" if fails > 0 else "PASS"
    fair_dicts = [asdict(r) for r in fr]
    return AuditRecord(
        dataset=name,
        accuracy=float(accuracy),
        fairness=fair_dicts,
        shap_flagged=list(shap_flagged),
        overall_verdict=overall,
        plain_summary=_summary(name, fr, shap_flagged),
    )


def save_audit(records: List[AuditRecord], filename: str = "audit_report"
               ) -> None:
    out_json = config.RESULTS_DIR / f"{filename}.json"
    out_json.write_text(json.dumps([asdict(r) for r in records], indent=2))
    out_md = config.RESULTS_DIR / f"{filename}.md"
    lines = ["# Automated Fairness Audit Report", ""]
    for r in records:
        emoji = "FAIL" if r.overall_verdict == "FAIL" else "PASS"
        lines.append(f"## {r.dataset} — {emoji}")
        lines.append(f"- Accuracy: **{r.accuracy*100:.2f}%**")
        for f in r.fairness:
            lines.append(
                f"  - {f['sensitive']}: DPD={f['dpd']:.3f}, DIR={f['dir']:.3f}, "
                f"EOD={f['eod']:.3f}, p={f['p_value']:.3f}, "
                f"verdict=**{f['verdict']}**"
            )
        if r.shap_flagged:
            lines.append(f"- SHAP-flagged features: {', '.join(r.shap_flagged)}")
        lines.append("")
        lines.append(f"> {r.plain_summary}")
        lines.append("")
    out_md.write_text("\n".join(lines))
    log("stage6", f"audit saved -> {out_json}, {out_md}")
