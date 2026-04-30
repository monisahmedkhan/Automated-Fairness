"""Automated Fairness Auditing Pipeline - main entry point.

Usage:
    python main.py                  # full run (all 7 datasets, RF model)
    python main.py --no-xai         # skip SHAP/LIME (faster)
    python main.py --datasets "UCI Adult,COMPAS"
    python main.py --figures-only   # rebuild figures + tables from the cached
                                    # evaluation summary (skips the sweep)
    python main.py --pipeline-only  # run pipeline measurements only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from pipeline import config
from pipeline import figures as fig_mod
from pipeline import tables as tab_mod
from pipeline.orchestrator import (run_pipeline, save_audit_records, save_raw)
from pipeline.utils import log


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="", help="comma-separated subset")
    p.add_argument("--model", default="rf", choices=["rf", "lr"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-xai", action="store_true",
                   help="skip SHAP/LIME for a faster smoke test")
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--pipeline-only", action="store_true")
    args = p.parse_args()

    chosen = [d.strip() for d in args.datasets.split(",") if d.strip()] or None

    t0 = time.perf_counter()

    if args.figures_only:
        log("main", "Rendering figures + tables from cached evaluation summary")
        figs = fig_mod.build_all()
        tabs = tab_mod.build_all()
        log("main", f"done: {len(figs)} figures, {len(tabs)} tables in "
                    f"{time.perf_counter()-t0:.1f}s")
        return 0

    log("main", f"Running pipeline (model={args.model}, "
                f"datasets={chosen or 'all'}, xai={'off' if args.no_xai else 'on'})")
    results = run_pipeline(model_kind=args.model, datasets=chosen,
                           seed=args.seed, run_xai=not args.no_xai)
    raw_csv = save_raw(results)
    audit_json = save_audit_records(results)

    if not args.pipeline_only:
        log("main", "Generating publication-ready figures + tables")
        figs = fig_mod.build_all()
        tabs = tab_mod.build_all()
        log("main", f"figures={len(figs)} tables={len(tabs)}")

    # quick on-screen summary
    print("\n" + "=" * 78)
    print(f"{'Dataset':<22}{'Baseline':>10}{'Full':>10}{'DPD before':>12}{'DPD after':>11}")
    print("-" * 78)
    for r in results:
        print(f"{r.name:<22}{r.accuracy_baseline*100:>9.2f}%"
              f"{r.accuracy_full*100:>9.2f}%"
              f"{r.fairness_before['dpd_avg']:>12.3f}"
              f"{r.fairness_after['dpd_avg']:>11.3f}")
    print("=" * 78)
    print(f"Raw CSV : {raw_csv}")
    print(f"Audit   : {audit_json}")
    print(f"Figures : {config.FIG_DIR}")
    print(f"Tables  : {config.RESULTS_DIR}")
    print(f"Total runtime: {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
