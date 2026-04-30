"""Loader for the cached cross-seed evaluation summary.

The end-to-end multi-seed sweep across all seven benchmark datasets is
expensive (≈ 50 min on a 16-vCPU host).  Once it has been run, the
aggregated metrics are cached in ``results/evaluation_summary.json`` so
that downstream artefacts (figures, tables, audit reports) can be
regenerated cheaply without re-running the full sweep.

The live pipeline (``pipeline.orchestrator``) still writes its own
per-run measurements to ``results/raw_pipeline_measurements.csv`` so a
fresh run can always be cross-checked against the cached aggregate.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List

from . import config

_SUMMARY_FILE = config.RESULTS_DIR / "evaluation_summary.json"


@lru_cache(maxsize=1)
def _load() -> Dict[str, Any]:
    if not _SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation summary not found at {_SUMMARY_FILE}. "
            "Run the multi-seed sweep (`python main.py`) to regenerate it."
        )
    return json.loads(_SUMMARY_FILE.read_text())


def datasets() -> Dict[str, Dict[str, Any]]:
    return _load()["datasets"]


def dataset_names() -> List[str]:
    return list(datasets().keys())


def accuracy(name: str, key: str) -> float:
    return datasets()[name]["accuracy"][key]


def fairness(name: str) -> Dict[str, float]:
    return datasets()[name]["fairness"]


def stability(name: str) -> Dict[str, float]:
    return datasets()[name]["stability_10runs"]


def leakage(name: str) -> Dict[str, float]:
    return datasets()[name]["leakage"]


def xai(name: str) -> Dict[str, float]:
    return datasets()[name]["xai_consistency"]


def shap_features(name: str) -> List[Dict[str, Any]]:
    return datasets()[name]["shap_features"]


def runtime() -> List[Dict[str, Any]]:
    return _load()["runtime_breakdown"]


def scalability() -> List[Dict[str, Any]]:
    return _load()["scalability"]


def statistics() -> List[Dict[str, Any]]:
    return _load()["statistics"]


def comparison() -> Dict[str, Dict[str, Any]]:
    return _load()["framework_comparison"]


def evaluation_protocol() -> Dict[str, Any]:
    return _load()["evaluation_protocol"]
