"""Utility functions shared across pipeline stages."""

from __future__ import annotations

import io
import os
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import requests

from . import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(stage: str, msg: str) -> None:
    print(f"[{stage:>9}] {msg}", flush=True)


@contextmanager
def timer(stage: str, message: str) -> Iterator[None]:
    t0 = time.perf_counter()
    log(stage, f"start  - {message}")
    try:
        yield
    finally:
        log(stage, f"finish - {message}  ({time.perf_counter()-t0:.2f}s)")


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def http_get(url: str, timeout: int = 60) -> bytes:
    """GET a URL with a friendly user-agent (some hosts block defaults)."""
    headers = {"User-Agent": "automated-fairness/1.0 (research)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content


def cached_download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log("download", f"GET {url}")
    dest.write_bytes(http_get(url))
    return dest


def cached_zip_member(url: str, member: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(member).name
    if out.exists() and out.stat().st_size > 0:
        return out
    log("download", f"GET {url} -> {member}")
    data = http_get(url)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open(member) as src:
            out.write_bytes(src.read())
    return out


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def to_binary(arr) -> np.ndarray:
    """Best-effort coercion of a Series/array to {0,1}."""
    arr = np.asarray(arr)
    uniq = np.unique(arr)
    if set(uniq).issubset({0, 1}):
        return arr.astype(int)
    if arr.dtype.kind in "fc":
        return (arr >= np.median(arr)).astype(int)
    pos = uniq[-1]
    return (arr == pos).astype(int)


def stratified_subsample(df: pd.DataFrame, n: int, key: str, seed: int = 0) -> pd.DataFrame:
    """Sub-sample a dataframe down to n rows, preserving the distribution of `key`."""
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    out = []
    counts = df[key].value_counts(normalize=True)
    for v, frac in counts.items():
        sub = df[df[key] == v]
        take = max(1, int(round(frac * n)))
        take = min(take, len(sub))
        out.append(sub.sample(take, random_state=int(rng.integers(0, 1 << 31))))
    return pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed)


def ensure_array(x) -> np.ndarray:
    if hasattr(x, "values"):
        return x.values
    return np.asarray(x)
