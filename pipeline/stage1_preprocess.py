"""Stage 1 - Data preprocessing and validation.

Implements Algorithm 1 from the paper:
  - separate numeric / categorical columns
  - impute missing values (median / mode)
  - label encode non-sensitive categoricals
  - standardise numeric features (z-score)
  - stratified train/test split

The output is a `ProcessedDataset` containing both the encoded matrices
ready for ML and a fitted `Preprocessor` that can transform new data
(used by counterfactual / GAN augmentation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config
from .datasets import RawDataset
from .utils import log, to_binary


def _is_categorical(s: pd.Series) -> bool:
    """A column is categorical if it isn't numeric or boolean."""
    return not is_numeric_dtype(s)


@dataclass
class Preprocessor:
    label_encoders: Dict[str, LabelEncoder]
    scaler: StandardScaler
    feature_names: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    sensitive_cols: List[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = df.copy()
        for c, le in self.label_encoders.items():
            if c in out.columns:
                out[c] = out[c].astype(str)
                # unseen categories -> map to most frequent class id 0
                known = set(le.classes_)
                out[c] = out[c].apply(lambda v: v if v in known else le.classes_[0])
                out[c] = le.transform(out[c])
        out = out[self.feature_names]
        for c in self.numeric_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        out_arr = out.to_numpy(dtype=float)
        out_arr[:, [self.feature_names.index(c) for c in self.numeric_cols]] = (
            self.scaler.transform(out_arr[:, [self.feature_names.index(c) for c in self.numeric_cols]])
        )
        return out_arr


@dataclass
class ProcessedDataset:
    name: str
    pre: Preprocessor
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    s_train: pd.DataFrame      # sensitive attributes (raw / pre-encoding)
    s_test: pd.DataFrame
    df_train: pd.DataFrame     # cleaned (un-scaled) train df  -- for counterfactual / GAN
    df_test: pd.DataFrame
    label: str
    sensitive: List[str]


def fit_preprocess(raw: RawDataset, seed: int = 42) -> ProcessedDataset:
    df = raw.df.copy()
    label = raw.label
    sensitive = list(raw.sensitive)

    if label not in df.columns:
        raise ValueError(f"{raw.name}: label '{label}' missing")
    df = df.dropna(subset=[label]).reset_index(drop=True)
    df[label] = to_binary(df[label])

    feature_cols = [c for c in df.columns if c != label]
    categorical_cols = [c for c in feature_cols if _is_categorical(df[c])]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)
    for c in categorical_cols:
        df[c] = df[c].astype(str).replace({"nan": None})
        mode_val = df[c].mode(dropna=True)
        fill_val = mode_val.iloc[0] if len(mode_val) else "missing"
        df[c] = df[c].fillna(fill_val)

    # split (stratified on label)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                                      random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df[label]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # label-encode categorical columns (fit on train)
    label_encoders: Dict[str, LabelEncoder] = {}
    for c in categorical_cols:
        le = LabelEncoder()
        le.fit(df_train[c].astype(str))
        label_encoders[c] = le
    # apply
    for c in categorical_cols:
        le = label_encoders[c]
        known = set(le.classes_)
        df_train[c] = le.transform(df_train[c].astype(str))
        df_test[c] = df_test[c].astype(str).apply(
            lambda v: v if v in known else le.classes_[0]
        )
        df_test[c] = le.transform(df_test[c])

    # sensitive frames (post-encoding, integer-coded for fairlearn)
    s_train = df_train[sensitive].copy()
    s_test = df_test[sensitive].copy()

    scaler = StandardScaler()
    if numeric_cols:
        # final defensive impute against any post-split NaN
        for c in numeric_cols:
            df_train[c] = pd.to_numeric(df_train[c], errors="coerce").fillna(0.0)
            df_test[c] = pd.to_numeric(df_test[c], errors="coerce").fillna(0.0)
        df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
        df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])
    df_train = df_train.fillna(0.0)
    df_test = df_test.fillna(0.0)

    pre = Preprocessor(label_encoders=label_encoders, scaler=scaler,
                       feature_names=feature_cols,
                       numeric_cols=numeric_cols,
                       categorical_cols=categorical_cols,
                       sensitive_cols=sensitive)

    X_train = df_train[feature_cols].to_numpy(dtype=float)
    X_test = df_test[feature_cols].to_numpy(dtype=float)
    y_train = df_train[label].to_numpy(dtype=int)
    y_test = df_test[label].to_numpy(dtype=int)

    log("stage1", f"{raw.name:<20} train={len(y_train):>6} test={len(y_test):>5} "
        f"feats={len(feature_cols):>3} sensitive={sensitive} pos%={y_train.mean()*100:5.1f}")

    return ProcessedDataset(
        name=raw.name, pre=pre,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        s_train=s_train, s_test=s_test,
        df_train=df_train, df_test=df_test,
        label=label, sensitive=sensitive,
    )
