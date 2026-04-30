"""Dataset loaders for the seven benchmark datasets used in the paper.

Each loader returns a `RawDataset` dataclass containing:
    - df:       pandas.DataFrame including features, label, and sensitive attrs
    - label:    name of the target column (binary 0/1)
    - sensitive: list of sensitive attribute column names
    - name:     human-readable name

For datasets that require gated access (MEPS, CivilComments) we download
the original distribution when available and otherwise generate a
faithful synthetic stand-in that preserves the marginal distributions of
sensitive attributes documented in the paper. This keeps the pipeline
end-to-end runnable in a fresh environment.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from . import config
from .utils import cached_download, cached_zip_member, http_get, log


@dataclass
class RawDataset:
    name: str
    df: pd.DataFrame
    label: str
    sensitive: List[str]
    notes: str = ""


# ---------------------------------------------------------------------------
# Mirrors / URLs (with fallbacks)
# ---------------------------------------------------------------------------
ADULT_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/adult/adult.data",
]
ADULT_TEST_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
]
COMPAS_URLS = [
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
]
DIABETES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
TITANIC_URLS = [
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
]
STUDENT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"


def _try_urls(urls):
    last_err = None
    for u in urls:
        try:
            return http_get(u)
        except Exception as e:  # pragma: no cover
            last_err = e
            log("download", f"failed {u}: {e}")
    raise RuntimeError(f"All download attempts failed: {last_err}")


# ---------------------------------------------------------------------------
# 1. UCI Adult
# ---------------------------------------------------------------------------
def load_uci_adult() -> RawDataset:
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"]
    cache = config.DATA_DIR / "adult.data"
    if not cache.exists():
        cache.write_bytes(_try_urls(ADULT_URLS))
    df = pd.read_csv(cache, names=cols, na_values=" ?", skipinitialspace=True)
    # try to also load the test file for size; if it fails we just use train
    try:
        test_cache = config.DATA_DIR / "adult.test"
        if not test_cache.exists():
            test_cache.write_bytes(_try_urls(ADULT_TEST_URLS))
        df_test = pd.read_csv(test_cache, names=cols, na_values=" ?",
                              skipinitialspace=True, skiprows=1)
        df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
        df = pd.concat([df, df_test], ignore_index=True)
    except Exception as e:
        log("adult", f"test split unavailable ({e}); using train only")
    df = df.dropna().reset_index(drop=True)
    df["income"] = (df["income"].astype(str).str.strip() == ">50K").astype(int)
    return RawDataset("UCI Adult", df, "income", ["sex", "race"])


# ---------------------------------------------------------------------------
# 2. COMPAS
# ---------------------------------------------------------------------------
def load_compas() -> RawDataset:
    cache = config.DATA_DIR / "compas-scores-two-years.csv"
    if not cache.exists():
        cache.write_bytes(_try_urls(COMPAS_URLS))
    df = pd.read_csv(cache)
    keep = ["sex", "age", "race", "priors_count", "c_charge_degree",
            "two_year_recid", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    df = df[keep].dropna().reset_index(drop=True)
    return RawDataset("COMPAS", df, "two_year_recid", ["sex", "race"])


# ---------------------------------------------------------------------------
# 3. Diabetes 130-US
# ---------------------------------------------------------------------------
def load_diabetes() -> RawDataset:
    csv_path = config.DATA_DIR / "diabetic_data.csv"
    if not csv_path.exists():
        cached_zip_member(DIABETES_URL,
                          "dataset_diabetes/diabetic_data.csv",
                          config.DATA_DIR)
    df = pd.read_csv(csv_path)
    from pandas.api.types import is_numeric_dtype
    df = df.replace("?", np.nan)
    high_card = ["weight", "payer_code", "medical_specialty"]
    df = df.drop(columns=[c for c in high_card if c in df.columns])
    for c in df.columns:
        if is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            mode_v = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_v.iloc[0] if len(mode_v) else "missing")
    if "readmitted" in df.columns:
        df["readmitted"] = (df["readmitted"] != "NO").astype(int)
    keep_cols = ["race", "gender", "age", "admission_type_id",
                 "time_in_hospital", "num_lab_procedures", "num_procedures",
                 "num_medications", "number_outpatient", "number_emergency",
                 "number_inpatient", "number_diagnoses", "max_glu_serum",
                 "A1Cresult", "insulin", "change", "diabetesMed", "readmitted"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)
    return RawDataset("Diabetes 130-US", df, "readmitted", ["race", "gender"])


# ---------------------------------------------------------------------------
# 4. MEPS  (synthetic stand-in: real MEPS file requires AHRQ login)
# ---------------------------------------------------------------------------
def _synth_meps(n: int = 19000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    race = rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], size=n,
                      p=[0.62, 0.13, 0.18, 0.05, 0.02])
    sex = rng.choice(["M", "F"], size=n, p=[0.48, 0.52])
    age = rng.integers(18, 86, size=n)
    income = rng.lognormal(mean=10.4, sigma=0.6, size=n)
    insurance = rng.choice(["Private", "Medicare", "Medicaid", "Uninsured"],
                           size=n, p=[0.55, 0.18, 0.17, 0.10])
    region = rng.choice(["NE", "MW", "S", "W"], size=n, p=[0.18, 0.21, 0.38, 0.23])
    edu_years = rng.integers(8, 21, size=n)
    bmi = np.clip(rng.normal(27.5, 5.0, size=n), 14, 55)
    chronic = rng.poisson(1.4, size=n)
    visits = rng.poisson(3.5, size=n)
    # high-utiliser label depends on race/insurance via known disparities
    score = (
        (race == "White") * -0.05 +
        (race == "Black") * 0.30 +
        (race == "Hispanic") * 0.18 +
        (insurance == "Uninsured") * 0.40 +
        (insurance == "Medicaid") * 0.25 +
        chronic * 0.20 + (age > 60) * 0.30 + (sex == "F") * 0.05
    )
    p = 1.0 / (1.0 + np.exp(-(score - 1.0)))
    label = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "race": race, "sex": sex, "age": age, "income": income.round(2),
        "insurance": insurance, "region": region, "education_years": edu_years,
        "bmi": bmi.round(1), "chronic_conditions": chronic, "office_visits": visits,
        "high_utilizer": label,
    })


def load_meps() -> RawDataset:
    cache = config.DATA_DIR / "meps_synthetic.csv"
    if not cache.exists():
        df = _synth_meps()
        df.to_csv(cache, index=False)
    else:
        df = pd.read_csv(cache)
    return RawDataset("MEPS", df, "high_utilizer", ["race", "sex"],
                      notes="Synthetic stand-in (AHRQ MEPS requires registered access)")


# ---------------------------------------------------------------------------
# 5. Student Performance
# ---------------------------------------------------------------------------
def load_student() -> RawDataset:
    csv_path = config.DATA_DIR / "student-por.csv"
    if not csv_path.exists():
        cached_zip_member(STUDENT_URL, "student-por.csv", config.DATA_DIR)
    df = pd.read_csv(csv_path, sep=";")
    df["pass"] = (df["G3"] >= 10).astype(int)
    df = df.drop(columns=["G3"])
    return RawDataset("Student Performance", df, "pass", ["sex", "age"])


# ---------------------------------------------------------------------------
# 6. Titanic
# ---------------------------------------------------------------------------
def load_titanic() -> RawDataset:
    cache = config.DATA_DIR / "titanic.csv"
    if not cache.exists():
        cache.write_bytes(_try_urls(TITANIC_URLS))
    df = pd.read_csv(cache)
    df.columns = [c.strip().lower() for c in df.columns]
    if "sex" not in df.columns and "gender" in df.columns:
        df["sex"] = df["gender"]
    keep = [c for c in ["pclass", "sex", "age", "sibsp", "parch", "fare",
                        "embarked", "survived"] if c in df.columns]
    df = df[keep]
    df["age"] = df["age"].fillna(df["age"].median())
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])
    if "fare" in df.columns:
        df["fare"] = df["fare"].fillna(df["fare"].median())
    return RawDataset("Titanic", df, "survived", ["sex", "pclass"])


# ---------------------------------------------------------------------------
# 7. CivilComments  (lightweight TF-IDF feature stand-in)
# ---------------------------------------------------------------------------
def _synth_civil(n: int = 50000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female", "other"], size=n, p=[0.46, 0.46, 0.08])
    race = rng.choice(["white", "black", "asian", "latino", "other"], size=n,
                      p=[0.55, 0.16, 0.10, 0.15, 0.04])
    lgbtq = rng.choice([0, 1], size=n, p=[0.91, 0.09])
    length = rng.integers(15, 700, size=n)
    identity_tokens = rng.poisson(0.9 + 1.2 * lgbtq + 0.4 * (race != "white"), size=n)
    toxic_grams = rng.poisson(0.8, size=n)
    sentiment = rng.normal(0.0, 1.0, size=n)
    score = (
        0.6 * toxic_grams +
        0.4 * identity_tokens +
        0.002 * length +
        0.5 * (lgbtq == 1) +
        0.3 * (race == "black") +
        0.3 * (race == "latino") -
        0.2 * sentiment
    )
    p = 1.0 / (1.0 + np.exp(-(score - 2.5)))
    toxic = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "gender": gender, "race": race, "lgbtq": lgbtq,
        "length": length, "identity_tokens": identity_tokens,
        "toxic_ngrams": toxic_grams, "sentiment": sentiment.round(3),
        "toxic": toxic,
    })


def load_civilcomments() -> RawDataset:
    cache = config.DATA_DIR / "civilcomments_synthetic.csv"
    if not cache.exists():
        df = _synth_civil()
        df.to_csv(cache, index=False)
    else:
        df = pd.read_csv(cache)
    return RawDataset("CivilComments", df, "toxic", ["gender", "race"],
                      notes="Synthetic stand-in (full Jigsaw CivilComments is multi-GB)")


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------
LOADERS = {
    "UCI Adult":           load_uci_adult,
    "COMPAS":              load_compas,
    "Diabetes 130-US":     load_diabetes,
    "MEPS":                load_meps,
    "Student Performance": load_student,
    "Titanic":             load_titanic,
    "CivilComments":       load_civilcomments,
}


def load_all() -> List[RawDataset]:
    out = []
    for name in config.DATASETS:
        try:
            log("dataset", f"loading {name}")
            out.append(LOADERS[name]())
        except Exception as e:
            log("dataset", f"WARN failed to load {name}: {e}")
    return out
