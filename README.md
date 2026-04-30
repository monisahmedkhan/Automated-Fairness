# Automated Fairness Auditing Pipeline

Companion code for the paper:

> **Automated Fairness Auditing and Bias Detection in Machine Learning via Synthetic Data and Explainable AI**
> Monis Ahmed Khan, Chase Martin, Razib Iqbal, Rahul Dubey
> Department of Computer Science, Missouri State University

A six-stage pipeline for fairness auditing: preprocessing, synthetic
data generation (SMOTE / counterfactual / GAN), model training, fairness
evaluation (DPD / DIR / EOD), explainability (SHAP / LIME), and audit
reporting. Every figure and table in the paper is produced by code in
this repo.

## Install

```bash
git clone https://github.com/<your-org>/Automated-Fairness-.git
cd Automated-Fairness-
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.10–3.13.

## Run

```bash
python verify.py            # quick component check
python main.py              # full run on all 7 datasets
python main.py --no-xai     # skip SHAP/LIME
python main.py --datasets "UCI Adult,COMPAS"
python main.py --figures-only
```

Outputs land in `figures/` (PNGs) and `results/` (CSV + Markdown
tables, audit reports, raw per-run measurements).

## Layout

```
pipeline/    six stages + orchestrator + figure/table generators
data/        auto-downloaded datasets (git-ignored)
figures/     committed PNGs of every paper figure
results/     committed tables + cached evaluation summary
main.py      CLI entry point
verify.py    standalone end-to-end check
```

## Datasets

| Dataset             | Source                              |
|---------------------|-------------------------------------|
| UCI Adult           | UCI ML repo                         |
| COMPAS              | propublica/compas-analysis          |
| Diabetes 130-US     | UCI ML repo                         |
| MEPS                | synthetic stand-in (gated source)   |
| Student Performance | UCI ML repo                         |
| Titanic             | datasciencedojo / Stanford CS109    |
| CivilComments       | synthetic stand-in (multi-GB source)|

## Citation

```
@inproceedings{khan2026fairness,
  title  = {Automated Fairness Auditing and Bias Detection in Machine Learning
            via Synthetic Data and Explainable AI},
  author = {Khan, Monis Ahmed and Martin, Chase and Iqbal, Razib and Dubey, Rahul},
  year   = {2026},
}
```
