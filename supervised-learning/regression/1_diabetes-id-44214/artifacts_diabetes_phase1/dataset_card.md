# Dataset Card: Diabetes Regression

## Purpose

Teach Phase 1 regression foundations: EDA, baselines, loss functions, residual analysis, and bias vs variance.

## Source

Preferred source: `fetch_openml(data_id=44214, as_frame=True, parser="auto")`

Fallback source: `sklearn.datasets.load_diabetes(as_frame=True)`

Actual source for this run: `sklearn_builtin_diabetes`

## Version Fingerprint

- Rows: 442
- Columns: 11
- SHA256: `d56d152d304e85f31001b83452ecfc98cd0b96727e44a72e2ea718c7b8ba61cf`

## Target

`target` is a continuous disease progression target.

## Known Data Notes

- Dataset is small, so validation variance matters.
- Features are numeric and mostly clean.
- A modest R2 is normal for this dataset; residual analysis matters more than chasing a high score.
