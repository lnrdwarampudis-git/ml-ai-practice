# Phase 2 Deep Dive: Industry Regression Readiness With Ames Housing

Dataset:

```python
from sklearn.datasets import fetch_openml

housing = fetch_openml(data_id=42165, as_frame=True, parser="auto")
housing_df = housing.frame
```

This phase moves from "can we fit a regression model?" to "can we trust this pipeline like an industry project?"

Core concepts:

```text
Data contracts
Leakage prevention
Validation strategy
Metric design
Baselines first
```

Implementation:

```text
housing_pipeline.py
```

## 1. Why Ames Housing For Phase 2

Ames Housing is useful because it is messy in realistic ways:

- Many numeric and categorical features.
- Missing values have meaning.
- Sale price is skewed.
- Neighborhood and quality features are strong predictors.
- Some fields may be unavailable depending on prediction timing.
- Evaluation can look good if validation assumptions are careless.

Industry question:

```text
Would this model still behave correctly on next month's property data?
```

## 2. Data Contracts

A data contract defines what valid input data looks like.

It answers:

- Which columns are required?
- What type should each column be?
- How much missingness is normal?
- What numeric ranges are expected?
- Which categorical values are common?
- What columns are extra or missing at prediction time?

The pipeline now exports:

```text
data_contract.json
contract_validation_train.json
```

Contract logic:

```text
for each training column:
  save dtype
  save missing rate
  save unique count
  if numeric: save min/max and quantiles
  if categorical: save top observed values
```

Why it matters:

```text
Model performance is irrelevant if production data no longer matches training data.
```

## 3. Leakage Prevention

Leakage means the model sees information it would not have at prediction time.

Common leakage types:

```text
Target leakage: feature directly encodes the answer
Time leakage: feature is known only after the prediction point
Preprocessing leakage: imputation/scaling fitted before train/test split
Group leakage: same entity appears in both train and validation
Selection leakage: choosing features after looking at test performance
```

The pipeline now exports:

```text
leakage_audit.json
```

It checks:

- Suspicious column names like `sale`, `price`, `target`, `outcome`.
- Very high numeric correlation with the target.
- Prediction-time assumption.

Important:

```text
The audit flags suspicious fields. A human still decides whether the field is truly leakage.
```

## 4. Validation Strategy

Validation should imitate deployment.

Current selected strategy:

```text
80/20 holdout + shuffled KFold on training data
```

Why:

```text
For this exercise, rows are treated as independent property sales.
```

But industry alternatives matter:

```text
If predicting future sales -> split by YrSold
If predicting unseen neighborhoods -> GroupKFold by Neighborhood
If predicting new geography -> spatial split
If tuning heavily on small data -> nested CV
```

The pipeline now exports:

```text
validation_strategy.json
```

Key rule:

```text
The validation design must match the deployment story.
```

## 5. Metric Design

Metric design connects model performance to real consequences.

For housing:

```text
RMSE: penalizes large dollar mistakes
MAE: easier to explain as average dollar error
RMSLE: useful for skewed prices and relative error
R2: variance explained, but not enough alone
```

The pipeline now exports:

```text
metric_design.json
```

Metric selection rule:

```text
Use CV RMSE to select a model.
Inspect MAE, RMSLE, R2, and residual plots before trusting it.
```

Why not only R2?

```text
A model can have high R2 and still systematically underprice expensive homes.
```

## 6. Baselines First

Industry ML begins with baselines.

Baselines in this phase:

```text
DummyRegressor median
Simple Ridge with log target
```

A complex model must beat simple baselines clearly enough to justify complexity.

The pipeline now exports:

```text
baseline_improvement_report.csv
plot_baseline_improvement.png
plot_model_comparison_rmse.png
plot_train_vs_cv_rmse.png
```

Interpretation:

```text
Small improvement over baseline -> keep the simple model
Large improvement but big train/CV gap -> possible overfitting
Strong CV performance + small gap -> better candidate
```

## 7. Train/Test Safe Preprocessing

All preprocessing lives inside sklearn `Pipeline` and `ColumnTransformer`.

This prevents leakage because these steps fit only on training folds:

```text
imputation
missing indicators
scaling
outlier clipping
one-hot encoding
polynomial/spline basis expansion
```

Wrong:

```text
impute full dataset, then split
```

Correct:

```text
split first
fit imputer inside pipeline on training data only
```

## 8. Production Monitoring Bridge

The `monitor` command now includes:

```text
missing required columns
extra columns
missing-rate drift
data contract validation
new/rare category share
numeric out-of-range share
```

Command:

```bash
/opt/anaconda3/bin/python housing_pipeline.py monitor \
  --artifact-dir artifacts_housing_phase2 \
  --input-csv artifacts_housing_phase2/sample_houses.csv \
  --output-json artifacts_housing_phase2/monitoring_report.json
```

## 9. Artifacts To Inspect

Core industry artifacts:

```text
data_contract.json
contract_validation_train.json
leakage_audit.json
validation_strategy.json
metric_design.json
baseline_improvement_report.csv
training_profile.json
monitoring_report.json
```

Model artifacts:

```text
housing_price_pipeline.joblib
metrics.json
cv_results.csv
test_predictions.csv
largest_prediction_errors.csv
```

Plots:

```text
plot_model_comparison_rmse.png
plot_baseline_improvement.png
plot_train_vs_cv_rmse.png
plot_actual_vs_predicted.png
plot_residuals.png
plot_missingness.png
plot_target_distribution.png
```

## 10. What You Should Learn In Phase 2

By the end of this phase, you should be able to explain:

- Why data contracts are needed before deployment.
- How leakage can make validation scores dishonest.
- Why validation strategy depends on deployment context.
- Why metric design is a business decision, not only a math choice.
- Why a dummy baseline and simple model are mandatory.
- Why preprocessing must live inside a pipeline.
- How monitoring connects training assumptions to production data.

This phase is what turns a notebook model into an industry-ready ML workflow.

## 11. Enterprise Extension Added

The Phase 2 implementation now also covers the deeper industry layer:

```text
Data versioning
Dataset card
Model card
Group/time validation comparison
Segment-level error analysis
Rejected-record style monitoring
Lightweight schema/leakage test report
```

New artifacts:

```text
dataset_version.json
dataset_card.md
model_card.md
schema_test_report.json
validation_comparison.json
segment_metrics.csv
segment_metrics.json
plot_segment_mae.png
```

Additional monitoring fields:

```text
data_contract_validation
row_validation_report
accepted_rows
rejected_rows
rejection_reasons
```

Additional validation checks:

```text
GroupKFold by Neighborhood
Latest-year holdout by YrSold
```

These checks answer questions that average KFold scores cannot:

```text
Does performance hold for unseen neighborhoods?
Does performance hold on the latest sale year?
Which segments have the largest errors?
Can we reproduce the exact data version later?
What rows should be reviewed before scoring?
```
