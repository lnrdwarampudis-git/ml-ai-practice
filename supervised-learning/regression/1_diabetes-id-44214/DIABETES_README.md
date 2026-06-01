# Phase 1 Deep Dive: Regression Foundations With Diabetes

Dataset target:

```python
from sklearn.datasets import fetch_openml

diabetes = fetch_openml(data_id=44214, as_frame=True, parser="auto")
diabetes_df = diabetes.frame
```

The implementation is in `diabetes_pipeline.py`. It can use OpenML when available or sklearn's built-in diabetes frame with `--no-openml` for offline work.

## 1. Problem Framing

Regression predicts a continuous numeric target:

```text
given X features -> predict y numeric outcome
```

For the diabetes dataset, the target is a disease progression score. The model is not deciding a class such as diabetic vs not diabetic. It is estimating a numeric response.

Core question:

```text
How close is y_hat to y?
```

Classification asks whether the label is right. Regression asks how large the error is.

## 2. Dataset And EDA

EDA should answer:

- What does the target distribution look like?
- Are any features missing?
- Which features correlate with the target?
- Are relationships linear, curved, or noisy?
- Are there outliers or leverage points?

Artifacts produced:

- `research_missingness_report.csv`
- `numeric_summary.csv`
- `numeric_target_correlations.csv`
- `plot_target_distribution.png`
- `plot_missingness.png`
- `plot_numeric_target_correlations.png`
- top feature-vs-target scatter/regression plots

Useful interpretation:

```text
Strong linear trend -> linear model may work
Curved trend -> splines/interactions may help
Extreme values -> robust regression may help
Weak correlations -> expect limited R2
```

## 3. Baseline Models

Start simple before adding complexity.

The pipeline compares:

```text
DummyRegressor
LinearRegression
Ridge
Lasso
ElasticNet
HuberRegressor
Ridge + interactions
Ridge + splines
RandomForest
ExtraTrees
HistGradientBoosting
```

The baseline principle:

```text
If a complex model cannot beat DummyRegressor and LinearRegression,
the complexity is not justified.
```

## 4. Regression Math

A regression model learns a function:

```text
y_hat = f(X)
```

For linear regression:

```text
y_hat = b0 + b1*x1 + b2*x2 + ... + bp*xp
```

The residual is:

```text
e_i = y_i - y_hat_i
```

OLS chooses coefficients that minimize squared error:

```text
min sum((y_i - y_hat_i)^2)
```

Matrix form:

```text
beta_hat = (X^T X)^-1 X^T y
```

When features are correlated or data is small, this can become unstable. That is why Ridge, Lasso, and ElasticNet matter.

## 5. Loss Functions

Loss functions define what the model treats as bad.

Squared loss:

```text
L = (y - y_hat)^2
```

Large errors are punished heavily. Good for smooth optimization, but sensitive to outliers.

Absolute loss:

```text
L = |y - y_hat|
```

More robust to outliers, but less smooth.

Huber loss:

```text
small errors -> squared loss
large errors -> absolute-like loss
```

This is why Huber can perform well on medical-style data with unusual observations.

Artifact:

- `plot_loss_functions.png`

## 6. Metrics

RMSE:

```text
sqrt(mean((y - y_hat)^2))
```

Use when large errors should be penalized strongly.

MAE:

```text
mean(|y - y_hat|)
```

Use when you want an easy-to-explain average error.

Median absolute error:

```text
median(|y - y_hat|)
```

Useful when a few large errors distort the mean.

R2:

```text
1 - SS_res / SS_total
```

Measures how much variance the model explains compared with predicting the mean.

Important:

```text
High R2 is not the only goal.
Residual behavior and stability matter too.
```

## 7. Residual Analysis

Residual plots diagnose what the model is missing.

Useful artifacts:

- `plot_actual_vs_predicted.png`
- `plot_residuals.png`
- `plot_residual_distribution.png`
- `plot_residual_qq.png`
- `largest_prediction_errors.csv`

Interpretation:

```text
Random cloud around zero -> model assumptions look reasonable
Curve in residuals -> missing nonlinear pattern
Fan shape -> heteroscedasticity
Long residual tails -> outliers/heavy-tailed errors
Systematic high/low regions -> segment-level bias
```

## 8. Bias vs Variance

Bias:

```text
Model is too simple and misses the true pattern.
Train error high, validation error high.
```

Variance:

```text
Model is too flexible and memorizes training noise.
Train error low, validation error much higher.
```

The pipeline exports:

- `plot_bias_variance_train_vs_cv.png`
- `plot_generalization_gap.png`

Generalization gap:

```text
CV RMSE - Train RMSE
```

Large positive gap suggests overfitting.

## 9. Regularization

Ridge:

```text
min MSE + alpha * sum(beta_j^2)
```

Shrinks coefficients but keeps all features.

Lasso:

```text
min MSE + alpha * sum(|beta_j|)
```

Can set coefficients exactly to zero.

ElasticNet:

```text
min MSE + alpha * (l1_ratio * L1 + (1-l1_ratio) * L2)
```

Useful when features are correlated.

## 10. Complete Pipeline Flow

The script follows this order:

```text
load data
infer target
split train/test
create train-only EDA artifacts
cross-validate candidate models
select best model by CV RMSE
fit final model on training set
evaluate holdout test set
save model
save metrics
save feature importance
save residual/error analysis
save training profile
support prediction and monitoring
```

## 11. Commands

Train with OpenML:

```bash
/opt/anaconda3/bin/python diabetes_pipeline.py train --output-dir artifacts_diabetes --n-splits 5
```

Train offline:

```bash
/opt/anaconda3/bin/python diabetes_pipeline.py train --no-openml --output-dir artifacts_diabetes --n-splits 5
```

Sample input:

```bash
/opt/anaconda3/bin/python diabetes_pipeline.py sample-input --no-openml --rows 5 --output-csv artifacts_diabetes/sample_diabetes.csv
```

Predict:

```bash
/opt/anaconda3/bin/python diabetes_pipeline.py predict \
  --artifact-dir artifacts_diabetes \
  --input-csv artifacts_diabetes/sample_diabetes.csv \
  --output-csv artifacts_diabetes/predictions.csv
```

Monitor:

```bash
/opt/anaconda3/bin/python diabetes_pipeline.py monitor \
  --artifact-dir artifacts_diabetes \
  --input-csv artifacts_diabetes/sample_diabetes.csv \
  --output-json artifacts_diabetes/monitoring_report.json
```

## 12. What To Learn From This Phase

By the end of this phase, you should be able to explain:

- Why regression uses residuals.
- Why RMSE and MAE tell different stories.
- Why a baseline is mandatory.
- Why train/test split must happen before EDA decisions.
- Why regularization improves stability.
- Why residual plots are more informative than one metric.
- Why train-vs-CV error reveals bias and variance.
- Why production pipelines save models, metrics, profiles, and predictions together.

This phase is the foundation. Every later regression topic is an extension of this workflow.

## 13. Phase 1 Completion Additions

The implementation now also includes the optional polish items that make Phase 1 complete:

```text
Learning curves
Validation curves
Ridge coefficient paths
Lasso coefficient paths
Dataset card
Model card
Dataset fingerprint/version
Artifact creation check
```

New artifacts:

```text
dataset_version.json
dataset_card.md
model_card.md
artifact_check_report.json
learning_curve.csv
validation_curve_ridge_alpha.csv
ridge_coefficient_paths.csv
lasso_coefficient_paths.csv
plot_learning_curve.png
plot_validation_curve_ridge_alpha.png
plot_ridge_coefficient_paths.png
plot_lasso_coefficient_paths.png
```

### Learning Curve

A learning curve asks:

```text
Will more training data likely improve the model?
```

Interpretation:

```text
High train error + high CV error -> high bias
Low train error + high CV error -> high variance
Train and CV curves close together -> stable generalization
CV error still falling -> more data may help
```

### Validation Curve

A validation curve asks:

```text
How does one hyperparameter affect bias and variance?
```

For Ridge:

```text
small alpha -> less regularization, more variance risk
large alpha -> more regularization, more bias risk
best alpha -> lowest CV RMSE
```

### Coefficient Paths

Coefficient paths show how model weights change as regularization increases.

Ridge:

```text
coefficients shrink smoothly toward zero
```

Lasso:

```text
some coefficients become exactly zero
```

This makes regularization visual instead of abstract.

### Dataset And Model Cards

The dataset card explains:

```text
source
target
fingerprint
known limitations
```

The model card explains:

```text
selected model
intended use
non-intended use
metrics
validation
limitations
```

This is lightweight governance for a learning project.
