# Model Card: Diabetes Phase 1 Regression

## Model

Selected model: `huber_robust`

Serialized artifact: `diabetes_regression_pipeline.joblib`

## Intended Use

Educational regression foundation project. Use it to study baseline comparison, regularization, loss functions, residuals, and bias vs variance.

## Not Intended For

- Medical decision-making.
- Patient diagnosis.
- Deployment without clinical validation.

## Metrics

- RMSE: 53.18
- MAE: 41.69
- Median absolute error: 33.98
- R2: 0.466

## Validation

The pipeline uses an 80/20 holdout and KFold cross-validation on the training split.

## Learning Artifacts

- `plot_loss_functions.png`
- `plot_learning_curve.png`
- `plot_validation_curve_ridge_alpha.png`
- `plot_ridge_coefficient_paths.png`
- `plot_lasso_coefficient_paths.png`
- `plot_residuals.png`
- `plot_residual_qq.png`

## Limitations

This is a small tabular dataset. Complex models can overfit easily, so simple regularized and robust linear models are often competitive.
