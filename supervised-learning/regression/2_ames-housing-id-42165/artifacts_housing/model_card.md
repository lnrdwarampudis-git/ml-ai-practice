# Model Card: Ames Housing Regression

## Model

Selected model: `simple_ridge`

Serialized artifact: `housing_price_pipeline.joblib`

## Intended Use

Estimate house sale price from pre-sale property attributes for learning and prototype analysis.

## Not Intended For

- Final appraisal decisions without human review.
- Deployment where sale-timing fields are unavailable or legally restricted.
- Unseen geographies without spatial/group validation.

## Primary Metrics

- RMSE: 27431.61
- MAE: 14541.21
- RMSLE: 0.111
- R2: 0.906

## Validation

Default split: 80/20 holdout plus KFold on training data.

Additional validation artifacts:

- `validation_comparison.json`
- `segment_metrics.csv`
- `plot_train_vs_cv_rmse.png`

## Leakage Review

Status: `review_required`

Flagged columns should be reviewed against the prediction-time assumption.

## Limitations

- Missingness and rare categories can shift in production.
- Neighborhood and sale-year effects may reduce generalization.
- High-price homes may have different residual behavior than typical homes.

## Monitoring

Use `monitor` to inspect missing required columns, extra columns, missing-rate drift, contract validation, and row-level review flags.
