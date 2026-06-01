# Dataset Card: Ames Housing OpenML 42165

## Purpose

Train and evaluate regression models that predict `SalePrice` from property attributes.

## Source

`fetch_openml(data_id=42165, as_frame=True, parser="auto")`

## Version Fingerprint

- Rows: 2930
- Columns: 82
- SHA256: `91bad56d794eba1c149fc13034263266e2c71c37e2ae4df83f4db075721d1eac`

## Target

`SalePrice` is a continuous house sale price target. It is right-skewed, so RMSLE and log-target models are useful companion checks.

## Known Data Concerns

- Missingness can be meaningful for some housing attributes.
- Sale timing fields require prediction-time review.
- Neighborhood can induce grouped/geographic dependence.
- Future deployment may require time-based validation by sale year.

## Recommended Validation

- Default exercise: holdout + shuffled KFold.
- Enterprise review: GroupKFold by `Neighborhood` and latest-year holdout by `YrSold`.

## Rows And Columns

- Dataset rows: 2930
- Dataset columns: 82
