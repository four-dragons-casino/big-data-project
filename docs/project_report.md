# Project Report

## Executive Summary
We built a full pipeline around 208k car sale ads to predict listing price (PLN). The cleaned dataset (207k rows) standardizes currency, caps outliers, enriches with vehicle age/mileage-per-year, and extracts the top 20 equipment flags. EDA highlights strong price lift for newer, powerful cars and premium brands. The baseline model (HistGradientBoostingRegressor) reaches **MAE ~ 8.98k PLN, RMSE ~ 18.98k PLN, R2 ~ 0.92** on a held-out test set, providing a solid foundation for deployment; hyperparameter search can be enabled for further gains.

## Problem Statement
Estimate a fair market price for a car listing given its technical specs, equipment, condition, and offer metadata. The model should generalize across brands/fuel types while handling messy real-world postings.

## Data Processing Highlights
- Currency normalized to PLN (EUR->PLN rate 4.5), price outliers above 99.5th percentile removed.
- Datetime parsing for registration and offer dates; derived `offer_year`, `offer_month`, `vehicle_age`, `mileage_per_year`.
- Missing-value strategy: brand-level medians for mileage/power, fuel-level medians for displacement/CO2, categorical fallback to `Unknown`, boolean `is_first_owner`.
- High-cardinality caps for `model`/`version`/`generation`/`offer_location`; rare values bucketed as `Other`.
- Features column parsed into a list; top 20 options converted to binary flags plus `feature_count`.
- Outputs: parquet dataset (`artifacts/processed/car_ads_processed.parquet`) and processing summary (`reports/metrics/processing_summary.json`).

## Exploratory Analysis (figures in `reports/figures`)
- **price_distribution.png**: heavy right tail; log-transform stabilizes variance.
- **top_brands.png**: VW, BMW, Audi, Opel dominate listing volume.
- **price_by_fuel.png**: electric/hybrid show premium median prices; LPG is lowest.
- **price_vs_mileage.png**: clear negative correlation, especially for SUVs/sedans.
- **age_effect.png**: steady depreciation; 2018-2021 cars maintain higher prices.
- **correlation_heatmap.png**: production year, power, and mileage are strongest numeric signals.

## Predictive Model (Option A)
- Features: engineered numerics + capped categoricals + 20 equipment flags.
- Model: HistGradientBoostingRegressor on log(price), trained via scikit-learn pipeline.
- Metrics (fast mode, no tuning): MAE 8.98k PLN, RMSE 18.98k PLN, R2 0.923.
- Importance snapshot (permutation): production year and horsepower dominate, followed by mileage, equipment count, and premium brand indicators.
- Artifacts: `artifacts/models/price_model.joblib`, `reports/metrics/model_metrics.json`.

## Data Product & Deployment Plan
- **API serving:** Wrap the saved pipeline with FastAPI; expose `/predict` accepting raw listing payload -> returns price + confidence interval. Containerize and deploy on Kubernetes/Databricks Jobs.
- **Batch scoring:** Scheduled job reads new parquet partitions, scores with the saved model, writes results to `predictions/offer_year=...` for BI tools.
- **Experiment tracking:** Integrate MLflow to log runs (params/metrics/artifacts) when running with tuning enabled.
- **Monitoring:** Track MAE drift on recent batches, feature drift (price quantiles, brand mix), and latency; alert when exceeding thresholds.

## Future Work
- Enable full hyperparameter search on a Spark cluster (or smaller folds locally) to tighten MAE/RMSE.
- Add geospatial signals from `offer_location` (city->region) and external fuel price indices.
- Calibrate prediction intervals and fairness checks across brands/fuel types.
- Implement CI hooks (linting, unit tests for transformations, data quality assertions) and automate pipeline via Airflow/Workflows.
- Collect user feedback loop from served API (accepted price vs. predicted) to retrain regularly.
