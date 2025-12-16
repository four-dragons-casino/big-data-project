# Big Data Final Project - Car Price Prediction (Option A)

## Overview
End-to-end project on 208k car sale ads. We built a reproducible pipeline (pandas/PyArrow with an optional PySpark job), performed EDA, and delivered a price prediction model (Option A: Predictive Model). Artifacts include processed parquet data, visualizations, and a trained scikit-learn pipeline.

## Repository Structure
- `src/` - preprocessing pipeline (`pipeline.py`), shared config, utilities, optional `spark_pipeline.py` for cluster runs.
- `analysis/eda.ipynb` - interactive EDA notebook that saves >=5 figures into `reports/figures`.
- `analysis/eda.py` - script version of the same plots (optional).
- `models/train_model.py` - trains/evaluates the predictive model, saves metrics + artifact.
- `docs/` - architecture notes and project report.
- `data/Car_sale_ads.csv` - raw dataset (208,304 rows, 25 cols).
- `artifacts/` - processed dataset + trained model (created after running scripts).
- `reports/` - metrics JSON + visualizations.

## Quickstart (local)
From repo root with Python 3.10+:
```bash
python3 -m src.pipeline                     # preprocess raw CSV -> parquet + summary JSON
python3 -m analysis.eda                     # generate EDA figures (or run the notebook)
python3 -m models.train_model --skip-tuning # fast model train (used for current metrics)
# Optional: run tuning (heavier). Adjust sample/iterations as needed
python3 -m models.train_model --sample-size 60000 --n-iter 6
```
Outputs land in `artifacts/processed`, `reports/figures`, `reports/metrics`, `artifacts/models`.

## What the Pipeline Does
- Renames columns -> snake_case, removes index column.
- Currency normalization (EUR->PLN at 4.5), drops price outliers above 99.5th percentile.
- Date parsing and derived fields: `offer_year`, `offer_month`, `registration_year`, `vehicle_age`, `mileage_per_year`.
- Numeric cleaning: caps unreal mileage/power/displacement; fills with brand/fuel medians; door count mode.
- Categorical cleaning: fills missing to `Unknown`; caps high-cardinality dims (`model`, `version`, `generation`, `offer_location`).
- Features parsing: reads equipment list, creates top-20 binary flags + `feature_count`.
- Writes parquet + `processing_summary.json` with row counts, top features, and null stats.

## EDA Highlights (reports/figures)
- Log-scale price distribution, top 15 brands volume, price by fuel type, price vs mileage scatter (8k sample), age/price line, numeric correlation heatmap.

## Model (Option A)
- Features: engineered numerics + capped categoricals + 20 equipment flags; log-target training.
- Estimator: `HistGradientBoostingRegressor` via scikit-learn pipeline.
- Current fast-run metrics (`reports/metrics/model_metrics.json`): **MAE ~ 8.98k PLN, RMSE ~ 18.98k PLN, R2 ~ 0.923**.
- Artifacts: `artifacts/models/price_model.joblib` (pipeline + preprocessing) and feature importance snapshot (permutation) in metrics JSON.

## Medallion Data Flow
- Bronze: raw ingestion (`data/Car_sale_ads.csv` or lake/raw) with schema checks.
- Silver: cleaned/enriched parquet (`artifacts/processed/car_ads_processed.parquet`) via `src.pipeline` or `src.spark_pipeline`.
- Gold: model-ready features and predictions (`artifacts/models/price_model.joblib`, batch/API scores written to `predictions/...` in production).

## PySpark Path
`src/spark_pipeline.py` mirrors preprocessing for Databricks/EMR:
```bash
spark-submit src/spark_pipeline.py \
  --input-path data/Car_sale_ads.csv \
  --output-path s3://your-bucket/processed/car_ads.parquet
```
Use Spark for scaling/partitioning; then run the modeling job against the parquet output (or port the scikit pipeline to SparkML).

## Team Split Suggestion (4 students)
1) Data engineering: productionize Spark pipeline + storage/partitions.
2) Data analysis: maintain EDA notebook, validate data quality, craft visuals/story.
3) ML engineering: hyperparameter search, feature refinement, model evaluation.
4) MLOps/product: API/batch serving, monitoring/alerts, CI/testing + documentation/presentation.

## Next Steps
- Run full tuning (or SparkML) and log experiments in MLflow.
- Enrich with region info from `offer_location` and external fuel price indices.
- Build FastAPI/Databricks serving endpoint + batch scoring with monitoring on MAE/drift.
- Add CI checks (lint, unit tests for transformations) and schedule the pipeline in Airflow/Workflows.
