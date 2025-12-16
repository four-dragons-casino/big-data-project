# Architecture & Pipeline

## Overview
- **Objective:** Build a scalable pipeline to ingest car sale ads, clean/enrich the data, explore insights, and deliver a price prediction model (Option A).
- **Stack:** Pandas + PyArrow for local processing, optional PySpark job for cluster-scale runs, scikit-learn for modeling, matplotlib/seaborn for EDA.

## Medallion Data Flow
```
Bronze (raw)  : data/Car_sale_ads.csv -> optional lake/raw/car_ads/<ingestion_date>.parquet
                   - schema inference, row count/null checks
                   - stored as-is for audit/replay

Silver (clean): src.pipeline or src.spark_pipeline
                   - currency to PLN, date parsing, outlier caps
                   - missing handling, top-k feature flags, high-card caps
                   - output artifacts/processed/car_ads_processed.parquet

Gold (serving): model-ready features and predictions
                   - scikit-learn pipeline + hist gradient boosting model
                   - artifacts/models/price_model.joblib
                   - scored outputs for API/batch (predictions/offer_year=YYYY/...)

EDA/notebooks operate on Silver; dashboards/API consume Gold.
```

## Key Transformations
- Currency harmonization to PLN (EUR rate=4.5).
- Date parsing (`First_registration_date`, `Offer_publication_date`) to derive `offer_year`, `offer_month`, `registration_year`, `vehicle_age`, `mileage_per_year`.
- Outlier handling: drop price above 99.5th percentile; cap mileage/power/displacement to realistic ceilings.
- Missing handling: brand-level medians for mileage/power, fuel-level medians for displacement/CO2; categorical fallback to `Unknown`; first-owner boolean.
- High-cardinality control: cap `model` (top 80), `version` (50), `generation` (50), `offer_location` (60) into `Other` buckets.
- Feature parsing: extract top 20 equipment flags (ABS, airbags, AC, etc.) and `feature_count`.

## Modeling Path (Option A)
- Feature space = engineered numerics + capped categoricals + equipment flags.
- Preprocessing = median/mode imputation + dense one-hot encoding.
- Estimator = HistGradientBoostingRegressor (log-target). Default fast mode + optional hyperparameter search.
- Outputs = `artifacts/models/price_model.joblib`, `reports/metrics/model_metrics.json` (MAE/RMSE/R2 + permutation importances).

## Orchestration
- Local: run Python modules directly; directory creation handled in-code. Use `analysis/eda.ipynb` for interactive visuals.
- Batch/cluster: `src/spark_pipeline.py` ready for `spark-submit` on Databricks/EMR; write Bronze->Silver parquet to data lake, then trigger notebook/model job for Gold.
- Hooks for scheduling: wrap commands in Airflow/Databricks Workflows with SLA alarms on row counts and null-rate checks (see `processing_summary.json`).

## Scalability & Reliability
- Parquet output enables predicate pushdown and partitioning by offer_year/month if needed.
- Spark job uses `approxQuantile` for outlier caps and `array_contains` for feature flags; collect-only for top-k features.
- Idempotent runs: overwrite mode for artifacts; config-driven constants in `src/config.py`.
- Monitoring candidates: track pipeline row deltas, price quantile drift, model MAE drift (via batch scoring + Prometheus/MLflow).

## Team Handoff
- Data engineer: productionize Spark pipeline + storage layout.
- Analyst: maintain EDA notebook + visualize drift.
- ML engineer: hyperparameter tuning, experiment tracking, model validation/serving.
- MLOps: orchestration, CI checks, monitoring/alerting.
