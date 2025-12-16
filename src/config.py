import os
from pathlib import Path

# Datalake targets (Databricks DBFS by default; override with env vars for local runs)
DATABRICKS_DATALAKE_BASE = os.getenv("DATABRICKS_DATALAKE_BASE", "dbfs:/mnt/datalake/car_price_project")
BRONZE_PATH = os.getenv("BRONZE_PATH", f"{DATABRICKS_DATALAKE_BASE}/bronze/car_sale_ads_raw")
SILVER_PATH = os.getenv("SILVER_PATH", f"{DATABRICKS_DATALAKE_BASE}/silver/car_ads_processed")
GOLD_MODEL_PATH = os.getenv("GOLD_MODEL_PATH", f"{DATABRICKS_DATALAKE_BASE}/gold/models/price_model.joblib")
GOLD_PREDICTIONS_PATH = os.getenv("GOLD_PREDICTIONS_PATH", f"{DATABRICKS_DATALAKE_BASE}/gold/predictions")

# Local convenience paths (kept for notebook/CLI usage)
DATA_PATH = Path("data/Car_sale_ads.csv")
PROCESSED_DATA_PATH = Path("artifacts/processed/car_ads_processed.parquet")
PROCESSING_SUMMARY_PATH = Path("reports/metrics/processing_summary.json")
FIGURES_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics/model_metrics.json")
MODEL_PATH = Path("artifacts/models/price_model.joblib")

# Domain settings
EUR_TO_PLN = 4.5
TOP_FEATURES = 20
PRICE_CLIP_QUANTILE = 0.995
MAX_MILEAGE = 1_000_000
MAX_POWER = 1_000
MAX_DISPLACEMENT = 8_000
DATE_FORMAT = "%d/%m/%Y"

# Modeling columns
TARGET_COLUMN = "price_pln"
NUMERIC_COLUMNS = [
    "production_year",
    "mileage_km",
    "power_hp",
    "displacement_cm3",
    "co2_emissions",
    "doors_number",
    "offer_year",
    "offer_month",
    "vehicle_age",
    "mileage_per_year",
    "feature_count",
    "brand_popularity",
    "model_popularity",
]

CATEGORICAL_COLUMNS = [
    "condition",
    "brand",
    "model_capped",
    "version_capped",
    "generation_capped",
    "fuel_type",
    "drive",
    "transmission",
    "vehicle_type",
    "colour",
    "origin_country",
    "location_capped",
]

# High-cardinality columns are pre-capped in the pipeline
HIGH_CARDINALITY_LIMITS = {
    "model": 80,
    "version": 50,
    "generation": 50,
    "offer_location": 60,
}
