import os
from pathlib import Path

# Simple Delta table destinations (saveAsTable only)
BRONZE_TABLE = os.getenv("BRONZE_TABLE", "car_price.bronze_car_sale_ads")
SILVER_TABLE = os.getenv("SILVER_TABLE", "car_price.silver_car_ads_processed")
GOLD_TABLE = os.getenv("GOLD_TABLE", "car_price.gold_car_ads_model_ready")
GOLD_PREDICTIONS_TABLE = os.getenv("GOLD_PREDICTIONS_TABLE", "car_price.gold_predictions")
GOLD_MODEL_PATH = os.getenv("GOLD_MODEL_PATH", "dbfs:/tmp/car_price/models/price_model.joblib")

# Local convenience paths (optional)
DATA_PATH = Path("data/Car_sale_ads.csv")
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
