import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import _column_transformer

# Compatibility shim for models pickled with newer scikit-learn where _RemainderColsList exists
# If running on an older sklearn, inject the missing class so joblib.load succeeds.
if not hasattr(_column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _column_transformer._RemainderColsList = _RemainderColsList

from src import config
from src.utils import ensure_directory

MODEL_PATH = config.MODEL_PATH
DATA_PATH = config.PROCESSED_DATA_PATH

st.set_page_config(page_title="Car Price Estimator", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Estimator (XGBoost demo)")
st.caption("Loads the trained XGBoost regression pipeline and predicts price from user inputs.")

@st.cache_resource(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Processed data not found at {DATA_PATH}. Run `python -m src.pipeline` first.")
        st.stop()
    return pd.read_parquet(DATA_PATH)

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        st.error("xgboost is not installed in this environment. Install with `pip install xgboost` and restart the app.")
        return None
    if not MODEL_PATH.exists():
        st.warning(f"Model file not found at {MODEL_PATH}. Train and save your XGBoost pipeline first.")
        return None
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

if model is None:
    st.info("No model loaded; UI available but predictions are disabled.")

# Precompute option lists and defaults
cat_cols = config.CATEGORICAL_COLUMNS
num_cols = config.NUMERIC_COLUMNS

value_counts = {col: df[col].value_counts() for col in cat_cols if col in df.columns}
mode_lookup = {col: vc.index[0] for col, vc in value_counts.items() if len(vc) > 0}

brand_popularity = df['brand'].value_counts(normalize=True) if 'brand' in df else pd.Series(dtype=float)
model_popularity = df['model_capped'].value_counts(normalize=True) if 'model_capped' in df else pd.Series(dtype=float)

def make_input_df(user_inputs: dict) -> pd.DataFrame:
    row = {}
    for col in num_cols:
        row[col] = user_inputs.get(col, df[col].median() if col in df else 0.0)
    for col in cat_cols:
        row[col] = user_inputs.get(col, mode_lookup.get(col, 'Unknown'))

    # derive fields if available
    if 'vehicle_age' in row and 'offer_year' in row and 'production_year' in row:
        row['vehicle_age'] = max(row['offer_year'] - row['production_year'], 0.0)
        # avoid zero age for mileage_per_year
        safe_age = row['vehicle_age'] if row['vehicle_age'] > 0 else 1.0
        if 'mileage_km' in row and 'mileage_per_year' in num_cols:
            row['mileage_per_year'] = row['mileage_km'] / safe_age
    if 'brand_popularity' in num_cols and 'brand' in row:
        row['brand_popularity'] = float(brand_popularity.get(row['brand'], brand_popularity.mean() if len(brand_popularity) else 0))
    if 'model_popularity' in num_cols and 'model_capped' in row:
        row['model_popularity'] = float(model_popularity.get(row['model_capped'], model_popularity.mean() if len(model_popularity) else 0))

    return pd.DataFrame([row])

st.sidebar.header("Input Features")

# Key numerics
prod_year = st.sidebar.number_input("Production year", min_value=1990, max_value=2030, value=int(df['production_year'].median()))
offer_year = st.sidebar.number_input("Offer year", min_value=2000, max_value=2030, value=int(df['offer_year'].median()))
offer_month = st.sidebar.number_input("Offer month", min_value=1, max_value=12, value=int(df['offer_month'].median()))
mileage_km = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=1_000_000, value=int(df['mileage_km'].median()))
power_hp = st.sidebar.number_input("Power (HP)", min_value=40, max_value=1000, value=int(df['power_hp'].median()))
displacement = st.sidebar.number_input("Displacement (cm3)", min_value=500, max_value=8000, value=int(df['displacement_cm3'].median()))
co2 = st.sidebar.number_input("CO2 emissions", min_value=0, max_value=1000, value=int(df['co2_emissions'].median()))

# Categoricals
brand = st.sidebar.selectbox("Brand", sorted(df['brand'].dropna().unique())) if 'brand' in df else 'Unknown'
fuel = st.sidebar.selectbox("Fuel type", sorted(df['fuel_type'].dropna().unique())) if 'fuel_type' in df else 'Unknown'
vehicle_type = st.sidebar.selectbox("Vehicle type", sorted(df['vehicle_type'].dropna().unique())) if 'vehicle_type' in df else 'Unknown'
drive = st.sidebar.selectbox("Drive", sorted(df['drive'].dropna().unique())) if 'drive' in df else 'Unknown'
transmission = st.sidebar.selectbox("Transmission", sorted(df['transmission'].dropna().unique())) if 'transmission' in df else 'Unknown'
colour = st.sidebar.selectbox("Colour", sorted(df['colour'].dropna().unique())) if 'colour' in df else 'Unknown'
condition = st.sidebar.selectbox("Condition", sorted(df['condition'].dropna().unique())) if 'condition' in df else 'Unknown'
origin = st.sidebar.selectbox("Origin country", sorted(df['origin_country'].dropna().unique())) if 'origin_country' in df else 'Unknown'
model_capped = st.sidebar.selectbox("Model (capped)", sorted(df['model_capped'].dropna().unique())) if 'model_capped' in df else 'Unknown'
version_capped = st.sidebar.selectbox("Version (capped)", sorted(df['version_capped'].dropna().unique())) if 'version_capped' in df else 'Unknown'
generation_capped = st.sidebar.selectbox("Generation (capped)", sorted(df['generation_capped'].dropna().unique())) if 'generation_capped' in df else 'Unknown'
location_capped = st.sidebar.selectbox("Location (capped)", sorted(df['location_capped'].dropna().unique())) if 'location_capped' in df else 'Unknown'

feature_count = int(df['feature_count'].median()) if 'feature_count' in df else 0
feature_count = st.sidebar.number_input("Feature count", min_value=0, max_value=100, value=feature_count)
doors_number = st.sidebar.number_input("Doors number", min_value=2, max_value=6, value=int(df['doors_number'].median()))

if st.sidebar.button("Predict price", type="primary"):
    if model is None:
        st.error("No model loaded. Train and save an XGBoost pipeline to MODEL_PATH before predicting.")
    else:
        user_inputs = {
            'production_year': prod_year,
            'offer_year': offer_year,
            'offer_month': offer_month,
            'mileage_km': mileage_km,
            'power_hp': power_hp,
            'displacement_cm3': displacement,
            'co2_emissions': co2,
            'fuel_type': fuel,
            'vehicle_type': vehicle_type,
            'drive': drive,
            'transmission': transmission,
            'colour': colour,
            'condition': condition,
            'origin_country': origin,
            'brand': brand,
            'model_capped': model_capped,
            'version_capped': version_capped,
            'generation_capped': generation_capped,
            'location_capped': location_capped,
            'feature_count': feature_count,
            'doors_number': doors_number,
        }

        input_df = make_input_df(user_inputs)
        pred_price = float(model.predict(input_df)[0])
        st.subheader("Estimated price (PLN)")
        st.metric(label="Prediction", value=f"{pred_price:,.0f} PLN")
        st.json(user_inputs)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: retrain and save an XGBoost pipeline to artifacts/models/price_model.joblib.")
