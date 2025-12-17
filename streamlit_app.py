import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import _column_transformer

from src import config

# Compatibility shim for models pickled with newer scikit-learn where _RemainderColsList exists
if not hasattr(_column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _column_transformer._RemainderColsList = _RemainderColsList

MODEL_PATH = config.MODEL_PATH
DATA_PATH = config.PROCESSED_DATA_PATH

# Current model feature set (aligns with training notebook)
NUMERIC_FEATURES = ["mileage_km", "power_hp", "vehicle_age"]
CATEGORICAL_FEATURES = ["brand", "model_capped", "fuel_type"]

st.set_page_config(page_title="Car Price Estimator", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Estimator")
st.caption("Predict listing prices using mileage, power, age, brand, model, and fuel type.")


@st.cache_resource(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Processed data not found at {DATA_PATH}. Run `python -m src.pipeline` first.")
        st.stop()
    return pd.read_parquet(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.warning(f"Model file not found at {MODEL_PATH}. Train and save your model first.")
        return None
    return joblib.load(MODEL_PATH)


df = load_data()
missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
if missing_cols:
    st.error(f"Processed data missing required columns: {missing_cols}")
    st.stop()

model = load_model()
if model is None:
    st.info("No model loaded; UI available but predictions are disabled.")

# Option lists and defaults per brand/model
brands = sorted(df["brand"].dropna().unique())
brand_default = "audi" if "audi" in brands else brands[0]

def get_brand_subset(selected_brand: str) -> pd.DataFrame:
    return df[df["brand"].str.lower() == selected_brand.lower()]

def get_brand_model_subset(selected_brand: str, selected_model: str) -> pd.DataFrame:
    return df[(df["brand"].str.lower() == selected_brand.lower()) & (df["model_capped"] == selected_model)]


def make_input_df(user_inputs: dict) -> pd.DataFrame:
    # Ensure all expected columns exist; use global medians/modes as fallbacks
    numeric_defaults = {col: float(df[col].median()) for col in NUMERIC_FEATURES}
    categorical_defaults = {col: df[col].mode(dropna=True).iloc[0] for col in CATEGORICAL_FEATURES if not df[col].dropna().empty}

    row = {col: user_inputs.get(col, numeric_defaults.get(col, 0.0)) for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        row[col] = user_inputs.get(col, categorical_defaults.get(col, "Unknown"))
    return pd.DataFrame([row])


st.sidebar.header("Input Features")
selected_brand = st.sidebar.selectbox("Brand", options=brands, index=brands.index(brand_default))
brand_df = get_brand_subset(selected_brand)

if brand_df.empty:
    st.warning(f"No rows found for brand '{selected_brand}'. Predictions may be unreliable.")

models_for_brand = sorted(brand_df["model_capped"].dropna().unique()) if not brand_df.empty else []
default_model = models_for_brand[0] if models_for_brand else "Unknown"
model_choice = st.sidebar.selectbox("Model", options=models_for_brand or ["Unknown"], index=0)

model_df = get_brand_model_subset(selected_brand, model_choice) if model_choice != "Unknown" else brand_df
fuel_options = sorted(model_df["fuel_type"].dropna().unique()) if not model_df.empty else sorted(df["fuel_type"].dropna().unique())
default_fuel = fuel_options[0] if fuel_options else "Unknown"

mileage_default = int(model_df["mileage_km"].median()) if "mileage_km" in model_df else 0
power_default = int(model_df["power_hp"].median()) if "power_hp" in model_df else 0
age_default = float(model_df["vehicle_age"].median()) if "vehicle_age" in model_df else 5.0

mileage_km = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=1_200_000, value=mileage_default)
power_hp = st.sidebar.number_input("Power (hp)", min_value=30, max_value=1_200, value=power_default)
vehicle_age = st.sidebar.number_input("Vehicle age (years)", min_value=0.0, max_value=40.0, value=age_default, step=0.5)

fuel_choice = st.sidebar.selectbox("Fuel type", options=fuel_options or ["Unknown"], index=0)

if st.sidebar.button("Predict price", type="primary"):
    if model is None:
        st.error("No model loaded. Train and save the model to MODEL_PATH before predicting.")
    else:
        user_inputs = {
            "mileage_km": mileage_km,
            "power_hp": power_hp,
            "vehicle_age": vehicle_age,
            "brand": selected_brand,
            "model_capped": model_choice,
            "fuel_type": fuel_choice,
        }

        input_df = make_input_df(user_inputs)
        pred_price = float(model.predict(input_df)[0])
        st.subheader("Estimated price (PLN)")
        st.metric(label="Prediction", value=f"{pred_price:,.0f} PLN")
        st.json(user_inputs)

st.sidebar.markdown("---")
st.sidebar.caption("Train the current model in model_training.ipynb and save to artifacts/models/price_model.joblib.")
