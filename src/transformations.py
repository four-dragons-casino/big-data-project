from __future__ import annotations

from ast import literal_eval
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import DATE_FORMAT, HIGH_CARDINALITY_LIMITS, MAX_DISPLACEMENT, MAX_MILEAGE, MAX_POWER
from .utils import slugify


COLUMN_RENAME = {
    "Price": "price",
    "Currency": "currency",
    "Condition": "condition",
    "Vehicle_brand": "brand",
    "Vehicle_model": "model",
    "Vehicle_version": "version",
    "Vehicle_generation": "generation",
    "Production_year": "production_year",
    "Mileage_km": "mileage_km",
    "Power_HP": "power_hp",
    "Displacement_cm3": "displacement_cm3",
    "Fuel_type": "fuel_type",
    "CO2_emissions": "co2_emissions",
    "Drive": "drive",
    "Transmission": "transmission",
    "Type": "vehicle_type",
    "Doors_number": "doors_number",
    "Colour": "colour",
    "Origin_country": "origin_country",
    "First_owner": "first_owner",
    "First_registration_date": "first_registration_date",
    "Offer_publication_date": "offer_publication_date",
    "Offer_location": "offer_location",
    "Features": "features_raw",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=COLUMN_RENAME)
    return renamed


def normalize_currency(df: pd.DataFrame, eur_to_pln: float) -> pd.DataFrame:
    conversion = df["currency"].map({"PLN": 1.0, "EUR": eur_to_pln}).fillna(1.0)
    df["price_pln"] = df["price"] * conversion
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["first_registration_date", "offer_publication_date"]:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce")

    df["offer_year"] = df["offer_publication_date"].dt.year.fillna(2021).astype(int)
    df["offer_month"] = df["offer_publication_date"].dt.month.fillna(1).astype(int)
    df["registration_year"] = df["first_registration_date"].dt.year
    df["registration_year"] = df["registration_year"].fillna(df["production_year"]).astype(int)
    return df


def clip_outliers(df: pd.DataFrame, price_quantile: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    price_cap = float(df["price_pln"].quantile(price_quantile))
    df = df[df["price_pln"] <= price_cap]
    df = df[df["price_pln"] > 0]

    df.loc[df["mileage_km"] > MAX_MILEAGE, "mileage_km"] = np.nan
    df.loc[df["power_hp"] > MAX_POWER, "power_hp"] = np.nan
    df.loc[df["displacement_cm3"] > MAX_DISPLACEMENT, "displacement_cm3"] = np.nan
    return df, {"price_cap": price_cap}


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_fillers = {
        "mileage_km": df.groupby("brand")["mileage_km"].transform("median"),
        "power_hp": df.groupby("brand")["power_hp"].transform("median"),
        "displacement_cm3": df.groupby("fuel_type")["displacement_cm3"].transform("median"),
        "co2_emissions": df.groupby("fuel_type")["co2_emissions"].transform("median"),
    }

    for col, group_median in numeric_fillers.items():
        df[col] = df[col].fillna(group_median)
        df[col] = df[col].fillna(df[col].median())

    df["doors_number"] = df["doors_number"].fillna(df["doors_number"].mode().iloc[0]).astype(int)
    return df


def fill_categorical(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "condition",
        "brand",
        "model",
        "version",
        "generation",
        "fuel_type",
        "drive",
        "transmission",
        "vehicle_type",
        "colour",
        "origin_country",
        "offer_location",
        "first_owner",
    ]:
        df[col] = df[col].fillna("Unknown")
    return df


def parse_feature_lists(df: pd.DataFrame) -> pd.DataFrame:
    def parse(value: str) -> List[str]:
        if pd.isna(value):
            return []
        try:
            parsed = literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            return []
        return []

    df["features_list"] = df["features_raw"].apply(parse)
    df["feature_count"] = df["features_list"].apply(len)
    return df


def extract_top_features(features: Iterable[List[str]], top_k: int) -> List[str]:
    counter: Counter[str] = Counter()
    for items in features:
        counter.update(items)
    return [name for name, _ in counter.most_common(top_k)]


def add_feature_flags(df: pd.DataFrame, top_features: List[str]) -> pd.DataFrame:
    for feat in top_features:
        col_name = f"feat_{slugify(feat)}"
        df[col_name] = df["features_list"].apply(lambda items: int(feat in items))
    return df


def cap_categories(series: pd.Series, top_n: int) -> pd.Series:
    top_values = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_values), "Other")


def apply_high_cardinality_caps(df: pd.DataFrame) -> pd.DataFrame:
    df["model_capped"] = cap_categories(df["model"], HIGH_CARDINALITY_LIMITS["model"])
    df["version_capped"] = cap_categories(df["version"], HIGH_CARDINALITY_LIMITS["version"])
    df["generation_capped"] = cap_categories(df["generation"], HIGH_CARDINALITY_LIMITS["generation"])
    df["location_capped"] = cap_categories(df["offer_location"], HIGH_CARDINALITY_LIMITS["offer_location"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_first_owner"] = (df["first_owner"].fillna("").str.lower() == "yes").astype(int)
    df["vehicle_age"] = (df["offer_year"] - df["production_year"]).clip(lower=0)
    df["vehicle_age"] = df["vehicle_age"].replace(0, 0.5)

    df["mileage_per_year"] = df["mileage_km"] / df["vehicle_age"]
    df["brand_popularity"] = df["brand"].map(df["brand"].value_counts(normalize=True))
    df["model_popularity"] = df["model_capped"].map(df["model_capped"].value_counts(normalize=True))
    return df
