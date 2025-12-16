from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.utils import ensure_directory


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def plot_price_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price_pln"], bins=60, log_scale=True, color="#1f77b4")
    plt.xlabel("Price (PLN, log scale)")
    plt.ylabel("Count")
    plt.title("Price Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "price_distribution.png")


def plot_brand_popularity(df: pd.DataFrame, out_dir: Path) -> None:
    top_brands = df["brand"].value_counts().head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_brands.values, y=top_brands.index, palette="Blues_r")
    plt.xlabel("Number of Listings")
    plt.ylabel("Brand")
    plt.title("Top 15 Brands by Listing Volume")
    plt.tight_layout()
    plt.savefig(out_dir / "top_brands.png")


def plot_price_by_fuel(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    order = df.groupby("fuel_type")["price_pln"].median().sort_values().index
    sns.boxplot(data=df, x="fuel_type", y="price_pln", order=order)
    plt.xticks(rotation=30, ha="right")
    plt.yscale("log")
    plt.xlabel("Fuel Type")
    plt.ylabel("Price (PLN, log scale)")
    plt.title("Price by Fuel Type")
    plt.tight_layout()
    plt.savefig(out_dir / "price_by_fuel.png")


def plot_price_vs_mileage(df: pd.DataFrame, out_dir: Path) -> None:
    sample = df.sample(min(len(df), 8000), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="mileage_km", y="price_pln", hue="vehicle_type", alpha=0.4, s=20)
    plt.yscale("log")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (PLN, log scale)")
    plt.title("Price vs Mileage (sampled)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.savefig(out_dir / "price_vs_mileage.png")


def plot_age_effect(df: pd.DataFrame, out_dir: Path) -> None:
    grouped = df.groupby("production_year")["price_pln"].median().reset_index()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=grouped, x="production_year", y="price_pln", marker="o")
    plt.yscale("log")
    plt.xlabel("Production Year")
    plt.ylabel("Median Price (PLN, log scale)")
    plt.title("Vehicle Age Effect on Price")
    plt.tight_layout()
    plt.savefig(out_dir / "age_effect.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = [
        "price_pln",
        "production_year",
        "mileage_km",
        "power_hp",
        "displacement_cm3",
        "co2_emissions",
        "feature_count",
        "vehicle_age",
        "mileage_per_year",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")


def run_eda(input_path: Path, output_dir: Path) -> None:
    ensure_directory(output_dir)
    df = load_data(input_path)
    sns.set_theme(style="ticks")

    plot_price_distribution(df, output_dir)
    plot_brand_popularity(df, output_dir)
    plot_price_by_fuel(df, output_dir)
    plot_price_vs_mileage(df, output_dir)
    plot_age_effect(df, output_dir)
    plot_correlation(df, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and generate figures.")
    parser.add_argument("--input-path", type=Path, default=config.PROCESSED_DATA_PATH, help="Processed parquet file.")
    parser.add_argument("--output-dir", type=Path, default=config.FIGURES_DIR, help="Directory for figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eda(input_path=args.input_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
