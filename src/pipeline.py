from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import config
from .transformations import (
    add_feature_flags,
    apply_high_cardinality_caps,
    clean_numeric,
    clip_outliers,
    engineer_features,
    extract_top_features,
    fill_categorical,
    normalize_currency,
    parse_dates,
    parse_feature_lists,
    rename_columns,
)
from .utils import ensure_directory, save_json


def run_pipeline(
    input_path: Path = config.DATA_PATH,
    output_path: str = config.SILVER_PATH,
    summary_path: Path = config.PROCESSING_SUMMARY_PATH,
    eur_to_pln: float = config.EUR_TO_PLN,
    top_features: int = config.TOP_FEATURES,
    price_clip_quantile: float = config.PRICE_CLIP_QUANTILE,
    bronze_path: str | None = config.BRONZE_PATH,
    local_output_path: Path | None = config.PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """
    Execute the end-to-end preprocessing pipeline on the raw car advertisement data.
    """
    raw_df = pd.read_csv(input_path)

    if bronze_path:
        # Bronze: raw ingested copy
        raw_df.to_parquet(bronze_path, index=False)
    df = rename_columns(raw_df)
    if "Index" in df.columns:
        df = df.drop(columns=["Index"])
    df = normalize_currency(df, eur_to_pln)
    df = parse_dates(df)
    df = fill_categorical(df)
    df = parse_feature_lists(df)
    df = apply_high_cardinality_caps(df)

    top_feature_list: List[str] = extract_top_features(df["features_list"], top_features)
    df = add_feature_flags(df, top_feature_list)

    df, outlier_stats = clip_outliers(df, price_clip_quantile)
    df = clean_numeric(df)
    df = engineer_features(df)

    df = df.drop(columns=["features_raw"])

    if isinstance(local_output_path, str) and local_output_path.strip() == "":
        local_output_path = None

    # Silver: cleaned/enriched dataset
    df.to_parquet(output_path, index=False)

    # Local convenience copy for notebooks/CLI
    if local_output_path:
        ensure_directory(local_output_path.parent)
        df.to_parquet(local_output_path, index=False)

    summary: Dict[str, Any] = {
        "input_rows": int(len(raw_df)),
        "output_rows": int(len(df)),
        "currency_rate_eur_to_pln": eur_to_pln,
        "top_features": top_feature_list,
        "outlier_rules": outlier_stats,
        "missing_after_processing": {k: int(v) for k, v in df.isna().sum().items()},
        "bronze_path": bronze_path,
        "silver_path": output_path,
        "local_output_path": str(local_output_path) if local_output_path else None,
    }
    save_json(summary_path, summary)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the car ads preprocessing pipeline.")
    parser.add_argument("--input-path", type=Path, default=config.DATA_PATH, help="Path to raw CSV file.")
    parser.add_argument("--output-path", type=str, default=config.SILVER_PATH, help="Silver layer destination (parquet).")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=config.PROCESSING_SUMMARY_PATH,
        help="Where to write processing summary JSON.",
    )
    parser.add_argument("--eur-to-pln", type=float, default=config.EUR_TO_PLN, help="Conversion rate for EUR prices.")
    parser.add_argument(
        "--top-features", type=int, default=config.TOP_FEATURES, help="Number of feature flags to extract."
    )
    parser.add_argument(
        "--price-clip-quantile",
        type=float,
        default=config.PRICE_CLIP_QUANTILE,
        help="Upper quantile for price clipping to drop extreme outliers.",
    )
    parser.add_argument("--bronze-path", type=str, default=config.BRONZE_PATH, help="Bronze layer raw parquet path.")
    parser.add_argument(
        "--local-output-path",
        type=Path,
        default=config.PROCESSED_DATA_PATH,
        help="Local copy of silver data for notebooks/CLI (set to '' to skip).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        eur_to_pln=args.eur_to_pln,
        top_features=args.top_features,
        price_clip_quantile=args.price_clip_quantile,
        bronze_path=args.bronze_path,
        local_output_path=args.local_output_path,
    )


if __name__ == "__main__":
    main()
