from __future__ import annotations

"""
PySpark implementation of the preprocessing pipeline.
This file is intended to be run on a Spark cluster or Databricks workspace.
"""

from pathlib import Path
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import config
from .transformations import COLUMN_RENAME
from .utils import slugify

DATE_PATTERN = "dd/MM/yyyy"


def build_spark(app_name: str = "CarAdsSparkPipeline") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def rename_columns(df):
    for old, new in COLUMN_RENAME.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    return df


def add_feature_flags(df, top_features: List[str]):
    for feat in top_features:
        col_name = f"feat_{slugify(feat)}"
        df = df.withColumn(col_name, F.array_contains("features_list", F.lit(feat)).cast("int"))
    return df


def run_spark_pipeline(
    input_path: Path = config.DATA_PATH,
    output_path: Path = config.PROCESSED_DATA_PATH,
    top_features: int = config.TOP_FEATURES,
    price_clip_quantile: float = config.PRICE_CLIP_QUANTILE,
) -> None:
    spark = build_spark()

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(input_path))
    )
    df = rename_columns(df)
    df = df.drop("Index") if "Index" in df.columns else df

    df = df.withColumn(
        "price_pln",
        F.when(F.col("currency") == "EUR", F.col("price") * F.lit(config.EUR_TO_PLN)).otherwise(F.col("price")),
    )

    df = df.withColumn("first_registration_date", F.to_date("first_registration_date", DATE_PATTERN))
    df = df.withColumn("offer_publication_date", F.to_date("offer_publication_date", DATE_PATTERN))
    df = df.withColumn("offer_year", F.coalesce(F.year("offer_publication_date"), F.lit(2021)))
    df = df.withColumn("offer_month", F.coalesce(F.month("offer_publication_date"), F.lit(1)))
    df = df.withColumn(
        "registration_year", F.coalesce(F.year("first_registration_date"), F.col("production_year")).cast("int")
    )

    fill_values = {
        "condition": "Unknown",
        "brand": "Unknown",
        "model": "Unknown",
        "version": "Unknown",
        "generation": "Unknown",
        "fuel_type": "Unknown",
        "drive": "Unknown",
        "transmission": "Unknown",
        "vehicle_type": "Unknown",
        "colour": "Unknown",
        "origin_country": "Unknown",
        "offer_location": "Unknown",
        "first_owner": "Unknown",
    }
    df = df.fillna(fill_values)

    df = df.withColumn("features_raw_clean", F.regexp_replace("features_raw", "'", "\""))
    df = df.withColumn("features_list", F.from_json("features_raw_clean", T.ArrayType(T.StringType())))
    df = df.withColumn("feature_count", F.size(F.col("features_list")))
    df = df.drop("features_raw_clean")

    feature_counts = (
        df.select(F.explode_outer("features_list").alias("feature"))
        .groupBy("feature")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_features)
    )
    top_feature_values = [row["feature"] for row in feature_counts.collect() if row["feature"]]
    df = add_feature_flags(df, top_feature_values)

    price_cap = df.approxQuantile("price_pln", [price_clip_quantile], 0.01)[0]
    df = df.filter((F.col("price_pln") > 0) & (F.col("price_pln") <= F.lit(price_cap)))
    df = df.filter(F.col("mileage_km") <= config.MAX_MILEAGE)
    df = df.filter(F.col("power_hp") <= config.MAX_POWER)
    df = df.filter(F.col("displacement_cm3") <= config.MAX_DISPLACEMENT)

    df = df.withColumn(
        "vehicle_age",
        F.when((F.col("offer_year") - F.col("production_year")) <= 0, F.lit(0.5)).otherwise(
            F.col("offer_year") - F.col("production_year")
        ),
    )
    df = df.withColumn("mileage_per_year", F.col("mileage_km") / F.col("vehicle_age"))
    df = df.withColumn("is_first_owner", (F.lower(F.col("first_owner")) == F.lit("yes")).cast("int"))

    brand_counts = df.groupBy("brand").count()
    df = df.join(
        brand_counts.withColumn("brand_popularity", F.col("count") / F.lit(df.count())),
        on="brand",
        how="left",
    )

    ensure_path = Path(output_path)
    ensure_path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(output_path))


if __name__ == "__main__":
    run_spark_pipeline()
