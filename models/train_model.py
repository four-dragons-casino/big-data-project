from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.utils import ensure_directory, save_json


def _get_spark_session():
    try:
        from pyspark.sql import SparkSession
    except Exception as e:
        raise RuntimeError("PySpark is required to read/write Delta paths on Databricks.") from e
    return SparkSession.builder.appName("CarPriceModel").getOrCreate()


def load_dataset(path: Path | str) -> pd.DataFrame:
    if isinstance(path, Path) and path.exists():
        return pd.read_parquet(path)
    path_str = str(path)
    if path_str.startswith(("dbfs:", "s3:", "abfss:")):
        spark = _get_spark_session()
        return spark.read.format("delta").load(path_str).toPandas()
    if "." in path_str and not path_str.startswith("/"):
        spark = _get_spark_session()
        return spark.table(path_str).toPandas()
    return pd.read_parquet(path_str)


def write_delta_from_pandas(df: pd.DataFrame, path: str | None, table: str | None) -> None:
    spark = _get_spark_session()
    writer = spark.createDataFrame(df).write.mode("overwrite").format("delta")
    if path:
        writer = writer.option("path", path)
    if table:
        writer.saveAsTable(table)
    else:
        writer.save(path)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    feature_flags = [col for col in df.columns if col.startswith("feat_")]
    numeric_features = config.NUMERIC_COLUMNS + ["is_first_owner", "registration_year"] + feature_flags
    categorical_features = config.CATEGORICAL_COLUMNS

    X = df[numeric_features + categorical_features].copy()
    y = df[config.TARGET_COLUMN]
    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    regressor = HistGradientBoostingRegressor(random_state=42)
    return Pipeline([("preprocess", preprocessor), ("regressor", regressor)])


def tune_hyperparameters(
    pipeline: Pipeline, X: pd.DataFrame, y_log: pd.Series, n_iter: int = 6
) -> Tuple[Pipeline, Dict[str, float]]:
    param_grid = {
        "regressor__max_depth": [6, 8, 10],
        "regressor__learning_rate": [0.05, 0.08, 0.1],
        "regressor__max_leaf_nodes": [31, 63, 127],
        "regressor__min_samples_leaf": [20, 50, 100],
        "regressor__l2_regularization": [0.0, 0.1, 0.2],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y_log)
    return search.best_estimator_, search.best_params_


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test_log: pd.Series) -> Dict[str, float]:
    pred_log = model.predict(X_test)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(pred_log)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_feature_importance(model: Pipeline, X: pd.DataFrame, y_log: pd.Series, top_n: int = 20) -> List[Dict[str, float]]:
    sample_size = min(len(X), 1200)
    sample = X.sample(sample_size, random_state=42)
    y_sample = y_log.loc[sample.index]

    result = permutation_importance(
        model, sample, y_sample, n_repeats=3, random_state=42, n_jobs=-1, scoring="neg_mean_squared_error"
    )
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = [
        {"feature": feature_names[i], "importance": float(imp)}
        for i, imp in enumerate(result.importances_mean)
    ]
    importances = sorted(importances, key=lambda item: item["importance"], reverse=True)
    return importances[:top_n]


def train(
    input_path: Path | str,
    model_path: Path,
    metrics_path: Path,
    sample_size: int = 80000,
    n_iter: int = 6,
    skip_tuning: bool = False,
    gold_model_path: str | None = config.GOLD_MODEL_PATH,
    gold_predictions_path: str | None = None,
    gold_predictions_table: str | None = config.GOLD_PREDICTIONS_TABLE,
    write_gold: bool = True,
) -> Dict[str, Any]:
    df = load_dataset(input_path)
    X, y, numeric_features, categorical_features = prepare_features(df)

    y_log = np.log1p(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42, stratify=df["fuel_type"]
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = build_pipeline(preprocessor)

    tuning_rows = 0
    if skip_tuning:
        tuned_model = pipeline
        best_params = tuned_model.named_steps["regressor"].get_params()
    else:
        if len(X_train) > sample_size:
            sample_idx = X_train.sample(sample_size, random_state=42).index
            X_tune = X_train.loc[sample_idx]
            y_tune = y_train.loc[sample_idx]
        else:
            X_tune, y_tune = X_train, y_train

        tuned_model, best_params = tune_hyperparameters(pipeline, X_tune, y_tune, n_iter=n_iter)
        tuning_rows = int(len(X_tune))
        best_params["tuning_rows"] = tuning_rows

    tuned_model.fit(X_train, y_train)

    metrics = evaluate(tuned_model, X_test, y_test)
    feature_importances = compute_feature_importance(tuned_model, X_test, y_test)

    ensure_directory(model_path.parent)
    joblib.dump(tuned_model, model_path)

    if write_gold and gold_model_path:
        joblib.dump(tuned_model, gold_model_path)

    if write_gold and (gold_predictions_path or gold_predictions_table):
        y_true = np.expm1(y_test)
        y_pred = np.expm1(tuned_model.predict(X_test))
        pred_df = X_test[["brand", "model_capped", "fuel_type", "vehicle_type"]].copy()
        pred_df["price_true"] = y_true
        pred_df["price_pred"] = y_pred
        write_delta_from_pandas(pred_df.reset_index(drop=True), gold_predictions_path, gold_predictions_table)

    payload = {
        "rows_used": int(len(df)),
        "best_params": best_params,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "sample_size_for_tuning": int(tuning_rows),
        "skip_tuning": skip_tuning,
        "gold_model_path": gold_model_path if write_gold else None,
        "gold_predictions_path": gold_predictions_path if write_gold else None,
        "gold_predictions_table": gold_predictions_table if write_gold else None,
    }
    save_json(metrics_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train predictive model for car price estimation.")
    parser.add_argument(
        "--input-path", type=str, default=str(config.GOLD_TABLE), help="Processed Delta/Parquet input."
    )
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH, help="Where to store the trained model.")
    parser.add_argument(
        "--metrics-path", type=Path, default=config.METRICS_PATH, help="Where to store evaluation metrics JSON."
    )
    parser.add_argument("--sample-size", type=int, default=80000, help="Rows to use for hyperparameter tuning.")
    parser.add_argument("--n-iter", type=int, default=6, help="Number of parameter sets to try during tuning.")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning for a fast run.")
    parser.add_argument(
        "--gold-model-path",
        type=str,
        default=config.GOLD_MODEL_PATH,
        help="Delta/DBFS destination for the trained model artifact.",
    )
    parser.add_argument(
        "--gold-predictions-path",
        type=str,
        default=None,
        help="Delta path for scored predictions (Gold layer).",
    )
    parser.add_argument(
        "--gold-predictions-table",
        type=str,
        default=config.GOLD_PREDICTIONS_TABLE,
        help="Delta table name for scored predictions (Gold layer).",
    )
    parser.add_argument("--no-write-gold", action="store_true", help="Skip writing Gold outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        input_path=args.input_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        sample_size=args.sample_size,
        n_iter=args.n_iter,
        skip_tuning=args.skip_tuning,
        gold_model_path=args.gold_model_path,
        gold_predictions_path=args.gold_predictions_path,
        gold_predictions_table=args.gold_predictions_table,
        write_gold=not args.no_write_gold,
    )


if __name__ == "__main__":
    main()
