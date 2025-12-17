# Big Data Final Project — Car Price Prediction

**Project:** End-to-end data pipeline + predictive model for estimating used-car listing prices in PLN.

**Repository:** This report documents the implementation in this workspace (pipeline in `src/`, modeling in `models/`, EDA in `eda.ipynb` / `analysis/eda.py`, and a lightweight data product in `streamlit_app.py`).

---

## Executive Summary

### Goal
The goal is to estimate a fair market **listing price** for a car advertisement given structured listing attributes (technical specs, offer metadata, equipment/features, and categorical descriptors). This supports price discovery, anomaly detection (over/under-priced listings), and can be embedded into a simple interactive estimator.

### Dataset
- Source file: `data/Car_sale_ads.csv`
- Raw size: **208,304 rows**
- Key fields include: price/currency, brand/model/version, year, mileage, engine details, transmission/drive, and a list-like “Features” field.

### What we built
1. **Reproducible preprocessing pipeline** (`python -m src.pipeline`)
   - Standardizes schema, cleans numerics, normalizes currency (EUR→PLN), handles missingness, caps high-cardinality categories, parses equipment lists into binary flags, and writes a processed Parquet dataset.
2. **EDA notebook/script** (`eda.ipynb`, `python -m analysis.eda`)
   - Generates a consistent set of figures in `reports/figures/` for understanding distributions and key relationships.
3. **Predictive model training** (`python -m models.train_model`)
   - Trains a scikit-learn pipeline using one-hot encoding for categoricals and a gradient-boosted tree regressor on log(price).
4. **Data product (interactive UI)** (`streamlit run streamlit_app.py`)
   - Loads the saved model and serves a simple form-based price estimator.

### Pipeline outputs (artifacts)
- Processed dataset: `artifacts/processed/car_ads_processed.parquet`
- Processing summary: `reports/metrics/processing_summary.json`
- Trained model artifact: `artifacts/models/price_model.joblib`
- Model evaluation + importances: `reports/metrics/model_metrics.json`

### Current results (saved metrics)
From `reports/metrics/model_metrics.json` (fast run, no hyperparameter tuning):
- Rows used after cleaning: **207,263**
- **MAE:** 8,981.94 PLN
- **RMSE:** 18,979.79 PLN
- **R²:** 0.9227

These metrics indicate strong predictive performance for a tabular baseline, with the usual caveats for marketplace data (label noise, missing fields, and non-stationarity).

### Key insights
- **Vehicle age / production year** is the strongest driver of price.
- **Power (hp)** and **mileage** are strong secondary signals.
- Equipment density (feature flags / `feature_count`) and brand effects also matter.

### Deliverable “data product”
A Streamlit application provides an accessible interface for non-technical users to input a small subset of features and obtain an estimated price in PLN.

---

## Technical Architecture

### High-level architecture
This project follows a simple Medallion-style flow:

- **Bronze (raw):** `data/Car_sale_ads.csv`
- **Silver (clean/enriched):** `artifacts/processed/car_ads_processed.parquet`
- **Gold (model + scoring):** `artifacts/models/price_model.joblib` and derived predictions in the data product

Conceptual flow:

```
          +--------------------+
          | Bronze             |
          | data/Car_sale_ads  |
          +---------+----------+
                    |
                    v
          +--------------------+
          | Silver             |
          | src.pipeline       |
          | parquet + summary  |
          +---------+----------+
                    |
           +--------+---------+
           |                  |
           v                  v
+-------------------+  +--------------------+
| EDA               |  | Modeling (Option A)|
| eda.ipynb         |  | models.train_model |
| reports/figures   |  | joblib + metrics   |
+-------------------+  +----------+---------+
                                 |
                                 v
                        +--------------------+
                        | Data product       |
                        | streamlit_app.py   |
                        +--------------------+
```

### Components and responsibilities

#### 1) Preprocessing (Pandas + PyArrow)
- Entry point: `src/pipeline.py`
- Main function: `run_pipeline(...)`
- Writes:
  - processed parquet to `artifacts/processed/`
  - JSON summary to `reports/metrics/processing_summary.json`

Key transformation stages (from `src/transformations.py`):
1. **Schema normalization**
   - Column rename to snake_case (mapping defined in `COLUMN_RENAME`).
   - Drop index column if present.
2. **Currency normalization**
   - Compute `price_pln = price * conversion` where EUR uses `EUR_TO_PLN = 4.5`.
3. **Date parsing + derived fields**
   - Parse `first_registration_date` and `offer_publication_date` using `DATE_FORMAT = "%d/%m/%Y"`.
   - Derive `offer_year`, `offer_month`.
   - Derive `registration_year` with fallback to production year.
4. **Categorical imputation**
   - Replace missing category values with `"Unknown"`.
5. **Equipment parsing and flags**
   - Parse `Features` into `features_list` (Python list).
   - Compute `feature_count`.
   - Extract top 20 features and create binary flags named `feat_<slug>`.
6. **Outlier handling and numeric cleaning**
   - Drop extreme prices above quantile `PRICE_CLIP_QUANTILE = 0.995`.
   - Replace unrealistic mileage/power/displacement with missing, then fill using group medians.
   - Ensure `doors_number` is int (mode fill).
7. **Feature engineering**
   - `is_first_owner` as a binary flag.
   - `vehicle_age` with a minimum of 0.5 for new vehicles.
   - `mileage_per_year`.
   - `brand_popularity` and `model_popularity` as relative frequencies.
8. **High-cardinality caps**
   - Bucket rare values into `Other` for `model`, `version`, `generation`, `offer_location`.

#### 2) Optional Spark pipeline (cluster-scale path)
- Entry point: `src/spark_pipeline.py`
- Purpose: provide a similar transformation logic with Spark operations.
- Notes:
  - Implements key steps (renaming, currency normalization, date parsing, feature flags, clipping, basic feature engineering).
  - Intended for large-scale runs (Databricks/EMR), writing Parquet to the configured output path.

#### 3) Modeling (scikit-learn)
- Entry point: `models/train_model.py`
- Approach:
  - Target: `price_pln`
  - Train on `log1p(price_pln)` and invert predictions with `expm1` during evaluation.
  - Split: 80/20 train-test (`train_test_split`) with stratification by `fuel_type`.

Pipeline structure:
- Numeric preprocessing: `SimpleImputer(strategy="median")`
- Categorical preprocessing: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`
- Estimator: `HistGradientBoostingRegressor(random_state=42)`

Outputs:
- Serialized model pipeline: `artifacts/models/price_model.joblib`
- Metrics JSON: `reports/metrics/model_metrics.json`
- Feature importance: permutation importance on the test set (sampled to max 1200 rows)

#### 4) Data product (Streamlit)
- Entry point: `streamlit_app.py`
- Behavior:
  - Loads processed data and model artifact.
  - Exposes a small feature subset UI (mileage, power, vehicle_age, brand, model_capped, fuel_type).
  - Produces a single predicted price in PLN.

### Configuration and paths
Paths and domain constants are centralized in `src/config.py`:
- Processed parquet path: `artifacts/processed/car_ads_processed.parquet`
- Model path: `artifacts/models/price_model.joblib`
- Metrics paths: `reports/metrics/*.json`
- Core constants: `EUR_TO_PLN=4.5`, `TOP_FEATURES=20`, `PRICE_CLIP_QUANTILE=0.995`

### Reproducibility & operational notes
- Deterministic splits and model randomness use `random_state=42`.
- Processing summary captures row counts, price cap, feature list used, and missing counts.
- The Parquet “silver” layer enables faster analytics and repeatable downstream work.

---

## Analysis Report

### Data quality and preprocessing impact
The pipeline performs a controlled set of cleaning steps to address typical marketplace issues:

- **Currency normalization:** ensures a single monetary unit (PLN) for modeling.
- **Outlier removal:** removes extreme prices beyond the 99.5th percentile cap.
- **Missingness handling:**
  - Numeric: uses brand-level medians (mileage/power) and fuel-level medians (displacement/CO2), then global median fallback.
  - Categorical: `Unknown` placeholder.
- **High-cardinality management:** avoids an explosion of one-hot dimensions by bucketing rare categories into `Other`.
- **Equipment parsing:** extracts consistent binary features from messy list strings.

From `reports/metrics/processing_summary.json`:
- Input rows: **208,304**
- Output rows: **207,263**
- Price cap applied: **544,900 PLN**
- Top 20 extracted equipment features include: ABS, Central locking, Electric windows, airbags, ESP, alloy wheels, etc.

### Exploratory data analysis (EDA)
EDA is produced by `eda.ipynb` (or `python -m analysis.eda`) and saved in `reports/figures/`. Typical observed relationships:

- **Price distribution** is heavy-tailed; log scaling stabilizes it.
- **Brand frequency** is long-tailed; a few brands dominate listings.
- **Price vs mileage** shows a strong inverse relationship.
- **Depreciation patterns** appear via `vehicle_age` and production year.
- **Fuel type differences** show systematic pricing differences.

If you regenerate figures, you should find files such as:
- `reports/figures/price_distribution.png`
- `reports/figures/top_brands.png`
- `reports/figures/price_by_fuel.png`
- `reports/figures/price_vs_mileage.png`
- `reports/figures/age_effect.png`
- `reports/figures/correlation_heatmap.png`

### Modeling methodology

#### Feature set
Training uses:
- Numeric features from `src/config.py` (`production_year`, `mileage_km`, `power_hp`, `vehicle_age`, `mileage_per_year`, etc.)
- Categorical features from `src/config.py` (`brand`, `fuel_type`, `transmission`, `vehicle_type`, etc.)
- Binary equipment flags (`feat_*`) created during preprocessing
- Additional engineered fields: `is_first_owner`, `registration_year`

#### Model choice
`HistGradientBoostingRegressor` was chosen as a strong tabular baseline:
- Handles non-linearities and interactions.
- Works well with one-hot encoded categorical inputs.
- Trains quickly for iterative experimentation.

#### Evaluation results
Saved evaluation from `reports/metrics/model_metrics.json`:
- **MAE:** 8,981.94 PLN
- **RMSE:** 18,979.79 PLN
- **R²:** 0.9227

Interpretation:
- The model explains ~92% of variance on the test split.
- MAE around ~9k PLN is practical for many pricing use cases, though errors can be higher in rare segments (e.g., very premium brands, limited trims).

#### Feature importance snapshot
Top permutation importance signals include:
- `production_year`
- `power_hp`
- `mileage_km`
- `feature_count`
- brand indicators and condition

This matches expected market mechanics: newer cars with more power and lower mileage command higher prices.

### Limitations and risks
- **Label noise:** listing price is not the final transaction price.
- **Unobserved confounders:** damage history, service status, and local demand are partially missing.
- **Temporal drift:** market prices change; training/test splits may not fully capture non-stationarity.
- **Sparse categories:** even after bucketing, rare brands/variants can be difficult.

---

## Data Product Guide

### What is the data product?
A simple **Streamlit web application** that loads:
- the processed dataset (for dropdown values and defaults), and
- a trained model artifact,

and provides an interactive UI to estimate a listing price.

### How to run
From repository root:

1) Create the processed dataset:
```bash
python3 -m src.pipeline
```

2) Train the model (fast mode):
```bash
python3 -m models.train_model --skip-tuning
```

3) Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### Inputs and outputs
Inputs currently used by the app:
- `mileage_km`
- `power_hp`
- `vehicle_age`
- `brand`
- `model_capped`
- `fuel_type`

Output:
- A single predicted price (PLN) from the loaded scikit-learn pipeline.

### Operational notes
- If the processed Parquet file is missing, the app stops and prompts you to run the pipeline.
- If the model artifact is missing, the UI still loads but predictions are disabled.
- The app includes a small compatibility shim for scikit-learn pickles (internal `ColumnTransformer` changes).

### Expected directories and artifacts
- Processed data: `artifacts/processed/car_ads_processed.parquet`
- Model: `artifacts/models/price_model.joblib`
- Metrics: `reports/metrics/model_metrics.json`

---

## References and Data Sources

### Primary data source
- `data/Car_sale_ads.csv` (car advertisement listings; provided as the project dataset)

### Libraries / frameworks
(See `requirements.txt` for pinned versions.)
- pandas, numpy
- pyarrow (Parquet I/O)
- scikit-learn (pipelines, preprocessing, modeling)
- matplotlib, seaborn (visualization)
- pyspark (optional cluster preprocessing)
- streamlit (data product UI)

### Project artifacts and documentation
- Pipeline: `src/pipeline.py`, `src/transformations.py`
- Spark pipeline (optional): `src/spark_pipeline.py`
- Model training: `models/train_model.py`
- EDA: `eda.ipynb`, `analysis/eda.py`
- Metrics and summaries: `reports/metrics/model_metrics.json`, `reports/metrics/processing_summary.json`
- Prior notes: `docs/architecture.md`, `docs/project_report.md`
