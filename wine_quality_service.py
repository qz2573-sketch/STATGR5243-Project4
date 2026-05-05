from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = Path(os.getenv("WINE_APP_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLEANED_DATA_PATH = DATA_DIR / "cleaned_red_wine.csv"
BUNDLE_PATH = ARTIFACT_DIR / "wine_quality_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"

RANDOM_STATE = 42
BEST_K = 2
DEFAULT_THRESHOLD = 0.5

RAW_FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

ENGINEERED_FEATURE_COLUMNS = [
    "cluster",
    "free_to_total_SO2",
    "total_acidity",
    "alcohol_acidity_ratio",
    "PC1_score",
    "PC2_score",
]

MODEL_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "scale_pos_weight": 6,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}


def _validate_cleaned_data(df: pd.DataFrame) -> None:
    missing = [column for column in RAW_FEATURE_COLUMNS + ["good_quality"] if column not in df.columns]
    if missing:
        raise ValueError(f"cleaned_red_wine.csv is missing columns: {missing}")


def load_cleaned_data() -> pd.DataFrame:
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/cleaned_red_wine.csv. Run data_preparation.py first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    _validate_cleaned_data(df)
    return df


def _build_feature_views(df: pd.DataFrame) -> tuple[StandardScaler, PCA, KMeans, pd.DataFrame]:
    raw_features = df[RAW_FEATURE_COLUMNS].copy()

    feature_scaler = StandardScaler()
    raw_scaled = feature_scaler.fit_transform(raw_features)

    pca_2d = PCA(n_components=2)
    pca_scores = pca_2d.fit_transform(raw_scaled)

    kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(raw_scaled)

    featured_df = df.copy()
    featured_df["cluster"] = cluster_labels.astype(int)
    featured_df["free_to_total_SO2"] = (
        featured_df["free_sulfur_dioxide"] / featured_df["total_sulfur_dioxide"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    featured_df["total_acidity"] = featured_df["fixed_acidity"] + featured_df["volatile_acidity"]
    featured_df["alcohol_acidity_ratio"] = (
        featured_df["alcohol"] / featured_df["volatile_acidity"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    featured_df["PC1_score"] = pca_scores[:, 0]
    featured_df["PC2_score"] = pca_scores[:, 1]

    return feature_scaler, pca_2d, kmeans, featured_df


def _build_input_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in RAW_FEATURE_COLUMNS:
        series = df[column]
        schema.append(
            {
                "name": column,
                "min": float(series.min()),
                "max": float(series.max()),
                "default": float(series.median()),
                "step": 0.01,
            }
        )
    return schema


def _evaluate_model(model: XGBClassifier, X_test_scaled: np.ndarray, y_test: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
        "precision": round(float(precision_score(y_test, predictions)), 3),
        "recall": round(float(recall_score(y_test, predictions)), 3),
        "f1": round(float(f1_score(y_test, predictions)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
    }


def train_and_export_artifacts(force: bool = False) -> dict[str, Any]:
    if BUNDLE_PATH.exists() and METRICS_PATH.exists() and not force:
        return {
            "bundle_path": str(BUNDLE_PATH),
            "metrics_path": str(METRICS_PATH),
            "created": False,
        }

    ARTIFACT_DIR.mkdir(exist_ok=True)

    cleaned_df = load_cleaned_data()
    feature_scaler, pca_2d, kmeans, featured_df = _build_feature_views(cleaned_df)

    X = featured_df[MODEL_FEATURE_COLUMNS].copy()
    y = featured_df["good_quality"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_scaler = StandardScaler()
    X_train_scaled = model_scaler.fit_transform(X_train)
    X_test_scaled = model_scaler.transform(X_test)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)

    metrics = _evaluate_model(model, X_test_scaled, y_test)

    bundle = {
        "feature_scaler": feature_scaler,
        "pca_2d": pca_2d,
        "kmeans": kmeans,
        "model_scaler": model_scaler,
        "model": model,
        "metadata": {
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
            "engineered_feature_columns": ENGINEERED_FEATURE_COLUMNS,
            "model_feature_columns": MODEL_FEATURE_COLUMNS,
            "best_k": BEST_K,
            "threshold": DEFAULT_THRESHOLD,
            "label_rule": "good_quality = 1 if quality >= 7 else 0",
            "model_name": "XGBoost (tuned)",
            "model_params": XGB_PARAMS,
            "input_schema": _build_input_schema(cleaned_df),
        },
        "metrics": metrics,
    }

    joblib.dump(bundle, BUNDLE_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "bundle_path": str(BUNDLE_PATH),
        "metrics_path": str(METRICS_PATH),
        "created": True,
        "metrics": metrics,
    }


def ensure_artifacts() -> dict[str, Any]:
    return train_and_export_artifacts(force=False)


def load_bundle() -> dict[str, Any]:
    ensure_artifacts()
    return joblib.load(BUNDLE_PATH)


def get_input_schema() -> list[dict[str, Any]]:
    bundle = load_bundle()
    return bundle["metadata"]["input_schema"]


def _coerce_input_value(sample: dict[str, Any], column: str) -> float:
    if column not in sample:
        raise ValueError(f"Missing required input field: {column}")
    return float(sample[column])


def _prepare_single_row(sample: dict[str, Any], bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    raw_values = {column: _coerce_input_value(sample, column) for column in RAW_FEATURE_COLUMNS}
    raw_df = pd.DataFrame([raw_values], columns=RAW_FEATURE_COLUMNS)

    feature_scaler: StandardScaler = bundle["feature_scaler"]
    pca_2d: PCA = bundle["pca_2d"]
    kmeans: KMeans = bundle["kmeans"]

    raw_scaled = feature_scaler.transform(raw_df[RAW_FEATURE_COLUMNS])
    pca_scores = pca_2d.transform(raw_scaled)
    cluster_value = int(kmeans.predict(raw_scaled)[0])

    featured_df = raw_df.copy()
    featured_df["cluster"] = cluster_value
    featured_df["free_to_total_SO2"] = (
        featured_df["free_sulfur_dioxide"] / featured_df["total_sulfur_dioxide"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    featured_df["total_acidity"] = featured_df["fixed_acidity"] + featured_df["volatile_acidity"]
    featured_df["alcohol_acidity_ratio"] = (
        featured_df["alcohol"] / featured_df["volatile_acidity"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    featured_df["PC1_score"] = pca_scores[:, 0]
    featured_df["PC2_score"] = pca_scores[:, 1]

    engineered = {
        "cluster": float(cluster_value),
        "free_to_total_SO2": float(featured_df.loc[0, "free_to_total_SO2"]),
        "total_acidity": float(featured_df.loc[0, "total_acidity"]),
        "alcohol_acidity_ratio": float(featured_df.loc[0, "alcohol_acidity_ratio"]),
        "PC1_score": float(featured_df.loc[0, "PC1_score"]),
        "PC2_score": float(featured_df.loc[0, "PC2_score"]),
    }

    return featured_df[MODEL_FEATURE_COLUMNS], engineered


def predict_wine_quality(sample: dict[str, Any]) -> dict[str, Any]:
    bundle = load_bundle()
    metadata = bundle["metadata"]
    threshold = float(metadata["threshold"])

    model_inputs, engineered = _prepare_single_row(sample, bundle)
    scaled_inputs = bundle["model_scaler"].transform(model_inputs[MODEL_FEATURE_COLUMNS])
    good_probability = float(bundle["model"].predict_proba(scaled_inputs)[0, 1])
    predicted_class = int(good_probability >= threshold)
    predicted_label = "Good" if predicted_class == 1 else "Bad"

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "good_quality_probability": round(good_probability, 4),
        "bad_quality_probability": round(1 - good_probability, 4),
        "threshold": threshold,
        "message": (
            "This wine is predicted to be good quality."
            if predicted_class == 1
            else "This wine is predicted to be bad quality."
        ),
        "engineered_features": engineered,
    }


def get_app_summary() -> dict[str, Any]:
    bundle = load_bundle()
    metrics = bundle["metrics"]

    top_features = []
    importance_path = OUTPUTS_DIR / "xgb_feature_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).head(5)
        top_features = [
            {
                "feature": str(row["feature"]),
                "gain_pct": round(float(row["gain_pct"]), 2),
            }
            for _, row in importance_df.iterrows()
        ]

    return {
        "label_rule": bundle["metadata"]["label_rule"],
        "model_name": bundle["metadata"]["model_name"],
        "metrics": metrics,
        "top_features": top_features,
    }
