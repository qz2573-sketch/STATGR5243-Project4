import json

from wine_quality_service import BUNDLE_PATH, MODEL_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, load_bundle, train_and_export_artifacts


def export_r_artifacts() -> dict:
    bundle = load_bundle()
    metadata = bundle["metadata"]

    feature_scaler = bundle["feature_scaler"]
    pca_2d = bundle["pca_2d"]
    kmeans = bundle["kmeans"]
    model_scaler = bundle["model_scaler"]
    model = bundle["model"]

    serving_bundle = {
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "threshold": metadata["threshold"],
        "label_rule": metadata["label_rule"],
        "model_name": metadata["model_name"],
        "metrics": bundle["metrics"],
        "input_schema": metadata["input_schema"],
        "feature_scaler": {
            "mean": feature_scaler.mean_.tolist(),
            "scale": feature_scaler.scale_.tolist(),
        },
        "pca_2d": {
            "mean": pca_2d.mean_.tolist(),
            "components": pca_2d.components_.tolist(),
        },
        "kmeans": {
            "cluster_centers": kmeans.cluster_centers_.tolist(),
        },
        "model_scaler": {
            "mean": model_scaler.mean_.tolist(),
            "scale": model_scaler.scale_.tolist(),
        },
    }

    serving_bundle_path = BUNDLE_PATH.parent / "serving_bundle.json"
    xgb_model_path = BUNDLE_PATH.parent / "xgb_model.json"

    serving_bundle_path.write_text(json.dumps(serving_bundle, indent=2), encoding="utf-8")
    model.get_booster().save_model(xgb_model_path)

    return {
        "serving_bundle_path": str(serving_bundle_path),
        "xgb_model_path": str(xgb_model_path),
    }


if __name__ == "__main__":
    summary = train_and_export_artifacts(force=True)
    summary.update(export_r_artifacts())
    print(json.dumps(summary, indent=2))
