#!/usr/bin/env python 3.11
# -*- coding: utf-8 -*-
# time: 2026/05/01
# name: Maya Rubin
# Part D: Model Interpretation

"""
Loads the processed train/test sets, refits the three supervised models
described in Part C (Logistic Regression, Random Forest, tuned XGBoost),
and produces all interpretation artifacts for the final report and slides:

  - XGBoost gain-based feature importance
  - Random Forest impurity-based feature importance
  - Logistic Regression standardized coefficients
  - Permutation importance on the test set (model-agnostic)
  - SHAP TreeExplainer values for the final XGBoost model
  - ROC and Precision-Recall curves overlaying all three models
  - A consolidated metrics comparison table

All figures are saved to figures/ and tables to outputs/.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    train = pd.read_csv("data/train_processed.csv")
    test = pd.read_csv("data/test_processed.csv")

    sample_weight = train["sample_weight"].values

    drop_cols_train = ["good_quality", "sample_weight"]
    X_train = train.drop(columns=drop_cols_train)
    y_train = train["good_quality"].values

    X_test = test.drop(columns=["good_quality"])
    y_test = test["good_quality"].values

    feature_names = list(X_train.columns)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Test  positive rate: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, sample_weight, feature_names


def fit_models(X_train, y_train, sample_weight):
    """Fit the three models exactly as in Part C, including tuned XGBoost."""
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train, sample_weight=sample_weight)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)

    # Final tuned XGBoost (max_depth=4, n_estimators=200) from Part C grid search.
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=6,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    return {"Logistic Regression": lr, "Random Forest": rf, "XGBoost (tuned)": xgb_model}


# ---------------------------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------------------------

def evaluate_models(models, X_test, y_test):
    rows = []
    probs = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        probs[name] = (y_pred, y_prob)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob),
            }
        )
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, digits=3))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    metrics_df = pd.DataFrame(rows).set_index("model").round(3)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_comparison_table.csv"))
    print("\nSaved: outputs/metrics_comparison_table.csv")
    print(metrics_df)
    return metrics_df, probs


# ---------------------------------------------------------------------------
# Plots: ROC + PR curves
# ---------------------------------------------------------------------------

def plot_roc_curves(probs, y_test):
    plt.figure(figsize=(7, 6))
    for name, (_, y_prob) in probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_curves_all_models.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/roc_curves_all_models.png")


def plot_pr_curves(probs, y_test):
    plt.figure(figsize=(7, 6))
    for name, (_, y_prob) in probs.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, lw=2, label=f"{name} (AP = {ap:.3f})")
    base_rate = y_test.mean()
    plt.axhline(base_rate, color="k", linestyle="--", alpha=0.4,
                label=f"Baseline = {base_rate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Test Set")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pr_curves_all_models.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/pr_curves_all_models.png")


# ---------------------------------------------------------------------------
# Feature importance — XGBoost gain
# ---------------------------------------------------------------------------

def plot_xgb_importance(model, feature_names):
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    # Booster keys may be either f0,f1,... or the original feature names
    # (depends on whether a DataFrame was passed at fit time). Handle both.
    importance = []
    for i, name in enumerate(feature_names):
        if name in score:
            importance.append(score[name])
        else:
            importance.append(score.get(f"f{i}", 0.0))
    importance = np.array(importance, dtype=float)
    # Normalize to a percentage of total gain so the scale is interpretable.
    if importance.sum() > 0:
        importance_pct = 100 * importance / importance.sum()
    else:
        importance_pct = importance

    order = np.argsort(importance_pct)
    plt.figure(figsize=(8, 7))
    plt.barh(np.array(feature_names)[order], importance_pct[order],
             color="#2E86AB", edgecolor="white")
    plt.xlabel("Gain importance (% of total)")
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance_xgb.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/feature_importance_xgb.png")

    df = pd.DataFrame({"feature": feature_names, "gain_pct": importance_pct})
    df = df.sort_values("gain_pct", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "xgb_feature_importance.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Feature importance — Random Forest impurity
# ---------------------------------------------------------------------------

def plot_rf_importance(model, feature_names):
    importance = model.feature_importances_
    order = np.argsort(importance)
    plt.figure(figsize=(8, 7))
    plt.barh(np.array(feature_names)[order], importance[order],
             color="#2ECC71", edgecolor="white")
    plt.xlabel("Mean decrease in impurity")
    plt.title("Random Forest Feature Importance (Impurity)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance_rf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/feature_importance_rf.png")

    df = pd.DataFrame({"feature": feature_names, "impurity": importance})
    df = df.sort_values("impurity", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "rf_feature_importance.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Logistic regression coefficients (features were standardized in Part B,
# so coefficient magnitudes are directly comparable)
# ---------------------------------------------------------------------------

def plot_lr_coefficients(model, feature_names):
    coef = model.coef_.ravel()
    order = np.argsort(coef)
    colors = ["#E74C3C" if c < 0 else "#3498DB" for c in coef[order]]

    plt.figure(figsize=(8, 7))
    plt.barh(np.array(feature_names)[order], coef[order], color=colors, edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardized coefficient")
    plt.title("Logistic Regression Coefficients\n(positive → higher P(Good Quality))")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "logistic_coefficients.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/logistic_coefficients.png")

    df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coef, "abs_coef": np.abs(coef)}
    ).sort_values("abs_coef", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "logistic_coefficients.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Permutation importance (model-agnostic) on the held-out test set
# ---------------------------------------------------------------------------

def plot_permutation_importance(model, X_test, y_test, feature_names,
                                model_label="XGBoost", filename="permutation_importance.png"):
    result = permutation_importance(
        model, X_test, y_test,
        scoring="f1", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1,
    )
    means = result.importances_mean
    stds = result.importances_std
    order = np.argsort(means)

    plt.figure(figsize=(8, 7))
    plt.barh(
        np.array(feature_names)[order],
        means[order],
        xerr=stds[order],
        color="#9B59B6",
        edgecolor="white",
        ecolor="gray",
    )
    plt.xlabel("Mean drop in F1 when feature is permuted")
    plt.title(f"Permutation Importance — {model_label} (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figures/{filename}")

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_drop_f1": means,
        "std_drop_f1": stds,
    }).sort_values("mean_drop_f1", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "permutation_importance_xgb.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# SHAP analysis for the tuned XGBoost
# ---------------------------------------------------------------------------

def shap_analysis(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Beeswarm summary
    plt.figure()
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/shap_summary.png")

    # Mean |SHAP| bar
    plt.figure()
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/shap_bar.png")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    df = df.sort_values("mean_abs_shap", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "shap_mean_abs.csv"), index=False)

    # Dependence plots for the top two features (gives directionality detail)
    top2 = df["feature"].head(2).tolist()
    for feat in top2:
        idx = feature_names.index(feat)
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            idx, shap_values, X_test,
            feature_names=feature_names, show=False,
        )
        plt.tight_layout()
        safe = feat.replace("/", "_")
        plt.savefig(
            os.path.join(FIG_DIR, f"shap_dependence_{safe}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        print(f"Saved: figures/shap_dependence_{safe}.png")

    return df


# ---------------------------------------------------------------------------
# Combined importance side-by-side (rank table for the report)
# ---------------------------------------------------------------------------

def combined_importance_table(xgb_df, rf_df, lr_df, perm_df, shap_df):
    out = (
        xgb_df.rename(columns={"gain_pct": "xgb_gain_pct"})
        .merge(rf_df.rename(columns={"impurity": "rf_impurity"}), on="feature")
        .merge(
            lr_df[["feature", "coefficient"]].rename(
                columns={"coefficient": "lr_coef"}
            ),
            on="feature",
        )
        .merge(
            perm_df[["feature", "mean_drop_f1"]].rename(
                columns={"mean_drop_f1": "perm_drop_f1"}
            ),
            on="feature",
        )
        .merge(shap_df, on="feature")
    )
    out = out.sort_values("xgb_gain_pct", ascending=False)
    out.to_csv(os.path.join(OUT_DIR, "combined_importance_table.csv"), index=False)
    print("Saved: outputs/combined_importance_table.csv")
    return out


def main():
    sns.set_theme(style="whitegrid")
    X_train, X_test, y_train, y_test, sample_weight, feature_names = load_data()

    models = fit_models(X_train, y_train, sample_weight)

    metrics_df, probs = evaluate_models(models, X_test, y_test)

    plot_roc_curves(probs, y_test)
    plot_pr_curves(probs, y_test)

    xgb_model = models["XGBoost (tuned)"]
    rf_model = models["Random Forest"]
    lr_model = models["Logistic Regression"]

    xgb_df = plot_xgb_importance(xgb_model, feature_names)
    rf_df = plot_rf_importance(rf_model, feature_names)
    lr_df = plot_lr_coefficients(lr_model, feature_names)
    perm_df = plot_permutation_importance(
        xgb_model, X_test, y_test, feature_names,
        model_label="XGBoost (tuned)", filename="permutation_importance.png",
    )
    shap_df = shap_analysis(xgb_model, X_test, feature_names)

    combined_importance_table(xgb_df, rf_df, lr_df, perm_df, shap_df)

    print("\nAll interpretation artifacts written to figures/ and outputs/.")


if __name__ == "__main__":
    main()
