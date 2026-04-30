#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Exploratory data analysis for the Red Wine Quality project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_preparation import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    SUMMARY_TABLE_PATH,
    create_summary_table,
    ensure_directories,
    prepare_data,
    save_dataframe,
)


QUALITY_DISTRIBUTION_PATH = OUTPUTS_DIR / "quality_distribution_table.csv"
BINARY_DISTRIBUTION_PATH = OUTPUTS_DIR / "binary_target_distribution_table.csv"
GROUP_COMPARISON_PATH = OUTPUTS_DIR / "key_feature_group_comparison.csv"
OUTLIER_SUMMARY_PATH = OUTPUTS_DIR / "outlier_summary.csv"


def save_figure(file_path: Path) -> None:
    """Apply consistent save settings to all figures."""
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_count_plot_quality(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="quality", hue="quality", palette="Blues", legend=False)
    plt.title("Distribution of Red Wine Quality Scores")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    save_figure(FIGURES_DIR / "quality_distribution.png")


def create_count_plot_binary_target(df: pd.DataFrame) -> None:
    label_map = {0: "Not Good Quality", 1: "Good Quality"}
    plt.figure(figsize=(8, 5))
    plot_df = df.copy()
    plot_df["quality_group"] = plot_df["good_quality"].map(label_map)
    sns.countplot(
        data=plot_df,
        x="quality_group",
        order=["Not Good Quality", "Good Quality"],
        hue="quality_group",
        palette="Set2",
        legend=False,
    )
    plt.title("Distribution of Binary Wine Quality Target")
    plt.xlabel("Wine Quality Group")
    plt.ylabel("Count")
    save_figure(FIGURES_DIR / "binary_target_distribution.png")


def create_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Heatmap of Numeric Variables")
    save_figure(FIGURES_DIR / "correlation_heatmap.png")


def create_combined_boxplots(df: pd.DataFrame, features: list[str]) -> None:
    label_map = {0: "Not Good Quality", 1: "Good Quality"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        plot_df = df.copy()
        plot_df["quality_group"] = plot_df["good_quality"].map(label_map)
        sns.boxplot(
            data=plot_df,
            x="quality_group",
            y=feature,
            order=["Not Good Quality", "Good Quality"],
            hue="quality_group",
            dodge=False,
            palette="Set3",
            legend=False,
            ax=ax,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_map[0], label_map[1]], rotation=10)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Wine Quality Group")
        ax.set_ylabel(feature.replace("_", " ").title())

    fig.suptitle("Boxplots of Key Features by Wine Quality Group", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(FIGURES_DIR / "combined_boxplots.png")


def create_combined_feature_histograms(df: pd.DataFrame, features: list[str]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        sns.histplot(data=df, x=feature, kde=True, color="steelblue", ax=ax)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Count")

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle("Distributions of Numeric Physicochemical Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(FIGURES_DIR / "combined_distributions.png")


def create_scatter_plot(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["quality_group"] = plot_df["good_quality"].map(
        {0: "Not Good Quality", 1: "Good Quality"}
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="alcohol",
        y="volatile_acidity",
        hue="quality_group",
        palette="Set1",
        alpha=0.8,
    )
    plt.title("Alcohol vs. Volatile Acidity by Wine Quality Group")
    plt.xlabel("Alcohol")
    plt.ylabel("Volatile Acidity")
    plt.legend(title="Wine Quality Group")
    save_figure(FIGURES_DIR / "alcohol_vs_volatile_acidity.png")


def create_quality_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df["quality"].value_counts().sort_index().rename_axis("quality").reset_index(name="count")
    table["percentage"] = (table["count"] / table["count"].sum() * 100).round(2)
    save_dataframe(table, QUALITY_DISTRIBUTION_PATH)
    return table


def create_binary_target_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df["good_quality"]
        .value_counts()
        .sort_index()
        .rename_axis("good_quality")
        .reset_index(name="count")
    )
    table["percentage"] = (table["count"] / table["count"].sum() * 100).round(2)
    table["quality_group"] = table["good_quality"].map(
        {0: "Not Good Quality", 1: "Good Quality"}
    )
    table = table[["good_quality", "quality_group", "count", "percentage"]]
    save_dataframe(table, BINARY_DISTRIBUTION_PATH)
    return table


def create_group_comparison_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    grouped = df.groupby("good_quality")
    rows = []

    for feature in features:
        not_good_mean = grouped[feature].mean().get(0)
        good_mean = grouped[feature].mean().get(1)
        not_good_median = grouped[feature].median().get(0)
        good_median = grouped[feature].median().get(1)
        rows.append(
            {
                "feature": feature,
                "mean_not_good_quality": round(not_good_mean, 4),
                "mean_good_quality": round(good_mean, 4),
                "median_not_good_quality": round(not_good_median, 4),
                "median_good_quality": round(good_median, 4),
                "difference_in_mean_good_minus_not_good": round(
                    good_mean - not_good_mean, 4
                ),
            }
        )

    table = pd.DataFrame(rows)
    save_dataframe(table, GROUP_COMPARISON_PATH)
    return table


def create_outlier_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        rows.append(
            {
                "feature": feature,
                "Q1": round(q1, 4),
                "Q3": round(q3, 4),
                "IQR": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "number_of_outliers": outlier_count,
                "percentage_of_outliers": round(outlier_count / len(df) * 100, 2),
            }
        )

    table = pd.DataFrame(rows)
    save_dataframe(table, OUTLIER_SUMMARY_PATH)
    return table


def main() -> None:
    """Run the exploratory analysis workflow."""
    ensure_directories()
    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    raw_df, cleaned_df, duplicate_count_before = prepare_data(print_report=False)

    boxplot_features = [
        "alcohol",
        "volatile_acidity",
        "sulphates",
        "citric_acid",
        "density",
        "chlorides",
    ]
    histogram_features = [
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

    create_count_plot_quality(cleaned_df)
    create_count_plot_binary_target(cleaned_df)
    create_correlation_heatmap(cleaned_df)
    create_combined_boxplots(cleaned_df, boxplot_features)
    create_combined_feature_histograms(cleaned_df, histogram_features)
    create_scatter_plot(cleaned_df)

    summary_table_df = create_summary_table(raw_df, cleaned_df, duplicate_count_before)
    quality_distribution_table = create_quality_distribution_table(cleaned_df)
    binary_target_distribution_table = create_binary_target_distribution_table(cleaned_df)
    group_comparison_table = create_group_comparison_table(cleaned_df, boxplot_features)
    outlier_summary = create_outlier_summary(cleaned_df, histogram_features)

    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print("\nPart A summary table:")
    print(summary_table_df.to_string(index=False))
    print(f"Saved: {SUMMARY_TABLE_PATH}")

    print("\nOriginal quality distribution table:")
    print(quality_distribution_table.to_string(index=False))
    print(f"Saved: {QUALITY_DISTRIBUTION_PATH}")

    print("\nBinary target distribution table:")
    print(binary_target_distribution_table.to_string(index=False))
    print(f"Saved: {BINARY_DISTRIBUTION_PATH}")

    print("\nKey feature group comparison:")
    print(group_comparison_table.to_string(index=False))
    print(f"Saved: {GROUP_COMPARISON_PATH}")

    print("\nIQR-based outlier summary:")
    print(outlier_summary.to_string(index=False))
    print(f"Saved: {OUTLIER_SUMMARY_PATH}")
    print("\nOutliers are kept because they may represent real chemical measurements.")


if __name__ == "__main__":
    main()
