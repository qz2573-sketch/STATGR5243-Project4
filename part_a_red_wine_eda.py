#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Part A: Data Acquisition, Cleaning, and Exploratory Data Analysis

Dataset source:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Expected project structure after download:
project_root/
├── winequality-red.csv
├── data/
├── figures/
└── outputs/

Notes for data acquisition:
1. Download `winequality-red.csv` manually from the Kaggle page above and place it
   in the project root directory, OR
2. Use the Kaggle API after setting up Kaggle credentials first. Kaggle API setup
   usually requires placing `kaggle.json` in the correct local folder. Example:

   pip install kaggle
   kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009

   After downloading, unzip the archive and move `winequality-red.csv` to the
   project root.

3. Use `kagglehub` to download the latest version programmatically. Example:

   import kagglehub
   path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
   print("Path to dataset files:", path)

This script does not assume the file is already present. If the CSV is missing,
it first tries `kagglehub`, and if that does not work, it prints a clear message
explaining where to download it.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASET_LINK = "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "winequality-red.csv"
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUMMARY_PATH = PROJECT_ROOT / "partA_eda_summary.txt"
OUTPUT_CLEANED_PATH = DATA_DIR / "cleaned_red_wine.csv"
SUMMARY_TABLE_PATH = OUTPUTS_DIR / "partA_summary_table.csv"


def ensure_directories() -> None:
    """Create output directories if they do not already exist."""
    for directory in (DATA_DIR, FIGURES_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def print_missing_file_message() -> None:
    """Print clear download guidance when the dataset file is missing."""
    print("=" * 80)
    print("Dataset file not found: 'winequality-red.csv'")
    print("Please download the dataset from:")
    print(DATASET_LINK)
    print()
    print("Expected location after download:")
    print(f"  {DATASET_PATH}")
    print()
    print("Optional Kaggle API example:")
    print("  pip install kaggle")
    print("  kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009")
    print("  # unzip the downloaded file and move 'winequality-red.csv' into the project root")
    print()
    print("Important: Kaggle API download may require setting up Kaggle API credentials")
    print("first (for example, a valid 'kaggle.json' file).")
    print("=" * 80)


def try_download_with_kagglehub() -> Path | None:
    """
    Try downloading the dataset with kagglehub and copy the CSV to project root.

    KaggleHub may still require local Kaggle authentication depending on the
    environment and account setup.
    """
    print("\n'winequality-red.csv' was not found in the project root.")
    print("Attempting automatic download with kagglehub...")

    try:
        import kagglehub  # type: ignore
    except ImportError:
        print("kagglehub is not installed. Install it with: pip install kagglehub")
        return None

    try:
        # Download latest version from Kaggle.
        download_path = Path(
            kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
        )
        print(f"Path to dataset files: {download_path}")
    except Exception as exc:
        print("Automatic download with kagglehub failed.")
        print(f"Reason: {exc}")
        return None

    csv_candidates = list(download_path.rglob("winequality-red.csv"))
    if not csv_candidates:
        print("Downloaded dataset folder does not contain 'winequality-red.csv'.")
        return None

    shutil.copy2(csv_candidates[0], DATASET_PATH)
    print(f"Copied dataset to project root: {DATASET_PATH}")
    return DATASET_PATH


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load the red wine dataset with pandas.

    The original UCI/Kaggle file typically uses a semicolon delimiter.
    """
    df = pd.read_csv(csv_path, sep=";")

    # Defensive fallback in case the file uses commas instead.
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path)

    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with standardized column names."""
    cleaned_df = df.copy()
    cleaned_df.columns = [
        column.strip().lower().replace(" ", "_")
        for column in cleaned_df.columns
    ]
    return cleaned_df


def save_figure(file_path: Path) -> None:
    """Apply consistent save settings to all figures."""
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_count_plot_quality(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="quality", palette="Blues")
    plt.title("Distribution of Red Wine Quality Scores")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    save_figure(FIGURES_DIR / "quality_distribution.png")


def create_count_plot_binary_target(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="good_quality", palette="Set2")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Good Quality", "Good Quality"])
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


def create_boxplots(df: pd.DataFrame, features: list[str]) -> None:
    label_map = {0: "Not Good Quality", 1: "Good Quality"}

    for feature in features:
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(data=df, x="good_quality", y=feature, palette="Set3")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_map[0], label_map[1]])
        plt.title(f"{feature.replace('_', ' ').title()} by Wine Quality Group")
        plt.xlabel("Wine Quality Group")
        plt.ylabel(feature.replace("_", " ").title())
        save_figure(FIGURES_DIR / f"boxplot_{feature}.png")


def create_combined_boxplots(df: pd.DataFrame, features: list[str]) -> None:
    """Create one report-friendly figure containing all requested boxplots."""
    label_map = {0: "Not Good Quality", 1: "Good Quality"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        sns.boxplot(data=df, x="good_quality", y=feature, palette="Set3", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_map[0], label_map[1]], rotation=10)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Wine Quality Group")
        ax.set_ylabel(feature.replace("_", " ").title())

    fig.suptitle("Boxplots of Key Features by Wine Quality Group", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(FIGURES_DIR / "combined_boxplots.png")


def create_feature_histograms(df: pd.DataFrame, features: list[str]) -> None:
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=feature, kde=True, color="steelblue")
        plt.title(f"Distribution of {feature.replace('_', ' ').title()}")
        plt.xlabel(feature.replace("_", " ").title())
        plt.ylabel("Count")
        save_figure(FIGURES_DIR / f"distribution_{feature}.png")


def create_combined_feature_histograms(df: pd.DataFrame, features: list[str]) -> None:
    """Create one report-friendly figure containing all feature distributions."""
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
    label_map = {0: "Not Good Quality", 1: "Good Quality"}
    plot_df = df.copy()
    plot_df["quality_group"] = plot_df["good_quality"].map(label_map)

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


def format_series_counts(series: pd.Series) -> str:
    """Format counts and percentages for readable console/text output."""
    counts = series.value_counts().sort_index()
    percentages = series.value_counts(normalize=True).sort_index().mul(100)
    lines = []

    for value in counts.index:
        lines.append(
            f"  {value}: {counts[value]} ({percentages[value]:.2f}%)"
        )

    return "\n".join(lines)


def create_summary_table(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    duplicate_count_before: int,
) -> pd.DataFrame:
    """Create the Part A summary table requested for reporting."""
    predictor_df = cleaned_df.drop(columns=["quality", "good_quality"], errors="ignore")
    numeric_feature_count = predictor_df.select_dtypes(include="number").shape[1]

    good_quality_percentages = (
        cleaned_df["good_quality"].value_counts(normalize=True).sort_index().mul(100)
    )
    quality_min = cleaned_df["quality"].min()
    quality_max = cleaned_df["quality"].max()

    summary_table = pd.DataFrame(
        {
            "number_of_rows_before_cleaning": [raw_df.shape[0]],
            "number_of_rows_after_removing_duplicates": [cleaned_df.shape[0]],
            "number_of_duplicate_rows_removed": [duplicate_count_before],
            "number_of_numeric_features": [numeric_feature_count],
            "total_missing_values": [int(cleaned_df.isna().sum().sum())],
            "original_quality_score_range": [f"{quality_min} to {quality_max}"],
            "percentage_good_quality_1": [round(good_quality_percentages.get(1, 0.0), 2)],
            "percentage_good_quality_0": [round(good_quality_percentages.get(0, 0.0), 2)],
        }
    )
    return summary_table


def write_summary_file(
    missing_values_sum: int,
    duplicate_count_before: int,
    original_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    quality_distribution: pd.Series,
    good_quality_distribution: pd.Series,
) -> None:
    """Write a markdown-style text summary for teammates."""
    quality_lines = [
        f"- Quality {index}: {value}"
        for index, value in quality_distribution.sort_index().items()
    ]
    target_lines = [
        f"- good_quality = {index}: {value}"
        for index, value in good_quality_distribution.sort_index().items()
    ]

    summary_text = f"""# Part A EDA Summary

## Dataset source link
- {DATASET_LINK}

## Dataset shape
- Before cleaning: {original_shape}
- After duplicate removal: {cleaned_shape}

## Missing value summary
- Total missing values in cleaned dataset: {missing_values_sum}

## Duplicate count
- Duplicate rows before cleaning: {duplicate_count_before}

## Original quality distribution
{chr(10).join(quality_lines)}

## Binary target distribution
{chr(10).join(target_lines)}

## Short written findings from EDA
- The `good_quality` target is imbalanced, so later models should not rely only on accuracy.
- `alcohol` appears positively related to better quality.
- `volatile_acidity` appears negatively related to better quality.
- Feature scales differ noticeably, so scaling will likely be useful in later modeling.
- Later supervised models should report precision, recall, F1-score, and ROC-AUC in addition to accuracy.
"""

    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")


def main() -> None:
    """Run the full Part A workflow."""
    ensure_directories()

    if not DATASET_PATH.exists():
        downloaded_path = try_download_with_kagglehub()
        if downloaded_path is None or not downloaded_path.exists():
            print_missing_file_message()
            sys.exit(1)

    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    # ------------------------------------------------------------------
    # 1. Data acquisition and loading
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART A: DATA ACQUISITION, CLEANING, AND EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print(f"Dataset source: {DATASET_LINK}")
    print(f"Loading dataset from: {DATASET_PATH}")

    raw_df = load_dataset(DATASET_PATH)

    # ------------------------------------------------------------------
    # 2. Load and inspect the data
    # ------------------------------------------------------------------
    print("\n[1] INITIAL DATA INSPECTION")
    print("-" * 80)
    print(f"Dataset shape: {raw_df.shape}")
    print("\nFirst five rows:")
    print(raw_df.head())
    print("\nColumn names:")
    print(list(raw_df.columns))
    print("\nData types:")
    print(raw_df.dtypes)
    print("\nSummary statistics:")
    print(raw_df.describe(include="all"))

    missing_values_before = raw_df.isna().sum()
    duplicate_count_before = int(raw_df.duplicated().sum())
    quality_distribution = raw_df["quality"].value_counts().sort_index()

    print("\nMissing values by column:")
    print(missing_values_before)
    print("\nDuplicate rows:")
    print(duplicate_count_before)
    print("\nDistribution of the original quality variable:")
    print(quality_distribution)

    # ------------------------------------------------------------------
    # 3. Clean the dataset
    # ------------------------------------------------------------------
    print("\n[2] DATA CLEANING")
    print("-" * 80)
    cleaned_df = standardize_column_names(raw_df)

    print("\nStandardized column names:")
    print(list(cleaned_df.columns))

    missing_values_cleaned = cleaned_df.isna().sum()
    print("\nMissing values after column-name standardization:")
    print(missing_values_cleaned)

    duplicate_count_cleaned_before_drop = int(cleaned_df.duplicated().sum())
    print("\nDuplicate rows before removal in cleaned dataset:")
    print(duplicate_count_cleaned_before_drop)

    shape_before_drop = cleaned_df.shape
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    shape_after_drop = cleaned_df.shape

    cleaned_df.to_csv(OUTPUT_CLEANED_PATH, index=False)
    print(f"\nCleaned dataset saved to: {OUTPUT_CLEANED_PATH}")
    print(f"Shape before duplicate removal: {shape_before_drop}")
    print(f"Shape after duplicate removal:  {shape_after_drop}")

    # ------------------------------------------------------------------
    # 4. Construct a binary target variable
    # ------------------------------------------------------------------
    print("\n[3] BINARY TARGET CONSTRUCTION")
    print("-" * 80)
    cleaned_df["good_quality"] = (cleaned_df["quality"] >= 7).astype(int)

    good_quality_counts = cleaned_df["good_quality"].value_counts().sort_index()
    good_quality_percentages = (
        cleaned_df["good_quality"].value_counts(normalize=True).sort_index() * 100
    )

    target_summary_df = pd.DataFrame(
        {
            "count": good_quality_counts,
            "percentage": good_quality_percentages.round(2),
        }
    )
    print("Distribution of `good_quality`:")
    print(target_summary_df)

    cleaned_df.to_csv(OUTPUT_CLEANED_PATH, index=False)
    print(f"\nUpdated cleaned dataset with binary target saved to: {OUTPUT_CLEANED_PATH}")

    summary_table_df = create_summary_table(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        duplicate_count_before=duplicate_count_before,
    )
    summary_table_df.to_csv(SUMMARY_TABLE_PATH, index=False)

    print("\nPart A summary table:")
    print(summary_table_df.to_string(index=False))
    print(f"Saved summary table to: {SUMMARY_TABLE_PATH}")

    # ------------------------------------------------------------------
    # 5. Exploratory Data Analysis
    # ------------------------------------------------------------------
    print("\n[4] EXPLORATORY DATA ANALYSIS")
    print("-" * 80)

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
    create_boxplots(cleaned_df, boxplot_features)
    create_combined_boxplots(cleaned_df, boxplot_features)
    create_feature_histograms(cleaned_df, histogram_features)
    create_combined_feature_histograms(cleaned_df, histogram_features)
    create_scatter_plot(cleaned_df)

    print(f"Saved quality distribution plot to: {FIGURES_DIR / 'quality_distribution.png'}")
    print(
        "Saved binary target distribution plot to: "
        f"{FIGURES_DIR / 'binary_target_distribution.png'}"
    )
    print(f"Saved correlation heatmap to: {FIGURES_DIR / 'correlation_heatmap.png'}")
    print("Saved boxplots for selected features by good_quality.")
    print(f"Saved combined boxplot figure to: {FIGURES_DIR / 'combined_boxplots.png'}")
    print("Saved histograms with KDE for all required physicochemical features.")
    print(
        "Saved combined distribution figure to: "
        f"{FIGURES_DIR / 'combined_distributions.png'}"
    )
    print(
        "Saved scatter plot to: "
        f"{FIGURES_DIR / 'alcohol_vs_volatile_acidity.png'}"
    )

    # ------------------------------------------------------------------
    # 6. Summary output
    # ------------------------------------------------------------------
    print("\n[5] EDA SUMMARY")
    print("-" * 80)

    most_common_quality_scores = cleaned_df["quality"].value_counts().head(3)
    missing_values_exist = bool(cleaned_df.isna().sum().sum() > 0)

    print(f"Dataset source link: {DATASET_LINK}")
    print(f"Missing values exist: {missing_values_exist}")
    print(f"Number of duplicate rows before cleaning: {duplicate_count_before}")
    print(f"Shape before duplicate removal: {shape_before_drop}")
    print(f"Shape after duplicate removal: {shape_after_drop}")
    print("\nMost common quality scores:")
    print(most_common_quality_scores)
    print("\nClass balance of good_quality:")
    print(format_series_counts(cleaned_df["good_quality"]))
    print("\nKey modeling implications:")
    print("- The target is imbalanced.")
    print("- Alcohol appears positively related to quality.")
    print("- Volatile acidity appears negatively related to quality.")
    print("- Scaling may be needed later because features are on different scales.")
    print("- Later models should use metrics beyond accuracy, such as precision,")
    print("  recall, F1-score, and ROC-AUC.")

    write_summary_file(
        missing_values_sum=int(cleaned_df.isna().sum().sum()),
        duplicate_count_before=duplicate_count_before,
        original_shape=raw_df.shape,
        cleaned_shape=cleaned_df.shape,
        quality_distribution=cleaned_df["quality"].value_counts(),
        good_quality_distribution=cleaned_df["good_quality"].value_counts(),
    )
    print(f"\nSaved text summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
