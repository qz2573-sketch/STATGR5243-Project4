#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data preparation for the Red Wine Quality project."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd


DATASET_LINK = "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "winequality-red.csv"
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_CLEANED_PATH = DATA_DIR / "cleaned_red_wine.csv"
SUMMARY_PATH = PROJECT_ROOT / "partA_eda_summary.txt"
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
    print("Optional kagglehub example:")
    print("  pip install kagglehub")
    print("  import kagglehub")
    print("  path = kagglehub.dataset_download('uciml/red-wine-quality-cortez-et-al-2009')")
    print("  print('Path to dataset files:', path)")
    print("=" * 80)


def try_download_with_kagglehub() -> Path | None:
    """Try downloading the dataset with kagglehub and copy the CSV to project root."""
    print("\n'winequality-red.csv' was not found in the project root.")
    print("Attempting automatic download with kagglehub...")

    try:
        import kagglehub  # type: ignore
    except ImportError:
        print("kagglehub is not installed. Install it with: python -m pip install kagglehub")
        return None

    try:
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
    """Load the red wine dataset with pandas."""
    df = pd.read_csv(csv_path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path)
    return df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe unless the target file is temporarily locked."""
    try:
        df.to_csv(output_path, index=False)
    except PermissionError:
        if output_path.exists():
            print(f"Warning: could not overwrite locked file, keeping existing file: {output_path}")
            return
        raise


def save_text(content: str, output_path: Path) -> None:
    """Write text unless the target file is temporarily locked."""
    try:
        output_path.write_text(content, encoding="utf-8")
    except PermissionError:
        if output_path.exists():
            print(f"Warning: could not overwrite locked file, keeping existing file: {output_path}")
            return
        raise


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with standardized column names."""
    cleaned_df = df.copy()
    cleaned_df.columns = [
        column.strip().lower().replace(" ", "_")
        for column in cleaned_df.columns
    ]
    return cleaned_df


def create_summary_table(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    duplicate_count_before: int,
) -> pd.DataFrame:
    """Create the summary table used in the project outputs."""
    physicochemical_features = [
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
    good_quality_percentages = (
        cleaned_df["good_quality"].value_counts(normalize=True).sort_index().mul(100)
    )

    return pd.DataFrame(
        {
            "number_of_rows_before_cleaning": [raw_df.shape[0]],
            "number_of_rows_after_removing_duplicates": [cleaned_df.shape[0]],
            "number_of_duplicate_rows_removed": [duplicate_count_before],
            "number_of_numeric_physicochemical_features": [len(physicochemical_features)],
            "total_missing_values": [int(cleaned_df.isna().sum().sum())],
            "original_quality_score_range": [
                f"{raw_df['quality'].min()} to {raw_df['quality'].max()}"
            ],
            "percentage_good_quality_1": [round(good_quality_percentages.get(1, 0.0), 2)],
            "percentage_good_quality_0": [round(good_quality_percentages.get(0, 0.0), 2)],
        }
    )


def write_summary_file(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    duplicate_count_before: int,
) -> None:
    """Write a short text summary for later reporting."""
    quality_distribution = cleaned_df["quality"].value_counts().sort_index()
    good_quality_distribution = cleaned_df["good_quality"].value_counts().sort_index()

    quality_lines = [
        f"- Quality {index}: {value}"
        for index, value in quality_distribution.items()
    ]
    target_lines = [
        f"- good_quality = {index}: {value}"
        for index, value in good_quality_distribution.items()
    ]

    summary_text = f"""# EDA Summary

## Dataset source link
- {DATASET_LINK}

## Dataset shape
- Before cleaning: {raw_df.shape}
- After duplicate removal: {cleaned_df.shape}

## Missing value summary
- Total missing values in cleaned dataset: {int(cleaned_df.isna().sum().sum())}

## Duplicate count
- Duplicate rows before cleaning: {duplicate_count_before}

## Original quality distribution
{chr(10).join(quality_lines)}

## Binary target distribution
{chr(10).join(target_lines)}

## Short written findings from EDA
- The good_quality target is imbalanced, so later models should not rely only on accuracy.
- Alcohol appears positively related to better quality.
- Volatile acidity appears negatively related to better quality.
- Feature scales differ noticeably, so scaling will likely be useful in later modeling.
- Later supervised models should report precision, recall, F1-score, and ROC-AUC in addition to accuracy.
"""
    save_text(summary_text, SUMMARY_PATH)


def prepare_data(print_report: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Load, inspect, clean, and save the red wine dataset.

    Returns the raw dataframe, cleaned dataframe, and duplicate count before removal.
    """
    ensure_directories()

    if not DATASET_PATH.exists():
        downloaded_path = try_download_with_kagglehub()
        if downloaded_path is None or not downloaded_path.exists():
            print_missing_file_message()
            sys.exit(1)

    raw_df = load_dataset(DATASET_PATH)
    cleaned_df = standardize_column_names(raw_df)
    duplicate_count_before = int(cleaned_df.duplicated().sum())
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    cleaned_df["good_quality"] = (cleaned_df["quality"] >= 7).astype(int)

    save_dataframe(cleaned_df, OUTPUT_CLEANED_PATH)
    summary_table_df = create_summary_table(raw_df, cleaned_df, duplicate_count_before)
    save_dataframe(summary_table_df, SUMMARY_TABLE_PATH)
    write_summary_file(raw_df, cleaned_df, duplicate_count_before)

    if print_report:
        print("\n" + "=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)
        print(f"Dataset source: {DATASET_LINK}")
        print(f"Dataset shape: {raw_df.shape}")
        print("\nFirst five rows:")
        print(raw_df.head())
        print("\nColumn names:")
        print(list(raw_df.columns))
        print("\nData types:")
        print(raw_df.dtypes)
        print("\nSummary statistics:")
        print(raw_df.describe(include="all"))
        print("\nMissing values by column:")
        print(raw_df.isna().sum())
        print("\nDuplicate rows before cleaning:")
        print(duplicate_count_before)
        print("\nDistribution of the original quality variable:")
        print(raw_df["quality"].value_counts().sort_index())
        print("\nDistribution of `good_quality`:")
        print(
            pd.DataFrame(
                {
                    "count": cleaned_df["good_quality"].value_counts().sort_index(),
                    "percentage": (
                        cleaned_df["good_quality"]
                        .value_counts(normalize=True)
                        .sort_index()
                        .mul(100)
                        .round(2)
                    ),
                }
            )
        )
        print(f"\nCleaned dataset saved to: {OUTPUT_CLEANED_PATH}")
        print(f"Summary table saved to: {SUMMARY_TABLE_PATH}")
        print(f"Text summary saved to: {SUMMARY_PATH}")

    return raw_df, cleaned_df, duplicate_count_before


def main() -> None:
    """Run the standalone data preparation workflow."""
    prepare_data(print_report=True)


if __name__ == "__main__":
    main()
