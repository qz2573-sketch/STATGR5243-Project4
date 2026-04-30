# GU5243 Project04
### Collaborators：
## Project Introduction
This project studies the Red Wine Quality dataset and builds an end-to-end machine learning workflow for data preparation, exploratory analysis, feature work, and predictive modeling. The repository currently includes a reproducible preprocessing pipeline, a separate exploratory data analysis script, saved figures, summary tables, and a cleaned dataset for downstream team tasks.

The dataset comes from the Kaggle/UCI Red Wine Quality source:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Main generated outputs in this repository include:
- `data_preparation.py`
- `eda.py`
- `data/cleaned_red_wine.csv`
- `figures/quality_distribution.png`
- `figures/binary_target_distribution.png`
- `figures/correlation_heatmap.png`
- `figures/combined_boxplots.png`
- `figures/combined_distributions.png`
- `figures/alcohol_vs_volatile_acidity.png`
- `outputs/partA_summary_table.csv`
- `outputs/quality_distribution_table.csv`
- `outputs/binary_target_distribution_table.csv`
- `outputs/key_feature_group_comparison.csv`
- `outputs/outlier_summary.csv`

Workflow overview:
- `data_preparation.py` handles dataset loading, column standardization, duplicate removal, binary target construction, and cleaned data export.
- `eda.py` generates the required visualizations and summary tables from the cleaned dataset.

To reproduce the current results, run:
- `python data_preparation.py`
- `python eda.py`

Initial exploratory findings suggest that the dataset has no missing values, duplicate rows were identified and removed, and wine quality scores are concentrated around 5 and 6. The binary quality label is imbalanced, alcohol shows a positive relationship with better quality, and volatile acidity shows a negative relationship. Sulphates and citric acid also show positive relationships with quality. Several variables are skewed and measured on different scales, indicating that later modeling steps should consider scaling and evaluation metrics beyond accuracy.
