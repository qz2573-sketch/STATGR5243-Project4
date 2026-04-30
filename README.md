# GU5243 Project04
### Collaborators：
## Project Introduction
This project studies the Red Wine Quality dataset and builds an end-to-end machine learning workflow for data preparation, exploratory analysis, feature work, and predictive modeling. The repository currently includes a reproducible preprocessing pipeline, a separate exploratory data analysis script, saved figures, summary tables, and a cleaned dataset for downstream team tasks.

The dataset comes from the Kaggle/UCI Red Wine Quality source:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Main generated outputs in this repository include:
- `data_preparation.py`
- `eda.py`
- `feature.py`
- `preprocessing.py`
- `data/cleaned_red_wine.csv`
- `data/wine_featured.csv`
- `data/train_processed.csv`
- `data/test_processed.csv`
- `figures/quality_distribution.png`
- `figures/binary_target_distribution.png`
- `figures/correlation_heatmap.png`
- `figures/combined_boxplots.png`
- `figures/combined_distributions.png`
- `figures/alcohol_vs_volatile_acidity.png`
- `figures/pca_variance.png`
- `figures/pca_2d.png`
- `figures/elbow_method.png`
- `figures/silhouette_scores.png`
- `figures/cluster_plot.png`
- `figures/cluster_summary_table.png`
- `figures/class_distribution.png`
- `outputs/partA_summary_table.csv`
- `outputs/quality_distribution_table.csv`
- `outputs/binary_target_distribution_table.csv`
- `outputs/key_feature_group_comparison.csv`
- `outputs/outlier_summary.csv`

Workflow overview:
- `data_preparation.py` handles dataset loading, column standardization, duplicate removal, binary target construction, and cleaned data export.
- `eda.py` generates the required visualizations and summary tables from the cleaned dataset.
- `features.py` applies PCA and K-Means clustering for unsupervised exploration, engineers six new features from EDA findings and unsupervised outputs, and exports the enriched dataset as `wine_featured.csv`.
- `preprocessing.py` performs stratified train/test splitting, StandardScaler normalization, and sample weighting to address class imbalance, and exports the processed train and test sets ready for modeling.

To reproduce the current results, run:
- `python data_preparation.py`
- `python eda.py`
- `python features.py`
- `python preprocessing.py`
  
Initial exploratory findings suggest that the dataset has no missing values, duplicate rows were identified and removed, and wine quality scores are concentrated around 5 and 6. The binary quality label is imbalanced, alcohol shows a positive relationship with better quality, and volatile acidity shows a negative relationship. Sulphates and citric acid also show positive relationships with quality. Several variables are skewed and measured on different scales, indicating that later modeling steps should consider scaling and evaluation metrics beyond accuracy.

PCA revealed that six components are needed to explain 80% of the total variance, and that Good Quality and Not Good Quality wines overlap substantially in the reduced feature space, supporting the use of nonlinear models. K-Means clustering with K = 2 identified two groups that differ nearly threefold in good_quality rate, validating that the physicochemical features contain quality-relevant structure. Six engineered features were added based on EDA patterns and unsupervised outputs, with alcohol_acidity_ratio showing the strongest correlation with good_quality (r = 0.37). The final processed dataset contains 17 features across 1,087 training and 272 test observations, with sample weights applied to handle the 86.5% / 13.5% class imbalance.
