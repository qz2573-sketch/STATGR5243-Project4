# GU5243 Project 04 — Red Wine Quality Prediction

End-to-end machine learning study of the UCI/Kaggle Red Wine Quality dataset.
The pipeline covers data cleaning, EDA, unsupervised exploration (PCA and
K-Means), feature engineering, supervised modeling (Logistic Regression,
Random Forest, XGBoost), and model interpretation (gain importance,
permutation importance, SHAP).

Dataset source:
<https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009>

## Team 3

| Member | UNI |
|--------|-----|
| Haowen Cui | hc3617 |
| Gujie Li | gl2957 |
| Maya Rubin | mr4459 |
| Qixian Zhou | qz2573 |

## Headline Result

The final model is a **tuned XGBoost classifier** (max_depth = 4,
n_estimators = 200, learning_rate = 0.05, scale_pos_weight = 6) predicting the
binary `good_quality` target (`quality >= 7`).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.772 | 0.360 | 0.865 | 0.508 | 0.893 | 0.579 |
| Random Forest | 0.846 | 0.449 | 0.595 | 0.512 | 0.886 | 0.519 |
| **XGBoost (tuned)** | **0.860** | **0.489** | 0.622 | **0.548** | **0.894** | **0.630** |

Across XGBoost gain importance, Random Forest impurity, Logistic Regression
standardized coefficients, permutation importance, and SHAP, the same drivers
emerge: **alcohol**, **sulphates**, and the engineered **alcohol_acidity_ratio**
are consistently the strongest predictors of "Good Quality" wine.

## Repository Structure

```
.
├── data_preparation.py            # Section 1 — load, dedupe, binary target
├── eda.py                         # Section 2 — EDA visuals & summary tables
├── feature.py                     # Sections 4–5 — PCA, K-Means, engineered features
├── preprocessing.py               # Section 5 — split, scale, sample weights
├── modeling and model comparison.py  # Section 6 — LR / RF / XGBoost + tuning
├── model_interpretation.py        # Sections 7–8 — gain, permutation, SHAP, ROC/PR
├── data/
│   ├── winequality-red.csv        # raw dataset
│   ├── cleaned_red_wine.csv
│   ├── wine_featured.csv
│   ├── train_processed.csv
│   └── test_processed.csv
├── figures/                       # all plots referenced in the report and slides
└── outputs/                       # summary tables (CSV) for the report
```

## Reproducing the Results

```bash
# 1. Section 1
python data_preparation.py

# 2. Section 2
python eda.py

# 3. Sections 4–5 (unsupervised + feature engineering)
python feature.py

# 4. Section 5 (preprocessing)
python preprocessing.py

# 5. Section 6 (supervised modeling)
python "modeling and model comparison.py"

# 6. Sections 7–8 (interpretation + final figures)
python model_interpretation.py
```

### Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
shap
```

On macOS, XGBoost requires the OpenMP runtime: `brew install libomp`.

## Shiny App

This repository now includes a Shiny web app for interactive red wine quality
prediction.

### Live app

- Deployed app: <https://statgr5243project.shinyapps.io/red-wine-quality-predictor/>

### App files

- `app.R` — Shiny user interface and server logic
- `build_artifacts.py` — exports deployable model artifacts from the Python pipeline
- `wine_quality_service.py` — Python-side training and artifact generation helpers
- `deploy_shinyapps.R` — helper script for publishing the app to `shinyapps.io`

### What the app does

- accepts the 11 raw wine chemistry inputs from the user
- recreates the engineered features used by the project pipeline
- predicts whether the wine is "Good" or "Bad" using the tuned XGBoost model
- shows prediction probabilities, model metrics, and input-field reference data

### Artifact export workflow

The original modeling workflow remains in Python. To refresh the web app model,
first export the serving artifacts:

```bash
python build_artifacts.py
```

This writes:

- `artifacts/serving_bundle.json` — scaler, PCA, K-Means, schema, metrics, metadata
- `artifacts/xgb_model.json` — exported XGBoost model used by `app.R`

### Local run instructions

1. Make sure the Python dependencies used by the modeling pipeline are installed.
2. Generate fresh Shiny serving artifacts:

   ```bash
   python build_artifacts.py
   ```

3. Install the R packages used by the app:

   ```r
   install.packages(c("shiny", "jsonlite", "xgboost"))
   ```

4. Start the app from the repository root:

   ```r
   shiny::runApp()
   ```

### shinyapps.io deployment

The deployed app serves predictions directly in R using the exported XGBoost
model and JSON metadata, which avoids Python environment issues on
`shinyapps.io`.

To deploy, set these environment variables in your R session or shell:

```r
Sys.setenv(
  SHINYAPPS_NAME = "your-account-name",
  SHINYAPPS_TOKEN = "your-token",
  SHINYAPPS_SECRET = "your-secret",
  SHINYAPP_APPNAME = "red-wine-quality-predictor"
)
source("deploy_shinyapps.R")
```

## Workflow Overview

1. **Data Acquisition & Cleaning (`data_preparation.py`).** Loads 1,599 rows,
   removes 240 duplicates, standardizes column names, constructs the binary
   `good_quality` label (`quality >= 7`) — final cleaned shape 1,359 × 12.
2. **EDA (`eda.py`).** Distributions, correlation heatmap, group comparisons,
   IQR outlier review, scatter of alcohol vs volatile acidity by quality.
3. **Unsupervised + Feature Engineering (`feature.py`).** PCA (6 components for
   80% variance), K-Means with K = 2 (silhouette = 0.205), six engineered
   features including `alcohol_acidity_ratio` (r = 0.37 with target),
   `free_to_total_SO2`, `total_acidity`, the K-Means cluster label, `PC1_score`
   and `PC2_score`.
4. **Preprocessing (`preprocessing.py`).** Stratified 80/20 split (1,087 / 272),
   `StandardScaler` fit on train only, balanced sample weights to address the
   86.5% / 13.5% class imbalance.
5. **Supervised Modeling (`modeling and model comparison.py`).** Trains
   Logistic Regression, Random Forest, and XGBoost; 5-fold CV; grid search on
   XGBoost (best: max_depth = 4, n_estimators = 200).
6. **Model Interpretation (`model_interpretation.py`).** Refits all three
   models, computes XGBoost gain importance, Random Forest impurity, Logistic
   Regression standardized coefficients, permutation importance on the test
   set, and SHAP TreeExplainer values for the final XGBoost. Outputs ROC and
   PR overlay curves and the consolidated metrics table.

## Member Contributions

| Member | Contribution |
|--------|--------------|
| Qixian Zhou (qz2573) | Sections 1–2: data acquisition, cleaning, and exploratory data analysis (`data_preparation.py`, `eda.py`). |
| Haowen Cui (hc3617) | Sections 4–5: PCA, K-Means, six engineered features (`feature.py`), and the preprocessing pipeline — stratified split, scaling, sample weighting (`preprocessing.py`). |
| Gujie Li (gl2957) | Section 6: supervised modeling (`modeling and model comparison.py`) — Logistic Regression, Random Forest, XGBoost, cross-validation, hyperparameter tuning. |
| Maya Rubin (mr4459) | Sections 7–9: model interpretation (`model_interpretation.py` — gain, impurity, coefficients, permutation, SHAP, ROC and PR overlays), final model selection writeup, limitations and conclusion, slide deck, and this README. |
