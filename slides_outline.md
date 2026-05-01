# Slide Deck — Red Wine Quality Prediction (10 minutes)

12 slides, ≈45 sec/slide. Drop the figures from `figures/` directly onto each
slide. Talking points are in plain English — read them, do not put the full
sentences on the slide. Keep slides visual-heavy.

---

## Slide 1 — Title (15 sec)
**Title:** Predicting Red Wine Quality from Chemistry
**Subtitle:** An End-to-End Machine Learning Study
**Team 3:** Haowen Cui (hc3617), Gujie Li (gl2957), Maya Rubin (mr4459), Qixian Zhou (qz2573)
GU5243 — Project 4

*Talk:* Brief framing — "We trained a model to predict whether a red wine
will be rated 'high quality' from its physicochemical profile."

---

## Slide 2 — The Question (45 sec)
**Bullets:**
- Can we predict a wine's sensory quality from chemistry alone?
- Target: binary `good_quality` (sensory score ≥ 7)
- Why it matters: thousands of physicochemical measurements, expensive sensory panels — model could pre-screen wines

*Talk:* Position the problem. Sensory tasting is expensive and subjective; if
chemistry alone predicts quality, we save time and money.

---

## Slide 3 — Data & Cleaning (45 sec)
**Bullets:**
- UCI / Kaggle Red Wine Quality dataset
- 1,599 → 1,359 rows after removing 240 duplicates
- 11 chemical features + sensory score
- Constructed binary target: `quality >= 7` → "Good Quality" (13.5% positive)

*Visual:* `figures/quality_distribution.png` and `figures/binary_target_distribution.png`

*Talk:* Highlight the heavy class imbalance — only ~14% are "good." This is
the single most important constraint for the rest of the project.

---

## Slide 4 — EDA Highlights (60 sec)
**Bullets:**
- Alcohol — strongest **positive** correlation with quality
- Volatile acidity — strongest **negative** correlation
- Sulphates and citric acid also positive
- Several features skewed and on different scales → scaling needed

*Visual:* `figures/correlation_heatmap.png` (left) + `figures/alcohol_vs_volatile_acidity.png` (right)

*Talk:* The picture from EDA already tells us most of the chemistry story.
Good wines tend to have higher alcohol and lower volatile acidity, but the
two classes overlap — a single feature isn't enough.

---

## Slide 5 — Unsupervised: PCA (45 sec)
**Bullets:**
- Standardized 11 features → PCA
- Need 6 components for 80% variance → features carry independent information
- 2D projection: classes overlap heavily → linear separation insufficient

*Visual:* `figures/pca_variance.png` (left) + `figures/pca_2d.png` (right)

*Talk:* PCA tells us we cannot aggressively compress; it also visually
confirms that linear models will struggle.

---

## Slide 6 — Unsupervised: K-Means (60 sec)
**Bullets:**
- K-Means on standardized features, K = 2 to 10
- Silhouette peaks at K = 2 (score = 0.205)
- Cluster 1 has 22.9% Good Quality vs Cluster 0 at 7.3% — **3× difference**
- K-Means recovered quality structure **without seeing the label**

*Visual:* `figures/cluster_plot.png` + small inset of `figures/silhouette_scores.png`

*Talk:* This is a strong validation — unsupervised clustering, blind to the
target, found groups that line up with quality.

---

## Slide 7 — Feature Engineering (45 sec)
**Bullets:**
- Six new features added:
  - `alcohol_acidity_ratio` — best engineered correlation r = 0.37
  - `free_to_total_SO2`, `total_acidity`
  - K-Means cluster label
  - `PC1_score`, `PC2_score`
- Final feature matrix: 17 features

*Visual:* small table of the six engineered features and their correlation with target

*Talk:* The engineered ratios collapse known EDA patterns into single
informative variables, and the unsupervised outputs feed back into the
supervised model.

---

## Slide 8 — Models Compared (60 sec)
**Bullets:**
- Three supervised models with stratified 80/20 split, scaled features, balanced sample weights
- Logistic Regression — linear baseline
- Random Forest — non-linear, ensemble
- XGBoost — gradient-boosted trees, tuned via grid search

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.51 | 0.89 | 0.58 |
| Random Forest | 0.51 | 0.89 | 0.52 |
| **XGBoost (tuned)** | **0.55** | **0.89** | **0.63** |

*Visual:* table on slide; reference confusion-matrix images if asked

*Talk:* All three are competitive on ROC-AUC, but XGBoost wins on F1 and on
**PR-AUC**, which is the honest metric under 13.5% imbalance.

---

## Slide 9 — Final Model: ROC + PR Curves (60 sec)
**Bullets:**
- Tuned XGBoost: max_depth = 4, n_estimators = 200, scale_pos_weight = 6
- Test F1 = 0.55, ROC-AUC = 0.89, PR-AUC = 0.63
- 5-fold CV ≈ test F1 → no overfitting

*Visual:* `figures/roc_curves_all_models.png` (left) + `figures/pr_curves_all_models.png` (right)

*Talk:* PR curve shows XGBoost dominates the precision-recall trade-off across
the full range — most decisive evidence under imbalance.

---

## Slide 10 — Why It Predicts What It Predicts (75 sec)
**Bullets:**
- Five different importance methods agree on the top features:

| | Top 1 | Top 2 | Top 3 |
|---|---|---|---|
| XGBoost gain | alcohol | alcohol_acidity_ratio | sulphates |
| Permutation | alcohol | sulphates | alcohol_acidity_ratio |
| SHAP | alcohol | sulphates | alcohol_acidity_ratio |
| LR coef | alcohol | sulphates | volatile_acidity |

- Engineered `alcohol_acidity_ratio` ranks top-3 in 3/4 methods → feature engineering paid off
- High alcohol + high sulphates push toward "Good," high volatile acidity pushes away

*Visual:* `figures/shap_summary.png` (beeswarm) on left, `figures/feature_importance_xgb.png` on right

*Talk:* The convergence across methods is the project's strongest result —
the model isn't just accurate, it's accurate for the right chemical reasons.

---

## Slide 11 — Limitations & Future Work (45 sec)
**Bullets:**
- Modest sample size (1,359 rows; 184 positives)
- Sensory target is subjective → ceiling on chemistry-only accuracy
- Generalization: red wine only, one source
- Binary collapse loses ordinal information
- Future: stacked ensemble, calibrated probabilities, ordinal regression on full 0–10 score, replicate on white wine

*Talk:* Be honest about scope. The model works for what it claims to do, not
beyond that.

---

## Slide 12 — Conclusion + Contributions (30 sec)
**Bullets:**
- Tuned XGBoost predicts "Good Quality" red wine with F1 = 0.55, PR-AUC = 0.63
- Same chemistry signal across EDA → unsupervised → engineered features → supervised importance: **alcohol, sulphates, and the alcohol-to-acidity ratio drive the prediction**
- Contributions:
  - Qixian — Data + EDA (Sections 1–2)
  - Haowen — Unsupervised + Preprocessing (Sections 4–5)
  - Gujie — Supervised Modeling (Section 6)
  - Maya — Interpretation, Selection, Limitations, README, Slides (Sections 7–9)

*Talk:* Close with the unifying insight: every layer of the pipeline tells
the same story.

---

## Q&A Backup Slides (optional, do not present unless asked)

**B1.** Confusion matrices for all three models — `figures/logistic.png`, `figures/random forest.png`, `figures/XGBoost.png`
**B2.** SHAP dependence plot for alcohol — `figures/shap_dependence_alcohol.png`
**B3.** K-Means cluster summary table — `figures/cluster_summary_table.png`
**B4.** Cross-validation F1 stability — XGBoost test F1 = 0.55, CV F1 ≈ 0.51
