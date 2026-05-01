# Sections 7–9 (Member D: Maya Rubin)

These sections continue directly from Part C (Section 6). All figures referenced
below are produced by `model_interpretation.py` and saved in `figures/`. The
underlying tables are saved in `outputs/`.

---

## 7. Model Interpretation

To move from "which model performs best" to "why does it predict what it
predicts," we examined feature importance from four complementary angles:
XGBoost gain, Random Forest impurity, Logistic Regression standardized
coefficients, and model-agnostic permutation importance on the held-out test
set. We then used SHAP (TreeExplainer) on the final tuned XGBoost to confirm
the directionality of each feature's effect.

### 7.1 XGBoost Feature Importance (Gain)

The gain-based importance for the tuned XGBoost ranks the top contributors as
**alcohol (24.6%)**, **alcohol_acidity_ratio (13.1%)**, **sulphates (8.1%)**,
**PC2_score (6.7%)**, and **pH (5.7%)**. Two of the top four features were
engineered in Part B, which directly validates the feature-engineering choices:
`alcohol_acidity_ratio` encoded the strongest EDA pattern as a single variable,
and `PC2_score` summarized the sulfur-dioxide-versus-alcohol contrast. The
K-Means cluster label received zero gain, indicating that the tree splits can
rederive the same partition from the underlying chemistry.

*Figure 13. XGBoost feature importance (gain).*

### 7.2 Logistic Regression Coefficients

Because all features were standardized in Part B, the logistic regression
coefficients are directly comparable in magnitude. The largest positive drivers
of P(Good Quality) are **alcohol (+0.95)**, **sulphates (+0.59)**, and
**free_to_total_SO2 (+0.44)**; the largest negative drivers are
**volatile_acidity (−0.47)**, **PC2_score (−0.40)**, and **chlorides (−0.40)**.
This is exactly the chemical story foreshadowed by the EDA in Part A: more
alcohol and more sulphates raise the odds of a "Good Quality" label, while
volatile acidity (a marker of vinegar-like compounds) and chlorides lower
them. The signs on the engineered features also make sense — `PC2_score`
contrasts sulfur dioxide against alcohol, so a negative coefficient agrees with
the positive coefficient on alcohol.

*Figure 14. Logistic regression standardized coefficients.*

### 7.3 Permutation Importance

Gain and impurity importances can be biased toward high-cardinality features,
so we computed permutation importance on the test set, scoring on F1 (the
metric we care about under class imbalance) with 20 repeats. The top features
by mean F1 drop are **alcohol (0.179)**, **sulphates (0.126)**,
**alcohol_acidity_ratio (0.050)**, and **pH (0.039)**. This independent,
model-agnostic check agrees with the XGBoost gain ranking on the top three
features and confirms that alcohol alone accounts for the largest drop in
predictive performance when scrambled.

*Figure 15. Permutation importance for tuned XGBoost on the test set.*

### 7.4 SHAP Analysis

SHAP values decompose each individual prediction into per-feature
contributions, giving us both global and local interpretability. The mean
|SHAP| ranking is **alcohol (1.06)**, **sulphates (0.87)**,
**alcohol_acidity_ratio (0.54)**, and **PC2_score (0.46)**. The beeswarm plot
makes the directionality explicit: high alcohol values push predictions toward
"Good Quality," and high sulphates do the same; high `PC2_score` (more sulfur
dioxide relative to alcohol) pushes predictions away from "Good Quality." The
SHAP dependence plot for `alcohol` shows a roughly monotonic positive effect
that flattens at very high alcohol levels, consistent with diminishing returns
above ~12% ABV.

*Figure 16. SHAP summary (beeswarm) for tuned XGBoost.*
*Figure 17. SHAP mean |SHAP| bar chart.*
*Figure 18. SHAP dependence plot for alcohol.*

### 7.5 Cross-Method Agreement

| Rank | XGB Gain | RF Impurity | Permutation | LR \|coef\| | SHAP |
|------|----------|-------------|-------------|------------|------|
| 1 | alcohol | alcohol | alcohol | alcohol | alcohol |
| 2 | alcohol_acidity_ratio | PC2_score | sulphates | sulphates | sulphates |
| 3 | sulphates | alcohol_acidity_ratio | alcohol_acidity_ratio | volatile_acidity | alcohol_acidity_ratio |
| 4 | PC2_score | sulphates | pH | free_to_total_SO2 | PC2_score |

Across five different importance views, **alcohol** and **sulphates** are
unanimously top-ranked, and the engineered features `alcohol_acidity_ratio`
and `PC2_score` consistently rank in the top four. This convergence is the
strongest evidence that the model is picking up real chemistry rather than
spurious patterns.

---

## 8. Final Model Selection and Justification

We selected the **tuned XGBoost** model (max_depth = 4, n_estimators = 200,
learning_rate = 0.05, scale_pos_weight = 6) as the final model. The choice is
justified on both quantitative and qualitative grounds.

### 8.1 Quantitative Comparison (Test Set, n = 272)

| Model | Accuracy | Precision (Good) | Recall (Good) | F1 (Good) | ROC-AUC | PR-AUC |
|-------|----------|------------------|---------------|-----------|---------|--------|
| Logistic Regression | 0.772 | 0.360 | **0.865** | 0.508 | 0.893 | 0.579 |
| Random Forest | 0.846 | 0.449 | 0.595 | 0.512 | 0.886 | 0.519 |
| **XGBoost (tuned)** | **0.860** | **0.489** | 0.622 | **0.548** | **0.894** | **0.630** |

XGBoost has the highest F1, ROC-AUC, and accuracy. The most decisive metric
under our 13.5% class imbalance is **PR-AUC**, where XGBoost (0.630) clearly
outperforms Random Forest (0.519) and Logistic Regression (0.579). PR-AUC is
more honest than ROC-AUC for imbalanced problems because it ignores the
abundant true negatives. Logistic Regression has higher recall than XGBoost,
but only because it predicts "Good Quality" much more aggressively (precision
0.36 means roughly two of every three positive predictions are wrong).

*Figure 19. ROC curves for all three models.*
*Figure 20. Precision-Recall curves for all three models.*

### 8.2 Qualitative Considerations

- **Robustness to imbalance.** XGBoost's `scale_pos_weight = 6` directly
  addresses the 6.4× class imbalance computed in Part B and is supported by
  cross-validated F1 stability (Section 6.3).
- **Interpretability.** Although XGBoost is a non-linear ensemble, the SHAP
  decomposition makes per-prediction explanations available. The cross-method
  agreement in Section 7.5 also shows that the XGBoost importances align with
  the linear LR coefficients, which is reassuring.
- **Captures non-linearity.** The PCA scatter from Part B and the EDA scatter
  in Part A both showed substantial overlap between the two classes — no
  linear boundary separates Good from Not Good. XGBoost's trees naturally
  model the interactions (alcohol × sulphates × volatile acidity) that a
  linear model cannot.
- **Generalization.** The tuned XGBoost's 5-fold CV F1 (~0.51) closely matches
  its test F1 (0.55), so we are not overfitting.

### 8.3 Final Retrain

The selected configuration was retrained on the full 1,087-row training set
using the standardized features and stratified split from Part B. The trained
model is what produced all evaluation numbers and SHAP values reported above.

---

## 9. Limitations and Conclusion

### 9.1 Limitations

- **Sample size.** 1,359 cleaned observations is modest for a tree ensemble.
  Confidence intervals on F1 and PR-AUC are correspondingly wide.
- **Class imbalance.** Only 184 wines are labeled "Good Quality" in total;
  37 land in the test set. A handful of misclassifications visibly moves
  metrics. We mitigated this with stratified splitting, balanced sample
  weights, and `scale_pos_weight`, but more high-quality examples would
  help.
- **Subjective target.** The original `quality` score is a sensory rating,
  not an objective measurement. Taster bias and inter-rater disagreement put
  a ceiling on how well any chemistry-only model can perform.
- **Generalization scope.** The data is red wine from one region and one
  source. The model should not be applied to white wine, sparkling wine,
  or wine from other regions without retraining.
- **Binary simplification.** Collapsing 0–10 quality into a binary label
  loses information, especially the distinction between average and
  slightly-above-average wines. An ordinal regression would preserve the
  full target structure.
- **No causal claims.** Feature importance shows what helps prediction; it
  does not say that increasing alcohol or sulphates causes higher quality.
  Confounders (vintage, grape variety, winemaking style) are not in the
  dataset.

### 9.2 Future Work

- Try a stacked ensemble (LR + RF + XGBoost) with a meta-learner — the three
  base models have complementary error profiles (LR over-predicts positives,
  RF under-predicts, XGBoost is balanced).
- Probability calibration via isotonic regression to make threshold tuning
  more honest under the heavy imbalance.
- Ordinal regression on the original 0–10 score to preserve target structure.
- Replicate the pipeline on the white-wine dataset and check whether the
  same features dominate.

### 9.3 Conclusion

We built an end-to-end pipeline — cleaning, EDA, unsupervised exploration,
feature engineering, and supervised modeling — that predicts whether a red
wine receives a "Good Quality" sensory rating from its physicochemical
profile. The tuned XGBoost classifier achieves F1 = 0.55, ROC-AUC = 0.89, and
PR-AUC = 0.63 on the held-out test set, a clear improvement over both the
linear and Random Forest baselines. Across five different interpretability
methods, the same chemistry rises to the top: more alcohol, more sulphates,
and lower volatile acidity make a wine more likely to be rated "Good." This
agreement between unsupervised structure (K-Means and PCA), exploratory
patterns (Section 2), engineered features (Section 5), and supervised
importance (Section 7) is the project's strongest result — the model isn't
just performing well, it is performing well for the right reasons.

---

## Member Contributions

| Member | UNI | Contribution |
|--------|-----|--------------|
| Qixian Zhou | qz2573 | Section 1 (Data Acquisition & Cleaning) and Section 2 (EDA): dataset sourcing, duplicate removal, binary target construction, distribution and correlation visualizations (`data_preparation.py`, `eda.py`). |
| Haowen Cui | hc3617 | Section 4 (Unsupervised Learning) and Section 5 (Feature Engineering & Preprocessing): PCA, K-Means, six engineered features, stratified split, scaling, sample weighting (`feature.py`, `preprocessing.py`). |
| Gujie Li | gl2957 | Section 6 (Supervised Modeling): Logistic Regression, Random Forest, and XGBoost training, cross-validation, hyperparameter tuning, and model comparison (`modeling and model comparison.py`). |
| **Maya Rubin** | **mr4459** | **Section 7 (Model Interpretation), Section 8 (Final Model Selection), Section 9 (Limitations & Conclusion). Built `model_interpretation.py` (gain / impurity / coefficient / permutation / SHAP). Produced all interpretation figures, ROC and PR overlays, the metrics comparison table, the slide deck, and the GitHub README.** |
