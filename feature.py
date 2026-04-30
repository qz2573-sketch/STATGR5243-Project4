#!/usr/bin/env python 3.11
# -*- coding: utf-8 -*-
# time: 2026/04/29
# name: Haowen Cui

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('data/cleaned_red_wine.csv')


def data_preview(df):
    """
       preview the dataset
       Parameters
       ---
       df : the dataset
       :return: df, X, y, feature_cols
    """
    print("  Loaded wine_cleaned.csv")
    print(f"  Shape  : {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # target = good_quality (binary), drop original quality score
    feature_cols = [c for c in df.columns
                  if c not in ('quality', 'good_quality')]
    X = df[feature_cols]
    y = df['good_quality']  # 0 = Not Good, 1 = Good

    print(f"\n  Target: good_quality")
    print(f"  Class distribution:")
    print(f"    Good Quality (1)    : {y.sum()} ({y.mean() * 100:.1f}%)")
    print(f"    Not Good Quality (0): {(1 - y).sum()} ({(1 - y.mean()) * 100:.1f}%)")

    return df, X, y, feature_cols


def run_pca(X_scaled, feature_cols, explained):
    """
        Fit PCA, print variance table, return 2D scores and loadings.
        Parameters
        ---
        X_scaled    : standardized feature matrix (numpy array)
        feature_cols: list of original feature column names
        explained   : explained variance ratio array from full PCA
        :return: X_pca2, loadings, explained, cumulative, n_80, n_95
    """
    cumulative = np.cumsum(explained)
    n_80 = int(np.argmax(cumulative >= 0.80)) + 1
    n_95 = int(np.argmax(cumulative >= 0.95)) + 1

    print(f"\n  Variance explained per component:")
    for i, (v, c) in enumerate(zip(explained, cumulative)):
        print(f"    PC{i + 1:02d}: {v * 100:5.2f}%   cumulative: {c * 100:6.2f}%")
    print(f"\n  → Components to explain 80%: {n_80}")
    print(f"  → Components to explain 95%: {n_95}")

    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca_2d.components_.T,
        index=feature_cols,
        columns=['PC1', 'PC2']
    ).round(3)
    print("\n  PCA Loadings (PC1 & PC2):")
    print(loadings.sort_values('PC1', ascending=False).to_string())

    return X_pca2, loadings, explained, cumulative, n_80, n_95


def plot_pca_2d(X_pca2, y, explained):
    """
        Plot 2D PCA scatter colored by good_quality binary label.
        Parameters
        ---
        X_pca2   : 2D PCA scores array, shape (n_samples, 2)
        y        : binary target series (0 = Not Good, 1 = Good)
        explained: explained variance ratio array from full PCA
        :return: None (saves figure to outputs/pca_2d.png)
    """
    colors_map = {0: '#e74c3c', 1: '#2ecc71'}
    labels_map = {0: 'Not Good Quality', 1: 'Good Quality'}

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X_pca2[mask, 0], X_pca2[mask, 1],
            c=colors_map[cls], label=labels_map[cls],
            alpha=0.6, s=45, edgecolors='white', linewidth=0.3
        )
    ax.set_xlabel(f'PC1 ({explained[0] * 100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained[1] * 100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA — 2D Projection by Wine Quality', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/pca_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/pca_2d.png")


def plot_pca_variance(explained, cumulative, n_80, n_95):
    """
        Plot scree plot and cumulative explained variance side by side.
        Parameters
        ---
        explained  : explained variance ratio array from full PCA
        cumulative : cumulative sum of explained variance ratio
        n_80       : number of components needed to explain 80% variance
        n_95       : number of components needed to explain 95% variance
        :return: None
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(range(1, len(explained) + 1), explained * 100,
                 'o-', color='#e74c3c', linewidth=2, markersize=6)
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance (%)', fontsize=12)
    axes[0].set_title('Scree Plot', fontsize=13, fontweight='bold')
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100,
                 's-', color='#3498db', linewidth=2, markersize=6)
    axes[1].axhline(80, color='orange', linestyle='--', linewidth=1.5, label='80%')
    axes[1].axhline(95, color='green', linestyle='--', linewidth=1.5, label='95%')
    axes[1].axvline(n_80, color='orange', linestyle=':', alpha=0.6)
    axes[1].axvline(n_95, color='green', linestyle=':', alpha=0.6)
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    axes[1].set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle('PCA — Explained Variance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/pca_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/pca_variance.png")


def find_best_k(X_scaled, k_range):
    """
        Run KMeans for each K in range, evaluate with silhouette score.
        Parameters
        ---
        X_scaled : standardized feature matrix (numpy array)
        k_range  : range of K values to try (e.g. range(2, 11))
        :return: inertias, sil_scores, best_k
    """
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
        print(f"  K={k}  Inertia={km.inertia_:8.1f}  Silhouette={sil_scores[-1]:.4f}")

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"\n  → Best K by Silhouette Score: {best_k}")
    return inertias, sil_scores, best_k


def plot_elbow(k_range, inertias):
    """
       Plot elbow method curve (inertia vs K).
       Parameters
       ---
       k_range  : range of K values used in find_best_k()
       inertias : list of inertia values for each K
       :return: None (saves figure to figures/elbow_method.png)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, 'o-', color='#e74c3c', linewidth=2, markersize=7)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (WSS)', fontsize=12)
    ax.set_title('Elbow Method — Optimal K Selection', fontsize=13, fontweight='bold')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('figures/elbow_method.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/elbow_method.png")


def plot_silhouette(k_range, sil_scores, best_k):
    """
        Plot silhouette scores for each K with best K highlighted.
        Parameters
        ---
        k_range    : range of K values used in find_best_k()
        sil_scores : list of silhouette scores for each K
        best_k     : optimal K selected by highest silhouette score
        :return: None (saves figure to figures/silhouette_scores.png)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), sil_scores, 's-', color='#3498db', linewidth=2, markersize=7)
    ax.axvline(best_k, color='green', linestyle='--', linewidth=1.5,
               label=f'Best K={best_k}')
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score by K', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('figures/silhouette_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/silhouette_scores.png")


def fit_kmeans(X_scaled, best_k):
    """
        Fit final KMeans with best K, return cluster labels.
        Parameters
        ---
        X_scaled : standardized feature matrix (numpy array)
        best_k   : optimal number of clusters from find_best_k()
        :return: labels (numpy array of cluster assignments)
    """
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels


def plot_clusters(X_pca2, cluster_labels, y, explained, best_k):
    """
    Side-by-side scatter: KMeans cluster labels vs actual quality label.
    Parameters
    ---
    X_pca2         : 2D PCA scores array, shape (n_samples, 2)
    cluster_labels : KMeans cluster assignment array from fit_kmeans()
    y              : binary target series (0 = Not Good, 1 = Good)
    explained      : explained variance ratio array from full PCA
    best_k         : number of clusters used in KMeans
    :return: None (saves figure to outputs/cluster_plot.png)
    """
    colors_map = {0: '#e74c3c', 1: '#2ecc71'}
    labels_map = {0: 'Not Good Quality', 1: 'Good Quality'}
    colors_k = plt.cm.Set1(np.linspace(0, 1, best_k))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for k in range(best_k):
        mask = cluster_labels == k
        axes[0].scatter(
            X_pca2[mask, 0], X_pca2[mask, 1],
            label=f'Cluster {k}', alpha=0.7, s=45,
            color=colors_k[k], edgecolors='white', linewidth=0.3
        )
    axes[0].set_xlabel(f'PC1 ({explained[0] * 100:.1f}%)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({explained[1] * 100:.1f}%)', fontsize=11)
    axes[0].set_title(f'K-Means Clusters (K={best_k})', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)

    for cls in [0, 1]:
        mask = y == cls
        axes[1].scatter(
            X_pca2[mask, 0], X_pca2[mask, 1],
            c=colors_map[cls], label=labels_map[cls],
            alpha=0.6, s=45, edgecolors='white', linewidth=0.3
        )
    axes[1].set_xlabel(f'PC1 ({explained[0] * 100:.1f}%)', fontsize=11)
    axes[1].set_ylabel(f'PC2 ({explained[1] * 100:.1f}%)', fontsize=11)
    axes[1].set_title('Actual Quality Label (PCA space)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.suptitle('K-Means Clusters vs Actual Quality Label', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/cluster_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/cluster_plot.png")


def get_cluster_profile(df, best_k):
    """
       Print and save cluster profile summary as a table image.
       Parameters
       ---
       df     : dataframe with cluster column and original features
       best_k : number of clusters, used to size the table figure
       :return: profile (DataFrame of per-cluster aggregated stats)
    """
    profile = df.groupby('cluster').agg(
        Count=('good_quality', 'count'),
        Good_Quality_Pct=('good_quality', lambda x: f"{x.mean() * 100:.1f}%"),
        Avg_Alcohol=('alcohol', 'mean'),
        Avg_VolAcid=('volatile_acidity', 'mean'),
        Avg_Sulphates=('sulphates', 'mean'),
        Avg_pH=('ph', 'mean'),
        Avg_CitricAcid=('citric_acid', 'mean'),
    ).round(3)

    print("\n  Cluster Profile:")
    print(profile.to_string())

    # Save as image
    fig, ax = plt.subplots(figsize=(14, 1 + best_k * 0.7))
    ax.axis('off')
    tbl = ax.table(
        cellText=profile.reset_index().values,
        colLabels=['Cluster'] + list(profile.columns),
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#ecf0f1')
    plt.title('Cluster Profile Summary', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figures/cluster_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/cluster_summary_table.png")

    return profile


def engineer_features(df, cluster_labels, X_pca2):
    """
    Create new features from domain knowledge, KMeans, and PCA results.
    Parameters
    ---
    df             : cleaned dataframe with original features
    cluster_labels : KMeans cluster assignment array from fit_kmeans()
    X_pca2         : 2D PCA scores array, shape (n_samples, 2)
    :return: df_feat (dataframe with original + 5 new features)
    """

    df_feat = df.copy()

    df_feat['free_to_total_SO2'] = (
            df_feat['free_sulfur_dioxide'] / df_feat['total_sulfur_dioxide']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    print("free_to_total_SO2     = free_SO2 / total_SO2")

    df_feat['total_acidity'] = (
            df_feat['fixed_acidity'] + df_feat['volatile_acidity']
    )
    print("total_acidity         = fixed_acidity + volatile_acidity")

    df_feat['alcohol_acidity_ratio'] = (
            df_feat['alcohol'] / df_feat['volatile_acidity']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    print("alcohol_acidity_ratio = alcohol / volatile_acidity")

    df_feat['cluster'] = cluster_labels
    print(f"cluster               = KMeans label")

    df_feat['PC1_score'] = X_pca2[:, 0]
    df_feat['PC2_score'] = X_pca2[:, 1]
    print("PC1_score, PC2_score  = PCA 2D scores")

    return df_feat


if __name__ == "__main__":
    df, X, y, feature_cols = data_preview(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("PCA")
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained = pca_full.explained_variance_ratio_

    X_pca2, loadings, explained, cumulative, n_80, n_95 = run_pca(
        X_scaled, feature_cols, explained
    )
    plot_pca_variance(explained, cumulative, n_80, n_95)
    plot_pca_2d(X_pca2, y, explained)

    print("K-Means Clustering")
    k_range = range(2, 11)
    inertias, sil_scores, best_k = find_best_k(X_scaled, k_range)
    plot_elbow(k_range, inertias)
    plot_silhouette(k_range, sil_scores, best_k)

    cluster_labels = fit_kmeans(X_scaled, best_k)
    df['cluster'] = cluster_labels
    plot_clusters(X_pca2, cluster_labels, y, explained, best_k)
    get_cluster_profile(df, best_k)

    print("Feature Engineering")
    df_feat = engineer_features(df, cluster_labels, X_pca2)

    df_feat.to_csv('data/wine_featured.csv', index=False)
    print(f"Saved: data/wine_featured.csv")