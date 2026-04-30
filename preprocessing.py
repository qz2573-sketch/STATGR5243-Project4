#!/usr/bin/env python 3.11
# -*- coding: utf-8 -*-
# time: 2026/04/30
# name: Haowen Cui

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

df = pd.read_csv('data/wine_featured.csv')

def data_preview(df):
    """
       preview the dataset
       Parameters
       ---
       df : the dataset
       :return: df, X, y, feature_cols
    """
    # target = good_quality (binary), drop original quality score
    feature_cols = [c for c in df.columns
                  if c not in ('quality', 'good_quality')]
    X = df[feature_cols]
    y = df['good_quality']  # 0 = Not Good, 1 = Good

    print(f"  Loaded data/wine_featured.csv")
    print(f"  Shape        : {df.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"\n  Target: good_quality")
    print(f"  Class distribution:")
    print(f"    Good Quality (1)    : {y.sum()} ({y.mean() * 100:.1f}%)")
    print(f"    Not Good Quality (0): {(1 - y).sum()} ({(1 - y.mean()) * 100:.1f}%)")

    return df, X, y, feature_cols

def split_data(X, y, test_size=0.2, random_state=42):
    """
        Stratified train/test split to preserve class ratio.
        Parameters
        ---
        X            : feature matrix
        y            : binary target series
        test_size    : proportion of test set (default 0.2)
        random_state : random seed for reproducibility
        :return: X_train, X_test, y_train, y_test
        """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # keeps good_quality ratio balanced in both sets
    )

    print(f"\n  Train size : {X_train.shape[0]}")
    print(f"  Test size  : {X_test.shape[0]}")
    print(f"\n  Train quality distribution:")
    print(f"    Good (1): {y_train.sum()} ({y_train.mean() * 100:.1f}%)")
    print(f"    Not  (0): {(1 - y_train).sum()} ({(1 - y_train.mean()) * 100:.1f}%)")
    print(f"\n  Test quality distribution:")
    print(f"    Good (1): {y_test.sum()} ({y_test.mean() * 100:.1f}%)")
    print(f"    Not  (0): {(1 - y_test).sum()} ({(1 - y_test.mean()) * 100:.1f}%)")

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, feature_cols):
    """
    Fit StandardScaler on train set only, apply to both sets.
    Fitting only on train prevents data leakage from test set.
    Parameters
    ---
    X_train      : training feature matrix
    X_test       : test feature matrix
    feature_cols : list of feature column names
    :return: X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()

    # fit ONLY on train, then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"     Train mean (should be ~0): {X_train_scaled.mean():.6f}")
    print(f"     Train std  (should be ~1): {X_train_scaled.std():.6f}")

    return X_train_scaled, X_test_scaled, scaler

def compute_weights(y_train):
    """
    Compute sample weights to handle class imbalance.
    Minority class (good_quality=1) gets higher weight.
    Parameters
    ---
    y_train : training target series
    :return: sample_weights (numpy array)
    """
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    # show average weight per class
    weight_df = pd.DataFrame({
        'good_quality': y_train,
        'weight': sample_weights
    }).groupby('good_quality')['weight'].mean().round(4)

    print(f"     Weight range: {sample_weights.min():.3f} – {sample_weights.max():.3f}")
    print(f"\n     Average weight by class:")
    print(f"       Not Good Quality (0): {weight_df[0]:.4f}  ← lower")
    print(f"       Good Quality     (1): {weight_df[1]:.4f}  ← higher (minority)")

    return sample_weights

def plot_class_distribution(y_train, y_test):
    """
    Bar chart comparing class distribution in train and test sets.
    Parameters
    ---
    y_train : training target series
    y_test  : test target series
    :return: None (saves figure to figures/class_distribution.png)
    """
    fig, axes=plt.subplots(1, 2, figsize=(10, 5))
    labels=['Not Good (0)', 'Good (1)']
    colors=['#e74c3c', '#2ecc71']

    for ax, y, title in zip(axes,
                            [y_train, y_test],
                            ['Train Set', 'Test Set']):
        counts = y.value_counts().sort_index()
        bars = ax.bar(labels, counts.values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)
        ax.set_ylim(0, counts.max() * 1.2)
        # add count + percentage labels on bars
        for bar, count in zip(bars, counts.values):
            pct = count / len(y) * 100
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', fontsize=10)

    plt.suptitle('Class Distribution — Train vs Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/class_distribution.png")

def save_processed(X_train_scaled, X_test_scaled,
                   y_train, y_test,
                   sample_weights, feature_cols):
    """
    Save scaled train/test sets as CSV for modeling step.
    Parameters
    ---
    X_train_scaled : scaled training feature matrix
    X_test_scaled  : scaled test feature matrix
    y_train        : training target series
    y_test         : test target series
    sample_weights : computed sample weights for training
    feature_cols   : list of feature column names
    :return: None (saves to data/train_processed.csv and data/test_processed.csv)
    """
    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df['good_quality'] = y_train.values
    train_df['sample_weight'] = sample_weights

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df['good_quality'] = y_test.values

    train_df.to_csv('data/train_processed.csv', index=False)
    test_df.to_csv('data/test_processed.csv', index=False)

    print(f"data/train_processed.csv  {train_df.shape}")
    print(f"data/test_processed.csv   {test_df.shape}")

if __name__=="__main__":
    df, X, y, feature_cols = data_preview(df)

    print("Split data")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Standard Scaling")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, feature_cols
    )

    print("Sample Weighting")
    sample_weights = compute_weights(y_train)

    plot_class_distribution(y_train, y_test)

    print("Saving Processed Data")
    save_processed(X_train_scaled, X_test_scaled,
                   y_train, y_test,
                   sample_weights, feature_cols)