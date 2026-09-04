"""Adaptive Balanced Forest (ABF).

BalancedRandomForestClassifier trains each tree on an independently resampled
subset, so its trees differ from one another and the ensemble gains the usual
bagging diversity. Running FIB-SMOTE once and fitting a single Random Forest on
the result provides no such diversity: every tree sees the same balanced dataset.

ABF applies the same mechanism to FIB-SMOTE. For each of n_estimators trees it
draws a class-stratified bootstrap of the original imbalanced training data, runs
FIB-SMOTE on that bootstrap with a fresh seed, and fits one decision tree.
Predictions are averaged by soft voting. This keeps FIB-SMOTE's feature-weighted,
borderline-aware interpolation while adding the resampling diversity that
BalancedRF relies on.

The bootstrap covers the majority class only. Bootstrapping the minority class as
well would interpolate between near-duplicate points, which adds noise rather than
diversity; FIB-SMOTE's own seeded choice of interpolation partner and ratio
already varies the minority side across trees.

Also defines ABF-Hybrid, which additionally undersamples the majority class in
each bag before FIB-SMOTE runs. ABF-Hybrid is the variant reported as the Stage 3
result in the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fib_smote import FIBSmote


class AdaptiveBalancedForest:
    def __init__(
        self,
        feature_importances: pd.Series,
        n_estimators: int = 200,
        k_neighbors: int = 5,
        sampling_ratio: float = 1.0,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        self.feature_importances = feature_importances
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.sampling_ratio = sampling_ratio
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaptiveBalancedForest":
        rng = check_random_state(self.random_state)
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        counts = y.value_counts()
        minority_label, majority_label = counts.idxmin(), counts.idxmax()
        maj_idx = y[y == majority_label].index.to_numpy()
        min_idx = y[y == minority_label].index.to_numpy()

        self.estimators_ = []
        for i in range(self.n_estimators):
            seed = rng.randint(0, 1_000_000)
            tree_rng = check_random_state(seed)

            boot_maj = tree_rng.choice(maj_idx, size=len(maj_idx), replace=True)
            boot_idx = np.concatenate([boot_maj, min_idx])
            X_boot, y_boot = X.iloc[boot_idx], y.iloc[boot_idx]

            fib = FIBSmote(
                feature_importances=self.feature_importances,
                k_neighbors=self.k_neighbors,
                sampling_ratio=self.sampling_ratio,
                random_state=seed,
            )
            X_bal, y_bal = fib.fit_resample(X_boot, y_boot)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt", random_state=seed,
            )
            tree.fit(X_bal, y_bal)
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AdaptiveBalancedForestHybrid(AdaptiveBalancedForest):
    """ABF variant: lightly undersample the majority class in each bag
    (to majority_undersample_ratio x minority count) before FIB-SMOTE
    brings the bag up to full 1:1 balance, combining both mechanisms
    instead of relying on oversampling alone."""

    def __init__(self, *args, majority_undersample_ratio: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.majority_undersample_ratio = majority_undersample_ratio

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        counts = y.value_counts()
        minority_label, majority_label = counts.idxmin(), counts.idxmax()
        maj_idx = y[y == majority_label].index.to_numpy()
        min_idx = y[y == minority_label].index.to_numpy()
        target_maj_size = min(len(maj_idx), int(self.majority_undersample_ratio * len(min_idx)))

        self.estimators_ = []
        for i in range(self.n_estimators):
            seed = rng.randint(0, 1_000_000)
            tree_rng = check_random_state(seed)

            boot_maj = tree_rng.choice(maj_idx, size=target_maj_size, replace=True)
            boot_idx = np.concatenate([boot_maj, min_idx])
            X_boot, y_boot = X.iloc[boot_idx], y.iloc[boot_idx]

            fib = FIBSmote(
                feature_importances=self.feature_importances, k_neighbors=self.k_neighbors,
                sampling_ratio=self.sampling_ratio, random_state=seed,
            )
            X_bal, y_bal = fib.fit_resample(X_boot, y_boot)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt", random_state=seed,
            )
            tree.fit(X_bal, y_bal)
            self.estimators_.append(tree)
        return self
