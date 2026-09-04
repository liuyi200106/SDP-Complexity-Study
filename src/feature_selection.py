"""Stage 1: two-stage feature selection.

Step A (filter): drop near-zero-variance features, fit a Random Forest on the
remainder, and rank the survivors by mean |SHAP value| (TreeExplainer, exact for
tree ensembles rather than a sampling approximation). For each highly correlated
feature pair, the feature with the lower importance is dropped. Ranking by a
measure that already accounts for feature interactions is what distinguishes this
step from running the wrapper alone.

Step B (wrapper): RFECV with a Random Forest estimator selects the final subset
and produces the feature importances that Stage 2 consumes. That reuse is what
links the two stages.

The filter criterion is swappable (shap, gini, mutual_info, variance); RQ4
compares the four.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, VarianceThreshold, mutual_info_classif
from sklearn.model_selection import StratifiedKFold


def _shap_importance_from_estimator(estimator, X: pd.DataFrame) -> pd.Series:
    explainer = shap.TreeExplainer(estimator)
    raw = explainer.shap_values(X, check_additivity=False)

    if isinstance(raw, list):
        matrix = raw[1] if len(raw) > 1 else raw[0]
    elif raw.ndim == 3:
        matrix = raw[:, :, 1] if raw.shape[2] > 1 else raw[:, :, 0]
    else:
        matrix = raw

    return pd.Series(np.abs(matrix).mean(axis=0), index=X.columns)


def _shap_importance(X: pd.DataFrame, y: pd.Series, random_state: int, n_estimators: int = 150) -> pd.Series:
    baseline = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=4)
    baseline.fit(X, y)
    return _shap_importance_from_estimator(baseline, X)


def _filter_criterion_scores(
    X: pd.DataFrame, y: pd.Series, criterion: str, random_state: int
) -> pd.Series:
    """Importance scores used to break correlated-pair ties in the filter step.

    'shap' is the proposed method; the others are cheaper conventional
    alternatives, used for the Stage 1 control experiment (Section 4.6) that
    tests whether SHAP earns its computational cost here.
    """
    if criterion == "shap":
        return _shap_importance(X, y, random_state=random_state)
    if criterion == "gini":
        est = RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=4)
        est.fit(X, y)
        return pd.Series(est.feature_importances_, index=X.columns)
    if criterion == "mutual_info":
        return pd.Series(
            mutual_info_classif(X, y, random_state=random_state), index=X.columns
        )
    if criterion == "variance":
        return X.var()
    raise ValueError(f"unknown filter criterion: {criterion!r}")


def filter_stage(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.90,
    var_threshold: float = 1e-5,
    top_k: int | None = None,
    filter_criterion: str = "shap",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    X = X.copy()

    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X)
    X = X.loc[:, vt.get_support()]

    shap_importance = _filter_criterion_scores(X, y, filter_criterion, random_state)

    corr_matrix = X.corr().abs()
    cols = list(X.columns)
    to_drop: set[str] = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if c1 in to_drop or c2 in to_drop:
                continue
            if corr_matrix.loc[c1, c2] > corr_threshold:
                to_drop.add(c2 if shap_importance[c1] >= shap_importance[c2] else c1)

    remaining = [c for c in cols if c not in to_drop]
    shap_importance = shap_importance[remaining].sort_values(ascending=False)

    if top_k is not None and top_k < len(remaining):
        remaining = shap_importance.index[:top_k].tolist()

    return X[remaining], shap_importance[remaining]


def wrapper_stage(
    X: pd.DataFrame,
    y: pd.Series,
    min_features: int = 3,
    n_estimators: int = 200,
    step: int = 1,
    inner_cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[str], pd.Series, RandomForestClassifier]:
    estimator = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=4
    )
    n_splits_possible = min(inner_cv_splits, y.value_counts().min())
    if n_splits_possible < 2:
        estimator.fit(X, y)
        importances = _shap_importance_from_estimator(estimator, X).sort_values(ascending=False)
        return list(X.columns), importances, estimator
    cv = StratifiedKFold(
        n_splits=n_splits_possible, shuffle=True, random_state=random_state
    )

    selector = RFECV(
        estimator,
        step=step,
        cv=cv,
        scoring="f1",
        min_features_to_select=min(min_features, X.shape[1]),
        n_jobs=4,
    )
    selector.fit(X, y)
    selected = X.columns[selector.support_].tolist()

    final_estimator = RandomForestClassifier(
        n_estimators=n_estimators + 100, random_state=random_state, n_jobs=4
    )
    final_estimator.fit(X[selected], y)
    importances = _shap_importance_from_estimator(
        final_estimator, X[selected]
    ).sort_values(ascending=False)

    return selected, importances, final_estimator


def two_stage_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.90,
    filter_top_k: int | None = None,
    min_features: int = 3,
    wrapper_step: int = 1,
    wrapper_inner_cv_splits: int = 5,
    filter_criterion: str = "shap",
    random_state: int = 42,
):
    """Returns (selected_features, feature_importances, fitted_estimator, X_after_filter).

    filter_criterion selects how correlated-pair ties are broken in the filter
    step ('shap' is the proposed method; 'gini', 'mutual_info' and 'variance'
    are cheaper alternatives used by the Stage 1 control experiment).
    """
    X_filtered, _mi = filter_stage(
        X, y, corr_threshold=corr_threshold, top_k=filter_top_k,
        filter_criterion=filter_criterion, random_state=random_state,
    )
    selected, importances, estimator = wrapper_stage(
        X_filtered, y, min_features=min_features, step=wrapper_step,
        inner_cv_splits=wrapper_inner_cv_splits, random_state=random_state,
    )
    return selected, importances, estimator, X_filtered
