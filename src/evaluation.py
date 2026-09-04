"""Shared evaluation metrics for all experiments.

Centralised so that every experiment reports the same metric set: F1, ROC-AUC,
PR-AUC (average precision), MCC, G-mean, precision, recall and specificity.

PR-AUC is reported alongside ROC-AUC because it ignores true negatives, which
makes it more informative than ROC-AUC when the positive class is rare.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["F1", "AUC", "PRAUC", "MCC", "Gmean", "Precision", "Recall", "Specificity"]


def g_mean(y_true, y_pred) -> float:
    rp = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rn = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return (rp * rn) ** 0.5


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """y_score: probability of the positive class (or a decision function)."""
    y_true = np.asarray(y_true)
    n_classes = len(np.unique(y_true))
    return {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score) if n_classes > 1 else np.nan,
        "PRAUC": average_precision_score(y_true, y_score) if n_classes > 1 else np.nan,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Gmean": g_mean(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def score_estimator(clf, X_test, y_test) -> dict:
    """Fit-free scoring helper: predicts with clf and returns the full metric set."""
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        y_score = clf.decision_function(X_test)
    return compute_metrics(y_test, y_pred, y_score)
