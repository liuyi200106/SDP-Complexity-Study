"""Stage 3: stacking ensemble over heterogeneous base learners.

Base learners are a Random Forest, XGBoost, a linear SVM and Logistic Regression.
A Logistic Regression meta-learner is trained on out-of-fold predictions from the
base learners; sklearn's StackingClassifier performs that internal split, so the
meta-learner never sees predictions from a model fitted on the same rows.

LinearSVC (liblinear) is used rather than a kernel SVC: at the size of JM1
(about 11k rows) a kernel SVM's O(n^2) to O(n^3) training cost makes a 5-fold CV
sweep impractical, and LinearSVC's decision_function is what StackingClassifier
requires from a base learner that lacks predict_proba.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def build_stacking_ensemble(random_state: int = 42, internal_cv: int = 3) -> StackingClassifier:
    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=4)),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state, n_jobs=4,
        )),
        ("svm", LinearSVC(random_state=random_state, dual="auto", max_iter=5000)),
        ("lr", LogisticRegression(max_iter=2000, random_state=random_state)),
    ]

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        cv=internal_cv,
        passthrough=False,
    )
