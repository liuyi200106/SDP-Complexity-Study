"""Bayesian hyperparameter optimization (Optuna) for the Stage 3 stacking ensemble,
run between Stage 2 (FIB-SMOTE) and Stage 3 (stacking).

The pipeline ablation found an untuned stacking ensemble statistically
indistinguishable from a single Random Forest, which leaves open whether the
ensemble was merely under-configured. This module searches the base learners' and
meta-learner's hyperparameters with Optuna/TPE, optimizing 3-fold CV F1 on the
already feature-selected, already balanced training data, before Stage 3 is fit.

Tuning runs once per dataset on the training fold passed in, not inside every fold
of the outer cross-validation. Fully nested tuning would multiply runtime by
roughly n_trials x inner_cv_folds, beyond this study's compute budget. The outer
train/test split is still respected for every reported metric; only the
hyperparameter values are chosen once. The paper reports this as a limitation.
"""
from __future__ import annotations

import optuna
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_from_params(params: dict, random_state: int = 42) -> StackingClassifier:
    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            random_state=random_state, n_jobs=4,
        )),
        ("xgb", XGBClassifier(
            n_estimators=params["xgb_n_estimators"],
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            eval_metric="logloss", random_state=random_state, n_jobs=4,
        )),
        ("svm", LinearSVC(C=params["svm_c"], random_state=random_state, dual="auto", max_iter=5000)),
        ("lr", LogisticRegression(C=params["lr_c"], max_iter=2000, random_state=random_state)),
    ]
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=params["meta_c"], max_iter=2000, random_state=random_state),
        cv=3,
        passthrough=False,
    )


def tune_stacking_ensemble(X, y, n_trials: int = 15, random_state: int = 42) -> dict:
    n_splits = min(3, y.value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "rf_n_estimators": trial.suggest_int("rf_n_estimators", 100, 300, step=50),
            "rf_max_depth": trial.suggest_int("rf_max_depth", 4, 20),
            "rf_min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5),
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300, step=50),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            "svm_c": trial.suggest_float("svm_c", 0.01, 10, log=True),
            "lr_c": trial.suggest_float("lr_c", 0.01, 10, log=True),
            "meta_c": trial.suggest_float("meta_c", 0.01, 10, log=True),
        }
        clf = _build_from_params(params, random_state)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def build_tuned_stacking_ensemble(best_params: dict, random_state: int = 42) -> StackingClassifier:
    return _build_from_params(best_params, random_state)
