
"""FIB-SMOTE component ablation and comparison against established resampling
baselines.

FIB-SMOTE combines four mechanisms that exist separately in the literature:
weighted distance, Borderline-SMOTE, an ADASYN-style quota and Tomek cleaning.
This experiment tests whether each one contributes, and how the assembled method
compares against the baselines it draws on.

All ten configurations share the same Stage 1 selected features per fold and the
same downstream Random Forest; only the resampling step varies, which isolates the
resampling method from feature selection.

  NoResample       Stage 1 features, no balancing (reference floor)
  CostSensitive    no oversampling, balanced class weights
  SMOTE            imbalanced-learn's SMOTE
  BorderlineSMOTE  imbalanced-learn's Borderline-SMOTE
  ADASYN           imbalanced-learn's ADASYN
  SMOTETomek       imbalanced-learn's SMOTE with Tomek cleaning
  FIB_ShapOnly     FIB-SMOTE with only the weighted distance enabled
  FIB_Borderline   + borderline/safe/noise classification
  FIB_Adaptive     + adaptive (ADASYN-style) quota allocation
  FIB_Full         + Tomek cleaning; the full method, as used in C_Reference
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42


def g_mean(y_true, y_pred) -> float:
    rp = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rn = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return (rp * rn) ** 0.5


def score(y_test, y_pred, y_proba) -> dict:
    auc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan
    return {
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Gmean": g_mean(y_test, y_pred),
    }


def fit_eval_rf(X_train, y_train, X_test, y_test, class_weight=None) -> dict:
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4, class_weight=class_weight)
    clf.fit(X_train, y_train)
    return score(y_test, clf.predict(X_test), clf.predict_proba(X_test)[:, 1])


def safe_resample_standardized(sampler, X: pd.DataFrame, y):
    """Fit a StandardScaler on X (always the current training fold, by
    construction of every caller here), resample in standardized space so
    raw-scale features (e.g. Halstead effort in the hundreds of thousands)
    don't dominate the resampler's internal distance computation regardless
    of the resampler's own logic, then inverse-transform the result back to
    the original feature scale (Random Forest is scale-invariant, but this
    keeps X_bal on the same scale as the untouched X_test for consistency).
    imbalanced-learn resamplers can fail on folds with very few minority
    instances relative to their internal k_neighbors; skip gracefully rather
    than crash the whole run.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    try:
        X_bal_scaled, y_bal = sampler.fit_resample(X_scaled, y)
    except Exception as e:
        return None, str(e)
    X_bal = pd.DataFrame(scaler.inverse_transform(X_bal_scaled), columns=X.columns)
    return X_bal, y_bal


def run_dataset(name: str, path: Path) -> pd.DataFrame:
    X, y, _ = load_defect_dataset(path)
    n_splits = min(N_SPLITS, y.value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        selected, importances, _est, _ = two_stage_feature_selection(
            X_train, y_train, corr_threshold=0.90, min_features=3,
            wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
        )
        X_train_sel, X_test_sel = X_train[selected], X_test[selected]
        min_count = y_train.value_counts().min()
        k_smote = max(1, min(5, min_count - 1))

        def add_row(tag, res):
            rows.append({"dataset": name, "fold": fold_i, "config": tag, **res})

        add_row("NoResample", fit_eval_rf(X_train_sel, y_train, X_test_sel, y_test))
        add_row("CostSensitive", fit_eval_rf(X_train_sel, y_train, X_test_sel, y_test, class_weight="balanced"))

        for tag, sampler in [
            ("SMOTE", SMOTE(k_neighbors=k_smote, random_state=RANDOM_STATE)),
            ("BorderlineSMOTE", BorderlineSMOTE(k_neighbors=k_smote, random_state=RANDOM_STATE)),
            ("ADASYN", ADASYN(n_neighbors=k_smote, random_state=RANDOM_STATE)),
            ("SMOTETomek", SMOTETomek(random_state=RANDOM_STATE)),
        ]:
            X_bal, y_bal = safe_resample_standardized(sampler, X_train_sel, y_train)
            if X_bal is None:
                print(f"  [skip] {name} fold{fold_i} {tag}: {y_bal}", flush=True)
                continue
            add_row(tag, fit_eval_rf(X_bal, y_bal, X_test_sel, y_test))

        ablation_configs = [
            ("FIB_ShapOnly", dict(use_feature_weighting=True, use_borderline=False, use_adaptive_quota=False, clean=False)),
            ("FIB_Borderline", dict(use_feature_weighting=True, use_borderline=True, use_adaptive_quota=False, clean=False)),
            ("FIB_Adaptive", dict(use_feature_weighting=True, use_borderline=True, use_adaptive_quota=True, clean=False)),
            ("FIB_Full", dict(use_feature_weighting=True, use_borderline=True, use_adaptive_quota=True, clean=True)),
        ]
        for tag, flags in ablation_configs:
            fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0,
                            random_state=RANDOM_STATE, **flags)
            X_bal, y_bal = fib.fit_resample(X_train_sel, y_train)
            add_row(tag, fit_eval_rf(X_bal, y_bal, X_test_sel, y_test))

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    out_dir = PROJECT_ROOT / "results" / "per_fold" / "resampling_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rel_path in targets.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"[skip] {name}")
            continue
        t0 = time.time()
        print(f"[running] {name} ...", flush=True)
        try:
            df = run_dataset(name, path)
        except Exception as e:
            print(f"  [error] {name}: {e}", flush=True)
            continue
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        print(df.groupby("config")[["F1", "AUC", "MCC", "Gmean"]].mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
