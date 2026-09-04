
"""Cross-validated ablation across all pipeline stages.

Configurations, each building on the previous:
  A_Baseline      raw features, no balancing, plain Random Forest
  B_FS            + Stage 1 (two-stage feature selection)
  C_FS_Sampling   + Stage 2 (FIB-SMOTE)
  D_Full_Untuned  + Stage 3 (stacking ensemble), default hyperparameters
  E_Full_Tuned    + Bayesian hyperparameter search over that ensemble

Stage 1 is fit once per fold and reused for B through E; Stage 2 is fit once per
fold and reused for C through E. Besides saving computation, this is what the
comparison requires: each stage is fit on the training fold only, never on the
held-out test fold.

Writes one CSV per dataset to results/per_fold/ablation/. Aggregation and
significance testing are handled by scripts/generate_tables.py and
scripts/statistical_analysis.py.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote
from src.stacking_ensemble import build_stacking_ensemble
from src.bayes_opt import build_tuned_stacking_ensemble, tune_stacking_ensemble

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42


def g_mean(y_true, y_pred) -> float:
    rp = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rn = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return (rp * rn) ** 0.5


def score(clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = clf.decision_function(X_test)
    auc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan
    return {
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Gmean": g_mean(y_test, y_pred),
    }


def run_dataset(name: str, path: Path) -> pd.DataFrame:
    X, y, _ = load_defect_dataset(path)
    n_splits = min(N_SPLITS, y.value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    selected_full, importances_full, _est, _ = two_stage_feature_selection(
        X, y, corr_threshold=0.90, min_features=3,
        wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
    )
    fib_full = FIBSmote(feature_importances=importances_full, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
    X_bal_full, y_bal_full = fib_full.fit_resample(X[selected_full], y)
    best_params = tune_stacking_ensemble(X_bal_full, y_bal_full, n_trials=15, random_state=RANDOM_STATE)
    print(f"  [{name}] Bayesian-tuned params: {best_params}", flush=True)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf_a = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        clf_a.fit(X_train, y_train)
        rows.append({"dataset": name, "fold": fold_i, "config": "A_Baseline", **score(clf_a, X_test, y_test)})

        selected, importances, _est, _ = two_stage_feature_selection(
            X_train, y_train, corr_threshold=0.90, min_features=3,
            wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
        )
        X_train_sel, X_test_sel = X_train[selected], X_test[selected]

        clf_b = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        clf_b.fit(X_train_sel, y_train)
        rows.append({"dataset": name, "fold": fold_i, "config": "B_FS", **score(clf_b, X_test_sel, y_test)})

        fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = fib.fit_resample(X_train_sel, y_train)

        clf_c = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        clf_c.fit(X_train_bal, y_train_bal)
        rows.append({"dataset": name, "fold": fold_i, "config": "C_FS_Sampling", **score(clf_c, X_test_sel, y_test)})

        clf_d = build_stacking_ensemble(random_state=RANDOM_STATE)
        clf_d.fit(X_train_bal, y_train_bal)
        rows.append({"dataset": name, "fold": fold_i, "config": "D_Full_Untuned", **score(clf_d, X_test_sel, y_test)})

        clf_e = build_tuned_stacking_ensemble(best_params, random_state=RANDOM_STATE)
        clf_e.fit(X_train_bal, y_train_bal)
        rows.append({"dataset": name, "fold": fold_i, "config": "E_Full_Tuned", **score(clf_e, X_test_sel, y_test)})

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Run only this dataset, save to results/per_fold/ablation/<name>.csv")
    args = parser.parse_args()

    per_dataset_dir = PROJECT_ROOT / "results" / "per_fold" / "ablation"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    for name, rel_path in targets.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"[skip] {name}: file not found")
            continue
        t0 = time.time()
        print(f"[running] {name} ...", flush=True)
        try:
            df = run_dataset(name, path)
        except Exception as e:
            print(f"  [error] {name}: {e}", flush=True)
            continue
        df.to_csv(per_dataset_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        elapsed = time.time() - t0
        summary = df.groupby("config")[["F1", "AUC", "MCC", "Gmean"]].mean()
        print(f"  done in {elapsed:.1f}s", flush=True)
        print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
