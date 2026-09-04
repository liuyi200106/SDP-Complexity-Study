
"""Parameter sensitivity of the two-stage pipeline (configuration C).

Two sweeps, both under the 5-fold CV used elsewhere, testing whether the reported
results depend on one particular hyperparameter setting:

  k_neighbors sweep    FIB-SMOTE's k in {3, 5, 7, 10}. Stage 1 is computed once
                       per fold and reused across all values, since feature
                       selection does not depend on k.
  feature_count sweep  Stage 1's filter_top_k in {5, 10, 15, None}, which changes
                       how many features survive the filter before RFE narrows
                       them further. Stage 1 is refit per value, since the filter
                       output itself changes.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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


def score(clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan
    return {
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Gmean": g_mean(y_test, y_pred),
    }


def k_neighbors_sensitivity(name: str, path: Path, k_values=(3, 5, 7, 10)) -> pd.DataFrame:
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

        for k in k_values:
            fib = FIBSmote(feature_importances=importances, k_neighbors=k, sampling_ratio=1.0, random_state=RANDOM_STATE)
            X_bal, y_bal = fib.fit_resample(X_train_sel, y_train)
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
            clf.fit(X_bal, y_bal)
            rows.append({"dataset": name, "fold": fold_i, "k_neighbors": k, **score(clf, X_test_sel, y_test)})

    return pd.DataFrame(rows)


def feature_count_sensitivity(name: str, path: Path, top_k_values=(5, 10, 15, None)) -> pd.DataFrame:
    X, y, _ = load_defect_dataset(path)
    n_splits = min(N_SPLITS, y.value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for top_k in top_k_values:
            selected, importances, _est, _ = two_stage_feature_selection(
                X_train, y_train, corr_threshold=0.90, filter_top_k=top_k, min_features=3,
                wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
            )
            X_train_sel, X_test_sel = X_train[selected], X_test[selected]
            fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
            X_bal, y_bal = fib.fit_resample(X_train_sel, y_train)
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
            clf.fit(X_bal, y_bal)
            rows.append({
                "dataset": name, "fold": fold_i,
                "requested_top_k": top_k if top_k is not None else 999,
                "n_features_selected": len(selected),
                **score(clf, X_test_sel, y_test),
            })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", choices=["k", "features", "both"], default="both")
    args = parser.parse_args()
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    out_dir = PROJECT_ROOT / "results" / "per_fold" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rel_path in targets.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"[skip] {name}")
            continue

        if args.mode in ("k", "both"):
            t0 = time.time()
            print(f"[k_neighbors] {name} ...", flush=True)
            df_k = k_neighbors_sensitivity(name, path)
            df_k.to_csv(out_dir / f"{name}_k_neighbors.csv", index=False, encoding="utf-8-sig")
            print(f"  done in {time.time()-t0:.1f}s", flush=True)
            print(df_k.groupby("k_neighbors")[["F1", "AUC", "MCC", "Gmean"]].mean().round(3).to_string(), flush=True)

        if args.mode in ("features", "both"):
            t0 = time.time()
            print(f"[feature_count] {name} ...", flush=True)
            df_f = feature_count_sensitivity(name, path)
            df_f.to_csv(out_dir / f"{name}_feature_count.csv", index=False, encoding="utf-8-sig")
            print(f"  done in {time.time()-t0:.1f}s", flush=True)
            print(df_f.groupby("requested_top_k")[["n_features_selected", "F1", "AUC", "MCC", "Gmean"]].mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
