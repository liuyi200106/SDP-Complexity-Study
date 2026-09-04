
"""Sweeps ABF-Hybrid's majority_undersample_ratio r (src/abf.py) under 5-fold CV.

r = 2.0 was originally selected as the local optimum on a single dataset
(KC2); that initial sweep is kept as results/summary/abf_ratio_sweep_kc2_initial.csv.
Because KC2 is also in the evaluation set, tuning on it risks overfitting the
benchmark, so this module repeats the sweep on every dataset to show whether
r = 2.0 sits on a plateau or is a KC2-specific artifact. Feeds Table 15.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.abf import AdaptiveBalancedForestHybrid
from src.feature_selection import two_stage_feature_selection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def g_mean(y_true, y_pred) -> float:
    rp = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rn = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return (rp * rn) ** 0.5


def sweep(dataset_name: str, ratios=(1.0, 1.5, 2.0, 3.0, 5.0)) -> pd.DataFrame:
    X, y, _ = load_defect_dataset(PROJECT_ROOT / DATASETS[dataset_name])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for ratio in ratios:
        f1s, gms, aucs, mccs = [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            selected, importances, _est, _ = two_stage_feature_selection(
                X_train, y_train, wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE
            )
            X_train_sel, X_test_sel = X_train[selected], X_test[selected]
            abf = AdaptiveBalancedForestHybrid(
                feature_importances=importances, n_estimators=200,
                majority_undersample_ratio=ratio, random_state=RANDOM_STATE,
            )
            abf.fit(X_train_sel, y_train)
            y_pred = abf.predict(X_test_sel)
            y_proba = abf.predict_proba(X_test_sel)[:, 1]
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            gms.append(g_mean(y_test, y_pred))
            aucs.append(roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan)
            mccs.append(matthews_corrcoef(y_test, y_pred))
        rows.append({
            "dataset": dataset_name, "majority_undersample_ratio": ratio,
            "F1": np.mean(f1s), "AUC": np.mean(aucs), "MCC": np.mean(mccs), "Gmean": np.mean(gms),
        })
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="sweep only this dataset (default: all 14)")
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASETS.keys())

    out_path = PROJECT_ROOT / "results" / "summary" / "abf_ratio_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    frames = []
    for name in targets:
        print(f"[sweeping r] {name} ...", flush=True)
        try:
            frames.append(sweep(name))
        except Exception as e:
            print(f"  [error] {name}: {e}", flush=True)
    new = pd.concat(frames, ignore_index=True)

    if not existing.empty:
        existing = existing[~existing["dataset"].isin(new["dataset"].unique())]
    df = pd.concat([existing, new], ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n=== Mean over datasets, per r ===")
    print(df.groupby("majority_undersample_ratio")[["F1", "AUC", "MCC", "Gmean"]].mean().round(4).to_string())
    print("\n=== Best r per dataset (by G-mean) ===")
    best = df.loc[df.groupby("dataset")["Gmean"].idxmax()][["dataset", "majority_undersample_ratio", "Gmean"]]
    print(best.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
