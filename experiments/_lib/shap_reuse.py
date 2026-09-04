
"""Cost and accuracy of computing SHAP once instead of twice. Reuse part of RQ5.

The pipeline runs an exact TreeExplainer pass twice per fold:
  1. in the filter step, to break ties between correlated features, and
  2. on the final selected-feature estimator, producing the importance vector
     handed to Stage 2.

Two strategies are compared with everything else held fixed:

  two_shap     both passes; the vector Stage 2 receives is the second pass
  single_shap  one pass, in the filter step. Its components for the finally
               selected features are reused as the vector handed to Stage 2, and
               the second pass is skipped entirely.

For each, F1/AUC/PR-AUC/MCC/G-mean on the held-out fold are recorded together with
timings at three levels: the TreeExplainer passes alone, the whole
feature-selection stage, and end-to-end training. The three are reported
separately because the saving differs sharply between them.

The two-stage selection is reproduced inline here rather than by modifying
src/feature_selection.py, so the other experiments run against unchanged code.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.evaluation import score_estimator
from src.feature_selection import _shap_importance_from_estimator
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42


def select_with_shap(X_train, y_train, reuse: bool):
    """Two-stage selection. Returns (selected, importances_for_stage2,
    shap_seconds, stage1_seconds). If reuse=True, SHAP is computed only in the
    filter step and its selected-feature components are reused for Stage 2; if
    False, a second SHAP pass on the final estimator produces the Stage 2
    vector. shap_seconds counts only the TreeExplainer passes; stage1_seconds
    is the wall-clock time of the whole feature-selection stage, so the two are
    reported separately rather than conflated."""
    shap_seconds = 0.0
    stage1_start = time.time()
    X = X_train.copy()

    vt = VarianceThreshold(threshold=1e-5).fit(X)
    X = X.loc[:, vt.get_support()]

    filt_rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=4)
    filt_rf.fit(X, y_train)
    t0 = time.time()
    filter_shap = _shap_importance_from_estimator(filt_rf, X)
    shap_seconds += time.time() - t0

    corr = X.corr().abs()
    cols = list(X.columns)
    drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if c1 in drop or c2 in drop:
                continue
            if corr.loc[c1, c2] > 0.90:
                drop.add(c2 if filter_shap[c1] >= filter_shap[c2] else c1)
    filtered = [c for c in cols if c not in drop]
    X = X[filtered]

    est = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
    n_splits = min(3, y_train.value_counts().min())
    if n_splits < 2:
        selected = filtered
        final_est = est.fit(X, y_train)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        sel = RFECV(est, step=2, cv=cv, scoring="f1",
                    min_features_to_select=min(3, X.shape[1]), n_jobs=4)
        sel.fit(X, y_train)
        selected = X.columns[sel.support_].tolist()
        final_est = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=4)
        final_est.fit(X[selected], y_train)

    if reuse:
        importances = filter_shap.reindex(selected).sort_values(ascending=False)
    else:
        t0 = time.time()
        importances = _shap_importance_from_estimator(final_est, X[selected]).sort_values(ascending=False)
        shap_seconds += time.time() - t0

    stage1_seconds = time.time() - stage1_start
    return selected, importances, shap_seconds, stage1_seconds


def run_dataset(name: str, path: Path) -> pd.DataFrame:
    X, y, _ = load_defect_dataset(path)
    n_splits = min(N_SPLITS, y.value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for strategy, reuse in [("two_shap", False), ("single_shap", True)]:
            pipeline_start = time.time()
            selected, importances, shap_sec, stage1_sec = select_with_shap(
                X_train, y_train, reuse=reuse)
            X_tr_sel, X_te_sel = X_train[selected], X_test[selected]
            fib = FIBSmote(feature_importances=importances, k_neighbors=5,
                           sampling_ratio=1.0, random_state=RANDOM_STATE)
            X_bal, y_bal = fib.fit_resample(X_tr_sel, y_train)
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
            clf.fit(X_bal, y_bal)
            pipeline_sec = time.time() - pipeline_start
            rows.append({
                "dataset": name, "fold": fold_i, "strategy": strategy,
                "shap_seconds": round(shap_sec, 3),
                "stage1_seconds": round(stage1_sec, 3),
                "pipeline_seconds": round(pipeline_sec, 3),
                **score_estimator(clf, X_te_sel, y_test),
            })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    out_dir = PROJECT_ROOT / "results" / "per_fold" / "shap_reuse"
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
        print(df.groupby("strategy")[["F1", "Gmean", "PRAUC", "shap_seconds", "stage1_seconds", "pipeline_seconds"]].mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
