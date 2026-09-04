
"""Agreement between the Stage 1 SHAP importance vector and the importance the final
model relies on. Consistency part of RQ5.

Reusing the Stage 1 vector rests on an assumption: that importances computed from
a Random Forest fit on the imbalanced, selected-feature training data still
describe what matters after Stage 2 has rebalanced the data and the final model
has been trained. If the two rankings agree, the Stage 1 vector is a faithful
summary and reuse is safe. If they diverge, the information shared across stages
is unstable, which is itself a reportable result.

Per fold, over the selected features:
  stage1_shap  mean |SHAP| of the Stage 1 estimator on the imbalanced,
               selected-feature training data, i.e. what Stage 2 receives
  final_shap   mean |SHAP| of the final Random Forest, trained on the
               FIB-SMOTE-balanced data, i.e. what the model uses

Agreement is measured by Spearman rank correlation and by top-half overlap, the
fraction of the top ceil(m/2) features shared, averaged over folds and datasets.
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.feature_selection import _shap_importance_from_estimator, two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42


def top_k_overlap(a: pd.Series, b: pd.Series, k: int) -> float:
    top_a = set(a.sort_values(ascending=False).index[:k])
    top_b = set(b.sort_values(ascending=False).index[:k])
    return len(top_a & top_b) / k


def run_dataset(name: str, path: Path) -> pd.DataFrame:
    X, y, _ = load_defect_dataset(path)
    n_splits = min(N_SPLITS, y.value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold_i, (train_idx, _test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        selected, stage1_shap, _est, _ = two_stage_feature_selection(
            X_train, y_train, corr_threshold=0.90, min_features=3,
            wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
        )
        if len(selected) < 2:
            continue
        X_train_sel = X_train[selected]

        fib = FIBSmote(feature_importances=stage1_shap, k_neighbors=5,
                       sampling_ratio=1.0, random_state=RANDOM_STATE)
        X_bal, y_bal = fib.fit_resample(X_train_sel, y_train)
        final_rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        final_rf.fit(X_bal, y_bal)
        final_shap = _shap_importance_from_estimator(final_rf, X_bal)

        s1 = stage1_shap.reindex(selected)
        sf = final_shap.reindex(selected)
        rho, _p = spearmanr(s1.values, sf.values)
        k = max(1, math.ceil(len(selected) / 2))
        rows.append({
            "dataset": name, "fold": fold_i, "n_features": len(selected),
            "spearman_rho": rho, f"top_half_overlap": top_k_overlap(s1, sf, k),
        })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    out_dir = PROJECT_ROOT / "results" / "per_fold" / "importance_consistency"
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
        print(f"  done in {time.time()-t0:.1f}s  "
              f"spearman={df['spearman_rho'].mean():.3f}  "
              f"top_half_overlap={df['top_half_overlap'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
