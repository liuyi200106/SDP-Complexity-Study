
"""RQ6: does Adaptive Balanced Forest (ABF) improve on purpose-built imbalanced
ensembles?

Reported in Section 4.8 of the paper. Full 5-fold CV comparison, all
configurations on Stage 1 SHAP+RFE selected features:

  C_Reference   single FIB-SMOTE pass and one plain Random Forest (Stage 1 + 2)
  BalancedRF    imbalanced-learn's per-tree random-undersampling forest
  EasyEnsemble  imbalanced-learn's AdaBoost over balanced subsets
  ABF           per-tree FIB-SMOTE with bagging
  ABF_Hybrid    ABF with light majority-class undersampling in each bag before
                FIB-SMOTE runs, combining both resampling mechanisms

Usage
-----
    python experiments/rq6_imbalanced_ensemble.py
    python experiments/rq6_imbalanced_ensemble.py --dataset KC2
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote
from src.abf import AdaptiveBalancedForest, AdaptiveBalancedForestHybrid

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

        fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
        X_bal, y_bal = fib.fit_resample(X_train_sel, y_train)
        clf_c = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        clf_c.fit(X_bal, y_bal)
        rows.append({"dataset": name, "fold": fold_i, "config": "C_Reference",
                     **score(y_test, clf_c.predict(X_test_sel), clf_c.predict_proba(X_test_sel)[:, 1])})

        brf = BalancedRandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4, sampling_strategy="all", replacement=True)
        brf.fit(X_train_sel, y_train)
        rows.append({"dataset": name, "fold": fold_i, "config": "BalancedRF",
                     **score(y_test, brf.predict(X_test_sel), brf.predict_proba(X_test_sel)[:, 1])})

        eec = EasyEnsembleClassifier(n_estimators=20, random_state=RANDOM_STATE, n_jobs=4)
        eec.fit(X_train_sel, y_train)
        rows.append({"dataset": name, "fold": fold_i, "config": "EasyEnsemble",
                     **score(y_test, eec.predict(X_test_sel), eec.predict_proba(X_test_sel)[:, 1])})

        abf = AdaptiveBalancedForest(feature_importances=importances, n_estimators=200, random_state=RANDOM_STATE)
        abf.fit(X_train_sel, y_train)
        rows.append({"dataset": name, "fold": fold_i, "config": "ABF",
                     **score(y_test, abf.predict(X_test_sel), abf.predict_proba(X_test_sel)[:, 1])})

        abf_h = AdaptiveBalancedForestHybrid(feature_importances=importances, n_estimators=200,
                                              majority_undersample_ratio=2.0, random_state=RANDOM_STATE)
        abf_h.fit(X_train_sel, y_train)
        rows.append({"dataset": name, "fold": fold_i, "config": "ABF_Hybrid",
                     **score(y_test, abf_h.predict(X_test_sel), abf_h.predict_proba(X_test_sel)[:, 1])})

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    out_dir = PROJECT_ROOT / "results" / "per_fold" / "imbalanced_ensemble"
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
