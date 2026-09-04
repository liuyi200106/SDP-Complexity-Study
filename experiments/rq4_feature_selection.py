
"""RQ4: does the SHAP-guided Stage 1 filter earn its cost?

Reported in Section 4.6 of the paper.

The weight-control experiment of Section 4.5 found that SHAP importances carry
no detectable feature-specific information into Stage 2's neighbor search. SHAP is
used in a second place, however: the Stage 1 filter, where it breaks ties between
highly correlated feature pairs by keeping the more important one. That use is the
more expensive of the two, since the exact TreeExplainer pass dominates pipeline
training time.

This experiment holds the entire pipeline fixed and varies only the filter's
tie-break criterion:

  shap         mean |SHAP| from a Random Forest
  gini         the same forest's built-in Gini/MDI importance, effectively free
  mutual_info  mutual information with the label; cheap and model-free
  variance     keep the higher-variance feature; the cheapest possible baseline

Everything downstream (the RFECV wrapper, the SHAP importances handed to Stage 2,
FIB-SMOTE, the Random Forest classifier) is identical across the four arms, so any
difference is attributable to the filter criterion alone.

Note that the wrapper stage still computes SHAP for the importances passed to
Stage 2 in all four arms. This experiment isolates the filter criterion, not the
removal of SHAP from the pipeline.

Usage
-----
    python experiments/rq4_feature_selection.py
    python experiments/rq4_feature_selection.py --dataset KC2
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.evaluation import score_estimator
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42
CRITERIA = ["shap", "gini", "mutual_info", "variance"]


def run_dataset(name: str, path: Path) -> pd.DataFrame:
    X, y, _ = load_defect_dataset(path)
    n_splits = min(N_SPLITS, y.value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for criterion in CRITERIA:
            t0 = time.time()
            selected, importances, _est, _ = two_stage_feature_selection(
                X_train, y_train, corr_threshold=0.90, min_features=3,
                wrapper_step=2, wrapper_inner_cv_splits=3,
                filter_criterion=criterion, random_state=RANDOM_STATE,
            )
            stage1_time = time.time() - t0

            X_train_sel, X_test_sel = X_train[selected], X_test[selected]
            fib = FIBSmote(
                feature_importances=importances, k_neighbors=5,
                sampling_ratio=1.0, random_state=RANDOM_STATE,
            )
            X_bal, y_bal = fib.fit_resample(X_train_sel, y_train)
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
            clf.fit(X_bal, y_bal)

            rows.append({
                "dataset": name, "fold": fold_i, "filter_criterion": criterion,
                "n_features_selected": len(selected),
                "stage1_time_s": round(stage1_time, 3),
                **score_estimator(clf, X_test_sel, y_test),
            })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    out_dir = PROJECT_ROOT / "results" / "per_fold" / "filter_criterion"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        print(df.groupby("filter_criterion")[
            ["F1", "AUC", "PRAUC", "MCC", "Gmean", "n_features_selected", "stage1_time_s"]
        ].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
