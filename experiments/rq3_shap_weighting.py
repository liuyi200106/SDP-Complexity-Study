
"""RQ3: does the Stage 1 SHAP importance vector carry useful, feature-specific
information into Stage 2's neighbor search?

Reported in Section 4.5 of the paper.

Weighting the neighbor search by SHAP importances is what links the two stages. A
weighted distance could improve results simply because some features are
down-weighted, regardless of which ones, so the weighting has to be tested against
controls that hold that effect constant.

Four weight modes share an identical pipeline (same Stage 1 selected features per
fold, same standardization, same borderline/quota/Tomek logic, same downstream
Random Forest) and differ only in the weight vector:

  true_shap  Stage 1's actual SHAP importances
  uniform    all weights equal, i.e. weighting switched off
  random     weights drawn uniformly at random, ignoring Stage 1 entirely
  shuffled   Stage 1's real importance values permuted across features: identical
             magnitude distribution, feature correspondence destroyed

shuffled is the decisive control. If true_shap does not outperform shuffled, the
assignment of importance to specific features carries no information and only the
spread of weight magnitudes matters.

random and shuffled are averaged over several seeds, so the comparison does not
turn on a single permutation.

Usage
-----
    python experiments/rq3_shap_weighting.py
    python experiments/rq3_shap_weighting.py --dataset KC2
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.evaluation import METRIC_NAMES, score_estimator
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42
CONTROL_SEEDS = [42, 43, 44, 45, 46]


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

        for mode in ["true_shap", "uniform", "random", "shuffled"]:
            seeds = [RANDOM_STATE] if mode in ("true_shap", "uniform") else CONTROL_SEEDS
            per_seed = []
            for seed in seeds:
                fib = FIBSmote(
                    feature_importances=importances, k_neighbors=5, sampling_ratio=1.0,
                    weight_mode=mode, random_state=seed,
                )
                X_bal, y_bal = fib.fit_resample(X_train_sel, y_train)
                clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
                clf.fit(X_bal, y_bal)
                per_seed.append(score_estimator(clf, X_test_sel, y_test))

            averaged = {m: float(np.nanmean([s[m] for s in per_seed])) for m in METRIC_NAMES}
            rows.append({
                "dataset": name, "fold": fold_i, "weight_mode": mode,
                "n_seeds": len(seeds), **averaged,
            })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    out_dir = PROJECT_ROOT / "results" / "per_fold" / "shap_weighting"
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
        print(df.groupby("weight_mode")[["F1", "AUC", "PRAUC", "MCC", "Gmean"]].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
