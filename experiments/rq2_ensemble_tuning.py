"""RQ2: does a heterogeneous stacking ensemble, with or without Bayesian
hyperparameter tuning, help once feature selection and oversampling are already
in place?

Reported in Section 4.4 (Table 5) of the paper.

Answered from configurations C, D and E of the pipeline ablation:

  C  feature selection + oversampling + a single Random Forest
  D  + heterogeneous stacking (Random Forest, XGBoost, linear SVM, logistic
       regression meta-learner), hyperparameters left at defaults
  E  + Bayesian hyperparameter search over that ensemble (Optuna TPE, 15 trials,
       tuned once per dataset)

All five configurations come from a single pass of the ablation, since they share
the cross-validation folds. Rerunning the pipeline separately for RQ2 would repeat
several hours of computation and, because folds are seeded identically, would
produce the same numbers. This script therefore runs the ablation only for
datasets whose results are missing, then reports the C/D/E comparison. If
rq1_imbalance_handling.py --part ablation has already been run, nothing is
recomputed.

Usage
-----
    python experiments/rq2_ensemble_tuning.py            # report (runs only what is missing)
    python experiments/rq2_ensemble_tuning.py --force    # recompute every dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._lib import ablation
from src.utils import DATASETS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "per_fold" / "ablation"
CONFIGS = ["C_FS_Sampling", "D_Full_Untuned", "E_Full_Tuned"]


def ensure_results(force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rel in DATASETS.items():
        target = OUT_DIR / f"{name}.csv"
        if target.exists() and not force:
            continue
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"[skip] {name}: dataset file not found. See data/README.md")
            continue
        print(f"[running ablation] {name} ...", flush=True)
        ablation.run_dataset(name, path).to_csv(target, index=False, encoding="utf-8-sig")


def report() -> None:
    files = sorted(OUT_DIR.glob("*.csv"))
    if not files:
        print("No ablation results found. Run rq1_imbalance_handling.py --part ablation first.")
        return
    fold = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dataset_level = fold.groupby(["dataset", "config"])[["F1", "AUC", "MCC", "Gmean"]].mean().reset_index()

    print(f"\n=== RQ2: ensemble and tuning on top of the rebalanced pipeline "
          f"(n = {dataset_level['dataset'].nunique()} datasets) ===")
    table = dataset_level.groupby("config")[["F1", "AUC", "MCC", "Gmean"]].mean().reindex(CONFIGS)
    print(table.round(3).to_string())
    print("\nSignificance tests for these comparisons: scripts/statistical_analysis.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="recompute the ablation even where results already exist")
    args = parser.parse_args()
    ensure_results(args.force)
    report()


if __name__ == "__main__":
    main()
