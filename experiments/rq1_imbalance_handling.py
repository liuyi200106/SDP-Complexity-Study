"""RQ1: how much does imbalance handling contribute, and does the choice of
resampling method matter?

Two parts, reported in Sections 4.3 and 4.3.1 of the paper.

  ablation    Progressive pipeline ablation over all 14 datasets:
                A  raw features, no balancing
                B  + feature selection (SHAP filter + RFECV)
                C  + oversampling (FIB-SMOTE)
                D  + heterogeneous stacking ensemble
                E  + Bayesian hyperparameter tuning
              A single run produces every configuration, since they share the
              cross-validation folds. Configurations D and E are what RQ2
              analyses, so rq2_ensemble_tuning.py consumes this same output
              instead of recomputing it: run this part once and both RQs are
              covered. Feeds Tables 3 and 5.

  resampling  Ten-way comparison holding the pipeline fixed and varying only the
              resampling step: no resampling, cost-sensitive learning, four
              established methods (SMOTE, Borderline-SMOTE, ADASYN, SMOTE-Tomek)
              and four cumulative FIB-SMOTE variants. Feeds Table 4.

Usage
-----
    python experiments/rq1_imbalance_handling.py --part ablation
    python experiments/rq1_imbalance_handling.py --part resampling
    python experiments/rq1_imbalance_handling.py --part both --dataset KC2

Results are written per dataset under results/per_fold/, so an interrupted run
can be resumed by re-invoking it for the datasets still missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._lib import ablation, resampling_comparison
from src.utils import DATASETS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_part(module, out_subdir: str, datasets: list[str]) -> None:
    out_dir = PROJECT_ROOT / "results" / "per_fold" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        path = PROJECT_ROOT / DATASETS[name]
        if not path.exists():
            print(f"[skip] {name}: dataset file not found at {path}")
            print("       See data/README.md for how to fetch the datasets.")
            continue
        print(f"[running] {name} ...", flush=True)
        df = module.run_dataset(name, path)
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  -> results/per_fold/{out_subdir}/{name}.csv", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["ablation", "resampling", "both"], default="both")
    parser.add_argument("--dataset", default=None,
                        help="run a single dataset (default: all 14)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS)

    if args.part in ("ablation", "both"):
        print("=== RQ1 part 1/2: pipeline ablation (A-E) ===")
        run_part(ablation, "ablation", datasets)
    if args.part in ("resampling", "both"):
        print("=== RQ1 part 2/2: resampling-method comparison ===")
        run_part(resampling_comparison, "resampling_comparison", datasets)

    print("\nNext: python scripts/generate_tables.py   (Tables 3, 4, 5)")
    print("      python scripts/statistical_analysis.py  (significance tests)")


if __name__ == "__main__":
    main()
