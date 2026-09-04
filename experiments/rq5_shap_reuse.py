"""RQ5: is the explanation vector computed early in the pipeline consistent with
what the final model relies on, and can the redundant second SHAP pass be removed
without loss?

Two parts, both reported in Section 4.7 of the paper.

  consistency  For each fold, compare the SHAP importance vector Stage 1 hands to
               the oversampling step against the SHAP importance of the final
               model, trained on the rebalanced data, by Spearman rank correlation
               and top-half feature overlap. High agreement is what makes reuse
               safe.

  reuse        Compare the two-pass pipeline against computing SHAP once and
               reusing it, measuring predictive metrics and timings at three
               levels: the TreeExplainer passes alone, the whole feature-selection
               stage, and end-to-end training. The three levels are reported
               separately because the saving differs sharply between them.

Excludes JM1 (10,885 instances): both parts run the full pipeline several times
per fold, and exact SHAP becomes prohibitive at that size. The paper states this
scope explicitly in Section 4.2; the cost itself is quantified in Section 4.11.

Usage
-----
    python experiments/rq5_shap_reuse.py --part both
    python experiments/rq5_shap_reuse.py --part reuse --dataset KC2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._lib import importance_consistency, shap_reuse
from src.utils import DATASETS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"JM1"}


def run_part(module, out_subdir: str, datasets: list[str]) -> None:
    out_dir = PROJECT_ROOT / "results" / "per_fold" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        if name in EXCLUDED:
            print(f"[skip] {name}: excluded from the SHAP diagnostics (see module docstring)")
            continue
        path = PROJECT_ROOT / DATASETS[name]
        if not path.exists():
            print(f"[skip] {name}: dataset file not found. See data/README.md")
            continue
        print(f"[running] {name} ...", flush=True)
        df = module.run_dataset(name, path)
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  -> results/per_fold/{out_subdir}/{name}.csv", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["consistency", "reuse", "both"], default="both")
    parser.add_argument("--dataset", default=None, help="run a single dataset")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS)

    if args.part in ("consistency", "both"):
        print("=== RQ5 part 1/2: Stage 1 vs final-model importance consistency ===")
        run_part(importance_consistency, "importance_consistency", datasets)
    if args.part in ("reuse", "both"):
        print("=== RQ5 part 2/2: single-SHAP reuse vs the two-pass pipeline ===")
        run_part(shap_reuse, "shap_reuse", datasets)

    print("\nNext: python scripts/generate_tables.py   (Table 8)")


if __name__ == "__main__":
    main()
