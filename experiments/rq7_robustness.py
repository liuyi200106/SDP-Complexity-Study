"""RQ7: interpretability, robustness and computational cost.

Five sub-experiments, reported in Sections 4.9 to 4.11 of the paper. Each is
implemented in its own module under experiments/_lib/ and dispatched from here, so
that one RQ maps to one entry point.

  interpretability  Which code metrics recur as important across projects, using
                    SHAP on the final model and evaluating on real instances only.
                    Restricted to the nine procedural-metric datasets, because
                    identifying recurring features requires comparable metric
                    names and the Java CK metrics share none with the
                    McCabe/Halstead family. Feeds Table 11.

  cross-project     Train on one project, test on another, within each feature
                    schema group. Feeds Table 12.

  burak             Whether the Burak nearest-neighbour instance filter recovers
                    any of the cross-project gap. Reported as a negative result.

  sensitivity       Sweeps of the oversampling neighbourhood size k and of the
                    Stage 1 filter cap. Feeds Tables 13 and 14.

  ratio             Sweep of ABF-Hybrid's majority-undersample ratio r across
                    datasets, checking whether r = 2.0 sits on a plateau rather
                    than at a dataset-specific peak. Feeds Table 15.

  efficiency        Per-stage wall-clock timing against a plain Random Forest
                    baseline. Feeds Table 16.

Usage
-----
    python experiments/rq7_robustness.py --part all
    python experiments/rq7_robustness.py --part efficiency
    python experiments/rq7_robustness.py --part sensitivity --dataset KC2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._lib import (
    abf_ratio_sweep,
    burak_filter,
    cross_project,
    efficiency,
    sensitivity,
    shap_explanation,
)

PARTS = {
    "interpretability": shap_explanation,
    "cross-project": cross_project,
    "burak": burak_filter,
    "sensitivity": sensitivity,
    "ratio": abf_ratio_sweep,
    "efficiency": efficiency,
}
ORDER = ["interpretability", "cross-project", "burak", "sensitivity", "ratio", "efficiency"]


def dispatch(part: str, dataset: str | None) -> None:
    module = PARTS[part]
    argv = [f"{part}"]
    if dataset and part in ("sensitivity", "ratio", "interpretability", "efficiency"):
        argv += ["--dataset", dataset]
    saved, sys.argv = sys.argv, argv
    try:
        module.main()
    finally:
        sys.argv = saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=[*PARTS, "all"], default="all")
    parser.add_argument("--dataset", default=None,
                        help="single dataset, where the sub-experiment supports it")
    args = parser.parse_args()

    parts = ORDER if args.part == "all" else [args.part]
    for part in parts:
        print(f"\n{'=' * 70}\n=== RQ7: {part} ===\n{'=' * 70}", flush=True)
        dispatch(part, args.dataset)


if __name__ == "__main__":
    main()
