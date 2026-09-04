"""Aggregate per-fold experiment output into the summary tables reported in the
paper.

Reads results/per_fold/<experiment>/<dataset>.csv and writes, for each
experiment, a fold-level concatenation and a dataset-level mean into
results/summary/. Every inferential test in the paper is computed on the
dataset-level means (one observation per dataset), never on individual folds --
folds within a dataset are not independent samples. See
scripts/statistical_analysis.py for the tests themselves.

Usage
-----
    python scripts/generate_tables.py                # all experiments
    python scripts/generate_tables.py --experiment ablation
    python scripts/generate_tables.py --check        # verify against committed summaries
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_FOLD = PROJECT_ROOT / "results" / "per_fold"
SUMMARY = PROJECT_ROOT / "results" / "summary"

EXPERIMENTS = {
    "ablation": ("ablation", "config",
                 ["F1", "AUC", "MCC", "Gmean"], "ablation"),
    "resampling": ("resampling_comparison", "config",
                   ["F1", "AUC", "MCC", "Gmean"], "resampling"),
    "imbalanced_ensemble": ("imbalanced_ensemble", "config",
                            ["F1", "AUC", "MCC", "Gmean"], "imbalanced_ensemble"),
    "shap_weighting": ("shap_weighting", "weight_mode",
                       ["F1", "AUC", "PRAUC", "MCC", "Gmean"], "shap_weighting"),
    "filter_criterion": ("filter_criterion", "filter_criterion",
                         ["F1", "AUC", "PRAUC", "MCC", "Gmean",
                          "n_features_selected", "stage1_time_s"], "filter_criterion"),
    "shap_reuse": ("shap_reuse", "strategy",
                   ["F1", "AUC", "PRAUC", "MCC", "Gmean",
                    "shap_seconds", "stage1_seconds", "pipeline_seconds"], "shap_reuse"),
    "importance_consistency": ("importance_consistency", None,
                               ["spearman_rho", "top_half_overlap", "n_features"],
                               "importance_consistency"),
}


def load_folds(subdir: str) -> pd.DataFrame | None:
    directory = PER_FOLD / subdir
    files = sorted(directory.glob("*.csv")) if directory.exists() else []
    frames = []
    for f in files:
        if f.stat().st_size == 0:
            print(f"  [skip] {f.name}: empty (run still in progress?)")
            continue
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f"  [skip] {f.name}: no parseable rows yet")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def aggregate(name: str, check_only: bool = False) -> None:
    subdir, config_col, metrics, stem = EXPERIMENTS[name]
    fold = load_folds(subdir)
    if fold is None:
        print(f"[{name}] no per-fold results found under results/per_fold/{subdir}/ -- skipped")
        return

    metrics = [m for m in metrics if m in fold.columns]
    keys = ["dataset"] + ([config_col] if config_col else [])
    dataset_level = fold.groupby(keys, as_index=False)[metrics].mean()

    n = fold["dataset"].nunique()
    targets = {
        SUMMARY / f"{stem}_fold_level.csv": fold,
        SUMMARY / f"{stem}_dataset_level.csv": dataset_level,
    }

    if check_only:
        for path, df in targets.items():
            if not path.exists():
                print(f"[{name}] MISSING committed summary: {path.name}")
                continue
            committed = pd.read_csv(path)
            same = (len(committed) == len(df)) and set(committed.columns) == set(df.columns)
            if same:
                merged = df[metrics].round(6).reset_index(drop=True)
                ref = committed[metrics].round(6).reset_index(drop=True)
                same = merged.equals(ref)
            print(f"[{name}] {path.name}: {'matches committed output' if same else 'DIFFERS'}")
        return

    SUMMARY.mkdir(parents=True, exist_ok=True)
    for path, df in targets.items():
        df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{name}] n = {n} datasets -> {stem}_fold_level.csv, {stem}_dataset_level.csv")

    if config_col:
        overall = dataset_level.groupby(config_col)[metrics].mean()
        print(overall.round(4).to_string())
    else:
        print(dataset_level[metrics].mean().round(4).to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default=None)
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare against the committed summary files "
                             "instead of overwriting them")
    args = parser.parse_args()

    names = [args.experiment] if args.experiment else list(EXPERIMENTS)
    for name in names:
        aggregate(name, check_only=args.check)


if __name__ == "__main__":
    main()
