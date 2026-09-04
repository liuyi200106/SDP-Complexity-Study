"""Significance testing for every comparison reported in the paper.

Testing protocol (Section 3.7 of the paper), applied uniformly:

  * Pairwise comparisons use the Wilcoxon signed-rank test over dataset-level
    mean scores -- one paired observation per independent dataset. Folds within
    a dataset are not independent, so per-fold values are never fed to a test.
  * Cliff's delta accompanies each comparison as an effect size, labelled by the
    conventional thresholds (<0.147 negligible, <0.33 small, <0.474 medium,
    >=0.474 large). It is a rank-based dominance measure usually introduced for
    independent samples; it is reported here because it is the effect size most
    widely used in the SDP literature, and is interpreted descriptively
    alongside the paired tests.
  * Holm-Bonferroni correction is applied separately per metric, across the
    pre-specified set of comparisons belonging to that experiment.
  * Friedman omnibus tests cover all configurations of an experiment; Nemenyi
    post-hoc tests are computed only where the omnibus null is rejected.

Sample sizes differ by experiment and are printed with every result: 14 datasets
for the main component comparisons, 13 for the SHAP diagnostics (JM1 excluded,
see experiments/rq5_shap_reuse.py), 9 for schema-specific analyses.

Usage
-----
    python scripts/statistical_analysis.py                     # all experiments
    python scripts/statistical_analysis.py --experiment ablation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "results" / "summary"
STATS = PROJECT_ROOT / "results" / "statistical_tests"

EXPERIMENTS = {
    "ablation": (
        "ablation_dataset_level.csv", "config",
        ["A_Baseline", "B_FS", "C_FS_Sampling", "D_Full_Untuned", "E_Full_Tuned"],
        ["F1", "AUC", "MCC", "Gmean"],
        [("A_Baseline", "C_FS_Sampling"), ("C_FS_Sampling", "D_Full_Untuned"),
         ("D_Full_Untuned", "E_Full_Tuned"), ("C_FS_Sampling", "E_Full_Tuned"),
         ("A_Baseline", "E_Full_Tuned")],
    ),
    "imbalanced_ensemble": (
        "imbalanced_ensemble_dataset_level.csv", "config",
        ["C_Reference", "BalancedRF", "EasyEnsemble", "ABF", "ABF_Hybrid"],
        ["F1", "Gmean"],
        [("C_Reference", "ABF_Hybrid"), ("BalancedRF", "ABF_Hybrid"),
         ("EasyEnsemble", "ABF_Hybrid"), ("C_Reference", "ABF"),
         ("BalancedRF", "ABF")],
    ),
    "shap_weighting": (
        "shap_weighting_dataset_level.csv", "weight_mode",
        ["true_shap", "uniform", "random", "shuffled"],
        ["F1", "AUC", "PRAUC", "MCC", "Gmean"],
        [("uniform", "true_shap"), ("random", "true_shap"), ("shuffled", "true_shap")],
    ),
    "filter_criterion": (
        "filter_criterion_dataset_level.csv", "filter_criterion",
        ["shap", "gini", "mutual_info", "variance"],
        ["F1", "AUC", "PRAUC", "MCC", "Gmean"],
        [("gini", "shap"), ("mutual_info", "shap"), ("variance", "shap")],
    ),
}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    return (gt - lt) / (len(x) * len(y))


def delta_label(d: float) -> str:
    a = abs(d)
    return ("negligible" if a < 0.147 else
            "small" if a < 0.33 else
            "medium" if a < 0.474 else "large")


def analyse(name: str, exclude: list[str] | None = None) -> None:
    summary_file, config_col, configs, metrics, pairs = EXPERIMENTS[name]
    path = SUMMARY / summary_file
    if not path.exists():
        print(f"[{name}] {summary_file} not found -- run scripts/generate_tables.py first")
        return

    dl = pd.read_csv(path)
    if exclude:
        dl = dl[~dl["dataset"].isin(exclude)]
    present = [c for c in configs if c in dl[config_col].unique()]
    n = dl["dataset"].nunique()
    suffix = f" (excluding {', '.join(exclude)})" if exclude else ""
    print(f"\n{'=' * 72}\n{name}{suffix}: n = {n} datasets\n{'=' * 72}")

    print(dl.groupby(config_col)[metrics].mean().reindex(present).round(4).to_string())

    rows = []
    for metric in metrics:
        pivot = dl.pivot(index="dataset", columns=config_col, values=metric)
        pvals, deltas, diffs, valid = [], [], [], []
        for a, b in pairs:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            _, p = wilcoxon(pivot[b], pivot[a])
            pvals.append(p)
            deltas.append(cliffs_delta(pivot[b].values, pivot[a].values))
            diffs.append((pivot[b] - pivot[a]).mean())
            valid.append((a, b))
        if not pvals:
            continue
        _, holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        for (a, b), p, hp, d, diff in zip(valid, pvals, holm, deltas, diffs):
            rows.append({
                "metric": metric, "comparison": f"{b} vs {a}",
                "mean_diff": round(diff, 4), "cliffs_delta": round(d, 4),
                "effect": delta_label(d), "p_raw": round(p, 4),
                "p_holm": round(hp, 4), "significant_after_holm": bool(hp < 0.05),
            })
    tests = pd.DataFrame(rows)
    print("\n--- Wilcoxon + Cliff's delta + Holm ---")
    print(tests.to_string(index=False))

    fried = []
    for metric in metrics:
        pivot = dl.pivot(index="dataset", columns=config_col, values=metric)[present].dropna()
        stat, p = friedmanchisquare(*[pivot[c].values for c in present])
        ranks = pivot.rank(axis=1, ascending=False).mean().reindex(present)
        fried.append({"metric": metric, "friedman_stat": round(stat, 4), "friedman_p": round(p, 4)})
        print(f"\n--- {metric}: Friedman chi2 = {stat:.3f}, p = {p:.4f} "
              f"({'reject' if p < 0.05 else 'retain'} omnibus null) ---")
        print("  average ranks (1 = best): " +
              ", ".join(f"{c}={ranks[c]:.2f}" for c in present))
        if p < 0.05:
            nem = sp.posthoc_nemenyi_friedman(pivot.values)
            nem.index = nem.columns = present
            print("  Nemenyi post-hoc p-values:")
            print(nem.round(4).to_string().replace("\n", "\n    "))

    if not exclude:
        STATS.mkdir(parents=True, exist_ok=True)
        tests.to_csv(STATS / f"{name}_tests_recomputed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fried).to_csv(STATS / f"{name}_friedman_recomputed.csv",
                                   index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default=None)
    args = parser.parse_args()

    for name in ([args.experiment] if args.experiment else list(EXPERIMENTS)):
        analyse(name)

    if args.experiment in (None, "imbalanced_ensemble"):
        analyse("imbalanced_ensemble", exclude=["KC2"])


if __name__ == "__main__":
    main()
