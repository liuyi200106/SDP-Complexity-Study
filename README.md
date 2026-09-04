# Does Added Complexity Pay Off?

Code, per-fold results and analysis scripts for *"Does Added Complexity Pay Off?
A Component-Wise Empirical Study of Feature Selection, Resampling and Ensembling
for Imbalanced Software Defect Prediction."*

Software defect prediction pipelines keep growing: feature selection, then
explanation-guided weighting, then synthetic oversampling, then heterogeneous
ensembles, then automated hyperparameter search. Each addition is usually
justified against a plain baseline rather than against the pipeline it is being
added to. This study measures each component's **marginal** contribution instead,
on 14 real datasets spanning two unrelated metric families.

The short version: **imbalance handling accounts for essentially all of the
measurable gain, and most of the remaining complexity does not pay for itself.**
One component does pay off: an explanation vector already computed earlier in
the pipeline can be reused instead of recomputed, cutting explanation time by
58% with no measurable loss.

## Headline results

Progressive ablation, 14 datasets, stratified 5-fold CV, dataset-level means:

| Configuration | F1 | AUC | MCC | G-mean |
|---|---:|---:|---:|---:|
| A: raw features, no balancing | 0.360 | 0.800 | 0.311 | 0.479 |
| B: + feature selection | 0.355 | 0.787 | 0.298 | 0.475 |
| **C: + oversampling (FIB-SMOTE)** | **0.461** | 0.789 | 0.355 | **0.645** |
| D: + heterogeneous stacking | 0.447 | 0.786 | 0.345 | 0.626 |
| E: + Bayesian tuning | 0.450 | 0.782 | 0.349 | 0.625 |

* **C vs A**: F1 +0.101 (delta = 0.37, medium), G-mean +0.167 (delta = 0.63,
  large), Wilcoxon p = 0.0001 for both, and both survive Holm correction.
* **D vs C, E vs D**: no comparison is significant; every effect size is
  negligible. Adding the ensemble and the tuner does not help.
* **Feature-importance weighting**: true SHAP weights (F1 0.463) beat neither
  uniform (0.464), random (0.457), nor **shuffled** weights (0.454). The shuffled
  control keeps the exact magnitude distribution of the real SHAP values and
  destroys only their feature correspondence, so it isolates whether the
  *identity* of the weights matters. Across 15 tests, none is significant after
  Holm and every effect size is negligible.
* **SHAP as the Stage 1 filter criterion**: F1 0.463 versus Gini 0.454, mutual
  information 0.453, variance 0.456: significant in 0 of 3 comparisons, at about
  1.5x the cost of the cheapest alternative.
* **Explanation reuse**: the Stage 1 importance vector agrees closely with what
  the final model relies on (Spearman rho = 0.83, top-half overlap 0.87), so the
  second SHAP pass can be dropped: explanation time 4.66 s -> 1.98 s (-57.6%),
  end-to-end 11.93 s -> 8.73 s (-26.9%), with no metric degrading.

Every number above is reproduced by `scripts/generate_tables.py` and
`scripts/statistical_analysis.py` from the per-fold CSVs in this repository.

## Layout

```
src/                    pipeline components (importable library)
  utils.py              dataset registry and loaders
  feature_selection.py  Stage 1: SHAP-guided filter + RFECV wrapper
  fib_smote.py          Stage 2: FIB-SMOTE oversampling
  abf.py                Stage 3: Adaptive Balanced Forest (+ Hybrid variant)
  stacking_ensemble.py  heterogeneous stacking ensemble
  bayes_opt.py          Optuna/TPE hyperparameter search
  evaluation.py         shared metric set used by every experiment
experiments/            one entry point per research question (rq1 ... rq7)
  _lib/                 the individual experiment implementations
scripts/                aggregation, significance testing, figures
results/
  per_fold/             raw per-fold output, one CSV per dataset  <- source of truth
  summary/              fold-level and dataset-level aggregates
  statistical_tests/    Wilcoxon / Cliff's delta / Holm / Friedman / Nemenyi
figures/                figures as they appear in the paper
data/                   see data/README.md - raw datasets are not redistributed
```

## Setup

```bash
pip install -r requirements.txt
```

Developed on Python 3.11.9. On Windows `python` may resolve to the Microsoft
Store stub; use `py -3.11` instead if `python -V` fails.

Then download the datasets as described in [`data/README.md`](data/README.md).
Data is only needed to *re-run* experiments; the analysis below works from the
committed results without it.

## Reproducing the paper from the committed results

This takes seconds and needs no dataset downloads.

```bash
python scripts/generate_tables.py --check
```

`--check` recomputes every summary table from `results/per_fold/` and compares it
against the committed file instead of overwriting it; all 14 report
`matches committed output`. Drop `--check` to regenerate them.

```bash
python scripts/statistical_analysis.py
```

Recomputes every significance test reported in the paper: Wilcoxon signed-rank
over dataset-level means, Cliff's delta, Holm-Bonferroni per metric, Friedman
omnibus, and Nemenyi post-hoc where the omnibus null is rejected.

```bash
python scripts/generate_figures.py
```

Redraws all three figures at 300 dpi (`--dpi` to override, `--figure N` for one).

## Re-running the experiments

Each research question has one entry point. These are slow: the full set is
several hours, dominated by the RFECV wrapper search on the largest datasets.

| Entry point | Question | Writes to | Paper |
|---|---|---|---|
| `experiments/rq1_imbalance_handling.py` | How much does imbalance handling contribute, and does the resampling method matter? | `per_fold/ablation/`, `per_fold/resampling_comparison/` | 4.3, 4.3.1 |
| `experiments/rq2_ensemble_tuning.py` | Does stacking, with or without Bayesian tuning, help once FS and oversampling are in place? | `per_fold/ablation/` | 4.4 |
| `experiments/rq3_shap_weighting.py` | Do SHAP weights carry feature-specific information into the neighbour search? | `per_fold/shap_weighting/` | 4.5 |
| `experiments/rq4_feature_selection.py` | Does the SHAP-guided filter beat cheaper criteria? | `per_fold/filter_criterion/` | 4.6 |
| `experiments/rq5_shap_reuse.py` | Is the early explanation vector consistent with the final model, and can the second pass go? | `per_fold/importance_consistency/`, `per_fold/shap_reuse/` | 4.7 |
| `experiments/rq6_imbalanced_ensemble.py` | Does ABF improve on purpose-built imbalanced ensembles? | `per_fold/imbalanced_ensemble/` | 4.8 |
| `experiments/rq7_robustness.py` | Interpretability, cross-project transfer, sensitivity, cost | `per_fold/sensitivity/`, `summary/` | 4.9-4.11 |

Most accept `--dataset NAME` to run a single dataset; `rq1`, `rq5` and `rq7` take
`--part` to run one sub-experiment. Run `--help` on any of them.

After re-running, regenerate the aggregates and tests:

```bash
python scripts/generate_tables.py && python scripts/statistical_analysis.py
```

## Notes on the experimental design

**Tests are computed on dataset-level means, never on individual folds.** Folds
within a dataset are not independent observations; treating 70 folds as 70
samples would inflate significance. Per-fold values are published and plotted
(Figure 3) for description only.

**Sample sizes differ by experiment and are printed with every result.** The main
component comparisons use all 14 datasets. The three SHAP diagnostics (RQ3, RQ4,
RQ5) use 13; JM1 is excluded because the exact TreeExplainer pass inside the
wrapper search did not finish in a practical budget for those
repeated-configuration designs. The cross-dataset feature-importance analysis
uses the 9 procedural-metric datasets, since the CK metrics share no feature
names with them.

**Preprocessing is fitted on the training fold only.** Feature selection,
standardization and oversampling all happen inside the CV loop.

**Seeds are fixed** (`RANDOM_STATE = 42`) and per-fold output is committed, so
every reported number is traceable to the run that produced it. Package versions
are pinned in `requirements.txt`; scikit-learn and imbalanced-learn drive the
resampling RNG, so a different release can shift individual fold scores slightly.

**Negative results are reported as such.** The Burak nearest-neighbour filter did
not recover the cross-project gap, and ABF is matched rather than exceeded by
BalancedRF on G-mean; both are published here in full.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE). The datasets are covered by their own licences;
see [`data/README.md`](data/README.md) for their sources.
