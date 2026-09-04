# Data

The raw defect datasets are **not redistributed here**. All three collections are
publicly available from their original maintainers, and mirroring them would
only risk this copy drifting from the canonical one. Download them into the
layout below and every script in the repository will find them; the paths are
declared in [`src/utils.py`](../src/utils.py) (`DATASETS`).

## Expected layout

```
data/raw/
├── promise/                                   # PROMISE repository, ARFF
│   ├── cm1.arff  jm1.arff  kc1.arff  kc2.arff  pc1.arff
├── nasa_mdp/
│   └── CleanedData/MDP/D''/                   # Shepperd et al. cleaned, D'' variant
│       ├── MW1.arff  PC3.arff  PC4.arff  MC2.arff
└── defectdata/
    └── inst/extdata/terapromise/ck/           # Jureczko & Madeyski CK metrics
        ├── ant-1.7.csv  camel-1.6.csv  tomcat.csv  xalan-2.4.csv  synapse-1.2.csv
```

## Sources

| Group | Origin |
|---|---|
| PROMISE | NASA datasets as distributed by the PROMISE Software Engineering Repository, `http://promise.site.uottawa.ca/SERepository/datasets-page.html` |
| NASA MDP | Cleaned NASA MDP data of Shepperd et al. (2014), **D''** variant (mirror maintained by C. Tantithamthavorn) |
| Java CK | `DefectData` R package, `github.com/klainfo/DefectData`; TeraPROMISE CK metrics collected by Jureczko & Madeyski |

Two NASA projects appear in both the PROMISE and the NASA MDP collections, and
the two differ: Shepperd et al.'s cleaning removes inconsistent and duplicated
modules, so their module counts are lower than the PROMISE originals. The table
below records which source each dataset was actually loaded from, so the counts
are unambiguous. PC1 comes from PROMISE; MW1, PC3, PC4 and MC2 come from the
cleaned D'' data.

## Dataset characteristics

Counts below were produced by loading each file through `src/utils.py`
(`load_defect_dataset`), i.e. after metadata columns are dropped and the target
is binarised, i.e. the same state the experiments see. `bug > 0` is treated as
defective for the CK datasets.

| Group | Dataset | Instances | Features | Defective | Defect rate |
|---|---|---:|---:|---:|---:|
| PROMISE | CM1 | 498 | 21 | 49 | 9.8% |
| PROMISE | JM1 | 10885 | 21 | 2106 | 19.3% |
| PROMISE | KC1 | 2109 | 21 | 326 | 15.5% |
| PROMISE | KC2 | 522 | 21 | 107 | 20.5% |
| PROMISE | PC1 | 1109 | 21 | 77 | 6.9% |
| NASA MDP | MW1 | 250 | 37 | 25 | 10.0% |
| NASA MDP | PC3 | 1053 | 37 | 130 | 12.3% |
| NASA MDP | PC4 | 1270 | 37 | 176 | 13.9% |
| NASA MDP | MC2 | 124 | 39 | 44 | 35.5% |
| Java CK | ant-1.7 | 745 | 20 | 166 | 22.3% |
| Java CK | camel-1.6 | 965 | 20 | 188 | 19.5% |
| Java CK | tomcat | 858 | 20 | 77 | 9.0% |
| Java CK | xalan-2.4 | 723 | 20 | 110 | 15.2% |
| Java CK | synapse-1.2 | 256 | 20 | 86 | 33.6% |

## Notes on dataset selection

**One version per project.** The CK collection contains several releases of each
Java project (ant-1.3 … ant-1.7 and so on). Only one release per project is used.
Successive releases of the same codebase share most of their modules, so counting
them as separate datasets would inflate *n* while breaking the independence
assumption the paired tests in `scripts/statistical_analysis.py` rely on.

**JM1 is excluded from the SHAP diagnostics.** RQ3, RQ4 and RQ5 run on 13 of the
14 datasets. JM1 has 10,885 modules and the exact TreeExplainer pass inside the
Stage 1 wrapper search did not complete in a practical time budget for the
repeated-configuration designs those RQs require. Every *n* is printed by the
analysis scripts alongside the result, and the paper states it per experiment.

**Feature schemas are not comparable across groups.** The McCabe/Halstead
procedural metrics and the CK object-oriented metrics share no feature names.
Cross-project transfer (`rq7_robustness.py --part cross-project`) is therefore
run within a group, never across groups, and the cross-dataset feature-importance
analysis is restricted to the nine procedural-metric datasets.
