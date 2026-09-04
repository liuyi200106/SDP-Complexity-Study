# Data

The raw defect datasets are **not redistributed here**. All of them are publicly
available from their original maintainers, and mirroring them would only risk
this copy drifting from the canonical one. Download them into the layout below
and every script in the repository will find them; the paths are declared in
[`src/utils.py`](../src/utils.py) (`DATASETS`).

The provenance recorded here matches the Data Availability Statement of the
accompanying paper.

## Expected layout

```
data/raw/
├── promise/                                   # PROMISE, ARFF
│   ├── cm1.arff  jm1.arff  kc1.arff  kc2.arff  pc1.arff
├── nasa_mdp/
│   └── CleanedData/MDP/D''/                   # Shepperd et al. cleaned, D'' variant
│       ├── MW1.arff  PC3.arff  PC4.arff  MC2.arff
└── defectdata/
    └── inst/extdata/terapromise/ck/           # Jureczko & Madeyski CK metrics
        ├── ant-1.7.csv  camel-1.6.csv  tomcat.csv  xalan-2.4.csv  synapse-1.2.csv
```

## Sources

**The nine procedural-metric datasets** — PROMISE (CM1, JM1, KC1, KC2, PC1) and
the cleaned NASA MDP D'' variants (MW1, PC3, PC4, MC2) — were retrieved from two
maintained GitHub mirrors:

| Group | Mirror |
|---|---|
| PROMISE | `https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository` |
| NASA MDP (D'') | `https://github.com/klainfo/NASADefectDataset` (maintained by Chakkrit Tantithamthavorn) |

The mirrors are used because the original PROMISE and NASA MDP hosting URLs are
subject to link rot. Both preserve the original file contents. Provenance for the
underlying data is documented in Sayyad Shirabad & Menzies (2005) and Shepperd et
al. (2013).

**The five Java datasets** (ant-1.7, camel-1.6, tomcat, xalan-2.4, synapse-1.2)
come from the Jureczko/Madeyski CK-metric collection as packaged in the
`DefectData` R package (`https://github.com/klainfo/DefectData`, MIT licence,
maintained by Chakkrit Tantithamthavorn; commit `e65993d`, 2018-01-01),
specifically the files `inst/extdata/terapromise/ck/<project>.csv`. These provide
20 CK object-oriented metrics per class plus a `bug` column holding the
post-release defect **count**. The count is binarised in the standard way for
this benchmark (`bug > 0` maps to the positive class). The files are used as
distributed, without additional cleaning.

Two NASA projects appear in both the PROMISE and the NASA MDP collections, and
the two differ: the cleaning of Shepperd et al. removes inconsistent and
duplicated modules, so their module counts are lower than the PROMISE originals.
The table below records which source each dataset was actually loaded from, so
the counts are unambiguous. PC1 comes from PROMISE; MW1, PC3, PC4 and MC2 come
from the cleaned D'' data.

## Dataset characteristics

Counts below were produced by loading each file through `src/utils.py`
(`load_defect_dataset`), i.e. after metadata columns are dropped and the target
is binarised, which is the same state the experiments see.

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
Java project (ant-1.3 ... ant-1.7 and so on). Only one release per project is
used. Successive releases of the same codebase share most of their modules, so
counting them as separate datasets would inflate *n* while breaking the
independence assumption the paired tests in `scripts/statistical_analysis.py`
rely on. This is stated in Section 3.2 of the paper.

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

## References

Sayyad Shirabad, J., & Menzies, T. (2005). *The PROMISE Repository of Software
Engineering Databases.* School of Information Technology and Engineering,
University of Ottawa, Canada.

Shepperd, M., Song, Q., Sun, Z., & Mair, C. (2013). Data quality: Some comments
on the NASA software defect datasets. *IEEE Transactions on Software
Engineering*, 39(9), 1208-1215.

Jureczko, M., & Madeyski, L. (2010). Towards identifying software project
clusters with regard to defect prediction. In *Proceedings of the 6th
International Conference on Predictive Models in Software Engineering (PROMISE)*.
