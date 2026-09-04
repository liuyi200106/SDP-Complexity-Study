"""Dataset registry and loaders.

Loads PROMISE and NASA MDP ARFF files and Jureczko CK metric CSVs into a common
(X, y) form, and defines DATASETS and DATASET_GROUPS, which every experiment
imports.

The ARFF files in this collection share one shape: numeric software metrics
followed by a single nominal class attribute in the last column, whose values vary
across files (true/false, Y/N, yes/no). The header is parsed here rather than
through a dependency, since the format is a flat table with no sparse or
relational features.

See data/README.md for where to obtain the raw files.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd


def load_arff(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    attr_names = []
    data_start = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("@attribute"):
            parts = stripped.split(None, 2)
            name = parts[1].strip("'\"")
            attr_names.append(name)
        elif low.startswith("@data"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"No @data section found in {path}")

    data_lines = [
        l.strip() for l in lines[data_start:]
        if l.strip() and not l.strip().startswith("%")
    ]
    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=attr_names)
    return df


def load_jureczko_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load a Jureczko/Madeyski CK-metric CSV (Java open-source projects).

    Layout differs from the ARFF files: three leading metadata columns
    (project name, version, fully-qualified class name -- pandas renames the
    duplicated 'name' header to 'name.1'), then 20 numeric CK/OO metrics, then
    a 'bug' column holding the DEFECT COUNT rather than a binary label. The
    count is binarised the standard way for this benchmark (bug > 0 -> 1),
    matching how these datasets are used throughout the SDP literature.
    """
    df = pd.read_csv(path)

    metadata_cols = [c for c in ("name", "version", "name.1") if c in df.columns]
    if "bug" not in df.columns:
        raise ValueError(f"{path}: expected a 'bug' column in a Jureczko CSV")

    X = df.drop(columns=metadata_cols + ["bug"]).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = (pd.to_numeric(df["bug"], errors="coerce").fillna(0) > 0).astype(int)

    return X.reset_index(drop=True), y.reset_index(drop=True), "bug"


def load_defect_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Return (X, y, label_name) with y encoded as 0/1, defect class = 1.

    Dispatches on file type: .csv is a Jureczko CK-metric file, anything else
    is one of the PROMISE / NASA MDP ARFF files. For ARFF, the defect class is
    always the minority class, so that is used instead of hard-coding every
    label spelling (true/Y/yes/...).
    """
    if Path(path).suffix.lower() == ".csv":
        return load_jureczko_csv(path)

    df = load_arff(path)
    label_col = df.columns[-1]
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    raw_label = df[label_col]
    if raw_label.dtype == object:
        counts = raw_label.value_counts()
        minority_value = counts.idxmin()
        y = (raw_label == minority_value).astype(int)
    else:
        counts = raw_label.value_counts()
        if set(counts.index) <= {0, 1}:
            y = raw_label.astype(int)
        else:
            minority_value = counts.idxmin()
            y = (raw_label == minority_value).astype(int)

    return X.reset_index(drop=True), y.reset_index(drop=True), label_col


DATASETS = {
    "CM1":  "data/raw/promise/cm1.arff",
    "JM1":  "data/raw/promise/jm1.arff",
    "KC1":  "data/raw/promise/kc1.arff",
    "KC2":  "data/raw/promise/kc2.arff",
    "PC1":  "data/raw/promise/pc1.arff",
    "MW1":  "data/raw/nasa_mdp/CleanedData/MDP/D''/MW1.arff",
    "PC3":  "data/raw/nasa_mdp/CleanedData/MDP/D''/PC3.arff",
    "PC4":  "data/raw/nasa_mdp/CleanedData/MDP/D''/PC4.arff",
    "MC2":  "data/raw/nasa_mdp/CleanedData/MDP/D''/MC2.arff",
    "ant-1.7":      "data/raw/defectdata/inst/extdata/terapromise/ck/ant-1.7.csv",
    "camel-1.6":    "data/raw/defectdata/inst/extdata/terapromise/ck/camel-1.6.csv",
    "tomcat":       "data/raw/defectdata/inst/extdata/terapromise/ck/tomcat.csv",
    "xalan-2.4":    "data/raw/defectdata/inst/extdata/terapromise/ck/xalan-2.4.csv",
    "synapse-1.2":  "data/raw/defectdata/inst/extdata/terapromise/ck/synapse-1.2.csv",
}

DATASET_GROUPS = {
    "PROMISE": ["CM1", "JM1", "KC1", "KC2", "PC1"],
    "NASA": ["MW1", "PC3", "PC4", "MC2"],
    "Java": ["ant-1.7", "camel-1.6", "tomcat", "xalan-2.4", "synapse-1.2"],
}
