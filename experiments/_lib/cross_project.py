
"""Cross-project defect prediction (CPDP).

Within-project results train and test on rows drawn from the same project, which
does not measure whether the pipeline generalizes to a project it has never seen.
That case, a new project with no labelled defect history, is a standard robustness
check in the SDP literature.

PROMISE and NASA MDP datasets use nearly the same 21 and 37 code metrics, but
column names differ across files by capitalization or by an aliased spelling. The
mapping in _canonicalize_promise_columns was read off each file's @attribute list
rather than inferred, so features line up across projects. The NASA group needs
only a column intersection, since MC2 carries two columns the others lack.

For each ordered (train_project, test_project) pair within a schema group, Stage 1
and Stage 2 are fit once per source project and reused across that project's
targets; a Random Forest trained on the source is then evaluated on the untouched
target.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

PROMISE_GROUP = ["CM1", "JM1", "KC1", "KC2", "PC1"]
NASA_GROUP = ["MW1", "PC3", "PC4", "MC2"]

_PROMISE_ALIASES = {"locodeandcomment": "loccodeandcomment"}


def _canonicalize_promise_columns(X: pd.DataFrame) -> pd.DataFrame:
    def canon(col: str) -> str:
        low = col.lower()
        return _PROMISE_ALIASES.get(low, low)

    return X.rename(columns=canon)


def g_mean(y_true, y_pred) -> float:
    rp = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rn = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return (rp * rn) ** 0.5


def score(clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan")
    return {
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Gmean": g_mean(y_test, y_pred),
    }


def load_group(group: list[str], canonicalize: bool) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    data = {}
    for name in group:
        X, y, _ = load_defect_dataset(PROJECT_ROOT / DATASETS[name])
        if canonicalize:
            X = _canonicalize_promise_columns(X)
        data[name] = (X, y)

    common_cols = set.intersection(*(set(X.columns) for X, _ in data.values()))
    common_cols = sorted(common_cols)
    return {name: (X[common_cols], y) for name, (X, y) in data.items()}


def run_group(group_name: str, group: list[str], canonicalize: bool) -> pd.DataFrame:
    data = load_group(group, canonicalize)
    rows = []

    for source in group:
        X_src, y_src = data[source]
        print(f"  [{group_name}] fitting source={source} (Stage1+Stage2 once, reused for all targets)", flush=True)
        selected, importances, _est, _ = two_stage_feature_selection(
            X_src, y_src, corr_threshold=0.90, min_features=3,
            wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
        )
        fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
        X_bal, y_bal = fib.fit_resample(X_src[selected], y_src)

        clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=4)
        clf.fit(X_bal, y_bal)

        for target in group:
            if target == source:
                continue
            X_tgt, y_tgt = data[target]
            result = score(clf, X_tgt[selected], y_tgt)
            rows.append({"group": group_name, "source": source, "target": target, "n_features": len(selected), **result})

    return pd.DataFrame(rows)


def main():
    out_dir = PROJECT_ROOT / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for group_name, group, canonicalize in [
        ("PROMISE", PROMISE_GROUP, True),
        ("NASA", NASA_GROUP, False),
    ]:
        print(f"[running group] {group_name}: {group}", flush=True)
        df = run_group(group_name, group, canonicalize)
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_dir / "cross_project.csv", index=False, encoding="utf-8-sig")

    print("\n=== Cross-project results (all source->target pairs) ===")
    print(result.round(3).to_string(index=False))

    print("\n=== Mean metrics per group (cross-project) ===")
    print(result.groupby("group")[["F1", "AUC", "MCC", "Gmean"]].mean().round(3).to_string())

    print(f"\nSaved: {out_dir / 'robustness_cross_project.csv'}")


if __name__ == "__main__":
    main()
