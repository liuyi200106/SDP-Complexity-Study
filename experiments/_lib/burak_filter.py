
"""Burak instance filter (Turhan et al., 2009) for cross-project defect prediction.

For every target-project instance, only the k nearest source instances in the
selected-feature space are kept; their union over all target instances becomes the
training set, which sits distributionally closer to the target than the full
source project. This addresses the generalization gap measured in
cross_project.py: within-project F1 around 0.44 against cross-project F1 of 0.19
to 0.27.

Compares NoAdapt (training on the whole source project, read from
results/summary/cross_project.csv) against BurakFilter on identical
source-to-target pairs, with the same selected features, the same Stage 2 and the
same classifier. Only the choice of source rows differs.

The paper reports this as a negative result.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._lib.cross_project import (NASA_GROUP, PROMISE_GROUP, RANDOM_STATE,
                                            load_group, score)
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def burak_filter_indices(X_source: pd.DataFrame, X_target: pd.DataFrame, k: int = 10) -> list[int]:
    nn = NearestNeighbors(n_neighbors=min(k, len(X_source))).fit(X_source.values)
    _, idx = nn.kneighbors(X_target.values)
    return sorted(set(idx.flatten().tolist()))


def run_group(group_name: str, group: list[str], canonicalize: bool, k: int = 10) -> pd.DataFrame:
    data = load_group(group, canonicalize)
    rows = []

    for source in group:
        X_src_full, y_src_full = data[source]
        print(f"  [{group_name}] Stage1 on source={source} (shared across its targets)", flush=True)
        selected, importances, _est, _ = two_stage_feature_selection(
            X_src_full, y_src_full, corr_threshold=0.90, min_features=3,
            wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
        )
        X_src_sel = X_src_full[selected]

        for target in group:
            if target == source:
                continue
            X_tgt_full, y_tgt = data[target]
            X_tgt_sel = X_tgt_full[selected]

            keep_idx = burak_filter_indices(X_src_sel, X_tgt_sel, k=k)
            X_src_filtered = X_src_sel.iloc[keep_idx]
            y_src_filtered = y_src_full.iloc[keep_idx]

            if y_src_filtered.nunique() < 2:
                print(f"  [skip] {source}->{target}: Burak filter left only one class", flush=True)
                continue

            fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
            X_bal, y_bal = fib.fit_resample(X_src_filtered, y_src_filtered)

            clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=4)
            clf.fit(X_bal, y_bal)
            result = score(clf, X_tgt_sel, y_tgt)
            rows.append({
                "group": group_name, "source": source, "target": target,
                "n_source_filtered": len(keep_idx), "n_source_total": len(X_src_sel),
                **result,
            })

    return pd.DataFrame(rows)


def main():
    out_dir = PROJECT_ROOT / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for group_name, group, canonicalize in [("PROMISE", PROMISE_GROUP, True), ("NASA", NASA_GROUP, False)]:
        print(f"[running group] {group_name} Burak-filter transfer learning", flush=True)
        df = run_group(group_name, group, canonicalize, k=10)
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_dir / "burak_filter.csv", index=False, encoding="utf-8-sig")

    print("\n=== Burak-filtered cross-project results ===")
    print(result.round(3).to_string(index=False))
    print("\n=== Mean by group (Burak-filtered) ===")
    print(result.groupby("group")[["F1", "AUC", "MCC", "Gmean"]].mean().round(3).to_string())

    baseline_path = out_dir / "cross_project.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        merged = result.merge(baseline, on=["group", "source", "target"], suffixes=("_burak", "_noadapt"))
        print("\n=== Improvement from Burak filter vs. no adaptation (mean across all pairs) ===")
        for m in ["F1", "AUC", "MCC", "Gmean"]:
            diff = (merged[f"{m}_burak"] - merged[f"{m}_noadapt"]).mean()
            print(f"  {m}: {diff:+.4f}")
        merged.to_csv(out_dir / "burak_vs_no_adaptation.csv", index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_dir / 'burak_vs_no_adaptation.csv'}")

    print(f"Saved: {out_dir / 'burak_filter.csv'}")


if __name__ == "__main__":
    main()
