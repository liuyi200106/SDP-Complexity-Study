
"""SHAP explanation of the final pipeline: configuration C, the SHAP-guided filter
with RFE, FIB-SMOTE and a Random Forest.

Per dataset, Stage 1 and Stage 2 are fit on the full data and the final Random
Forest is trained on the balanced training set. Its predictions are explained on
the original, non-synthetic samples only, so the explanation describes real code
metrics rather than interpolated points.

Writes to results/summary/shap_explanation/:
  <name>_feature_importance.csv   ranked mean |SHAP value| per feature
  <name>_summary.png              SHAP beeswarm summary plot
  cross_dataset_top_features.csv  which metrics recur as top-ranked across
                                  datasets

Restricted to the nine procedural-metric datasets: identifying recurring features
requires comparable metric names, and the Java CK metrics share none with the
McCabe/Halstead family.
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def explain_dataset(name: str, path: Path, out_dir: Path) -> pd.Series:
    X, y, _ = load_defect_dataset(path)

    selected, importances, _est, _ = two_stage_feature_selection(
        X, y, corr_threshold=0.90, min_features=3,
        wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
    )
    X_sel = X[selected]

    fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
    X_bal, y_bal = fib.fit_resample(X_sel, y)

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=4)
    clf.fit(X_bal, y_bal)

    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X_sel, check_additivity=False)
    if isinstance(raw, list):
        matrix = raw[1] if len(raw) > 1 else raw[0]
    elif raw.ndim == 3:
        matrix = raw[:, :, 1] if raw.shape[2] > 1 else raw[:, :, 0]
    else:
        matrix = raw

    mean_abs_shap = pd.Series(
        __import__("numpy").abs(matrix).mean(axis=0), index=selected
    ).sort_values(ascending=False)
    mean_abs_shap.to_csv(out_dir / f"{name}_feature_importance.csv", header=["mean_abs_shap"])

    plt.figure()
    shap.summary_plot(matrix, X_sel, show=False, max_display=15)
    plt.title(f"{name}: SHAP summary (final RF, Stage1+2 pipeline)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_summary.png", dpi=120)
    plt.close()

    print(f"  [{name}] top-5 features by |SHAP|: {mean_abs_shap.head(5).round(4).to_dict()}", flush=True)
    return mean_abs_shap


def main():
    out_dir = PROJECT_ROOT / "results" / "summary" / "shap_explanation"
    out_dir.mkdir(parents=True, exist_ok=True)

    top3_counter = Counter()
    for name, rel_path in DATASETS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"[skip] {name}")
            continue
        print(f"[running] {name} ...", flush=True)
        importance = explain_dataset(name, path, out_dir)
        for feat in importance.head(3).index:
            top3_counter[feat] += 1

    summary = pd.DataFrame(top3_counter.most_common(), columns=["feature", "n_datasets_in_top3"])
    summary.to_csv(out_dir / "cross_dataset_top_features.csv", index=False, encoding="utf-8-sig")
    print("\n=== Features most often in the top-3 across datasets ===")
    print(summary.to_string(index=False))
    print(f"\nSaved per-dataset importances + summary plots to {out_dir}")
    print(f"Saved: {out_dir / 'cross_dataset_top_features.csv'}")


if __name__ == "__main__":
    main()
