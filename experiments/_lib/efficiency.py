
"""Per-stage wall-clock cost of the pipeline against a plain baseline classifier.

Reports training time by stage and in total, inference time, and how both scale
with dataset size (n_samples, n_features), so the accuracy differences reported
elsewhere can be weighed against what they cost to obtain.

Timing uses one 70/30 split per dataset rather than the full 5-fold CV, since a
realistic-scale measurement does not need the cross-validation structure. Each
stage's time is the median of 3 repeated measurements, which smooths OS
scheduling noise. No run is discarded as warm-up, as neither sklearn nor XGBoost
JIT-compiles.
"""
from __future__ import annotations

import statistics
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATASETS, load_defect_dataset
from src.feature_selection import two_stage_feature_selection
from src.fib_smote import FIBSmote

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
N_REPEATS = 3


def timed(fn, n_repeats: int = N_REPEATS):
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, statistics.median(times)


def profile_dataset(name: str, path: Path) -> dict:
    X, y, _ = load_defect_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

    def fit_baseline():
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        clf.fit(X_train, y_train)
        return clf

    baseline_clf, baseline_train_time = timed(fit_baseline)
    _, baseline_predict_time = timed(lambda: baseline_clf.predict(X_test))

    def run_stage1():
        return two_stage_feature_selection(
            X_train, y_train, corr_threshold=0.90, min_features=3,
            wrapper_step=2, wrapper_inner_cv_splits=3, random_state=RANDOM_STATE,
        )

    (selected, importances, _est, _), stage1_time = timed(run_stage1)
    X_train_sel, X_test_sel = X_train[selected], X_test[selected]

    def run_stage2():
        fib = FIBSmote(feature_importances=importances, k_neighbors=5, sampling_ratio=1.0, random_state=RANDOM_STATE)
        return fib.fit_resample(X_train_sel, y_train)

    (X_bal, y_bal), stage2_time = timed(run_stage2)

    def fit_final():
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=4)
        clf.fit(X_bal, y_bal)
        return clf

    final_clf, final_train_time = timed(fit_final)
    _, final_predict_time = timed(lambda: final_clf.predict(X_test_sel))

    total_pipeline_train_time = stage1_time + stage2_time + final_train_time

    return {
        "dataset": name,
        "n_samples": len(X),
        "n_features_raw": X.shape[1],
        "n_features_selected": len(selected),
        "n_train_after_balance": len(X_bal),
        "baseline_train_s": round(baseline_train_time, 4),
        "baseline_predict_s": round(baseline_predict_time, 5),
        "stage1_feature_selection_s": round(stage1_time, 4),
        "stage2_adaptive_smote_s": round(stage2_time, 4),
        "final_rf_train_s": round(final_train_time, 4),
        "pipeline_total_train_s": round(total_pipeline_train_time, 4),
        "pipeline_predict_s": round(final_predict_time, 5),
        "overhead_vs_baseline_x": round(total_pipeline_train_time / baseline_train_time, 2) if baseline_train_time > 0 else float("nan"),
    }


def main():
    rows = []
    for name, rel_path in DATASETS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"[skip] {name}")
            continue
        print(f"[profiling] {name} ...", flush=True)
        row = profile_dataset(name, path)
        rows.append(row)
        print(f"  n={row['n_samples']}, features {row['n_features_raw']}->{row['n_features_selected']}, "
              f"baseline_train={row['baseline_train_s']}s, pipeline_train={row['pipeline_total_train_s']}s "
              f"({row['overhead_vs_baseline_x']}x)", flush=True)

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "results" / "summary" / "efficiency_analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== Efficiency summary (sorted by dataset size) ===")
    print(df.sort_values("n_samples")[[
        "dataset", "n_samples", "n_features_raw", "n_features_selected",
        "baseline_train_s", "pipeline_total_train_s", "overhead_vs_baseline_x", "pipeline_predict_s",
    ]].to_string(index=False))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
