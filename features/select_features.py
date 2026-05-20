"""
Feature selection using BoostARoota (XGBoost-native Boruta variant).
Saves selected feature list to models/saved/selected_features.json.

Run once before training:
    python features/select_features.py

If boostARoota is not installed:
    pip install boostARoota
Falls back to XGBoost importance percentile if not available.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.engineer import FEATURE_COLS, MERGED

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import HORIZON_META as HORIZON, TRAINING_CUTOFF_DATE as CUTOFF_DATE

SAMPLE_ROWS     = 200_000
LABEL_THRESHOLD = 0.001   # kept for reference, not used in label construction
OUTPUT_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "saved", "selected_features.json")


def run():
    print(f"Loading {MERGED}...", flush=True)
    df = pd.read_csv(MERGED, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)

    cutoff_ts = pd.Timestamp(CUTOFF_DATE, tz="UTC")
    before    = len(df)
    df        = df[df["time"] <= cutoff_ts].copy()
    print(f"Cutoff {CUTOFF_DATE}: {before:,} → {len(df):,} rows", flush=True)

    # Build target matching train.py exactly — all rows, simple up/down
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    df           = df.dropna(subset=["target", "close"])
    df           = df.dropna(subset=FEATURE_COLS)
    print(f"Rows after dropna: {len(df):,}  (label balance: {df['target'].mean():.3f})", flush=True)

    # Use recent tail — more representative of current market regime
    sample = df.tail(SAMPLE_ROWS).copy()
    X = sample[FEATURE_COLS].values.astype(np.float32)
    y = sample["target"].values.astype(int)
    print(f"Running feature selection on {len(sample):,} rows × {len(FEATURE_COLS)} features...", flush=True)

    try:
        from boostARoota import BoostARoota
        selector = BoostARoota(metric="logloss", max_rounds=10, delta=0.1, silent=False)
        selector.fit(X, y)
        kept_mask     = selector.keep_vars_
        kept_features = [f for f, k in zip(FEATURE_COLS, kept_mask) if k]
        method        = "BoostARoota"
    except ImportError:
        print("boostARoota not installed — falling back to XGBoost importance (top 75%)", flush=True)
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.05, subsample=0.8,
                                   eval_metric="logloss", verbosity=0)
        model.fit(X, y)
        importances   = model.feature_importances_
        threshold     = np.percentile(importances[importances > 0], 25)
        kept_features = [f for f, imp in zip(FEATURE_COLS, importances)
                         if imp >= threshold]
        method        = f"XGBoost importance ≥ {threshold:.6f} (top 75%)"

    print(f"\n{method}: selected {len(kept_features)} / {len(FEATURE_COLS)} features")
    print("-" * 50)
    dropped = [f for f in FEATURE_COLS if f not in kept_features]
    print(f"KEPT ({len(kept_features)}):")
    for f in kept_features:
        print(f"  {f}")
    print(f"\nDROPPED ({len(dropped)}):")
    for f in dropped:
        print(f"  {f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as fp:
        json.dump({
            "selected":       kept_features,
            "dropped":        dropped,
            "total_original": len(FEATURE_COLS),
            "total_selected": len(kept_features),
            "method":         method,
            "sample_rows":    len(sample),
            "label_threshold": LABEL_THRESHOLD,
            "cutoff_date":    CUTOFF_DATE,
        }, fp, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    run()
