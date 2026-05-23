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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from features.engineer import BILSTM_FEAT_COLS, TFT_FEAT_COLS

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import HORIZON_META as HORIZON, TRAINING_CUTOFF_DATE as CUTOFF_DATE

ALL_FEATURES    = TFT_FEAT_COLS + BILSTM_FEAT_COLS   # 48 features
SAMPLE_ROWS     = 200_000
LABEL_THRESHOLD = 0.001   # kept for reference, not used in label construction
OUTPUT_PATH      = os.path.join(ROOT_DIR, "models", "saved", "selected_features.json")
YEARLY_DIR       = os.path.join(ROOT_DIR, "data", "yearly_merged")


def _load_yearly(pattern: str) -> pd.DataFrame:
    now_year = pd.Timestamp.now(tz="UTC").year
    dfs = []
    for y in range(now_year - 6, now_year + 1):
        path = os.path.join(YEARLY_DIR, f"{y}_{pattern}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["time"])
            if df["time"].dt.tz is None:
                df["time"] = pd.to_datetime(df["time"], utc=True)
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No {pattern} files found in {YEARLY_DIR}. Run features/engineer.py.")
    return (pd.concat(dfs)
              .sort_values("time")
              .drop_duplicates("time")
              .reset_index(drop=True))


def run():
    print("Loading tft_merged and bilstm_merged...", flush=True)
    df_tft    = _load_yearly("tft_merged")
    df_bilstm = _load_yearly("bilstm_merged")

    # Align bilstm features onto tft timestamps (4h resolution) via nearest join
    df = pd.merge_asof(
        df_tft.sort_values("time"),
        df_bilstm[["time"] + BILSTM_FEAT_COLS].sort_values("time"),
        on="time", direction="nearest", tolerance=pd.Timedelta("3h"),
    )

    cutoff_ts = pd.Timestamp(CUTOFF_DATE, tz="UTC")
    before    = len(df)
    df        = df[df["time"] <= cutoff_ts].copy()
    print(f"Cutoff {CUTOFF_DATE}: {before:,} → {len(df):,} rows", flush=True)

    # Build target matching train.py exactly — all rows, simple up/down
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    df           = df.dropna(subset=["target", "close"])
    df           = df.dropna(subset=ALL_FEATURES)
    print(f"Rows after dropna: {len(df):,}  (label balance: {df['target'].mean():.3f})", flush=True)

    # Use recent tail — more representative of current market regime
    sample = df.tail(SAMPLE_ROWS).copy()
    X = sample[ALL_FEATURES].values.astype(np.float32)
    y = sample["target"].values.astype(int)
    print(f"Running feature selection on {len(sample):,} rows × {len(ALL_FEATURES)} features...", flush=True)

    try:
        from boostARoota import BoostARoota
        selector = BoostARoota(metric="logloss", max_rounds=10, delta=0.1, silent=False)
        selector.fit(X, y)
        kept_mask     = selector.keep_vars_
        kept_features = [f for f, k in zip(ALL_FEATURES, kept_mask) if k]
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
        kept_features = [f for f, imp in zip(ALL_FEATURES, importances)
                         if imp >= threshold]
        method        = f"XGBoost importance ≥ {threshold:.6f} (top 75%)"

    print(f"\n{method}: selected {len(kept_features)} / {len(ALL_FEATURES)} features")
    print("-" * 50)
    dropped = [f for f in ALL_FEATURES if f not in kept_features]
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
            "total_original": len(ALL_FEATURES),
            "total_selected": len(kept_features),
            "method":         method,
            "sample_rows":    len(sample),
            "label_threshold": LABEL_THRESHOLD,
            "cutoff_date":    CUTOFF_DATE,
        }, fp, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    run()
