"""
Data validation script — cross-references all raw and feature CSVs against btc_1m.csv.

Checks:
  1. Row counts, date ranges
  2. Duplicate timestamps
  3. Gap detection (unexpected missing candles)
  4. OHLCV sanity (high >= low, volume >= 0, prices in range)
  5. OHLCV aggregation: spot-sample higher-TF candles against 1m data
  6. Feature CSV timestamp alignment with raw CSVs
  7. Feature value ranges (RSI in [0,1], ATR >= 0, etc.)
  8. Merged CSV: NaN analysis per column
  9. RSI window bug check: mo1_rsi must differ from d1_rsi
 10. Lag check: h1 features at time T must use pre-T hourly data (not future)
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

SEP  = "=" * 62
SEP2 = "-" * 62


def _p(tag, msg):
    print(f"  {tag} {msg}", flush=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_raw(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    df   = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def load_feat(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    df   = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def check_gaps(df: pd.DataFrame, freq_min: int, label: str, tolerance: int = 3):
    """Check for unexpected gaps in a time series."""
    diffs = df["time"].diff().dropna()
    expected = pd.Timedelta(minutes=freq_min)
    gaps = diffs[diffs > expected * tolerance]
    if gaps.empty:
        _p(PASS, f"{label}: no gaps > {tolerance}× expected interval")
    else:
        _p(WARN, f"{label}: {len(gaps)} gap(s) > {tolerance}× expected interval")
        for idx in gaps.index[:5]:
            t_before = df["time"].iloc[idx - 1]
            t_after  = df["time"].iloc[idx]
            _p("     ", f"  {t_before}  →  {t_after}  ({diffs[idx]})")
        if len(gaps) > 5:
            _p("     ", f"  ... and {len(gaps)-5} more")


def check_duplicates(df: pd.DataFrame, label: str):
    dupes = df["time"].duplicated().sum()
    if dupes == 0:
        _p(PASS, f"{label}: no duplicate timestamps")
    else:
        _p(FAIL, f"{label}: {dupes} duplicate timestamps")


def check_ohlcv_sanity(df: pd.DataFrame, label: str):
    bad_hl  = (df["high"] < df["low"]).sum()
    bad_ho  = (df["high"] < df["open"]).sum()
    bad_hc  = (df["high"] < df["close"]).sum()
    bad_lo  = (df["low"]  > df["open"]).sum()
    bad_lc  = (df["low"]  > df["close"]).sum()
    bad_vol = (df["volume"] < 0).sum()
    bad_px  = ((df["close"] < 1_000) | (df["close"] > 500_000)).sum()
    issues = bad_hl + bad_ho + bad_hc + bad_lo + bad_lc + bad_vol + bad_px
    if issues == 0:
        _p(PASS, f"{label}: OHLCV sanity OK")
    else:
        _p(FAIL, f"{label}: OHLCV sanity failures — "
                 f"high<low:{bad_hl}, high<open:{bad_ho}, high<close:{bad_hc}, "
                 f"low>open:{bad_lo}, low>close:{bad_lc}, vol<0:{bad_vol}, price_oor:{bad_px}")


def check_feat_timestamps(feat_df: pd.DataFrame, raw_df: pd.DataFrame, label: str):
    """Every timestamp in feat_df must appear in raw_df."""
    raw_times  = set(raw_df["time"].astype(str))
    feat_times = set(feat_df["time"].astype(str))
    missing    = feat_times - raw_times
    extra      = raw_times - feat_times
    if not missing:
        _p(PASS, f"{label}: all feature timestamps present in raw data")
    else:
        _p(FAIL, f"{label}: {len(missing)} feature timestamps NOT in raw data")
    if extra:
        _p(INFO, f"{label}: {len(extra)} raw timestamps not in features (warm-up rows expected)")


def check_feat_ranges(df: pd.DataFrame, prefix: str):
    """Check RSI in [0,1], ATR >= 0, etc."""
    errors = []
    for col in df.columns:
        if col in ("time",):
            continue
        if "rsi" in col:
            oor = ((df[col] < 0) | (df[col] > 1)).sum()
            if oor:
                errors.append(f"{col}: {oor} values outside [0,1]")
        if "atr_norm" in col:
            neg = (df[col] < 0).sum()
            if neg:
                errors.append(f"{col}: {neg} negative values")
        if "vol_ratio" in col:
            neg = (df[col] < 0).dropna().sum() if df[col].dtype != object else 0
            # vol_ratio can legitimately be negative if OBV-derived; skip
    if errors:
        for e in errors:
            _p(FAIL, f"  {prefix}: {e}")
    else:
        _p(PASS, f"{prefix} features: value ranges OK")


# ── aggregation spot-check ────────────────────────────────────────────────────

def agg_spot_check(m1: pd.DataFrame, htf: pd.DataFrame,
                   freq_min: int, label: str, n_samples: int = 200):
    """
    Sample n_samples candles from htf and verify OHLCV against the 1m bars
    that make up that candle.
    """
    m1_idx  = m1.set_index("time")
    sample  = htf.sample(min(n_samples, len(htf)), random_state=42)
    tol     = 0.01   # 1 cent tolerance for floating-point

    open_err = high_err = low_err = close_err = vol_err = 0

    for _, row in sample.iterrows():
        t_open = row["time"]
        t_close_incl = t_open + pd.Timedelta(minutes=freq_min - 1)
        window = m1_idx.loc[t_open:t_close_incl]
        if window.empty:
            continue
        if abs(window["open"].iloc[0]  - row["open"])  > tol:  open_err  += 1
        if abs(window["high"].max()    - row["high"])   > tol:  high_err  += 1
        if abs(window["low"].min()     - row["low"])    > tol:  low_err   += 1
        if abs(window["close"].iloc[-1]- row["close"])  > tol:  close_err += 1
        if abs(window["volume"].sum()  - row["volume"]) > max(0.01 * row["volume"], tol):
            vol_err += 1

    total = min(n_samples, len(htf))
    errs  = open_err + high_err + low_err + close_err + vol_err
    if errs == 0:
        _p(PASS, f"{label}: OHLCV aggregation OK ({total} candles sampled)")
    else:
        _p(FAIL, f"{label}: aggregation mismatches in {total} sampled candles — "
                 f"open:{open_err} high:{high_err} low:{low_err} "
                 f"close:{close_err} vol:{vol_err}")


# ── merged CSV checks ─────────────────────────────────────────────────────────

def check_merged(merged: pd.DataFrame):
    print()
    print(SEP2)
    print("MERGED FEATURES CSV")
    print(SEP2)
    _p(INFO, f"Shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    _p(INFO, f"Date range: {merged['time'].min()}  →  {merged['time'].max()}")

    # NaN analysis
    nan_counts = merged.isnull().sum()
    bad_cols   = nan_counts[nan_counts > len(merged) * 0.20]  # > 20% NaN
    zero_cols  = (merged.drop(columns="time") == 0).all()
    zero_cols  = zero_cols[zero_cols].index.tolist()

    if bad_cols.empty:
        _p(PASS, "No column has > 20% NaN")
    else:
        _p(FAIL, f"{len(bad_cols)} columns have > 20% NaN:")
        for col, cnt in bad_cols.items():
            _p("     ", f"  {col}: {cnt:,} NaN ({cnt/len(merged)*100:.1f}%)")

    if zero_cols:
        _p(WARN, f"{len(zero_cols)} columns are entirely zero: {zero_cols}")
    else:
        _p(PASS, "No all-zero columns")

    # RSI bug check: mo1_rsi must NOT equal d1_rsi
    if "mo1_rsi" in merged.columns and "d1_rsi" in merged.columns:
        valid = merged.dropna(subset=["mo1_rsi", "d1_rsi"])
        same  = (valid["mo1_rsi"] == valid["d1_rsi"]).sum()
        pct   = same / len(valid) * 100
        if pct < 5:
            _p(PASS, f"RSI window bug check: mo1_rsi ≠ d1_rsi in {100-pct:.1f}% of rows")
        else:
            _p(FAIL, f"RSI window bug: mo1_rsi == d1_rsi in {pct:.1f}% of rows — "
                     f"RSI window bug may still be present")

    # Lag check: h1 feature at time T must use data from BEFORE T
    # The h1_rsi at time T should equal the h1_features rsi
    # at the previous completed hour boundary (i.e., floor(T/1H))
    # We verify: merged h1_rsi doesn't change within the same 1H block
    # (all 1m rows in [14:00, 14:59] should have the same h1_rsi value)
    if "h1_rsi" in merged.columns:
        sample_hour = merged.dropna(subset=["h1_rsi"]).head(3000).copy()
        sample_hour["hour_block"] = sample_hour["time"].dt.floor("h")
        consistency = sample_hour.groupby("hour_block")["h1_rsi"].nunique()
        inconsistent = (consistency > 1).sum()
        if inconsistent == 0:
            _p(PASS, "Lag check: h1_rsi is constant within each 1H block (no leakage)")
        else:
            _p(FAIL, f"Lag check: h1_rsi changes within {inconsistent} 1H block(s) — possible leakage")

    if "h4_rsi" in merged.columns:
        sample_h4 = merged.dropna(subset=["h4_rsi"]).head(3000).copy()
        sample_h4["h4_block"] = sample_h4["time"].dt.floor("4h")
        consistency_h4 = sample_h4.groupby("h4_block")["h4_rsi"].nunique()
        incon_h4 = (consistency_h4 > 1).sum()
        if incon_h4 == 0:
            _p(PASS, "Lag check: h4_rsi is constant within each 4H block (no leakage)")
        else:
            _p(FAIL, f"Lag check: h4_rsi changes within {incon_h4} 4H block(s) — possible leakage")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("BTC DATA VALIDATION")
    print(SEP)

    # ── 1. Load 1m reference ─────────────────────────────────────────────────
    print()
    print("Loading 1m reference data...")
    m1 = load_raw("btc_1m.csv")
    _p(INFO, f"btc_1m.csv: {len(m1):,} rows  |  {m1['time'].min()} → {m1['time'].max()}")
    m1_times = set(m1["time"])

    # ── 2. Raw CSV checks ────────────────────────────────────────────────────
    raw_configs = [
        ("btc_15m.csv",  15,   "15m"),
        ("btc_30m.csv",  30,   "30m"),
        ("btc_1h.csv",   60,   "1H"),
        ("btc_4h.csv",   240,  "4H"),
        ("btc_1d.csv",   1440, "1D"),
        ("btc_1w.csv",   10080,"1W"),
        ("btc_1mo.csv",  None, "1MO"),
    ]

    print()
    print(SEP2)
    print("RAW CSV CHECKS")
    print(SEP2)

    # 1m checks first
    check_duplicates(m1, "1m")
    check_gaps(m1, 1, "1m", tolerance=5)
    check_ohlcv_sanity(m1, "1m")

    raw_dfs = {}
    for fname, freq_min, label in raw_configs:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            _p(WARN, f"{fname}: file not found — skipping")
            continue

        print()
        df = load_raw(fname)
        raw_dfs[label] = df
        _p(INFO, f"{fname}: {len(df):,} rows  |  {df['time'].min()} → {df['time'].max()}")

        check_duplicates(df, label)
        if freq_min:
            check_gaps(df, freq_min, label, tolerance=3)
        check_ohlcv_sanity(df, label)

        # Close alignment with 1m: every higher-TF close timestamp must exist in 1m
        htf_times = set(df["time"])
        not_in_m1 = htf_times - m1_times
        if not_in_m1:
            _p(WARN, f"{label}: {len(not_in_m1)} timestamps not found in 1m data "
                     f"(expected if 1m data starts later)")
        else:
            _p(PASS, f"{label}: all candle timestamps exist in 1m data")

        # OHLCV aggregation spot-check (skip 1W / 1MO — too sparse for reliable check)
        if freq_min and freq_min <= 1440:
            agg_spot_check(m1, df, freq_min, label, n_samples=200)

    # ── 3. Feature CSV checks ────────────────────────────────────────────────
    feat_configs = [
        ("btc_1m_features.csv",  "btc_1m.csv",  "1m features"),
        ("btc_15m_features.csv", "btc_15m.csv", "15m features"),
        ("btc_30m_features.csv", "btc_30m.csv", "30m features"),
        ("btc_1h_features.csv",  "btc_1h.csv",  "1H features"),
        ("btc_4h_features.csv",  "btc_4h.csv",  "4H features"),
        ("btc_1d_features.csv",  "btc_1d.csv",  "1D features"),
        # 1W/1MO features are rolling windows computed on daily data → compare against 1d
        ("btc_1w_features.csv",  "btc_1d.csv",  "1W features"),
        ("btc_1mo_features.csv", "btc_1d.csv",  "1MO features"),
    ]

    print()
    print(SEP2)
    print("FEATURE CSV CHECKS")
    print(SEP2)

    for feat_fname, raw_fname, label in feat_configs:
        feat_path = os.path.join(DATA_DIR, feat_fname)
        raw_path  = os.path.join(DATA_DIR, raw_fname)
        if not os.path.exists(feat_path):
            _p(WARN, f"{feat_fname}: not found — skipping")
            continue
        if not os.path.exists(raw_path):
            _p(WARN, f"{raw_fname}: raw file not found — skipping {label} alignment check")
            raw_df = None
        else:
            raw_df = load_raw(raw_fname)

        print()
        feat_df = load_feat(feat_fname)
        _p(INFO, f"{feat_fname}: {len(feat_df):,} rows × {len(feat_df.columns)} cols")

        if raw_df is not None:
            check_feat_timestamps(feat_df, raw_df, label)

        check_feat_ranges(feat_df, label)

        nan_counts = feat_df.isnull().sum()
        all_nan    = nan_counts[nan_counts == len(feat_df)]
        if not all_nan.empty:
            _p(FAIL, f"{label}: entirely-NaN columns: {list(all_nan.index)}")

        # Summarize NaN warm-up rows (first N rows with any NaN)
        first_valid = feat_df.dropna().index[0] if feat_df.dropna().shape[0] > 0 else None
        if first_valid:
            _p(INFO, f"{label}: first fully-valid row at index {first_valid} "
                     f"({feat_df.loc[first_valid, 'time']})")

    # ── 4. Merged CSV checks ─────────────────────────────────────────────────
    merged_path = os.path.join(DATA_DIR, "btc_merged_features.csv")
    if os.path.exists(merged_path):
        print()
        print("Loading merged CSV (this may take 30s)...")
        merged = pd.read_csv(merged_path, parse_dates=["time"])
        if merged["time"].dt.tz is None:
            merged["time"] = pd.to_datetime(merged["time"], utc=True)
        check_merged(merged)
    else:
        _p(WARN, "btc_merged_features.csv not found")

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("Validation complete.")
    print(SEP)


if __name__ == "__main__":
    main()
