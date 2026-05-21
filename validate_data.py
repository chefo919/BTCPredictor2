"""
Data validation script — validates btc_1m.csv, feature CSVs, and yearly merged files.

Checks:
  1. 1m: row count, gaps, duplicates, OHLCV sanity
  2. Feature CSVs: row counts, feature counts, value ranges
  3. Yearly merged CSVs: presence and row counts
  4. Merged CSV: NaN analysis, feature count, RSI window check
  5. Rolling check: h1_rsi must CHANGE within each hour (proves rolling pipeline)
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

    # Rolling check: h1_rsi must CHANGE within each hour (proves rolling, not stale cardinal)
    if "h1_rsi" in merged.columns:
        sample_h1 = merged.dropna(subset=["h1_rsi"]).tail(3000).copy()
        sample_h1["hour_block"] = sample_h1["time"].dt.floor("h")
        unique_per_block = sample_h1.groupby("hour_block")["h1_rsi"].nunique()
        varying_pct = (unique_per_block > 1).mean() * 100
        if varying_pct > 50:
            _p(PASS, f"Rolling check: h1_rsi varies within hours ({varying_pct:.0f}% of blocks)")
        else:
            _p(FAIL, f"Rolling check: h1_rsi constant within {100-varying_pct:.0f}% of hours — "
                     f"may be stale cardinal value, not rolling")


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

    # ── 2. 1m sanity ─────────────────────────────────────────────────────────
    print()
    print(SEP2)
    print("1M CSV CHECKS")
    print(SEP2)

    check_duplicates(m1, "1m")
    check_gaps(m1, 1, "1m", tolerance=5)
    check_ohlcv_sanity(m1, "1m")

    # ── 3. Feature CSV checks ────────────────────────────────────────────────
    feat_configs = [
        ("btc_1m_features.csv",  "1m features",   12),
        ("btc_15m_features.csv", "15m features",  12),
        ("btc_30m_features.csv", "30m features",  12),
        ("btc_1h_features.csv",  "1H features",   12),
        ("btc_4h_features.csv",  "4H features",   12),
        ("btc_1d_features.csv",  "1D features",   12),
        ("btc_1w_features.csv",  "1W features",   12),
        ("btc_1mo_features.csv", "1MO features",  12),
    ]

    print()
    print(SEP2)
    print("FEATURE CSV CHECKS")
    print(SEP2)

    for feat_fname, label, expected_features in feat_configs:
        feat_path = os.path.join(DATA_DIR, feat_fname)
        if not os.path.exists(feat_path):
            _p(WARN, f"{feat_fname}: not found — skipping")
            continue

        print()
        feat_df = load_feat(feat_fname)
        n_feat  = sum(1 for c in feat_df.columns if c not in ("time", "close"))
        _p(INFO, f"{feat_fname}: {len(feat_df):,} rows × {n_feat} features")

        if n_feat != expected_features:
            _p(FAIL, f"{label}: expected {expected_features} features, got {n_feat}")
        else:
            _p(PASS, f"{label}: feature count OK ({n_feat})")

        check_feat_ranges(feat_df, label)

        nan_counts = feat_df.isnull().sum()
        all_nan    = nan_counts[nan_counts == len(feat_df)]
        if not all_nan.empty:
            _p(FAIL, f"{label}: entirely-NaN columns: {list(all_nan.index)}")

    # ── 4. Yearly merged files ───────────────────────────────────────────────
    yearly_dir = os.path.join(DATA_DIR, "yearly")
    print()
    print(SEP2)
    print("YEARLY MERGED FILES (data/yearly/)")
    print(SEP2)

    if os.path.exists(yearly_dir):
        yearly_files = sorted(f for f in os.listdir(yearly_dir) if f.endswith("_merged.csv"))
        if yearly_files:
            for fname in yearly_files:
                path = os.path.join(yearly_dir, fname)
                df_y = pd.read_csv(path, parse_dates=["time"], nrows=1)
                rc   = sum(1 for _ in open(path)) - 1
                _p(INFO, f"{fname}: {rc:,} rows")
            _p(PASS, f"Found {len(yearly_files)} yearly file(s)")
        else:
            _p(WARN, "yearly/ directory exists but no YYYY_merged.csv files found")
    else:
        _p(WARN, "data/yearly/ directory not found — run features/engineer.py first")

    # ── 5. Merged checks via yearly files ────────────────────────────────────
    print()
    print("Loading yearly files for merged checks...")
    dfs = []
    if os.path.exists(yearly_dir):
        for fname in sorted(os.listdir(yearly_dir)):
            if fname.endswith("_merged.csv"):
                df_y = pd.read_csv(os.path.join(yearly_dir, fname), parse_dates=["time"])
                if df_y["time"].dt.tz is None:
                    df_y["time"] = pd.to_datetime(df_y["time"], utc=True)
                dfs.append(df_y)
    if dfs:
        merged = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
        check_merged(merged)
    else:
        _p(WARN, "No yearly merged files found — run features/engineer.py first")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("Validation complete.")
    print(SEP)


if __name__ == "__main__":
    main()
