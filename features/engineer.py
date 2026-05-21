import os
import json
import numpy as np
import pandas as pd
import ta
from typing import Optional

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
YEARLY_DIR = os.path.join(DATA_DIR, "yearly")

PATH_1M = os.path.join(DATA_DIR, "btc_1m.csv")

FEAT_1M  = os.path.join(DATA_DIR, "btc_1m_features.csv")
FEAT_15M = os.path.join(DATA_DIR, "btc_15m_features.csv")
FEAT_30M = os.path.join(DATA_DIR, "btc_30m_features.csv")
FEAT_1H  = os.path.join(DATA_DIR, "btc_1h_features.csv")
FEAT_4H  = os.path.join(DATA_DIR, "btc_4h_features.csv")
FEAT_1D  = os.path.join(DATA_DIR, "btc_1d_features.csv")

_BASE = ["rsi", "macd_diff_pct",
         "ema9_ratio", "ema21_ratio", "ema50_ratio",
         "atr_norm", "bb_pct", "bb_width",
         "obv_zscore", "vol_ratio",
         "body_ratio", "adx"]

FEATURE_1M  = list(_BASE)
FEATURE_15M = [f"m15_{c}" for c in _BASE]
FEATURE_30M = [f"m30_{c}" for c in _BASE]
FEATURE_1H  = [f"h1_{c}"  for c in _BASE]
FEATURE_4H  = [f"h4_{c}"  for c in _BASE]
FEATURE_1D  = [f"d1_{c}"  for c in _BASE]
FEATURE_COLS = (FEATURE_1M + FEATURE_15M + FEATURE_30M
                + FEATURE_1H + FEATURE_4H + FEATURE_1D)
# Total: 12 indicators x 6 timeframes = 72 features, all genuinely distinct
# Each timeframe uses windows scaled by tf_mult so RSI/EMA/etc. measure
# the same number of *periods* at each timeframe's own resolution.


def get_feature_groups() -> dict:
    """
    Feature split for the TFT-ACB-XML architecture.

    BiLSTM (micro, SEQ=240):  1m/15m/30m  — 36 features
    TFT    (macro, SEQ=1440): 1H/4H/1D    — 36 features, all dynamic (no static covariates)
    """
    return {
        "bilstm":      FEATURE_1M + FEATURE_15M + FEATURE_30M,   # 36
        "tft_dynamic": FEATURE_1H  + FEATURE_4H  + FEATURE_1D,   # 36
        "tft_static":  [],                                         # none
    }


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _clean(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    for col in feat_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).astype("float64")
    return df.dropna(subset=feat_cols).reset_index(drop=True)


# ── Rolling OHLCV from 1m data ────────────────────────────────────────────────

def _rolling_ohlcv_series(df_1m: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    At each 1m row T:
      open   = open of candle T-(window-1)  [oldest in window]
      high   = rolling max high over window
      low    = rolling min low  over window
      close  = close(T)                     [current — no shift, no leakage]
      volume = rolling sum of volume over window
    First (window-1) rows are NaN and dropped.
    """
    return pd.DataFrame({
        "time":   df_1m["time"].values,
        "open":   df_1m["open"].shift(window - 1),
        "high":   df_1m["high"].rolling(window).max(),
        "low":    df_1m["low"].rolling(window).min(),
        "close":  df_1m["close"],
        "volume": df_1m["volume"].rolling(window).sum(),
    }).dropna().reset_index(drop=True)


# ── 12 indicators with timeframe-scaled windows ───────────────────────────────

def _compute_std_indicators(df: pd.DataFrame, prefix: str = "",
                             tf_mult: int = 1) -> pd.DataFrame:
    """
    Compute all 12 indicators with windows scaled by tf_mult so that each
    timeframe measures the same *number of periods* at its own resolution.

    Examples:
      tf_mult=1   (1m):  RSI(14),  EMA(9),   BB(20)
      tf_mult=15  (15m): RSI(210), EMA(135), BB(300)
      tf_mult=60  (1H):  RSI(840), EMA(540), BB(1200)
      tf_mult=1440(1D):  RSI(20160), EMA(12960), BB(28800)

    This makes m15_rsi genuinely reflect 14 fifteen-minute candles of momentum,
    h4_rsi genuinely reflect 14 four-hour candles, etc.
    """
    result = df[["time"]].copy()
    if prefix == "" and "close" in df.columns:
        result["close"] = df["close"].values

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    open_  = df["open"].astype(float)
    volume = df["volume"].astype(float)
    p = prefix

    w_rsi  = 14 * tf_mult
    w_ema9 = 9  * tf_mult
    w_ema21= 21 * tf_mult
    w_ema50= 50 * tf_mult
    w_atr  = 14 * tf_mult
    w_bb   = 20 * tf_mult
    w_obv  = 50 * tf_mult
    w_vol  = 20 * tf_mult
    w_adx  = 14 * tf_mult
    # MACD: fast=12*tf_mult, slow=26*tf_mult, signal=9*tf_mult
    w_fast = 12 * tf_mult
    w_slow = 26 * tf_mult
    w_sig  = 9  * tf_mult

    result[f"{p}rsi"]           = ta.momentum.RSIIndicator(close, window=w_rsi).rsi() / 100.0

    macd_fast = close.ewm(span=w_fast, adjust=False).mean()
    macd_slow = close.ewm(span=w_slow, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.ewm(span=w_sig, adjust=False).mean()
    result[f"{p}macd_diff_pct"] = (macd_line - macd_signal) / close

    result[f"{p}ema9_ratio"]    = ta.trend.EMAIndicator(close, window=w_ema9).ema_indicator()  / close - 1
    result[f"{p}ema21_ratio"]   = ta.trend.EMAIndicator(close, window=w_ema21).ema_indicator() / close - 1
    result[f"{p}ema50_ratio"]   = ta.trend.EMAIndicator(close, window=w_ema50).ema_indicator() / close - 1

    result[f"{p}atr_norm"]      = ta.volatility.AverageTrueRange(
                                      high, low, close, window=w_atr).average_true_range() / close

    bb = ta.volatility.BollingerBands(close, window=w_bb, window_dev=2)
    result[f"{p}bb_pct"]        = bb.bollinger_pband()
    result[f"{p}bb_width"]      = (bb.bollinger_hband() - bb.bollinger_lband()) / close

    obv      = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_mean = obv.rolling(w_obv, min_periods=1).mean()
    obv_std  = obv.rolling(w_obv, min_periods=1).std().replace(0, np.nan)
    result[f"{p}obv_zscore"]    = (obv - obv_mean) / obv_std

    vol_ma = volume.rolling(w_vol, min_periods=1).mean().replace(0, np.nan)
    result[f"{p}vol_ratio"]     = volume / vol_ma

    # body_ratio: uses rolling open (open of candle T-window+1) and rolling h/l
    # — already captures the full-timeframe candle body, no extra scaling needed
    result[f"{p}body_ratio"]    = (close - open_) / (high - low).replace(0, np.nan)

    result[f"{p}adx"]           = ta.trend.ADXIndicator(
                                      high, low, close, window=w_adx).adx() / 100.0

    feat_cols = [c for c in result.columns if c != "time" and c != "close"]
    return _clean(result, feat_cols)


# ── Build individual feature files ────────────────────────────────────────────
# tf_mult = number of 1m bars in one period of that timeframe

_TF_TASKS = [
    (FEAT_1M,  "1m",  1,    lambda m1, m: _compute_std_indicators(m1, "",      1)),
    (FEAT_15M, "15m", 15,   lambda m1, m: _compute_std_indicators(_rolling_ohlcv_series(m1,   15), "m15_",  15)),
    (FEAT_30M, "30m", 30,   lambda m1, m: _compute_std_indicators(_rolling_ohlcv_series(m1,   30), "m30_",  30)),
    (FEAT_1H,  "1H",  60,   lambda m1, m: _compute_std_indicators(_rolling_ohlcv_series(m1,   60), "h1_",   60)),
    (FEAT_4H,  "4H",  240,  lambda m1, m: _compute_std_indicators(_rolling_ohlcv_series(m1,  240), "h4_",  240)),
    (FEAT_1D,  "1D",  1440, lambda m1, m: _compute_std_indicators(_rolling_ohlcv_series(m1, 1440), "d1_", 1440)),
]


def _build_individual():
    m1 = _load(PATH_1M)
    if m1 is None or m1.empty:
        print("[Features] btc_1m.csv not found — cannot build features", flush=True)
        return

    print(f"[Features] Loaded btc_1m.csv: {len(m1):,} rows", flush=True)

    for feat_path, label, mult, fn in _TF_TASKS:
        print(f"[Features] Computing {label} (tf_mult={mult}, "
              f"RSI={14*mult}, EMA9={9*mult}, BB={20*mult})...", flush=True)
        feat = fn(m1, mult)
        feat.to_csv(feat_path, index=False)
        n_feat = len([c for c in feat.columns if c not in ("time", "close")])
        print(f"[Features] {label}: {len(feat):,} rows  {n_feat} features  saved", flush=True)


# ── Build merged CSV + yearly split ──────────────────────────────────────────

_INTERMEDIATE_OHLCV = [
    "btc_15m.csv", "btc_30m.csv", "btc_1h.csv",
    "btc_4h.csv",  "btc_1d.csv",  "btc_1w.csv", "btc_1mo.csv",
]


def _build_merged():
    feat_files = [FEAT_1M, FEAT_15M, FEAT_30M, FEAT_1H, FEAT_4H, FEAT_1D]
    dfs = [_load(p) for p in feat_files]
    if any(df is None for df in dfs):
        missing = [p for p, d in zip(feat_files, dfs) if d is None]
        raise RuntimeError(f"Missing feature CSVs: {missing}")

    print(f"[Merged] Merging {len(dfs[0]):,} 1m rows across 6 timeframes...", flush=True)
    merged = dfs[0].sort_values("time")
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="time", how="inner")

    before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"[Merged] Dropped {before - len(merged):,} NaN rows -> {len(merged):,} final rows",
          flush=True)

    # Yearly split
    os.makedirs(YEARLY_DIR, exist_ok=True)
    for year, group in merged.groupby(merged["time"].dt.year):
        path = os.path.join(YEARLY_DIR, f"{year}_merged.csv")
        group.to_csv(path, index=False)
        print(f"[Merged] Saved yearly/{year}_merged.csv ({len(group):,} rows)", flush=True)

    # Delete leftover intermediate OHLCV CSVs
    for fname in _INTERMEDIATE_OHLCV:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
            print(f"[Merged] Deleted intermediate: {fname}", flush=True)

    return merged


# ── Incremental update: current year only ────────────────────────────────────

_MAX_WINDOW = 1440 * 50  # 50 days — enough warm-up for d1_ with tf_mult=1440


def _update_current_year_features() -> int:
    """
    Fast incremental update after new 1m candles are appended.
    Loads the last _MAX_WINDOW rows (enough warm-up for all scaled windows),
    recomputes all 6 rolling feature sets, and appends only new rows to the
    current year's merged CSV.
    """
    if not os.path.exists(PATH_1M):
        return 0

    m1_full = _load(PATH_1M)
    if m1_full is None or m1_full.empty:
        return 0

    tail = m1_full.tail(_MAX_WINDOW + 500).reset_index(drop=True)

    feature_sets = [fn(tail, mult) for _, _, mult, fn in _TF_TASKS]

    merged = feature_sets[0].sort_values("time")
    for df in feature_sets[1:]:
        merged = pd.merge(merged, df, on="time", how="inner")
    merged = merged.dropna(subset=FEATURE_COLS)
    if merged.empty:
        return 0

    year = pd.Timestamp.now(tz="UTC").year
    year_path = os.path.join(YEARLY_DIR, f"{year}_merged.csv")
    os.makedirs(YEARLY_DIR, exist_ok=True)

    if os.path.exists(year_path):
        existing_ts = pd.read_csv(year_path, usecols=["time"], parse_dates=["time"])
        if existing_ts["time"].dt.tz is None:
            existing_ts["time"] = pd.to_datetime(existing_ts["time"], utc=True)
        last_ts  = existing_ts["time"].max()
        new_rows = merged[merged["time"] > last_ts]
    else:
        new_rows = merged

    if new_rows.empty:
        return 0

    write_header = not os.path.exists(year_path)
    new_rows.to_csv(year_path, mode="a", header=write_header, index=False)
    return len(new_rows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def _get_last_csv_time(path: str) -> Optional[pd.Timestamp]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 8192))
        data = f.read()
    for line in reversed(data.decode("utf-8", errors="replace").split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            return pd.to_datetime(line.split(",")[0].strip(), utc=True)
        except Exception:
            pass
    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def update_features() -> dict:
    """Full rebuild or incremental update. Returns stats dict."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(YEARLY_DIR, exist_ok=True)

    yearly_files = (
        [f for f in os.listdir(YEARLY_DIR) if f.endswith("_merged.csv")]
        if os.path.exists(YEARLY_DIR) else []
    )

    if not yearly_files:
        _build_individual()
        _build_merged()
        total = sum(_count_rows(os.path.join(YEARLY_DIR, f))
                    for f in os.listdir(YEARLY_DIR) if f.endswith("_merged.csv"))
        return {"merged_added": total, "partial_recalculated": False}

    n_added = _update_current_year_features()
    return {"merged_added": n_added, "partial_recalculated": False}


# ── Backward compat shim ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ind = _compute_std_indicators(df, prefix="", tf_mult=1)
    for col in ind.columns:
        if col not in ("time", "close"):
            df[col] = ind[col].values if len(ind) == len(df) else np.nan
    return df


# ── Verification ──────────────────────────────────────────────────────────────

def _verify():
    print("\n" + "=" * 50)
    print("FEATURE VERIFICATION REPORT")
    print("=" * 50)

    passes, fails, fail_list = 0, 0, []

    def chk(label, ok, detail=""):
        nonlocal passes, fails
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""))
        if ok:
            passes += 1
        else:
            fails += 1
            fail_list.append(f"{label}: {detail}")

    # ── File summary ─────────────────────────────────────────────────────────
    file_specs = [
        (FEAT_1M,  "btc_1m_features.csv",  12, FEATURE_1M),
        (FEAT_15M, "btc_15m_features.csv", 12, FEATURE_15M),
        (FEAT_30M, "btc_30m_features.csv", 12, FEATURE_30M),
        (FEAT_1H,  "btc_1h_features.csv",  12, FEATURE_1H),
        (FEAT_4H,  "btc_4h_features.csv",  12, FEATURE_4H),
        (FEAT_1D,  "btc_1d_features.csv",  12, FEATURE_1D),
    ]
    print(f"\n{'File':<30} {'Rows':>10}  {'Features':>9}  Status")
    print("-" * 60)
    for path, fname, n_expected, cols in file_specs:
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=0)
            row_count = sum(1 for _ in open(path)) - 1
            feat_count = sum(1 for c in df.columns if c not in ("time", "close"))
            ok = feat_count == n_expected
            print(f"{fname:<30} {row_count:>10,}  {feat_count:>9}  {'OK' if ok else 'FAIL'}")
        else:
            print(f"{fname:<30} {'MISSING':>10}")

    # ── Yearly files ──────────────────────────────────────────────────────────
    print(f"\nYEARLY FILES (data/yearly/):")
    print("-" * 60)
    if os.path.exists(YEARLY_DIR):
        found = sorted(f for f in os.listdir(YEARLY_DIR) if f.endswith("_merged.csv"))
        if found:
            for fname in found:
                path = os.path.join(YEARLY_DIR, fname)
                rc = sum(1 for _ in open(path)) - 1
                print(f"  {fname:<30} {rc:>10,} rows")
        else:
            print("  (empty)")
    else:
        print("  (no yearly directory)")

    # ── Load yearly files for checks ─────────────────────────────────────────
    print(f"\nINDIVIDUAL CHECKS (data/yearly/ combined):")
    print("-" * 60)

    if not os.path.exists(YEARLY_DIR):
        print("[FAIL] data/yearly/ not found")
        return 0, 1
    yearly_csvs = sorted(f for f in os.listdir(YEARLY_DIR) if f.endswith("_merged.csv"))
    if not yearly_csvs:
        print("[FAIL] No yearly files found")
        return 0, 1

    dfs = []
    for fname in yearly_csvs:
        df_y = pd.read_csv(os.path.join(YEARLY_DIR, fname), parse_dates=["time"])
        if df_y["time"].dt.tz is None:
            df_y["time"] = pd.to_datetime(df_y["time"], utc=True)
        dfs.append(df_y)
    merged = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)

    chk("C1  Row count > 2,000,000",          len(merged) > 2_000_000,   f"{len(merged):,}")
    tz_ok = merged["time"].dt.tz is not None and "UTC" in str(merged["time"].dt.tz).upper()
    chk("C2  time column UTC datetime",        tz_ok,                     str(merged["time"].dtype))
    n_feat = sum(1 for c in merged.columns if c not in ("time", "close"))
    chk("C3  Feature count = 72",              n_feat == 72,              f"{n_feat} total features")
    chk("C4  All 1m  features present (12)",   all(c in merged.columns for c in FEATURE_1M),
        str([c for c in FEATURE_1M  if c not in merged.columns] or "OK"))
    chk("C5  All 15m features present (12)",   all(c in merged.columns for c in FEATURE_15M),
        str([c for c in FEATURE_15M if c not in merged.columns] or "OK"))
    chk("C6  All 30m features present (12)",   all(c in merged.columns for c in FEATURE_30M),
        str([c for c in FEATURE_30M if c not in merged.columns] or "OK"))
    chk("C7  All 1H  features present (12)",   all(c in merged.columns for c in FEATURE_1H),
        str([c for c in FEATURE_1H  if c not in merged.columns] or "OK"))
    chk("C8  All 4H  features present (12)",   all(c in merged.columns for c in FEATURE_4H),
        str([c for c in FEATURE_4H  if c not in merged.columns] or "OK"))
    chk("C9  All 1D  features present (12)",   all(c in merged.columns for c in FEATURE_1D),
        str([c for c in FEATURE_1D  if c not in merged.columns] or "OK"))

    nan_total = merged[FEATURE_COLS].isna().sum().sum()
    chk("C10 No NaN in any feature",           nan_total == 0,            f"{nan_total}")
    inf_total = np.isinf(merged[FEATURE_COLS].values.astype(float)).sum()
    chk("C11 No Inf in any feature",           inf_total == 0,            f"{inf_total}")
    sort_ok = (merged["time"].diff().dt.total_seconds().dropna() > 0).all()
    chk("C12 Sorted ascending by time",        sort_ok)
    rsi_ok = ((merged["rsi"] >= 0) & (merged["rsi"] <= 1)).all()
    chk("C13 RSI in [0, 1]",                   rsi_ok,
        f"min={merged['rsi'].min():.4f} max={merged['rsi'].max():.4f}")
    atr_ok = (merged["atr_norm"] > 0).all()
    chk("C14 ATR values positive",             atr_ok,
        f"min={merged['atr_norm'].min():.6f}")

    # C15: verify timeframe scaling is genuine — rsi != h4_rsi
    if "h4_rsi" in merged.columns:
        identical_pct = (merged["rsi"].round(6) == merged["h4_rsi"].round(6)).mean() * 100
        chk("C15 1m_rsi != h4_rsi (scaled windows working)",
            identical_pct < 1, f"identical in {identical_pct:.1f}% of rows")

    # C16: h1_rsi should vary within each hour (rolling, not stale)
    sample = merged.tail(5000)
    h_groups = sample.groupby(sample["time"].dt.floor("h"))["h1_rsi"].nunique()
    chk("C16 Rolling: h1_rsi varies within each hour",
        (h_groups > 1).mean() > 0.5, f"avg unique h1_rsi per hour: {h_groups.mean():.1f}")

    chk("C17 close column present",            "close" in merged.columns)

    last_date = merged["time"].max().date()
    today     = pd.Timestamp.now(tz="UTC").date()
    chk("C18 Last row is today or yesterday",  (today - last_date).days <= 1,
        f"{last_date} (today={today})")

    # ── Market context ────────────────────────────────────────────────────────
    print("\nCURRENT MARKET CONTEXT (latest row):")
    row = merged.iloc[-1]
    for label, rsi_c, macd_c, atr_c in [
        ("1m",  "rsi",      "macd_diff_pct",    "atr_norm"),
        ("15m", "m15_rsi",  "m15_macd_diff_pct","m15_atr_norm"),
        ("30m", "m30_rsi",  "m30_macd_diff_pct","m30_atr_norm"),
        ("1H",  "h1_rsi",   "h1_macd_diff_pct", "h1_atr_norm"),
        ("4H",  "h4_rsi",   "h4_macd_diff_pct", "h4_atr_norm"),
        ("1D",  "d1_rsi",   "d1_macd_diff_pct", "d1_atr_norm"),
    ]:
        try:
            rsi_v  = float(row[rsi_c]) * 100
            macd_v = float(row[macd_c])
            atr_v  = float(row[atr_c]) * 100
            trend  = "neutral" if abs(macd_v) < 0.0001 else ("bullish" if macd_v > 0 else "bearish")
            print(f"[{label:<3}]  RSI: {rsi_v:5.1f}  MACD: {macd_v:+.4f}  ATR: {atr_v:.3f}%  Trend: {trend}")
        except (KeyError, TypeError):
            print(f"[{label:<3}]  (no data)")

    print()
    print("=" * 50)
    if fails == 0:
        print(f"ALL {passes} CHECKS PASSED -- READY FOR TRAINING")
    else:
        print(f"{fails} CHECKS FAILED -- fix issues before training")
        for f in fail_list:
            print(f"  - {f}")
    print("=" * 50)
    return passes, fails


# ── Manifest update ───────────────────────────────────────────────────────────

def _update_manifest():
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                content = f.read().strip()
            if content:
                manifest = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            pass

    file_specs = [
        (FEAT_1M,  "btc_1m_features",  12),
        (FEAT_15M, "btc_15m_features", 12),
        (FEAT_30M, "btc_30m_features", 12),
        (FEAT_1H,  "btc_1h_features",  12),
        (FEAT_4H,  "btc_4h_features",  12),
        (FEAT_1D,  "btc_1d_features",  12),
    ]
    for path, key, n_feat in file_specs:
        if os.path.exists(path):
            rows = sum(1 for _ in open(path)) - 1
            manifest[key] = {"rows": rows, "features": n_feat, "verified": False}

    if os.path.exists(YEARLY_DIR):
        yearly_info = {}
        for fname in sorted(os.listdir(YEARLY_DIR)):
            if fname.endswith("_merged.csv"):
                path = os.path.join(YEARLY_DIR, fname)
                rows = sum(1 for _ in open(path)) - 1
                yearly_info[fname] = {"rows": rows}
        manifest["yearly_files"] = yearly_info

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[Manifest] Updated with all feature files")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    update_features()
    _verify()
    _update_manifest()
