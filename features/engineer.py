import os
import json
import numpy as np
import pandas as pd
import ta
from typing import Optional

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
YEARLY_DIR = os.path.join(DATA_DIR, "yearly_merged")

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

# ── Model-specific feature column sets ───────────────────────────────────────
# BiLSTM (micro, SEQ=96 @ 15min): 15m/1h indicators — 24 features
# TFT    (macro, SEQ=180 @ 4h):   4h/1d indicators  — 24 features
BILSTM_FEAT_COLS = FEATURE_15M + FEATURE_1H
TFT_FEAT_COLS    = FEATURE_4H  + FEATURE_1D


def get_feature_groups() -> dict:
    """
    Feature split for the TFT-ACB-XML architecture (1-day horizon).

    BiLSTM (micro, SEQ=96 @ 15min):  15m/1H  — 24 features
    TFT    (macro, SEQ=180 @ 4h):    4H/1D   — 24 features, all dynamic (no static covariates)
    """
    return {
        "bilstm":      BILSTM_FEAT_COLS,   # 24
        "tft_dynamic": TFT_FEAT_COLS,      # 24
        "tft_static":  [],                  # none
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


# ── Build model-specific merged CSVs ─────────────────────────────────────────

def _at_15min_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows at exact 15-minute clock boundaries (:00, :15, :30, :45)."""
    mask = (df["time"].dt.minute % 15 == 0) & (df["time"].dt.second == 0)
    return df[mask].reset_index(drop=True)


def _at_4h_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows at exact 4-hour clock boundaries (00:00, 04:00, ..., 20:00 UTC)."""
    mask = (df["time"].dt.hour % 4 == 0) & (df["time"].dt.minute == 0) & (df["time"].dt.second == 0)
    return df[mask].reset_index(drop=True)


def _build_bilstm_merged():
    """
    Build {YYYY}_bilstm_merged.csv: m15_* + h1_* features sampled at 15-minute boundaries.
    Rows are at :00, :15, :30, :45 of each hour. All features still computed from
    rolling 1m windows (no data leakage).
    """
    df_15m = _load(FEAT_15M)
    df_1h  = _load(FEAT_1H)
    if df_15m is None or df_1h is None:
        raise RuntimeError("Missing btc_15m_features.csv or btc_1h_features.csv")

    print(f"[BiLSTM Merged] Merging 15m ({len(df_15m):,}) and 1h ({len(df_1h):,}) features...",
          flush=True)
    merged = pd.merge(df_15m, df_1h, on="time", how="inner")

    # Keep close from 15m side (they share the same close, but 15m has it from 1m passthrough)
    # Both have 'time'; 15m has 'close' from _compute_std_indicators for 1m prefix="" path.
    # For FEAT_15M/1H, close is NOT added (prefix != ""). We need to get close from FEAT_1M.
    df_1m_feat = _load(FEAT_1M)
    if df_1m_feat is not None and "close" in df_1m_feat.columns:
        merged = pd.merge(merged, df_1m_feat[["time", "close"]], on="time", how="inner")

    merged = _at_15min_boundary(merged)
    before = len(merged)
    merged = merged.dropna(subset=BILSTM_FEAT_COLS).reset_index(drop=True)
    print(f"[BiLSTM Merged] After 15min filter + dropna: {before} -> {len(merged):,} rows",
          flush=True)

    os.makedirs(YEARLY_DIR, exist_ok=True)
    for year, group in merged.groupby(merged["time"].dt.year):
        path = os.path.join(YEARLY_DIR, f"{year}_bilstm_merged.csv")
        group.to_csv(path, index=False)
        print(f"[BiLSTM Merged] Saved yearly_merged/{year}_bilstm_merged.csv ({len(group):,} rows)",
              flush=True)
    return merged


def _build_tft_merged():
    """
    Build {YYYY}_tft_merged.csv: h4_* + d1_* features sampled at 4-hour boundaries.
    Rows are at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC. All features still
    computed from rolling 1m windows (no data leakage).
    """
    df_4h = _load(FEAT_4H)
    df_1d = _load(FEAT_1D)
    if df_4h is None or df_1d is None:
        raise RuntimeError("Missing btc_4h_features.csv or btc_1d_features.csv")

    print(f"[TFT Merged] Merging 4h ({len(df_4h):,}) and 1d ({len(df_1d):,}) features...",
          flush=True)
    merged = pd.merge(df_4h, df_1d, on="time", how="inner")

    df_1m_feat = _load(FEAT_1M)
    if df_1m_feat is not None and "close" in df_1m_feat.columns:
        merged = pd.merge(merged, df_1m_feat[["time", "close"]], on="time", how="inner")

    merged = _at_4h_boundary(merged)
    before = len(merged)
    merged = merged.dropna(subset=TFT_FEAT_COLS).reset_index(drop=True)
    print(f"[TFT Merged] After 4h filter + dropna: {before} -> {len(merged):,} rows", flush=True)

    os.makedirs(YEARLY_DIR, exist_ok=True)
    for year, group in merged.groupby(merged["time"].dt.year):
        path = os.path.join(YEARLY_DIR, f"{year}_tft_merged.csv")
        group.to_csv(path, index=False)
        print(f"[TFT Merged] Saved yearly_merged/{year}_tft_merged.csv ({len(group):,} rows)", flush=True)
    return merged


def _build_merged():
    """Build both bilstm_merged and tft_merged yearly CSVs."""
    _build_bilstm_merged()
    _build_tft_merged()

    # Delete leftover intermediate OHLCV CSVs
    _INTERMEDIATE_OHLCV = [
        "btc_15m.csv", "btc_30m.csv", "btc_1h.csv",
        "btc_4h.csv",  "btc_1d.csv",  "btc_1w.csv", "btc_1mo.csv",
    ]
    for fname in _INTERMEDIATE_OHLCV:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
            print(f"[Merged] Deleted intermediate: {fname}", flush=True)


# ── Incremental update: current year only ────────────────────────────────────

_MAX_WINDOW = 1440 * 50  # 50 days — enough warm-up for d1_ with tf_mult=1440


def _update_current_year_features() -> dict:
    """
    Fast incremental update after new 1m candles are appended.
    Loads the last _MAX_WINDOW rows (enough warm-up for all scaled windows),
    recomputes all 6 rolling feature sets, then appends only new boundary rows
    to the current year's bilstm_merged (15min boundaries) and tft_merged (4h boundaries).
    """
    if not os.path.exists(PATH_1M):
        return {"bilstm_added": 0, "tft_added": 0}

    m1_full = _load(PATH_1M)
    if m1_full is None or m1_full.empty:
        return {"bilstm_added": 0, "tft_added": 0}

    tail = m1_full.tail(_MAX_WINDOW + 500).reset_index(drop=True)

    feature_sets = [fn(tail, mult) for _, _, mult, fn in _TF_TASKS]

    # Map label to computed DataFrame
    feat_map = {label: feature_sets[i] for i, (_, label, _, _) in enumerate(_TF_TASKS)}

    year = pd.Timestamp.now(tz="UTC").year
    os.makedirs(YEARLY_DIR, exist_ok=True)

    # ── BiLSTM merged update (15min boundaries) ───────────────────────────────
    bilstm_added = 0
    feat_15m = feat_map["15m"]
    feat_1h  = feat_map["1H"]
    feat_1m_close = feat_map["1m"][["time", "close"]] if "close" in feat_map["1m"].columns else None

    bilstm_new = pd.merge(feat_15m, feat_1h, on="time", how="inner")
    if feat_1m_close is not None:
        bilstm_new = pd.merge(bilstm_new, feat_1m_close, on="time", how="inner")
    bilstm_new = _at_15min_boundary(bilstm_new)
    bilstm_new = bilstm_new.dropna(subset=BILSTM_FEAT_COLS)

    bilstm_path = os.path.join(YEARLY_DIR, f"{year}_bilstm_merged.csv")
    if os.path.exists(bilstm_path):
        existing_ts = pd.read_csv(bilstm_path, usecols=["time"], parse_dates=["time"])
        if existing_ts["time"].dt.tz is None:
            existing_ts["time"] = pd.to_datetime(existing_ts["time"], utc=True)
        last_ts = existing_ts["time"].max()
        bilstm_new = bilstm_new[bilstm_new["time"] > last_ts]

    if not bilstm_new.empty:
        write_header = not os.path.exists(bilstm_path)
        bilstm_new.to_csv(bilstm_path, mode="a", header=write_header, index=False)
        bilstm_added = len(bilstm_new)

    # ── TFT merged update (4h boundaries) ─────────────────────────────────────
    tft_added = 0
    feat_4h = feat_map["4H"]
    feat_1d = feat_map["1D"]

    tft_new = pd.merge(feat_4h, feat_1d, on="time", how="inner")
    if feat_1m_close is not None:
        tft_new = pd.merge(tft_new, feat_1m_close, on="time", how="inner")
    tft_new = _at_4h_boundary(tft_new)
    tft_new = tft_new.dropna(subset=TFT_FEAT_COLS)

    tft_path = os.path.join(YEARLY_DIR, f"{year}_tft_merged.csv")
    if os.path.exists(tft_path):
        existing_ts = pd.read_csv(tft_path, usecols=["time"], parse_dates=["time"])
        if existing_ts["time"].dt.tz is None:
            existing_ts["time"] = pd.to_datetime(existing_ts["time"], utc=True)
        last_ts = existing_ts["time"].max()
        tft_new = tft_new[tft_new["time"] > last_ts]

    if not tft_new.empty:
        write_header = not os.path.exists(tft_path)
        tft_new.to_csv(tft_path, mode="a", header=write_header, index=False)
        tft_added = len(tft_new)

    return {"bilstm_added": bilstm_added, "tft_added": tft_added}


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

    bilstm_files = (
        [f for f in os.listdir(YEARLY_DIR) if f.endswith("_bilstm_merged.csv")]
        if os.path.exists(YEARLY_DIR) else []
    )
    tft_files = (
        [f for f in os.listdir(YEARLY_DIR) if f.endswith("_tft_merged.csv")]
        if os.path.exists(YEARLY_DIR) else []
    )

    if not bilstm_files or not tft_files:
        _build_individual()
        _build_merged()
        bilstm_total = sum(_count_rows(os.path.join(YEARLY_DIR, f))
                           for f in os.listdir(YEARLY_DIR) if f.endswith("_bilstm_merged.csv"))
        tft_total    = sum(_count_rows(os.path.join(YEARLY_DIR, f))
                           for f in os.listdir(YEARLY_DIR) if f.endswith("_tft_merged.csv"))
        return {"bilstm_added": bilstm_total, "tft_added": tft_total,
                "merged_added": bilstm_total + tft_total}

    counts = _update_current_year_features()
    counts["merged_added"] = counts["bilstm_added"] + counts["tft_added"]
    return counts


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
    print("\n" + "=" * 60)
    print("FEATURE VERIFICATION REPORT")
    print("=" * 60)

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

    # ── Individual feature file summary ──────────────────────────────────────
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

    # ── Yearly merged files ───────────────────────────────────────────────────
    for suffix, label in [("_bilstm_merged.csv", "BILSTM"), ("_tft_merged.csv", "TFT")]:
        print(f"\nYEARLY {label} FILES (data/yearly_merged/):")
        print("-" * 60)
        if os.path.exists(YEARLY_DIR):
            found = sorted(f for f in os.listdir(YEARLY_DIR) if f.endswith(suffix))
            if found:
                for fname in found:
                    path = os.path.join(YEARLY_DIR, fname)
                    rc = sum(1 for _ in open(path)) - 1
                    print(f"  {fname:<40} {rc:>10,} rows")
            else:
                print("  (empty)")
        else:
            print("  (no yearly directory)")

    # ── Load and check bilstm_merged ─────────────────────────────────────────
    print(f"\nBILSTM MERGED CHECKS:")
    print("-" * 60)
    if os.path.exists(YEARLY_DIR):
        bilstm_files = sorted(f for f in os.listdir(YEARLY_DIR) if f.endswith("_bilstm_merged.csv"))
    else:
        bilstm_files = []

    if bilstm_files:
        bilstm_dfs = []
        for fname in bilstm_files:
            df_y = pd.read_csv(os.path.join(YEARLY_DIR, fname), parse_dates=["time"])
            if df_y["time"].dt.tz is None:
                df_y["time"] = pd.to_datetime(df_y["time"], utc=True)
            bilstm_dfs.append(df_y)
        bm = pd.concat(bilstm_dfs, ignore_index=True).sort_values("time").reset_index(drop=True)

        n_feat = sum(1 for c in bm.columns if c not in ("time", "close"))
        chk("B1  Feature count = 24", n_feat == 24, f"{n_feat}")
        chk("B2  All 15m features present (12)", all(c in bm.columns for c in FEATURE_15M),
            str([c for c in FEATURE_15M if c not in bm.columns] or "OK"))
        chk("B3  All 1H features present (12)", all(c in bm.columns for c in FEATURE_1H),
            str([c for c in FEATURE_1H  if c not in bm.columns] or "OK"))
        chk("B4  close column present", "close" in bm.columns)

        # Check 15-min spacing
        diffs = bm["time"].diff().dt.total_seconds().dropna()
        chk("B5  Rows spaced at 15min (900s)", (diffs == 900).mean() > 0.95,
            f"{(diffs == 900).mean()*100:.1f}% at 900s")

        nan_total = bm[BILSTM_FEAT_COLS].isna().sum().sum()
        chk("B6  No NaN in BiLSTM features", nan_total == 0, f"{nan_total}")

        last_date = bm["time"].max().date()
        today = pd.Timestamp.now(tz="UTC").date()
        chk("B7  Last row is today or recent", (today - last_date).days <= 2,
            f"{last_date} (today={today})")
    else:
        print("  [FAIL] No bilstm_merged files found")
        fails += 1

    # ── Load and check tft_merged ─────────────────────────────────────────────
    print(f"\nTFT MERGED CHECKS:")
    print("-" * 60)
    if os.path.exists(YEARLY_DIR):
        tft_files = sorted(f for f in os.listdir(YEARLY_DIR) if f.endswith("_tft_merged.csv"))
    else:
        tft_files = []

    if tft_files:
        tft_dfs = []
        for fname in tft_files:
            df_y = pd.read_csv(os.path.join(YEARLY_DIR, fname), parse_dates=["time"])
            if df_y["time"].dt.tz is None:
                df_y["time"] = pd.to_datetime(df_y["time"], utc=True)
            tft_dfs.append(df_y)
        tm = pd.concat(tft_dfs, ignore_index=True).sort_values("time").reset_index(drop=True)

        n_feat = sum(1 for c in tm.columns if c not in ("time", "close"))
        chk("T1  Feature count = 24", n_feat == 24, f"{n_feat}")
        chk("T2  All 4H features present (12)", all(c in tm.columns for c in FEATURE_4H),
            str([c for c in FEATURE_4H if c not in tm.columns] or "OK"))
        chk("T3  All 1D features present (12)", all(c in tm.columns for c in FEATURE_1D),
            str([c for c in FEATURE_1D if c not in tm.columns] or "OK"))
        chk("T4  close column present", "close" in tm.columns)

        # Check 4h spacing
        diffs = tm["time"].diff().dt.total_seconds().dropna()
        chk("T5  Rows spaced at 4h (14400s)", (diffs == 14400).mean() > 0.95,
            f"{(diffs == 14400).mean()*100:.1f}% at 14400s")

        nan_total = tm[TFT_FEAT_COLS].isna().sum().sum()
        chk("T6  No NaN in TFT features", nan_total == 0, f"{nan_total}")

        last_date = tm["time"].max().date()
        today = pd.Timestamp.now(tz="UTC").date()
        chk("T7  Last row is today or recent", (today - last_date).days <= 2,
            f"{last_date} (today={today})")
    else:
        print("  [FAIL] No tft_merged files found")
        fails += 1

    print()
    print("=" * 60)
    if fails == 0:
        print(f"ALL {passes} CHECKS PASSED -- READY FOR TRAINING")
    else:
        print(f"{fails} CHECKS FAILED -- fix issues before training")
        for f in fail_list:
            print(f"  - {f}")
    print("=" * 60)
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
            if fname.endswith("_bilstm_merged.csv") or fname.endswith("_tft_merged.csv"):
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
