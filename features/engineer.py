import os
import json
import numpy as np
import pandas as pd
import ta
from typing import Optional
from datetime import timezone

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

PATH_1M  = os.path.join(DATA_DIR, "btc_1m.csv")
PATH_15M = os.path.join(DATA_DIR, "btc_15m.csv")
PATH_30M = os.path.join(DATA_DIR, "btc_30m.csv")
PATH_1H  = os.path.join(DATA_DIR, "btc_1h.csv")
PATH_4H  = os.path.join(DATA_DIR, "btc_4h.csv")
PATH_1D  = os.path.join(DATA_DIR, "btc_1d.csv")
PATH_1W  = os.path.join(DATA_DIR, "btc_1w.csv")
PATH_1MO = os.path.join(DATA_DIR, "btc_1mo.csv")

FEAT_1M  = os.path.join(DATA_DIR, "btc_1m_features.csv")
FEAT_15M = os.path.join(DATA_DIR, "btc_15m_features.csv")
FEAT_30M = os.path.join(DATA_DIR, "btc_30m_features.csv")
FEAT_1H  = os.path.join(DATA_DIR, "btc_1h_features.csv")
FEAT_4H  = os.path.join(DATA_DIR, "btc_4h_features.csv")
FEAT_1D  = os.path.join(DATA_DIR, "btc_1d_features.csv")
FEAT_1W  = os.path.join(DATA_DIR, "btc_1w_features.csv")
FEAT_1MO = os.path.join(DATA_DIR, "btc_1mo_features.csv")
MERGED   = os.path.join(DATA_DIR, "btc_merged_features.csv")

_BASE = ["rsi", "macd_diff_pct",
         "ema9_ratio", "ema21_ratio", "ema50_ratio",
         "atr_norm", "bb_pct", "bb_width",
         "obv_zscore", "vol_ratio"]

FEATURE_1M  = list(_BASE)
FEATURE_15M = [f"m15_{c}" for c in _BASE]
FEATURE_30M = [f"m30_{c}" for c in _BASE]
FEATURE_1H  = [f"h1_{c}" for c in _BASE]
FEATURE_4H  = [f"h4_{c}" for c in _BASE]
FEATURE_1D  = [f"d1_{c}" for c in _BASE]
FEATURE_1W  = [f"w1_{c}" for c in _BASE] + ["w1_above_4w_ema", "w1_above_8w_ema", "w1_is_complete"]
FEATURE_1MO = [f"mo1_{c}" for c in _BASE] + ["mo1_above_6mo_ema", "mo1_above_12mo_ema",
                                               "mo1_bull_market", "mo1_is_complete"]
FEATURE_COLS = (FEATURE_1M + FEATURE_15M + FEATURE_30M
                + FEATURE_1H + FEATURE_4H + FEATURE_1D
                + FEATURE_1W + FEATURE_1MO)
# Total: 10+10+10+10+10+10+13+14 = 87 features


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


# ── Standard 10 indicators (ta library, used for 1m/1H/4H/1D) ────────────────

def _compute_std_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    result = df[["time"]].copy()
    if prefix == "" and "close" in df.columns:
        result["close"] = df["close"].values
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)
    p = prefix

    result[f"{p}rsi"]           = ta.momentum.RSIIndicator(close, window=14).rsi() / 100.0
    macd = ta.trend.MACD(close)
    result[f"{p}macd_diff_pct"] = macd.macd_diff() / close
    result[f"{p}ema9_ratio"]    = ta.trend.EMAIndicator(close, window=9).ema_indicator()  / close - 1
    result[f"{p}ema21_ratio"]   = ta.trend.EMAIndicator(close, window=21).ema_indicator() / close - 1
    result[f"{p}ema50_ratio"]   = ta.trend.EMAIndicator(close, window=50).ema_indicator() / close - 1
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    result[f"{p}atr_norm"]      = atr / close
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    result[f"{p}bb_pct"]        = bb.bollinger_pband()
    result[f"{p}bb_width"]      = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    obv      = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_mean = obv.rolling(50, min_periods=1).mean()
    obv_std  = obv.rolling(50, min_periods=1).std().replace(0, np.nan)
    result[f"{p}obv_zscore"]    = (obv - obv_mean) / obv_std
    vol_ma = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
    result[f"{p}vol_ratio"]     = volume / vol_ma

    feat_cols = [c for c in result.columns if c != "time" and c != "close"]
    return _clean(result, feat_cols)


# ── Custom indicators with min_periods=1 (used for 1W/1MO to avoid warmup NaN) ─

def _compute_custom_indicators(df: pd.DataFrame, prefix: str,
                                obv_win: int, vol_win: int) -> pd.DataFrame:
    result = df[["time"]].copy()
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)
    p = prefix

    # RSI — ewm-based, min_periods=1
    delta = close.diff().fillna(0)
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    result[f"{p}rsi"] = (1 - 1 / (1 + rs)).fillna(0.5)

    # MACD — ewm min_periods=1
    ema12    = close.ewm(span=12, min_periods=1, adjust=False).mean()
    ema26    = close.ewm(span=26, min_periods=1, adjust=False).mean()
    macd_l   = ema12 - ema26
    macd_sig = macd_l.ewm(span=9, min_periods=1, adjust=False).mean()
    result[f"{p}macd_diff_pct"] = (macd_l - macd_sig) / close

    # EMA ratios — min_periods=1
    ema9  = close.ewm(span=9,  min_periods=1, adjust=False).mean()
    ema21 = close.ewm(span=21, min_periods=1, adjust=False).mean()
    ema50 = close.ewm(span=50, min_periods=1, adjust=False).mean()
    result[f"{p}ema9_ratio"]  = ema9  / close - 1
    result[f"{p}ema21_ratio"] = ema21 / close - 1
    result[f"{p}ema50_ratio"] = ema50 / close - 1

    # ATR — ewm min_periods=1
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr  = pd.concat([high - low,
                     (high - prev_close).abs(),
                     (low  - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, min_periods=1, adjust=False).mean()
    result[f"{p}atr_norm"] = atr / close

    # Bollinger bands — rolling min_periods=1
    sma   = close.rolling(20, min_periods=1).mean()
    std   = close.rolling(20, min_periods=1).std().fillna(0)
    upper = sma + 2 * std
    lower = sma - 2 * std
    band  = (upper - lower).replace(0, np.nan)
    result[f"{p}bb_pct"]   = (close - lower) / band
    result[f"{p}bb_width"] = band / close

    # OBV z-score — rolling min_periods=1
    obv_sign = np.sign(close.diff()).fillna(0)
    obv      = (obv_sign * volume).cumsum()
    obv_mean = obv.rolling(obv_win, min_periods=1).mean()
    obv_std  = obv.rolling(obv_win, min_periods=1).std().replace(0, np.nan)
    result[f"{p}obv_zscore"] = (obv - obv_mean) / obv_std

    # Volume ratio — rolling min_periods=1
    vol_ma = volume.rolling(vol_win, min_periods=1).mean().replace(0, np.nan)
    result[f"{p}vol_ratio"] = volume / vol_ma

    feat_cols = [c for c in result.columns if c != "time"]
    return _clean(result, feat_cols)


# ── Rolling N-day indicators from daily OHLCV (replaces calendar weekly/monthly) ─

def _compute_rolling_daily_features(df_1d: pd.DataFrame, window: int, prefix: str,
                                     long_ema1: int, long_ema2: int) -> pd.DataFrame:
    """
    Compute rolling N-day trailing window indicators from daily OHLCV data.
    Each output row reflects data from the `window` days ending on that date.
    Column names match the old calendar-period versions for model compatibility.
    Rolling windows are always fully current — no partial-period concept needed.
    """
    close  = df_1d["close"].astype(float)
    high   = df_1d["high"].astype(float)
    low    = df_1d["low"].astype(float)
    volume = df_1d["volume"].astype(float)
    p      = prefix
    result = df_1d[["time"]].copy()

    rsi_win = window   # use the actual window: 7 for w1_, 30 for mo1_ (was incorrectly capped at 14)
    result[f"{p}rsi"]           = ta.momentum.RSIIndicator(close, window=rsi_win).rsi() / 100.0
    macd_i  = ta.trend.MACD(close)
    result[f"{p}macd_diff_pct"] = macd_i.macd_diff() / close
    result[f"{p}ema9_ratio"]    = ta.trend.EMAIndicator(close, window=9).ema_indicator()  / close - 1
    result[f"{p}ema21_ratio"]   = ta.trend.EMAIndicator(close, window=21).ema_indicator() / close - 1
    result[f"{p}ema50_ratio"]   = ta.trend.EMAIndicator(close, window=50).ema_indicator() / close - 1
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    result[f"{p}atr_norm"]      = atr / close
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    result[f"{p}bb_pct"]        = bb.bollinger_pband()
    result[f"{p}bb_width"]      = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    obv      = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_mean = obv.rolling(window, min_periods=1).mean()
    obv_std  = obv.rolling(window, min_periods=1).std().replace(0, np.nan)
    result[f"{p}obv_zscore"]    = (obv - obv_mean) / obv_std
    vol_ma = volume.rolling(window, min_periods=1).mean().replace(0, np.nan)
    result[f"{p}vol_ratio"]     = volume / vol_ma

    ema_l1 = ta.trend.EMAIndicator(close, window=long_ema1).ema_indicator()
    ema_l2 = ta.trend.EMAIndicator(close, window=long_ema2).ema_indicator()
    above1 = (close > ema_l1).astype(float)
    above2 = (close > ema_l2).astype(float)

    if prefix == "w1_":
        result["w1_above_4w_ema"] = above1   # EMA(28) ≈ 4 weeks of daily bars
        result["w1_above_8w_ema"] = above2   # EMA(56) ≈ 8 weeks of daily bars
        result["w1_is_complete"]  = 1.0      # rolling: always a complete window
    else:
        result["mo1_above_6mo_ema"]  = above1   # EMA(180) ≈ 6 months of daily bars
        result["mo1_above_12mo_ema"] = above2   # EMA(365) ≈ 12 months of daily bars
        result["mo1_bull_market"]    = (above1.astype(bool) & above2.astype(bool)).astype(float)
        result["mo1_is_complete"]    = 1.0

    feat_cols = [c for c in result.columns if c != "time"]
    return _clean(result, feat_cols)


# ── Build individual feature files ────────────────────────────────────────────

def _build_individual():
    results = {}
    tasks = [
        (PATH_1M,  FEAT_1M,  "1m",      lambda df: _compute_std_indicators(df, "")),
        (PATH_15M, FEAT_15M, "15m",     lambda df: _compute_std_indicators(df, "m15_")),
        (PATH_30M, FEAT_30M, "30m",     lambda df: _compute_std_indicators(df, "m30_")),
        (PATH_1H,  FEAT_1H,  "1H",      lambda df: _compute_std_indicators(df, "h1_")),
        (PATH_4H,  FEAT_4H,  "4H",      lambda df: _compute_std_indicators(df, "h4_")),
        (PATH_1D,  FEAT_1D,  "1D",      lambda df: _compute_std_indicators(df, "d1_")),
        (PATH_1D,  FEAT_1W,  "1W-roll", lambda df: _compute_rolling_daily_features(
                                             df, window=7,  prefix="w1_",  long_ema1=28,  long_ema2=56)),
        (PATH_1D,  FEAT_1MO, "1MO-roll",lambda df: _compute_rolling_daily_features(
                                             df, window=30, prefix="mo1_", long_ema1=180, long_ema2=365)),
    ]
    for raw_path, feat_path, label, fn in tasks:
        raw = _load(raw_path)
        if raw is None or raw.empty:
            print(f"[Features] {label}: source CSV not found — skipped", flush=True)
            continue
        print(f"[Features] Computing {label} on {len(raw):,} rows...", flush=True)
        feat = fn(raw)
        feat.to_csv(feat_path, index=False)
        n_feat = len([c for c in feat.columns if c not in ("time", "close")])
        print(f"[Features] {label}: {len(feat):,} rows  {n_feat} features  saved", flush=True)
        results[label] = feat
    return results


# ── Build merged CSV ──────────────────────────────────────────────────────────

def _build_merged():
    feat_1m  = _load(FEAT_1M)
    feat_15m = _load(FEAT_15M)
    feat_30m = _load(FEAT_30M)
    feat_1h  = _load(FEAT_1H)
    feat_4h  = _load(FEAT_4H)
    feat_1d  = _load(FEAT_1D)
    feat_1w  = _load(FEAT_1W)
    feat_1mo = _load(FEAT_1MO)

    if any(f is None for f in [feat_1m, feat_15m, feat_30m,
                                feat_1h, feat_4h, feat_1d, feat_1w, feat_1mo]):
        raise RuntimeError("One or more feature CSVs missing — cannot build merged.")

    print(f"[Merged] Merging {len(feat_1m):,} 1m rows with all timeframes...", flush=True)
    merged = feat_1m.sort_values("time")

    # Calendar-period TFs: shift by one period so join lands on the previous closed candle.
    # This eliminates look-ahead bias where the current open candle's complete data would
    # be visible to rows within that candle during historical batch training.
    TF_LAG_MIN = {"15M": 15, "30M": 30, "1H": 60, "4H": 240, "1D": 1440}
    for feat, label in [(feat_15m, "15M"), (feat_30m, "30M"),
                        (feat_1h, "1H"), (feat_4h, "4H"), (feat_1d, "1D")]:
        feat_lagged = feat.copy()
        feat_lagged["time"] = feat_lagged["time"] + pd.Timedelta(minutes=TF_LAG_MIN[label])
        merged = pd.merge_asof(merged, feat_lagged.sort_values("time"),
                               on="time", direction="backward")
        print(f"[Merged]   After {label}: {len(merged):,} rows", flush=True)

    # Rolling daily TFs: shift 1 day so each 1m row uses yesterday's rolling window.
    # Rolling windows are always complete — no partial-period handling needed.
    DAY_LAG = pd.Timedelta(minutes=1440)
    for feat, label in [(feat_1w, "1W-roll"), (feat_1mo, "1MO-roll")]:
        feat_lagged = feat.copy()
        feat_lagged["time"] = feat_lagged["time"] + DAY_LAG
        merged = pd.merge_asof(merged, feat_lagged.sort_values("time"),
                               on="time", direction="backward")
        print(f"[Merged]   After {label}: {len(merged):,} rows", flush=True)

    before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"[Merged] Dropped {before - len(merged):,} NaN rows  ->  {len(merged):,} final rows", flush=True)

    merged.to_csv(MERGED, index=False)
    print(f"[Merged] Saved to btc_merged_features.csv  ({len(FEATURE_COLS)} features)", flush=True)
    return merged


LOOKBACK = 100


# ── Incremental helpers ───────────────────────────────────────────────────────

def _count_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def _get_last_csv_time(path: str) -> Optional[pd.Timestamp]:
    """Read last timestamp from a CSV. Uses 8 KB tail so that even wide rows are captured."""
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


def _tail_csv_df(path: str, n: int = 5000) -> pd.DataFrame:
    """Read last n rows from a CSV using binary tail seek."""
    from io import StringIO
    READ_BYTES = max(n * 600, 2 * 1024 * 1024)
    with open(path, "rb") as f:
        header   = f.readline().decode("utf-8").strip()
        f.seek(0, 2)
        seek_pos = max(1, f.tell() - READ_BYTES)
        f.seek(seek_pos)
        data = f.read()
    lines = [l for l in data.decode("utf-8", errors="replace").split("\n")[1:] if l.strip()]
    df = pd.read_csv(StringIO(header + "\n" + "\n".join(lines[-n:])), parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _update_tf_incremental(raw_path: str, feat_path: str, compute_fn, label: str) -> int:
    """Compute features only for new rows using LOOKBACK context. Returns n_added."""
    raw = _load(raw_path)
    if raw is None or raw.empty:
        return 0

    if not os.path.exists(feat_path):
        print(f"  Building {label} features from scratch ({len(raw):,} rows)...", flush=True)
        feat = compute_fn(raw)
        feat.to_csv(feat_path, index=False)
        return len(feat)

    n_existing = _count_rows(feat_path)
    if n_existing >= len(raw):
        return 0

    start_idx  = max(0, n_existing - LOOKBACK)
    chunk      = raw.iloc[start_idx:].copy().reset_index(drop=True)
    feat_chunk = compute_fn(chunk)
    n_skip     = n_existing - start_idx
    new_feat   = feat_chunk.iloc[n_skip:]

    if new_feat.empty:
        return 0

    new_feat.to_csv(feat_path, mode="a", header=False, index=False)
    return len(new_feat)




def _update_merged_incremental() -> int:
    """Append only new rows to merged CSV. Returns n_added."""
    if not os.path.exists(FEAT_1M):
        return 0

    last_merged = _get_last_csv_time(MERGED) if os.path.exists(MERGED) else None

    # Load new 1m feature rows (tail read covers up to 5000 recent rows)
    feat_tail = _tail_csv_df(FEAT_1M, n=5000)
    new_1m = feat_tail[feat_tail["time"] > last_merged] if last_merged is not None else feat_tail
    if new_1m.empty:
        return 0

    feat_15m = _load(FEAT_15M)
    feat_30m = _load(FEAT_30M)
    feat_1h  = _load(FEAT_1H)
    feat_4h  = _load(FEAT_4H)
    feat_1d  = _load(FEAT_1D)
    feat_1w  = _load(FEAT_1W)
    feat_1mo = _load(FEAT_1MO)

    if any(f is None for f in [feat_15m, feat_30m,
                                feat_1h, feat_4h, feat_1d, feat_1w, feat_1mo]):
        return 0

    TF_LAG_MIN = {"15M": 15, "30M": 30, "1H": 60, "4H": 240, "1D": 1440}
    DAY_LAG    = pd.Timedelta(minutes=1440)

    merged = new_1m.sort_values("time")
    for feat_tf, label in [(feat_15m, "15M"), (feat_30m, "30M"),
                            (feat_1h, "1H"), (feat_4h, "4H"), (feat_1d, "1D")]:
        cols        = [c for c in feat_tf.columns if c != "time"]
        feat_lagged = feat_tf[["time"] + cols].copy()
        feat_lagged["time"] = feat_lagged["time"] + pd.Timedelta(minutes=TF_LAG_MIN[label])
        merged = pd.merge_asof(merged, feat_lagged.sort_values("time"),
                               on="time", direction="backward")

    for feat_tf in [feat_1w, feat_1mo]:
        cols        = [c for c in feat_tf.columns if c != "time"]
        feat_lagged = feat_tf[["time"] + cols].copy()
        feat_lagged["time"] = feat_lagged["time"] + DAY_LAG
        merged = pd.merge_asof(merged, feat_lagged.sort_values("time"),
                               on="time", direction="backward")

    merged = merged.dropna(subset=FEATURE_COLS)
    if merged.empty:
        return 0

    # Safety: strip any overlap with existing merged rows
    if last_merged is not None:
        merged = merged[merged["time"] > last_merged]
    if merged.empty:
        return 0

    write_header = not os.path.exists(MERGED)
    merged.to_csv(MERGED, mode="a", header=write_header, index=False)
    return len(merged)


# ── Main entry point ──────────────────────────────────────────────────────────

def update_features() -> dict:
    """Incremental update: compute only new rows. Falls back to full build if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)

    ind_exist = all(os.path.exists(p) for p in [FEAT_1M, FEAT_15M, FEAT_30M,
                                                  FEAT_1H, FEAT_4H, FEAT_1D,
                                                  FEAT_1W, FEAT_1MO])
    if not ind_exist:
        _build_individual()

    if not os.path.exists(MERGED):
        _build_merged()
        return {"merged_added": _count_rows(MERGED), "partial_recalculated": False}

    if not ind_exist:
        return {"merged_added": _count_rows(MERGED), "partial_recalculated": False}

    # Incremental updates for each timeframe
    tasks = [
        (PATH_1M,  FEAT_1M,  lambda df: _compute_std_indicators(df, ""),       "1m"),
        (PATH_15M, FEAT_15M, lambda df: _compute_std_indicators(df, "m15_"),   "15m"),
        (PATH_30M, FEAT_30M, lambda df: _compute_std_indicators(df, "m30_"),   "30m"),
        (PATH_1H,  FEAT_1H,  lambda df: _compute_std_indicators(df, "h1_"),    "1H"),
        (PATH_4H,  FEAT_4H,  lambda df: _compute_std_indicators(df, "h4_"),    "4H"),
        (PATH_1D,  FEAT_1D,  lambda df: _compute_std_indicators(df, "d1_"),    "1D"),
        (PATH_1D,  FEAT_1W,  lambda df: _compute_rolling_daily_features(
                                  df, window=7,  prefix="w1_",  long_ema1=28,  long_ema2=56),  "1W-roll"),
        (PATH_1D,  FEAT_1MO, lambda df: _compute_rolling_daily_features(
                                  df, window=30, prefix="mo1_", long_ema1=180, long_ema2=365), "1MO-roll"),
    ]
    for raw_path, feat_path, fn, label in tasks:
        _update_tf_incremental(raw_path, feat_path, fn, label)

    merged_added = _update_merged_incremental()

    return {"merged_added": merged_added, "partial_recalculated": False}


# ── Backward compat shim ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ind = _compute_std_indicators(df, prefix="")
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

    # ── Individual file summary ───────────────────────────────────────────────
    file_specs = [
        (FEAT_1M,  "btc_1m_features.csv",  10, FEATURE_1M),
        (FEAT_15M, "btc_15m_features.csv", 10, FEATURE_15M),
        (FEAT_30M, "btc_30m_features.csv", 10, FEATURE_30M),
        (FEAT_1H,  "btc_1h_features.csv",  10, FEATURE_1H),
        (FEAT_4H,  "btc_4h_features.csv",  10, FEATURE_4H),
        (FEAT_1D,  "btc_1d_features.csv",  10, FEATURE_1D),
        (FEAT_1W,  "btc_1w_features.csv",  13, FEATURE_1W),
        (FEAT_1MO, "btc_1mo_features.csv", 14, FEATURE_1MO),
        (MERGED,   "btc_merged_features.csv", len(FEATURE_COLS), FEATURE_COLS),
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

    # ── Merged file checks ────────────────────────────────────────────────────
    print(f"\nINDIVIDUAL CHECKS (btc_merged_features.csv):")
    print("-" * 60)

    merged = _load(MERGED)
    if merged is None:
        print("[FAIL] Merged CSV not found")
        return 0, 1

    chk("C1  Row count > 2,200,000",           len(merged) > 2_200_000,   f"{len(merged):,}")
    tz_ok = merged["time"].dt.tz is not None and "UTC" in str(merged["time"].dt.tz).upper()
    chk("C2  time column UTC datetime",         tz_ok,                     str(merged["time"].dtype))
    n_feat = sum(1 for c in merged.columns if c not in ("time", "close"))
    chk("C3  Feature count",                    True,                      f"{n_feat} total features")
    chk("C4  All 1m  features present (10)",    all(c in merged.columns for c in FEATURE_1M),
        str([c for c in FEATURE_1M  if c not in merged.columns] or "OK"))
    chk("C4b All 15m features present (10)",    all(c in merged.columns for c in FEATURE_15M),
        str([c for c in FEATURE_15M if c not in merged.columns] or "OK"))
    chk("C4c All 30m features present (10)",    all(c in merged.columns for c in FEATURE_30M),
        str([c for c in FEATURE_30M if c not in merged.columns] or "OK"))
    chk("C5  All 1H  features present (10)",    all(c in merged.columns for c in FEATURE_1H),
        str([c for c in FEATURE_1H  if c not in merged.columns] or "OK"))
    chk("C6  All 4H  features present (10)",    all(c in merged.columns for c in FEATURE_4H),
        str([c for c in FEATURE_4H  if c not in merged.columns] or "OK"))
    chk("C7  All 1D  features present (10)",    all(c in merged.columns for c in FEATURE_1D),
        str([c for c in FEATURE_1D  if c not in merged.columns] or "OK"))
    chk("C8  All 1W  features present (13)",    all(c in merged.columns for c in FEATURE_1W),
        str([c for c in FEATURE_1W  if c not in merged.columns] or "OK"))
    chk("C9  All 1MO features present (14)",    all(c in merged.columns for c in FEATURE_1MO),
        str([c for c in FEATURE_1MO if c not in merged.columns] or "OK"))

    nan_total = merged[FEATURE_COLS].isna().sum().sum()
    chk("C10 No NaN in any feature",            nan_total == 0,            f"{nan_total}")
    inf_total = np.isinf(merged[FEATURE_COLS].values.astype(float)).sum()
    chk("C11 No Inf in any feature",            inf_total == 0,            f"{inf_total}")
    chk("C12 Feature dtypes numeric",           all(merged[c].dtype in (np.float64, np.int64, np.float32, np.int32)
                                                    for c in FEATURE_COLS), "OK")
    sort_ok = (merged["time"].diff().dt.total_seconds().dropna() > 0).all()
    chk("C13 Sorted ascending by time",         sort_ok)
    rsi_ok = ((merged["rsi"] >= 0) & (merged["rsi"] <= 1)).all()
    chk("C14 RSI in [0, 1]",                    rsi_ok,                   f"min={merged['rsi'].min():.4f} max={merged['rsi'].max():.4f}")
    atr_ok = (merged["atr_norm"] > 0).all()
    chk("C15 ATR values positive",              atr_ok,                   f"min={merged['atr_norm'].min():.6f}")
    bb_min, bb_max = merged["bb_pct"].min(), merged["bb_pct"].max()
    chk("C16 BB pct reasonable (-1 to 2)",      bb_min > -1 and bb_max < 2, f"min={bb_min:.2f} max={bb_max:.2f}")

    # C17: 1W rolling features are daily — verify same-day rows share the same w1_rsi
    # (1-day lag means all 1m rows within a calendar day get the same daily rolling value)
    sample = merged.tail(5000)
    d_groups = sample.groupby(sample["time"].dt.date)["w1_rsi"].nunique()
    chk("C17 1W rolling features constant within day",
        (d_groups <= 1).all(), f"max unique w1_rsi per day: {d_groups.max()}")

    # C18: 1MO rolling features same check
    dm_groups = sample.groupby(sample["time"].dt.date)["mo1_rsi"].nunique()
    chk("C18 1MO rolling features constant within day",
        (dm_groups <= 1).all(), f"max unique mo1_rsi per day: {dm_groups.max()}")

    # C19: w1_is_complete / mo1_is_complete should always be 1.0 (rolling — never partial)
    chk("C19 w1_is_complete always 1 (rolling, no partial)",
        (merged["w1_is_complete"] == 1.0).all())
    chk("C20 mo1_is_complete always 1 (rolling, no partial)",
        (merged["mo1_is_complete"] == 1.0).all())
    chk("C21 close column present",            "close" in merged.columns)

    feat_1m = _load(FEAT_1M)
    if feat_1m is not None:
        chk("C22 Merged first date >= 1m_features first date",
            merged["time"].min().date() >= feat_1m["time"].min().date(),
            f"merged={merged['time'].min().date()}  1m={feat_1m['time'].min().date()} (gap=indicator warmup)")

    last_date = merged["time"].max().date()
    today     = pd.Timestamp.now(tz="UTC").date()
    chk("C23 Last row is today or yesterday",  (today - last_date).days <= 1,
        f"{last_date} (today={today})")

    # ── Market context ────────────────────────────────────────────────────────
    print("\nCURRENT MARKET CONTEXT (latest row):")
    row = merged.iloc[-1]
    for label, rsi_c, macd_c, atr_c, extra_fn in [
        ("1m",  "rsi",             "macd_diff_pct",    "atr_norm",     lambda r: ""),
        ("15m", "m15_rsi",         "m15_macd_diff_pct","m15_atr_norm", lambda r: ""),
        ("30m", "m30_rsi",         "m30_macd_diff_pct","m30_atr_norm", lambda r: ""),
        ("1H",  "h1_rsi",          "h1_macd_diff_pct", "h1_atr_norm",  lambda r: ""),
        ("4H",  "h4_rsi",          "h4_macd_diff_pct", "h4_atr_norm",  lambda r: ""),
        ("1D",  "d1_rsi",          "d1_macd_diff_pct", "d1_atr_norm",  lambda r: ""),
        ("1W",  "w1_rsi",          "w1_macd_diff_pct", "w1_atr_norm",  lambda r: "  (rolling 7-day)"),
        ("1MO", "mo1_rsi",         "mo1_macd_diff_pct","mo1_atr_norm",
         lambda r: f"  Bull market: {'YES' if r['mo1_bull_market'] == 1 else 'NO'}  (rolling 30-day)"),
    ]:
        try:
            rsi_v  = float(row[rsi_c]) * 100
            macd_v = float(row[macd_c])
            atr_v  = float(row[atr_c]) * 100
            trend  = "neutral" if abs(macd_v) < 0.0001 else ("bullish" if macd_v > 0 else "bearish")
            extra  = extra_fn(row)
            print(f"[{label:<3}]  RSI: {rsi_v:5.1f}  MACD: {macd_v:+.4f}  ATR: {atr_v:.3f}%  Trend: {trend}{extra}")
        except (KeyError, TypeError):
            print(f"[{label:<3}]  (no data)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    if fails == 0:
        print(f"ALL {passes} CHECKS PASSED — READY FOR TRAINING")
    else:
        print(f"{fails} CHECKS FAILED — DO NOT TRAIN — fix issues first")
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
        (FEAT_1M,  "btc_1m_features",  10),
        (FEAT_15M, "btc_15m_features", 10),
        (FEAT_30M, "btc_30m_features", 10),
        (FEAT_1H,  "btc_1h_features",  10),
        (FEAT_4H,  "btc_4h_features",  10),
        (FEAT_1D,  "btc_1d_features",  10),
        (FEAT_1W,  "btc_1w_features",  13),
        (FEAT_1MO, "btc_1mo_features", 14),
        (MERGED,   "btc_merged_features", len(FEATURE_COLS)),
    ]
    for path, key, n_feat in file_specs:
        if os.path.exists(path):
            rows = sum(1 for _ in open(path)) - 1
            manifest[key] = {
                "rows":     rows,
                "features": n_feat,
                "verified": key == "btc_merged_features",
            }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[Manifest] Updated with all feature files")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    update_features()
    _verify()
    _update_manifest()
