"""
Triple-Barrier Labeling for BTCPredictor2 (Marcos López de Prado, AFML Ch. 3).

Labels each merged-CSV row based on which barrier is hit first within the
forward horizon, scanning btc_1m.csv OHLC for bar-level accuracy:

  +1  — take-profit barrier (+tp_pct) hit before SL or vertical barrier
   0  — stop-loss barrier  (-sl_pct) OR vertical barrier (time expiry) hit first
  NaN — last rows where forward 1m data is unavailable (dropped after labeling)

Fast path: numba JIT-compiled inner loop (~0.5B iter/s).
Fallback:  vectorized numpy (slower but no extra dependency).
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Optional numba acceleration ───────────────────────────────────────────────

_HAS_NUMBA = False
try:
    import numba

    @numba.njit(cache=True)
    def _scan_barriers_jit(close, start_idxs, end_idxs,
                            m1_high, m1_low, tp_pct, sl_pct):
        """JIT inner loop: for each merged row scan forward 1m bars for barrier hits."""
        n = len(close)
        labels = np.zeros(n, dtype=np.float64)
        for t in range(n):
            s = start_idxs[t]
            e = end_idxs[t]
            if s >= e:
                continue
            tp = close[t] * (1.0 + tp_pct)
            sl = close[t] * (1.0 - sl_pct)
            label = 0
            for i in range(s, e):
                if m1_high[i] >= tp:
                    label = 1
                    break
                if m1_low[i] <= sl:
                    label = 0
                    break
            labels[t] = label
        return labels

    _HAS_NUMBA = True
except ImportError:
    pass


def _scan_barriers_numpy(close, start_idxs, end_idxs, m1_high, m1_low,
                          tp_pct, sl_pct):
    """Numpy fallback: vectorized inner check per row (no extra deps required)."""
    n = len(close)
    labels = np.zeros(n, dtype=np.float64)
    for t in range(n):
        s, e = int(start_idxs[t]), int(end_idxs[t])
        if s >= e:
            continue
        tp = close[t] * (1.0 + tp_pct)
        sl = close[t] * (1.0 - sl_pct)
        w_h = m1_high[s:e]
        w_l = m1_low[s:e]
        tp_hits = np.nonzero(w_h >= tp)[0]
        sl_hits = np.nonzero(w_l <= sl)[0]
        tp_first = int(tp_hits[0]) if len(tp_hits) else len(w_h)
        sl_first = int(sl_hits[0]) if len(sl_hits) else len(w_h)
        if tp_first < sl_first:
            labels[t] = 1.0
    return labels


# ── Core labeling function ────────────────────────────────────────────────────

def triple_barrier_labels(
    close: np.ndarray,
    m1_times: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    merged_times: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    horizon_minutes: int,
) -> np.ndarray:
    """
    Compute triple-barrier labels for each merged-CSV row.

    Args:
        close:           Close prices at merged timestamps (aligned to merged_times).
        m1_times:        1m bar timestamps (sorted ascending, datetime64[ns]).
        m1_high:         1m bar high prices.
        m1_low:          1m bar low prices.
        merged_times:    Merged CSV timestamps (datetime64[ns]).
        tp_pct:          Take-profit fraction (e.g. 0.06 = 6%).
        sl_pct:          Stop-loss fraction   (e.g. 0.03 = 3%).
        horizon_minutes: Forward scan window  (e.g. 1440 = 24h).

    Returns:
        float64 array: 1.0 (TP hit), 0.0 (SL/expiry), NaN (no label possible).
    """
    # Convert to int64 nanoseconds for fast comparison (works in numba)
    mt_ns  = merged_times.astype("datetime64[ns]").view(np.int64)
    m1t_ns = m1_times.astype("datetime64[ns]").view(np.int64)
    horizon_ns = int(np.timedelta64(horizon_minutes, "m") / np.timedelta64(1, "ns"))

    # Vectorized window boundaries
    # start: first 1m bar AFTER the merged timestamp (don't include current bar)
    # end:   first 1m bar AFTER the horizon endpoint
    start_idxs = np.searchsorted(m1t_ns, mt_ns,              side="right").astype(np.int64)
    end_times  = mt_ns + horizon_ns
    end_idxs   = np.searchsorted(m1t_ns, end_times,          side="right").astype(np.int64)

    close_f64  = close.astype(np.float64)
    m1_high_f  = m1_high.astype(np.float64)
    m1_low_f   = m1_low.astype(np.float64)

    if _HAS_NUMBA:
        raw = _scan_barriers_jit(close_f64, start_idxs, end_idxs,
                                  m1_high_f, m1_low_f, tp_pct, sl_pct)
    else:
        print("  [labels] numba not found — using numpy fallback (slower for large datasets)",
              flush=True)
        raw = _scan_barriers_numpy(close_f64, start_idxs, end_idxs,
                                    m1_high_f, m1_low_f, tp_pct, sl_pct)

    # Mark rows where 1m data doesn't cover the full horizon as NaN
    labels = raw.copy()
    last_m1_ns = int(m1t_ns[-1]) if len(m1t_ns) > 0 else 0
    tail_mask  = (mt_ns + horizon_ns) > last_m1_ns
    labels[tail_mask] = np.nan

    return labels


# ── Data loading helper ───────────────────────────────────────────────────────

def load_ohlc_aligned(path_1m: str) -> tuple:
    """
    Load 1m OHLC and return (times, high, low) as sorted numpy arrays.
    path_1m may be:
      - a single CSV file  (data/btc_1m.csv)
      - a directory        (data/yearly_1m/) containing *_btc_1m.csv yearly splits
    """
    import glob as _glob

    if os.path.isdir(path_1m):
        files = sorted(_glob.glob(os.path.join(path_1m, "*_btc_1m.csv")))
        if not files:
            raise FileNotFoundError(f"No *_btc_1m.csv files found in {path_1m}")
        dfs = [pd.read_csv(f, parse_dates=["time"]) for f in files]
        df_1m = pd.concat(dfs, ignore_index=True)
    else:
        df_1m = pd.read_csv(path_1m, parse_dates=["time"])

    if df_1m["time"].dt.tz is None:
        df_1m["time"] = pd.to_datetime(df_1m["time"], utc=True)
    df_1m = (df_1m.sort_values("time")
                  .drop_duplicates("time")
                  .reset_index(drop=True))
    return (
        df_1m["time"].values,
        df_1m["high"].values.astype(np.float64),
        df_1m["low"].values.astype(np.float64),
    )


# ── Public entry point ────────────────────────────────────────────────────────

def apply_triple_barrier(
    df: pd.DataFrame,
    path_1m: str,
    tp_pct: float,
    sl_pct: float,
    horizon_minutes: int,
) -> pd.DataFrame:
    """
    Apply triple-barrier labels to a merged-CSV DataFrame in-place copy.

    Adds 'target' column: 1 = TP hit first, 0 = SL or time expiry.
    Drops tail rows where no label is computable (no forward 1m data).

    Raises ValueError if 'close' or 'time' columns are missing.
    """
    if "close" not in df.columns or "time" not in df.columns:
        raise ValueError("df must have 'time' and 'close' columns")

    df = df.copy()
    before = len(df)

    print(f"  [labels] Triple-barrier: TP={tp_pct*100:.0f}%  SL={sl_pct*100:.0f}%  "
          f"horizon={horizon_minutes}min  rows={before:,}", flush=True)

    m1_times, m1_high, m1_low = load_ohlc_aligned(path_1m)

    labels = triple_barrier_labels(
        close=df["close"].values,
        m1_times=m1_times,
        m1_high=m1_high,
        m1_low=m1_low,
        merged_times=df["time"].values,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        horizon_minutes=horizon_minutes,
    )

    df["target"] = labels
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    tp_rate = float(df["target"].mean())
    dropped = before - len(df)
    print(f"  [labels] Done: {len(df):,} rows (dropped {dropped} tail rows)  "
          f"TP={tp_rate:.3f}  SL={1-tp_rate:.3f}", flush=True)

    return df
