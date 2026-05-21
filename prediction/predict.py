import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["DNNL_VERBOSE"]          = "0"

import sys
import logging
from io import StringIO
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_manager import update_data
from features.engineer import FEATURE_COLS, get_feature_groups, update_features
from models import tft_model, bilstm_model, meta_model
from config import BUY_THRESHOLD, SELL_THRESHOLD

ROOT       = os.path.dirname(os.path.dirname(__file__))
YEARLY_DIR = os.path.join(ROOT, "data", "yearly")

_groups          = get_feature_groups()
BILSTM_FEATURES  = _groups["bilstm"]
TFT_DYN_FEATURES = _groups["tft_dynamic"]
TFT_STA_FEATURES = _groups["tft_static"]
ALL_FEATURES     = BILSTM_FEATURES + TFT_DYN_FEATURES + TFT_STA_FEATURES

# Load tail needs enough rows for the TFT (largest window)
SEQ_LEN = tft_model.SEQ_LEN


def _load_tail(path: str, n: int = 200) -> pd.DataFrame:
    READ_BYTES = 2 * 1024 * 1024
    with open(path, "rb") as f:
        header   = f.readline().decode("utf-8").strip()
        f.seek(0, 2)
        seek_pos = max(f.tell() - READ_BYTES, 1)
        f.seek(seek_pos)
        data = f.read()
    lines   = data.decode("utf-8", errors="replace").split("\n")
    lines   = [l for l in lines[1:] if l.strip()]
    csv_str = header + "\n" + "\n".join(lines[-n:])
    df = pd.read_csv(StringIO(csv_str), parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _load_yearly_tail(n: int) -> pd.DataFrame:
    """Load the last n rows from yearly files (current + previous year for seq warm-up)."""
    if not os.path.exists(YEARLY_DIR):
        raise RuntimeError("data/yearly/ not found. Run features/engineer.py first.")
    now_year = pd.Timestamp.now(tz="UTC").year
    dfs = []
    for y in [now_year - 1, now_year]:
        path = os.path.join(YEARLY_DIR, f"{y}_merged.csv")
        if os.path.exists(path):
            dfs.append(_load_tail(path, n=n))
    if not dfs:
        raise RuntimeError("No yearly merged files found for current or previous year.")
    df = (pd.concat(dfs)
            .sort_values("time")
            .drop_duplicates("time")
            .tail(n)
            .reset_index(drop=True))
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _market_context(row: pd.Series) -> list:
    items = []
    for tf, rsi_col, macd_col in [
        ("1H",  "h1_rsi",  "h1_macd_diff_pct"),
        ("4H",  "h4_rsi",  "h4_macd_diff_pct"),
        ("1D",  "d1_rsi",  "d1_macd_diff_pct"),
    ]:
        if rsi_col not in row.index:
            continue
        rsi_val  = float(row[rsi_col]) * 100
        macd_val = float(row.get(macd_col, 0))
        trend    = "neutral" if abs(macd_val) < 0.0001 else ("bullish" if macd_val > 0 else "bearish")
        items.append((tf, rsi_val, trend, ""))
    return items


def generate_signal(fetch: bool = True) -> dict:
    if fetch:
        print("Fetching latest data...", flush=True)
        update_data()
        update_features()

    df = _load_yearly_tail(n=SEQ_LEN + 10)
    df = df.dropna(subset=ALL_FEATURES)

    if len(df) < SEQ_LEN:
        raise RuntimeError(f"Not enough clean rows: {len(df)} < {SEQ_LEN}")

    now           = datetime.now(tz=timezone.utc)
    last_row      = df.iloc[-1]
    last_candle   = last_row["time"]
    current_price = float(last_row["close"]) if "close" in df.columns else float("nan")
    lag_minutes   = (now - last_candle).total_seconds() / 60

    X_dyn    = df[TFT_DYN_FEATURES].values
    X_sta    = df[TFT_STA_FEATURES].values
    X_bi     = df[BILSTM_FEATURES].values

    p_tft    = tft_model.predict_proba(X_dyn, X_sta)
    p_bilstm = bilstm_model.predict_proba(X_bi)

    if meta_model.is_trained():
        final_score = meta_model.predict_proba(p_tft, p_bilstm, last_row,
                                                TFT_DYN_FEATURES + TFT_STA_FEATURES)
    else:
        final_score = (p_tft + p_bilstm) / 2.0

    if final_score > BUY_THRESHOLD:
        signal = "BUY"
        conf   = "HIGH" if final_score > 0.75 else "MEDIUM"
    elif final_score < SELL_THRESHOLD:
        signal = "SELL"
        conf   = "HIGH" if final_score < 0.25 else "MEDIUM"
    else:
        signal = "HOLD"
        conf   = "LOW"

    return {
        "signal_time": now,
        "last_candle": last_candle,
        "lag_minutes": lag_minutes,
        "price":       current_price,
        "tft_prob":    p_tft,
        "bilstm_prob": p_bilstm,
        "final_score": final_score,
        "signal":      signal,
        "confidence":  conf,
        "context":     _market_context(last_row),
    }


if __name__ == "__main__":
    if not tft_model.is_trained() or not bilstm_model.is_trained():
        print("Models not trained. Run: python training/train.py --force")
        sys.exit(1)
    s = generate_signal()
    print(f"Signal: {s['signal']} ({s['confidence']})  score={s['final_score']:.3f}  "
          f"TFT={s['tft_prob']:.3f}  BiLSTM={s['bilstm_prob']:.3f}")
