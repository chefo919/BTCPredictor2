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
from features.engineer import FEATURE_COLS, update_features
from models import tft_model, bilstm_model, meta_model

ROOT        = os.path.dirname(os.path.dirname(__file__))
MERGED_PATH = os.path.join(ROOT, "data", "btc_merged_features.csv")

# Use feature-selected subset if available
_sel_path = os.path.join(ROOT, "models", "saved", "selected_features.json")
if os.path.exists(_sel_path):
    import json as _json
    ACTIVE_FEATURES = _json.load(open(_sel_path))["selected"]
else:
    ACTIVE_FEATURES = FEATURE_COLS

from config import BUY_THRESHOLD, SELL_THRESHOLD
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


def _market_context(row: pd.Series) -> list:
    items = []
    for tf, rsi_col, macd_col in [
        ("1m",  "rsi",            "macd_diff_pct"),
        ("1H",  "h1_rsi",         "h1_macd_diff_pct"),
        ("4H",  "h4_rsi",         "h4_macd_diff_pct"),
        ("1D",  "d1_rsi",         "d1_macd_diff_pct"),
    ]:
        if rsi_col not in row.index:
            continue
        rsi_val  = float(row[rsi_col]) * 100
        macd_val = float(row.get(macd_col, 0))
        trend    = "neutral" if abs(macd_val) < 0.0001 else ("bullish" if macd_val > 0 else "bearish")
        items.append((tf, rsi_val, trend, ""))
    return items


def _key_drivers(row: pd.Series) -> list:
    candidates = []
    for tf, col in [("1m", "rsi"), ("1H", "h1_rsi"), ("4H", "h4_rsi"), ("1D", "d1_rsi")]:
        if col not in row.index:
            continue
        val = float(row[col]) * 100
        if val < 30:
            label, strength = "oversold",   (30 - val) / 30
        elif val > 70:
            label, strength = "overbought", (val - 70) / 30
        else:
            label, strength = "neutral", 0.0
        candidates.append((strength, tf, "RSI", f"{val:.1f}", label))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:3]


def generate_signal(fetch: bool = True) -> dict:
    if fetch:
        print("Fetching latest data...", flush=True)
        update_data()
        update_features()

    if not os.path.exists(MERGED_PATH):
        raise RuntimeError("btc_merged_features.csv not found. Run features/engineer.py first.")

    df = _load_tail(MERGED_PATH, n=SEQ_LEN + 10)
    df = df.dropna(subset=FEATURE_COLS)

    if len(df) < SEQ_LEN:
        raise RuntimeError(f"Not enough clean rows: {len(df)} < {SEQ_LEN}")

    now           = datetime.now(tz=timezone.utc)
    last_row      = df.iloc[-1]
    last_candle   = last_row["time"]
    current_price = float(last_row["close"]) if "close" in df.columns else float("nan")
    lag_minutes   = (now - last_candle).total_seconds() / 60

    X_seq    = df[ACTIVE_FEATURES].values
    p_tft    = tft_model.predict_proba(X_seq, ACTIVE_FEATURES)
    p_bilstm = bilstm_model.predict_proba(X_seq, ACTIVE_FEATURES)
    if meta_model.is_trained():
        final_score = meta_model.predict_proba(p_tft, p_bilstm, df.iloc[-1], ACTIVE_FEATURES)
    else:
        final_score = (p_tft + p_bilstm) / 2.0   # fallback before meta is trained

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
        "drivers":     _key_drivers(last_row),
    }


def print_signal(s: dict):
    SEP        = "=" * 42
    signal_str = s["signal_time"].strftime("%Y-%m-%d %H:%M UTC")
    candle_str = s["last_candle"].strftime("%Y-%m-%d %H:%M UTC")
    lag        = s["lag_minutes"]

    print(SEP)
    print("BTC Live Signal — Adaptive TFT")
    print(SEP)
    print(f"Signal time:  {signal_str}")
    print(f"Last candle:  {candle_str}")
    if lag <= 2:
        print(f"Data lag:     {lag:.0f} min OK")
    else:
        print(f"Data lag:     {lag:.0f} min  WARNING: data is behind")
    print(f"BTC Price:    ${s['price']:,.0f}")

    context = s.get("context", [])
    if context:
        print()
        print("Market Context:")
        for tf, rsi, trend, note in context:
            print(f"[{tf:<3}]  RSI: {rsi:5.1f}  Trend: {trend}{note}")

    print()
    print(f"TFT Score:    {s['tft_prob']:.3f}")
    print(f"Signal:       {s['signal']}")
    print(f"Confidence:   {s['confidence']}")
    print(SEP)


if __name__ == "__main__":
    if not tft_model.is_trained() or not bilstm_model.is_trained():
        print("Models not trained. Run: python training/train.py --force")
        sys.exit(1)
    s = generate_signal()
    print_signal(s)
