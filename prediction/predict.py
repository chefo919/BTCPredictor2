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
from features.engineer import (BILSTM_FEAT_COLS, TFT_FEAT_COLS,
                                get_feature_groups, update_features)
from models import tft_model, bilstm_model, meta_model
from config import BUY_THRESHOLD, SELL_THRESHOLD, SEQ_LEN_TFT, SEQ_LEN_BILSTM

ROOT       = os.path.dirname(os.path.dirname(__file__))
YEARLY_DIR = os.path.join(ROOT, "data", "yearly_merged")

_groups          = get_feature_groups()
BILSTM_FEATURES  = _groups["bilstm"]       # m15_* + h1_*
TFT_DYN_FEATURES = _groups["tft_dynamic"]  # h4_* + d1_*
TFT_STA_FEATURES = _groups["tft_static"]   # []
ALL_GATE_FEATURES = TFT_DYN_FEATURES + BILSTM_FEATURES


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


def _load_bilstm_tail(n: int) -> pd.DataFrame:
    """Load last n rows from bilstm_merged (15min-sampled, m15_* + h1_* features)."""
    if not os.path.exists(YEARLY_DIR):
        raise RuntimeError("data/yearly_merged/ not found. Run features/engineer.py first.")
    now_year = pd.Timestamp.now(tz="UTC").year
    dfs = []
    for y in [now_year - 1, now_year]:
        path = os.path.join(YEARLY_DIR, f"{y}_bilstm_merged.csv")
        if os.path.exists(path):
            dfs.append(_load_tail(path, n=n))
    if not dfs:
        raise RuntimeError("No bilstm_merged files found for current or previous year.")
    df = (pd.concat(dfs)
            .sort_values("time")
            .drop_duplicates("time")
            .tail(n)
            .reset_index(drop=True))
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _load_tft_tail(n: int) -> pd.DataFrame:
    """Load last n rows from tft_merged (4h-sampled, h4_* + d1_* features)."""
    if not os.path.exists(YEARLY_DIR):
        raise RuntimeError("data/yearly_merged/ not found. Run features/engineer.py first.")
    now_year = pd.Timestamp.now(tz="UTC").year
    dfs = []
    for y in [now_year - 1, now_year]:
        path = os.path.join(YEARLY_DIR, f"{y}_tft_merged.csv")
        if os.path.exists(path):
            dfs.append(_load_tail(path, n=n))
    if not dfs:
        raise RuntimeError("No tft_merged files found for current or previous year.")
    df = (pd.concat(dfs)
            .sort_values("time")
            .drop_duplicates("time")
            .tail(n)
            .reset_index(drop=True))
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _market_context(tft_row: pd.Series, bilstm_row: pd.Series) -> list:
    items = []
    for tf, rsi_col, macd_col, row in [
        ("15m", "m15_rsi", "m15_macd_diff_pct", bilstm_row),
        ("1H",  "h1_rsi",  "h1_macd_diff_pct",  bilstm_row),
        ("4H",  "h4_rsi",  "h4_macd_diff_pct",  tft_row),
        ("1D",  "d1_rsi",  "d1_macd_diff_pct",  tft_row),
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

    # Load pre-sampled model inputs
    df_tft    = _load_tft_tail(n=SEQ_LEN_TFT + 10)
    df_bilstm = _load_bilstm_tail(n=SEQ_LEN_BILSTM + 10)

    df_tft    = df_tft.dropna(subset=TFT_DYN_FEATURES)
    df_bilstm = df_bilstm.dropna(subset=BILSTM_FEATURES)

    if len(df_tft) < SEQ_LEN_TFT:
        raise RuntimeError(f"Not enough TFT rows: {len(df_tft)} < {SEQ_LEN_TFT}")
    if len(df_bilstm) < SEQ_LEN_BILSTM:
        raise RuntimeError(f"Not enough BiLSTM rows: {len(df_bilstm)} < {SEQ_LEN_BILSTM}")

    now             = datetime.now(tz=timezone.utc)
    last_tft_row    = df_tft.iloc[-1]
    last_bilstm_row = df_bilstm.iloc[-1]
    last_candle     = last_tft_row["time"]
    current_price   = float(last_tft_row["close"]) if "close" in df_tft.columns else float("nan")
    lag_minutes     = (now - last_candle).total_seconds() / 60

    X_dyn = df_tft[TFT_DYN_FEATURES].values
    X_sta = df_tft[TFT_STA_FEATURES].values if TFT_STA_FEATURES else np.zeros((len(df_tft), 0))
    X_bi  = df_bilstm[BILSTM_FEATURES].values

    p_tft    = tft_model.predict_proba(X_dyn, X_sta)
    p_bilstm = bilstm_model.predict_proba(X_bi)

    if meta_model.is_trained():
        # Combine last TFT and BiLSTM rows into a single gate snapshot
        gate_snapshot = pd.concat([last_tft_row, last_bilstm_row])
        final_score   = meta_model.predict_proba(p_tft, p_bilstm, gate_snapshot,
                                                  ALL_GATE_FEATURES)
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
        "context":     _market_context(last_tft_row, last_bilstm_row),
    }


if __name__ == "__main__":
    if not tft_model.is_trained() or not bilstm_model.is_trained():
        print("Models not trained. Run: python training/train.py --force")
        sys.exit(1)
    s = generate_signal()
    print(f"Signal: {s['signal']} ({s['confidence']})  score={s['final_score']:.3f}  "
          f"TFT={s['tft_prob']:.3f}  BiLSTM={s['bilstm_prob']:.3f}")
