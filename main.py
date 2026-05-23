"""
BTC Paper Trader — Web Dashboard
Run: python main.py
Opens: http://localhost:8080
"""
import os, sys, json, time, socket, threading, webbrowser, logging
from datetime import datetime, timezone
from io import StringIO
from contextlib import redirect_stdout

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["DNNL_VERBOSE"]          = "0"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from data_manager import update_csv_from_coinbase, get_live_price
from prediction.predict import generate_signal
from models import tft_model, bilstm_model, meta_model
from features.engineer import BILSTM_FEAT_COLS, TFT_FEAT_COLS, get_feature_groups

_groups          = get_feature_groups()
BILSTM_FEATURES  = _groups["bilstm"]
TFT_DYN_FEATURES = _groups["tft_dynamic"]
TFT_STA_FEATURES = _groups["tft_static"]
ALL_GATE_FEATURES = TFT_DYN_FEATURES + BILSTM_FEATURES
from papertrading.paper_trader import (
    PaperTrader, _hourly_range_pct,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MIN_HOLD_MIN, MAX_HOLD_MIN, CHOP_FILTER_PCT,
    STALE_SKIP_MIN, STARTING_USDT, FEE_RATE,
)

HOST              = "127.0.0.1"
PORT              = 8080
PRICE_HISTORY_MAX = 240
TEMPLATES_DIR     = os.path.join(ROOT, "templates")
MODELS_DIR        = os.path.join(ROOT, "models", "saved")
YEARLY_DIR        = os.path.join(ROOT, "data", "yearly_merged")
from config import BUY_THRESHOLD, SELL_THRESHOLD, SEQ_LEN_TFT
SEQ_LSTM = SEQ_LEN_TFT   # warmup window = TFT lookback (the larger of the two)


# ── JSON sanitiser ─────────────────────────────────────────────────────────────

def _j(obj):
    if isinstance(obj, bool):           return bool(obj)   # before int check
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
    if isinstance(obj, dict):  return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_j(x) for x in obj]
    return obj



# ── Shared state ───────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.lock             = threading.Lock()
        self.trader           = PaperTrader()
        self.signal           = None
        self.last_price       = 0.0
        self.price_source     = "unknown"
        self.csv_result       = {"ok": True, "new_candles": 0, "minutes_behind": 0.0}
        self.price_history    = []
        self.last_update      = None
        self.is_running       = True
        self.startup_done     = False
        self.session_start    = datetime.now(tz=timezone.utc)
        self.training_cutoff  = None   # ISO string — last date in training data
        self.backtest_from    = None   # "YYYY-MM-DD" — portfolio seed date
        self.live_since       = None   # datetime — when real-time loop started


class _BtState:
    def __init__(self):
        self.running    = False
        self.progress   = 0      # 0-100
        self.phase      = "idle" # loading|models|inference|simulating|complete|error
        self.error      = None
        self.start_date = None


state    = AppState()
bt_state = _BtState()


# ── Training cutoff loader ─────────────────────────────────────────────────────

def _load_training_cutoff() -> str | None:
    """Return the training data cutoff as 'YYYY-MM-DD'. Falls back to config if file missing."""
    path = os.path.join(MODELS_DIR, "training_cutoff.txt")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return str(pd.Timestamp(f.read().strip()).date())
        except Exception:
            pass
    from config import TRAINING_CUTOFF_DATE
    return str(pd.Timestamp(TRAINING_CUTOFF_DATE).date())


# ── Backtest helpers ───────────────────────────────────────────────────────────

def _bt_to_ts(np_time):
    """Convert numpy datetime64 to timezone-aware datetime."""
    ts = pd.Timestamp(np_time).to_pydatetime()
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _bt_close(local_trader, price, ts, reason, entry_price):
    gross   = local_trader.btc * price
    fee     = gross * FEE_RATE
    net     = gross - fee
    pnl     = net - local_trader.cost_basis
    pnl_pct = pnl / local_trader.cost_basis * 100
    hold_m  = (ts - local_trader.entry_time).total_seconds() / 60 if local_trader.entry_time else 0
    sold    = local_trader.btc

    local_trader.usdt       += net
    local_trader.btc         = 0.0
    local_trader.entry_price = 0.0
    local_trader.trades.append({
        "action":        "SELL",
        "time":          ts.strftime("%Y-%m-%d %H:%M UTC"),
        "time_iso":      ts.isoformat(),
        "price":         round(price, 2),
        "btc_amount":    round(sold, 5),
        "fee":           round(fee, 2),
        "usdt_received": round(net, 2),
        "pnl":           round(pnl, 2),
        "pnl_pct":       round(pnl_pct, 2),
        "hold_min":      round(hold_m, 1),
        "reason":        reason,
    })


# ── Backtest thread ────────────────────────────────────────────────────────────

def _run_backtest(start_date_str: str):
    """
    Runs full batch backtest on historical CSV, then seeds state.trader with results.
    Real-time trading loop skips trade execution while this runs.
    """
    try:
        bt_state.running  = True
        bt_state.error    = None
        bt_state.progress = 0
        bt_state.phase    = "loading"

        start_dt = pd.Timestamp(start_date_str, tz="UTC")
        print(f"[Backtest] Starting from {start_date_str}...", flush=True)

        # Load from yearly files (include one year before start for seq warm-up)
        def _load_yearly(pattern):
            dfs = []
            for fname in sorted(os.listdir(YEARLY_DIR)):
                if not fname.endswith(f"_{pattern}.csv"):
                    continue
                year = int(fname.split("_")[0])
                if year < start_dt.year - 1:
                    continue
                df_y = pd.read_csv(os.path.join(YEARLY_DIR, fname), parse_dates=["time"])
                if df_y["time"].dt.tz is None:
                    df_y["time"] = pd.to_datetime(df_y["time"], utc=True)
                dfs.append(df_y)
            if not dfs:
                return None
            return (pd.concat(dfs, ignore_index=True)
                      .sort_values("time")
                      .drop_duplicates("time")
                      .pipe(lambda d: d[d["time"] <= pd.Timestamp.now(tz="UTC")])
                      .reset_index(drop=True))

        df_tft    = _load_yearly("tft_merged")
        df_bilstm = _load_yearly("bilstm_merged")
        if df_tft is None or df_bilstm is None:
            bt_state.error   = "No yearly feature files found. Run features/engineer.py."
            bt_state.phase   = "error"
            bt_state.running = False
            return

        df_tft    = df_tft.dropna(subset=TFT_DYN_FEATURES).reset_index(drop=True)
        df_bilstm = df_bilstm.dropna(subset=BILSTM_FEATURES).reset_index(drop=True)

        mask_tft = df_tft["time"] >= start_dt
        if not mask_tft.any():
            bt_state.error   = f"No TFT data found from {start_date_str}"
            bt_state.phase   = "error"
            bt_state.running = False
            return

        candidate_tft    = int(np.where(mask_tft.values)[0][0])
        first_idx_tft    = max(candidate_tft, SEQ_LSTM + 1)

        mask_bilstm      = df_bilstm["time"] >= start_dt
        candidate_bilstm = int(np.where(mask_bilstm.values)[0][0]) if mask_bilstm.any() else len(df_bilstm)
        from config import SEQ_LEN_BILSTM as _SEQ_BI
        first_idx_bilstm = max(candidate_bilstm, _SEQ_BI + 1)

        n = len(df_tft) - first_idx_tft

        print(f"[Backtest] {n:,} rows to process", flush=True)

        # Load models — ensemble handled internally by model modules
        bt_state.phase    = "models"
        bt_state.progress = 5
        from models import tft_model as _tft_mod, bilstm_model as _bi_mod, meta_model as _meta

        bt_state.progress = 10
        bt_state.phase    = "inference"
        batch = 512

        # n_static for TFT (0 for current models — no static covariates)
        _ns_val  = 0
        _ns_path = os.path.join(MODELS_DIR, "tft_n_static.txt")
        if os.path.exists(_ns_path):
            with open(_ns_path) as _f:
                _ns_val = int(_f.read().strip() or "0")

        # TFT pass — 3-seed ensemble averaged internally
        print("[Backtest] TFT ensemble inference...", flush=True)
        X_dyn    = df_tft[TFT_DYN_FEATURES].values.astype(np.float32)
        X_sta    = np.zeros((len(df_tft), _ns_val), dtype=np.float32)
        _tft_all = _tft_mod.predict_proba_batch(X_dyn, X_sta, batch_size=batch)
        tft_probs = _tft_all[first_idx_tft:first_idx_tft + n].astype(np.float32)
        bt_state.progress = 45

        # BiLSTM pass — 3-seed ensemble averaged internally
        print("[Backtest] BiLSTM ensemble inference...", flush=True)
        X_bi_all         = df_bilstm[BILSTM_FEATURES].values.astype(np.float32)
        _bi_all          = _bi_mod.predict_proba_batch(X_bi_all, batch_size=batch)
        n_bilstm         = len(df_bilstm) - first_idx_bilstm
        bilstm_probs_raw = _bi_all[first_idx_bilstm:first_idx_bilstm + n_bilstm].astype(np.float32)
        bt_state.progress = 80

        # Align bilstm probs onto tft timestamps via nearest join
        tft_times_active    = pd.DatetimeIndex(df_tft["time"].values[first_idx_tft:])
        bilstm_times_active = pd.DatetimeIndex(df_bilstm["time"].values[first_idx_bilstm:])
        _df_bi_p = pd.DataFrame({"time": bilstm_times_active, "p_bi": bilstm_probs_raw})
        _df_tft_t = pd.DataFrame({"time": tft_times_active})
        _aligned = pd.merge_asof(_df_tft_t.sort_values("time"),
                                  _df_bi_p.sort_values("time"),
                                  on="time", direction="nearest",
                                  tolerance=pd.Timedelta("3h"))
        bilstm_probs = _aligned["p_bi"].fillna(0.5).values.astype(np.float32)

        # Build combined gate snapshot (tft features + bilstm features aligned)
        tft_slice    = df_tft.iloc[first_idx_tft:].reset_index(drop=True)
        bilstm_slice = df_bilstm.iloc[first_idx_bilstm:].reset_index(drop=True)
        bilstm_snap  = pd.merge_asof(
            tft_slice[["time"]].sort_values("time"),
            bilstm_slice[["time"] + BILSTM_FEATURES].sort_values("time"),
            on="time", direction="nearest", tolerance=pd.Timedelta("3h"),
        ).drop(columns=["time"])
        df_gate = pd.concat([tft_slice[TFT_DYN_FEATURES].reset_index(drop=True),
                              bilstm_snap.reset_index(drop=True)], axis=1)

        # Meta pass
        final_probs = _meta.predict_proba_batch(tft_probs, bilstm_probs, df_gate, ALL_GATE_FEATURES)
        bt_state.progress = 85

        bt_state.phase    = "simulating"
        bt_state.progress = 85
        print(f"[Backtest] Simulating trades...", flush=True)

        local_trader = PaperTrader()
        closes   = df_tft["close"].values.astype(np.float64)
        times    = df_tft["time"].values
        close_s  = pd.Series(closes)
        chop_arr = ((close_s.rolling(20, min_periods=3).max()
                     - close_s.rolling(20, min_periods=3).min()) / close_s).values
        entry_price = 0.0
        entry_time  = None

        for step in range(n):
            abs_i  = first_idx_tft + step
            price  = closes[abs_i]
            ts     = _bt_to_ts(times[abs_i])
            score  = float(final_probs[step])

            if score > BUY_THRESHOLD:
                signal = "BUY";  conf = "HIGH" if score > 0.75 else "MEDIUM"
            elif score < SELL_THRESHOLD:
                signal = "SELL"; conf = "HIGH" if score < 0.25 else "MEDIUM"
            else:
                signal = "HOLD"; conf = "LOW"

            local_trader.value_history.append(local_trader.usdt + local_trader.btc * price)

            if local_trader.btc > 0:
                pnl_f    = (price - entry_price) / entry_price
                hold_min = (ts - entry_time).total_seconds() / 60 if entry_time else 0
                reason = None
                if   pnl_f <= -STOP_LOSS_PCT:                              reason = "STOP_LOSS"
                elif pnl_f >= TAKE_PROFIT_PCT:                             reason = "TAKE_PROFIT"
                elif hold_min >= MAX_HOLD_MIN:                             reason = "MAX_HOLD"
                elif (signal == "SELL" and conf in ("HIGH","MEDIUM")
                      and hold_min >= MIN_HOLD_MIN):                       reason = "SIGNAL"
                if reason:
                    _bt_close(local_trader, price, ts, reason, entry_price)

            elif (local_trader.btc == 0 and signal == "BUY"
                  and conf in ("HIGH","MEDIUM")):
                rv = chop_arr[abs_i]
                if not np.isnan(rv) and rv >= CHOP_FILTER_PCT:
                    cap  = local_trader.usdt if conf == "HIGH" else local_trader.usdt * 0.5
                    fee  = cap * FEE_RATE
                    btcr = (cap - fee) / price
                    local_trader.usdt        -= cap
                    local_trader.btc          = btcr
                    local_trader.entry_price  = price
                    local_trader.entry_time   = ts
                    local_trader.cost_basis   = cap
                    local_trader.entry_lstm_prob = score
                    local_trader.entry_xgb_prob  = score
                    local_trader.entry_transformer_prob = None
                    entry_price = price
                    entry_time  = ts
                    local_trader.trades.append({
                        "action":    "BUY",
                        "time":      ts.strftime("%Y-%m-%d %H:%M UTC"),
                        "time_iso":  ts.isoformat(),
                        "price":     round(price, 2),
                        "btc_amount": round(btcr, 5),
                        "fee":       round(fee, 2),
                        "usdt_spent": round(cap, 2),
                        "confidence": conf,
                        "tft_prob":   round(score, 4),
                        "pnl": None, "pnl_pct": None, "hold_min": None,
                    })

            if step % 5000 == 0 and n:
                bt_state.progress = 85 + int(step / n * 14)

        # Close any open position at end
        if local_trader.btc > 0:
            last_ts = _bt_to_ts(times[-1])
            _bt_close(local_trader, closes[-1], last_ts, "END_OF_BACKTEST", entry_price)

        # Build price history for 4-hour chart (last PRICE_HISTORY_MAX minutes)
        trade_times = {}
        for t in local_trader.trades:
            ts_key = t.get("time", "")[:16]   # "YYYY-MM-DD HH:MM"
            trade_times[ts_key] = t["action"]

        ph_start = max(0, n - PRICE_HISTORY_MAX)
        price_hist = []
        for step in range(ph_start, n):
            abs_i  = first_idx_tft + step
            ts     = _bt_to_ts(times[abs_i])
            ts_key = ts.strftime("%Y-%m-%d %H:%M")
            price_hist.append({
                "time":   ts.isoformat(),
                "price":  float(closes[abs_i]),
                "action": trade_times.get(ts_key),
            })

        # Atomically swap trader + price history
        with state.lock:
            state.trader        = local_trader
            state.price_history = price_hist
            state.last_price    = float(closes[-1])
            state.startup_done  = True

        sells = [t for t in local_trader.trades if t["action"] == "SELL"
                 and t.get("reason") != "END_OF_BACKTEST"]
        wins  = sum(1 for t in sells if (t.get("pnl") or 0) >= 0)
        final = local_trader._total(closes[-1])
        print(f"[Backtest] Complete — ${final:,.2f}  |  {len(sells)} trades  {wins}W/{len(sells)-wins}L",
              flush=True)

        bt_state.progress = 100
        bt_state.phase    = "complete"

    except Exception as e:
        import traceback; traceback.print_exc()
        bt_state.error = str(e)
        bt_state.phase = "error"
    finally:
        bt_state.running = False


# ── Real-time trade execution ──────────────────────────────────────────────────

def _execute_trade(trader: PaperTrader, price: float, sig: dict) -> str | None:
    """Mirror of paper_trader.py exit/entry logic. Must be called with state.lock held."""
    if bt_state.running:
        return None   # pause real-time trades during backtest

    signal = sig["signal"]
    conf   = sig["confidence"]

    if trader.btc > 0:
        pnl_frac = (price - trader.entry_price) / trader.entry_price
        hold_min = (datetime.now(tz=timezone.utc) - trader.entry_time).total_seconds() / 60

        if pnl_frac <= -STOP_LOSS_PCT:
            trade = trader._sell(price)
            if trade:
                print(f"[Trade] STOP LOSS   @ ${price:,.0f}  PnL: {trade['pnl']:+.2f}", flush=True)
                return "SELL"
        elif pnl_frac >= TAKE_PROFIT_PCT:
            trade = trader._sell(price)
            if trade:
                print(f"[Trade] TAKE PROFIT @ ${price:,.0f}  PnL: {trade['pnl']:+.2f}", flush=True)
                return "SELL"
        elif hold_min >= MAX_HOLD_MIN:
            trade = trader._sell(price)
            if trade:
                print(f"[Trade] MAX HOLD    @ ${price:,.0f}  PnL: {trade['pnl']:+.2f}  ({hold_min:.0f}min)", flush=True)
                return "SELL"
        elif signal == "SELL" and conf in ("HIGH", "MEDIUM") and hold_min >= MIN_HOLD_MIN:
            trade = trader._sell(price)
            if trade:
                print(f"[Trade] SELL        @ ${price:,.0f}  PnL: {trade['pnl']:+.2f}", flush=True)
                return "SELL"

    elif trader.btc == 0 and signal == "BUY" and conf in ("HIGH", "MEDIUM"):
        if _hourly_range_pct() >= CHOP_FILTER_PCT:
            tft_p = sig.get("tft_prob", 0.5)
            trade = trader._buy(price, tft_p, tft_p, conf, None)
            if trade:
                print(f"[Trade] BUY         @ ${price:,.0f}  [{conf}]  TFT:{tft_p:.3f}  "
                      f"${trade['usdt_spent']:,.2f}", flush=True)
                return "BUY"

    return None


# ── Real-time trading cycle ────────────────────────────────────────────────────

def _run_cycle():
    try:
        csv_result = update_csv_from_coinbase()
    except Exception as e:
        csv_result = {"ok": False, "new_candles": 0, "minutes_behind": 999, "message": str(e)}

    try:
        live_price, price_source = get_live_price()
    except Exception:
        live_price, price_source = state.last_price, "last known"

    if live_price <= 0:
        return

    lag = csv_result.get("minutes_behind", 0)
    sig = None
    if lag < STALE_SKIP_MIN:
        try:
            sig = generate_signal(fetch=False)
        except Exception as e:
            print(f"[Signal] Error: {e}", flush=True)

    with state.lock:
        action = None
        if sig and not bt_state.running:
            action = _execute_trade(state.trader, live_price, sig)
            state.signal = sig
        elif sig:
            state.signal = sig   # keep signal fresh even during backtest

        state.last_price   = live_price
        state.price_source = price_source
        state.csv_result   = csv_result
        state.last_update  = datetime.now(tz=timezone.utc).isoformat()
        state.trader.value_history.append(state.trader._total(live_price))
        state.price_history.append({"time": state.last_update, "price": live_price, "action": action})
        if len(state.price_history) > PRICE_HISTORY_MAX:
            state.price_history = state.price_history[-PRICE_HISTORY_MAX:]
        if not bt_state.running:
            state.startup_done = True


def _trading_thread():
    print("[Trader] Paper trading loop started.", flush=True)
    while state.is_running:
        t0 = time.time()
        try:
            _run_cycle()
        except Exception as e:
            print(f"[Trader] Cycle error: {e}", flush=True)
        for _ in range(max(0, int(60 - (time.time() - t0)))):
            if not state.is_running:
                break
            time.sleep(1)
    print("[Trader] Stopped.", flush=True)


# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="BTC Paper Trader")


@app.get("/")
def root():
    return FileResponse(os.path.join(TEMPLATES_DIR, "dashboard.html"))


@app.get("/api/signal")
def api_signal():
    with state.lock:
        if not state.startup_done and not bt_state.running:
            return JSONResponse({"status": "loading"})
        sig = state.signal
        if sig is None:
            return JSONResponse({"status": "no_signal"})
        return JSONResponse(_j({
            "status":         "ok",
            "signal":         sig["signal"],
            "confidence":     sig["confidence"],
            "final_score":    sig["final_score"],
            "tft_prob":       sig.get("tft_prob"),
            "last_update":    state.last_update,
            "price":          state.last_price,
            "price_source":   state.price_source,
            "minutes_behind": state.csv_result.get("minutes_behind", 0),
        }))


@app.get("/api/portfolio")
def api_portfolio():
    with state.lock:
        trader  = state.trader
        price   = state.last_price or 0.0
        btc_val = round(trader.btc * price, 2)
        total   = trader._total(price)
        pnl_abs = total - STARTING_USDT
        pnl_pct = pnl_abs / STARTING_USDT * 100

        if len(trader.value_history) >= 2:
            v      = np.array(trader.value_history)
            peak   = np.maximum.accumulate(v)
            max_dd = float(np.min((v - peak) / peak * 100))
        else:
            max_dd = 0.0

        sells  = [t for t in trader.trades if t["action"] == "SELL"]
        rets   = [t["pnl_pct"] for t in sells if t.get("pnl_pct") is not None]
        sharpe = 0.0
        if len(rets) >= 2:
            std    = float(np.std(rets))
            sharpe = round(float(np.mean(rets)) / std, 2) if std > 0 else 0.0

        wins = [t for t in sells if t.get("pnl", 0) >= 0]

        position = None
        if trader.btc > 0 and trader.entry_price:
            pos_pnl = (price - trader.entry_price) / trader.entry_price * 100
            hold_m  = (datetime.now(tz=timezone.utc) - trader.entry_time).total_seconds() / 60 \
                      if trader.entry_time else 0
            position = {
                "entry_price": trader.entry_price,
                "entry_time":  trader.entry_time.isoformat() if trader.entry_time else None,
                "btc_amount":  round(trader.btc, 5),
                "btc_value":   btc_val,
                "pnl_pct":     round(pos_pnl, 2),
                "pnl_abs":     round((price - trader.entry_price) * trader.btc, 2),
                "hold_min":    round(hold_m, 0),
            }

        return JSONResponse(_j({
            "usdt":             round(trader.usdt, 2),
            "btc":              round(trader.btc, 5),
            "btc_value":        btc_val,
            "total_value":      round(total, 2),
            "pnl_abs":          round(pnl_abs, 2),
            "pnl_pct":          round(pnl_pct, 2),
            "max_drawdown":     round(max_dd, 2),
            "sharpe_ratio":     sharpe,
            "total_trades":     len(sells),
            "winning_trades":   len(wins),
            "losing_trades":    len(sells) - len(wins),
            "win_rate":         round(len(wins) / len(sells) * 100, 1) if sells else 0.0,
            "starting_capital": STARTING_USDT,
            "position":         position,
            "session_start":    state.session_start.isoformat(),
            "training_cutoff":  state.training_cutoff,
            "backtest_from":    state.backtest_from,
            "live_since":       state.live_since.isoformat() if state.live_since else None,
        }))


@app.get("/api/trades")
def api_trades():
    with state.lock:
        return JSONResponse({"trades": _j(state.trader.trades[-50:])})


@app.get("/api/context")
def api_context():
    with state.lock:
        sig = state.signal
        if sig is None:
            return JSONResponse({"context": [], "drivers": []})
        return JSONResponse(_j({
            "context": sig.get("context", []),
            "drivers": [list(d) for d in sig.get("drivers", [])],
        }))


def _load_model_accuracies() -> dict:
    path = os.path.join(MODELS_DIR, "model_accuracies.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@app.get("/api/models")
def api_models():
    with state.lock:
        sig  = state.signal
        accs = _load_model_accuracies()
        return JSONResponse(_j({
            "model":           "TFT-ACB-XML",
            "tft_accuracy":    accs.get("tft"),
            "bilstm_accuracy": accs.get("bilstm"),
            "meta_accuracy":   accs.get("meta"),
            "tft_prob":        sig.get("tft_prob")    if sig else None,
            "bilstm_prob":     sig.get("bilstm_prob") if sig else None,
            "final_score":     sig.get("final_score") if sig else None,
            "training_cutoff": state.training_cutoff,
        }))


@app.get("/api/price")
def api_price():
    with state.lock:
        return JSONResponse({"history": _j(state.price_history)})


@app.get("/api/equity")
def api_equity():
    with state.lock:
        trades = state.trader.trades
        sells  = [t for t in trades if t["action"] == "SELL"
                  and t.get("reason") != "END_OF_BACKTEST"]
        capital = STARTING_USDT
        curve   = []
        for i, t in enumerate(sells):
            pnl     = t.get("pnl") or 0.0
            capital += pnl
            curve.append({
                "n":      int(i + 1),
                "time":   str(t.get("time", "")),
                "pnl":    float(round(pnl, 2)),
                "total":  float(round(capital, 2)),
                "win":    bool(pnl >= 0),
                "reason": str(t.get("reason", "signal")),
                "price":  float(t.get("price", 0) or 0),
            })
        return JSONResponse({"curve": _j(curve), "starting": STARTING_USDT})


# ── Backtest endpoints ─────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    start_date: str   # "YYYY-MM-DD"


@app.post("/api/backtest/start")
def api_backtest_start(req: BacktestRequest):
    if bt_state.running:
        return JSONResponse({"ok": False, "error": "Backtest already running"})
    bt_state.start_date = req.start_date
    threading.Thread(target=_run_backtest, args=(req.start_date,),
                     daemon=True, name="Backtest").start()
    return JSONResponse({"ok": True})


@app.get("/api/backtest/status")
def api_backtest_status():
    return JSONResponse({
        "running":    bt_state.running,
        "progress":   bt_state.progress,
        "phase":      bt_state.phase,
        "error":      bt_state.error,
        "start_date": bt_state.start_date,
    })


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 46)
    print("  BTC Paper Trader  —  Web Dashboard")
    print("=" * 46)

    if not tft_model.is_trained() or not bilstm_model.is_trained() or not meta_model.is_trained():
        print("ERROR: Models not trained. Run: python training/train.py --force")
        sys.exit(1)

    with socket.socket() as s:
        try:
            s.bind((HOST, PORT))
        except OSError:
            print(f"ERROR: Port {PORT} is already in use.")
            sys.exit(1)

    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    # ── Load training cutoff and seed portfolio from backtest ─────────────────
    cutoff = _load_training_cutoff()
    if cutoff:
        state.training_cutoff = cutoff
        state.backtest_from   = cutoff
        print(f"[Init] Models trained up to: {cutoff}", flush=True)
        print(f"[Init] Seeding portfolio from backtest ({cutoff} -> today)...", flush=True)
        _run_backtest(cutoff)   # runs synchronously before server starts
        if bt_state.error:
            print(f"[Init] Backtest error: {bt_state.error} — starting real-time only", flush=True)
        else:
            sells = [t for t in state.trader.trades if t["action"] == "SELL"]
            wins  = sum(1 for t in sells if (t.get("pnl") or 0) >= 0)
            total = state.trader._total(state.last_price or 0)
            pnl   = total - STARTING_USDT
            sign  = "+" if pnl >= 0 else ""
            print(f"[Init] Backtest complete: {len(sells)} trades  "
                  f"{wins}W/{len(sells)-wins}L  "
                  f"Portfolio: ${total:,.2f} ({sign}${pnl:,.2f})", flush=True)
    else:
        print(f"[Init] No training cutoff found — starting real-time only", flush=True)

    # ── Start real-time trading loop ──────────────────────────────────────────
    state.live_since  = datetime.now(tz=timezone.utc)
    state.startup_done = True   # backtest already populated state

    trade_thread = threading.Thread(target=_trading_thread, daemon=True, name="TradingLoop")
    trade_thread.start()

    print(f"[Init] Live trading started at {state.live_since.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    print(f"[Init] Dashboard at http://{HOST}:{PORT}", flush=True)

    def _open():
        time.sleep(1.5)
        webbrowser.open(f"http://{HOST}:{PORT}")

    threading.Thread(target=_open, daemon=True).start()
    print("Press Ctrl+C to stop.\n", flush=True)

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    state.is_running = False


if __name__ == "__main__":
    main()
