import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["DNNL_VERBOSE"]          = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from features.engineer import FEATURE_COLS

import json as _json
_sel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "selected_features.json")
ACTIVE_FEATURES = (_json.load(open(_sel_path))["selected"]
                   if os.path.exists(_sel_path) else FEATURE_COLS)

from models.tft_model    import GRN, VSN, SEQ_LEN as SEQ_TFT
from models.bilstm_model import _SumPool
from models import bilstm_model as _bilstm_mod, meta_model as _meta_mod

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.dirname(__file__))
MERGED_PATH   = os.path.join(ROOT, "data", "btc_merged_features.csv")
MODELS_DIR    = os.path.join(ROOT, "models", "saved")
BACKTESTS_DIR = os.path.join(os.path.dirname(__file__), "backtests")

# ── Trading constants (from central config) ───────────────────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (FEE_RATE, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
                    MIN_HOLD_MIN as MIN_HOLD_ROWS, MAX_HOLD_MIN as MAX_HOLD_ROWS,
                    CHOP_FILTER_PCT, BUY_THRESHOLD, SELL_THRESHOLD, STARTING_USDT)
DEFAULT_CAPITAL = STARTING_USDT
INFER_BATCH     = 512

SEP = "=" * 46


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models():
    tft_path    = os.path.join(MODELS_DIR, "tft.keras")
    tft_s_path  = os.path.join(MODELS_DIR, "tft_scaler.pkl")
    bi_path     = os.path.join(MODELS_DIR, "bilstm.keras")
    bi_s_path   = os.path.join(MODELS_DIR, "bilstm_scaler.pkl")
    meta_path   = os.path.join(MODELS_DIR, "meta_xgb.pkl")

    for p in [tft_path, tft_s_path, bi_path, bi_s_path, meta_path]:
        if not os.path.exists(p):
            print(f"[Backtest] Model not found: {p}. Run: python training/train.py --force")
            sys.exit(1)

    print("[Backtest] Loading TFT-ACB-XML models...", flush=True)
    tft_m    = tf.keras.models.load_model(tft_path, custom_objects={"GRN": GRN, "VSN": VSN})
    tft_s    = joblib.load(tft_s_path)
    bilstm_m = tf.keras.models.load_model(bi_path, custom_objects={"_SumPool": _SumPool})
    bilstm_s = joblib.load(bi_s_path)
    meta_m   = joblib.load(meta_path)["model"]
    meta_w   = joblib.load(meta_path)["weights"]
    print(f"[Backtest] Models loaded — TFT weight: {meta_w['tft']:.3f}  BiLSTM weight: {meta_w['bilstm']:.3f}",
          flush=True)
    return tft_m, tft_s, bilstm_m, bilstm_s, meta_m, meta_w


# ── Batch inference ───────────────────────────────────────────────────────────

def _batch_infer(X_all, times_all, first_idx, tft_m, tft_s, bilstm_m, bilstm_s, meta_m, meta_w, df):
    n         = len(X_all) - first_idx
    n_batches = (n + INFER_BATCH - 1) // INFER_BATCH
    SEQ_BILSTM = _bilstm_mod.SEQ_LEN

    # ── TFT inference ────────────────────────────────────────────────────────
    print("[Backtest] Running TFT inference...", flush=True)
    tft_sc    = tft_s.transform(X_all).astype(np.float32)
    tft_probs = np.full(n, 0.5, dtype=np.float32)
    last_pct  = -1
    for b in range(n_batches):
        b_start = b * INFER_BATCH
        b_end   = min(b_start + INFER_BATCH, n)
        abs_idx = np.arange(first_idx + b_start, first_idx + b_end)
        win_idx = abs_idx[:, None] + np.arange(-SEQ_TFT + 1, 1)[None, :]
        preds   = tft_m.predict(tft_sc[win_idx], verbose=0, batch_size=INFER_BATCH)
        tft_probs[b_start:b_end] = preds.flatten()
        pct = int(b_end / n * 50)
        if pct // 10 > last_pct // 10:
            row_ts = pd.Timestamp(times_all[first_idx + b_end - 1]).strftime("%b %Y")
            print(f"[Backtest] TFT... {pct}% ({row_ts})", flush=True)
            last_pct = pct
    del tft_sc

    # ── BiLSTM inference ─────────────────────────────────────────────────────
    print("[Backtest] Running BiLSTM inference...", flush=True)
    bilstm_sc    = bilstm_s.transform(X_all).astype(np.float32)
    bilstm_probs = np.full(n, 0.5, dtype=np.float32)
    last_pct = -1
    for b in range(n_batches):
        b_start = b * INFER_BATCH
        b_end   = min(b_start + INFER_BATCH, n)
        abs_idx = np.arange(first_idx + b_start, first_idx + b_end)
        win_idx = abs_idx[:, None] + np.arange(-SEQ_BILSTM + 1, 1)[None, :]
        win_idx = np.clip(win_idx, 0, len(bilstm_sc) - 1)
        preds   = bilstm_m.predict(bilstm_sc[win_idx], verbose=0, batch_size=INFER_BATCH)
        bilstm_probs[b_start:b_end] = preds.flatten()
        pct = 50 + int(b_end / n * 40)
        if pct // 10 > last_pct // 10:
            row_ts = pd.Timestamp(times_all[first_idx + b_end - 1]).strftime("%b %Y")
            print(f"[Backtest] BiLSTM... {pct}% ({row_ts})", flush=True)
            last_pct = pct
    del bilstm_sc

    # ── XGBoost meta-learner ──────────────────────────────────────────────────
    print("[Backtest] Running meta-learner...", flush=True)
    w_tft, w_bilstm = meta_w["tft"], meta_w["bilstm"]
    final_probs = _meta_mod.predict_proba_batch(tft_probs, bilstm_probs,
                                                  df.iloc[first_idx:].reset_index(drop=True),
                                                  ACTIVE_FEATURES)
    print(f"[Backtest] Processing... 100%", flush=True)
    return final_probs


# ── Trade simulation ──────────────────────────────────────────────────────────

def _simulate(df, first_idx, tft_probs, starting_capital):
    closes    = df["close"].values.astype(np.float64)
    times_arr = df["time"].values
    n         = len(df) - first_idx

    # Pre-compute hourly chop filter
    close_s  = pd.Series(closes)
    roll_max = close_s.rolling(60, min_periods=5).max()
    roll_min = close_s.rolling(60, min_periods=5).min()
    chop_arr = ((roll_max - roll_min) / close_s).values

    usdt    = float(starting_capital)
    btc     = 0.0
    entry_price = entry_row = cost_basis = 0.0
    entry_time  = None

    trades   = []
    exits    = {"STOP_LOSS": [], "TAKE_PROFIT": [], "MAX_HOLD": [],
                "SIGNAL": [], "END_OF_BACKTEST": []}
    value_history = []
    hold_sig = hold_buy = 0
    win_trades = loss_trades = 0

    def _do_sell(abs_i, price, reason):
        nonlocal usdt, btc, entry_price, entry_row, cost_basis, entry_time
        nonlocal win_trades, loss_trades
        gross = btc * price
        fee   = gross * FEE_RATE
        net   = gross - fee
        pnl   = net - cost_basis
        hold_rows = abs_i - entry_row
        if pnl >= 0:
            win_trades += 1
        else:
            loss_trades += 1
        ts = pd.Timestamp(times_arr[abs_i]).isoformat()
        trades.append({
            "action":     "SELL",
            "time":       ts,
            "price":      round(price, 2),
            "btc_amount": round(btc, 6),
            "gross":      round(gross, 2),
            "fee":        round(fee, 2),
            "net":        round(net, 2),
            "pnl":        round(pnl, 2),
            "pnl_pct":    round(pnl / cost_basis * 100, 3),
            "hold_rows":  hold_rows,
            "reason":     reason,
        })
        exits[reason].append(pnl)
        usdt        = net
        btc         = 0.0
        entry_price = entry_row = cost_basis = 0.0
        entry_time  = None
        return pnl

    for step in range(n):
        abs_i = first_idx + step
        price = float(closes[abs_i])
        score = float(tft_probs[step])
        value_history.append(usdt + btc * price)

        # ── Signal ────────────────────────────────────────────────────────────
        if score > BUY_THRESHOLD:
            signal = "BUY"
            conf   = "HIGH" if score > 0.75 else "MEDIUM"
        elif score < SELL_THRESHOLD:
            signal = "SELL"
            conf   = "HIGH" if score < 0.25 else "MEDIUM"
        else:
            signal = "HOLD"
            conf   = "LOW"
            hold_sig += 1

        # ── Exit ──────────────────────────────────────────────────────────────
        if btc > 0:
            pnl_frac  = (price - entry_price) / entry_price
            hold_rows = abs_i - entry_row
            reason    = None

            if pnl_frac <= -STOP_LOSS_PCT:
                reason = "STOP_LOSS"
            elif pnl_frac >= TAKE_PROFIT_PCT:
                reason = "TAKE_PROFIT"
            elif hold_rows >= MAX_HOLD_ROWS:
                reason = "MAX_HOLD"
            elif signal == "SELL" and conf in ("HIGH", "MEDIUM") and hold_rows >= MIN_HOLD_ROWS:
                reason = "SIGNAL"

            if reason:
                _do_sell(abs_i, price, reason)

        # ── Entry ─────────────────────────────────────────────────────────────
        elif btc == 0 and signal == "BUY" and conf in ("HIGH", "MEDIUM"):
            range_pct = chop_arr[abs_i]
            if np.isnan(range_pct) or range_pct < CHOP_FILTER_PCT:
                pass  # choppy — skip
            else:
                hold_buy  += 1
                capital    = usdt if conf == "HIGH" else usdt * 0.5
                fee        = capital * FEE_RATE
                btc_rcvd   = (capital - fee) / price
                usdt      -= capital
                btc        = btc_rcvd
                entry_price = price
                entry_row   = abs_i
                cost_basis  = capital
                entry_time  = pd.Timestamp(times_arr[abs_i])
                ts = pd.Timestamp(times_arr[abs_i]).isoformat()
                trades.append({
                    "action":    "BUY",
                    "time":      ts,
                    "price":     round(price, 2),
                    "btc_amount": round(btc, 6),
                    "capital":   round(capital, 2),
                    "fee":       round(fee, 2),
                    "confidence": conf,
                    "score":     round(score, 4),
                })

    # Close any open position at end
    if btc > 0:
        price = float(closes[first_idx + n - 1])
        _do_sell(first_idx + n - 1, price, "END_OF_BACKTEST")

    return {
        "trades":        trades,
        "value_history": value_history,
        "exits":         exits,
        "final_usdt":    usdt,
        "win_trades":    win_trades,
        "loss_trades":   loss_trades,
        "hold_sig":      hold_sig,
        "hold_buy":      hold_buy,
    }


# ── Stats & display ──────────────────────────────────────────────────────────

def _compute_stats(result, capital, df, first_idx, start_dt, end_dt):
    trades = result["trades"]
    exits  = result["exits"]

    sell_trades = [t for t in trades if t["action"] == "SELL" and t.get("reason") != "END_OF_BACKTEST"]
    total       = len(sell_trades)
    wins        = result["win_trades"]
    losses      = result["loss_trades"]
    final_val   = result["final_usdt"]
    total_pnl   = final_val - capital

    value_hist = np.array(result["value_history"])
    peak       = np.maximum.accumulate(value_hist)
    drawdown   = (value_hist - peak) / peak
    max_dd     = float(drawdown.min()) if len(drawdown) else 0.0

    returns    = np.diff(value_hist) / value_hist[:-1] if len(value_hist) > 1 else np.array([0.0])
    sharpe     = (returns.mean() / returns.std() * np.sqrt(525_600)) if returns.std() > 0 else 0.0

    # Buy-and-hold comparison
    start_price = float(df["close"].iloc[first_idx])
    end_price   = float(df["close"].iloc[-1])
    bh_pct      = (end_price / start_price - 1) * 100

    # Signal profitability
    buy_sigs   = [t for t in trades if t["action"] == "BUY"]
    n_buy_sigs = len(buy_sigs)
    buy_wins   = wins
    sell_sigs  = sum(1 for r in exits.get("SIGNAL", []) if r >= 0)
    n_sell_sigs = len(exits.get("SIGNAL", []))

    # By month
    monthly = defaultdict(float)
    monthly_cnt = defaultdict(int)
    for t in sell_trades:
        m = t["time"][:7]
        monthly[m] += t["pnl"]
        monthly_cnt[m] += 1

    hold_times = [t["hold_rows"] for t in sell_trades if "hold_rows" in t]
    avg_hold   = np.mean(hold_times) if hold_times else 0

    pnls = [t["pnl"] for t in sell_trades]
    best  = max(pnls) if pnls else 0
    worst = min(pnls) if pnls else 0
    avg   = np.mean(pnls) if pnls else 0

    best_t  = next((t for t in sell_trades if t.get("pnl") == best),  {})
    worst_t = next((t for t in sell_trades if t.get("pnl") == worst), {})

    return {
        "final_val": final_val, "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / capital * 100,
        "max_drawdown": max_dd * 100, "sharpe": sharpe,
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate": wins / total * 100 if total else 0,
        "best": best, "worst": worst, "avg": avg,
        "best_time":  best_t.get("time", "")[:16],
        "worst_time": worst_t.get("time", "")[:16],
        "avg_hold": avg_hold,
        "bh_pct": bh_pct, "bh_start": start_price, "bh_end": end_price,
        "n_buy_sigs": n_buy_sigs, "buy_wins": buy_wins,
        "n_sell_sigs": n_sell_sigs, "sell_wins": sell_sigs,
        "hold_sig": result["hold_sig"],
        "chop_filtered": 0,  # not tracked separately
        "monthly": dict(monthly), "monthly_cnt": dict(monthly_cnt),
        "exit_reasons": {k: len(v) for k, v in result["exits"].items()},
    }


def _print_results(s, start_dt, end_dt):
    dur_days  = (end_dt - start_dt).days
    dur_months = dur_days / 30.44

    print()
    print(SEP)
    print("BACKTEST RESULTS")
    print(SEP)
    print(f"Period:           {start_dt.date()} -> {end_dt.date()}")
    print(f"Duration:         {dur_months:.0f} months  ({dur_days} days)")
    print(f"Starting capital: ${1000.00:,.2f}")
    print()
    print("PERFORMANCE")
    pnl_sign = "+" if s["total_pnl"] >= 0 else ""
    print(f"Final value:      ${s['final_val']:,.2f}")
    print(f"Total PnL:        {pnl_sign}${s['total_pnl']:.2f}  ({pnl_sign}{s['total_pnl_pct']:.2f}%)")
    print(f"vs Buy and Hold:  BTC went {'+' if s['bh_pct']>=0 else ''}{s['bh_pct']:.2f}% same period")
    print(f"Max drawdown:     {s['max_drawdown']:.2f}%")
    print(f"Sharpe ratio:     {s['sharpe']:.2f}")
    print()
    print("TRADES")
    print(f"Total trades:     {s['total_trades']}")
    print(f"Winning:          {s['wins']}  ({s['win_rate']:.1f}%)")
    print(f"Losing:           {s['losses']}  ({100-s['win_rate']:.1f}%)")
    print(f"Best trade:       +${s['best']:.2f}  ({s['best_time']})")
    print(f"Worst trade:      -${abs(s['worst']):.2f}  ({s['worst_time']})")
    print(f"Avg hold time:    {s['avg_hold']:.0f} minutes")
    print(f"Avg profit/trade: ${s['avg']:.2f}")
    print()
    print("EXIT REASONS")
    for reason, count in s["exit_reasons"].items():
        print(f"  {reason:<20} {count}")
    print()
    if s["monthly"]:
        print("BY MONTH")
        for m in sorted(s["monthly"]):
            pnl = s["monthly"][m]
            cnt = s["monthly_cnt"][m]
            sign = "+" if pnl >= 0 else ""
            print(f"  {m}: ${pnl:>8.2f}  ({sign}{pnl/1000*100:.1f}%)  {cnt} trades")
    print()
    print("SIGNAL BREAKDOWN")
    buy_pct = s['buy_wins'] / s['n_buy_sigs'] * 100 if s['n_buy_sigs'] else 0
    print(f"  BUY signals:      {s['n_buy_sigs']:,}  ({buy_pct:.0f}% profitable)")
    print(f"  HOLD:             {s['hold_sig']:,}")
    print()
    verdict = "PROFITABLE" if s["total_pnl"] > 0 else "Unprofitable — needs improvement"
    print("VERDICT")
    pct_str = f"{'+' if s['total_pnl_pct']>=0 else ''}{s['total_pnl_pct']:.1f}%"
    bh_str  = f"BTC +{s['bh_pct']:.1f}%" if s["bh_pct"] >= 0 else f"BTC {s['bh_pct']:.1f}%"
    print(f"  {verdict}  ({pct_str} vs {bh_str})")
    print(SEP)


def _save_result(s, result, start_dt, end_dt):
    os.makedirs(BACKTESTS_DIR, exist_ok=True)
    fname = f"backtest_{datetime.now().strftime('%Y-%m-%d')}.json"
    fpath = os.path.join(BACKTESTS_DIR, fname)
    out = {
        "period":      {"start": str(start_dt.date()), "end": str(end_dt.date())},
        "performance": {k: s[k] for k in
                        ["final_val","total_pnl","total_pnl_pct","max_drawdown","sharpe"]},
        "trades_summary": {k: s[k] for k in
                           ["total_trades","wins","losses","win_rate","avg_hold"]},
        "exit_reasons": s["exit_reasons"],
        "monthly": s["monthly"],
        "trades":  result["trades"],
    }
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return fpath


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest Adaptive TFT on historical BTC data")
    parser.add_argument("--start",   type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help=f"Starting USDT (default: {DEFAULT_CAPITAL})")
    args = parser.parse_args()

    start_dt = pd.Timestamp(args.start, tz="UTC")
    end_dt   = pd.Timestamp.now(tz="UTC")

    print(f"[Backtest] {start_dt.date()} -> {end_dt.date()}", flush=True)
    print("[Backtest] Loading btc_merged_features.csv...", flush=True)

    df = pd.read_csv(MERGED_PATH, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    df = (df.sort_values("time")
            .reset_index(drop=True)
            .pipe(lambda d: d[d["time"] <= pd.Timestamp.now(tz="UTC")])
            .reset_index(drop=True))
    df = df.dropna(subset=ACTIVE_FEATURES).reset_index(drop=True)

    mask      = df["time"] >= start_dt
    if not mask.any():
        print(f"ERROR: No data from {args.start}")
        sys.exit(1)

    candidate = int(np.where(mask.values)[0][0])
    first_idx = max(candidate, SEQ_TFT + 1)
    n         = len(df) - first_idx
    print(f"[Backtest] {n:,} rows to process", flush=True)

    tft_m, tft_s, bilstm_m, bilstm_s, meta_m, meta_w = _load_models()

    X_all     = df[ACTIVE_FEATURES].values.astype(np.float32)
    times_all = df["time"].values

    final_probs = _batch_infer(X_all, times_all, first_idx,
                                tft_m, tft_s, bilstm_m, bilstm_s, meta_m, meta_w, df)

    print("[Backtest] Simulating trades...", flush=True)
    result = _simulate(df, first_idx, final_probs, args.capital)
    print("[Backtest] Processing... 100% complete", flush=True)

    stats = _compute_stats(result, args.capital, df, first_idx, start_dt, end_dt)
    _print_results(stats, start_dt, end_dt)
    fpath = _save_result(stats, result, start_dt, end_dt)
    print(f"\nFull trade log saved to: {fpath}")


if __name__ == "__main__":
    main()
