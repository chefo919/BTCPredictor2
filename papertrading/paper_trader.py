import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["DNNL_VERBOSE"]          = "0"

import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_manager import update_csv_from_coinbase, get_live_price, PATH_1M
from prediction.predict import generate_signal

SESSIONS_DIR   = os.path.join(os.path.dirname(__file__), "sessions")
LOOP_INTERVAL  = 60
NO_TRADE_SECS  = 30 * 60
STALE_WARN_MIN = 10
STALE_SKIP_MIN = 60
SEP            = "=" * 46

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (STARTING_USDT, FEE_RATE, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
                    MIN_HOLD_MIN, MAX_HOLD_MIN, CHOP_FILTER_PCT)


def _hourly_range_pct() -> float:
    try:
        df   = pd.read_csv(PATH_1M, usecols=["high", "low", "close"])
        if len(df) < 5:
            return 1.0
        tail  = df.tail(60)
        high  = tail["high"].max()
        low   = tail["low"].min()
        close = tail["close"].iloc[-1]
        return (high - low) / close
    except Exception:
        return 1.0


def _fmt_pnl(val: float) -> str:
    return f"+${val:.2f}" if val >= 0 else f"-${abs(val):.2f}"


def _runtime(start: datetime) -> str:
    s    = int((datetime.now(tz=timezone.utc) - start).total_seconds())
    h, r = divmod(s, 3600)
    m, _ = divmod(r, 60)
    return f"{h}h {m:02d}m"


class PaperTrader:
    def __init__(self):
        self.start_time      = datetime.now(tz=timezone.utc)
        self.usdt            = STARTING_USDT
        self.btc             = 0.0
        self.entry_price     = 0.0
        self.entry_time      = None
        self.cost_basis      = 0.0
        self.entry_tft_prob  = 0.0
        self.entry_bi_prob   = 0.0
        self.entry_final     = 0.0
        self.trades          = []
        self.value_history   = [STARTING_USDT]
        self.last_trade_time     = self.start_time
        self.last_no_trade_print = self.start_time
        self.last_live_price     = 0.0

    # ── Portfolio maths ───────────────────────────────────────────────────────

    def _total(self, price: float) -> float:
        return self.usdt + self.btc * price

    def _pnl(self, price: float):
        total   = self._total(price)
        abs_pnl = total - STARTING_USDT
        pct_pnl = abs_pnl / STARTING_USDT * 100
        return abs_pnl, pct_pnl

    def _max_drawdown(self) -> float:
        if len(self.value_history) < 2:
            return 0.0
        v    = np.array(self.value_history)
        peak = np.maximum.accumulate(v)
        dd   = (v - peak) / peak * 100
        return float(np.min(dd))

    def _sharpe(self) -> float:
        closed = [t for t in self.trades if t["action"] == "SELL"]
        if len(closed) < 2:
            return 0.0
        rets = [t["pnl_pct"] for t in closed]
        std  = np.std(rets)
        return round(np.mean(rets) / std, 2) if std > 0 else 0.0

    def _win_stats(self):
        closed = [t for t in self.trades if t["action"] == "SELL"]
        if not closed:
            return 0, 0, 0, 0.0, 0.0, 0.0
        pnls = [t["pnl"] for t in closed]
        wins = sum(1 for p in pnls if p >= 0)
        return wins, len(closed) - wins, len(closed), max(pnls), min(pnls), float(np.mean(pnls))

    def _avg_hold(self) -> float:
        sells = [t for t in self.trades if t["action"] == "SELL"]
        return float(np.mean([t["hold_min"] for t in sells])) if sells else 0.0

    # ── Trade execution ───────────────────────────────────────────────────────

    def _buy(self, price: float, tft_prob: float, bi_prob: float,
             conf: str = "HIGH", final_score: float | None = None) -> dict | None:
        if self.btc > 0 or self.usdt <= 0:
            return None
        capital         = self.usdt if conf == "HIGH" else self.usdt * 0.5
        fee             = capital * FEE_RATE
        btc_received    = (capital - fee) / price
        self.entry_time = datetime.now(tz=timezone.utc)
        self.cost_basis = capital
        self.btc        = btc_received
        self.usdt      -= capital
        self.entry_price    = price
        self.entry_tft_prob = tft_prob
        self.entry_bi_prob  = bi_prob
        self.entry_final    = final_score or tft_prob
        self.last_trade_time = self.entry_time
        t = {
            "action":      "BUY",
            "time":        self.entry_time.strftime("%Y-%m-%d %H:%M UTC"),
            "time_iso":    self.entry_time.isoformat(),
            "price":       price,
            "btc_amount":  round(btc_received, 5),
            "fee":         round(fee, 2),
            "usdt_spent":  round(capital, 2),
            "confidence":  conf,
            "tft_prob":    round(tft_prob, 4),
            "bilstm_prob": round(bi_prob, 4),
            "final_score": round(self.entry_final, 4),
            "pnl":         None,
            "pnl_pct":     None,
            "hold_min":    None,
        }
        self.trades.append(t)
        return t

    def _sell(self, price: float) -> dict | None:
        if self.btc <= 0:
            return None
        gross        = self.btc * price
        fee          = gross * FEE_RATE
        net          = gross - fee
        pnl          = net - self.cost_basis
        pnl_pct      = pnl / self.cost_basis * 100
        hold_min     = (datetime.now(tz=timezone.utc) - self.entry_time).total_seconds() / 60
        sell_time    = datetime.now(tz=timezone.utc)
        btc_sold     = self.btc
        self.usdt   += net
        self.btc     = 0.0
        self.entry_price     = 0.0
        self.last_trade_time = sell_time
        t = {
            "action":        "SELL",
            "time":          sell_time.strftime("%Y-%m-%d %H:%M UTC"),
            "time_iso":      sell_time.isoformat(),
            "price":         price,
            "btc_amount":    round(btc_sold, 5),
            "fee":           round(fee, 2),
            "usdt_received": round(net, 2),
            "pnl":           round(pnl, 2),
            "pnl_pct":       round(pnl_pct, 2),
            "hold_min":      round(hold_min, 1),
        }
        self.trades.append(t)
        return t

    # ── Print helpers ─────────────────────────────────────────────────────────

    def _print_loop(self, sig, action_str: str, live_price: float,
                    csv_result: dict, price_source: str):
        pnl_abs, pnl_pct = self._pnl(live_price)
        wins, losses, closed, best, worst, avg = self._win_stats()
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lag     = csv_result["minutes_behind"]

        print(SEP)
        print(f"BTC Paper Trader -- {now_str}")
        print(SEP)

        if csv_result["ok"]:
            n          = csv_result["new_candles"]
            candle_str = f"+{n} candle{'s' if n != 1 else ''}" if n > 0 else "up to date"
            print(f"CSV:          Coinbase {candle_str}  lag: {lag:.0f} min  OK")
        else:
            print(f"CSV:          Coinbase unavailable  lag: {lag:.0f} min  WARNING")

        note = ""
        if price_source == "Kraken":        note = "  (Coinbase down)"
        elif price_source == "Binance":     note = "  (Coinbase + Kraken down)"
        elif price_source == "CSV fallback":note = "  (all sources down)"
        print(f"Live price:   ${live_price:,.0f} via {price_source}{note}")
        print()

        if lag >= STALE_SKIP_MIN:
            print(f"WARNING  CSV is {lag/60:.1f} hours behind — skipping trade")
            print()
        elif lag >= STALE_WARN_MIN:
            print(f"WARNING  CSV is {lag:.0f}+ minutes behind — stale signal")
            print()

        if sig is not None:
            print(f"Signal:       {sig['signal']}  (confidence: {sig['confidence']})")
            print(f"TFT:          {sig.get('tft_prob', 0):.3f}")
            print(f"BiLSTM:       {sig.get('bilstm_prob', 0):.3f}")
            print(f"Final score:  {sig['final_score']:.3f}")
        print(f"Action:       {action_str}")
        print()

        print("Portfolio:")
        print(f"  USDT:         ${self.usdt:,.2f}")
        print(f"  BTC:          {self.btc:.5f}")
        print(f"  Total value:  ${self._total(live_price):,.2f}")
        print(f"  PnL:          {_fmt_pnl(pnl_abs)}  ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)")
        print()
        print("Session stats:")
        print(f"  Runtime:      {_runtime(self.start_time)}")
        print(f"  Total trades: {closed}")
        if closed > 0:
            print(f"  Wins:         {wins}  ({wins / closed * 100:.1f}%)")
            print(f"  Losses:       {losses}")
            print(f"  Best trade:   {_fmt_pnl(best)}")
            print(f"  Worst trade:  {_fmt_pnl(worst)}")
        print(SEP, flush=True)

    def _save_session(self, end_time, price, pnl_abs, pnl_pct) -> str:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        fname = f"session_{self.start_time.strftime('%Y-%m-%d-%H-%M')}.json"
        fpath = os.path.join(SESSIONS_DIR, fname)
        wins, losses, closed, best, worst, avg = self._win_stats()
        data = {
            "start_time":       self.start_time.isoformat(),
            "end_time":         end_time.isoformat(),
            "starting_capital": STARTING_USDT,
            "final_value":      round(self._total(price), 2),
            "trades":           self.trades,
            "metrics": {
                "total_pnl":        round(pnl_abs, 2),
                "total_pnl_pct":    round(pnl_pct, 2),
                "max_drawdown":     round(self._max_drawdown(), 2),
                "sharpe_ratio":     self._sharpe(),
                "total_trades":     closed,
                "winning_trades":   wins,
                "losing_trades":    losses,
                "best_trade":       round(best, 2) if closed else 0,
                "worst_trade":      round(worst, 2) if closed else 0,
                "avg_trade":        round(avg, 2) if closed else 0,
                "avg_hold_minutes": round(self._avg_hold(), 1),
            },
        }
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return os.path.join("papertrading", "sessions", fname)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print(f"Starting BTC Paper Trader -- ${STARTING_USDT:,.2f} USDT")
        print(f"[Config] Stop loss:    {STOP_LOSS_PCT:.0%}")
        print(f"[Config] Take profit:  {TAKE_PROFIT_PCT:.0%}")
        print(f"[Config] Min hold:     {MIN_HOLD_MIN} min")
        print(f"[Config] Chop filter:  {CHOP_FILTER_PCT:.1%} hourly range minimum")
        print("Press Ctrl+C to stop.\n", flush=True)

        last_sig     = None
        csv_result   = {"ok": True, "new_candles": 0, "minutes_behind": 0.0}
        live_price   = 0.0
        price_source = "unknown"

        try:
            while True:
                loop_start = time.time()

                try:
                    csv_result = update_csv_from_coinbase()
                except Exception as e:
                    csv_result = {"ok": False, "new_candles": 0,
                                  "minutes_behind": csv_result.get("minutes_behind", 0) + 1,
                                  "message": str(e)}

                try:
                    live_price, price_source = get_live_price()
                    self.last_live_price = live_price
                except Exception:
                    live_price   = self.last_live_price or 0.0
                    price_source = "last known"

                if live_price <= 0:
                    time.sleep(LOOP_INTERVAL)
                    continue

                self.value_history.append(self._total(live_price))
                lag = csv_result["minutes_behind"]

                if lag >= STALE_SKIP_MIN:
                    self._print_loop(None, f"SKIPPED (CSV {lag:.0f} min behind)",
                                     live_price, csv_result, price_source)
                    time.sleep(max(0, LOOP_INTERVAL - (time.time() - loop_start)))
                    continue

                try:
                    sig      = generate_signal(fetch=False)
                    last_sig = sig
                except Exception as e:
                    print(f"[Signal] Error: {e}", flush=True)
                    time.sleep(max(0, LOOP_INTERVAL - (time.time() - loop_start)))
                    continue

                signal = sig["signal"]
                conf   = sig["confidence"]
                action_str = "No action (HOLD)"

                if lag >= STALE_WARN_MIN:
                    action_str = f"No action (CSV {lag:.0f} min behind)"

                elif self.btc > 0:
                    pnl_frac = (live_price - self.entry_price) / self.entry_price
                    hold_min = (datetime.now(tz=timezone.utc) - self.entry_time).total_seconds() / 60

                    if pnl_frac <= -STOP_LOSS_PCT:
                        trade = self._sell(live_price)
                        if trade:
                            action_str = f"STOP LOSS: SOLD @ ${live_price:,.0f}  PnL: {_fmt_pnl(trade['pnl'])}"

                    elif pnl_frac >= TAKE_PROFIT_PCT:
                        trade = self._sell(live_price)
                        if trade:
                            action_str = f"TAKE PROFIT: SOLD @ ${live_price:,.0f}  PnL: {_fmt_pnl(trade['pnl'])}"

                    elif hold_min >= MAX_HOLD_MIN:
                        trade = self._sell(live_price)
                        if trade:
                            action_str = f"MAX HOLD: SOLD @ ${live_price:,.0f}  PnL: {_fmt_pnl(trade['pnl'])}"

                    elif signal == "SELL" and conf in ("HIGH", "MEDIUM") and hold_min >= MIN_HOLD_MIN:
                        trade = self._sell(live_price)
                        if trade:
                            action_str = f"SELL signal: SOLD @ ${live_price:,.0f}  PnL: {_fmt_pnl(trade['pnl'])}"

                elif self.btc == 0 and signal == "BUY" and conf in ("HIGH", "MEDIUM"):
                    if _hourly_range_pct() < CHOP_FILTER_PCT:
                        action_str = "No action (market too choppy)"
                    else:
                        trade = self._buy(live_price,
                                          sig.get("tft_prob", 0.5),
                                          sig.get("bilstm_prob", 0.5),
                                          conf,
                                          sig.get("final_score"))
                        if trade:
                            action_str = (f"BOUGHT {trade['btc_amount']:.5f} BTC"
                                          f" @ ${live_price:,.0f}  [{conf}]"
                                          f"  score: {sig['final_score']:.3f}")

                self._print_loop(sig, action_str, live_price, csv_result, price_source)
                time.sleep(max(0, LOOP_INTERVAL - (time.time() - loop_start)))

        except KeyboardInterrupt:
            price = self.last_live_price or (last_sig["price"] if last_sig else 0.0)
            pnl_abs, pnl_pct = self._pnl(price)
            end_time = datetime.now(tz=timezone.utc)
            print(f"\nStopped.  Final value: ${self._total(price):,.2f}  PnL: {_fmt_pnl(pnl_abs)}")
            path = self._save_session(end_time, price, pnl_abs, pnl_pct)
            print(f"Session saved to {path}", flush=True)


if __name__ == "__main__":
    from models import tft_model, bilstm_model, meta_model
    if not tft_model.is_trained() or not bilstm_model.is_trained() or not meta_model.is_trained():
        print("Models not trained. Run: python training/train.py --force")
        sys.exit(1)
    PaperTrader().run()
