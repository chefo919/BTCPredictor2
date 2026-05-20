import sys
import json
import os


def view(path: str):
    with open(path) as f:
        data = json.load(f)

    m      = data["metrics"]
    trades = data.get("trades", [])
    closed = [t for t in trades if t["action"] == "SELL"]
    SEP    = "=" * 42

    start = data["start_time"][:16].replace("T", " ") + " UTC"
    end   = data["end_time"][:16].replace("T", " ") + " UTC"

    from datetime import datetime, timezone
    t0  = datetime.fromisoformat(data["start_time"])
    t1  = datetime.fromisoformat(data["end_time"])
    dur = t1 - t0
    h, r = divmod(int(dur.total_seconds()), 3600)
    mn, _ = divmod(r, 60)

    pnl_abs = m["total_pnl"]
    pnl_pct = m["total_pnl_pct"]

    def fmt(v):
        return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"

    print(SEP)
    print("PAPER TRADING SESSION")
    print(SEP)
    print(f"Started:        {start}")
    print(f"Ended:          {end}")
    print(f"Duration:       {h}h {mn:02d}m")
    print()
    print("PERFORMANCE")
    print(f"Starting capital:  ${data['starting_capital']:,.2f}")
    print(f"Final value:       ${data['final_value']:,.2f}")
    print(f"Total PnL:         {fmt(pnl_abs)}  ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)")
    print(f"Max drawdown:      {m['max_drawdown']:.2f}%")
    print(f"Sharpe ratio:      {m['sharpe_ratio']:.2f}")
    print()
    print("TRADES")
    print(f"Total trades:      {m['total_trades']}")
    if m["total_trades"] > 0:
        w = m["winning_trades"]
        l = m["losing_trades"]
        print(f"Winning trades:    {w}  ({w / m['total_trades'] * 100:.1f}%)")
        print(f"Losing trades:     {l}  ({l / m['total_trades'] * 100:.1f}%)")
        print(f"Best trade:        {fmt(m['best_trade'])}")
        print(f"Worst trade:       {fmt(m['worst_trade'])}")
        print(f"Average trade:     {fmt(m['avg_trade'])}")
        print(f"Average hold time: {m['avg_hold_minutes']:.0f} minutes")
    print()
    if trades:
        print("FULL TRADE LOG")
        for t in trades:
            hhmm = t["time"][11:16]
            if t["action"] == "BUY":
                print(f"  {hhmm}  BUY   {t['btc_amount']:.5f} BTC @ ${t['price']:,.0f}"
                      f"  fee: ${t['fee']:.2f}")
            else:
                pnl_str = fmt(t["pnl"])
                print(f"  {hhmm}  SELL  {t['btc_amount']:.5f} BTC @ ${t['price']:,.0f}"
                      f"  fee: ${t['fee']:.2f}  PnL: {pnl_str}")
    print(SEP)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python papertrading/session_viewer.py <path-to-session.json>")
        sys.exit(1)
    view(sys.argv[1])
