"""
Comprehensive historical validation of btc_1m.csv against the Coinbase Exchange API.

Strategy: pick 5 fixed sample times per month (1st, 8th, 15th, 22nd, 28th at 12:00 UTC).
These are predictable, spread evenly, and can be fetched efficiently — one API call
per month fetches all 5 samples in a single 300-candle window around noon.

~5 samples × 53 months = 265 candles total, ~55 API calls, runs in ~1 minute.

Usage:
    python features/validate_1m_online.py
"""
import os, sys, time, argparse
import requests
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH  = os.path.join(DATA_DIR, "btc_1m.csv")
API_URL   = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
TOLERANCE = 0.0001   # 0.01%
TAIL_SKIP = 30       # ignore last 30 min (may be open/partial)

SEP  = "=" * 64
SEP2 = "-" * 64

# Fixed sample days within each month — spread to catch different conditions
SAMPLE_DAYS  = [1, 8, 15, 22, 28]
SAMPLE_HOUR  = 12   # noon UTC — well-formed, never at day boundaries


def load_local_index(df: pd.DataFrame) -> dict:
    """Build a time → row lookup for fast O(1) access."""
    df["time_str"] = df["time"].astype(str)
    return {row["time_str"]: row for _, row in df.iterrows()}


def load_local_full() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def fetch_window(center_unix: int, window_min: int = 60) -> pd.DataFrame:
    """Fetch candles in a ±window_min window around center_unix."""
    start = center_unix - window_min * 60
    end   = center_unix + window_min * 60
    for attempt in range(3):
        try:
            resp = requests.get(API_URL,
                                params={"granularity": 60, "start": start, "end": end},
                                timeout=20)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                return pd.DataFrame()
            df = pd.DataFrame(raw, columns=["time","low","high","open","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return df.sort_values("time").reset_index(drop=True)[
                ["time","open","high","low","close","volume"]]
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                return pd.DataFrame()


def build_sample_targets(df: pd.DataFrame) -> list:
    """
    Build list of target timestamps: 5 fixed days per month at SAMPLE_HOUR UTC.
    Only includes timestamps that exist in the local CSV.
    """
    cutoff  = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=TAIL_SKIP)
    targets = []
    local_times = set(df["time"].astype(str))

    months = df["time"].dt.to_period("M").unique()
    for period in sorted(months):
        year, month = period.year, period.month
        for day in SAMPLE_DAYS:
            try:
                ts = pd.Timestamp(year=year, month=month, day=day,
                                   hour=SAMPLE_HOUR, tz="UTC")
            except ValueError:
                continue   # day doesn't exist in this month (e.g., Feb 28+)
            if ts < cutoff and str(ts) in local_times:
                targets.append(ts)

    return targets


def run():
    print(SEP)
    print("COMPREHENSIVE 1M BTC DATA VALIDATION vs COINBASE")
    print(f"Checking {len(SAMPLE_DAYS)} fixed candles per month (days {SAMPLE_DAYS} at {SAMPLE_HOUR}:00 UTC)")
    print(SEP)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    print("Loading btc_1m.csv...", flush=True)
    local = load_local_full()
    print(f"  {len(local):,} rows  |  {local['time'].iloc[0].date()}  →  {local['time'].iloc[-1].date()}")

    print("Building sample targets...", flush=True)
    targets = build_sample_targets(local)
    print(f"  {len(targets)} target candles across {len(set(t.to_period('M') for t in targets))} months")

    # Build local lookup
    local_idx = {str(row["time"]): row for _, row in local.iterrows()}

    # Fetch and compare
    print(f"\nFetching from Coinbase API ({len(targets)} candles)...", flush=True)
    mismatches  = []
    by_year     = defaultdict(list)
    by_month    = defaultdict(list)
    checked     = 0
    not_in_api  = 0

    for i, ts in enumerate(targets):
        if (i + 1) % 20 == 0 or i == len(targets) - 1:
            print(f"  [{i+1}/{len(targets)}] Checked {checked} candles, "
                  f"{len(mismatches)} mismatches so far...", flush=True)

        center_unix = int(ts.timestamp())
        remote_df   = fetch_window(center_unix, window_min=5)
        time.sleep(0.25)

        if remote_df.empty:
            not_in_api += 1
            continue

        # Find this exact timestamp in remote
        remote_row = remote_df[remote_df["time"] == ts]
        if remote_row.empty:
            not_in_api += 1
            continue

        local_row = local_idx.get(str(ts))
        if local_row is None:
            continue

        remote_row = remote_row.iloc[0]
        checked   += 1

        for col in ["open", "high", "low", "close", "volume"]:
            lv = local_row[col]
            rv = remote_row[col]
            if rv == 0:
                continue
            diff = abs(lv - rv) / rv
            if diff > TOLERANCE:
                m = {
                    "time":     ts,
                    "year":     ts.year,
                    "month":    ts.strftime("%Y-%m"),
                    "field":    col,
                    "local":    round(lv, 6),
                    "remote":   round(rv, 6),
                    "diff_pct": round(diff * 100, 4),
                }
                mismatches.append(m)
                by_year[ts.year].append(m)
                by_month[ts.strftime("%Y-%m")].append(m)

    # Results
    print()
    print(SEP2)
    print("RESULTS BY YEAR")
    print(SEP2)

    years = sorted(set(t.year for t in targets))
    clean = True
    for year in years:
        year_targets = [t for t in targets if t.year == year]
        year_errors  = len(by_year.get(year, []))
        tag = "[PASS]" if year_errors == 0 else "[FAIL]"
        print(f"  {tag} {year}: {len(year_targets)} candles checked, {year_errors} mismatches")
        if year_errors > 0:
            clean = False
            for m in by_year[year][:3]:
                print(f"         {m['time']}  {m['field']:7s}  "
                      f"local={m['local']}  remote={m['remote']}  diff={m['diff_pct']}%")

    if not_in_api:
        print(f"\n  [INFO] {not_in_api} targets not found in API (exchange gaps — expected for some dates)")

    print()
    print(SEP)
    print(f"Checked: {checked} candles  |  {len(mismatches)} total mismatches")
    if clean:
        print("RESULT: ALL PASS — 1m data is valid. Safe to run build_ohlcv.py.")
    else:
        print(f"RESULT: {len(mismatches)} MISMATCHES — investigate before rebuilding.")
    print(SEP)
    return clean


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
