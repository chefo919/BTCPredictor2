"""
Rebuilds all OHLCV CSVs from btc_1m.csv using strict UTC cardinal time boundaries.

Cardinal time rules (all UTC):
  15m  → :00, :15, :30, :45 boundaries
  30m  → :00, :30 boundaries
  1h   → :00 each hour
  4h   → 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
  1d   → 00:00 UTC midnight
  1w   → Monday 00:00 UTC (ISO calendar week)
  1mo  → 1st of each month 00:00 UTC

Aggregation rule (all timeframes):
  open   = first open in period
  high   = max high in period
  low    = min low in period
  close  = last close in period
  volume = sum of volumes
  time   = period START (the boundary label)

Data leakage prevention: period label = start of period. The merge layer in
engineer.py lag-shifts each timeframe forward by one period before joining
to the 1m base, so no row ever sees its own period's data during training.

Usage:
    python features/build_ohlcv.py --dry-run   # validate only, no changes
    python features/build_ohlcv.py             # backup 1m, delete derived, rebuild all
"""
import os, sys, shutil, argparse, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEP  = "=" * 64
SEP2 = "-" * 64

DERIVED_CSVS = [
    "btc_15m.csv", "btc_30m.csv", "btc_1h.csv",
    "btc_4h.csv",  "btc_1d.csv",  "btc_1w.csv", "btc_1mo.csv",
]


# ── IO helpers ────────────────────────────────────────────────────────────────

def _load(fname: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, fname)
    df   = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def _save(df: pd.DataFrame, fname: str):
    df.to_csv(os.path.join(DATA_DIR, fname), index=False)


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate_floor(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate OHLCV using pandas dt.floor() for sub-day frequencies.
    freq: '15min', '30min', '1h', '4h', '1D'
    Period label = floor (start) of the period.
    """
    df = df.copy()
    df["bucket"] = df["time"].dt.floor(freq)
    agg = (
        df.groupby("bucket", sort=True)
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"), close=("close","last"),
             volume=("volume","sum"))
        .rename_axis("time")
        .reset_index()
    )
    return agg[agg["volume"] > 0].reset_index(drop=True)


# Public alias — used by features/engineer.py for proper timeframe resampling
aggregate_floor = _aggregate_floor


def _aggregate_grouper(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate OHLCV using pd.Grouper for calendar-aligned weekly/monthly.
    freq: 'W-MON' (Monday start) or 'MS' (month start)
    Period label = start of the period.
    """
    df = df.set_index("time").sort_index()
    agg = df.groupby(pd.Grouper(freq=freq, closed="left", label="left")).agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"),
        volume=("volume","sum")
    ).reset_index()
    agg = agg.rename(columns={"time": "time"})
    agg["time"] = agg["time"].dt.tz_localize("UTC") if agg["time"].dt.tz is None else agg["time"]
    return agg[agg["volume"] > 0].reset_index(drop=True)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame, label: str, freq_min: int = None) -> bool:
    ok = True

    dupes = df["time"].duplicated().sum()
    if dupes:
        print(f"  [FAIL] {label}: {dupes} duplicate timestamps"); ok = False
    else:
        print(f"  [PASS] {label}: no duplicates")

    bad = (
        (df["high"] < df["low"]).sum() +
        (df["high"] < df["open"]).sum() +
        (df["high"] < df["close"]).sum() +
        (df["low"]  > df["open"]).sum() +
        (df["low"]  > df["close"]).sum() +
        (df["volume"] < 0).sum()
    )
    if bad:
        print(f"  [FAIL] {label}: {bad} OHLCV violations"); ok = False
    else:
        print(f"  [PASS] {label}: OHLCV sanity OK")

    if freq_min:
        diffs = df["time"].diff().dt.total_seconds().dropna() / 60
        gaps  = (diffs > freq_min * 3).sum()
        if gaps:
            print(f"  [WARN] {label}: {gaps} gap(s) > 3× expected interval")
        else:
            print(f"  [PASS] {label}: no unusual gaps")

    return ok


def _cross_check(lower: pd.DataFrame, higher: pd.DataFrame, freq: str,
                  label: str, n: int = 300) -> bool:
    """Re-aggregate n random rows of `lower` and verify they match `higher`."""
    h_idx      = higher.set_index("time")
    lower_cp   = lower.copy()
    lower_cp["bucket"] = lower_cp["time"].dt.floor(freq)
    buckets    = random.sample(list(higher["time"]), min(n, len(higher)))
    errors     = 0

    for b in buckets:
        rows = lower_cp[lower_cp["bucket"] == b]
        if rows.empty or b not in h_idx.index:
            continue
        saved = h_idx.loc[b]
        tol   = 0.01
        if abs(rows["open"].iloc[0]  - saved["open"])  > tol: errors += 1
        if abs(rows["high"].max()    - saved["high"])   > tol: errors += 1
        if abs(rows["low"].min()     - saved["low"])    > tol: errors += 1
        if abs(rows["close"].iloc[-1]- saved["close"])  > tol: errors += 1

    if errors == 0:
        print(f"  [PASS] {label}: cross-check OK ({len(buckets)} candles sampled)")
        return True
    else:
        print(f"  [FAIL] {label}: {errors} cross-check mismatches in {len(buckets)} candles")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> bool:
    print(SEP)
    print("OHLCV REBUILD — CARDINAL TIME ALIGNMENT")
    if dry_run:
        print("DRY RUN — no files will be written or deleted")
    print(SEP)

    m1_path = os.path.join(DATA_DIR, "btc_1m.csv")
    if not os.path.exists(m1_path):
        print(f"ERROR: btc_1m.csv not found in {DATA_DIR}")
        sys.exit(1)

    print("Loading btc_1m.csv...", flush=True)
    m1 = _load("btc_1m.csv")
    print(f"  {len(m1):,} rows  |  {m1['time'].iloc[0]}  →  {m1['time'].iloc[-1]}")
    print()

    # Validate 1m source
    print("Validating 1m source data...")
    if not _validate(m1, "1m", freq_min=1):
        print("\nERROR: 1m data invalid. Run validate_1m_online.py first.")
        sys.exit(1)
    print()

    if not dry_run:
        # Back up btc_1m.csv
        backup = os.path.join(DATA_DIR, "btc_1m_backup.csv")
        shutil.copy(m1_path, backup)
        print(f"Backup created: btc_1m_backup.csv")

        # Delete all derived CSVs
        print("Deleting derived CSVs...")
        for fname in DERIVED_CSVS:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Deleted: {fname}")
        print()

    all_ok = True

    # ── Sub-day aggregations using floor() ────────────────────────────────────
    prev_df    = m1
    prev_label = "1m"
    chain = [
        ("btc_15m.csv", "15min", 15),
        ("btc_30m.csv", "30min", 30),
        ("btc_1h.csv",  "1h",    60),
        ("btc_4h.csv",  "4h",    240),
        ("btc_1d.csv",  "1D",    1440),
    ]
    built = {}

    for fname, freq, freq_min in chain:
        label = fname.replace("btc_","").replace(".csv","").upper()
        print(SEP2)
        print(f"Building {fname} from {prev_label} (boundary: {freq} UTC floor)...")
        agg = _aggregate_floor(prev_df, freq)
        print(f"  → {len(agg):,} candles  |  {agg['time'].iloc[0]}  →  {agg['time'].iloc[-1]}")

        ok = _validate(agg, label, freq_min)
        # Cross-check against 1m for sub-day
        if freq_min <= 1440 and freq != "1D":
            ok &= _cross_check(m1, agg, freq, label)
        elif freq == "1D":
            ok &= _cross_check(prev_df, agg, freq, label)

        all_ok &= ok
        built[label.lower().replace("m","")] = agg

        if not dry_run and ok:
            _save(agg, fname)
            print(f"  Saved: {fname}")

        prev_df    = agg
        prev_label = label
        print()

    df_1d = built.get("1d", _aggregate_floor(m1, "1D"))

    # ── Rolling weekly — last 7 daily candles (inclusive) ─────────────────────
    print(SEP2)
    print("Building btc_1w.csv (rolling 7-day OHLCV from daily, no data leakage)...")
    w1 = _rolling_ohlcv(df_1d, window=7)
    print(f"  → {len(w1):,} rows  |  {w1['time'].iloc[0].date()}  →  {w1['time'].iloc[-1].date()}")
    print(f"  Rule: at daily row T, open=T-6 open, high=max(T-6..T), low=min(T-6..T), close=T close")
    ok = _validate(w1, "1W")
    all_ok &= ok
    if not dry_run and ok:
        _save(w1, "btc_1w.csv")
        print(f"  Saved: btc_1w.csv")
    print()

    # ── Rolling monthly — last 30 daily candles (inclusive) ───────────────────
    print(SEP2)
    print("Building btc_1mo.csv (rolling 30-day OHLCV from daily, no data leakage)...")
    mo1 = _rolling_ohlcv(df_1d, window=30)
    print(f"  → {len(mo1):,} rows  |  {mo1['time'].iloc[0].date()}  →  {mo1['time'].iloc[-1].date()}")
    print(f"  Rule: at daily row T, open=T-29 open, high=max(T-29..T), low=min(T-29..T), close=T close")
    ok = _validate(mo1, "1MO")
    all_ok &= ok
    if not dry_run and ok:
        _save(mo1, "btc_1mo.csv")
        print(f"  Saved: btc_1mo.csv")
    print()

    print(SEP)
    if all_ok:
        print("ALL STEPS PASSED")
        if dry_run:
            print("Dry run complete — no files written. Run without --dry-run to build.")
        else:
            print("All CSVs rebuilt. Next: python features/engineer.py")
    else:
        print("SOME STEPS FAILED — review output above before continuing.")
    print(SEP)

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only — no deletes or writes")
    args = parser.parse_args()
    ok   = run(dry_run=args.dry_run)
    sys.exit(0 if ok else 1)
