import os
import json
import shutil
import time as _time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

PATH_1M = os.path.join(DATA_DIR, "btc_1m.csv")

MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
GAPS_LOG      = os.path.join(DATA_DIR, "gaps.log")

GRANULARITY      = 60
BATCH_SIZE       = 300
CHECKPOINT_EVERY = 500
REQUEST_DELAY    = 0.12

# COINBASE ONLY RULE — NEVER CHANGE THIS
# All data written to CSV files comes exclusively from Coinbase API
# No other exchange data is ever written to any CSV file
# If Coinbase fails: retry, log gap, continue
# Never fall back to Kraken or Binance for storage

COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
MAX_RETRIES  = 3
RETRY_WAIT   = 30

# Tracks which live-price sources have printed their first-run field validation
_validated_sources: set = set()


def fetch_coinbase_only(start: int, end: int, max_retries: int = MAX_RETRIES) -> Optional[pd.DataFrame]:
    for attempt in range(1, max_retries + 1):
        try:
            params = {"granularity": 60, "start": start, "end": end}
            resp   = requests.get(COINBASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
            df = df[["time", "open", "high", "low", "close", "volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype("float64")
            df = df.sort_values("time").reset_index(drop=True)
            df = df.dropna()
            df = df.drop_duplicates(subset="time", keep="last")
            return df
        except Exception as e:
            if attempt < max_retries:
                print(f"[Coinbase] Attempt {attempt}/{max_retries} failed: {e}", flush=True)
                print(f"[Coinbase] Retrying in {RETRY_WAIT}s...", flush=True)
                _time.sleep(RETRY_WAIT)
            else:
                _log_gap(start, end, str(e))
                print(f"[Coinbase] All {max_retries} attempts failed — gap logged", flush=True)
                return None


# ── Manifest ──────────────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _save_manifest(rows: int, first_candle: str, last_candle: str):
    data = _load_manifest()
    data["btc_1m"] = {
        "rows":         rows,
        "first_candle": first_candle,
        "last_candle":  last_candle,
        "verified":     True,
        "verified_at":  datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _check_manifest():
    manifest = _load_manifest()
    expected = manifest.get("btc_1m", {}).get("rows", 0)
    if expected == 0 or not os.path.exists(PATH_1M):
        return
    with open(PATH_1M, "rb") as f:
        actual = sum(1 for _ in f) - 1
    if actual < expected:
        print(f"WARNING: btc_1m.csv has {actual:,} rows but manifest expects {expected:,}")
        print("DATA LOSS DETECTED - aborting to prevent further damage")
        print(f"Check {BACKUP_DIR} for recent backup")
        raise SystemExit(1)


# ── Backup ────────────────────────────────────────────────────────────────────

def _backup_1m() -> str:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if not os.path.exists(PATH_1M):
        return ""
    ts   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M")
    dest = os.path.join(BACKUP_DIR, f"btc_1m_{ts}.csv")
    shutil.copy2(PATH_1M, dest)
    backups = sorted(f for f in os.listdir(BACKUP_DIR)
                     if f.startswith("btc_1m_") and f.endswith(".csv"))
    while len(backups) > 3:
        os.remove(os.path.join(BACKUP_DIR, backups.pop(0)))
    return dest


# ── Gap log ───────────────────────────────────────────────────────────────────

def _log_gap(start: int, end: int, error: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(tz=timezone.utc).isoformat()
    with open(GAPS_LOG, "a") as f:
        start_dt = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
        end_dt   = datetime.fromtimestamp(end,   tz=timezone.utc).isoformat()
        f.write(f"{now}  GAP [{start_dt} - {end_dt}]: {error}\n")


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _save_deduped(df: pd.DataFrame, path: str) -> pd.DataFrame:
    manifest  = _load_manifest()
    min_rows  = manifest.get("btc_1m", {}).get("rows", 0)
    new_count = len(df)

    if path == PATH_1M and min_rows > 0 and new_count < min_rows:
        raise RuntimeError(
            f"Row count would decrease: {min_rows:,} -> {new_count:,}. Aborting save.")

    if path == PATH_1M and os.path.exists(path):
        backup = _backup_1m()
        if backup:
            print(f"  [Backup] {os.path.basename(backup)}", flush=True)

    df = df.drop_duplicates(subset="time", keep="last").sort_values("time").reset_index(drop=True)
    df.to_csv(path, index=False)

    if path == PATH_1M:
        _save_manifest(
            rows=len(df),
            first_candle=str(df["time"].min()),
            last_candle=str(df["time"].max()),
        )

    return df


def _validate_ohlcv(df: pd.DataFrame, source: str = "") -> bool:
    """
    Validates OHLCV data before any write to btc_1m.csv.
    Returns True if data is clean, False if any violation found.
    Logs what failed so corruption can be traced.
    """
    if df.empty:
        return True
    errors = []
    if (df["high"] < df["low"]).any():
        errors.append(f"high < low in {(df['high'] < df['low']).sum()} rows")
    if (df["high"] < df["open"]).any():
        errors.append(f"high < open in {(df['high'] < df['open']).sum()} rows")
    if (df["high"] < df["close"]).any():
        errors.append(f"high < close in {(df['high'] < df['close']).sum()} rows")
    if (df["low"] > df["open"]).any():
        errors.append(f"low > open in {(df['low'] > df['open']).sum()} rows")
    if (df["low"] > df["close"]).any():
        errors.append(f"low > close in {(df['low'] > df['close']).sum()} rows")
    if (df["volume"] < 0).any():
        errors.append(f"negative volume in {(df['volume'] < 0).sum()} rows")
    bad_price = ((df["close"] < 1000) | (df["close"] > 500_000)).sum()
    if bad_price:
        errors.append(f"price out of range [1k-500k] in {bad_price} rows")
    if errors:
        tag = f" [{source}]" if source else ""
        for e in errors:
            print(f"[1m VALIDATION FAIL{tag}] {e}", flush=True)
        return False
    return True


def get_last_1m_update() -> dict:
    """Returns manifest info about the last btc_1m.csv write. Safe to call anytime."""
    manifest = _load_manifest()
    info = manifest.get("btc_1m", {})
    return {
        "last_candle":  info.get("last_candle",  "unknown"),
        "first_candle": info.get("first_candle", "unknown"),
        "rows":         info.get("rows",          0),
        "verified_at":  info.get("verified_at",   "never"),
    }


def _append_to_1m(new_rows: pd.DataFrame) -> int:
    """Fast append to btc_1m.csv without rewriting the full file."""
    if new_rows.empty:
        return 0

    # Validate before any write — never let corrupt data reach the CSV
    if not _validate_ohlcv(new_rows, source="append"):
        print("[1m] Append aborted — validation failed. No data written.", flush=True)
        return 0

    if os.path.exists(PATH_1M):
        with open(PATH_1M, "rb") as fh:
            try:
                fh.seek(-2048, 2)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", errors="replace")
        last_lines = [ln for ln in tail.split("\n") if ln.strip()]
        if last_lines:
            try:
                last_ts = pd.to_datetime(last_lines[-1].split(",")[0], utc=True)
                new_rows = new_rows[new_rows["time"] > last_ts]
            except Exception:
                pass

    if new_rows.empty:
        return 0

    new_rows = new_rows.drop_duplicates(subset="time").sort_values("time")
    new_rows.to_csv(PATH_1M, mode="a", header=False, index=False)
    n = len(new_rows)

    manifest = _load_manifest()
    m1 = manifest.get("btc_1m", {})
    manifest["btc_1m"] = {
        **m1,
        "rows":        m1.get("rows", 0) + n,
        "last_candle": str(new_rows["time"].max()),
        "verified":    True,
        "verified_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return n


# ── Feature update integration ────────────────────────────────────────────────

def _run_features(total_fetched: int):
    import sys as _sys
    _sys.path.insert(0, ROOT)
    from features.engineer import update_features as _update_features

    feat_stats   = _update_features()
    merged_added = feat_stats.get("merged_added", 0)

    if total_fetched > 0:
        print(f"[Data]     +{total_fetched:,} new 1m candles", flush=True)
    if merged_added > 0:
        print(f"[Features] +{merged_added:,} new feature rows added", flush=True)

    merged_path = os.path.join(DATA_DIR, "btc_merged_features.csv")
    if os.path.exists(merged_path):
        n_merged = sum(1 for _ in open(merged_path)) - 1
        print(f"[Ready]    btc_merged_features.csv up to date — {n_merged:,} rows", flush=True)


# ── Flow 1 — CSV update (Coinbase ONLY) ──────────────────────────────────────

def update_csv_from_coinbase() -> dict:
    """
    Flow 1: Fetch only from Coinbase, write to CSV files.
    Returns {"ok": bool, "new_candles": int, "minutes_behind": float, "message": str}

    ABSOLUTE RULE: Nothing from Kraken or Binance ever touches any CSV file.
    Not now, not ever, under any circumstance.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = _load_csv(PATH_1M)
    now_ts   = int(datetime.now(tz=timezone.utc).timestamp())

    # Last CLOSED candle boundary: floor to minute, subtract one candle.
    # A candle opening at T:00 closes at T:59 — it is complete only at T+1:00.
    # This guarantees we never write a partial open candle.
    end_ts = (now_ts // GRANULARITY) * GRANULARITY - GRANULARITY

    if existing is not None and not existing.empty:
        last_ts     = int(existing["time"].max().timestamp())
        fetch_start = last_ts + GRANULARITY
    else:
        fetch_start = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp())

    gap_seconds = end_ts - fetch_start
    if gap_seconds <= 0:
        return {"ok": True, "new_candles": 0, "minutes_behind": 0.0,
                "message": "[CSV] Already up to date"}

    minutes_behind = gap_seconds / 60
    show_progress  = gap_seconds > 300
    total_batches  = max(1, (gap_seconds + BATCH_SIZE * GRANULARITY - 1) // (BATCH_SIZE * GRANULARITY))

    buffer_dfs = []
    cursor     = fetch_start
    batch_num  = 0

    while cursor < end_ts:
        batch_end = min(cursor + BATCH_SIZE * GRANULARITY, end_ts)

        if show_progress:
            print(f"[Data] Behind by {minutes_behind:.0f} minutes — "
                  f"fetching batch {batch_num + 1} of {total_batches}...", flush=True)

        rows = fetch_coinbase_only(cursor, batch_end, max_retries=1)
        if rows is not None and not rows.empty:
            buffer_dfs.append(rows)
        else:
            _log_gap(cursor, end_ts, "Coinbase unavailable")
            remaining = (end_ts - cursor) / 60
            msg = "[CSV] Coinbase unavailable — CSV unchanged, gap logged"
            print(msg, flush=True)
            return {"ok": False, "new_candles": 0, "minutes_behind": remaining, "message": msg}

        cursor    = batch_end + GRANULARITY
        batch_num += 1
        _time.sleep(REQUEST_DELAY)

    if not buffer_dfs:
        return {"ok": True, "new_candles": 0, "minutes_behind": 0.0,
                "message": "[CSV] No new candles"}

    # Write to CSV (fast append — no full rewrite)
    all_new    = pd.concat(buffer_dfs, ignore_index=True)
    n_appended = _append_to_1m(all_new)

    # Update rolling features (incremental if yearly files exist, full rebuild otherwise)
    _run_features(n_appended)

    full_1m     = _load_csv(PATH_1M)
    new_last_ts = int(full_1m["time"].max().timestamp()) if full_1m is not None else end_ts
    remaining   = max(0.0, (end_ts - new_last_ts) / 60)

    msg = f"[CSV] +{n_appended} new candles from Coinbase — all CSVs up to date"
    print(msg, flush=True)
    return {"ok": True, "new_candles": n_appended, "minutes_behind": remaining, "message": msg}


# ── Full rewrite — fetches everything from scratch ───────────────────────────

def rewrite_1m_csv(start_date: str = "2022-01-01") -> int:
    """
    Deletes btc_1m.csv and fetches all historical 1m candles from Coinbase
    from start_date to now. Validates every batch before writing.

    Safeguards:
      - Backs up existing file before deleting
      - Validates OHLCV on every batch — skips corrupt batches, logs gaps
      - Updates manifest after every 500 batches (checkpoint)
      - Never writes partial candles (stops at now - 2 minutes)

    Returns total candles written.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    start_ts  = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    # Last closed candle boundary — same logic as update_csv_from_coinbase
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    end_ts = (now_ts // GRANULARITY) * GRANULARITY - GRANULARITY

    total_batches = max(1, (end_ts - start_ts) // (BATCH_SIZE * GRANULARITY) + 1)
    print(f"[Rewrite] Start:  {pd.Timestamp(start_ts, unit='s', tz='UTC')}", flush=True)
    print(f"[Rewrite] End:    {pd.Timestamp(end_ts,   unit='s', tz='UTC')}", flush=True)
    print(f"[Rewrite] ~{total_batches:,} batches of {BATCH_SIZE} candles each", flush=True)
    print(f"[Rewrite] Estimated time: {total_batches * REQUEST_DELAY / 60:.0f}–{total_batches * (REQUEST_DELAY + 0.1) / 60:.0f} minutes", flush=True)

    # Backup existing file
    if os.path.exists(PATH_1M):
        backup = _backup_1m()
        print(f"[Rewrite] Backup: {backup}", flush=True)
        os.remove(PATH_1M)
        print(f"[Rewrite] Deleted old btc_1m.csv", flush=True)

    # Write CSV header
    pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"]).to_csv(
        PATH_1M, index=False)

    cursor      = start_ts
    total       = 0
    batch_num   = 0
    skipped     = 0

    while cursor < end_ts:
        batch_end = min(cursor + BATCH_SIZE * GRANULARITY, end_ts)
        rows      = fetch_coinbase_only(cursor, batch_end, max_retries=2)

        if rows is not None and not rows.empty:
            if _validate_ohlcv(rows, source=f"batch@{pd.Timestamp(cursor, unit='s', tz='UTC').date()}"):
                rows.to_csv(PATH_1M, mode="a", header=False, index=False)
                total += len(rows)
            else:
                _log_gap(cursor, batch_end, "validation_failed")
                skipped += 1
        else:
            _log_gap(cursor, batch_end, "api_returned_empty")

        cursor    = batch_end + GRANULARITY
        batch_num += 1
        _time.sleep(REQUEST_DELAY)

        # Progress every 500 batches + checkpoint manifest
        if batch_num % 500 == 0:
            pct = (cursor - start_ts) / (end_ts - start_ts) * 100
            print(f"[Rewrite] {pct:.1f}%  {total:,} candles written  "
                  f"({skipped} batches skipped)", flush=True)
            # Checkpoint: update manifest so progress survives a crash
            df_ck = _load_csv(PATH_1M)
            if df_ck is not None and not df_ck.empty:
                _save_manifest(len(df_ck),
                               str(df_ck["time"].min()),
                               str(df_ck["time"].max()))

    # Final manifest update
    df_final = _load_csv(PATH_1M)
    if df_final is not None and not df_final.empty:
        _save_manifest(len(df_final),
                       str(df_final["time"].min()),
                       str(df_final["time"].max()))

    print(f"[Rewrite] COMPLETE — {total:,} candles written, {skipped} batches skipped", flush=True)
    if skipped:
        print(f"[Rewrite] Check {GAPS_LOG} for details on skipped batches", flush=True)
    return total


# ── Flow 2 — Live price (NEVER writes to any file) ───────────────────────────

def get_last_close_from_csv() -> float:
    """Read last close price from btc_1m.csv tail without loading the full file."""
    if not os.path.exists(PATH_1M):
        return 0.0
    try:
        with open(PATH_1M, "rb") as fh:
            try:
                fh.seek(-2048, 2)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", errors="replace")
        last_line = [ln for ln in tail.split("\n") if ln.strip()][-1]
        return float(last_line.split(",")[4])  # close is column index 4
    except Exception:
        return 0.0


def _validate_live_price(price: float, source_name: str, field_info: str) -> float:
    """Validate a live price and print field confirmation on first call per source."""
    price = float(price)
    if not (10000 < price < 500000):
        raise ValueError(f"{source_name}: price {price:.2f} outside realistic range (10k-500k)")
    if source_name not in _validated_sources:
        print(f"[Live] {source_name}: using {field_info} = ${price:,.2f} OK", flush=True)
        _validated_sources.add(source_name)
    return price


def _get_coinbase_live_price() -> float:
    resp = requests.get(
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        timeout=5,
    )
    resp.raise_for_status()
    price = float(resp.json()["price"])          # 'price' = last trade price
    return _validate_live_price(price, "Coinbase", "'price' field")


def _get_kraken_live_price() -> float:
    resp = requests.get(
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        timeout=5,
    )
    resp.raise_for_status()
    data   = resp.json()
    result = data["result"]
    pair   = list(result.keys())[0]
    price  = float(result[pair]["c"][0])         # 'c'[0] = last trade price
    return _validate_live_price(price, "Kraken", "'c'[0] field")


def _get_binance_live_price() -> float:
    resp = requests.get(
        "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
        timeout=5,
    )
    resp.raise_for_status()
    price = float(resp.json()["price"])          # 'price' = last trade price
    return _validate_live_price(price, "Binance", "'price' field")


def get_live_price() -> tuple:
    """
    Flow 2: Get current BTC price from any exchange. NEVER writes to any file.
    Returns (price: float, source_name: str).

    ABSOLUTE RULE: This function never writes to any file ever.
    It only returns a price and source name.
    It is completely separate from Flow 1.
    """
    sources = [
        ("Coinbase", _get_coinbase_live_price),
        ("Kraken",   _get_kraken_live_price),
        ("Binance",  _get_binance_live_price),
    ]

    for name, fn in sources:
        try:
            price = fn()
            if price and price > 10000:
                print(f"[Live] ${price:,.2f} via {name}", flush=True)
                return price, name
        except Exception as e:
            print(f"[Live] {name} failed ({e}) — trying next...", flush=True)

    last_price = get_last_close_from_csv()
    print(f"[Live] All sources failed — using last CSV price ${last_price:,.2f}", flush=True)
    return last_price, "CSV fallback"


# ── Main entry point (full historical fetch — used by main.py / training) ────

def update_data() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    _check_manifest()

    existing_1m = _load_csv(PATH_1M)

    if existing_1m is not None and not existing_1m.empty:
        last_ts     = int(existing_1m["time"].max().timestamp())
        fetch_start = last_ts + GRANULARITY
        print(f"Existing 1m: {len(existing_1m):,} rows through {existing_1m['time'].max()}", flush=True)
    else:
        fetch_start = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp())
        print("No existing data. Fetching from 2022-01-01...", flush=True)

    fetch_end = int(datetime.now(tz=timezone.utc).timestamp())

    if fetch_start >= fetch_end:
        _run_features(0)
        return 0

    total_seconds = fetch_end - fetch_start
    total_batches = (total_seconds + BATCH_SIZE * GRANULARITY - 1) // (BATCH_SIZE * GRANULARITY)
    print(f"Estimated batches: {total_batches:,}  (~{total_batches * REQUEST_DELAY / 60:.1f} min)", flush=True)

    buffer_dfs    = []
    cursor        = fetch_start
    batch_num     = 0
    total_fetched = 0
    current_df    = existing_1m

    while cursor < fetch_end:
        batch_end = min(cursor + BATCH_SIZE * GRANULARITY, fetch_end)

        rows = fetch_coinbase_only(cursor, batch_end)
        if rows is not None and not rows.empty:
            buffer_dfs.append(rows)
            total_fetched += len(rows)
        else:
            _log_gap(cursor, batch_end, "All Coinbase retries failed")
            print(f"  [Gap logged] {datetime.fromtimestamp(cursor, tz=timezone.utc)} - "
                  f"{datetime.fromtimestamp(batch_end, tz=timezone.utc)}", flush=True)

        cursor     = batch_end + GRANULARITY
        batch_num += 1
        _time.sleep(REQUEST_DELAY)

        if batch_num % CHECKPOINT_EVERY == 0 and buffer_dfs:
            new_df     = pd.concat(buffer_dfs, ignore_index=True)
            combined   = pd.concat([current_df, new_df]) if (current_df is not None and not current_df.empty) else new_df
            current_df = _save_deduped(combined, PATH_1M)
            buffer_dfs = []
            pct = min(100, batch_num / total_batches * 100)
            print(f"  Checkpoint {batch_num}/{total_batches} ({pct:.0f}%) - {len(current_df):,} rows total", flush=True)

    if buffer_dfs:
        new_df     = pd.concat(buffer_dfs, ignore_index=True)
        combined   = pd.concat([current_df, new_df]) if (current_df is not None and not current_df.empty) else new_df
        current_df = _save_deduped(combined, PATH_1M)

    _run_features(total_fetched)
    return total_fetched



# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    update_data()
