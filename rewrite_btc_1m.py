"""
Rewrites btc_1m.csv from scratch by fetching all 1-minute BTC/USD candles
from Coinbase Exchange API from 2022-01-01 to now.

Safeguards:
  - Backs up the current file before deleting
  - Validates every batch (OHLCV sanity) before writing
  - Never writes partial open candles (stops 2 minutes before now)
  - Updates manifest every 500 batches so progress survives interruption
  - All gaps logged to data/gaps.log

After this completes, run:
    python features/build_ohlcv.py
    python features/engineer.py
    python validate_data.py

Usage:
    python rewrite_btc_1m.py
    python rewrite_btc_1m.py --start 2023-01-01   (partial rewrite)
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_manager import rewrite_1m_csv, get_last_1m_update

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01",
                        help="Start date for rewrite (default: 2022-01-01)")
    args = parser.parse_args()

    print("=" * 62)
    print("BTC 1-MINUTE CSV REWRITE")
    print("=" * 62)

    info = get_last_1m_update()
    print(f"Current CSV:  {info['rows']:,} rows")
    print(f"First candle: {info['first_candle']}")
    print(f"Last candle:  {info['last_candle']}")
    print(f"Last updated: {info['verified_at']}")
    print()
    print(f"Rewriting from {args.start} to now using Coinbase Exchange API.")
    print("The current file will be backed up before deletion.")
    print()

    response = input("Proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Aborted.")
        sys.exit(0)

    print()
    total = rewrite_1m_csv(start_date=args.start)
    print()
    print("=" * 62)
    info = get_last_1m_update()
    print(f"Done. {total:,} candles written.")
    print(f"First: {info['first_candle']}")
    print(f"Last:  {info['last_candle']}")
    print()
    print("Next steps:")
    print("  python features/validate_1m_online.py")
    print("  python features/build_ohlcv.py --dry-run")
    print("  python features/build_ohlcv.py")
    print("  python features/engineer.py")
    print("  python validate_data.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
