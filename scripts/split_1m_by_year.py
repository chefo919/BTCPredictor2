"""
One-time utility: split btc_1m.csv into data/yearly_1m/YYYY_btc_1m.csv files.

Run once locally before pushing to GitHub:
    python scripts/split_1m_by_year.py

Each yearly file is ~25-35 MB (well under GitHub's 100 MB limit).
After running, commit data/yearly_1m/ and you can delete btc_1m.csv locally.
"""
import os
import sys
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC     = os.path.join(ROOT, "data", "btc_1m.csv")
OUT_DIR = os.path.join(ROOT, "data", "yearly_1m")


def split():
    if not os.path.exists(SRC):
        print(f"Not found: {SRC}")
        sys.exit(1)

    print(f"Loading {SRC} ...", flush=True)
    df = pd.read_csv(SRC, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    print(f"  {len(df):,} rows  ({df['time'].iloc[0]} to {df['time'].iloc[-1]})", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for year, group in df.groupby(df["time"].dt.year):
        out_path = os.path.join(OUT_DIR, f"{year}_btc_1m.csv")
        group.reset_index(drop=True).to_csv(out_path, index=False)
        mb = os.path.getsize(out_path) / 1024**2
        print(f"  {year}: {len(group):,} rows  ->  {out_path}  ({mb:.1f} MB)", flush=True)

    print(f"\nDone. {OUT_DIR}/ is ready to commit.")
    print("After pushing, btc_1m.csv is no longer needed on Colab.")


if __name__ == "__main__":
    split()
