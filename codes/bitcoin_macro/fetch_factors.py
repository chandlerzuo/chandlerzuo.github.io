"""Fama-French daily factors + momentum.

Parsing quirks handled here (all confirmed against the live files):
  * a multi-line text preamble precedes the header row
  * dates are bare YYYYMMDD integers
  * missing values are -99.99 / -999
  * a trailing copyright line, and in some files an ANNUAL block appended
    after the daily block -- detected by row length, not by line number
  * values are in PERCENT, converted to decimal here
  * the files lag ~2 months behind today (CRSP lag), so factor-based analysis
    cannot run to the BTC sample end. That truncation is explicit, not silent.
"""
from __future__ import annotations

import io
import sys
import zipfile

import pandas as pd

from fetchlib import PROC, fetch

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "industry49_daily": "49_Industry_Portfolios_daily_CSV.zip",
}


def parse(path, *, daily: bool) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        text = zf.read(name).decode("latin-1")

    width = 8 if daily else 6
    recs = []
    for line in io.StringIO(text):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        # A monthly/annual block appended after a daily block has shorter dates;
        # keying on token length keeps only the intended frequency.
        if len(parts[0]) != width:
            continue
        try:
            vals = [float(p) for p in parts[1:] if p != ""]
        except ValueError:
            continue
        recs.append([parts[0]] + vals)

    if not recs:
        raise RuntimeError(f"no rows parsed from {path}")

    ncol = max(len(r) for r in recs)
    recs = [r for r in recs if len(r) == ncol]
    df = pd.DataFrame(recs)
    df[0] = pd.to_datetime(df[0], format="%Y%m%d" if daily else "%Y%m")
    df = df.set_index(0)
    df.index.name = "date"

    # Column names are positional in these files; assign by known layout.
    layouts = {
        6: ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        4: ["Mkt-RF", "SMB", "HML", "RF"],
        1: ["Mom"],
    }
    if ncol - 1 in layouts:
        df.columns = layouts[ncol - 1]
    else:
        df.columns = [f"c{i}" for i in range(ncol - 1)]

    df = df.astype(float).replace([-99.99, -999.0, -99.0], float("nan")) / 100.0
    return df.dropna(how="all")


def main() -> int:
    out = {}
    for key, fname in FILES.items():
        try:
            path = fetch(BASE + fname, f"ff_{key}.zip")
            df = parse(path, daily="daily" in key)
        except Exception as exc:  # noqa: BLE001
            print(f"{key:20s} FAIL {str(exc)[:80]}")
            continue
        out[key] = df
        print(f"{key:20s} OK  n={len(df):6d}  {df.index.min().date()} -> "
              f"{df.index.max().date()}  cols={list(df.columns)}")

    # Merge the daily research factors into one table (the common analysis input).
    ff5, mom = out.get("ff5_daily"), out.get("mom_daily")
    if ff5 is not None:
        daily = ff5.join(mom, how="left") if mom is not None else ff5
        daily.to_parquet(PROC / "ff_daily.parquet")
        print(f"\nwrote {PROC / 'ff_daily.parquet'} n={len(daily)} "
              f"{daily.index.min().date()} -> {daily.index.max().date()}")
    for key in ("ff5_monthly", "mom_monthly", "industry49_daily"):
        if key in out:
            out[key].to_parquet(PROC / f"{key}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
