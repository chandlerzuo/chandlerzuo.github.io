#!/usr/bin/env python3
"""
Fetch U.S. Treasury yield data from FRED (no API key needed) and build a
structured, cleaned panel for downstream analysis.

Series pulled (all daily, business-day frequency, percent):
  Constant-maturity par yields (used for curve fitting + bond pricing):
    DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30
  Bill rates for the rolling short-rate strategy:
    DTB4WK  -> 4-week bill secondary market rate (discount basis), from 2001-07
    DTB3    -> 3-month bill secondary market rate (discount basis), from 1954
              (used as pre-2001 proxy to extend the short-rate history)

Outputs (data/):
  raw/<SERIES>.csv                 raw FRED downloads
  processed/yields_daily.csv       wide daily panel of all series (float, NaN for missing)
  processed/short_rate_daily.csv   spliced 4-week short rate (DTB4WK, DTB3 proxy pre-2001)
  processed/metadata.json          provenance + coverage
"""
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# maturity (in years) for each constant-maturity series
CMT = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2,
    "DGS3": 3, "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30,
}
BILLS = ["DTB4WK", "DTB3"]
ALL_SERIES = list(CMT) + BILLS

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_series(sid: str) -> pd.Series:
    url = FRED_CSV.format(sid=sid)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    (RAW / f"{sid}.csv").write_text(raw)
    df = pd.read_csv(io.StringIO(raw))
    # FRED CSVs: first col is date (name varies: 'DATE' or 'observation_date'), second is value
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")  # '.' -> NaN
    s = df.set_index(date_col)[val_col]
    s.name = sid
    s.index.name = "date"
    return s


def main():
    print("Fetching FRED series...")
    series = {}
    for sid in ALL_SERIES:
        try:
            s = fetch_series(sid)
            series[sid] = s
            valid = s.dropna()
            print(f"  {sid:8s} {len(valid):6d} obs  "
                  f"{valid.index.min().date()} -> {valid.index.max().date()}")
        except Exception as e:  # noqa
            print(f"  {sid:8s} FAILED: {e}", file=sys.stderr)

    panel = pd.concat(series.values(), axis=1).sort_index()
    panel = panel[(panel.index >= "1970-01-01")]
    panel.to_csv(PROC / "yields_daily.csv")
    print(f"\nSaved daily panel: {panel.shape[0]} rows x {panel.shape[1]} cols "
          f"-> processed/yields_daily.csv")

    # ---- spliced 4-week short rate: DTB4WK where available, else DTB3 (proxy) ----
    short = panel["DTB4WK"].copy()
    proxy = panel["DTB3"].copy()
    spliced = short.copy()
    used_proxy = short.isna() & proxy.notna()
    spliced[used_proxy] = proxy[used_proxy]
    short_df = pd.DataFrame({
        "short_rate_pct": spliced,
        "source": np.where(short.notna(), "DTB4WK",
                           np.where(used_proxy, "DTB3_proxy", "missing")),
    })
    short_df = short_df.dropna(subset=["short_rate_pct"])
    short_df.to_csv(PROC / "short_rate_daily.csv")
    n4 = (short_df["source"] == "DTB4WK").sum()
    npx = (short_df["source"] == "DTB3_proxy").sum()
    print(f"Saved spliced short rate: {len(short_df)} obs "
          f"({n4} DTB4WK, {npx} DTB3 proxy) -> processed/short_rate_daily.csv")

    meta = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "source": "FRED (fred.stlouisfed.org) CSV endpoint, no API key",
        "cmt_maturities_years": CMT,
        "bill_series": BILLS,
        "coverage": {
            sid: {
                "n": int(series[sid].dropna().shape[0]),
                "start": str(series[sid].dropna().index.min().date()),
                "end": str(series[sid].dropna().index.max().date()),
            } for sid in series
        },
        "notes": [
            "DGS* are constant-maturity PAR yields (%). Coupon of a par bond == par yield.",
            "DTB4WK/DTB3 are secondary-market DISCOUNT-basis bill rates (%).",
            "Short-rate series splices DTB3 as a proxy before DTB4WK begins 2001-07.",
        ],
    }
    (PROC / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved processed/metadata.json")


if __name__ == "__main__":
    main()
