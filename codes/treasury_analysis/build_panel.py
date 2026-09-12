#!/usr/bin/env python3
"""
Build structured panels from the raw Treasury.gov daily CSVs.

Inputs  (data/raw/):
  tsy_curve_<YYYY>.csv   Daily Treasury Par Yield Curve Rates (percent)
  tsy_bill_<YYYY>.csv    Daily Treasury Bill Rates (bank discount + coupon-equivalent)

Outputs (data/processed/):
  yields_daily.csv       wide daily panel; columns = canonical tenor labels (percent)
  short_rate_daily.csv   4-week short rate for the rolling strategy, with source flag
  metadata.json          provenance + coverage
"""
from __future__ import annotations

import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# canonical tenor -> maturity in years
TENORS = {
    "1Mo": 1/12, "1.5Mo": 1.5/12, "2Mo": 2/12, "3Mo": 0.25, "4Mo": 4/12,
    "6Mo": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10,
    "20Y": 20, "30Y": 30,
}


def normalize_tenor(label: str):
    """Map a raw Treasury column header to a canonical tenor key, or None."""
    s = label.strip().strip('"').lower()
    m = re.match(r"([\d.]+)\s*(mo|month|weeks?|wk|yr|year)", s)
    if not m:
        return None
    num = float(m.group(1)); unit = m.group(2)
    if unit.startswith("mo") or unit.startswith("month"):
        key = f"{num:g}Mo"
    elif unit.startswith("yr") or unit.startswith("year"):
        key = f"{num:g}Y"
    else:
        return None
    return key if key in TENORS else None


def parse_curve():
    frames = []
    for f in sorted(glob.glob(str(RAW / "tsy_curve_*.csv"))):
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        colmap = {c: normalize_tenor(c) for c in df.columns if c != "Date" and c != "date"}
        colmap = {c: k for c, k in colmap.items() if k}
        sub = df[["date"] + list(colmap)].rename(columns=colmap)
        frames.append(sub)
    panel = pd.concat(frames, ignore_index=True).groupby("date").mean()
    panel = panel.sort_index()
    # order columns by maturity
    cols = [t for t in TENORS if t in panel.columns]
    return panel[cols]


def parse_bills():
    """Return DataFrame indexed by date with 4-week coupon-equivalent yield (%)."""
    frames = []
    for f in sorted(glob.glob(str(RAW / "tsy_bill_*.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        ce = [c for c in df.columns if "4 WEEKS" in c.upper() and "COUPON" in c.upper()]
        if not ce:
            continue
        sub = df[["date", ce[0]]].rename(columns={ce[0]: "bill4wk_ce"})
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["bill4wk_ce"])
    return pd.concat(frames, ignore_index=True).groupby("date").mean().sort_index()


def main():
    curve = parse_curve()
    curve.to_csv(PROC / "yields_daily.csv")
    print(f"yields_daily.csv: {curve.shape[0]} days x {curve.shape[1]} tenors "
          f"({curve.index.min().date()} -> {curve.index.max().date()})")
    print("  tenors:", list(curve.columns))
    print("  30Y coverage:", int(curve['30Y'].notna().sum()), "days; missing years:",
          sorted(set(curve.index.year) - set(curve['30Y'].dropna().index.year)))

    bills = parse_bills()

    # short-rate splice: prefer true 4-week bill CE; else 1Mo par; else 3Mo par (proxy)
    idx = curve.index.union(bills.index).sort_values()
    b4 = bills["bill4wk_ce"].reindex(idx) if not bills.empty else pd.Series(index=idx, dtype=float)
    m1 = curve["1Mo"].reindex(idx) if "1Mo" in curve else pd.Series(index=idx, dtype=float)
    m3 = curve["3Mo"].reindex(idx)
    rate = b4.copy()
    src = pd.Series("bill4wk_ce", index=idx)
    use_m1 = rate.isna() & m1.notna()
    rate[use_m1] = m1[use_m1]; src[use_m1] = "par_1Mo"
    use_m3 = rate.isna() & m3.notna()
    rate[use_m3] = m3[use_m3]; src[use_m3] = "par_3Mo_proxy"
    src[rate.isna()] = "missing"
    short = pd.DataFrame({"short_rate_pct": rate, "source": src}).dropna(subset=["short_rate_pct"])
    short.index.name = "date"
    short.to_csv(PROC / "short_rate_daily.csv")
    vc = short["source"].value_counts().to_dict()
    print(f"short_rate_daily.csv: {len(short)} days; sources={vc}")

    meta = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "source": "U.S. Department of the Treasury (home.treasury.gov) daily rate CSVs",
        "datasets": ["daily_treasury_yield_curve (par yields, %)",
                     "daily_treasury_bill_rates (4wk coupon-equivalent, %)"],
        "tenors_years": TENORS,
        "curve_coverage": {"start": str(curve.index.min().date()),
                            "end": str(curve.index.max().date()),
                            "n_days": int(curve.shape[0])},
        "short_rate_sources": vc,
        "notes": [
            "Par yields: coupon of a par bond == par yield at issue.",
            "30Y par yield discontinued Feb-2002..Feb-2006; NS fit interpolates/extrapolates.",
            "4-week bill coupon-equivalent (investment) yield used for rolling strategy; "
            "1Mo/3Mo par yields splice earlier history as proxy.",
        ],
    }
    (PROC / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print("metadata.json written")


if __name__ == "__main__":
    main()
