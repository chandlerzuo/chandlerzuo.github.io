"""Fetch the FRED backbone, one series per request, and report coverage.

Series are grouped by the role they play in the study so that a gap in one
group is immediately visible as a gap in an economic channel, not just a
missing file.
"""
from __future__ import annotations

import sys

import pandas as pd

from fetchlib import PROC, fred

CATALOG: dict[str, list[str]] = {
    # --- nominal rates and curve -------------------------------------------
    "rates_nominal": ["DFF", "DGS1MO", "DGS3MO", "DGS1", "DGS2", "DGS5",
                      "DGS10", "DGS20", "DGS30", "T10Y2Y", "T10Y3M"],
    # --- real rates and inflation compensation (the 'liquidity/real rate' thesis)
    # TIPS-based series start only in 2003. The Cleveland Fed model-based pair
    # (EXPINF10YR / REAINTRATREARAT10Y, monthly from 1982) is what lets the
    # real-rate channel be measured in the pre-TIPS era, which the historical
    # analogue matching in Phase 6 needs.
    "rates_real": ["DFII5", "DFII10", "DFII20", "DFII30",
                   "T5YIE", "T10YIE", "T5YIFR",
                   "EXPINF10YR", "REAINTRATREARAT10Y", "MICH",
                   "THREEFYTP10", "THREEFY10"],
    # --- dollar -------------------------------------------------------------
    "dollar": ["DTWEXBGS", "DTWEXM", "DTWEXAFEGS", "DEXUSEU", "DEXJPUS",
               "DEXCHUS", "DEXUSUK"],
    # --- risk appetite / credit (the 'risk asset' thesis) -------------------
    # WARNING: every ICE BofA (BAML*) series is licence-truncated on the public
    # fredgraph endpoint to a rolling ~3 years. They are kept for the recent
    # sample only. The Moody's BAA10Y/AAA10Y spreads (daily, 1983/1986+) and
    # BAA/AAA (monthly, 1919+) are the unrestricted long-history credit proxies
    # and carry the credit channel for everything before 2023.
    "risk": ["VIXCLS", "VXOCLS", "VXNCLS", "OVXCLS", "STLFSI4", "NFCI", "ANFCI",
             "BAA10Y", "AAA10Y", "DBAA", "DAAA", "BAA", "AAA", "TEDRATE",
             "RIFSPPFAAD90NB", "USEPUINDXD",
             "BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A0HYM2EY"],
    # --- credit / bond total return indices (tradable proxies, 3y only) -----
    "total_return": ["BAMLHYH0A0HYM2TRIV", "BAMLCC0A0CMTRIV",
                     "BAMLHE00EHYITRIV", "BAMLEMCBPITRIV"],
    # --- central bank liquidity (weekly) -----------------------------------
    "liquidity": ["WALCL", "WRESBAL", "RRPONTSYD", "WTREGEN", "WSHOSHO",
                  "SWPT", "M1SL", "M2SL", "BOGMBASE", "TOTRESNS",
                  "SOFR", "EFFR", "IORB"],
    # --- equity and commodity price levels ---------------------------------
    # SP500 and DJIA are licence-truncated to ~10 years. The market factor for
    # the full history comes from Fama-French Mkt-RF (daily, 1926+) instead;
    # NASDAQCOM (1971+) and NASDAQ100 (1986+) are unrestricted.
    "equity": ["NASDAQCOM", "NASDAQ100", "NIKKEI225", "SP500", "DJIA"],
    "commodity": ["DCOILWTICO", "WTISPLC", "DCOILBRENTEU", "DHHNGSP",
                  "PCOPPUSDM", "PALLFNFINDEXM", "IR14270"],
    # --- macro activity / inflation ----------------------------------------
    # XTEXVA01USM667S / XTIMVA01USM667S are monthly US exports and imports from
    # 1955; their difference is the trade balance used as a macro-state axis in
    # the analogue search, where a 1973+ history is needed.
    "macro": ["WEI", "CPIAUCSL", "CPILFESL", "INDPRO", "UNRATE", "PAYEMS",
              "GDPC1", "UMCSENT", "USEPUINDXD",
              "XTEXVA01USM667S", "XTIMVA01USM667S", "NETEXP"],
    # --- crypto cross-check -------------------------------------------------
    "crypto": ["CBBTCUSD", "CBETHUSD"],
}


def load(series: str) -> pd.Series:
    path = fred(series)
    df = pd.read_csv(path)
    datecol = df.columns[0]
    df[datecol] = pd.to_datetime(df[datecol])
    # FRED now emits empty strings for missing values (the legacy "." is gone).
    out = pd.to_numeric(df[series], errors="coerce")
    out.index = df[datecol]
    out.name = series
    return out.dropna()


def main() -> int:
    rows, frames, failures = [], {}, []
    for group, series_list in CATALOG.items():
        for s in series_list:
            try:
                ser = load(s)
            except Exception as exc:  # noqa: BLE001 - want the reason, keep going
                failures.append((group, s, str(exc)[:90]))
                continue
            frames[s] = ser
            freq = pd.infer_freq(ser.index[:60]) or "irregular"
            rows.append({
                "group": group, "series": s, "n": len(ser),
                "start": ser.index.min().date(), "end": ser.index.max().date(),
                "inferred_freq": freq,
            })

    cov = pd.DataFrame(rows).sort_values(["group", "start"])
    cov.to_csv(PROC.parent / "fred_coverage.csv", index=False)
    print(cov.to_string(index=False))

    if failures:
        print("\n--- UNAVAILABLE (dropped, not silently zero-filled) ---")
        for g, s, e in failures:
            print(f"  {g:14s} {s:22s} {e}")

    # Store as a long table: heterogeneous frequencies must not be force-aligned
    # here. Alignment is a modelling decision made in build_panel.py.
    long = (pd.concat(frames, names=["series", "date"])
              .rename("value").reset_index())
    long.to_parquet(PROC / "fred_long.parquet", index=False)
    print(f"\nwrote {PROC / 'fred_long.parquet'}  rows={len(long):,}  series={len(frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
