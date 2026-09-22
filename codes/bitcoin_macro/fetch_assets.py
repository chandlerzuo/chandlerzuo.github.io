"""Analogue-asset and hard-asset price history from keyless sources.

  LBMA          gold / silver daily from 1968, platinum / palladium from 1990
  SPDR (SSGA)   ETF NAV history from 2003 (SPY, GLD, XLK, DIA, XLF, XLE, MDY)
  FRED/OECD     monthly national share price indices, several back to 1960
  Philly Fed    ADS business-conditions index
  JST           Jorda-Schularick-Taylor macrohistory, 18 countries from 1870

Notes on sources that do NOT work keylessly, recorded so they are not retried:
  iShares NAV history  - site is client-rendered; the legacy .ajax endpoint now
                         returns 1.5 MB of HTML with HTTP 200 (a silent failure)
  Invesco NAV history  - HTTP 406
  ICE BofA on FRED     - licence-truncated to a rolling 3 years
  Thailand / Taiwan / Hong Kong indices - outside the OECD series family
Substitutes used instead: synthetic constant-maturity Treasury returns for TLT,
Moody's BAA for HY spreads, Fama-French SMB for IWM.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from fetchlib import PROC, fetch, fred

LBMA = {"gold": "gold_pm", "silver": "silver",
        "platinum": "platinum_pm", "palladium": "palladium_pm"}

SPDR = ["spy", "gld", "xlk", "dia", "xlf", "xle", "mdy"]

# OECD monthly share-price indices on FRED: SPASTT01{CC}M661N
OECD_EQUITY = {
    "mx": "Mexico IPC (1970+) - 1994 Tequila crisis analogue",
    "kr": "Korea KOSPI (1981+) - 1997 Asian crisis analogue",
    "br": "Brazil Bovespa (1980+) - hyperinflation / EM boom-bust",
    "tr": "Turkey BIST (1988+) - chronic-inflation equity analogue",
    "cn": "China (1999+) - 2015 retail mania analogue",
    "ru": "Russia RTS (1997+) - 1998 default analogue",
    "id": "Indonesia (1997+) - Asian crisis analogue",
    "za": "South Africa (1960+) - commodity/EM",
    "jp": "Japan (1959+) - 1980s bubble and workout",
    "us": "United States (1957+)",
    "de": "Germany (1960+)",
    "gb": "United Kingdom (1957+)",
}


def lbma() -> pd.DataFrame:
    out = {}
    for name, slug in LBMA.items():
        path = fetch(f"https://prices.lbma.org.uk/json/{slug}.json",
                     f"lbma_{name}.json", ua=True, force=True)
        rows = json.loads(path.read_text())
        s = pd.Series(
            {pd.Timestamp(r["d"]): (r["v"][0] if r.get("v") else None) for r in rows},
            dtype="float64").dropna().sort_index()
        s = s[s > 0]
        out[name] = s
        print(f"lbma {name:10s} n={len(s):6d}  {s.index.min().date()} -> {s.index.max().date()}")
    return pd.DataFrame(out)


def spdr_nav() -> pd.DataFrame:
    """SSGA NAV history workbooks.

    Two parsing traps handled: rows are newest-first, and SSGA overlays a legal
    disclaimer into column A for roughly the first 29 rows, so rows are selected
    by matching a dd-MMM-yyyy date pattern rather than by a fixed header offset.
    """
    out = {}
    for tic in SPDR:
        try:
            path = fetch(
                f"https://www.ssga.com/library-content/products/fund-data/etfs/us/"
                f"navhist-us-en-{tic}.xlsx", f"spdr_{tic}.xlsx", ua=True, force=True)
            raw = pd.read_excel(path, header=None, engine="openpyxl")
        except Exception as exc:  # noqa: BLE001
            print(f"spdr {tic:6s} FAIL {str(exc)[:70]}")
            continue

        a = raw[0].astype(str).str.strip()
        mask = a.str.match(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")
        df = raw[mask].copy()
        dates = pd.to_datetime(df[0].astype(str).str.strip(), format="%d-%b-%Y",
                               errors="coerce")
        nav = pd.to_numeric(df[1], errors="coerce")
        s = pd.Series(nav.values, index=dates).dropna().sort_index()
        s = s[s > 0]
        out[tic] = s
        print(f"spdr {tic:6s} n={len(s):6d}  {s.index.min().date()} -> {s.index.max().date()}")
        time.sleep(0.3)
    return pd.DataFrame(out)


def oecd_equity() -> pd.DataFrame:
    out = {}
    for cc, desc in OECD_EQUITY.items():
        sid = f"SPASTT01{cc.upper()}M661N"
        try:
            df = pd.read_csv(fred(sid))
        except Exception:  # noqa: BLE001
            print(f"oecd {cc} FAIL")
            continue
        s = pd.Series(pd.to_numeric(df[sid], errors="coerce").values,
                      index=pd.to_datetime(df[df.columns[0]])).dropna()
        out[cc] = s
        print(f"oecd {cc:4s} n={len(s):5d}  {s.index.min().date()} -> "
              f"{s.index.max().date()}  {desc}")
    return pd.DataFrame(out)


def extras() -> dict[str, pd.Series]:
    """Long commodity and activity series that backfill the modern daily ones."""
    out = {}
    for sid in ("WTISPLC", "PURANUSDM", "PCOPPUSDM", "PALLFNFINDEXM",
                "CPIAUCSL", "NASDAQCOM", "NIKKEI225", "BBKMLEIX"):
        try:
            df = pd.read_csv(fred(sid))
            s = pd.Series(pd.to_numeric(df[sid], errors="coerce").values,
                          index=pd.to_datetime(df[df.columns[0]])).dropna()
            out[sid] = s
            print(f"extra {sid:16s} n={len(s):6d}  {s.index.min().date()} -> {s.index.max().date()}")
        except Exception:  # noqa: BLE001
            print(f"extra {sid:16s} FAIL")
    return out


def jst() -> pd.DataFrame | None:
    """Jorda-Schularick-Taylor: 18 advanced economies, annual, from 1870."""
    try:
        path = fetch("https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.dta",
                     "jst_r6.dta", ua=True)
        df = pd.read_stata(path)
    except Exception as exc:  # noqa: BLE001
        print(f"jst FAIL {str(exc)[:90]}")
        return None
    print(f"jst  n={len(df)}  years {int(df['year'].min())}-{int(df['year'].max())}  "
          f"countries={df['country'].nunique()}  cols={len(df.columns)}")
    keep = [c for c in ("year", "country", "iso", "eq_tr", "housing_tr", "bond_tr",
                        "bill_rate", "ltrate", "cpi", "rgdpmad", "eq_capgain",
                        "eq_dp", "stir", "safe_tr") if c in df.columns]
    print(f"     asset-return cols present: {[c for c in keep if 'tr' in c or 'rate' in c]}")
    return df[keep]


def main() -> int:
    print("--- LBMA precious metals ---")
    metals = lbma()
    print("\n--- SPDR ETF NAV history ---")
    etfs = spdr_nav()
    print("\n--- OECD national equity indices (monthly) ---")
    oecd = oecd_equity()
    print("\n--- long commodity / activity series ---")
    ex = extras()
    print("\n--- JST macrohistory ---")
    j = jst()

    metals.to_parquet(PROC / "metals_daily.parquet")
    etfs.to_parquet(PROC / "etf_nav_daily.parquet")
    oecd.to_parquet(PROC / "oecd_equity_monthly.parquet")
    pd.DataFrame(ex).to_parquet(PROC / "long_macro.parquet")
    if j is not None:
        j.to_parquet(PROC / "jst_annual.parquet")

    print("\nwrote metals_daily, etf_nav_daily, oecd_equity_monthly, long_macro"
          + (", jst_annual" if j is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
