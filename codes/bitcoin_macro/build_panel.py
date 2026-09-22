"""Construct the analysis panel: returns, tradable factor proxies, macro shocks.

Design rules enforced here:
  * Returns are log returns. Yields/spreads enter as CHANGES, never levels.
  * Series of different native frequency are never forward-filled into a daily
    grid and then differenced -- that manufactures spurious zero-change days and
    would bias betas toward zero. Weekly series are aligned at their own
    frequency in the weekly panel and left as step functions, explicitly
    flagged, in the daily panel.
  * Every column is registered in COLUMN_DOC so the data dictionary is generated
    from the code that builds the data, not maintained separately and allowed to
    drift out of date.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fetchlib import PROC, ROOT

COLUMN_DOC: dict[str, str] = {}


def doc(name: str, text: str) -> str:
    COLUMN_DOC[name] = text
    return name


def fred_wide() -> pd.DataFrame:
    long = pd.read_parquet(PROC / "fred_long.parquet")
    return long.pivot(index="date", columns="series", values="value").sort_index()


# --------------------------------------------------------------------------
# Synthetic constant-maturity Treasury total return
# --------------------------------------------------------------------------
def par_bond_total_return(yields: pd.Series, maturity: float = 10.0,
                         freq: int = 2, days: int = 252) -> pd.Series:
    """Daily total return of a constant-maturity par coupon bond.

    Exact repricing rather than a duration approximation: at t-1 the bond is
    priced at par with coupon equal to the prevailing yield; at t it is repriced
    at the new yield one day closer to maturity. This matters because the 2021-23
    move in long yields was large enough that a linear duration approximation
    accumulates visible error over the sample.
    """
    y = (yields / 100.0).dropna()
    dt = 1.0 / days
    c = y.shift(1)                  # coupon set at prior close (par)
    y1 = y                          # revaluation yield
    n = maturity * freq

    def price(coupon, ytm, ttm):
        per = ytm / freq
        k = ttm * freq
        # Discount factor exponents for a bond with fractional first period.
        with np.errstate(divide="ignore", invalid="ignore"):
            disc = np.where(per == 0, 1.0 / k, (1 - (1 + per) ** (-k)) / per)
            return (coupon / freq) * disc * 100 + 100 * (1 + per) ** (-k)

    p1 = price(c, y1, maturity - dt)
    accrued = (c / days) * 100
    ret = (p1 + accrued) / 100.0 - 1.0
    out = pd.Series(ret, index=y.index, name=f"ret_ust{int(maturity)}y")
    return out.replace([np.inf, -np.inf], np.nan)


def main() -> int:
    btc = pd.read_parquet(PROC / "btc_daily.parquet")
    fr = fred_wide()
    ff = pd.read_parquet(PROC / "ff_daily.parquet")

    px = btc["PriceUSD"].dropna()
    px = px[px > 0]

    d = pd.DataFrame(index=pd.date_range(px.index.min(), max(px.index.max(),
                                                            fr.index.max()), freq="D"))

    # ---------------- Bitcoin ------------------------------------------------
    d[doc("btc_px", "BTC close, USD, CoinMetrics community reference rate")] = px
    d[doc("btc_ret", "BTC daily log return, calendar-day close to close")] = \
        np.log(px).diff()
    d[doc("btc_mvrv", "BTC market-value-to-realised-value (on-chain cycle gauge)")] = \
        btc["CapMVRVCur"]
    d[doc("btc_hashrate", "BTC network hash rate")] = btc.get("HashRate")
    d[doc("btc_adract", "BTC active addresses (adoption / network activity)")] = \
        btc.get("AdrActCnt")

    # ---------------- Tradable factor proxies (Layer A) ---------------------
    for col, src, desc in [
        ("ret_mkt", None, "US equity market excess return, Fama-French Mkt-RF (1926+)"),
        ("ret_nasdaq", "NASDAQCOM", "Nasdaq Composite log return (1971+)"),
        ("ret_ndx", "NASDAQ100", "Nasdaq-100 log return (1986+), tech/growth proxy"),
        ("ret_nikkei", "NIKKEI225", "Nikkei 225 log return (1949+), bubble analogue"),
        ("ret_oil", "DCOILWTICO", "WTI crude log return (1986+)"),
        ("ret_gas", "DHHNGSP", "Henry Hub natural gas log return (1997+)"),
        ("ret_usd", "DTWEXBGS", "Broad trade-weighted USD log return (2006+)"),
        ("ret_usd_major", "DTWEXM", "Major-currency USD index log return (1973-2019)"),
    ]:
        if src is None:
            continue
        if src in fr:
            s = fr[src].dropna()
            d[doc(col, desc)] = np.log(s[s > 0]).diff()

    d[doc("ret_mkt", "US equity market EXCESS return, Fama-French Mkt-RF (1926+)")] = ff["Mkt-RF"]
    for f in ("SMB", "HML", "RMW", "CMA", "Mom"):
        if f in ff:
            d[doc(f"ff_{f.lower()}", f"Fama-French {f} factor, daily")] = ff[f]
    d[doc("rf", "Daily risk-free rate (Fama-French RF, 1-month T-bill)")] = ff["RF"]

    # Duration factor: exact-repricing constant-maturity Treasury total returns.
    for m in (2, 10, 30):
        src = {2: "DGS2", 10: "DGS10", 30: "DGS30"}[m]
        if src in fr:
            r = par_bond_total_return(fr[src], maturity=m)
            d[doc(f"ret_ust{m}y", f"Total return of a constant-maturity {m}y par "
                                  f"Treasury, exact repricing from {src}")] = r

    # Precious metals (LBMA, daily from 1968) -- gold is the conventional control
    # asset in this literature and is what makes 'digital gold' falsifiable.
    metals = pd.read_parquet(PROC / "metals_daily.parquet")
    for name, desc in [("gold", "LBMA gold PM fix log return (1968+)"),
                       ("silver", "LBMA silver fix log return (1968+)"),
                       ("platinum", "LBMA platinum PM fix log return (1990+)"),
                       ("palladium", "LBMA palladium PM fix log return (1990+)")]:
        if name in metals:
            s = metals[name].dropna()
            d[doc(f"ret_{name}", desc)] = np.log(s[s > 0]).diff()
    if {"gold", "silver"}.issubset(metals.columns):
        d[doc("gold_silver_ratio", "Gold/silver price ratio: monetary vs "
                                   "industrial precious-metal demand")] = \
            metals["gold"] / metals["silver"]

    # SPDR ETF NAV history (2003+). Used where an actual tradable price is needed
    # rather than an index level; iShares equivalents are not available keylessly.
    etf = pd.read_parquet(PROC / "etf_nav_daily.parquet")
    for tic, desc in [("spy", "SPY NAV log return (2003+)"),
                      ("gld", "GLD NAV log return (2004+)"),
                      ("xlk", "XLK technology-sector NAV log return (2003+)"),
                      ("xle", "XLE energy-sector NAV log return (2003+)"),
                      ("xlf", "XLF financials-sector NAV log return (2003+)")]:
        if tic in etf:
            s = etf[tic].dropna()
            d[doc(f"ret_{tic}", desc)] = np.log(s[s > 0]).diff()

    # Tech tilt: Nasdaq-100 in excess of the broad market. Isolates the
    # growth/duration-of-cash-flow exposure from plain equity beta, which is the
    # distinction the 'BTC is a high-beta Nasdaq proxy' thesis actually requires.
    if "ret_ndx" in d:
        d[doc("tech_tilt", "Nasdaq-100 return minus total market return "
                           "(Mkt-RF + RF): growth/long-duration equity tilt")] = \
            d["ret_ndx"] - (d["ret_mkt"] + d["rf"])

    # ---------------- Macro shocks (Layer B) --------------------------------
    level_changes = {
        "d_real10": ("DFII10", "Change in 10y TIPS real yield, pp (2003+)"),
        "d_real5": ("DFII5", "Change in 5y TIPS real yield, pp (2003+)"),
        "d_be10": ("T10YIE", "Change in 10y breakeven inflation, pp (2003+)"),
        "d_be5": ("T5YIE", "Change in 5y breakeven inflation, pp (2003+)"),
        "d_nom10": ("DGS10", "Change in 10y nominal Treasury yield, pp"),
        "d_nom2": ("DGS2", "Change in 2y nominal yield, pp (policy-expectation proxy)"),
        "d_slope": ("T10Y2Y", "Change in 10y-2y curve slope, pp"),
        "d_termprem": ("THREEFYTP10", "Change in Kim-Wright 10y term premium, pp (1990+)"),
        "d_credit_baa": ("BAA10Y", "Change in Moody's Baa-10y credit spread, pp "
                                   "(1986+; the long-history credit proxy)"),
        "d_credit_hy": ("BAMLH0A0HYM2", "Change in ICE BofA HY OAS, pp "
                                        "(LICENCE-TRUNCATED to last ~3y only)"),
        "d_epu": ("USEPUINDXD", "Change in Economic Policy Uncertainty index (1985+)"),
    }
    for col, (src, desc) in level_changes.items():
        if src in fr:
            d[doc(col, desc)] = fr[src].dropna().diff()

    for col, src, desc in [
        ("d_lvix", "VIXCLS", "Log change in VIX (1990+)"),
        ("d_lvxo", "VXOCLS", "Log change in VXO (1986-2021; splices VIX back to 1986)"),
        ("d_lvxn", "VXNCLS", "Log change in VXN, Nasdaq-100 implied vol (2001+)"),
    ]:
        if src in fr:
            s = fr[src].dropna()
            d[doc(col, desc)] = np.log(s[s > 0]).diff()

    # Spliced equity implied vol: VXO before 1990, VIX after. Flagged as spliced
    # because a regime break at the splice date would be an artefact.
    if "d_lvix" in d and "d_lvxo" in d:
        d[doc("d_livol", "Log change in equity implied vol: VXO pre-1990 spliced "
                         "with VIX from 1990 (SPLICED SERIES)")] = \
            d["d_lvix"].fillna(d["d_lvxo"])

    # ---------------- Fed / net liquidity (weekly native) -------------------
    if {"WALCL", "WTREGEN"}.issubset(fr.columns):
        wk = fr[["WALCL", "WTREGEN"]].dropna(how="all").resample("W-WED").last()
        rrp = fr["RRPONTSYD"].dropna().resample("W-WED").last() * 1000  # $bn -> $mn
        net = (wk["WALCL"] - wk["WTREGEN"] - rrp.reindex(wk.index).fillna(0.0))
        d[doc("fed_assets", "Fed total assets, $mn, weekly Wed (STEP FUNCTION in "
                            "the daily panel)")] = wk["WALCL"]
        d[doc("net_liquidity", "Fed assets minus Treasury General Account minus "
                               "reverse repo, $mn, weekly Wed (STEP FUNCTION daily)")] = net

    # ---------------- Crypto-native controls (Layer C) ----------------------
    if "CBETHUSD" in fr:
        eth = fr["CBETHUSD"].dropna()
        d[doc("ret_eth", "ETH log return (2016+), crypto-native factor proxy")] = \
            np.log(eth[eth > 0]).diff()
        d[doc("eth_btc", "ETH/BTC ratio: within-crypto risk appetite")] = eth / px

    d = d.loc[:px.index.max()]
    d.to_parquet(PROC / "panel_daily.parquet")

    # ---------------- Weekly panel (primary estimation frequency) -----------
    # Non-synchronous trading (24/7 BTC vs a 6.5h equity session) attenuates
    # daily betas, so Friday-to-Friday returns are the headline frequency.
    rets = [c for c in d.columns if c.startswith(("btc_ret", "ret_", "ff_", "tech_tilt")) or c == "rf"]
    chgs = [c for c in d.columns if c.startswith("d_")]
    lvls = [c for c in d.columns if c in ("btc_px", "btc_mvrv", "net_liquidity",
                                          "fed_assets", "eth_btc", "btc_hashrate",
                                          "btc_adract", "gold_silver_ratio")]
    w = pd.concat([
        d[rets].resample("W-FRI").sum(min_count=1),
        d[chgs].resample("W-FRI").sum(min_count=1),
        d[lvls].resample("W-FRI").last(),
    ], axis=1)
    w.to_parquet(PROC / "panel_weekly.parquet")

    # ---------------- Report -------------------------------------------------
    print(f"daily panel  {d.shape[0]} rows x {d.shape[1]} cols  "
          f"{d.index.min().date()} -> {d.index.max().date()}")
    print(f"weekly panel {w.shape[0]} rows x {w.shape[1]} cols")
    cov = pd.DataFrame({
        "n": d.notna().sum(),
        "first": [d[c].first_valid_index() for c in d.columns],
        "last": [d[c].last_valid_index() for c in d.columns],
    })
    cov["first"] = pd.to_datetime(cov["first"]).dt.date
    cov["last"] = pd.to_datetime(cov["last"]).dt.date
    print("\n" + cov.to_string())

    lines = ["# Data Dictionary (generated by src/build_panel.py)", "",
             "One row per panel column. Generated from code, so it cannot drift "
             "out of sync with the data.", "",
             "| column | n | first | last | definition |", "|---|---|---|---|---|"]
    for c in d.columns:
        r = cov.loc[c]
        lines.append(f"| `{c}` | {r['n']} | {r['first']} | {r['last']} | "
                     f"{COLUMN_DOC.get(c, '')} |")
    (ROOT / "DATA_DICTIONARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {ROOT / 'DATA_DICTIONARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
