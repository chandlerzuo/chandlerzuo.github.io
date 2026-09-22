"""Crypto-native factors (Layer C of the plan's taxonomy).

Purpose: REPORT.md section 11 item 4 -- the largest gap between plan and
execution. Without these, a macro loading may simply be absorbing a
crypto-idiosyncratic shock that happened to land near a macro event (FTX in
Nov-2022 sits inside the tightening regime; Luna in May-2022 likewise).

Sources, all keyless:
  CoinMetrics community  13 altcoin prices + market caps; USDT/USDC supply
  Binance USD-M futures  BTCUSDT perpetual funding rate (2019-09 onward)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from fetchlib import PROC, fetch

# sol is 403 on the community tier; these 13 are all verified 200.
ALTS = ["eth", "ltc", "xrp", "bch", "ada", "doge", "bnb", "link",
        "xlm", "etc", "zec", "dash", "xmr"]
STABLES = ["usdt", "usdc"]

CM = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
      "?assets={a}&metrics={m}&frequency=1d&page_size=10000&format=csv"
      "&start_time=2012-01-01")


def cm_series(asset: str, metrics: str) -> pd.DataFrame:
    path = fetch(CM.format(a=asset, m=metrics), f"cm_{asset}_{metrics.replace(',', '_')}.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    return df.drop(columns=["asset", "time"]).set_index("date").sort_index()


def binance_funding() -> pd.Series:
    """BTCUSDT perpetual funding rate, paged backwards from launch.

    Funding is the cleanest available proxy for crypto-native leverage demand:
    persistently positive funding means levered longs are paying to stay long.
    """
    out, start = [], 1568000000000          # ~2019-09, contract launch
    for i in range(60):
        url = ("https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT"
               f"&startTime={start}&limit=1000")
        try:
            p = fetch(url, f"binance_funding_{i}.json", min_bytes=20)
            rows = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            break
        if not rows:
            break
        out += rows
        last = int(rows[-1]["fundingTime"])
        if last <= start:
            break
        start = last + 1
        time.sleep(0.15)

    if not out:
        return pd.Series(dtype=float, name="funding")
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    # 3 settlements/day -> daily total funding paid
    return (df.set_index("date")["fundingRate"].sort_index()
              .resample("D").sum(min_count=1).rename("funding"))


def main() -> int:
    px, cap = {}, {}
    for a in ALTS:
        try:
            d = cm_series(a, "PriceUSD,CapMrktCurUSD")
        except Exception as exc:  # noqa: BLE001
            print(f"  {a:6s} FAIL {str(exc)[:60]}")
            continue
        s = pd.to_numeric(d.get("PriceUSD"), errors="coerce").dropna()
        px[a] = s[s > 0]
        if "CapMrktCurUSD" in d:
            cap[a] = pd.to_numeric(d["CapMrktCurUSD"], errors="coerce")
        print(f"  {a:6s} n={len(px[a]):5d}  {px[a].index.min().date()} -> "
              f"{px[a].index.max().date()}")
        time.sleep(0.2)

    P, C = pd.DataFrame(px), pd.DataFrame(cap)
    R = np.log(P).diff()

    # ---- CMKT ex-BTC: cap-weighted altcoin market return --------------------
    # Weights lagged one day so the factor is implementable and not
    # contaminated by same-day cap moves.
    W = C.shift(1).reindex(R.index).where(R.notna())
    W = W.div(W.sum(axis=1), axis=0)
    cmkt_ex = (R * W).sum(axis=1, min_count=2).rename("cmkt_ex_btc")

    # ---- CSIZE / CMOM among the available alts ------------------------------
    # Crude versions of Liu-Tsyvinski-Wu's size and momentum factors: with only
    # 13 coins these are indicative, not the published factors, and are labelled
    # as such wherever they appear.
    capr = C.shift(1).reindex(R.index)
    med = capr.median(axis=1)
    small = R.where(capr.lt(med, axis=0)).mean(axis=1)
    large = R.where(capr.ge(med, axis=0)).mean(axis=1)
    csize = (small - large).rename("csize")

    mom = np.log(P).diff(21).shift(1).reindex(R.index)
    mmed = mom.median(axis=1)
    win = R.where(mom.gt(mmed, axis=0)).mean(axis=1)
    lose = R.where(mom.le(mmed, axis=0)).mean(axis=1)
    cmom = (win - lose).rename("cmom")

    # ---- stablecoin supply growth ------------------------------------------
    sup = {}
    for s in STABLES:
        try:
            d = cm_series(s, "SplyCur")
            v = pd.to_numeric(d["SplyCur"], errors="coerce").dropna()
            sup[s] = v[v > 1000]
            print(f"  {s:6s} supply n={len(sup[s])}  {sup[s].index.min().date()} ->")
        except Exception:  # noqa: BLE001
            print(f"  {s} supply FAIL")
    S = pd.DataFrame(sup)
    stbl = S.sum(axis=1, min_count=1)
    stbl_gr = np.log(stbl.where(stbl > 0)).diff().rename("stbl_gr")

    print("  binance funding ...")
    fund = binance_funding()
    if len(fund):
        print(f"  funding n={len(fund)}  {fund.index.min().date()} -> "
              f"{fund.index.max().date()}")

    out = pd.concat([cmkt_ex, csize, cmom, stbl_gr, fund,
                     R["eth"].rename("ret_eth_cm"),
                     stbl.rename("stbl_supply")], axis=1)
    out.to_parquet(PROC / "crypto_native.parquet")
    R.to_parquet(PROC / "alt_returns.parquet")
    print(f"\nwrote crypto_native.parquet {out.shape}, alt_returns.parquet {R.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
