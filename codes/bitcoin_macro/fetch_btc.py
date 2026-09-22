"""Bitcoin price history from independent sources, with cross-source QA.

Primary   : CoinMetrics community (2010-07-18 -> today, one request, no key)
Cross-check: Bitstamp OHLC (2011-09 -> today, paged), FRED CBBTCUSD (2014-12 ->)

The point of pulling three is not redundancy for its own sake: pre-2014 BTC
prices are dominated by Mt. Gox, where wash trading is documented, so the study
needs an explicit, quantified view of where sources disagree rather than an
assumption that any single feed is clean.
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

from fetchlib import PROC, RAW, fetch, fred

CM_URL = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
          "?assets=btc&metrics={metrics}&frequency=1d&page_size=10000"
          "&format=csv&start_time=2009-01-01")
# Community tier exposes 31 metrics; VtyDayRet30d and CapRealUSD are 403 (paid).
# MVRV is kept because it is a widely used on-chain cycle indicator and gives an
# independent, non-macro regime signal to race against the macro-beta regimes.
CM_METRICS = ("PriceUSD,CapMrktCurUSD,CapMrktEstUSD,CapMVRVCur,"
              "AdrActCnt,TxCnt,SplyCur,HashRate")


def coinmetrics() -> pd.DataFrame:
    path = fetch(CM_URL.format(metrics=CM_METRICS), "btc_coinmetrics.csv", force=True)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df = df.drop(columns=["asset", "time"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def bitstamp(step: int = 86400, pages: int = 8) -> pd.DataFrame:
    """Page backwards through Bitstamp's 1000-row-capped OHLC endpoint."""
    frames, start = [], 1315000000  # ~2011-09, the start of their history
    for i in range(pages):
        url = (f"https://www.bitstamp.net/api/v2/ohlc/btcusd/"
               f"?step={step}&limit=1000&start={start}")
        path = fetch(url, f"btc_bitstamp_{step}_{i}.json", force=True, min_bytes=100)
        rows = json.loads(path.read_text())["data"]["ohlc"]
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        last = int(rows[-1]["timestamp"])
        if last <= start:
            break
        start = last + step
        time.sleep(0.4)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.drop_duplicates("date", keep="last")
            .set_index("date").sort_index()[["open", "high", "low", "close", "volume"]])
    return df[df["close"] > 0]


def main() -> int:
    cm = coinmetrics()
    print(f"coinmetrics  n={len(cm):5d}  {cm.index.min().date()} -> {cm.index.max().date()}")
    print(f"             cols={list(cm.columns)}")

    bs = bitstamp()
    print(f"bitstamp     n={len(bs):5d}  {bs.index.min().date()} -> {bs.index.max().date()}")

    cb = pd.read_csv(fred("CBBTCUSD"))
    cb["date"] = pd.to_datetime(cb[cb.columns[0]])
    cb = (cb.set_index("date")["CBBTCUSD"].pipe(pd.to_numeric, errors="coerce").dropna())
    print(f"fred/coinbase n={len(cb):4d}  {cb.index.min().date()} -> {cb.index.max().date()}")

    # ---- cross-source QA: where do independent feeds actually disagree? ----
    qa = pd.DataFrame({"cm": cm["PriceUSD"], "bs": bs["close"], "cb": cb}).dropna(how="all")
    qa["d_cm_bs"] = (qa["cm"] / qa["bs"] - 1).abs()
    qa["d_cm_cb"] = (qa["cm"] / qa["cb"] - 1).abs()
    qa.to_parquet(PROC / "btc_source_qa.parquet")

    print("\n--- cross-source absolute close disagreement ---")
    for col, label in (("d_cm_bs", "CoinMetrics vs Bitstamp"),
                       ("d_cm_cb", "CoinMetrics vs Coinbase")):
        s = qa[col].dropna()
        if s.empty:
            continue
        print(f"{label:26s} n={len(s):5d}  median={s.median():.4%}  "
              f"p99={s.quantile(.99):.3%}  max={s.max():.2%}  >2%: {(s > .02).sum()} days")

    by_year = qa.groupby(qa.index.year)["d_cm_bs"].median().dropna()
    print("\nmedian CoinMetrics-vs-Bitstamp gap by year (data-quality regime check):")
    print("  " + "  ".join(f"{y}:{v:.3%}" for y, v in by_year.items()))

    out = cm.copy()
    out["close_bitstamp"] = bs["close"]
    out["close_coinbase"] = cb
    out.to_parquet(PROC / "btc_daily.parquet")
    print(f"\nwrote {PROC / 'btc_daily.parquet'} n={len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
