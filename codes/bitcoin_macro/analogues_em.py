"""Follow-up 5: extend the analogue panel with emerging-market equity indices.

REPORT.md section 11 item 5. The daily panel's weakness was composition: 49 of
its 65 assets are DIVERSIFIED US industry portfolios, which structurally cannot
reach Bitcoin's volatility, so "no credible analogue" was partly a statement
about the panel rather than about Bitcoin. Emerging-market equity indices are the
obvious missing class -- high volatility, weak developed-market correlation in
calm periods, violent crisis behaviour -- and the OECD share-price series already
fetched reach back to 1960-1970.

Cost of admission: OECD indices are MONTHLY only. So this runs the whole matching
exercise at monthly frequency, which is a genuinely different test rather than an
extension of the daily one:
  + emerging markets, and 1960s-80s history the daily panel cannot reach
  - short regimes become untestable (a 14-month regime gives 14 observations),
    so only regimes of >= 24 months are matched, and correlation features are
    noisy even then. Flagged in the output rather than glossed.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from analogues import (FEATURES, dtw_distance, industry49, mahalanobis_setup,
                       mdist_to, resample_path, window_features)
from fetchlib import PROC, ROOT

warnings.filterwarnings("ignore")

MONTHS_PER_YEAR = 12
MIN_MONTHS = 24
MIN_CORR_N = 20   # monthly windows are short; see _rowwise_corr docstring

# Regimes long enough to characterise at monthly frequency, plus two aggregates.
TARGETS = [
    ("R2 2014-16 bear/recovery", "2013-12-07", "2016-12-31"),
    ("R3 2017-18 ICO cycle",     "2017-01-01", "2018-12-31"),
    ("R7 2023-25 ETF era",       "2023-01-01", "2025-07-18"),
    ("BTC 2013-2019 (pre-macro)", "2013-01-01", "2019-12-31"),
    ("BTC 2020-2026 (macro era)", "2020-01-01", "2026-09-30"),
    ("BTC full 2013-2026",        "2013-01-01", "2026-09-30"),
]

COUNTRY = {"mx": "Mexico", "kr": "Korea", "br": "Brazil", "tr": "Turkey",
           "cn": "China", "ru": "Russia", "id": "Indonesia", "za": "South Africa",
           "jp": "Japan", "us": "United States", "de": "Germany",
           "gb": "United Kingdom"}


def monthly_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    metals = pd.read_parquet(PROC / "metals_daily.parquet")
    etf = pd.read_parquet(PROC / "etf_nav_daily.parquet")
    oecd = pd.read_parquet(PROC / "oecd_equity_monthly.parquet")
    fr = (pd.read_parquet(PROC / "fred_long.parquet")
            .pivot(index="date", columns="series", values="value").sort_index())
    ffm = pd.read_parquet(PROC / "ff5_monthly.parquet")

    cand: dict[str, pd.Series] = {}

    def add(name, s):
        s = s.dropna()
        if len(s) >= MIN_MONTHS + 6:
            cand[name] = s

    # --- emerging + developed market equity indices (the point of this script)
    for cc, nm in COUNTRY.items():
        if cc in oecd:
            p = oecd[cc].dropna()
            p = p[p > 0]
            add(f"em_{cc}_{nm.replace(' ', '')}", np.log(p).diff())

    # --- daily assets aggregated to month end
    for m in ("gold", "silver", "platinum", "palladium"):
        if m in metals:
            p = metals[m].dropna().resample("ME").last()
            add(f"metal_{m}", np.log(p[p > 0]).diff())
    for sid, nm in (("NASDAQCOM", "idx_nasdaq_comp"), ("NASDAQ100", "idx_nasdaq100"),
                    ("NIKKEI225", "idx_nikkei"), ("DCOILWTICO", "cmdty_wti"),
                    ("WTISPLC", "cmdty_wti_long"), ("DHHNGSP", "cmdty_natgas"),
                    ("PURANUSDM", "cmdty_uranium"), ("PCOPPUSDM", "cmdty_copper")):
        if sid in fr:
            p = fr[sid].dropna().resample("ME").last()
            add(nm, np.log(p[p > 0]).diff())
    for tic in ("spy", "gld", "xlk", "xle", "xlf"):
        if tic in etf:
            p = etf[tic].dropna().resample("ME").last()
            add(f"etf_{tic}", np.log(p[p > 0]).diff())

    # --- US industry portfolios, compounded to monthly
    ind = industry49()
    indm = (1 + ind).resample("ME").prod() - 1
    for c in indm.columns:
        add(f"ind_{c}", indm[c])

    assets = pd.DataFrame(cand)
    assets.index = assets.index.to_period("M").to_timestamp("M")
    assets = assets.groupby(level=0).last()

    # --- benchmarks for the correlation features
    ffm2 = ffm.copy()
    ffm2.index = ffm2.index.to_period("M").to_timestamp("M")
    goldm = metals["gold"].dropna().resample("ME").last()
    usd = None
    for sid in ("DTWEXM", "DTWEXBGS"):
        if sid in fr:
            u = np.log(fr[sid].dropna().resample("ME").last()).diff()
            usd = u if usd is None else u.combine_first(usd)
    bench = pd.DataFrame({
        "eq": ffm2["Mkt-RF"],
        "rf": ffm2["RF"],
        "gold": np.log(goldm[goldm > 0]).diff(),
        "d10y": fr["DGS10"].dropna().resample("ME").last().diff()
        if "DGS10" in fr else np.nan,
        "usd": usd,
    })
    bench.index = bench.index.to_period("M").to_timestamp("M")
    bench = bench.groupby(level=0).last()
    return assets, bench


def ref_windows(assets: pd.DataFrame, bench: pd.DataFrame, L: int) -> pd.DataFrame:
    frames = []
    for name in assets.columns:
        s = assets[name].dropna()
        if len(s) < L + 2:
            continue
        starts = np.arange(0, len(s) - L + 1, 1)
        f = window_features(s, bench, MONTHS_PER_YEAR, L, starts,
                            min_corr_n=MIN_CORR_N)
        f["asset"] = name
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[np.isfinite(out["vol_ann"])]


def main() -> int:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    btc = pd.read_parquet(PROC / "panel_daily.parquet")["btc_px"].dropna()
    btcm = np.log(btc.resample("ME").last()).diff().dropna()
    btcm.index = btcm.index.to_period("M").to_timestamp("M")

    assets, bench = monthly_panel()

    emit("=" * 104)
    emit("FOLLOW-UP 5. MONTHLY ANALOGUE PANEL WITH EMERGING MARKETS")
    emit("=" * 104)
    kinds = pd.Series([c.split("_")[0] for c in assets.columns]).value_counts()
    emit(f"  assets: {len(assets.columns)}  ({', '.join(f'{k}={v}' for k, v in kinds.items())})")
    emit(f"  monthly span: {assets.index.min().date()} to {assets.index.max().date()}")
    em_cols = [c for c in assets.columns if c.startswith("em_")]
    emit(f"  emerging/developed country indices added: {len(em_cols)}")
    for c in sorted(em_cols):
        s = assets[c].dropna()
        emit(f"    {c:28s} n={len(s):4d}  {s.index.min().date()} -> "
             f"{s.index.max().date()}  vol={s.std() * np.sqrt(12):.2f}")
    emit(f"  BTC monthly obs: {len(btcm)}  "
         f"{btcm.index.min().date()} -> {btcm.index.max().date()}")
    emit("\n  CAVEAT: with 24-81 monthly observations per window, correlation")
    emit("  features carry large standard errors. Rankings below are indicative.")

    rows, nulls = [], []
    for name, a, b in TARGETS:
        y = btcm.loc[a:b]
        L = len(y)
        if L < MIN_MONTHS:
            emit(f"\n--- {name}: only {L} months, below the {MIN_MONTHS}-month "
                 f"floor, skipped")
            continue

        ref = ref_windows(assets, bench, L)
        if ref.empty:
            continue
        btc_f = window_features(y, bench, MONTHS_PER_YEAR, L, np.array([0]),
                               min_corr_n=MIN_CORR_N).iloc[0]
        feats = [f for f in FEATURES if ref[f].notna().mean() > 0.6
                 and np.isfinite(btc_f.get(f, np.nan))]
        mu, sd, Ci, ok = mahalanobis_setup(ref, feats)
        refv = ref[ok].reset_index(drop=True)
        refv["d_maha"] = mdist_to(refv, btc_f, mu, sd, Ci, feats)

        top = refv.nsmallest(40, "d_maha").copy()
        bpath = resample_path(y)
        bstd = ((y - y.mean()) / y.std()).values
        dt, ws = [], []
        for _, r in top.iterrows():
            s = assets[r["asset"]].loc[r["start"]:r["end"]].dropna()
            dt.append(dtw_distance(bpath, resample_path(s)))
            ws.append(wasserstein_distance(bstd, ((s - s.mean()) / s.std()).values))
        top["d_dtw"], top["d_wass"] = dt, ws
        for c in ("d_maha", "d_dtw", "d_wass"):
            top[f"rk_{c}"] = top[c].rank()
        top["rank_sum"] = top[[f"rk_{c}" for c in
                              ("d_maha", "d_dtw", "d_wass")]].sum(axis=1)

        # same null as the daily study: pseudo-targets scan the panel too
        pool = refv.sample(min(300, len(refv)), random_state=3)
        bn = []
        for _, pr in pool.iterrows():
            other = refv[refv["asset"] != pr["asset"]]
            if other.empty:
                continue
            d = mdist_to(other, pr, mu, sd, Ci, feats)
            if np.isfinite(d).any():
                bn.append(np.nanmin(d))
        bn = np.array(bn)
        best = float(top["d_maha"].min())
        p = float(np.mean(bn <= best)) if len(bn) else np.nan

        emit(f"\n--- {name}  [{a} .. {b}]  window = {L} months ---")
        emit(f"    {len(refv)} reference windows from {refv['asset'].nunique()} "
             f"assets; features = {len(feats)}")
        emit(f"    BTC: vol={btc_f['vol_ann']:.2f} sharpe={btc_f['sharpe']:+.2f} "
             f"skew={btc_f['skew']:+.2f} maxDD={btc_f['max_dd']:.2f} "
             f"rho_eq={btc_f['rho_eq']:+.2f}")
        emit(f"    best distance = {best:.2f};  null median {np.median(bn):.2f}, "
             f"5th pct {np.percentile(bn, 5):.2f};  p = {p:.3f}")
        emit("    -> " + ("BTC IS unusually well matched" if p < 0.05 else
                          "no credible analogue (not closer than chance)"))
        emit(f"      {'asset':30s} {'window':20s} {'maha':>6s} {'dtw':>6s} "
             f"{'wass':>6s} {'vol':>6s} {'shrp':>6s} {'rhoEq':>6s}")
        for _, r in top.nsmallest(8, "rank_sum").iterrows():
            emit(f"      {r['asset']:30s} "
                 f"{str(r['start'].date())[:7] + '..' + str(r['end'].date())[:7]:20s} "
                 f"{r['d_maha']:6.2f} {r['d_dtw']:6.3f} {r['d_wass']:6.3f} "
                 f"{r['vol_ann']:6.2f} {r['sharpe']:+6.2f} {r['rho_eq']:+6.2f}")
            rows.append({"target": name, **r[["asset", "start", "end", "d_maha",
                                             "d_dtw", "d_wass", "vol_ann",
                                             "sharpe", "rho_eq"]].to_dict()})

        # does adding EM change the answer at all?
        em_in_top = [r["asset"] for _, r in top.nsmallest(20, "rank_sum").iterrows()
                     if r["asset"].startswith("em_")]
        emit(f"    EM/country indices in top 20: {len(em_in_top)}"
             + (f"  ({', '.join(sorted(set(em_in_top))[:5])})" if em_in_top else ""))
        nulls.append({"target": name, "months": L, "best": best,
                     "null_median": float(np.median(bn)), "p": p,
                     "n_em_top20": len(em_in_top)})

    emit("\n" + "=" * 104)
    emit("DOES ADDING EMERGING MARKETS CHANGE THE CONCLUSION?")
    emit("=" * 104)
    nt = pd.DataFrame(nulls)
    if not nt.empty:
        emit(nt.round(3).to_string(index=False))
        emit("")
        if (nt["p"] >= 0.05).all():
            emit("  No. Even with emerging-market indices and 1960s-80s history in")
            emit("  the panel, Bitcoin is never matched closer than chance. The")
            emit("  daily study's conclusion was not an artefact of a US-equity-")
            emit("  heavy panel.")
        else:
            emit("  Yes -- at least one target is now matched better than chance.")
        emit(f"  EM indices reach the top 20 in "
             f"{int((nt['n_em_top20'] > 0).sum())} of {len(nt)} targets.")

    if rows:
        pd.DataFrame(rows).to_csv(
            ROOT / "outputs" / "tables" / "analogue_matches_monthly_em.csv",
            index=False)
    if not nt.empty:
        nt.to_csv(ROOT / "outputs" / "tables" / "analogue_em_null_tests.csv",
                 index=False)
    (ROOT / "outputs" / "tables" / "followup5_em_analogues.txt").write_text(
        "\n".join(lines) + "\n")
    print("\nwrote outputs/tables/followup5_em_analogues.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
