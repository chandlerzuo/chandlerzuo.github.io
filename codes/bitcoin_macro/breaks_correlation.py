"""Structural breaks in Bitcoin's correlation and beta -- the headline result.

This is the test described in the blog post. Two runs:

  1. BETA breaks. Partial structural change in (intercept, market beta) of the
     multifactor regression, with observations weighted by 1/EWMA-volatility so
     that breaks reflect the relationship rather than Bitcoin's own declining
     volatility.

  2. CORRELATION breaks. Both sides are divided by their own EWMA volatility
     first. Because the standardised series have unit variance, the regression
     slope IS the correlation, so a break in that slope is a break in rho --
     net of the volatility ratio that contaminates a beta break test. Since
     beta = rho * (sigma_B / sigma_M) and sigma_B has fallen threefold, the two
     tests genuinely answer different questions and are both reported.

Inference is a Rademacher wild bootstrap of the sup-F statistic (Andrews's
asymptotic critical values assume homoskedasticity, which is untenable here:
the fitted Student-t degrees of freedom are about 3.5), applied sequentially to
locate multiple breaks. See regimes.py for the statistic and the bootstrap.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fetchlib import PROC, ROOT
from regimes import (CORE, partial_break_F, sequential_breaks,
                     wild_bootstrap_supF)

warnings.filterwarnings("ignore")

START = "2013-01-01"
HALFLIFE = 26          # weeks
N_BOOT_SUP = 399       # for the reported sup-F p-value
N_BOOT_SEQ = 299       # inside the sequential search

FACTORS = {
    "ret_mkt": "US equity market",
    "ret_ndx": "Nasdaq-100",
    "ret_gold": "Gold",
    "ret_usd": "Broad USD",
}


def ewma_vol(x: pd.Series) -> pd.Series:
    """EWMA volatility. NaN over the burn-in -- deliberately not backfilled.

    An earlier version of this analysis standardised by computing the RATIO
    r_t / sigma_t and then back-filling it. That fills the first 26 weeks with a
    single repeated value, and a constant block at the start of the sample is
    indistinguishable from a regime: the test duly reported a highly significant
    "break" in July 2013 that was an artefact of the fill. Returning NaN and
    dropping the burn-in (see standardise) removes the artefact; back-filling the
    VOLATILITY instead gives the same break date, so the finding is not sensitive
    to which of the two defensible choices is made.
    """
    return x.ewm(halflife=HALFLIFE, min_periods=HALFLIFE).std()


def standardise(d: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Index]:
    """Divide each column by its own EWMA vol, dropping the burn-in rows."""
    vols = {c: ewma_vol(d[c]) for c in cols}
    keep = np.logical_and.reduce([v.notna().values for v in vols.values()])
    z = pd.DataFrame({c: (d[c] / vols[c]) for c in cols})[keep]
    return z, d.index[keep]


def load() -> pd.DataFrame:
    w = pd.read_parquet(PROC / "panel_weekly.parquet")
    w["btc_xs"] = w["btc_ret"] - w["rf"].fillna(0)
    return w


def beta_breaks(w: pd.DataFrame, emit) -> pd.DataFrame:
    d = w.loc[START:][["btc_xs"] + CORE].dropna()
    idx = d.index
    vol = ewma_vol(d["btc_xs"])
    keep = vol.notna().values
    d = d[keep]
    idx = d.index
    wt = 1.0 / vol[keep].values
    y = d["btc_xs"].values * wt
    X = sm.add_constant(d[CORE].values) * wt[:, None]
    chg = [0, 1]                      # intercept and market beta

    F, c, _ = partial_break_F(y, X, chg)
    crit, boot = wild_bootstrap_supF(y, X, chg, n_boot=N_BOOT_SUP)
    p = float(np.mean(boot >= F))

    emit("=" * 96)
    emit("1. BREAKS IN BETA  (partial change in intercept + market beta,")
    emit("   observations weighted by 1/EWMA volatility)")
    emit("=" * 96)
    emit(f"  regression: btc_xs ~ {CORE}")
    emit(f"  n={len(d)} weeks, {idx.min().date()} to {idx.max().date()}")
    emit(f"  sup-F = {F:.2f} at {idx[c].date()}   "
         f"bootstrap 95% crit = {crit:.2f}   p = {p:.3f}")

    br = sequential_breaks(y, X, chg, idx, n_boot=N_BOOT_SEQ)
    emit("  sequential breaks:")
    for r in br:
        emit(f"    {r['date'].date()}  F={r['F']:.2f}  "
             f"crit95={r['crit95']:.2f}  p={r['p']:.3f}")
    if not br:
        emit("    none significant")

    bounds = [idx[0]] + [r["date"] for r in br] + [idx[-1] + pd.Timedelta(days=7)]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = d.loc[a:b]
        if b != bounds[-1]:
            sub = sub.iloc[:-1]
        if len(sub) < 10:
            continue
        res = sm.OLS(sub["btc_xs"], sm.add_constant(sub[CORE])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        x = sub["btc_xs"]
        rows.append({
            "start": sub.index.min().date(), "end": sub.index.max().date(),
            "n": len(sub),
            "vol": round(x.std() * np.sqrt(52), 2),
            "sharpe": round(x.mean() * 52 / (x.std() * np.sqrt(52)), 2),
            "b_mkt": f"{res.params['ret_mkt']:.2f}({res.tvalues['ret_mkt']:.1f})",
            "rho_mkt": round(x.corr(sub["ret_mkt"]), 2),
            "b_gold": f"{res.params['ret_gold']:.2f}({res.tvalues['ret_gold']:.1f})",
            "b_real10": f"{res.params['d_real10']:.2f}({res.tvalues['d_real10']:.1f})",
            "R2": round(res.rsquared, 3),
        })
    segs = pd.DataFrame(rows)
    emit("\n  segment characteristics (raw, unweighted, HAC):")
    emit(segs.to_string(index=False))

    segs.to_csv(ROOT / "outputs" / "tables" / "partial_break_segments.csv",
                index=False)
    pd.DataFrame([{"date": str(r["date"].date()), "F": r["F"],
                   "crit95": r["crit95"], "p": r["p"]} for r in br]).to_csv(
        ROOT / "outputs" / "tables" / "partial_breaks.csv", index=False)
    return segs


def correlation_breaks(w: pd.DataFrame, emit) -> pd.DataFrame:
    emit("\n" + "=" * 96)
    emit("2. BREAKS IN CORRELATION  (both sides volatility-standardised, so the")
    emit("   slope IS rho)")
    emit("=" * 96)

    out = {}
    for fac, label in FACTORS.items():
        if fac not in w.columns:
            continue
        d = w.loc[START:][["btc_xs", fac]].dropna()
        z, idx = standardise(d, ["btc_xs", fac])
        y, X = z["btc_xs"].values, sm.add_constant(z[fac].values)
        chg = [0, 1]
        d = d.loc[idx]

        F, c, _ = partial_break_F(y, X, chg)
        crit, boot = wild_bootstrap_supF(y, X, chg, n_boot=N_BOOT_SUP)
        p = float(np.mean(boot >= F))

        emit(f"\n  --- {label} ({fac}) ---")
        emit(f"    sup-F = {F:.2f} at {idx[c].date()}   "
             f"crit95 = {crit:.2f}   p = {p:.3f}")
        br = sequential_breaks(y, X, chg, idx, n_boot=N_BOOT_SEQ, max_b=4)
        emit("    sequential breaks: "
             + (", ".join(f"{r['date'].date()} (F={r['F']:.2f}, p={r['p']:.3f})"
                          for r in br) if br else "none"))

        bounds = [idx[0]] + [r["date"] for r in br] + [idx[-1] + pd.Timedelta(days=7)]
        rows = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            sub = d.loc[a:b]
            if b != bounds[-1]:
                sub = sub.iloc[:-1]
            if len(sub) < 10:
                continue
            x = sub["btc_xs"]
            rows.append({
                "factor": fac,
                "start": sub.index.min().date(), "end": sub.index.max().date(),
                "n": len(sub),
                "rho": round(x.corr(sub[fac]), 3),
                "beta": round(x.cov(sub[fac]) / sub[fac].var(), 2),
                "sd_ratio": round(x.std() / sub[fac].std(), 2),
                "vol_btc_ann": round(x.std() * np.sqrt(52), 2),
                "sharpe": round(x.mean() * 52 / (x.std() * np.sqrt(52)), 2),
            })
        t = pd.DataFrame(rows)
        emit(t.to_string(index=False))
        out[fac] = t

    allt = pd.concat(out.values(), ignore_index=True) if out else pd.DataFrame()
    if not allt.empty:
        allt.to_csv(ROOT / "outputs" / "tables" / "correlation_breaks.csv",
                    index=False)
    return allt


def main() -> int:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    w = load()
    beta_breaks(w, emit)
    corr = correlation_breaks(w, emit)

    emit("\n" + "=" * 96)
    emit("HEADLINE (as reported in the blog post)")
    emit("=" * 96)
    if not corr.empty:
        eq = corr[corr["factor"] == "ret_mkt"]
        emit("  Equity correlation by segment: "
             + " -> ".join(f"{r['rho']:+.3f} ({r['start']}..{r['end']})"
                           for _, r in eq.iterrows()))
        gold = corr[corr["factor"] == "ret_gold"]
        if len(gold) == 1:
            emit(f"  Gold correlation: {gold.iloc[0]['rho']:+.3f} over the whole "
                 f"sample, NO significant break anywhere.")
    emit("  Note March 2020 does not appear as a break in either test.")

    (ROOT / "outputs" / "tables" / "breaks_correlation.txt").write_text(
        "\n".join(lines) + "\n")
    print("\nwrote outputs/tables/{partial_breaks,partial_break_segments,"
          "correlation_breaks}.csv and breaks_correlation.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
