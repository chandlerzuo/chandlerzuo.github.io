"""Phase 2: static factor models and the three-thesis horse race.

Deliberately runs before any time-varying machinery so that later, fancier
estimates have a transparent baseline to be checked against. If a Kalman filter
disagrees with a clean subsample OLS on the sign of a major beta, the filter is
wrong until proven otherwise.

Inference: Newey-West HAC throughout. Weekly (Friday) is the primary frequency
because 24/7 BTC trading against a 6.5-hour equity session attenuates daily
betas; the daily Dimson (lead-lag) specification quantifies how much.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fetchlib import PROC, ROOT

# The three competing narratives, as explicit factor sets.
THESES: dict[str, list[str]] = {
    "liquidity_realrate": ["d_real10", "d_be10", "net_liq_gr", "d_nom2"],
    "risk_asset_tech":    ["ret_mkt", "tech_tilt", "d_livol", "d_credit_baa"],
    "digital_gold":       ["ret_gold", "ret_usd", "d_be10", "ret_ust10y"],
}

REGIMES_PRIOR: list[tuple[str, str, str]] = [
    ("2013 retail cycle",      "2013-01-01", "2014-12-31"),
    ("2015-16 recovery",       "2015-01-01", "2016-12-31"),
    ("2017-18 ICO boom/bust",  "2017-01-01", "2018-12-31"),
    ("2019-pre-COVID",         "2019-01-01", "2020-02-19"),
    ("COVID crash",            "2020-02-20", "2020-04-30"),
    ("2020-21 liquidity era",  "2020-05-01", "2021-11-09"),
    ("2022 tightening/credit", "2021-11-10", "2022-12-31"),
    ("2023 banking stress",    "2023-01-01", "2023-12-31"),
    ("2024-26 ETF era",        "2024-01-01", "2026-12-31"),
]


def hac_lags(n: int) -> int:
    """Newey-West bandwidth, Newey-West (1994) plug-in rule of thumb."""
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def load(freq: str = "W") -> pd.DataFrame:
    f = "panel_weekly.parquet" if freq == "W" else "panel_daily.parquet"
    d = pd.read_parquet(PROC / f)

    # Net liquidity enters as a growth rate, not a level or a raw difference,
    # so its scale is comparable across a balance sheet that grew ~10x.
    if "net_liquidity" in d:
        nl = d["net_liquidity"].replace(0, np.nan)
        d["net_liq_gr"] = np.log(nl.where(nl > 0)).diff()

    d["btc_xs"] = d["btc_ret"] - d["rf"].fillna(0.0)
    return d


def regress(y: pd.Series, X: pd.DataFrame, label: str) -> dict | None:
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 30:
        return None
    Xc = sm.add_constant(df[X.columns])
    res = sm.OLS(df["y"], Xc).fit(cov_type="HAC",
                                 cov_kwds={"maxlags": hac_lags(len(df))})
    out = {"model": label, "n": len(df), "r2": res.rsquared,
           "r2_adj": res.rsquared_adj}
    for k in Xc.columns:
        out[f"b_{k}"] = res.params[k]
        out[f"t_{k}"] = res.tvalues[k]
    return out


def fmt(v: float, t: float, scale: float = 1.0) -> str:
    stars = "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""
    return f"{v * scale:7.3f}{stars:<3s}({t:5.2f})"


def horse_race(d: pd.DataFrame, periods: list[tuple[str, str, str]],
               per_year: float) -> pd.DataFrame:
    """Compare the three theses on the same sample: who explains BTC variance?"""
    rows = []
    for name, a, b in periods:
        sub = d.loc[a:b]
        if len(sub.dropna(subset=["btc_xs"])) < 30:
            continue
        rec = {"period": name, "start": a, "end": b}
        for thesis, cols in THESES.items():
            have = [c for c in cols if c in sub.columns and sub[c].notna().sum() > 25]
            if not have:
                rec[f"r2_{thesis}"] = np.nan
                continue
            r = regress(sub["btc_xs"], sub[have], thesis)
            rec[f"r2_{thesis}"] = r["r2"] if r else np.nan
            rec[f"n_{thesis}"] = r["n"] if r else 0
        # combined model, for how much the theses overlap
        allc = [c for c in set(sum(THESES.values(), [])) if c in sub.columns
                and sub[c].notna().sum() > 25]
        r = regress(sub["btc_xs"], sub[allc], "all")
        rec["r2_combined"] = r["r2"] if r else np.nan
        x = sub["btc_xs"].dropna()
        rec["vol_ann"] = x.std() * np.sqrt(per_year)
        rec["ret_ann"] = x.mean() * per_year
        rec["sharpe"] = rec["ret_ann"] / rec["vol_ann"] if rec["vol_ann"] else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def dimson(d: pd.DataFrame, factor: str = "ret_mkt") -> pd.DataFrame:
    """Daily beta with lead/lag terms, to size the non-synchronous-trading bias.

    BTC trades continuously; US equities do not. The sum of the lead, contemporaneous
    and lagged coefficients is the bias-corrected beta. If the corrected beta is
    materially larger than the contemporaneous one, calendar-day betas understate
    true exposure and the weekly frequency is doing necessary work.
    """
    # Restrict to days the factor actually trades BEFORE shifting. On a calendar
    # index a .shift() crosses weekends, so a Friday's "lead" is Saturday (NaN)
    # and a Monday's "lag" is Sunday (NaN); the subsequent dropna then silently
    # discards every Friday and Monday -- roughly 40% of the sample, and not at
    # random. Reindexing to trading days first makes the shift a true lead/lag.
    td = d[d[factor].notna()].copy()

    rows = []
    for name, a, b in REGIMES_PRIOR:
        sub = td.loc[a:b]
        X = pd.DataFrame({
            "lead": sub[factor].shift(-1),
            "contemp": sub[factor],
            "lag": sub[factor].shift(1),
        }, index=sub.index)
        r = regress(sub["btc_xs"], X, name)
        if not r:
            continue
        tot = r["b_lead"] + r["b_contemp"] + r["b_lag"]
        rows.append({"period": name, "n": r["n"], "beta_contemp": r["b_contemp"],
                     "t_contemp": r["t_contemp"], "beta_lead": r["b_lead"],
                     "beta_lag": r["b_lag"], "beta_dimson_sum": tot,
                     "ratio_sum_to_contemp": tot / r["b_contemp"]
                     if r["b_contemp"] else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    out_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    w = load("W")
    dd = load("D")

    # ---------------------------------------------------------------- 1
    emit("=" * 100)
    emit("1. FULL-SAMPLE AND SPLIT-SAMPLE SPANNING REGRESSIONS (weekly, HAC)")
    emit("   Does a liquid-asset portfolio span BTC? alpha in % per week.")
    emit("=" * 100)

    specs = {
        "CAPM": ["ret_mkt"],
        "FF5": ["ret_mkt", "ff_smb", "ff_hml", "ff_rmw", "ff_cma"],
        "FF5+Mom": ["ret_mkt", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_mom"],
        "Mkt+Tech": ["ret_mkt", "tech_tilt"],
        "Mkt+Tech+Dur+Cred+USD": ["ret_mkt", "tech_tilt", "ret_ust10y",
                                  "d_credit_baa", "ret_usd"],
    }
    samples = [("full 2013+", "2013-01-01", "2026-12-31"),
               ("early 2013-2019", "2013-01-01", "2019-12-31"),
               ("late 2020+", "2020-01-01", "2026-12-31")]

    for sname, a, b in samples:
        emit(f"\n--- {sname} ---")
        emit(f"{'spec':24s} {'n':>4s} {'R2':>6s}  {'alpha%/wk':>18s}  betas")
        for spec, cols in specs.items():
            have = [c for c in cols if c in w.columns]
            r = regress(w.loc[a:b, "btc_xs"], w.loc[a:b, have], spec)
            if not r:
                continue
            bs = "  ".join(f"{c}={r[f'b_{c}']:.2f}({r[f't_{c}']:.1f})" for c in have)
            emit(f"{spec:24s} {r['n']:4d} {r['r2']:6.3f}  "
                 f"{fmt(r['b_const'], r['t_const'], 100):>18s}  {bs}")

    # ---------------------------------------------------------------- 2
    emit("\n" + "=" * 100)
    emit("2. MACRO SHOCK BETAS, UNIVARIATE BY PRIOR REGIME (weekly, HAC)")
    emit("   Coefficient on a 1pp shock (rates/spreads) or 1 log unit (vol/FX).")
    emit("=" * 100)
    macro = ["d_real10", "d_be10", "d_nom2", "d_slope", "d_credit_baa",
             "d_livol", "ret_usd", "net_liq_gr", "d_termprem"]
    tab = {}
    for name, a, b in REGIMES_PRIOR:
        col = {}
        for m in macro:
            if m not in w.columns:
                continue
            r = regress(w.loc[a:b, "btc_xs"], w.loc[a:b, [m]], m)
            col[m] = f"{r[f'b_{m}']:6.2f}({r[f't_{m}']:5.2f})" if r else ""
        tab[name] = col
    emit(pd.DataFrame(tab).to_string())

    # ---------------------------------------------------------------- 3
    emit("\n" + "=" * 100)
    emit("3. THESIS HORSE RACE: R-squared by regime (weekly)")
    emit("   liquidity/real-rate vs risk-asset/tech vs digital-gold")
    emit("=" * 100)
    hr = horse_race(w, REGIMES_PRIOR, 52.0)
    show = hr[["period", "vol_ann", "ret_ann", "sharpe", "r2_liquidity_realrate",
               "r2_risk_asset_tech", "r2_digital_gold", "r2_combined"]].copy()
    show.columns = ["period", "vol", "ret", "sharpe", "R2_liq_rate",
                    "R2_risk_tech", "R2_gold", "R2_all"]
    emit(show.round(3).to_string(index=False))
    hr.to_csv(ROOT / "outputs" / "tables" / "horse_race_prior_regimes.csv", index=False)

    # ---------------------------------------------------------------- 4
    emit("\n" + "=" * 100)
    emit("4. NON-SYNCHRONOUS TRADING: Dimson lead-lag betas vs market (daily)")
    emit("   ratio>1 means calendar-day betas UNDERSTATE true exposure.")
    emit("=" * 100)
    dm = dimson(dd)
    emit(dm.round(3).to_string(index=False))
    dm.to_csv(ROOT / "outputs" / "tables" / "dimson_betas.csv", index=False)

    (ROOT / "outputs" / "tables" / "phase2_static.txt").write_text("\n".join(out_lines) + "\n")
    print(f"\nwrote outputs/tables/phase2_static.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
