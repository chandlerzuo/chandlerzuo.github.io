"""Follow-up 4: do Bitcoin's macro loadings survive crypto-native controls?

REPORT.md section 11 item 4. The worry is concrete: FTX (Nov-2022) and Luna
(May-2022) both sit inside the 2022 tightening regime, so a "credit spread beta"
or "equity beta" estimated there may be picking up an exchange collapse that
merely coincided with a macro tightening.

CHOICE OF CONTROLS -- this is the whole design, and it is easy to get wrong.

  Included as controls (potential CONFOUNDERS -- crypto-idiosyncratic drivers
  that could coincide with macro events and manufacture a spurious loading):
    stablecoin supply growth, perpetual funding rate (leverage demand),
    ETH/BTC (within-crypto risk appetite), on-chain activity growth,
    MVRV change, and dated event dummies for the major crypto failures.

  EXCLUDED as BAD CONTROLS -- each is a transform of the dependent variable, so
  conditioning on it would drive macro loadings to zero for a mechanical reason
  that says nothing about macro exposure:
    * MVRV change. MVRV = market cap / realised cap, and realised cap moves
      slowly, so d(MVRV) ~ 2x the BTC return. On the 2019+ sample
      corr(btc_xs, d_mvrv) = 0.926 and d_mvrv ALONE gives R2 = 0.858. A first
      version of this script included it and reported macro loadings being
      "killed" -- they were being absorbed by an accounting near-identity.
      The full-sample correlation is only 0.155, which is what hid the problem.
    * ETH/BTC change. Equals r_ETH - r_BTC, i.e. it contains the dependent
      variable with coefficient -1 by construction.
    * Crypto market return ex-BTC (corr 0.637). A sibling asset, not a
      confounder; kept only as a reported diagnostic.

  Remaining endogeneity handled by LAGGING. Funding, on-chain activity and
  stablecoin growth all respond to the same-week price move, so they enter as
  one-week lags: "crypto conditions entering the week", which is predetermined
  with respect to that week's return. Event dummies are exogenous dated facts
  and enter contemporaneously.

  Sample note: perpetual funding begins 2019-09, so requiring it silently cut
  the estimation sample from 657 to 360 weeks. Two specifications are therefore
  reported -- one without funding on the full sample, one with it post-2019.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fetchlib import PROC, ROOT

warnings.filterwarnings("ignore")

REGIMES = [
    ("R2 2014-16", "2013-12-07", "2016-12-31"),
    ("R3 2017-18", "2017-01-01", "2018-12-31"),
    ("R5 2020-21", "2020-02-20", "2021-11-09"),
    ("R6 2022", "2021-11-10", "2022-12-31"),
    ("R7 2023-25", "2023-01-01", "2025-07-18"),
    ("R8 2025-26", "2025-07-19", "2026-09-30"),
    ("full 2014+", "2014-01-01", "2026-09-30"),
]

MACRO = ["ret_mkt", "d_be10", "d_livol", "d_credit_baa", "ret_usd",
         "d_real10", "net_liq_gr"]

# Dated crypto-idiosyncratic shocks. Windows are deliberately short and set from
# the public record, not chosen to fit the data.
EVENTS = {
    "ev_mtgox":   ("2014-02-20", "2014-03-10"),
    "ev_chinaban": ("2017-09-01", "2017-09-18"),
    "ev_chinamine": ("2021-05-17", "2021-06-30"),
    "ev_luna":    ("2022-05-06", "2022-05-23"),
    "ev_3ac":     ("2022-06-10", "2022-07-05"),
    "ev_ftx":     ("2022-11-04", "2022-11-28"),
    "ev_svb":     ("2023-03-08", "2023-03-20"),
    "ev_etf":     ("2024-01-08", "2024-01-19"),
}


def build() -> pd.DataFrame:
    d = pd.read_parquet(PROC / "panel_daily.parquet")
    cn = pd.read_parquet(PROC / "crypto_native.parquet")
    d = d.join(cn, how="left")

    d["btc_xs"] = d["btc_ret"] - d["rf"].fillna(0)
    nl = d["net_liquidity"].replace(0, np.nan)
    d["net_liq_gr"] = np.log(nl.where(nl > 0)).diff()

    # crypto-native controls, all as changes/growth rates
    d["d_eth_btc"] = np.log(d["eth_btc"].where(d["eth_btc"] > 0)).diff()
    d["hash_gr"] = np.log(d["btc_hashrate"].where(d["btc_hashrate"] > 0)).diff()
    d["adr_gr"] = np.log(d["btc_adract"].where(d["btc_adract"] > 0)).diff()
    d["d_mvrv"] = d["btc_mvrv"].diff()

    for name, (a, b) in EVENTS.items():
        d[name] = 0.0
        d.loc[a:b, name] = 1.0

    rets = ["btc_xs", "btc_ret", "rf", "ret_mkt", "ret_usd", "tech_tilt",
            "ret_gold", "ret_ust10y", "cmkt_ex_btc", "csize", "cmom",
            "d_eth_btc", "hash_gr", "adr_gr", "stbl_gr"]
    chgs = [c for c in d.columns if c.startswith("d_")] + ["net_liq_gr"]
    lvls = ["funding"]
    evs = list(EVENTS)

    w = pd.concat([
        d[[c for c in rets if c in d]].resample("W-FRI").sum(min_count=1),
        d[[c for c in sorted(set(chgs)) if c in d]].resample("W-FRI").sum(min_count=1),
        d[[c for c in lvls if c in d]].resample("W-FRI").sum(min_count=1),
        d[evs].resample("W-FRI").max(),
    ], axis=1)
    w = w.loc[:, ~w.columns.duplicated()]
    # Predetermined versions of the endogenous crypto state variables.
    for c in ("funding", "stbl_gr", "hash_gr", "adr_gr"):
        if c in w:
            w[c + "_l1"] = w[c].shift(1)
    return w


def fit(w: pd.DataFrame, cols: list[str], a: str, b: str):
    sub = w.loc[a:b]
    have = [c for c in cols if c in sub.columns and sub[c].notna().sum() > 20
            and sub[c].std() > 0]
    df = pd.concat([sub["btc_xs"].rename("y"), sub[have]], axis=1).dropna()
    if len(df) < 35 or not have:
        return None, have
    X = sm.add_constant(df[have])
    lags = max(1, int(4 * (len(df) / 100) ** (2 / 9)))
    return sm.OLS(df["y"], X).fit(cov_type="HAC",
                                 cov_kwds={"maxlags": lags}), have


def stars(t):
    return "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""


def main() -> int:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    w = build()
    # Primary spec: no funding (preserves the full 2014+ sample), lagged
    # crypto state, contemporaneous event dummies.
    controls = ["stbl_gr_l1", "hash_gr_l1", "adr_gr_l1"] + list(EVENTS)
    controls_fund = controls + ["funding_l1"]

    emit("=" * 100)
    emit("FOLLOW-UP 4. DO MACRO LOADINGS SURVIVE CRYPTO-NATIVE CONTROLS?")
    emit("  weekly, Newey-West HAC. *** p<1%, ** p<5%, * p<10%")
    emit("  controls: LAGGED stablecoin supply growth, hashrate growth,")
    emit("            active-address growth + 8 exogenous event dummies.")
    emit("  MVRV change, ETH/BTC and crypto-market-ex-BTC are EXCLUDED as bad")
    emit("  controls (transforms of the dependent variable) -- see module docstring.")
    emit("=" * 100)

    rows = []
    for name, a, b in REGIMES:
        m0, _ = fit(w, MACRO, a, b)
        m1, used = fit(w, MACRO + controls, a, b)
        if m0 is None or m1 is None:
            emit(f"\n--- {name}: insufficient data")
            continue
        emit(f"\n--- {name} [{a} .. {b}] ---")
        emit(f"    n={int(m0.nobs)}  R2 macro-only={m0.rsquared:.3f}  "
             f"R2 with controls={m1.rsquared:.3f}  "
             f"(+{m1.rsquared - m0.rsquared:.3f})")
        emit(f"    {'macro var':14s} {'macro-only':>20s} "
             f"{'+ controls':>20s}  verdict")
        for v in MACRO:
            if v not in m0.params or v not in m1.params:
                continue
            b0, t0 = m0.params[v], m0.tvalues[v]
            b1, t1 = m1.params[v], m1.tvalues[v]
            sig0, sig1 = abs(t0) > 1.96, abs(t1) > 1.96
            if sig0 and sig1:
                vd = "survives"
            elif sig0 and not sig1:
                vd = "KILLED"
            elif not sig0 and sig1:
                vd = "appears"
            else:
                vd = "ns both"
            shift = (b1 - b0) / abs(b0) if b0 != 0 else np.nan
            emit(f"    {v:14s} {b0:8.3f}{stars(t0):<3s}(t={t0:5.2f}) "
                 f"{b1:8.3f}{stars(t1):<3s}(t={t1:5.2f})  {vd:>14s}")
            rows.append({"regime": name, "var": v, "b_macro": b0, "t_macro": t0,
                        "b_ctrl": b1, "t_ctrl": t1, "rel_shift": shift,
                        "verdict": vd})

        # which controls actually mattered
        sigc = [(c, m1.params[c], m1.tvalues[c]) for c in used
                if c in controls and c in m1.params and abs(m1.tvalues[c]) > 1.96]
        if sigc:
            emit("    significant crypto-native controls: "
                 + ", ".join(f"{c}={v:+.3f}(t={t:.1f})" for c, v, t in sigc))
        else:
            emit("    significant crypto-native controls: none")

    emit("\n" + "=" * 100)
    emit("SECONDARY SPEC: adding LAGGED perpetual funding (sample starts 2019-09)")
    emit("=" * 100)
    mf0, _ = fit(w, MACRO, "2019-10-01", "2026-09-30")
    mf1, usedf = fit(w, MACRO + controls_fund, "2019-10-01", "2026-09-30")
    if mf0 is not None and mf1 is not None:
        emit(f"  n={int(mf0.nobs)}  R2 macro-only={mf0.rsquared:.3f} -> "
             f"with controls+funding {mf1.rsquared:.3f}")
        for v in MACRO:
            if v in mf0.params and v in mf1.params:
                emit(f"  {v:14s} {mf0.params[v]:8.3f}{stars(mf0.tvalues[v]):<3s}"
                     f"(t={mf0.tvalues[v]:5.2f}) {mf1.params[v]:8.3f}"
                     f"{stars(mf1.tvalues[v]):<3s}(t={mf1.tvalues[v]:5.2f})")
        if "funding_l1" in mf1.params:
            emit(f"  {'funding_l1':14s} {'':>20s} {mf1.params['funding_l1']:8.3f}"
                 f"{stars(mf1.tvalues['funding_l1']):<3s}"
                 f"(t={mf1.tvalues['funding_l1']:5.2f})")

    res = pd.DataFrame(rows)
    emit("\n" + "=" * 100)
    emit("SUMMARY: stability of macro loadings")
    emit("=" * 100)
    if not res.empty:
        vc = res["verdict"].value_counts()
        emit("  across all regime x variable cells: "
             + ", ".join(f"{k}={v}" for k, v in vc.items()))
        killed = res[res["verdict"] == "KILLED"]
        if killed.empty:
            emit("  NO macro loading that was significant loses significance when")
            emit("  crypto-native controls are added -- the macro betas are not")
            emit("  artefacts of crypto-idiosyncratic shocks.")
        else:
            emit("  loadings killed by controls (were significant, now are not):")
            for _, r in killed.iterrows():
                emit(f"    {r['regime']:12s} {r['var']:14s} "
                     f"{r['b_macro']:+.3f}(t={r['t_macro']:.2f}) -> "
                     f"{r['b_ctrl']:+.3f}(t={r['t_ctrl']:.2f})")
        surv = res[res["verdict"] == "survives"]
        if not surv.empty:
            emit(f"\n  median |relative coefficient shift| among surviving loadings: "
                 f"{surv['rel_shift'].abs().median():.1%}")

    # ---------------- bad-control diagnostic --------------------------------
    emit("\n" + "=" * 100)
    emit("DIAGNOSTIC: why the crypto market factor is NOT used as a control")
    emit("=" * 100)
    emit("  Contemporaneous correlation with the dependent variable (btc_xs):")
    for c in ("d_mvrv", "d_eth_btc", "cmkt_ex_btc", "funding", "stbl_gr",
              "hash_gr", "adr_gr"):
        if c not in w:
            continue
        x = w[["btc_xs", c]].dropna()
        x19 = w.loc["2019-10-01":, ["btc_xs", c]].dropna()
        tag = "  <-- EXCLUDED as bad control" if c in ("d_mvrv", "d_eth_btc",
                                                      "cmkt_ex_btc") else ""
        emit(f"    {c:14s} full {x['btc_xs'].corr(x[c]):+.3f} (n={len(x)})   "
             f"2019+ {x19['btc_xs'].corr(x19[c]):+.3f} (n={len(x19)}){tag}")
    emit("  Note d_mvrv: +0.155 full sample but +0.926 post-2019. The full-sample")
    emit("  figure is what concealed the problem in the first version.")
    m_bad, _ = fit(w, MACRO + ["cmkt_ex_btc"], "2014-01-01", "2026-09-30")
    m_ref, _ = fit(w, MACRO, "2014-01-01", "2026-09-30")
    if m_bad is not None and m_ref is not None:
        emit(f"  full-sample R2: macro-only {m_ref.rsquared:.3f} -> "
             f"with cmkt_ex_btc {m_bad.rsquared:.3f}")
        emit(f"  {'var':14s} {'macro-only':>20s} {'+ cmkt_ex_btc':>20s}")
        for v in MACRO:
            if v in m_ref.params and v in m_bad.params:
                emit(f"  {v:14s} {m_ref.params[v]:8.3f}"
                     f"{stars(m_ref.tvalues[v]):<3s}(t={m_ref.tvalues[v]:5.2f}) "
                     f"{m_bad.params[v]:8.3f}{stars(m_bad.tvalues[v]):<3s}"
                     f"(t={m_bad.tvalues[v]:5.2f})")
        emit(f"  cmkt_ex_btc    {m_bad.params['cmkt_ex_btc']:8.3f}"
             f"{stars(m_bad.tvalues['cmkt_ex_btc']):<3s}"
             f"(t={m_bad.tvalues['cmkt_ex_btc']:5.2f})")
        emit("  Conditioning on an 0.8-correlated sibling asset absorbs the")
        emit("  dependent variable, not a confounder. This is reported for")
        emit("  transparency and is NOT the specification the verdict rests on.")

    if not res.empty:
        res.to_csv(ROOT / "outputs" / "tables" / "crypto_control_stability.csv",
                  index=False)
    (ROOT / "outputs" / "tables" / "followup4_crypto_controls.txt").write_text(
        "\n".join(lines) + "\n")
    print("\nwrote outputs/tables/followup4_crypto_controls.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
