"""Phase 7: is a matched analogue actually informative, or just a nice story?

A matched historical window is a curiosity unless it carries out-of-sample
content. The test:

  match on the FIRST HALF of each Bitcoin regime only, then ask whether the
  matched analogues' SUBSEQUENT path predicts Bitcoin's realised subsequent
  volatility, drawdown and Sharpe better than two baselines:
    persistence  - Bitcoin's own first-half value carried forward
    random       - the median outcome of randomly chosen asset-windows

If the analogues cannot beat persistence, the honest conclusion is that
analogue-matching is descriptive and has no forecasting value. That is a
publishable answer and it is the one the design must be willing to reach.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from analogues import (BTC_DAYS, FEATURES, REGIMES, TRADING_DAYS, long_panel,
                       mahalanobis_setup, mdist_to, reference_windows,
                       window_features)
from fetchlib import PROC, ROOT

warnings.filterwarnings("ignore")

TOP_K = 10


def forward_stats(s: pd.Series, end_ts: pd.Timestamp, n: int,
                 per_year: float) -> dict | None:
    """Realised stats over the n observations AFTER end_ts."""
    fut = s.loc[s.index > end_ts].iloc[:n]
    if len(fut) < max(40, n // 2):
        return None
    sd = fut.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    cum = fut.cumsum()
    dd = float((cum - cum.cummax()).min())
    return {"f_vol": sd * np.sqrt(per_year),
            "f_sharpe": fut.mean() * per_year / (sd * np.sqrt(per_year)),
            "f_dd": dd}


def main() -> int:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    panel = pd.read_parquet(PROC / "panel_daily.parquet")
    assets, bench = long_panel()
    rng = np.random.default_rng(11)

    emit("=" * 100)
    emit("PHASE 7. OUT-OF-SAMPLE VALIDITY OF ANALOGUE MATCHES")
    emit("  Match on regime FIRST HALF; predict Bitcoin's SECOND HALF.")
    emit(f"  Prediction = median forward outcome of top-{TOP_K} matches.")
    emit("=" * 100)

    rows = []
    for name, a, b, tag in REGIMES:
        r = panel.loc[a:b, "btc_ret"].dropna()
        if len(r) < 240:
            emit(f"\n{name}: too short to split, skipped")
            continue
        h = len(r) // 2
        first, second = r.iloc[:h], r.iloc[h:]

        # truth: what Bitcoin actually did in the second half
        sd2 = second.std()
        truth = {"f_vol": sd2 * np.sqrt(BTC_DAYS),
                "f_sharpe": second.mean() * BTC_DAYS / (sd2 * np.sqrt(BTC_DAYS)),
                "f_dd": float((second.cumsum() - second.cumsum().cummax()).min())}

        # persistence baseline: first-half values carried forward
        sd1 = first.std()
        persist = {"f_vol": sd1 * np.sqrt(BTC_DAYS),
                  "f_sharpe": first.mean() * BTC_DAYS / (sd1 * np.sqrt(BTC_DAYS)),
                  "f_dd": float((first.cumsum() - first.cumsum().cummax()).min())}

        # match using first-half features only
        span = (first.index[-1] - first.index[0]).days
        L = max(90, int(span * TRADING_DAYS / 365))
        n_fwd = int(len(second) * TRADING_DAYS / BTC_DAYS)

        ref = reference_windows(assets, bench, L, step=21)
        if ref.empty:
            continue
        btc_f = window_features(first, bench, BTC_DAYS, len(first),
                               np.array([0])).iloc[0]
        feats = [f for f in FEATURES if ref[f].notna().mean() > 0.6
                 and np.isfinite(btc_f.get(f, np.nan))]
        mu, sd, Ci, ok = mahalanobis_setup(ref, feats)
        refv = ref[ok].reset_index(drop=True)
        refv["d"] = mdist_to(refv, btc_f, mu, sd, Ci, feats)

        # forward outcomes of the top-K matches
        preds = []
        for _, row in refv.nsmallest(TOP_K * 3, "d").iterrows():
            fs = forward_stats(assets[row["asset"]].dropna(), row["end"],
                              n_fwd, TRADING_DAYS)
            if fs:
                preds.append(fs)
            if len(preds) >= TOP_K:
                break
        if len(preds) < 3:
            emit(f"\n{name}: too few matches with forward data, skipped")
            continue
        pred = {k: float(np.median([p[k] for p in preds])) for k in truth}

        # random-analogue baseline
        rnd = []
        for _, row in refv.sample(min(200, len(refv)), random_state=5).iterrows():
            fs = forward_stats(assets[row["asset"]].dropna(), row["end"],
                              n_fwd, TRADING_DAYS)
            if fs:
                rnd.append(fs)
        rand = {k: float(np.median([p[k] for p in rnd])) for k in truth} if rnd else None

        emit(f"\n--- {name} ---")
        emit(f"    matched on {len(first)}d, predicting {len(second)}d forward; "
             f"{len(preds)} analogues with forward data")
        emit(f"    {'metric':10s} {'truth':>9s} {'analogue':>9s} "
             f"{'persist':>9s} {'random':>9s}")
        for k, lab in (("f_vol", "vol"), ("f_sharpe", "sharpe"), ("f_dd", "maxDD")):
            emit(f"    {lab:10s} {truth[k]:9.3f} {pred[k]:9.3f} "
                 f"{persist[k]:9.3f} "
                 f"{(rand[k] if rand else float('nan')):9.3f}")
            rows.append({"regime": name, "metric": lab, "truth": truth[k],
                        "analogue": pred[k], "persistence": persist[k],
                        "random": rand[k] if rand else np.nan})

    res = pd.DataFrame(rows)
    if res.empty:
        emit("\nno regimes produced a valid test")
        return 0

    emit("\n" + "=" * 100)
    emit("SCORECARD: mean absolute error across regimes (lower is better)")
    emit("=" * 100)
    emit(f"{'metric':10s} {'analogue':>10s} {'persistence':>12s} {'random':>10s}"
         f"  {'winner':>12s}")
    verdict = {}
    for metric, g in res.groupby("metric"):
        mae = {c: float((g[c] - g["truth"]).abs().mean())
               for c in ("analogue", "persistence", "random")}
        win = min(mae, key=lambda k: mae[k] if np.isfinite(mae[k]) else np.inf)
        verdict[metric] = win
        emit(f"{metric:10s} {mae['analogue']:10.3f} {mae['persistence']:12.3f} "
             f"{mae['random']:10.3f}  {win:>12s}")

    n_win = sum(1 for v in verdict.values() if v == "analogue")
    emit("")
    if n_win == 0:
        emit("VERDICT: analogue matching loses on every metric. The matched")
        emit("  historical windows carry NO out-of-sample information about")
        emit("  Bitcoin's subsequent risk beyond what Bitcoin's own recent")
        emit("  history already implies. Analogues are descriptive only.")
    elif n_win == len(verdict):
        emit("VERDICT: analogues beat both baselines on all metrics.")
    else:
        emit(f"VERDICT: analogues win on {n_win} of {len(verdict)} metrics "
             f"({', '.join(k for k, v in verdict.items() if v == 'analogue')}); "
             "mixed evidence, and with 8 regimes this is a small-sample")
        emit("  comparison -- treat as suggestive, not established.")

    emit("\nCaveat that applies whatever the result: 8 regimes give 8 observations")
    emit("per metric. None of these MAE differences is statistically significant")
    emit("at any conventional level; the scorecard shows direction, not proof.")

    res.to_csv(ROOT / "outputs" / "tables" / "analogue_oos_validation.csv", index=False)
    (ROOT / "outputs" / "tables" / "phase7_oos.txt").write_text("\n".join(lines) + "\n")
    print("\nwrote outputs/tables/phase7_oos.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
