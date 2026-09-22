"""Phase 4: regime identification by three structurally different methods.

The user's framing is that regimes follow FROM beta exposure. Doing that naively
is circular: estimate betas with a rolling window, look for breaks in the
smoothed output, declare regimes. So three methods with different assumptions
are run and reconciled:

  1. Bai-Perron   multiple structural breaks in the factor REGRESSION itself
                  (global SSR minimisation by dynamic programming, BIC-selected,
                  with break-date confidence intervals). The formally correct
                  test for "did these coefficients change?"
  2. Markov-switching regression   betas AND residual variance switch across K
                  states; regimes may RECUR rather than being phases passed
                  through once.
  3. Gaussian HMM on a monthly feature vector of estimated betas/vol/rho.
                  Persistence modelled, not assumed (hence HMM, not k-means).

Reconciliation rule, fixed BEFORE looking at output: a boundary counts as robust
only if >=2 of 3 methods place a break within +/-2 months. Single-method breaks
are reported as tentative. Disagreements are reported, not hidden.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fetchlib import PROC, ROOT

warnings.filterwarnings("ignore")

CORE = ["ret_mkt", "tech_tilt", "ret_gold", "d_real10", "ret_usd"]
MIN_SEG = 26          # weeks: a regime shorter than ~6 months is not a regime
MAX_BREAKS = 6


# ------------------------------------------------------------- Bai-Perron
def ssr_matrix(y: np.ndarray, X: np.ndarray, min_seg: int) -> np.ndarray:
    """SSR for every admissible segment [i, j). Recursive least squares would be
    faster; n is small enough that direct solves are fine and less error-prone."""
    n = len(y)
    S = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        for j in range(i + min_seg, n + 1):
            Xs, ys = X[i:j], y[i:j]
            if Xs.shape[0] <= Xs.shape[1] + 2:
                continue
            try:
                beta, res, rank, _ = np.linalg.lstsq(Xs, ys, rcond=None)
                if rank < Xs.shape[1]:
                    continue
                S[i, j] = float(res[0]) if res.size else float(
                    np.sum((ys - Xs @ beta) ** 2))
            except np.linalg.LinAlgError:
                continue
    return S


def bai_perron(y: np.ndarray, X: np.ndarray, max_breaks: int = MAX_BREAKS,
              min_seg: int = MIN_SEG) -> dict:
    """Global SSR minimisation over break configurations by dynamic programming."""
    n, k = X.shape
    S = ssr_matrix(y, X, min_seg)

    # cost[m][j] = min SSR segmenting [0, j) with m breaks
    cost = [[np.inf] * (n + 1) for _ in range(max_breaks + 1)]
    arg = [[-1] * (n + 1) for _ in range(max_breaks + 1)]
    for j in range(min_seg, n + 1):
        cost[0][j] = S[0, j]
    for m in range(1, max_breaks + 1):
        for j in range((m + 1) * min_seg, n + 1):
            best, bi = np.inf, -1
            for i in range(m * min_seg, j - min_seg + 1):
                if not np.isfinite(cost[m - 1][i]) or not np.isfinite(S[i, j]):
                    continue
                c = cost[m - 1][i] + S[i, j]
                if c < best:
                    best, bi = c, i
            cost[m][j], arg[m][j] = best, bi

    out = {}
    for m in range(max_breaks + 1):
        if not np.isfinite(cost[m][n]):
            continue
        # recover break points
        brks, j, mm = [], n, m
        while mm > 0:
            i = arg[mm][j]
            if i < 0:
                break
            brks.append(i)
            j, mm = i, mm - 1
        brks = sorted(brks)
        sigma2 = cost[m][n] / n
        npar = (m + 1) * k + m
        bic = n * np.log(sigma2) + npar * np.log(n)
        out[m] = {"ssr": cost[m][n], "breaks": brks, "bic": bic,
                  "n_par": npar}
    return out


def sup_f(y: np.ndarray, X: np.ndarray, brk: int) -> float:
    """Chow F statistic for a single known break, for reporting break strength."""
    n, k = X.shape
    b_all, *_ = np.linalg.lstsq(X, y, rcond=None)
    ssr_r = np.sum((y - X @ b_all) ** 2)
    ssr_u = 0.0
    for a, b in ((0, brk), (brk, n)):
        bb, *_ = np.linalg.lstsq(X[a:b], y[a:b], rcond=None)
        ssr_u += np.sum((y[a:b] - X[a:b] @ bb) ** 2)
    if ssr_u <= 0:
        return np.nan
    return ((ssr_r - ssr_u) / k) / (ssr_u / (n - 2 * k))


def break_ci(y: np.ndarray, X: np.ndarray, brk: int, window: int = 26) -> tuple[int, int]:
    """Crude break-date interval: the set of dates whose two-segment SSR is
    within a likelihood-ratio tolerance of the optimum. Reported so that regime
    boundaries are never presented as if they were known to the week."""
    n = len(y)
    lo, hi = max(MIN_SEG, brk - window), min(n - MIN_SEG, brk + window)
    best, vals = np.inf, {}
    for c in range(lo, hi + 1):
        s = 0.0
        ok = True
        for a, b in ((0, c), (c, n)):
            if b - a <= X.shape[1] + 2:
                ok = False
                break
            bb, *_ = np.linalg.lstsq(X[a:b], y[a:b], rcond=None)
            s += np.sum((y[a:b] - X[a:b] @ bb) ** 2)
        if ok:
            vals[c] = s
            best = min(best, s)
    tol = best * (1 + 2.0 * X.shape[1] / n)   # ~LR tolerance
    keep = [c for c, v in vals.items() if v <= tol]
    return (min(keep), max(keep)) if keep else (brk, brk)


# ------------------------------------------------------------- Markov switching
def markov_betas(w: pd.DataFrame, factors: list[str], k_regimes: int = 3,
                switch_only: tuple[str, ...] = ("ret_mkt",)) -> dict:
    """Markov-switching regression where only SELECTED betas switch.

    Letting all five betas plus the variance switch across three states is ~21
    parameters on ~700 weekly observations; the first attempt did exactly that
    and produced a degenerate 6-week state with an annualised Sharpe of 19,
    i.e. it fitted a handful of extreme weeks. Restricting the switching set to
    the market beta (plus intercept and variance) is both estimable and the
    economically interesting object: this is the regime-switching-beta model the
    literature scan flagged as an open slot.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    df = w[["btc_xs"] + factors].dropna()
    exog = sm.add_constant(df[factors].values)
    names = ["const"] + list(factors)
    sw = [n == "const" or n in switch_only for n in names]
    mod = MarkovRegression(df["btc_xs"].values, k_regimes=k_regimes,
                          exog=exog, switching_exog=sw,
                          switching_variance=True)
    res = mod.fit(search_reps=30, maxiter=1000, disp=False)
    smoothed = pd.DataFrame(res.smoothed_marginal_probabilities, index=df.index)
    return {"res": res, "smoothed": smoothed, "index": df.index,
            "state": smoothed.idxmax(axis=1), "bic": res.bic, "aic": res.aic,
            "llf": res.llf}


# ------------------------------------------------------------- HMM on features
def feature_hmm(n_states: int = 4) -> dict | None:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:  # noqa: BLE001
        return None

    tv = PROC / "tvp"
    feats = {}
    for f in ("ret_mkt", "ret_ndx", "ret_gold", "ret_usd"):
        p = tv / f"roll_{f}.parquet"
        if p.exists():
            rb = pd.read_parquet(p)
            feats[f"beta_{f}"] = rb["beta"]
            feats[f"rho_{f}"] = rb["rho"]
    if not feats:
        return None
    w = pd.read_parquet(PROC / "panel_weekly.parquet")
    w["btc_xs"] = w["btc_ret"] - w["rf"].fillna(0)
    feats["vol_btc"] = w["btc_xs"].rolling(52).std()

    F = pd.DataFrame(feats).dropna()
    Fm = F.resample("ME").last().dropna()
    Z = (Fm - Fm.mean()) / Fm.std()

    hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                     n_iter=500, random_state=0)
    hmm.fit(Z.values)
    states = pd.Series(hmm.predict(Z.values), index=Fm.index)
    return {"states": states, "features": Fm, "model": hmm,
            "score": hmm.score(Z.values)}


def runs(s: pd.Series) -> list[tuple]:
    """Collapse a state series into contiguous (state, start, end, n) runs."""
    out, cur, start = [], s.iloc[0], s.index[0]
    prev = s.index[0]
    for t, v in s.items():
        if v != cur:
            out.append((cur, start, prev))
            cur, start = v, t
        prev = t
    out.append((cur, start, prev))
    return out


def main() -> int:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    w = pd.read_parquet(PROC / "panel_weekly.parquet")
    w["btc_xs"] = w["btc_ret"] - w["rf"].fillna(0)
    d = w.loc["2013-01-01":][["btc_xs"] + CORE].dropna()
    idx = d.index

    # Volatility weighting. An unweighted SSR criterion is dominated by BTC's
    # 2013 volatility (~141% annualised vs ~45% recently), so the first run of
    # this found a single break in Dec-2013 and nothing in 2020: it was detecting
    # a VARIANCE change, not a BETA change, because Bai-Perron assumes
    # homoskedasticity. Dividing each observation by an EWMA volatility estimate
    # makes segments scale-comparable, so breaks reflect changes in the
    # relationship rather than in the level of noise. Both are reported.
    ew_vol = d["btc_xs"].ewm(halflife=26, min_periods=26).std().bfill()
    wt = 1.0 / ew_vol.values

    y_raw = d["btc_xs"].values
    X_raw = sm.add_constant(d[CORE].values)
    y = y_raw * wt
    X = X_raw * wt[:, None]

    emit("=" * 100)
    emit("METHOD 1. BAI-PERRON MULTIPLE STRUCTURAL BREAKS")
    emit(f"  regression: btc_xs ~ {CORE}")
    emit(f"  weekly, {len(d)} obs {idx.min().date()} to {idx.max().date()}, "
         f"min segment {MIN_SEG}w")
    emit("  observations volatility-weighted by 1/EWMA(sd, halflife=26w) so that")
    emit("  breaks reflect beta changes rather than BTC's declining volatility.")
    emit("=" * 100)

    bp_unw = bai_perron(y_raw, X_raw)
    m_unw = min(bp_unw, key=lambda m: bp_unw[m]["bic"])
    emit(f"  [unweighted, for contrast] BIC picks m={m_unw}: "
         + ", ".join(str(idx[b].date()) for b in bp_unw[m_unw]["breaks"]))
    emit("")

    bp = bai_perron(y, X)
    emit(f"{'m':>2s} {'SSR':>10s} {'BIC':>10s}  break dates")
    for m, r in sorted(bp.items()):
        dates = ", ".join(str(idx[b].date()) for b in r["breaks"])
        emit(f"{m:2d} {r['ssr']:10.4f} {r['bic']:10.2f}  {dates}")

    best_m = min(bp, key=lambda m: bp[m]["bic"])
    emit(f"\nBIC-selected number of breaks: m = {best_m}")
    bp_breaks = bp[best_m]["breaks"]
    emit(f"\n{'break date':>12s} {'Chow F':>8s}  {'90% CI (approx)':>28s}")
    bp_dates = []
    for b in bp_breaks:
        f = sup_f(y, X, b)
        lo, hi = break_ci(y, X, b)
        bp_dates.append(idx[b])
        emit(f"{str(idx[b].date()):>12s} {f:8.2f}  "
             f"{str(idx[lo].date())} .. {str(idx[hi].date())}")
    emit("\n  Break-date intervals are wide by construction: with ~13 years of")
    emit("  weekly data a regime boundary is simply not identified to the week.")

    # per-segment betas
    emit("\n  Segment betas (OLS, HAC):")
    bounds = [0] + bp_breaks + [len(d)]
    seg_rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = d.iloc[a:b]
        Xs = sm.add_constant(sub[CORE])
        r = sm.OLS(sub["btc_xs"], Xs).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        row = {"start": sub.index.min().date(), "end": sub.index.max().date(),
               "n": len(sub), "r2": round(r.rsquared, 3),
               "vol_ann": round(sub["btc_xs"].std() * np.sqrt(52), 2),
               "sharpe": round(sub["btc_xs"].mean() * 52 /
                              (sub["btc_xs"].std() * np.sqrt(52)), 2)}
        for c in CORE:
            row[c] = f"{r.params[c]:.2f}({r.tvalues[c]:.1f})"
        seg_rows.append(row)
    segs = pd.DataFrame(seg_rows)
    emit(segs.to_string(index=False))
    segs.to_csv(ROOT / "outputs" / "tables" / "bai_perron_segments.csv", index=False)

    # ---------------------------------------------------------------- method 2
    emit("\n" + "=" * 100)
    emit("METHOD 2. MARKOV-SWITCHING REGRESSION (betas and variance switch)")
    emit("=" * 100)
    ms_best, ms_dates = None, []
    for k in (2, 3):
        try:
            m = markov_betas(w.loc["2013-01-01":], CORE, k)
        except Exception as exc:  # noqa: BLE001
            emit(f"  k={k}: failed ({str(exc)[:60]})")
            continue
        emit(f"  k={k}: llf={m['llf']:.1f}  AIC={m['aic']:.1f}  BIC={m['bic']:.1f}")
        if ms_best is None or m["bic"] < ms_best["bic"]:
            ms_best = m
    if ms_best is not None:
        k = ms_best["smoothed"].shape[1]
        emit(f"\n  BIC-selected k = {k}")
        st = ms_best["state"]
        # regime characteristics
        emit("\n  state occupancy and characteristics:")
        wx = w.loc[st.index, "btc_xs"]
        for s in sorted(st.unique()):
            mask = st == s
            x = wx[mask].dropna()
            emit(f"    state {s}: {mask.sum():4d} weeks ({mask.mean():5.1%})  "
                 f"vol={x.std() * np.sqrt(52):5.2f}  "
                 f"mean_ann={x.mean() * 52:+6.2f}  "
                 f"sharpe={x.mean() * 52 / (x.std() * np.sqrt(52)):+5.2f}")
        emit("\n  persistent runs (>= 26 weeks, matched to the Bai-Perron")
        emit("  minimum segment length so the three methods are comparable):")
        for s, a, b in runs(st):
            wk = len(st.loc[a:b])
            if wk >= 26:
                emit(f"    state {s}  {a.date()} -> {b.date()}  ({wk} weeks)")
                ms_dates.append(a)

    # ---------------------------------------------------------------- method 3
    emit("\n" + "=" * 100)
    emit("METHOD 3. GAUSSIAN HMM ON MONTHLY BETA/RHO/VOL FEATURE VECTOR")
    emit("=" * 100)
    hm = feature_hmm(4)
    hmm_dates = []
    if hm is None:
        emit("  unavailable")
    else:
        st = hm["states"]
        emit(f"  features: {list(hm['features'].columns)}")
        emit(f"  log-likelihood {hm['score']:.1f}, {len(st)} monthly obs")
        emit("\n  state means (original units):")
        prof = hm["features"].groupby(st).mean().round(3)
        emit(prof.to_string())
        emit("\n  persistent runs (>= 4 months):")
        for s, a, b in runs(st):
            mo = len(st.loc[a:b])
            if mo >= 4:
                emit(f"    state {s}  {a.date()} -> {b.date()}  ({mo} months)")
                hmm_dates.append(a)

    # ---------------------------------------------------------------- reconcile
    emit("\n" + "=" * 100)
    emit("RECONCILIATION: boundaries found by >=2 of 3 methods within +/- 2 months")
    emit("=" * 100)
    cand: dict[pd.Timestamp, list[str]] = {}
    for name, dates in (("bai-perron", bp_dates), ("markov", ms_dates),
                        ("hmm", hmm_dates)):
        for dt in dates:
            cand.setdefault(pd.Timestamp(dt), []).append(name)

    allc = sorted(cand)
    used, clusters = set(), []
    for dt in allc:
        if dt in used:
            continue
        grp = [o for o in allc if abs((o - dt).days) <= 62]
        methods = sorted({m for o in grp for m in cand[o]})
        clusters.append((min(grp), max(grp), methods))
        used.update(grp)

    emit(f"{'window':>26s}  {'#methods':>8s}  methods")
    robust = []
    for a, b, ms in clusters:
        tag = "ROBUST" if len(ms) >= 2 else "tentative"
        emit(f"{str(a.date()) + ' .. ' + str(b.date()):>26s}  {len(ms):8d}  "
             f"{', '.join(ms):34s} {tag}")
        if len(ms) >= 2:
            robust.append((a, b))

    emit("\nRobust boundaries: " + (", ".join(f"{a.date()}" for a, _ in robust)
                                    if robust else "NONE"))
    emit("\nHonest reading: methods agreeing on a boundary is evidence; a boundary")
    emit("found by one method only is a property of that estimator, not of Bitcoin.")

    pd.DataFrame([{"start": a, "end": b, "methods": ",".join(m)}
                  for a, b, m in clusters]).to_csv(
        ROOT / "outputs" / "tables" / "regime_boundaries.csv", index=False)
    if ms_best is not None:
        ms_best["state"].rename("ms_state").to_frame().to_parquet(
            PROC / "ms_states.parquet")
    if hm is not None:
        hm["states"].rename("hmm_state").to_frame().to_parquet(
            PROC / "hmm_states.parquet")
    pd.DataFrame({"break_date": [str(x.date()) for x in bp_dates]}).to_csv(
        ROOT / "outputs" / "tables" / "bai_perron_breaks.csv", index=False)

    (ROOT / "outputs" / "tables" / "phase4_regimes.txt").write_text("\n".join(lines) + "\n")
    print("\nwrote outputs/tables/phase4_regimes.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------- partial structural breaks
def partial_break_F(y: np.ndarray, X: np.ndarray, chg: list[int],
                   min_seg: int = MIN_SEG) -> tuple[float, int, np.ndarray]:
    """sup-F for a break in a SUBSET of coefficients (Bai-Perron partial change).

    Testing whether a whole 6-coefficient vector broke has very low power when
    the regression R-squared is ~0.03: the BIC penalty for a full extra
    coefficient set swamps the SSR gain, and the test reports no break even when
    a single loading has clearly moved. Restricting the break to the intercept
    and the market beta costs 2 parameters instead of 6 and is the hypothesis of
    economic interest.

    Returns (max F, argmax index, F series).
    """
    n, k = X.shape
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    ssr_r = float(np.sum((y - X @ b0) ** 2))
    q = len(chg)

    Fs = np.full(n, np.nan)
    for c in range(min_seg, n - min_seg + 1):
        dummy = np.zeros((n, q))
        dummy[c:, :] = X[c:, chg]
        Xa = np.column_stack([X, dummy])
        try:
            ba, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        ssr_u = float(np.sum((y - Xa @ ba) ** 2))
        if ssr_u <= 0:
            continue
        Fs[c] = ((ssr_r - ssr_u) / q) / (ssr_u / (n - k - q))
    if np.all(np.isnan(Fs)):
        return np.nan, -1, Fs
    return float(np.nanmax(Fs)), int(np.nanargmax(Fs)), Fs


def wild_bootstrap_supF(y: np.ndarray, X: np.ndarray, chg: list[int],
                       n_boot: int = 399, seed: int = 0,
                       min_seg: int = MIN_SEG) -> tuple[float, np.ndarray]:
    """Critical values for sup-F by wild bootstrap under the no-break null.

    Asymptotic Andrews critical values assume homoskedasticity and a fixed
    trimming fraction; BTC returns are heteroskedastic and fat-tailed enough that
    those tables are not trustworthy here. Rademacher wild bootstrap preserves
    the conditional heteroskedasticity pattern while imposing H0 exactly.
    """
    rng = np.random.default_rng(seed)
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit, resid = X @ b0, y - X @ b0
    stats = np.empty(n_boot)
    for i in range(n_boot):
        e = resid * rng.choice([-1.0, 1.0], size=len(resid))
        stats[i] = partial_break_F(fit + e, X, chg, min_seg)[0]
    return float(np.nanpercentile(stats, 95)), stats


def sequential_breaks(y: np.ndarray, X: np.ndarray, chg: list[int], idx,
                     alpha: float = 0.05, max_b: int = 5,
                     min_seg: int = MIN_SEG, n_boot: int = 299) -> list[dict]:
    """Sequential partial-break detection: test, split, recurse."""
    found: list[dict] = []
    segments = [(0, len(y))]
    while segments and len(found) < max_b:
        nxt = []
        for a, b in segments:
            if b - a < 2 * min_seg + 4:
                continue
            ys, Xs = y[a:b], X[a:b]
            F, c, _ = partial_break_F(ys, Xs, chg, min_seg)
            if not np.isfinite(F):
                continue
            crit, boot = wild_bootstrap_supF(ys, Xs, chg, n_boot=n_boot,
                                            seed=a, min_seg=min_seg)
            pval = float(np.mean(boot >= F))
            if F > crit and pval <= alpha:
                found.append({"idx": a + c, "date": idx[a + c], "F": F,
                             "crit95": crit, "p": pval,
                             "seg": (str(idx[a].date()), str(idx[b - 1].date()))})
                nxt += [(a, a + c), (a + c, b)]
        segments = nxt
    return sorted(found, key=lambda r: r["idx"])
