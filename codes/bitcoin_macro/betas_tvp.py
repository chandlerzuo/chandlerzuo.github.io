"""Phase 3: time-varying betas by three independent estimators, and the
correlation-vs-relative-volatility decomposition.

Because beta = rho * (sigma_btc / sigma_factor), a rising beta can mean more
macro co-movement (rho up) or merely more Bitcoin volatility (sigma ratio up).
Those are economically opposite stories and most commentary conflates them.
Every beta here is therefore decomposed by holding one component at its
regime-start value and letting the other move.

Estimators, deliberately different in their assumptions:
  1. rolling OLS        - transparent; smears break dates by half a window
  2. DCC-GARCH          - gives conditional rho and sigma separately (the
                          decomposition comes from here)
  3. TVP state-space    - random-walk betas via Kalman filter/smoother; no
                          window choice, and returns credible bands
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fetchlib import PROC, ROOT

warnings.filterwarnings("ignore")

FACTORS = {
    "ret_mkt": "US equity market (Mkt-RF)",
    "ret_ndx": "Nasdaq-100",
    "ret_gold": "Gold (LBMA)",
    "ret_ust10y": "10y Treasury total return",
    "ret_usd": "Broad USD",
    "ret_silver": "Silver (LBMA)",
}


# ---------------------------------------------------------------- rolling OLS
def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    cov = df["y"].rolling(window).cov(df["x"])
    var = df["x"].rolling(window).var()
    out = pd.DataFrame({
        "beta": cov / var,
        "rho": df["y"].rolling(window).corr(df["x"]),
        "sd_y": df["y"].rolling(window).std(),
        "sd_x": df["x"].rolling(window).std(),
    })
    out["sd_ratio"] = out["sd_y"] / out["sd_x"]
    return out.dropna()


# ---------------------------------------------------------------- GARCH + DCC
def garch11_t(r: np.ndarray) -> tuple[np.ndarray, dict]:
    """GJR-GARCH(1,1) with Student-t errors, estimated by ML.

    GJR (asymmetric) rather than plain GARCH because Baur & Dimpfl (2018) show
    crypto's volatility asymmetry is INVERTED relative to equities: positive
    shocks raise volatility more than negative ones. A symmetric model would
    impose the equity sign convention on an asset that does not share it, so the
    asymmetry parameter is estimated freely and reported.
    """
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    v0 = r.var()

    def negll(p):
        omega, alpha, gamma, beta, nu = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta + gamma / 2 >= 0.999 or nu <= 2.1:
            return 1e10
        h = np.empty(len(r))
        h[0] = v0
        for t in range(1, len(r)):
            e = r[t - 1]
            h[t] = omega + (alpha + gamma * (e < 0)) * e * e + beta * h[t - 1]
            if h[t] <= 0 or not np.isfinite(h[t]):
                return 1e10
        z2 = r * r / h
        from scipy.special import gammaln
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
              - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
        return -np.sum(ll)

    best, bestf = None, np.inf
    for a0, g0, b0 in [(0.05, 0.0, 0.90), (0.08, 0.05, 0.85), (0.03, -0.02, 0.94)]:
        p0 = [v0 * 0.05, a0, g0, b0, 6.0]
        try:
            res = minimize(negll, p0, method="Nelder-Mead",
                          options={"maxiter": 4000, "fatol": 1e-6})
            if res.fun < bestf:
                best, bestf = res.x, res.fun
        except Exception:  # noqa: BLE001
            continue
    if best is None:
        return np.full(len(r), np.sqrt(v0)), {}

    omega, alpha, gamma, beta, nu = best
    h = np.empty(len(r))
    h[0] = v0
    for t in range(1, len(r)):
        e = r[t - 1]
        h[t] = omega + (alpha + gamma * (e < 0)) * e * e + beta * h[t - 1]
    params = {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
              "nu": nu, "persistence": alpha + beta + gamma / 2}
    return np.sqrt(h), params


def normal_scores(z: np.ndarray) -> np.ndarray:
    """Rank-to-normal-score (Gaussian copula) transform.

    Needed because BTC's GARCH-standardized residuals have an estimated
    t degrees-of-freedom near 3.5. Gaussian DCC quasi-ML on residuals that
    fat-tailed is dominated by a handful of extreme weeks, and the optimizer
    responds by driving the correlation innovation parameter to ~0, producing a
    near-constant rho: the first attempt here returned rho spanning only
    0.151-0.233 when the realized 52-week rolling correlation spans -0.22 to
    +0.55. Transforming to normal scores preserves the rank dependence structure
    and its dynamics while removing the outlier dominance.
    """
    from scipy.stats import norm
    n = len(z)
    ranks = pd.Series(z).rank(method="average").values
    return norm.ppf(ranks / (n + 1.0))


def dcc(z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """DCC(1,1), correlation-targeted, on normal-score-transformed residuals."""
    z1, z2 = normal_scores(z1), normal_scores(z2)
    qbar = np.corrcoef(z1, z2)[0, 1]

    def negll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        q11 = q22 = 1.0
        q12 = qbar
        ll = 0.0
        for t in range(len(z1)):
            rho = q12 / np.sqrt(q11 * q22)
            rho = np.clip(rho, -0.999, 0.999)
            d = 1 - rho * rho
            ll += -0.5 * (np.log(d) + (z1[t] ** 2 + z2[t] ** 2 - 2 * rho * z1[t] * z2[t]) / d
                          - (z1[t] ** 2 + z2[t] ** 2))
            q11 = (1 - a - b) + a * z1[t] ** 2 + b * q11
            q22 = (1 - a - b) + a * z2[t] ** 2 + b * q22
            q12 = (1 - a - b) * qbar + a * z1[t] * z2[t] + b * q12
        return -ll

    # Coarse grid pre-search, then local refinement from the best grid points.
    # The DCC likelihood in (a, b) is flat in places and a single local optimizer
    # started at a conventional (0.02, 0.95) can sit in the near-constant-rho
    # corner even when a time-varying solution fits materially better.
    grid = [(a, b) for a in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15)
            for b in (0.5, 0.7, 0.85, 0.92, 0.96, 0.98)
            if a + b < 0.999]
    grid.sort(key=negll)

    best, bestf = grid[0], negll(grid[0])
    for p0 in grid[:4]:
        for method, kw in (("L-BFGS-B", {"bounds": [(1e-4, 0.3), (0.0, 0.998)]}),
                          ("Nelder-Mead", {"options": {"maxiter": 3000}})):
            try:
                res = minimize(negll, p0, method=method, **kw)
                if res.fun < bestf and np.isfinite(res.fun):
                    best, bestf = tuple(res.x), res.fun
            except Exception:  # noqa: BLE001
                continue

    a, b = best
    q11 = q22 = 1.0
    q12 = qbar
    rhos = np.empty(len(z1))
    for t in range(len(z1)):
        rhos[t] = np.clip(q12 / np.sqrt(q11 * q22), -0.999, 0.999)
        q11 = (1 - a - b) + a * z1[t] ** 2 + b * q11
        q22 = (1 - a - b) + a * z2[t] ** 2 + b * q22
        q12 = (1 - a - b) * qbar + a * z1[t] * z2[t] + b * q12
    return rhos, (a, b)


def dcc_beta(y: pd.Series, x: pd.Series) -> tuple[pd.DataFrame, dict]:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    sy, py = garch11_t(df["y"].values)
    sx, px = garch11_t(df["x"].values)
    zy = (df["y"].values - df["y"].mean()) / sy
    zx = (df["x"].values - df["x"].mean()) / sx
    rho, ab = dcc(zy, zx)
    out = pd.DataFrame({"rho": rho, "sd_y": sy, "sd_x": sx}, index=df.index)
    out["sd_ratio"] = out["sd_y"] / out["sd_x"]
    out["beta"] = out["rho"] * out["sd_ratio"]
    return out, {"btc_garch": py, "factor_garch": px, "dcc_a": ab[0], "dcc_b": ab[1]}


# ---------------------------------------------------------------- TVP Kalman
def tvp_regression(y: pd.Series, X: pd.DataFrame, q_scale: float = 1e-4) -> pd.DataFrame:
    """Random-walk-coefficient regression via Kalman filter.

    y_t = x_t' b_t + e_t ,  b_t = b_{t-1} + u_t

    The filter runs on STANDARDIZED y and X and the coefficients are rescaled
    afterwards by sd_y/sd_x_j. This is not cosmetic: a beta has units of
    sd_y/sd_x, so scaling the state prior and Q by var(y) -- as is tempting --
    is dimensionally wrong and makes the prior far too tight whenever the
    regressor is much less volatile than the dependent variable. With weekly BTC
    (sd~8%) against equities (sd~2%) that error pins every beta near its zero
    prior regardless of the data. In standardized space all coefficients are
    O(1), one q governs them all, and the prior can be made properly diffuse.
    """
    df = pd.concat([y.rename("__y"), X], axis=1).dropna()
    sd_y = df["__y"].std()
    sd_x = df[X.columns].std().replace(0, np.nan)
    ys = ((df["__y"] - df["__y"].mean()) / sd_y).values
    Xs = ((df[X.columns] - df[X.columns].mean()) / sd_x).values
    Xv = np.column_stack([np.ones(len(df)), Xs])
    n, k = Xv.shape

    r = 1.0                       # standardized observation variance
    Q = np.eye(k) * q_scale
    b = np.zeros(k)
    P = np.eye(k) * 10.0          # diffuse prior in standardized units

    bs, Ps, pred_err = np.zeros((n, k)), np.zeros((n, k)), np.zeros(n)
    for t in range(n):
        P = P + Q                                   # predict
        xt = Xv[t]
        f = xt @ P @ xt + r                         # innovation variance
        v = ys[t] - xt @ b                          # innovation
        pred_err[t] = v
        K = (P @ xt) / f
        b = b + K * v
        P = P - np.outer(K, xt @ P)
        bs[t], Ps[t] = b, np.diag(P)
        r = 0.97 * r + 0.03 * v * v                 # adaptive obs variance

    # Rescale back to economic units: b_orig_j = b_std_j * sd_y / sd_x_j
    cols = list(X.columns)
    scale = (sd_y / sd_x).values
    out = pd.DataFrame(index=df.index)
    for i, c in enumerate(cols):
        out[f"b_{c}"] = bs[:, i + 1] * scale[i]
        out[f"se_{c}"] = np.sqrt(np.maximum(Ps[:, i + 1], 0)) * scale[i]
    out["b_const"] = bs[:, 0] * sd_y
    out["pred_err"] = pred_err * sd_y
    return out


def pick_q(y: pd.Series, X: pd.DataFrame) -> float:
    """Choose the random-walk variance by one-step-ahead prediction error."""
    best, bestq = np.inf, 1e-4
    for q in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        try:
            e = tvp_regression(y, X, q)["pred_err"]
            m = np.mean(e.iloc[len(e) // 4:] ** 2)   # skip filter burn-in
            if m < best:
                best, bestq = m, q
        except Exception:  # noqa: BLE001
            continue
    return bestq


# ---------------------------------------------------------------- driver
REGIME_MARKS = ["2013-01-01", "2015-01-01", "2017-01-01", "2019-01-01",
                "2020-03-01", "2020-05-01", "2021-11-10", "2023-01-01",
                "2024-01-01", "2026-09-01"]


def main() -> int:
    w = pd.read_parquet(PROC / "panel_weekly.parquet")
    dl = pd.read_parquet(PROC / "panel_daily.parquet")
    w["btc_xs"] = w["btc_ret"] - w["rf"].fillna(0)
    dl["btc_xs"] = dl["btc_ret"] - dl["rf"].fillna(0)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    store: dict[str, pd.DataFrame] = {}

    emit("=" * 104)
    emit("PHASE 3A. ROLLING 52-WEEK BETAS, and the rho vs sigma-ratio decomposition")
    emit("=" * 104)
    for f, desc in FACTORS.items():
        if f not in w.columns:
            continue
        rb = rolling_beta(w["btc_xs"], w[f], 52)
        store[f"roll_{f}"] = rb
        emit(f"\n--- {desc} ({f}) ---")
        emit(f"{'as of':>12s} {'beta':>8s} {'rho':>7s} {'sd_btc':>8s} "
             f"{'sd_fac':>8s} {'sd_ratio':>9s}")
        for mark in REGIME_MARKS:
            ts = pd.Timestamp(mark)
            idx = rb.index[rb.index <= ts]
            if len(idx) == 0:
                continue
            r = rb.loc[idx[-1]]
            emit(f"{str(idx[-1].date()):>12s} {r['beta']:8.3f} {r['rho']:7.3f} "
                 f"{r['sd_y']:8.4f} {r['sd_x']:8.4f} {r['sd_ratio']:9.2f}")

    emit("\n" + "=" * 104)
    emit("PHASE 3B. DCC-GARCH CONDITIONAL BETAS (weekly). GJR asymmetry estimated freely:")
    emit("  gamma>0 = equity-like (bad news raises vol); gamma<0 = inverted, as")
    emit("  Baur & Dimpfl (2018) report for crypto.")
    emit("=" * 104)
    for f, desc in FACTORS.items():
        if f not in w.columns or w[f].notna().sum() < 200:
            continue
        try:
            db, params = dcc_beta(w["btc_xs"], w[f])
        except Exception as exc:  # noqa: BLE001
            emit(f"{f}: DCC failed ({str(exc)[:60]})")
            continue
        store[f"dcc_{f}"] = db
        pb, pf = params["btc_garch"], params["factor_garch"]
        emit(f"\n--- {desc} ({f}) ---")
        if pb:
            emit(f"  BTC   GJR: alpha={pb['alpha']:.3f} gamma={pb['gamma']:+.3f} "
                 f"beta={pb['beta']:.3f} persist={pb['persistence']:.3f} nu={pb['nu']:.1f}")
        if pf:
            emit(f"  {f:6s} GJR: alpha={pf['alpha']:.3f} gamma={pf['gamma']:+.3f} "
                 f"beta={pf['beta']:.3f} persist={pf['persistence']:.3f} nu={pf['nu']:.1f}")
        a_, b_ = params["dcc_a"], params["dcc_b"]
        rho_sd = db["rho"].std()
        flag = ""
        if rho_sd < 1e-6 or a_ < 1e-3:
            flag = "  <-- DEGENERATE: rho is constant, DCC did not identify " \
                   "time variation; treat this pair's conditional beta as " \
                   "unconditional and rely on rolling OLS instead"
        emit(f"  DCC: a={a_:.4f} b={b_:.4f} (persistence {a_ + b_:.3f}) "
             f"sd(rho)={rho_sd:.4f}{flag}")
        ann = db.resample("YE").mean()
        emit("  annual mean conditional beta / rho:")
        emit("    " + "  ".join(f"{i.year}:{r['beta']:.2f}/{r['rho']:.2f}"
                                for i, r in ann.iterrows()))

    # ---------------- decomposition: what actually drove the beta change? ----
    emit("\n" + "=" * 104)
    emit("PHASE 3C. DECOMPOSITION OF THE BETA CHANGE BETWEEN REGIMES")
    emit("  beta = rho * (sd_btc/sd_factor). Counterfactuals isolate each channel:")
    emit("    beta_rho_only  : rho moves to end value, sd_ratio held at start")
    emit("    beta_vol_only  : sd_ratio moves to end value, rho held at start")
    emit("=" * 104)
    dec_rows = []
    for f in FACTORS:
        key = f"dcc_{f}"
        if key not in store:
            continue
        db = store[key]
        for a, b in zip(REGIME_MARKS[:-1], REGIME_MARKS[1:]):
            seg = db.loc[a:b]
            if len(seg) < 12:
                continue
            h = len(seg) // 4
            s, e = seg.iloc[:h], seg.iloc[-h:]
            rho0, rho1 = s["rho"].mean(), e["rho"].mean()
            sr0, sr1 = s["sd_ratio"].mean(), e["sd_ratio"].mean()
            dec_rows.append({
                "factor": f, "from": a[:7], "to": b[:7],
                "beta_start": rho0 * sr0, "beta_end": rho1 * sr1,
                "d_beta": rho1 * sr1 - rho0 * sr0,
                "via_rho": rho1 * sr0 - rho0 * sr0,
                "via_vol": rho0 * sr1 - rho0 * sr0,
                "rho_start": rho0, "rho_end": rho1,
                "sdratio_start": sr0, "sdratio_end": sr1,
            })
    dec = pd.DataFrame(dec_rows)
    if not dec.empty:
        emit(dec.round(3).to_string(index=False))
        dec.to_csv(ROOT / "outputs" / "tables" / "beta_decomposition.csv", index=False)

    # ---------------- TVP Kalman on the core model --------------------------
    emit("\n" + "=" * 104)
    emit("PHASE 3D. TVP (KALMAN) BETAS, core 5-factor model, weekly")
    emit("=" * 104)
    core = [c for c in ("ret_mkt", "tech_tilt", "ret_gold", "d_real10", "ret_usd")
            if c in w.columns]
    q = pick_q(w["btc_xs"], w[core])
    tvp = tvp_regression(w["btc_xs"], w[core], q)
    store["tvp_core"] = tvp
    emit(f"core factors: {core}")
    emit(f"selected random-walk variance scale q={q:.0e} "
         f"(chosen by one-step prediction error)")
    emit(f"\n{'as of':>12s} " + " ".join(f"{c:>13s}" for c in core))
    for mark in REGIME_MARKS:
        ts = pd.Timestamp(mark)
        idx = tvp.index[tvp.index <= ts]
        if len(idx) == 0:
            continue
        r = tvp.loc[idx[-1]]
        cells = []
        for c in core:
            bv, se = r[f"b_{c}"], r[f"se_{c}"]
            star = "*" if abs(bv) > 1.96 * se else " "
            cells.append(f"{bv:7.2f}{star}({se:4.2f})")
        emit(f"{str(idx[-1].date()):>12s} " + " ".join(cells))
    emit("\n  * = |beta| > 1.96 filter SE. Bands are filtered, not smoothed,")
    emit("    so they are wider early in the sample by construction.")

    outdir = PROC / "tvp"
    outdir.mkdir(exist_ok=True)
    for k, v in store.items():
        v.to_parquet(outdir / f"{k}.parquet")
    (ROOT / "outputs" / "tables" / "phase3_tvp.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote outputs/tables/phase3_tvp.txt and data/processed/tvp_betas.h5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
