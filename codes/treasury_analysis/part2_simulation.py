#!/usr/bin/env python3
"""
PART 2 -- Dynamic Nelson-Siegel + VAR(1) simulation of future IRR.

Pipeline
--------
1. Fit Nelson-Siegel factors (level b0, slope b1, curvature b2) at FIXED lambda
   (Diebold-Li) to each month-end curve -> monthly factor panel.
2. Estimate a VAR(1) on the factors:  X_t = c + A X_{t-1} + e_t,  e_t ~ N(0, Sigma).
3. From today's fitted factors, Monte-Carlo simulate N monthly factor paths out to
   10y. For each path & horizon h in {1,3,5,10}y:
      * sale yield  = NS(30-h) from the simulated curve at h  -> bond sale price
      * bond IRR    from fixed cash coupons (coupon = today's 30y par yield) + sale
      * bill IRR    = compounding the simulated short rate (NS at ~1M) monthly
   -> distribution of IRRs, forward-looking Sharpe, curve fan chart.

This is a PHYSICAL-measure projection (real-world dynamics), suitable for
"what IRR is likely", not an arbitrage-free pricing exercise.

Outputs (outputs/):
  part2_ns_factors.csv          monthly fitted factors + fit RMSE
  part2_var_params.json         VAR(1) c, A, Sigma, eigenvalues, today's state
  part2_sim_irr.csv             per-(horizon) simulated IRR distribution stats
  part2_sim_paths_sample.csv    a sample of simulated 30y-yield paths
  part2_*.png                   charts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tsy_lib as T

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

HORIZONS = [1, 3, 5, 10]
LAM = T.DL_LAMBDA
N_SIM = 20000
RNG = np.random.default_rng(42)


def fit_factor_panel():
    y = pd.read_csv(PROC / "yields_daily.csv", parse_dates=["date"]).set_index("date")
    tenors = [t for t in T.TENORS if t in y.columns]
    taus = np.array([T.TENORS[t] for t in tenors])
    monthly = y.resample("ME").last()
    recs = []
    for dt, row in monthly.iterrows():
        vals = row[tenors].values.astype(float) / 100.0
        betas, rmse = T.fit_ns_ols(taus, vals, LAM)
        if np.all(np.isfinite(betas)):
            recs.append((dt, betas[0], betas[1], betas[2], rmse))
    f = pd.DataFrame(recs, columns=["date", "b0", "b1", "b2", "rmse"]).set_index("date")
    return f, y, tenors, taus


def fit_var1(X):
    """OLS VAR(1). X: (T,k). Returns c (k,), A (k,k), Sigma (k,k)."""
    Y = X[1:]
    Xl = X[:-1]
    Z = np.column_stack([np.ones(len(Xl)), Xl])          # (T-1, k+1)
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)            # (k+1, k)
    c = B[0]
    A = B[1:].T                                          # so that Y = c + A Xl^T ... row form
    resid = Y - Z @ B
    Sigma = np.cov(resid.T, ddof=Z.shape[1])
    return c, A, Sigma, resid


def simulate(c, A, Sigma, x0, n_months, n_sim):
    """Return array (n_sim, n_months+1, k) of factor paths."""
    k = len(x0)
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(k))
    paths = np.empty((n_sim, n_months + 1, k))
    paths[:, 0, :] = x0
    for t in range(1, n_months + 1):
        shocks = RNG.standard_normal((n_sim, k)) @ L.T
        paths[:, t, :] = c + paths[:, t - 1, :] @ A.T + shocks
    return paths


def ns_from_factors(betas, tau):
    return T.ns_yield(tau, betas, LAM)


def main():
    f, y, tenors, taus = fit_factor_panel()
    f.to_csv(OUT / "part2_ns_factors.csv")
    print(f"NS factors: {len(f)} months, mean fit RMSE = {f.rmse.mean()*1e4:.1f} bp")

    X = f[["b0", "b1", "b2"]].values
    c, A, Sigma, resid = fit_var1(X)
    eig = np.linalg.eigvals(A)
    print("VAR(1) eigenvalue moduli:", np.round(np.abs(eig), 3),
          "(all < 1 => stationary)" if np.all(np.abs(eig) < 1) else "(NON-stationary!)")

    x0 = X[-1]                        # today's factor state
    today = f.index[-1]
    coupon = ns_from_factors(x0, 30.0)     # today's fitted 30y par yield (decimal)
    obs_30 = y["30Y"].dropna().iloc[-1] / 100.0
    print(f"Today ({today.date()}): fitted 30y={coupon*100:.2f}% (observed {obs_30*100:.2f}%), "
          f"factors b0={x0[0]*100:.2f} b1={x0[1]*100:.2f} b2={x0[2]*100:.2f}")

    n_months = 12 * max(HORIZONS)
    paths = simulate(c, A, Sigma, x0, n_months, N_SIM)

    # short rate path (annualized), use ~1M NS point (vectorized via loadings)
    short_tau = 1/12
    L1 = np.array([1.0, (1 - np.exp(-short_tau/LAM))/(short_tau/LAM),
                   (1 - np.exp(-short_tau/LAM))/(short_tau/LAM) - np.exp(-short_tau/LAM)])
    short_path = paths @ L1                                   # (n_sim, n_months+1)

    results = []
    irr_samples = {}
    for h in HORIZONS:
        m = 12 * h
        rem = 30 - h
        Lrem = np.array([1.0, (1 - np.exp(-rem/LAM))/(rem/LAM),
                         (1 - np.exp(-rem/LAM))/(rem/LAM) - np.exp(-rem/LAM)])
        sale_yield = paths[:, m, :] @ Lrem                    # (n_sim,)
        # bond IRR per path
        bond_irr = np.empty(N_SIM)
        for i in range(N_SIM):
            sp = T.bond_price(coupon, sale_yield[i], rem)
            times, cfs = T.par_bond_cashflows(coupon, h, sp)
            bond_irr[i] = T.irr_annual(times, cfs)
        # bill IRR per path: compound monthly short rate (annualized simple / 12)
        r_m = short_path[:, 1:m+1] / 12.0
        growth = np.prod(1.0 + r_m, axis=1)
        bill_irr = growth ** (1.0 / h) - 1.0

        def q(a, p):
            return float(np.nanpercentile(a, p))
        excess = bond_irr - bill_irr
        sharpe = np.nanmean(excess) / np.nanstd(excess, ddof=1)
        row = dict(horizon=h, remaining_maturity=rem,
                   bond_mean=np.nanmean(bond_irr), bond_std=np.nanstd(bond_irr, ddof=1),
                   bond_p5=q(bond_irr, 5), bond_p25=q(bond_irr, 25), bond_median=q(bond_irr, 50),
                   bond_p75=q(bond_irr, 75), bond_p95=q(bond_irr, 95),
                   bill_mean=np.nanmean(bill_irr), bill_std=np.nanstd(bill_irr, ddof=1),
                   bill_median=q(bill_irr, 50),
                   excess_mean=np.nanmean(excess), sharpe_forward=sharpe,
                   prob_bond_beats_bill=float(np.nanmean(excess > 0)),
                   prob_bond_negative=float(np.nanmean(bond_irr < 0)))
        results.append(row)
        irr_samples[h] = (bond_irr, bill_irr)

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT / "part2_sim_irr.csv", index=False)
    print("\n=== Simulated forward IRR (buy 30y today, sell at horizon) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(rdf[["horizon", "bond_mean", "bond_std", "bond_p5", "bond_median", "bond_p95",
                   "bill_mean", "excess_mean", "sharpe_forward", "prob_bond_beats_bill",
                   "prob_bond_negative"]])

    # save VAR params + today's state
    params = dict(lambda_=LAM, n_months_of_data=len(f),
                  c=c.tolist(), A=A.tolist(), Sigma=Sigma.tolist(),
                  eig_moduli=np.abs(eig).tolist(), today=str(today.date()),
                  x0=x0.tolist(), coupon_today=coupon, obs_30y_today=obs_30,
                  n_sim=N_SIM)
    (OUT / "part2_var_params.json").write_text(json.dumps(params, indent=2, default=str))

    # sample 30y-yield paths for inspection
    L30 = np.array([1.0, (1 - np.exp(-30/LAM))/(30/LAM),
                    (1 - np.exp(-30/LAM))/(30/LAM) - np.exp(-30/LAM)])
    y30_paths = paths[:200] @ L30
    months = np.arange(n_months + 1)
    pd.DataFrame(y30_paths.T, index=months).to_csv(OUT / "part2_sim_paths_sample.csv")

    # save simulated IRR samples (for boxplot / re-analysis)
    samp = {}
    for h in HORIZONS:
        b, l = irr_samples[h]
        samp[f"bond_{h}y"] = b
        samp[f"bill_{h}y"] = l
    pd.DataFrame(samp).to_csv(OUT / "part2_sim_irr_samples.csv", index=False)

    # today's observed curve for reference in the predicted-curve chart
    obs_row = y.iloc[-1]
    obs_mat = np.array([T.TENORS[t] for t in y.columns])
    obs_val = obs_row.values.astype(float)
    ok = np.isfinite(obs_val)
    obs_curve = (obs_mat[ok], obs_val[ok])

    predicted_curves(paths, obs_curve)          # future term-structure fan
    make_charts(f, paths, y30_paths, irr_samples, rdf, coupon)
    make_boxplot(irr_samples)
    print("\nCharts written to outputs/")


def predicted_curves(paths, obs_curve):
    """Predicted future yield CURVES (term structure) at each horizon, with bands.
    Saves part2_predicted_curves.csv and part2_predicted_curves.png."""
    mat = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    Lg = T._ns_loadings(mat, LAM)               # (len(mat), 3)
    recs = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, h in zip(axes.ravel(), HORIZONS):
        m = 12 * h
        Y = paths[:, m, :] @ Lg.T * 100          # (n_sim, len(mat))
        pct = {p: np.percentile(Y, p, axis=0) for p in [5, 25, 50, 75, 95]}
        for p in [5, 25, 50, 75, 95]:
            for mt, v in zip(mat, pct[p]):
                recs.append(dict(horizon=h, maturity=mt, pctile=p, yield_pct=v))
        ax.fill_between(mat, pct[5], pct[95], color="#2E5A88", alpha=.15, label="5–95%")
        ax.fill_between(mat, pct[25], pct[75], color="#2E5A88", alpha=.3, label="25–75%")
        ax.plot(mat, pct[50], "o-", color="#2E5A88", lw=2, label="median")
        ax.plot(obs_curve[0], obs_curve[1], "s--", color="black", lw=1.2,
                ms=4, label="today (observed)")
        ax.set_title(f"predicted curve, {h}y ahead", fontsize=11)
        ax.set_xlabel("maturity (years)"); ax.set_ylabel("par yield (%)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Simulated future Treasury yield curves (DNS-VAR, physical measure)",
                 fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "part2_predicted_curves.png", dpi=130); plt.close(fig)
    pd.DataFrame(recs).to_csv(OUT / "part2_predicted_curves.csv", index=False)


def make_boxplot(irr_samples):
    """Boxplot of simulated forward IRR: bond vs bill across horizons."""
    fig, ax = plt.subplots(figsize=(11, 6))
    data, pos, labels, colors = [], [], [], []
    for i, h in enumerate(HORIZONS):
        b, l = irr_samples[h]
        data.append(b * 100); pos.append(i * 3 + 1); labels.append(f"bond {h}y"); colors.append("#2E5A88")
        data.append(l * 100); pos.append(i * 3 + 2); labels.append(f"bill {h}y"); colors.append("#C55A11")
    bp = ax.boxplot(data, positions=pos, widths=.8, patch_artist=True, showmeans=True,
                    whis=(5, 95), showfliers=False)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c); box.set_alpha(.6)
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="grey", lw=.6); ax.set_ylabel("annualized IRR (%)")
    ax.set_title("Simulated forward IRR distribution (box = IQR, whiskers = 5–95%)")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "part2_irr_boxplot.png", dpi=130); plt.close(fig)


def make_charts(f, paths, y30_paths, irr_samples, rdf, coupon):
    # 1) factor history
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, lab, col_c in [("b0", "level b0", "#2E5A88"), ("b1", "slope b1", "#C55A11"),
                            ("b2", "curvature b2", "#548235")]:
        ax.plot(f.index, f[col] * 100, label=lab, color=col_c)
    ax.axhline(0, color="grey", lw=.6); ax.legend(); ax.grid(alpha=.3)
    ax.set_ylabel("%"); ax.set_title("Nelson-Siegel factors (monthly, 1990-2026)")
    fig.tight_layout(); fig.savefig(OUT / "part2_ns_factors.png", dpi=130); plt.close(fig)

    # 2) 30y yield fan chart
    months = np.arange(paths.shape[1])
    L30 = np.array([1.0, (1 - np.exp(-30/LAM))/(30/LAM),
                    (1 - np.exp(-30/LAM))/(30/LAM) - np.exp(-30/LAM)])
    y30 = paths @ L30 * 100
    pct = {p: np.percentile(y30, p, axis=0) for p in [5, 25, 50, 75, 95]}
    fig, ax = plt.subplots(figsize=(12, 5))
    yr = months / 12
    ax.fill_between(yr, pct[5], pct[95], color="#2E5A88", alpha=.15, label="5-95%")
    ax.fill_between(yr, pct[25], pct[75], color="#2E5A88", alpha=.3, label="25-75%")
    ax.plot(yr, pct[50], color="#2E5A88", lw=2, label="median")
    ax.set_xlabel("years ahead"); ax.set_ylabel("simulated 30y par yield (%)")
    ax.set_title("Simulated 30y yield fan (DNS-VAR, physical measure)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "part2_yield_fan.png", dpi=130); plt.close(fig)

    # 3) IRR distributions per horizon
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, h in zip(axes.ravel(), HORIZONS):
        b, l = irr_samples[h]
        ax.hist(b * 100, bins=80, alpha=.6, color="#2E5A88", label="30y bond", density=True)
        ax.hist(l * 100, bins=80, alpha=.5, color="#C55A11", label="4wk bill roll", density=True)
        ax.axvline(np.nanmedian(b) * 100, color="#2E5A88", ls="--")
        ax.axvline(0, color="grey", lw=.6)
        ax.set_title(f"h = {h}y  (median bond {np.nanmedian(b)*100:.2f}%)")
        ax.set_xlabel("annualized IRR (%)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Simulated forward IRR distributions (buy 30y today)", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "part2_irr_distributions.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
