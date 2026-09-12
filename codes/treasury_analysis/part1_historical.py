#!/usr/bin/env python3
"""
PART 1 -- Historical IRR comparison: 30-year Treasury bond vs. rolling 4-week bills.

For each start year Y (first business day) and horizon h in {1,3,5,10} years:

  BOND strategy
    * Buy a par 30y bond at 100; coupon rate = 30y par yield at Y (semiannual).
    * Hold to Y+h collecting cash coupons; sell at the (30-h)-year point on the
      NS-fitted curve at Y+h. IRR from the full cash-flow stream (cash coupons).

  BILL strategy
    * Invest 100 at Y; roll into 4-week bills every ~28 days at the prevailing
      4-week rate (spliced DTB4WK / DTB3 proxy). IRR == realized CAGR.

Outputs (outputs/):
  part1_irr_by_start_year.csv     one row per (start_year, horizon): bond & bill IRR
  part1_summary_by_horizon.csv    distribution stats + Sharpe per horizon
  part1_*.png                     charts
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
CMT = T.TENORS  # canonical tenor label -> years


def load():
    y = pd.read_csv(PROC / "yields_daily.csv", parse_dates=["date"]).set_index("date")
    short = pd.read_csv(PROC / "short_rate_daily.csv", parse_dates=["date"]).set_index("date")
    return y, short


def curve_on(date, yields_daily, window_days=7):
    """Fitted NS betas using the last available observation on/near `date`.
    Returns (betas, lam, obs_date) or (None, None, None)."""
    sub = yields_daily.loc[:date].tail(window_days)
    if sub.empty:
        return None, None, None
    row = sub.iloc[-1]
    obs_date = sub.index[-1]
    taus, ys = [], []
    for sid, tau in CMT.items():
        v = row.get(sid, np.nan)
        if np.isfinite(v):
            taus.append(tau); ys.append(v / 100.0)
    if len(taus) < 4:
        return None, None, None
    betas, lam, _ = T.fit_ns_nls(taus, ys)
    return betas, lam, obs_date


def first_bday_on_or_after(idx, year):
    target = pd.Timestamp(year=year, month=1, day=1)
    pos = idx.searchsorted(target)
    if pos >= len(idx):
        return None
    return idx[pos]


def yield_30y_at(date, yields_daily):
    """Observed 30y par yield near date (decimals); fall back to NS(30)."""
    sub = yields_daily.loc[:date, "30Y"].dropna()
    if not sub.empty and (date - sub.index[-1]).days <= 10:
        return sub.iloc[-1] / 100.0
    betas, lam, _ = curve_on(date, yields_daily)
    if betas is None:
        return np.nan
    return T.ns_yield(30.0, betas, lam)


def bill_irr(start_date, horizon_years, short):
    """Roll 4-week bills from start_date for horizon_years. Return (irr, ok)."""
    end_date = start_date + pd.DateOffset(years=horizon_years)
    s = short["short_rate_pct"].loc[start_date:end_date].dropna()
    if s.empty:
        return np.nan, False
    # step ~28 days using prevailing rate (asof)
    value = 100.0
    t = start_date
    all_rates = short["short_rate_pct"].dropna()
    last_used = None
    while t < end_date:
        step_end = min(t + pd.Timedelta(days=28), end_date)
        dt = (step_end - t).days / 365.25
        r = all_rates.asof(t)
        if not np.isfinite(r):
            return np.nan, False
        value *= (1.0 + (r / 100.0) * dt)
        last_used = t
        t = step_end
    if last_used is None:
        return np.nan, False
    years = (end_date - start_date).days / 365.25
    irr = (value / 100.0) ** (1.0 / years) - 1.0
    return irr, True


def bond_irr(start_date, horizon_years, yields_daily):
    """Buy par 30y bond at start_date, sell at horizon. Return (irr, coupon, sale)."""
    coupon = yield_30y_at(start_date, yields_daily)
    if not np.isfinite(coupon):
        return np.nan, np.nan, np.nan
    sale_date = start_date + pd.DateOffset(years=horizon_years)
    rem = 30 - horizon_years
    betas, lam, obs = curve_on(sale_date, yields_daily)
    if betas is None:
        return np.nan, coupon, np.nan
    ytm = T.ns_yield(rem, betas, lam)
    if not np.isfinite(ytm):
        return np.nan, coupon, np.nan
    sale_price = T.bond_price(coupon, ytm, rem)
    times, cfs = T.par_bond_cashflows(coupon, horizon_years, sale_price)
    irr = T.irr_annual(times, cfs)
    return irr, coupon, sale_price


def main():
    yields_daily, short = load()
    idx = yields_daily.index

    # start years: from when 30y exists, ensure 10y horizon has sale data
    dgs30 = yields_daily["30Y"].dropna()
    first_year = max(dgs30.index.min().year, short.index.min().year)
    last_data = min(yields_daily.index.max(), short.index.max())
    rows = []
    for year in range(first_year, last_data.year + 1):
        sd = first_bday_on_or_after(idx, year)
        if sd is None:
            continue
        for h in HORIZONS:
            sale_date = sd + pd.DateOffset(years=h)
            if sale_date > last_data:
                continue
            b_irr, coupon, sale = bond_irr(sd, h, yields_daily)
            l_irr, ok = bill_irr(sd, h, short)
            rows.append(dict(start_year=year, start_date=sd.date(), horizon=h,
                             coupon30y=coupon, sale_price=sale,
                             bond_irr=b_irr, bill_irr=l_irr,
                             excess=b_irr - l_irr if (np.isfinite(b_irr) and np.isfinite(l_irr)) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "part1_irr_by_start_year.csv", index=False)
    print(f"Wrote {len(df)} (start_year,horizon) rows -> part1_irr_by_start_year.csv")

    # ---- summary + Sharpe by horizon ----
    summ = []
    for h in HORIZONS:
        d = df[df.horizon == h].dropna(subset=["bond_irr", "bill_irr"])
        if d.empty:
            continue
        b, l, e = d.bond_irr.values, d.bill_irr.values, d.excess.values
        def stats(x):
            return dict(mean=np.mean(x), std=np.std(x, ddof=1) if len(x) > 1 else np.nan,
                        min=np.min(x), p25=np.percentile(x, 25), median=np.median(x),
                        p75=np.percentile(x, 75), max=np.max(x))
        sb, sl = stats(b), stats(l)
        # Sharpe of the bond's excess return over the matched bill investment
        sharpe_excess = np.mean(e) / np.std(e, ddof=1) if len(e) > 1 and np.std(e, ddof=1) > 0 else np.nan
        # Standalone Sharpe of each strategy using the bill as the risk-free leg
        sharpe_bond_rf = np.mean(b - l) / np.std(b, ddof=1) if np.std(b, ddof=1) > 0 else np.nan
        summ.append(dict(horizon=h, n=len(d),
                         bond_mean=sb["mean"], bond_std=sb["std"],
                         bill_mean=sl["mean"], bill_std=sl["std"],
                         excess_mean=np.mean(e), excess_std=np.std(e, ddof=1),
                         sharpe_excess=sharpe_excess, sharpe_bond_rf=sharpe_bond_rf,
                         bond_min=sb["min"], bond_p25=sb["p25"], bond_med=sb["median"],
                         bond_p75=sb["p75"], bond_max=sb["max"],
                         bill_min=sl["min"], bill_med=sl["median"], bill_max=sl["max"],
                         prob_bond_beats_bill=np.mean(e > 0)))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT / "part1_summary_by_horizon.csv", index=False)
    print("\n=== Summary by horizon (annualized IRR) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(sdf[["horizon", "n", "bond_mean", "bond_std", "bill_mean", "bill_std",
                   "excess_mean", "excess_std", "sharpe_excess", "prob_bond_beats_bill"]])

    make_charts(df, sdf)
    make_histograms(df)
    print("\nCharts written to outputs/")


def make_histograms(df):
    """Histograms of realized IRR: rows = horizon, cols = bond / bill."""
    fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(12, 3 * len(HORIZONS)))
    for i, h in enumerate(HORIZONS):
        d = df[df.horizon == h].dropna(subset=["bond_irr", "bill_irr"])
        for j, (col, name, color) in enumerate(
                [("bond_irr", "30y bond", "#2E5A88"), ("bill_irr", "4wk bill roll", "#C55A11")]):
            ax = axes[i, j]
            x = d[col].values * 100
            ax.hist(x, bins=12, color=color, alpha=.75, edgecolor="white")
            ax.axvline(np.mean(x), color="black", ls="--", lw=1,
                       label=f"mean {np.mean(x):.2f}%")
            ax.axvline(0, color="grey", lw=.6)
            ax.set_title(f"{name} — hold {h}y  (n={len(d)})", fontsize=10)
            ax.set_xlabel("annualized IRR (%)"); ax.set_ylabel("count")
            ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Historical distribution of realized IRR (1990–2026 start years)", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "part1_irr_histograms.png", dpi=130); plt.close(fig)


def make_charts(df, sdf):
    # 1) IRR time series by start year for each horizon
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, h in zip(axes.ravel(), HORIZONS):
        d = df[df.horizon == h]
        ax.plot(d.start_year, d.bond_irr * 100, "o-", label="30y bond", color="#2E5A88")
        ax.plot(d.start_year, d.bill_irr * 100, "s-", label="4wk bill roll", color="#C55A11")
        ax.axhline(0, color="grey", lw=.6)
        ax.set_title(f"h = {h}y  (annualized IRR)")
        ax.set_ylabel("IRR (%)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Historical annualized IRR by start year", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "part1_irr_timeseries.png", dpi=130); plt.close(fig)

    # 2) IRR distributions (box) per horizon
    fig, ax = plt.subplots(figsize=(11, 6))
    data, labels, pos = [], [], []
    for i, h in enumerate(HORIZONS):
        d = df[df.horizon == h].dropna(subset=["bond_irr", "bill_irr"])
        data.append(d.bond_irr.values * 100); pos.append(i * 3 + 1); labels.append(f"bond {h}y")
        data.append(d.bill_irr.values * 100); pos.append(i * 3 + 2); labels.append(f"bill {h}y")
    bp = ax.boxplot(data, positions=pos, widths=.8, patch_artist=True, showmeans=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("#2E5A88" if i % 2 == 0 else "#C55A11"); box.set_alpha(.6)
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="grey", lw=.6); ax.set_ylabel("annualized IRR (%)")
    ax.set_title("Distribution of realized IRRs by strategy & horizon"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "part1_irr_distributions.png", dpi=130); plt.close(fig)

    # 3) Sharpe (excess) by horizon
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sdf.horizon.astype(str) + "y", sdf.sharpe_excess, color="#2E5A88", alpha=.8)
    ax.axhline(0, color="grey", lw=.6)
    ax.set_ylabel("Sharpe of bond excess return over bills")
    ax.set_title("Sharpe ratio (bond minus bill, per start-year) by horizon"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "part1_sharpe.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
