#!/usr/bin/env python3
"""
Add an S&P 500 (total-return) benchmark to the Part 1 historical comparison,
on the SAME start-year / horizon / Sharpe basis as the bond and bill strategies.

Data: Shiller-based monthly S&P 500 (price index + annualized dividend per share),
      data/raw/sp500_shiller.csv. Monthly total return reinvests Dividend/12.

Sharpe convention (matches part1_historical.py): per start-year,
  excess = strategy_IRR - matched rolling-bill IRR;  Sharpe = mean(excess)/std(excess).

Outputs:
  outputs/part1_sp500_comparison.csv
  outputs/part1_sharpe_vs_sp500.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
HORIZONS = [1, 3, 5, 10]


def build_tr_index():
    df = pd.read_csv(RAW / "sp500_shiller.csv")
    df = df[["Date", "SP500", "Dividend"]].copy()
    df["date"] = pd.to_datetime(df["Date"])
    df["price"] = pd.to_numeric(df["SP500"], errors="coerce")
    df["div"] = pd.to_numeric(df["Dividend"], errors="coerce")
    df = df.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
    # recent months may have Dividend==0 (not yet reported) -> forward-fill last real
    df.loc[df["div"] <= 0, "div"] = np.nan
    df["div"] = df["div"].ffill()
    # monthly total-return growth: (P_t + Div_t/12) / P_{t-1}
    p = df["price"].values
    d = df["div"].values
    growth = np.ones(len(df))
    growth[1:] = (p[1:] + d[1:] / 12.0) / p[:-1]
    df["tr_index"] = np.cumprod(growth)
    return df.set_index("date")[["price", "tr_index"]]


def sp_irr(tr, start_year, h):
    """Annualized total-return CAGR from Jan of start_year to Jan of start_year+h."""
    def jan_val(yr):
        seg = tr.loc[f"{yr}-01-01":f"{yr}-02-15", "tr_index"]
        return seg.iloc[0] if len(seg) else np.nan
    v0, v1 = jan_val(start_year), jan_val(start_year + h)
    if not (np.isfinite(v0) and np.isfinite(v1)):
        return np.nan
    return (v1 / v0) ** (1.0 / h) - 1.0


def main():
    tr = build_tr_index()
    print(f"S&P TR index: {tr.index.min().date()} -> {tr.index.max().date()}")

    base = pd.read_csv(OUT / "part1_irr_by_start_year.csv")
    rows = []
    for _, r in base.iterrows():
        yr, h = int(r.start_year), int(r.horizon)
        rows.append(dict(start_year=yr, horizon=h,
                         bond_irr=r.bond_irr, bill_irr=r.bill_irr,
                         sp500_irr=sp_irr(tr, yr, h)))
    df = pd.DataFrame(rows)

    summ = []
    for h in HORIZONS:
        d = df[df.horizon == h].dropna(subset=["bond_irr", "bill_irr", "sp500_irr"])
        b_ex = (d.bond_irr - d.bill_irr).values
        s_ex = (d.sp500_irr - d.bill_irr).values
        def sharpe(x):
            return np.mean(x) / np.std(x, ddof=1) if len(x) > 1 and np.std(x, ddof=1) > 0 else np.nan
        summ.append(dict(
            horizon=h, n=len(d),
            bond_mean=d.bond_irr.mean(), bond_std=d.bond_irr.std(),
            sp500_mean=d.sp500_irr.mean(), sp500_std=d.sp500_irr.std(),
            bill_mean=d.bill_irr.mean(),
            sharpe_bond=sharpe(b_ex), sharpe_sp500=sharpe(s_ex),
            sp500_prob_neg=(d.sp500_irr < 0).mean(),
            bond_prob_neg=(d.bond_irr < 0).mean(),
            sp500_beats_bond=(d.sp500_irr > d.bond_irr).mean()))
    sdf = pd.DataFrame(summ)
    df.to_csv(OUT / "part1_sp500_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(sdf[["horizon", "n", "bond_mean", "sp500_mean", "bill_mean",
                   "sharpe_bond", "sharpe_sp500", "sp500_prob_neg",
                   "sp500_beats_bond"]])

    # chart: Sharpe (excess over bills) bond vs S&P by horizon
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(HORIZONS)); w = 0.38
    ax.bar(x - w/2, sdf.sharpe_bond, w, label="30y bond", color="#2E5A88", alpha=.85)
    ax.bar(x + w/2, sdf.sharpe_sp500, w, label="S&P 500 (total return)", color="#548235", alpha=.85)
    for i, (b, s) in enumerate(zip(sdf.sharpe_bond, sdf.sharpe_sp500)):
        ax.text(i - w/2, b + .03, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, s + .03, f"{s:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{h}y" for h in HORIZONS])
    ax.axhline(0, color="grey", lw=.6)
    ax.set_ylabel("Sharpe of excess return over rolling bills")
    ax.set_title("Historical Sharpe by horizon: 30y bond vs. S&P 500 (1990–2026)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "part1_sharpe_vs_sp500.png", dpi=130); plt.close(fig)
    print("\nSaved part1_sp500_comparison.csv and part1_sharpe_vs_sp500.png")


if __name__ == "__main__":
    main()
