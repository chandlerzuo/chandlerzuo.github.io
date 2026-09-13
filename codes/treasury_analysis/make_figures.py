"""Generate four additional figures for the treasury blog post."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Dropbox/Public/blog/treasury_analysis/outputs/")

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BOND = "#1f4e79"    # deep blue
BILL = "#c98a2b"    # ochre
SP500 = "#7a1f2b"   # dark red
HORIZONS = [1, 3, 5, 10]


def fig1_irr_boxplot():
    """Bond vs T-bill realised IRR, boxplots side-by-side per horizon."""
    df = pd.read_csv(OUT + "part1_irr_by_start_year.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.34
    for i, h in enumerate(HORIZONS):
        sub = df[df.horizon == h]
        bp = ax.boxplot([sub.bond_irr.values * 100], positions=[i - width / 2 - 0.02],
                        widths=width, patch_artist=True, showfliers=True,
                        medianprops=dict(color="black"))
        for b in bp["boxes"]:
            b.set(facecolor=BOND, alpha=0.8)
        bp = ax.boxplot([sub.bill_irr.values * 100], positions=[i + width / 2 + 0.02],
                        widths=width, patch_artist=True, showfliers=True,
                        medianprops=dict(color="black"))
        for b in bp["boxes"]:
            b.set(facecolor=BILL, alpha=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h}y" for h in HORIZONS])
    ax.set_xlabel("Holding horizon")
    ax.set_ylabel("Realised annualised IRR (%)")
    ax.set_title("Realised IRR by horizon: 30-year bond vs rolling 4-week bills\n(distribution across 1990-2026 start years)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BOND, alpha=0.8),
               plt.Rectangle((0, 0), 1, 1, fc=BILL, alpha=0.8)]
    ax.legend(handles, ["30-year bond", "Rolling 4-week bills"], frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT + "part1_irr_boxplot_bond_vs_bill.png", bbox_inches="tight")
    plt.close(fig)


def _sharpe_by_horizon(df, col):
    """Sharpe of a strategy's excess return over rolling bills, by horizon."""
    out = {}
    for h in HORIZONS:
        sub = df[df.horizon == h]
        e = sub[col].values - sub.bill_irr.values
        out[h] = e.mean() / e.std(ddof=1)
    return out


def fig2_sharpe_bars():
    """Bond vs S&P 500 excess-return Sharpe, grouped bars per horizon."""
    df = pd.read_csv(OUT + "part1_sp500_comparison.csv")
    bond = _sharpe_by_horizon(df, "bond_irr")
    sp = _sharpe_by_horizon(df, "sp500_irr")
    x = np.arange(len(HORIZONS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [bond[h] for h in HORIZONS], width, color=BOND, label="30-year bond")
    ax.bar(x + width / 2, [sp[h] for h in HORIZONS], width, color=SP500, label="S&P 500 (total return)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}y" for h in HORIZONS])
    ax.set_xlabel("Holding horizon")
    ax.set_ylabel("Sharpe ratio of excess return over bills")
    ax.set_title("Risk-adjusted edge by horizon: bond vs equities\n(Sharpe of excess return over rolling bills, 1990-2026)")
    ax.legend(frameon=False, loc="upper left")
    for xi, h in zip(x - width / 2, HORIZONS):
        ax.annotate(f"{bond[h]:.2f}", (xi, bond[h]), ha="center", va="bottom", fontsize=9)
    for xi, h in zip(x + width / 2, HORIZONS):
        ax.annotate(f"{sp[h]:.2f}", (xi, sp[h]), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "part1_sharpe_bond_vs_sp500_bars.png", bbox_inches="tight")
    plt.close(fig)


def _todays_curve(maturities):
    """Latest observed par yields (%) mapped onto the plotting maturity grid."""
    yd = pd.read_csv(os.path.expanduser(
        "~/Dropbox/Public/blog/treasury_analysis/data/processed/yields_daily.csv"))
    today = yd.iloc[-1]
    col_for = {0.25: "3Mo", 0.5: "6Mo", 1.0: "1Y", 2.0: "2Y", 3.0: "3Y",
               5.0: "5Y", 7.0: "7Y", 10.0: "10Y", 20.0: "20Y", 30.0: "30Y"}
    return today["date"], [float(today[col_for[m]]) for m in maturities]


def fig3_simulated_curve_distribution():
    """Percentile fan of simulated future yield curves by maturity, per horizon."""
    pc = pd.read_csv(OUT + "part2_predicted_curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, h in zip(axes.ravel(), HORIZONS):
        sub = pc[pc.horizon == h]
        piv = sub.pivot(index="maturity", columns="pctile", values="yield_pct").sort_index()
        m = piv.index.values
        ax.fill_between(m, piv[5], piv[95], color=BOND, alpha=0.15, label="5-95 pctile")
        ax.fill_between(m, piv[25], piv[75], color=BOND, alpha=0.30, label="25-75 pctile")
        ax.plot(m, piv[50], color=BOND, lw=2, label="median")
        today_date, today_y = _todays_curve(m)
        ax.plot(m, today_y, color="black", lw=1.6, ls="--", label="actual curve today")
        ax.set_title(f"{h} years ahead")
        ax.set_xscale("log")
        ax.set_xticks([0.25, 1, 3, 10, 30])
        ax.set_xticklabels(["3m", "1y", "3y", "10y", "30y"])
    axes[1, 0].set_xlabel("Maturity")
    axes[1, 1].set_xlabel("Maturity")
    axes[0, 0].set_ylabel("Yield (%)")
    axes[1, 0].set_ylabel("Yield (%)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Distribution of simulated future yield curves (Nelson-Siegel + VAR(1), 20,000 paths)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT + "part2_simulated_curve_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig4_ns_components_fit():
    """Historical Nelson-Siegel factors (level/slope/curvature) and fit error."""
    nf = pd.read_csv(OUT + "part2_ns_factors.csv", parse_dates=["date"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(nf.date, nf.b0 * 100, color=BOND, lw=1.6, label=r"level $\beta_0$")
    ax1.plot(nf.date, nf.b1 * 100, color=BILL, lw=1.6, label=r"slope $\beta_1$")
    ax1.plot(nf.date, nf.b2 * 100, color=SP500, lw=1.6, label=r"curvature $\beta_2$")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_ylabel("Factor value (%)")
    ax1.set_title("Nelson-Siegel factors fitted to the historical curve (month-end, 1990-2026)")
    ax1.legend(frameon=False, ncol=3, loc="upper right")
    ax2.plot(nf.date, nf.rmse * 1e4, color="grey", lw=1.0)
    ax2.set_ylabel("Fit RMSE (bp)")
    ax2.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT + "part2_ns_components_fit.png", bbox_inches="tight")
    plt.close(fig)


def fig5_forward_irr_distributions():
    """Distribution of simulated forward bond IRR by horizon (Part 2)."""
    s = pd.read_csv(OUT + "part2_sim_irr_samples.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    for ax, h in zip(axes.ravel(), HORIZONS):
        bond = s[f"bond_{h}y"].values * 100
        bill_med = np.median(s[f"bill_{h}y"].values) * 100
        p_neg = (bond < 0).mean() * 100
        ax.hist(bond, bins=80, color=BOND, alpha=0.85)
        ax.axvline(0, color="grey", lw=0.9)
        ax.axvline(bill_med, color=BILL, lw=1.8, ls="--", label=f"bill median {bill_med:.1f}%")
        ax.set_title(f"{h} years ahead   (P[IRR<0] = {p_neg:.0f}%)")
        ax.set_xlabel("Bond IRR (%)")
        ax.legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylabel("Simulated paths")
    axes[1, 0].set_ylabel("Simulated paths")
    fig.suptitle("Distribution of simulated forward bond IRR by horizon (20,000 paths)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT + "part2_irr_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig6_forward_sharpe():
    """Forward-looking Sharpe of bond excess return over bills, by horizon (simulation)."""
    s = pd.read_csv(OUT + "part2_sim_irr_samples.csv")
    sharpe = {}
    for h in HORIZONS:
        e = s[f"bond_{h}y"].values - s[f"bill_{h}y"].values
        sharpe[h] = e.mean() / e.std(ddof=1)
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, [sharpe[h] for h in HORIZONS], width=0.55, color=BOND)
    for xi, h in zip(x, HORIZONS):
        ax.annotate(f"{sharpe[h]:.2f}", (xi, sharpe[h]), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}y" for h in HORIZONS])
    ax.set_xlabel("Holding horizon")
    ax.set_ylabel("Forward Sharpe of excess return over bills")
    ax.set_title("Forward-looking Sharpe ratio by horizon\n(Nelson-Siegel + VAR(1) simulation, 20,000 paths, from today)")
    ax.set_ylim(0, max(sharpe.values()) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT + "part2_forward_sharpe.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_irr_boxplot()
    fig2_sharpe_bars()
    fig3_simulated_curve_distribution()
    fig4_ns_components_fit()
    fig5_forward_irr_distributions()
    fig6_forward_sharpe()
    print("wrote 6 figures to", OUT)
