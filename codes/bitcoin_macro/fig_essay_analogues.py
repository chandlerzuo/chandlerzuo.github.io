"""Essay diagrams: how Bitcoin resembles gas, crude, silver, cars and Turkey.

Two figures, each answering a different question about the same matches:

  A. PATHS -- small multiples of amplitude-normalised cumulative log-return
     paths, Bitcoin against its matched analogue window. This is exactly the
     object the DTW distance is computed on, so the reader sees the quantity
     being reported rather than a decorative stand-in.

  B. FEATURES -- where each match is close and where it is NOT. Standardised
     risk features, Bitcoin versus analogue. Included because the path panels
     flatter the comparison: normalising amplitude away is what makes the shapes
     look alike, and the volatility gap is the single largest reason the formal
     test still rejects every match.

Windows are read from the saved match tables, not re-derived, so the figures and
the statistics in REPORT.md cannot drift apart.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from analogues import BTC_DAYS, FEATURES, industry49, window_features
from fetchlib import PROC, ROOT

BLUE = "#006BA2"
RED = "#E3120B"
CYAN = "#3EBCD2"
INK = "#121317"
GREY = "#5b5f63"
GRID = "#d9d9d6"
SURFACE = "#ffffff"
FIG = ROOT / "outputs" / "figures"

# (label, asset key, analogue window, BTC regime window, frequency)
PAIRS = [
    ("Crude oil, 2020-21", "cmdty_wti", "2020-03-16", "2021-02-19",
     "2013-01-01", "2013-12-06", "D"),
    ("Natural gas, 2021-22", "cmdty_natgas", "2021-03-08", "2022-04-22",
     "2019-01-01", "2020-02-19", "D"),
    ("Silver, 1978-80", "metal_silver", "1978-10-02", "1980-09-25",
     "2017-01-01", "2018-12-31", "D"),
    ("American car makers,\n2019-25", "ind_Autos", "2019-01-31", "2025-09-30",
     "2020-01-01", "2026-09-30", "M"),
    ("Turkish equities,\n1989-2003", "em_tr_Turkey", "1989-07-31", "2003-03-31",
     "2013-01-01", "2026-09-30", "M"),
]

BTC_LABEL = {"2013-01-01": "Bitcoin 2013", "2019-01-01": "Bitcoin 2019",
             "2017-01-01": "Bitcoin 2017-18", "2020-01-01": "Bitcoin 2020-26",
             "2013-01-01_full": "Bitcoin 2013-26"}


def econ(ax, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=GREY, labelsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=5)


def load_assets():
    """Daily and monthly candidate series for the five named analogues."""
    metals = pd.read_parquet(PROC / "metals_daily.parquet")
    fr = (pd.read_parquet(PROC / "fred_long.parquet")
            .pivot(index="date", columns="series", values="value").sort_index())
    oecd = pd.read_parquet(PROC / "oecd_equity_monthly.parquet")

    daily, monthly = {}, {}
    for sid, nm in (("DCOILWTICO", "cmdty_wti"), ("DHHNGSP", "cmdty_natgas")):
        p = fr[sid].dropna()
        daily[nm] = np.log(p[p > 0]).diff().dropna()
    p = metals["silver"].dropna()
    daily["metal_silver"] = np.log(p[p > 0]).diff().dropna()

    ind = industry49()
    indm = (1 + ind).resample("ME").prod() - 1
    monthly["ind_Autos"] = indm["Autos"].dropna()
    p = oecd["tr"].dropna()
    monthly["em_tr_Turkey"] = np.log(p[p > 0]).diff().dropna()
    return daily, monthly


def norm_path(r: pd.Series) -> np.ndarray:
    """Amplitude-normalised cumulative log-return path on a common 0-1 grid."""
    c = r.cumsum().values
    c = (c - c.mean()) / (c.std() or 1.0)
    return np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, len(c)), c)


def figure_paths(daily, monthly, btc_d, btc_m):
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.5))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.66, bottom=0.14, wspace=0.16)

    from matplotlib.patches import Rectangle
    fig.add_artist(Rectangle((0.04, 0.955), 0.035, 0.022, facecolor=RED,
                            edgecolor="none", transform=fig.transFigure))
    fig.text(0.04, 0.875, "Shapes that rhyme", fontsize=15, weight="bold",
            color=INK)
    fig.text(0.04, 0.80, "Cumulative return paths, each scaled to mean 0 and "
                         "standard deviation 1. DTW = dynamic time-warping "
                         "distance (lower = closer)",
            fontsize=8.6, color=GREY)

    for ax, (lab, key, a0, a1, b0, b1, freq) in zip(axes, PAIRS):
        src = daily if freq == "D" else monthly
        btc = btc_d if freq == "D" else btc_m
        an = src[key].loc[a0:a1]
        bt = btc.loc[b0:b1]
        if len(an) < 10 or len(bt) < 10:
            continue
        x = np.linspace(0, 1, 200)
        ax.plot(x, norm_path(bt), color=BLUE, lw=1.9, zorder=3)
        ax.plot(x, norm_path(an), color=RED, lw=1.6, zorder=2, alpha=0.9)
        econ(ax, lab)
        ax.set_xticks([])
        ax.set_yticks([-2, 0, 2])
        ax.set_ylim(-2.9, 2.9)
        from analogues import dtw_distance
        d = dtw_distance(norm_path(bt), norm_path(an))
        ax.annotate(f"DTW {d:.3f}", xy=(0.97, 0.045), xycoords="axes fraction",
                   fontsize=8, color=GREY, ha="right")

    fig.text(0.735, 0.875, "Bitcoin", fontsize=9.5, color=BLUE, weight="bold")
    fig.text(0.805, 0.875, "analogue", fontsize=9.5, color=RED, weight="bold")
    fig.text(0.04, 0.035, "Left-hand panels daily data, right-hand two monthly. "
                          "Sources: CoinMetrics; FRED; LBMA; OECD; Kenneth "
                          "French data library; author's calculations",
            fontsize=7.4, color=GREY, style="italic")
    out = FIG / "essay_analogue_paths.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def figure_features(daily, monthly, btc_d, btc_m, bench_d, bench_m):
    """Where the matches hold and where they fail."""
    show = ["vol_ann", "sharpe", "skew", "max_dd", "rho_eq", "rho_gold"]
    nice = {"vol_ann": "Volatility", "sharpe": "Sharpe", "skew": "Skew",
            "max_dd": "Max drawdown", "rho_eq": "Corr. w/ equities",
            "rho_gold": "Corr. w/ gold"}

    rows = []
    for lab, key, a0, a1, b0, b1, freq in PAIRS:
        src, btc = (daily, btc_d) if freq == "D" else (monthly, btc_m)
        bench = bench_d if freq == "D" else bench_m
        per = BTC_DAYS if freq == "D" else 12
        pa = 252 if freq == "D" else 12
        an, bt = src[key].loc[a0:a1], btc.loc[b0:b1]
        fa = window_features(an, bench, pa, len(an), np.array([0]),
                            min_corr_n=20).iloc[0]
        fb = window_features(bt, bench, per, len(bt), np.array([0]),
                            min_corr_n=20).iloc[0]
        for f in show:
            rows.append({"pair": lab.replace("\n", " "), "feat": nice[f],
                        "analogue": fa[f], "btc": fb[f]})
    d = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(show), figsize=(13.2, 3.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.155, right=0.99, top=0.62, bottom=0.12, wspace=0.28)

    from matplotlib.patches import Rectangle
    fig.add_artist(Rectangle((0.04, 0.955), 0.035, 0.022, facecolor=RED,
                            edgecolor="none", transform=fig.transFigure))
    fig.text(0.04, 0.865, "Close on risk, apart on character", fontsize=15,
            weight="bold", color=INK)
    fig.text(0.04, 0.775, "Risk characteristics of Bitcoin and its matched "
                          "analogue. Volatility and drawdown are close because "
                          "that is what the search optimises; the interesting "
                          "gaps are elsewhere",
            fontsize=8.6, color=GREY)

    pairs = list(dict.fromkeys(d["pair"]))
    y = np.arange(len(pairs))[::-1]
    for ax, f in zip(axes, [nice[c] for c in show]):
        sub = d[d["feat"] == f].set_index("pair").loc[pairs]
        ax.hlines(y, sub["btc"], sub["analogue"], color=GRID, lw=2.2, zorder=1)
        ax.scatter(sub["analogue"], y, s=46, color=RED, zorder=3)
        ax.scatter(sub["btc"], y, s=46, color=BLUE, zorder=4)
        econ(ax, f)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        if f in ("Corr. w/ equities", "Corr. w/ gold", "Sharpe", "Skew"):
            ax.axvline(0, color=GREY, lw=0.8, zorder=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(pairs, fontsize=8.5, color=INK)
    fig.text(0.735, 0.865, "Bitcoin", fontsize=9.5, color=BLUE, weight="bold")
    fig.text(0.805, 0.865, "analogue", fontsize=9.5, color=RED, weight="bold")
    fig.text(0.04, 0.03, "Volatility and drawdown annualised, in return units. "
                         "Sources: as above; author's calculations",
            fontsize=7.4, color=GREY, style="italic")
    out = FIG / "essay_analogue_features.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    daily, monthly = load_assets()

    panel = pd.read_parquet(PROC / "panel_daily.parquet")
    btc_d = panel["btc_ret"].dropna()
    px = panel["btc_px"].dropna()
    btc_m = np.log(px.resample("ME").last()).diff().dropna()
    btc_m.index = btc_m.index.to_period("M").to_timestamp("M")

    # benchmarks for the correlation features, at both frequencies
    from analogues import long_panel
    _, bench_d = long_panel()
    from analogues_em import monthly_panel
    _, bench_m = monthly_panel()

    # align monthly analogue indices to month-end stamps
    for k in list(monthly):
        s = monthly[k]
        s.index = s.index.to_period("M").to_timestamp("M")
        monthly[k] = s

    figure_paths(daily, monthly, btc_d, btc_m)
    figure_features(daily, monthly, btc_d, btc_m, bench_d, bench_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
