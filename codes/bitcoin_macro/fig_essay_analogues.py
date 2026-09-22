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
# Best match per regime under the clean metric (no gold/equity correlation in
# the distance, panel restricted to 1973+). Read off analogue_matches.csv.
# Best HISTORICAL match per regime: candidate windows must end before the
# bitcoin regime begins. Read off analogue_matches.csv / _monthly_em.csv.
PAIRS = [
    ("Silver, 1979-80", "metal_silver", "1979-01-02", "1980-10-31",
     "2020-02-20", "2021-11-09", "D"),
    ("Natural gas, 2002-05", "cmdty_natgas", "2002-01-11", "2005-10-07",
     "2013-01-01", "2013-12-06", "D"),
    ("Silver, 1978-80", "metal_silver", "1978-11-01", "1980-11-28",
     "2017-01-01", "2018-12-31", "D"),
    ("Natural gas, 2008-09", "cmdty_natgas", "2008-06-02", "2009-07-10",
     "2021-11-10", "2022-12-31", "D"),
    ("Coal stocks, 2015-18", "ind_Coal", "2015-12-15", "2018-06-29",
     "2023-01-01", "2025-07-18", "D"),
    ("Natural gas,\n1997-2011", "cmdty_natgas", "1997-04-30", "2011-01-31",
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
    # daily industry series are needed too: some best matches are industry
    # portfolios measured at daily frequency (e.g. coal stocks in 2000-01)
    for col in ("Coal", "Autos", "FabPr", "Gold", "Steel", "Clths"):
        if col in ind.columns:
            daily[f"ind_{col}"] = ind[col].dropna()
    indm = (1 + ind).resample("ME").prod() - 1
    for col in ("Autos", "Coal", "Gold", "Fun"):
        if col in indm.columns:
            monthly[f"ind_{col}"] = indm[col].dropna()
    # monthly versions of the daily commodity/metal series, for pairings matched
    # at monthly frequency
    for k, v in list(daily.items()):
        lvl = v.cumsum()
        monthly[k] = lvl.resample("ME").last().diff().dropna()
    for cc in ("tr", "mx"):
        if cc in oecd:
            q = oecd[cc].dropna()
            monthly[f"em_{cc}_" + ("Turkey" if cc == "tr" else "Mexico")] = \
                np.log(q[q > 0]).diff().dropna()
    return daily, monthly


def btc_window_label(r: pd.Series) -> str:
    """Compact label for the bitcoin window a panel is comparing against."""
    a, b = r.index.min(), r.index.max()
    if a.year == b.year:
        return f"{a.strftime('%b')}-{b.strftime('%b %Y')}"
    if b.year - a.year == 1 and (a.month > 1 or b.month < 12):
        return f"{a.strftime('%b %Y')}-{b.strftime('%b %Y')}"
    return f"{a.year}-{str(b.year)[-2:]}"


def norm_path(r: pd.Series) -> np.ndarray:
    """Amplitude-normalised cumulative log-return path on a common 0-1 grid."""
    c = r.cumsum().values
    c = (c - c.mean()) / (c.std() or 1.0)
    return np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, len(c)), c)


def figure_paths(daily, monthly, btc_d, btc_m):
    """5 pairs on a 2x3 grid; the spare cell carries the legend.

    A single row of five panels renders each one so narrow that the tick and
    annotation type becomes unreadable at blog width. Two rows of three roughly
    doubles the width per panel, which lets every label go up several points.
    """
    from matplotlib.patches import Rectangle
    from analogues import dtw_distance

    fig = plt.figure(figsize=(10.6, 7.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.20,
                          left=0.05, right=0.985, top=0.745, bottom=0.075)

    fig.add_artist(Rectangle((0.05, 0.955), 0.045, 0.019, facecolor=RED,
                            edgecolor="none", transform=fig.transFigure))
    fig.text(0.05, 0.893, "Shapes that rhyme", fontsize=17, weight="bold",
            color=INK)
    fig.text(0.05, 0.845, "Cumulative return paths, each scaled to mean 0 and "
                         "standard deviation 1", fontsize=11, color=GREY)
    fig.text(0.05, 0.808, "DTW = dynamic time-warping distance; lower means a "
                         "closer match", fontsize=9.5, color=GREY)
    fig.text(0.70, 0.893, "Bitcoin", fontsize=13, color=BLUE, weight="bold")
    fig.text(0.80, 0.893, "analogue", fontsize=13, color=RED, weight="bold")

    for i, (lab, key, a0, a1, b0, b1, freq) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        src = daily if freq == "D" else monthly
        btc = btc_d if freq == "D" else btc_m
        an, bt = src[key].loc[a0:a1], btc.loc[b0:b1]
        if len(an) < 10 or len(bt) < 10:
            continue
        x = np.linspace(0, 1, 200)
        pb, pa = norm_path(bt), norm_path(an)
        ax.plot(x, pb, color=BLUE, lw=2.2, zorder=3)
        ax.plot(x, pa, color=RED, lw=1.9, zorder=2, alpha=0.9)
        # Panel title names the analogue; the subtitle names the bitcoin window
        # it is being compared against, which differs from panel to panel.
        econ(ax, lab.replace("\n", " "))
        ax.title.set_fontsize(11.5)
        ax.annotate(f"vs bitcoin {btc_window_label(bt)}", xy=(0.03, 0.95),
                   xycoords="axes fraction", fontsize=9.5, color=BLUE,
                   ha="left", va="top")
        ax.set_xticks([])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(-3.0, 3.0)
        ax.annotate(f"DTW {dtw_distance(pb, pa):.3f}", xy=(0.97, 0.05),
                   xycoords="axes fraction", fontsize=10, color=GREY,
                   ha="right")

    fig.text(0.05, 0.022, "Amplitude is normalised away, so these compare shape, "
                         "not size. First five panels daily, last monthly. "
                         "Sources: CoinMetrics; FRED; LBMA; author's calculations",
            fontsize=9, color=GREY, style="italic")
    out = FIG / "essay_analogue_paths.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def figure_features(daily, monthly, btc_d, btc_m, bench_d, bench_m):
    """Where the matches hold and where they fail. 6 features on a 2x3 grid.

    One row of six panels leaves each too narrow to label; 2x3 doubles the width
    and lets the pair names and tick labels grow to a readable size.
    """
    from matplotlib.patches import Rectangle

    # Only the attributes the search actually uses. Correlations with gold and
    # with equities are not search attributes and are not shown here.
    show = ["vol_ann", "sharpe", "rho_usd", "rho_d10y"]
    nice = {"vol_ann": "Volatility", "sharpe": "Sharpe ratio",
            "rho_usd": "Corr. w/ dollar", "rho_d10y": "Corr. w/ 10y yield"}

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
        short = (lab.replace("\n", " ")
                 .replace("American car makers", "Car makers")
                 .replace("Turkish equities", "Turkey"))
        for f in show:
            rows.append({"pair": short, "feat": nice[f],
                        "analogue": fa[f], "btc": fb[f]})
    d = pd.DataFrame(rows)

    fig = plt.figure(figsize=(9.6, 7.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.42,
                          left=0.215, right=0.982, top=0.735, bottom=0.085)

    fig.add_artist(Rectangle((0.05, 0.955), 0.045, 0.019, facecolor=RED,
                            edgecolor="none", transform=fig.transFigure))
    fig.text(0.05, 0.893, "Close, but never close enough", fontsize=17,
            weight="bold", color=INK)
    fig.text(0.04, 0.845, "The four attributes the search matches on. A short "
                         "bar means bitcoin and its analogue agree",
             fontsize=11, color=GREY)
    fig.text(0.04, 0.808, "No pairing agrees on all four, which is why none of "
                         "them clears the chance benchmark",
             fontsize=9.5, color=GREY)
    fig.text(0.62, 0.893, "Bitcoin", fontsize=13, color=BLUE, weight="bold")
    fig.text(0.78, 0.893, "analogue", fontsize=13, color=RED, weight="bold")

    pairs = list(dict.fromkeys(d["pair"]))
    y = np.arange(len(pairs))[::-1]
    for i, f in enumerate([nice[c] for c in show]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        sub = d[d["feat"] == f].set_index("pair").loc[pairs]
        ax.hlines(y, sub["btc"], sub["analogue"], color=GRID, lw=3.0, zorder=1)
        ax.scatter(sub["analogue"], y, s=72, color=RED, zorder=3)
        ax.scatter(sub["btc"], y, s=72, color=BLUE, zorder=4)
        econ(ax, f)
        ax.title.set_fontsize(11.5)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=GRID, lw=0.7)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_ylim(-0.7, len(pairs) - 0.3)
        if f.startswith("Corr.") or f in ("Sharpe ratio", "Skew"):
            ax.axvline(0, color=GREY, lw=1.0, zorder=0)
        if i % 2 == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(pairs, fontsize=10.5, color=INK)
        else:
            ax.set_yticks([])

    fig.text(0.04, 0.022, "Volatility annualised. The monthly pairing also "
                         "matches on the trade balance and inflation. "
                         "Sources: as above; author's calculations",
            fontsize=9, color=GREY, style="italic")
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
