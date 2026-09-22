"""Essay figure: bitcoin versus the dollar.

Two panels, each a single message and a single series, so no dual axis and no
legend is needed:

  A. The rolling correlation with the broad trade-weighted dollar, which was
     mildly POSITIVE until 2018 and has been reliably negative since. The sign
     flip is the point; the Feb-2018 break date comes from the same sup-F test
     used elsewhere (see breaks_correlation.py).

  B. Mean bitcoin return by quintile of weekly dollar return, post-2018. A
     monotone staircase is much harder to dismiss as an artefact of two or three
     outlying weeks than a correlation coefficient is, which is why it is here.

Note the sign convention carefully: ret_usd RISES when the dollar strengthens.
A negative coefficient therefore means bitcoin gains when the dollar WEAKENS.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Rectangle

from fetchlib import PROC, ROOT

BLUE = "#006BA2"
RED = "#E3120B"
INK = "#121317"
GREY = "#5b5f63"
GRID = "#d9d9d6"
SURFACE = "#ffffff"
FIG = ROOT / "outputs" / "figures"

BREAK = "2018-02-02"


def econ(ax, title, ylab=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, colors=GREY, labelsize=10)
    ax.tick_params(axis="x", length=3, colors=GREY, labelsize=10)
    ax.yaxis.tick_right()
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9.5, color=GREY)


def main() -> int:
    w = pd.read_parquet(PROC / "panel_weekly.parquet")
    w["btc_xs"] = w["btc_ret"] - w["rf"].fillna(0)
    d = w.loc["2013-01-01":][["btc_xs", "ret_usd"]].dropna()

    fig = plt.figure(figsize=(9.2, 7.2))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, hspace=0.50, left=0.08, right=0.90,
                          top=0.78, bottom=0.11)

    fig.add_artist(Rectangle((0.08, 0.945), 0.05, 0.018, facecolor=RED,
                             edgecolor="none", transform=fig.transFigure))
    fig.text(0.08, 0.885, "The debasement trade, at last", fontsize=17,
             color=INK, weight="bold")
    fig.text(0.08, 0.838, "Bitcoin used to ignore the dollar. Since 2018 it has "
                          "risen when the dollar falls",
             fontsize=11, color=GREY)

    # ---- panel A: rolling correlation ------------------------------------
    ax = fig.add_subplot(gs[0])
    rc = d["btc_xs"].rolling(52).corr(d["ret_usd"]).dropna()
    ax.plot(rc.index, rc.values, color=BLUE, lw=2.0)
    ax.axhline(0, color=INK, lw=1.0)
    ax.axvline(pd.Timestamp(BREAK), color=RED, lw=1.6, ls=(0, (5, 3)))
    ax.annotate("correlation flips sign,\nFebruary 2018",
                xy=(pd.Timestamp(BREAK), 0.20), xytext=(14, 0),
                textcoords="offset points", fontsize=9.5, color=RED,
                ha="left", va="center")
    econ(ax, "Correlation of weekly returns with the broad trade-weighted "
             "dollar, 52-week rolling", "correlation")
    ax.set_ylim(-0.58, 0.34)
    ax.text(pd.Timestamp("2015-06-01"), -0.50, "mean +0.07", fontsize=10,
            color=GREY, ha="center", weight="bold")
    ax.text(pd.Timestamp("2022-06-01"), -0.50, "mean -0.23", fontsize=10,
            color=GREY, ha="center", weight="bold")

    # ---- panel B: quintile staircase ------------------------------------
    ax2 = fig.add_subplot(gs[1])
    s = d.loc[BREAK:].copy()
    s["q"] = pd.qcut(s["ret_usd"], 5, labels=False)
    g = s.groupby("q").agg(dollar=("ret_usd", "mean"), btc=("btc_xs", "mean"))
    xs = np.arange(5)
    cols = [BLUE if v > 0 else RED for v in g["btc"]]
    ax2.bar(xs, g["btc"] * 100, width=0.62, color=cols, edgecolor=SURFACE, lw=1.4,
            zorder=3)
    ax2.axhline(0, color=INK, lw=1.0)
    econ(ax2, "Mean weekly bitcoin return, by quintile of weekly dollar return\n"
              "since February 2018 (left = dollar falling)",
         "bitcoin return, % per week")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{v * 100:+.2f}%" for v in g["dollar"]], fontsize=10,
                        color=INK)
    for x, v in zip(xs, g["btc"] * 100):
        off = 8 if v > 0 else -16
        ax2.annotate(f"{v:+.2f}", xy=(x, v), xytext=(0, off),
                     textcoords="offset points", ha="center", fontsize=10,
                     color=INK, weight="bold")
    ax2.set_ylim(min(g["btc"] * 100) - 1.1, max(g["btc"] * 100) + 1.3)

    r = sm.OLS(s["btc_xs"], sm.add_constant(s["ret_usd"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4})
    fig.text(0.08, 0.033,
             f"Slope {r.params['ret_usd']:.2f} (t = {r.tvalues['ret_usd']:.2f}), "
             f"n = {int(r.nobs)} weeks.  Sources: CoinMetrics; FRED; "
             f"author's calculations",
             fontsize=9, color=GREY, style="italic")

    out = FIG / "essay_dollar.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
