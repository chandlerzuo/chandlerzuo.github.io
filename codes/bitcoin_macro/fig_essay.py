"""Chart for the essay.

Palette: substituted for an Economist-style brand instance (blue #006BA2,
accent red #E3120B) per the dataviz skill's "swap the parameters, keep the
method" rule. One series per panel, so there is no categorical adjacency to
validate and no legend is needed -- the panel title names the series. Two
measures of different scale are shown as stacked small multiples sharing an
x-axis, never a dual axis.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from fetchlib import PROC, ROOT

BLUE = "#006BA2"
RED = "#E3120B"
INK = "#121317"
GREY = "#5b5f63"
GRID = "#d9d9d6"
SURFACE = "#ffffff"

FIG = ROOT / "outputs" / "figures"


def econ_axes(ax, panel_title, ylab=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, colors=GREY, labelsize=8.5)
    ax.tick_params(axis="x", length=3, colors=GREY, labelsize=8.5)
    ax.yaxis.tick_right()
    ax.set_title(panel_title, loc="left", fontsize=9.5, color=INK, pad=6)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8, color=GREY)


def main() -> int:
    rb = pd.read_parquet(PROC / "tvp" / "roll_ret_mkt.parquet")
    rb = rb.loc["2012-01-01":]

    fig = plt.figure(figsize=(7.4, 6.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, hspace=0.30, left=0.07, right=0.90,
                         top=0.78, bottom=0.10)

    # Economist furniture: red tab, headline, subtitle
    fig.add_artist(Rectangle((0.07, 0.945), 0.055, 0.018,
                            facecolor=RED, edgecolor="none",
                            transform=fig.transFigure))
    fig.text(0.07, 0.885, "Two conversions, six years apart", fontsize=16,
            color=INK, weight="bold", ha="left")
    fig.text(0.07, 0.845, "Bitcoin's correlation shifted in 2019; its beta only "
                          "in 2025, because volatility kept falling",
            fontsize=10, color=GREY, ha="left")

    # ---- panel 1: correlation -------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.plot(rb.index, rb["rho"], color=BLUE, lw=1.8, solid_joinstyle="round")
    ax.axhline(0, color=GREY, lw=0.8)
    econ_axes(ax, "Correlation of weekly returns, 52-week rolling")
    ax.set_ylim(-0.32, 0.68)

    brk = pd.Timestamp("2019-08-16")
    ax.axvline(brk, color=RED, lw=1.5, ls=(0, (5, 3)), zorder=4)
    ax.annotate("correlation breaks,\nAugust 2019", xy=(brk, 0.60),
               xytext=(-108, 0), textcoords="offset points", fontsize=8,
               color=RED, ha="left", va="center")
    ax.text(pd.Timestamp("2015-06-01"), -0.265, "mean 0.04", fontsize=8.5,
           color=GREY, ha="center", weight="bold")
    ax.text(pd.Timestamp("2023-06-01"), -0.265, "mean 0.30", fontsize=8.5,
           color=GREY, ha="center", weight="bold")
    # x labels belong on the bottom panel only
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", length=0)

    # ---- panel 2: relative volatility ------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.plot(rb.index, rb["sd_ratio"], color=BLUE, lw=1.8)
    ax2.axvline(pd.Timestamp("2025-07-18"), color=RED, lw=1.5,
               ls=(0, (5, 3)), zorder=4)
    ax2.annotate("beta breaks,\nJuly 2025", xy=(pd.Timestamp("2025-07-18"), 12.6),
                xytext=(-96, 0), textcoords="offset points", fontsize=8,
                color=RED, ha="left", va="center")
    econ_axes(ax2, "Bitcoin's volatility as a multiple of the stockmarket's")
    ax2.set_ylim(0, 16)
    for yr, val, lab, dx, dy in ((2014, 15.4, "15x", 26, -4),
                                (2018, 14.9, "15x", 26, -4),
                                (2026, 3.4, "3x", 16, 6)):
        ax2.annotate(lab, xy=(pd.Timestamp(f"{yr}-03-01"), val),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, ha="center", weight="bold")

    fig.text(0.07, 0.035,
            "Sources: CoinMetrics; Kenneth French data library; author's calculations",
            fontsize=7.5, color=GREY, ha="left", style="italic")

    out = FIG / "essay_coupled_and_calmer.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
