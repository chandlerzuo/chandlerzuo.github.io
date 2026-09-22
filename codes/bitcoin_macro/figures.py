"""Figures for the report.

Palette: the dataviz reference instance's categorical slots, used in fixed order
(never cycled). Light surface. One y-axis per panel -- where two measures of
different scale must be shown together they get stacked panels sharing an x-axis,
never a dual axis. Grid and spines are recessive; text uses ink tokens rather
than series colors.
"""
from __future__ import annotations

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fetchlib import PROC, ROOT

warnings.filterwarnings("ignore")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

REGIME_SPANS = [
    ("R1", "2013-01-01", "2013-12-06"),
    ("R2", "2013-12-07", "2016-12-31"),
    ("R3", "2017-01-01", "2018-12-31"),
    ("R4", "2019-01-01", "2020-02-19"),
    ("R5", "2020-02-20", "2021-11-09"),
    ("R6", "2021-11-10", "2022-12-31"),
    ("R7", "2023-01-01", "2025-07-18"),
    ("R8", "2025-07-19", "2026-09-30"),
]


def style(ax, title=None, ylab=None, xlab=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if ylab:
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
    if xlab:
        ax.set_xlabel(xlab, color=INK2, fontsize=9)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote figures/{name}")


# ----------------------------------------------------------------- figure 1
def fig_decomposition():
    """The central result: beta decomposed into correlation and relative vol."""
    rb = pd.read_parquet(PROC / "tvp" / "roll_ret_mkt.parquet")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True,
                            gridspec_kw={"hspace": 0.28})

    ax = axes[0]
    ax.plot(rb.index, rb["beta"], color=S[0], lw=2)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 4)))
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (1, 3)))
    style(ax, "Bitcoin's beta to US equities is the product of two very "
              "different stories", "beta (52-week)")
    ax.annotate("beta ~1.2 here is\nlow rho x huge vol ratio",
               xy=(pd.Timestamp("2013-01-01"), 1.18), xytext=(14, -66),
               textcoords="offset points", fontsize=8, color=INK2,
               ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("beta ~1.8 here is\nhigh rho x smaller vol ratio",
               xy=(pd.Timestamp("2026-06-01"), 1.78), xytext=(-186, -30),
               textcoords="offset points", fontsize=8, color=INK2,
               ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    ax = axes[1]
    ax.plot(rb.index, rb["rho"], color=S[1], lw=2)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 4)))
    style(ax, "Correlation: ~zero before 2020; elevated but unstable 2020-2023; "
              "back to zero in 2024; highest on record now",
          "correlation (52-week)")

    ax = axes[2]
    ax.plot(rb.index, rb["sd_ratio"], color=S[2], lw=2)
    style(ax, "Relative volatility: spiky, but each peak is lower -- 15x in "
              "2014 and 2018, ~7x in 2022, ~3.5x now",
          "sd(BTC) / sd(equities)")

    for ax in axes:
        for _, a, b in REGIME_SPANS:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#000000",
                      alpha=0.022, lw=0, zorder=0)
    save(fig, "01_beta_decomposition.png")


# ----------------------------------------------------------------- figure 2
def fig_regime_map():
    p = pd.read_parquet(PROC / "panel_daily.parquet")
    px = p["btc_px"].dropna()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(px.index, px.values, color=S[0], lw=1.4)
    ax.set_yscale("log")
    style(ax, "Bitcoin price with the reconciled regime chronology",
          "USD (log scale)")
    ymin = px.min()
    for i, (lab, a, b) in enumerate(REGIME_SPANS):
        A, B = pd.Timestamp(a), pd.Timestamp(b)
        ax.axvspan(A, B, color="#000000", alpha=0.03 if i % 2 else 0.0, lw=0)
        ax.text(A + (B - A) / 2, ymin * 1.15, lab, ha="center", fontsize=8,
               color=INK2)
    for d, lab in (("2013-12-06", "statistically\nrobust break"),
                   ("2025-07-18", "statistically\nrobust break")):
        ax.axvline(pd.Timestamp(d), color=S[1], lw=1.6, ls=(0, (5, 3)))
        ax.annotate(lab, xy=(pd.Timestamp(d), px.max() * 0.55),
                   xytext=(6, 0), textcoords="offset points",
                   fontsize=8, color=S[1], va="center")
    save(fig, "02_regime_map.png")


# ----------------------------------------------------------------- figure 3
def fig_horse_race():
    f = ROOT / "outputs" / "tables" / "horse_race_prior_regimes.csv"
    if not f.exists():
        return
    hr = pd.read_csv(f)
    cols = [("r2_liquidity_realrate", "Liquidity / real rates"),
            ("r2_risk_asset_tech", "Risk asset / tech"),
            ("r2_digital_gold", "Digital gold")]
    cols = [(c, l) for c, l in cols if c in hr.columns]
    x = np.arange(len(hr))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (c, lab) in enumerate(cols):
        ax.bar(x + (i - 1) * w, hr[c], width=w - 0.02, label=lab,
              color=S[i], edgecolor=SURFACE, lw=1.2, zorder=3)
    style(ax, "Which macro thesis explains Bitcoin? The winner rotates by regime",
          "share of weekly return variance explained (R-squared)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" ", "\n", 1) for p in hr["period"]],
                      fontsize=8, color=INK2)
    leg = ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, "03_thesis_horse_race.png")


# ----------------------------------------------------------------- figure 4
def fig_sharpe_ci():
    f = ROOT / "outputs" / "tables" / "regime_profiles.csv"
    if not f.exists():
        return
    pr = pd.read_csv(f)
    y = np.arange(len(pr))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hlines(y, pr["sharpe_lo95"], pr["sharpe_hi95"], color=MUTED, lw=2)
    ax.scatter(pr["sharpe"], y, s=64, color=S[0], zorder=4,
              edgecolor=SURFACE, lw=1.5)
    ax.axvline(0, color=INK2, lw=1)
    style(ax, "Regime Sharpe ratios: every interval is wide, and the spread "
              "across regimes\nis not distinguishable from random splitting "
              "(placebo p = 0.22)",
          None, "annualised Sharpe (95% block-bootstrap interval)")
    ax.set_yticks(y)
    ax.set_yticklabels(pr["regime"], fontsize=9, color=INK2)
    save(fig, "04_regime_sharpe.png")


# ----------------------------------------------------------------- figure 5
def fig_analogue_null():
    f = ROOT / "outputs" / "tables" / "analogue_null_tests.csv"
    if not f.exists():
        return
    nt = pd.read_csv(f)
    x = np.arange(len(nt))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - 0.18, nt["btc_best_maha"], width=0.34, label="Bitcoin's best match",
          color=S[0], edgecolor=SURFACE, lw=1.2, zorder=3)
    ax.bar(x + 0.18, nt["null_median"], width=0.34,
          label="a random asset's best match (null)", color=S[3],
          edgecolor=SURFACE, lw=1.2, zorder=3)
    style(ax, "No credible historical analogue: Bitcoin's closest match is always "
              "far worse\nthan a random asset's closest match -- though the gap "
              "narrows as volatility falls",
          "Mahalanobis distance (lower = more similar)")
    ax.set_xticks(x)
    ax.set_xticklabels([r.split()[0] + "\n" + " ".join(r.split()[1:2])
                       for r in nt["regime"]], fontsize=8, color=INK2)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, "05_analogue_null.png")


# ----------------------------------------------------------------- figure 6
def fig_rho_vol_space():
    """Where each regime sits in (correlation, relative volatility) space.

    Encoding choice: the eight regimes are ORDERED in time, and there are more
    of them than the categorical palette has slots -- cycling hues would imply
    R1 and R8 are the same thing. So this uses a single-hue sequential ramp
    (light = early, dark = late) plus a connecting trajectory, which is both the
    correct encoding for an ordered variable and self-documenting about direction.
    """
    rb = pd.read_parquet(PROC / "tvp" / "roll_ret_mkt.parquet")
    ramp = plt.cm.Blues(np.linspace(0.32, 0.95, len(REGIME_SPANS)))
    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    cen = []
    for i, (lab, a_, b_) in enumerate(REGIME_SPANS):
        seg = rb.loc[a_:b_]
        if seg.empty:
            continue
        ax.scatter(seg["rho"], seg["sd_ratio"], s=11, alpha=0.30,
                  color=ramp[i], lw=0, zorder=2)
        cen.append((lab, seg["rho"].mean(), seg["sd_ratio"].mean(), ramp[i]))

    ax.plot([c[1] for c in cen], [c[2] for c in cen], color=MUTED, lw=1.2,
           ls=(0, (5, 3)), zorder=3)
    # label offsets tuned per point to avoid the R1/R2/R3 cluster colliding
    offs = {"R1": (-26, 12), "R2": (16, 10), "R3": (2, -22), "R4": (-24, -14),
            "R5": (-4, 15), "R6": (14, 6), "R7": (-26, 8), "R8": (12, -14)}
    for lab, x, y, c in cen:
        ax.scatter([x], [y], s=165, color=c, edgecolor=SURFACE, lw=2, zorder=6)
        dx, dy = offs.get(lab, (0, 13))
        ax.annotate(lab, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                   ha="center", fontsize=9.5, color=INK, weight="bold", zorder=7)

    for bval, ls in ((0.5, (0, (1, 3))), (1.0, (0, (4, 4))), (2.0, (0, (1, 3)))):
        r = np.linspace(0.05, 0.62, 80)
        ax.plot(r, bval / r, color=MUTED, lw=0.8, ls=ls, zorder=1)
        ax.text(0.615, min(bval / 0.615, 11.4), f"beta={bval}", fontsize=7.5,
               color=MUTED, va="center", ha="left")

    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_ylim(0, 12)
    ax.set_xlim(-0.26, 0.70)
    style(ax, "Regimes in correlation-vs-relative-volatility space\n"
              "Bitcoin has travelled right (more coupled) and down (less wild); "
              "shade = early to late",
          "sd(BTC) / sd(equities)", "correlation with US equities (52-week)")
    save(fig, "06_rho_vol_space.png")



# ----------------------------------------------------------------- figure 7
def fig_control_stability():
    """Follow-up 4: macro loadings before and after crypto-native controls."""
    f = ROOT / "outputs" / "tables" / "crypto_control_stability.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d["regime"] != "full 2014+"]
    # show only cells significant in at least one specification
    d = d[(d["t_macro"].abs() > 1.96) | (d["t_ctrl"].abs() > 1.96)].copy()
    if d.empty:
        return
    d["label"] = d["regime"].str.split().str[0] + "  " + d["var"]
    d = d.sort_values("b_macro")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.52 * len(d) + 1.6)))
    ax.hlines(y, d["b_macro"], d["b_ctrl"], color=MUTED, lw=1.4, zorder=2)
    ax.scatter(d["b_macro"], y, s=74, color=S[0], zorder=4,
              edgecolor=SURFACE, lw=1.5, label="macro only")
    ax.scatter(d["b_ctrl"], y, s=74, color=S[1], zorder=4,
              edgecolor=SURFACE, lw=1.5, label="+ crypto-native controls")
    ax.axvline(0, color=INK2, lw=1)
    style(ax, "Macro loadings mostly survive crypto-native controls\n"
              "(only loadings significant in at least one specification shown)",
          None, "coefficient")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=9, color=INK2)
    for yi, vd in zip(y, d["verdict"]):
        ax.annotate(vd, xy=(1.005, yi), xycoords=("axes fraction", "data"),
                   fontsize=8, color=INK2 if vd != "KILLED" else S[1],
                   va="center")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, "07_control_stability.png")


# ----------------------------------------------------------------- figure 8
def fig_em_null():
    """Follow-up 5: does adding emerging markets find an analogue? (No.)"""
    f = ROOT / "outputs" / "tables" / "analogue_em_null_tests.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.bar(x - 0.18, d["best"], width=0.34, color=S[0], edgecolor=SURFACE,
          lw=1.2, zorder=3, label="Bitcoin's best match")
    ax.bar(x + 0.18, d["null_median"], width=0.34, color=S[3],
          edgecolor=SURFACE, lw=1.2, zorder=3, label="random asset's best (null)")
    style(ax, "Adding emerging markets does not produce an analogue\n"
              "Monthly panel, 77 assets incl. 12 country indices back to 1957",
          "Mahalanobis distance (lower = more similar)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" (", "\n(").replace(" 20", "\n20")
                       for t in d["target"]], fontsize=7.5, color=INK2)
    for xi, (p, n) in enumerate(zip(d["p"], d["n_em_top20"])):
        ax.annotate(f"p={p:.2f}", xy=(xi, max(d['best'][xi], d['null_median'][xi])),
                   xytext=(0, 5), textcoords="offset points", ha="center",
                   fontsize=8, color=INK2)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, "08_em_analogue_null.png")


def main() -> int:
    print("building figures")
    for fn in (fig_decomposition, fig_regime_map, fig_horse_race,
               fig_sharpe_ci, fig_analogue_null, fig_rho_vol_space,
               fig_control_stability, fig_em_null):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {fn.__name__} FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
