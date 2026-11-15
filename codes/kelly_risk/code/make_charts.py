"""Figures for the Kelly / drawdown / stopping post, in the same Economist style as the other
posts. Reads ../data/*.csv (run sim.py first). Outputs -> ../outputs."""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))
os.makedirs(OUT, exist_ok=True)

EC = {"red": "#E3120B", "blue": "#006BA2", "cyan": "#3EBCD2", "green": "#379A8B",
      "yellow": "#EBB434", "olive": "#B4BA39", "purple": "#9A607F", "ink": "#0C0C0C",
      "grey": "#758D99", "grid": "#D9D9D9", "line": "#B7C6CF", "shade": "#E9EDF0"}
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": EC["ink"], "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": EC["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.color": EC["ink"], "axes.labelcolor": EC["ink"], "xtick.color": EC["ink"],
    "ytick.color": EC["ink"], "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})
SOURCE = "Source: author's simulations (20,000 traders, 1,260 daily bets each)"

A = pd.read_csv(os.path.join(DATA, "results_A.csv"))
B = pd.read_csv(os.path.join(DATA, "results_B.csv"))
A = A[A.get("segment").isna()] if "segment" in A else A
Bseg = B[B.segment.notna()]; B = B[B.segment.isna()]
fan = pd.read_csv(os.path.join(DATA, "fan_bands.csv"))
fr = pd.read_csv(os.path.join(DATA, "frontier.csv"))
th = pd.read_csv(os.path.join(DATA, "theory_hit.csv"))
tb = pd.read_csv(os.path.join(DATA, "theory_barrier.csv"))


def frame(fig, title, subtitle, note="", top=0.97, source=None):
    fig.add_artist(Line2D([0, 1], [top, top], color=EC["red"], lw=1.3, transform=fig.transFigure))
    fig.add_artist(Rectangle((0, top - 0.022), 0.045, 0.022, color=EC["red"], lw=0, transform=fig.transFigure))
    fig.text(0, top - 0.045, title, fontsize=14, weight="bold", va="top")
    fig.text(0, top - 0.092, subtitle, fontsize=10.5, va="top")
    fig.text(0, 0, (note + "\n" if note else "") + (source or SOURCE), fontsize=7.8, color=EC["grey"], va="bottom")


def strip(ax, bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)


SHORT = {"Fixed stake, 5% of start": "Fixed stake, 5% of start capital", "1% of equity": "1% of equity (1/6 Kelly)",
         "Half Kelly, drawdown throttle": "Half Kelly + drawdown throttle",
         "Half Kelly, quit at 25% drawdown": "Half Kelly, quit at 25% drawdown",
         "Half Kelly, quit at -30% or +100%": "Half Kelly, quit at −30% / +100%",
         "Bayesian half Kelly": "Bayesian half Kelly",
         "Bayesian half Kelly, quit at 25% drawdown": "Bayesian half Kelly, quit at 25% DD"}
nm = lambda s: SHORT.get(s, s)


# ============================================================ FIG 1: fans ====
def fig1():
    names = ["Quarter Kelly", "Half Kelly", "Full Kelly", "Double Kelly"]
    fig, axs = plt.subplots(1, 4, figsize=(11, 4.3), sharey=True)
    for ax, n in zip(axs, names):
        d = fan[fan.strategy == n]; t = d.t / 252
        ax.fill_between(t, d.p5.clip(lower=1e-4), d.p95, color=EC["shade"], lw=0, label="5–95%")
        ax.fill_between(t, d.p25.clip(lower=1e-4), d.p75, color=EC["line"], lw=0, label="25–75%")
        ax.plot(t, d.p50, color=EC["red"], lw=1.8, label="Median")
        ax.axhline(1, color=EC["ink"], lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(1e-3, 1e3); ax.set_xlim(0, 5)
        ax.set_title(n, loc="left", fontsize=10.5, weight="bold")
        ax.set_xticks([0, 1, 2, 3, 4, 5]); strip(ax)
        ax.set_xlabel("Years", fontsize=9)
        end = A.set_index("strategy").loc[n]
        ax.text(0.04, 0.05, f"median {end.median_multiple:.1f}×\n5th pct {end.p5_multiple:.2f}×",
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axs[0].set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3])
    axs[0].set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1,000"])
    axs[0].set_ylabel("Account value (× starting capital, log scale)", fontsize=9)
    frame(fig, "Each step up in bet size fattens the tails; past Kelly, even the median stops growing",
          "Wealth over 5 years, 53% win probability at even odds (Kelly fraction 6%); same random outcomes in every panel")
    fig.tight_layout(rect=(0, 0.07, 1, 0.85))
    fig.savefig(os.path.join(OUT, "fig1_fans.png")); plt.close(fig)


# ============================================================ FIG 2: bars ====
def fig2():
    order = ["Fixed stake, 5% of start", "1% of equity", "Quarter Kelly", "Half Kelly", "Full Kelly",
             "Double Kelly", "Half Kelly, drawdown throttle", "Bayesian half Kelly"]
    d = A.set_index("strategy").loc[order][::-1]
    labels = [nm(s) for s in d.index]; y = np.arange(len(d))
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    specs = [("median_cagr", "Median annual growth, %", 100, EC["blue"]),
             ("p_mdd50", "Chance of a ≥50% drawdown, %", 100, EC["red"]),
             ("p_lose_money", "Chance of ending below the start, %", 100, EC["grey"])]
    for ax, (col, ttl, k, c) in zip(axs, specs):
        v = d[col] * k
        ax.barh(y, v, height=0.62, color=c, lw=0, zorder=2)
        for i, x in enumerate(v):
            ax.text(x + max(v.max(), 1) * 0.015, i, f"{x:.1f}" if abs(x) < 10 else f"{x:.0f}", va="center", fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        ax.set_title(ttl, loc="left", fontsize=10, weight="bold"); ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5); ax.grid(axis="y", visible=False)
        ax.xaxis.tick_top(); strip(ax, bottom=False); ax.axvline(0, color=EC["ink"], lw=0.9)
        ax.set_ylim(-0.6, len(d) - 0.3)
        ax.set_xlim(min(0, v.min() * 1.2) if col == "median_cagr" else 0, max(v.max() * 1.18, 1))
    axs[0].tick_params(axis="y", length=0)
    frame(fig, "Growth rises with bet size; so does every risk",
          "Five-year results for a 53% win probability at even odds, edge known exactly (scenario A)",
          note="A drawdown is a fall from the account's running peak. The fixed-stake trader's ending balance is 0 in 8.7% of runs (ruin).")
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    fig.savefig(os.path.join(OUT, "fig2_strategies.png")); plt.close(fig)


# ============================================================ FIG 3: theory ==
def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axs[0]
    cs = np.linspace(0.15, 1.6, 100)
    for x, col in ((0.5, EC["blue"]), (0.7, EC["cyan"])):
        ax.plot(cs, x ** ((2 - cs) / cs) * 100, color=col, lw=1.8)
        d = th[th.x == x]
        ax.scatter(d.c, d.sim * 100, s=34, color=col, zorder=4, edgecolor="white", lw=0.6)
    ax.text(0.03, 80, "…falls to 70% of its start", color=EC["cyan"], fontsize=9, weight="bold")
    ax.text(0.03, 70, "…falls to 50% of its start", color=EC["blue"], fontsize=9, weight="bold")
    ax.axvline(1, color=EC["grey"], lw=1, ls=":")
    ax.text(1.02, 3, "Full\nKelly", color=EC["grey"], fontsize=8.5)
    ax.set_xlabel("Bet size, as a multiple of the Kelly fraction", fontsize=9)
    ax.set_title("Probability the account ever…", loc="left", fontsize=10.5, weight="bold")
    ax.set_ylim(0, 100); ax.set_xlim(0, 1.65); strip(ax)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax = axs[1]
    d = tb[np.isfinite(tb.target)]; u = np.linspace(1.05, 6, 200)
    thv, b = tb.theta.iloc[0], np.log(1 / 0.7)
    a = np.log(u)
    ax.plot(u, (1 - np.exp(-thv * a)) / (np.exp(thv * b) - np.exp(-thv * a)) * 100, color=EC["red"], lw=1.8)
    ax.scatter(d.target, d.sim_p_floor * 100, s=34, color=EC["red"], zorder=4, edgecolor="white", lw=0.6)
    inf = tb[~np.isfinite(tb.target)].iloc[0]
    ax.axhline(inf.theory_p_floor * 100, color=EC["grey"], lw=1, ls=":")
    ax.text(1.1, inf.theory_p_floor * 100 + 1.2, f"never stop on gains: {inf.theory_p_floor*100:.0f}%", color=EC["grey"], fontsize=8.5)
    ax.set_xlabel("Profit target, as a multiple of starting capital", fontsize=9)
    ax.set_title("Half Kelly: chance of hitting the 30% floor before the target", loc="left", fontsize=10.5, weight="bold")
    ax.set_ylim(0, 40); ax.set_xlim(1, 6); strip(ax)
    ax.set_yticks([0, 10, 20, 30, 40]); ax.set_yticklabels(["0", "10", "20", "30", "40%"])
    frame(fig, "Two formulas pin down the risk of ruin and what a stopping rule does to it",
          "Lines: theory (exponent 2m/s² from the mean m and variance s² of log-wealth growth per bet); dots: simulation",
          note="Left: 12,000 bets, 5,000 paths per bet size. Right: half Kelly, 20,000 paths. Dots sit slightly below the lines because discrete bets can jump past a barrier.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.86))
    fig.savefig(os.path.join(OUT, "fig3_theory.png")); plt.close(fig)


# ============================================================ FIG 4: frontier
def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    risk = lambda p5: (1 - np.asarray(p5)) * 100
    for ax, tag, big in zip(axs, "AB", (A, B)):
        f = fr[fr.scenario == tag]
        s = f[f.family == "size"].sort_values("param")
        s = s[(s.param <= 1.5) & (s.param >= (0.2 if tag == "A" else 0.1))]
        ax.plot(risk(s.p5_multiple), s.median_cagr * 100, color=EC["blue"], lw=1.8, marker="o", ms=4)
        for _, r in s.iterrows():
            if r.param in ((0.3, 0.5, 1.0, 1.5) if tag == "B" else (0.3, 0.5, 1.0, 1.5)):
                ax.annotate(f"{r.param:g}×", (risk(r.p5_multiple), r.median_cagr * 100), xytext=(6, 4),
                            textcoords="offset points", fontsize=8.5, color=EC["blue"])
        q = f[f.family == "quit_c0.5"].sort_values("param")
        ax.plot(risk(q.p5_multiple), q.median_cagr * 100, color=EC["red"], lw=1.8, marker="s", ms=4)
        for _, r in q.iterrows():
            if r.param in (0.10, 0.40):
                ax.annotate(f"quit at {r.param*100:.0f}% DD", (risk(r.p5_multiple), r.median_cagr * 100),
                            xytext=(0, -14), textcoords="offset points", fontsize=8.5, color=EC["red"], ha="center")
        b_ = big.set_index("strategy")
        offs = {"A": ((8, 8), (8, -14)), "B": ((-8, 12), (10, 6))}[tag]
        for (n_, col), off in zip((("Bayesian half Kelly", EC["green"]), ("Half Kelly, drawdown throttle", EC["purple"])), offs):
            r = b_.loc[n_]
            ax.scatter(risk(r.p5_multiple), r.median_cagr * 100, s=60, color=col, zorder=5, marker="D")
            ax.annotate(nm(n_), (risk(r.p5_multiple), r.median_cagr * 100), xytext=off, textcoords="offset points",
                        fontsize=8.5, color=col, weight="bold", ha="right" if off[0] < 0 else "left")
        ax.axhline(0, color=EC["ink"], lw=0.9)
        ax.set_title({"A": "A. The edge is known", "B": "B. The edge is uncertain (and overestimated)"}[tag],
                     loc="left", fontsize=10.5, weight="bold")
        ax.set_xlabel("Bad-luck outcome: change in capital at the 5th percentile, loss shown positive (%)  →  riskier", fontsize=8.5)
        strip(ax, bottom=True)
    axs[0].set_xlim(-15, 105); axs[1].set_xlim(-30, 105)
    axs[0].set_ylabel("Median annual growth, %", fontsize=9)
    frame(fig, "Shrinking the bet beats stopping: more growth for the same bad-luck outcome",
          "Risk–growth frontier. Blue: bet size as a multiple of Kelly. Red: half Kelly with a permanent quit after a drawdown of the given size",
          note="B: the trader sizes to a 53% win rate but the truth varies across traders, p ~ N(52%, 2%). Bayesian: Beta(200,200) prior on p.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig4_frontier.png")); plt.close(fig)


# ============================================================ FIG 5: segments
def fig5():
    order = ["Half Kelly", "1% of equity", "Half Kelly, quit at 25% drawdown", "Bayesian half Kelly",
             "Bayesian half Kelly, quit at 25% drawdown"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, seg, ttl, col in zip(axs, ("edge", "no_edge"), ("Traders whose edge is real (p > 50%): 85%",
                                                            "Traders with no edge (p ≤ 50%): 15%"),
                                 (EC["blue"], EC["red"])):
        d = Bseg[Bseg.segment == seg].set_index("strategy").loc[order]
        ax.barh(y, d.median_multiple, height=0.6, color=col, lw=0, zorder=2)
        ax.scatter(d.p5_multiple, y, color=EC["ink"], s=18, zorder=5)
        for yi, (m_, p5) in zip(y, zip(d.median_multiple, d.p5_multiple)):
            ax.text(max(m_, p5) + 0.1, yi, f"{m_:.2f}×", va="center", fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        ax.axvline(1, color=EC["ink"], lw=0.9); ax.set_xlim(0, 4.6)
        ax.set_title(ttl, loc="left", fontsize=10.5, weight="bold")
        ax.set_yticks(y); ax.set_yticklabels([nm(s) for s in order]); ax.grid(axis="y", visible=False)
        ax.xaxis.tick_top(); strip(ax, bottom=False)
    frame(fig, "Stopping rules and learning rescue the traders with no edge, at a price for the rest",
          "Scenario B, five years: median final wealth (bar) and 5th percentile (dot), × starting capital",
          note="Black line: starting capital. Quitting is permanent; a trader who quits keeps what is left.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.80))
    fig.savefig(os.path.join(OUT, "fig5_uncertain_edge.png")); plt.close(fig)


# ============================================================ FIG 6: growth hump
def fig6():
    """Growth rate and ruin exponent against bet size: the case for fractional Kelly."""
    c = np.linspace(0.02, 2.0, 300)
    g = c * (2 - c)                                   # growth as a share of the Kelly maximum
    lam = (2 - c) / c
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.plot(c, g * 100, color=EC["blue"], lw=2)
    ax.fill_between(c, 0, g * 100, where=c <= 1, color=EC["shade"], lw=0)
    for k in (0.25, 0.5, 1.0, 1.5):
        ax.scatter(k, k * (2 - k) * 100, color=EC["blue"], s=34, zorder=4)
        ax.annotate(f"{k * (2 - k) * 100:.0f}%", (k, k * (2 - k) * 100), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, weight="bold")
    ax.axhline(0, color=EC["ink"], lw=0.9)
    ax.set_xlim(0, 2); ax.set_ylim(0, 115); strip(ax)
    ax.set_xlabel("Bet size, as a multiple of the Kelly fraction", fontsize=9)
    ax.set_title("Long-run growth, as % of the maximum", loc="left", fontsize=10.5, weight="bold")
    ax = axs[1]
    p50 = 0.5 ** lam * 100
    ax.plot(c, p50, color=EC["red"], lw=2)
    for k in (0.25, 0.5, 1.0, 1.5):
        v = 0.5 ** ((2 - k) / k) * 100
        ax.scatter(k, v, color=EC["red"], s=34, zorder=4)
        ax.annotate(f"{v:.1f}%" if v < 10 else f"{v:.0f}%", (k, v), xytext=(-4, 9), textcoords="offset points",
                    ha="center", fontsize=9, weight="bold", color=EC["red"])
    ax.set_xlim(0, 2); ax.set_ylim(0, 105); strip(ax)
    ax.set_xlabel("Bet size, as a multiple of the Kelly fraction", fontsize=9)
    ax.set_title("Chance the account ever falls to half its starting value", loc="left", fontsize=10.5, weight="bold")
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    frame(fig, "Cutting the bet in half costs a quarter of the growth and three quarters of the ruin risk",
          "Both curves from theory: growth ∝ c(2−c); ruin chance = 0.5^((2−c)/c), for a bet of c times the Kelly fraction",
          source="Source: author's calculations")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig6_growth_vs_risk.png")); plt.close(fig)


# ============================================================ FIG 7: drawdown vs horizon
def fig7():
    h = pd.read_csv(os.path.join(DATA, "horizon_mdd.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = {0.25: EC["cyan"], 0.5: EC["blue"], 1.0: EC["red"]}
    for cc, d in h.groupby("c"):
        ax.plot(d.years, d.median_mdd * 100, color=cols[cc], lw=2, marker="o", ms=4)
        ax.text(d.years.iloc[-1] * 1.05, d.median_mdd.iloc[-1] * 100,
                {0.25: "Quarter Kelly", 0.5: "Half Kelly", 1.0: "Full Kelly"}[cc], color=cols[cc],
                fontsize=9.5, weight="bold", va="center")
    ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20, 40]); ax.set_xticklabels(["1", "5", "10", "20", "40 years"])
    ax.set_xlim(0.8, 70); ax.set_ylim(0, 100); strip(ax)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.minorticks_off()
    frame(fig, "Drawdowns never stop deepening; they just slow down",
          "Median maximum drawdown from the running peak, by years played (log scale)",
          note="5,000 simulated traders per line, 53% win rate at even odds.", source="Source: author's simulations")
    fig.tight_layout(rect=(0, 0.07, 1, 0.85))
    fig.savefig(os.path.join(OUT, "fig7_drawdown_horizon.png")); plt.close(fig)


# ============================================================ FIG 8: one trader's underwater curve
def fig8():
    sp = pd.read_csv(os.path.join(DATA, "sample_paths.csv"))
    t = np.arange(len(sp)) / 252
    names = [("Quarter Kelly", EC["cyan"]), ("Half Kelly", EC["blue"]), ("Full Kelly", EC["red"])]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    for n_, col in names:
        w = sp[n_].values
        axs[0].plot(t, w, color=col, lw=1.4)
        axs[1].plot(t, -(1 - w / np.maximum.accumulate(w)) * 100, color=col, lw=1.2)
        axs[0].text(t[-1] + 0.05, w[-1], n_, color=col, fontsize=9, weight="bold", va="center")
    axs[0].set_yscale("log"); axs[0].set_yticks([0.5, 1, 2, 5, 10, 20, 50]); axs[0].set_yticklabels(["0.5×", "1×", "2×", "5×", "10×", "20×", "50×"])
    axs[0].minorticks_off(); axs[0].axhline(1, color=EC["ink"], lw=0.8)
    axs[0].set_title("Account value (log scale)", loc="left", fontsize=10, weight="bold")
    axs[1].set_title("Drawdown from the running peak", loc="left", fontsize=10, weight="bold")
    axs[1].set_yticks([0, -25, -50, -75]); axs[1].set_yticklabels(["0", "−25%", "−50%", "−75%"])
    axs[1].axhline(0, color=EC["ink"], lw=0.8); axs[1].set_ylim(-90, 3)
    axs[1].set_xlabel("Years", fontsize=9); axs[1].set_xlim(0, 5.9)
    for a in axs:
        strip(a)
    frame(fig, "The same trader, the same luck, three bet sizes",
          "One simulated trader's 5 years at a quarter, a half and the full Kelly fraction (53% win rate, even odds)",
          source="Source: author's simulations (one of 20,000 traders)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    fig.savefig(os.path.join(OUT, "fig8_one_trader.png")); plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
    f()
print("ok")
