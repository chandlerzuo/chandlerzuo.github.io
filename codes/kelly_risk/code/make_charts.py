"""Figures for the Kelly / running-maximum / stopping-time post, in the same Economist style as the other
posts. Reads ../data/*.csv (run sim.py first). Outputs -> ../outputs."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim
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
SOURCE = "Source: author's simulations (20,000 paths, n = 1,260 steps each)"

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


SHORT = {"Fixed bet size, 0.05 W0": "Fixed bet size 0.05 $W_0$", "Fixed fraction 1%": "Fixed fraction 1% (⅙ Kelly)",
         "Half Kelly, depth throttle": "Half Kelly + depth throttle",
         "Half Kelly, stop at depth 25%": "Half Kelly, stop at depth 25%",
         "Half Kelly, exit band 0.7 W0 / 2 W0": "Half Kelly, exit at 0.7 $W_0$ or 2 $W_0$",
         "Bayesian half Kelly": "Bayesian half Kelly",
         "Bayesian half Kelly, stop at depth 25%": "Bayesian half Kelly, stop at depth 25%"}
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
        ax.set_xlabel("Steps n (in units of 252)", fontsize=9)
        end = A.set_index("strategy").loc[n]
        ax.text(0.04, 0.05, f"median {end.median_multiple:.1f}×\n5th pct {end.p5_multiple:.2f}×",
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axs[0].set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3])
    axs[0].set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1,000"])
    axs[0].set_ylabel("$W_n / W_0$ (log scale)", fontsize=9)
    frame(fig, "Each step up in bet fraction fattens the tails; past Kelly, even the median stops growing",
          "Quantiles of $W_n$ over n = 1,260 steps; P(X = +1) = 0.53, f* = 0.06; the same random steps in every panel")
    fig.tight_layout(rect=(0, 0.07, 1, 0.85))
    fig.savefig(os.path.join(OUT, "fig1_fans.png")); plt.close(fig)


# ============================================================ FIG 2: bars ====
def fig2():
    order = ["Fixed bet size, 0.05 W0", "Fixed fraction 1%", "Quarter Kelly", "Half Kelly", "Full Kelly",
             "Double Kelly", "Half Kelly, depth throttle", "Bayesian half Kelly"]
    d = A.set_index("strategy").loc[order][::-1]
    labels = [nm(s) for s in d.index]; y = np.arange(len(d))
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    specs = [("median_cagr", "Median growth per 252 steps, %", 100, EC["blue"]),
             ("p_mdd50", "P(depth below running max ≥ 50%), %", 100, EC["red"]),
             ("p_lose_money", "P($W_n < W_0$), %", 100, EC["grey"])]
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
    frame(fig, "Growth rises with the bet fraction; so does every risk measure",
          "n = 1,260 steps, P(X = +1) = 0.53 known exactly (scenario A)",
          note="Depth below the running maximum is 1 − $W_n$ / (max over k ≤ n of $W_k$) (the drawdown). The fixed-bet process is absorbed at 0 in 8.7% of paths.")
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
    ax.text(0.03, 80, "…falls to 0.7 $W_0$", color=EC["cyan"], fontsize=9, weight="bold")
    ax.text(0.03, 70, "…falls to 0.5 $W_0$", color=EC["blue"], fontsize=9, weight="bold")
    ax.axvline(1, color=EC["grey"], lw=1, ls=":")
    ax.text(1.02, 3, "Full\nKelly", color=EC["grey"], fontsize=8.5)
    ax.set_xlabel("Bet fraction, as a multiple of the Kelly fraction c", fontsize=9)
    ax.set_title("Probability the process ever…", loc="left", fontsize=10.5, weight="bold")
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
    ax.set_xlabel("Upper barrier, as a multiple of $W_0$", fontsize=9)
    ax.set_title("Half Kelly: P(hit 0.7 $W_0$ before the upper barrier)", loc="left", fontsize=10.5, weight="bold")
    ax.set_ylim(0, 40); ax.set_xlim(1, 6); strip(ax)
    ax.set_yticks([0, 10, 20, 30, 40]); ax.set_yticklabels(["0", "10", "20", "30", "40%"])
    frame(fig, "Two formulas pin down first-passage risk and what a stopping time does to it",
          "Lines: theory (exponent λ ≈ 2m/s² from the mean m and variance s² of the log-increment); dots: simulation",
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
                ax.annotate(f"stop at depth {r.param*100:.0f}%", (risk(r.p5_multiple), r.median_cagr * 100),
                            xytext=(0, -14), textcoords="offset points", fontsize=8.5, color=EC["red"], ha="center")
        b_ = big.set_index("strategy")
        offs = {"A": ((8, 8), (8, -14)), "B": ((-8, 12), (10, 6))}[tag]
        for (n_, col), off in zip((("Bayesian half Kelly", EC["green"]), ("Half Kelly, depth throttle", EC["purple"])), offs):
            r = b_.loc[n_]
            ax.scatter(risk(r.p5_multiple), r.median_cagr * 100, s=60, color=col, zorder=5, marker="D")
            ax.annotate(nm(n_), (risk(r.p5_multiple), r.median_cagr * 100), xytext=off, textcoords="offset points",
                        fontsize=8.5, color=col, weight="bold", ha="right" if off[0] < 0 else "left")
        ax.axhline(0, color=EC["ink"], lw=0.9)
        ax.set_title({"A": "A. The edge is known", "B": "B. The edge is uncertain (and overestimated)"}[tag],
                     loc="left", fontsize=10.5, weight="bold")
        ax.set_xlabel("Lower tail: 1 − (5th percentile of $W_n / W_0$), in %  →  riskier", fontsize=8.5)
        strip(ax, bottom=True)
    axs[0].set_xlim(-15, 105); axs[1].set_xlim(-30, 105)
    axs[0].set_ylabel("Median growth per 252 steps, %", fontsize=9)
    frame(fig, "Shrinking the bet fraction beats stopping: more growth for the same lower tail",
          "Growth–tail frontier. Blue: bet fraction as a multiple of Kelly. Red: half Kelly, stopped for good at the given depth below the running maximum",
          note="B: the policy assumes p = 0.53 but the true p varies across paths, p ~ N(0.52, 0.02²). Bayesian: Beta(200, 200) prior on p.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig4_frontier.png")); plt.close(fig)


# ============================================================ FIG 5: segments
def fig5():
    order = ["Half Kelly", "Fixed fraction 1%", "Half Kelly, stop at depth 25%", "Bayesian half Kelly",
             "Bayesian half Kelly, stop at depth 25%"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, seg, ttl, col in zip(axs, ("edge", "no_edge"), ("Paths with positive drift (p > ½): 85%",
                                                            "Paths with no drift (p ≤ ½): 15%"),
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
    frame(fig, "Stopping and learning rescue the zero-drift paths, at a price for the rest",
          "Scenario B, n = 1,260: median (bar) and 5th percentile (dot) of $W_n / W_0$",
          note="Black line: $W_0$. Stopping is permanent; a stopped path stays where it is.")
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
    ax.set_xlabel("Bet fraction, as a multiple of the Kelly fraction c", fontsize=9)
    ax.set_title("Growth rate m(c), as % of its maximum m*", loc="left", fontsize=10.5, weight="bold")
    ax = axs[1]
    p50 = 0.5 ** lam * 100
    ax.plot(c, p50, color=EC["red"], lw=2)
    for k in (0.25, 0.5, 1.0, 1.5):
        v = 0.5 ** ((2 - k) / k) * 100
        ax.scatter(k, v, color=EC["red"], s=34, zorder=4)
        ax.annotate(f"{v:.1f}%" if v < 10 else f"{v:.0f}%", (k, v), xytext=(-4, 9), textcoords="offset points",
                    ha="center", fontsize=9, weight="bold", color=EC["red"])
    ax.set_xlim(0, 2); ax.set_ylim(0, 105); strip(ax)
    ax.set_xlabel("Bet fraction, as a multiple of the Kelly fraction c", fontsize=9)
    ax.set_title("P(process ever falls to ½ $W_0$) = (½)^λ", loc="left", fontsize=10.5, weight="bold")
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    frame(fig, "Halving the bet fraction costs a quarter of the growth rate and three quarters of the hitting probability",
          "Both curves from theory: m(c) ∝ c(2−c); hitting probability = (½)^((2−c)/c), for the fraction c f*",
          source="Source: author's calculations")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig6_growth_vs_risk.png")); plt.close(fig)


# ============================================================ FIG 7: drawdown vs horizon
def fig7():
    h = pd.read_csv(os.path.join(DATA, "horizon_mdd.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = {0.25: EC["cyan"], 0.5: EC["blue"], 1.0: EC["red"]}
    for cc, d in h.groupby("c"):
        ax.plot(d.years, d.median_mdd * 100, color=cols[cc], lw=0, marker="o", ms=5)
        f_ = cc * sim.KELLY; m_ = sim.log_moments(f_)[0]; lam_ = sim.exact_exponent(f_)
        yy = np.logspace(np.log10(0.8), np.log10(40), 100)
        dd = np.log(1 + m_ * yy * 252 * lam_ / np.log(2)) / lam_
        ax.plot(yy, (1 - np.exp(-dd)) * 100, color=cols[cc], lw=1.8, ls="--")
        ax.text(d.years.iloc[-1] * 1.15, d.median_mdd.iloc[-1] * 100,
                {0.25: "Quarter Kelly", 0.5: "Half Kelly", 1.0: "Full Kelly"}[cc], color=cols[cc],
                fontsize=9.5, weight="bold", va="center")
    ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20, 40]); ax.set_xticklabels(["252", "1,260", "2,520", "5,040", "10,080 steps"])
    ax.set_xlim(0.8, 70); ax.set_ylim(0, 100); strip(ax)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.minorticks_off()
    frame(fig, "Depth below the running maximum never stops growing; it just slows down",
          "Median maximum depth below the running maximum. Dots: simulation; dashed: formula",
          note="5,000 simulated paths per line, P(X = +1) = 0.53.", source="Source: author's simulations")
    fig.tight_layout(rect=(0, 0.07, 1, 0.85))
    fig.savefig(os.path.join(OUT, "fig7_drawdown_horizon.png")); plt.close(fig)


# ============================================================ FIG 8: one path, underwater curve
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
    axs[0].set_title("$W_n / W_0$ (log scale)", loc="left", fontsize=10, weight="bold")
    axs[1].set_title("Depth below the running maximum", loc="left", fontsize=10, weight="bold")
    axs[1].set_yticks([0, -25, -50, -75]); axs[1].set_yticklabels(["0", "−25%", "−50%", "−75%"])
    axs[1].axhline(0, color=EC["ink"], lw=0.8); axs[1].set_ylim(-90, 3)
    axs[1].set_xlabel("Steps n (in units of 252)", fontsize=9); axs[1].set_xlim(0, 5.9)
    for a in axs:
        strip(a)
    frame(fig, "One sample path, three bet fractions",
          "The same 1,260 random steps driving the process at a quarter, a half and the full Kelly fraction",
          source="Source: author's simulations (one of 20,000 paths)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    fig.savefig(os.path.join(OUT, "fig8_one_path.png")); plt.close(fig)


# ============================================================ FIG 9: law of large numbers
def fig9():
    d = pd.read_csv(os.path.join(DATA, "slln.csv"))
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.9), sharey=True)
    for ax, (c, g) in zip(axs, d.groupby("c")):
        ax.fill_between(g.n, g.p5 * 1e3, g.p95 * 1e3, color=EC["shade"], lw=0)
        ax.plot(g.n, g.p50 * 1e3, color=EC["blue"], lw=1.8)
        m = g.m.iloc[0] * 1e3
        ax.axhline(m, color=EC["red"], lw=1.4, ls="--"); ax.axhline(0, color=EC["ink"], lw=0.8)
        ax.set_xscale("log"); ax.set_xlim(10, 20000); ax.set_xticks([10, 100, 1000, 10000])
        ax.set_xticklabels(["10", "100", "1k", "10k"]); ax.minorticks_off()
        ax.set_title(f"c = {c:g}" + ("  (Kelly)" if c == 1 else ""), loc="left", fontsize=10.5, weight="bold")
        ax.text(0.97, 0.06, f"m(c) = {m:.2f}×10⁻³", transform=ax.transAxes, ha="right", color=EC["red"], fontsize=8.5)
        ax.set_xlabel("n", fontsize=9); strip(ax)
    axs[0].set_ylabel("$S_n / n$  (×10⁻³)", fontsize=9)
    axs[0].set_ylim(-14, 14)
    frame(fig, "Why Kelly wins: $S_n/n \\to m(c)$ almost surely, and $m(c)$ peaks at $c = 1$",
          "5th–95th percentile band and median of $S_n / n$, with $S_n$ = ln($W_n / W_0$); dashed: the limit $m(c)$")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    fig.savefig(os.path.join(OUT, "fig9_slln.png")); plt.close(fig)


# ============================================================ FIG 10: competitive optimality
def fig10():
    d = pd.read_csv(os.path.join(DATA, "competitive.csv"))
    cols = {"Half Kelly": EC["cyan"], "1.5× Kelly": EC["blue"], "Double Kelly": EC["purple"],
            "Chase losses (adaptive)": EC["red"]}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    K = np.logspace(0, 3, 100)
    ax.plot(K, 1 / K, color=EC["ink"], lw=1.4, ls="--")
    ax.text(150, 0.014, "Bound 1/K", fontsize=9.5, rotation=-36, rotation_mode="anchor")
    for n_, g in d.groupby("rival"):
        g = g[g.prob > 0]
        ax.plot(g.K, g.prob, color=cols[n_], lw=2)
        ax.text(g.K.iloc[-1] * 1.08, g.prob.iloc[-1], n_, color=cols[n_], fontsize=9, weight="bold", va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 3000); ax.set_ylim(1e-5, 1.3)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.minorticks_off(); strip(ax); ax.set_xlabel("K", fontsize=9)
    frame(fig, "No policy beats Kelly by a factor K with probability above 1/K, and one nearly attains it",
          "P( sup over n ≤ 1,260 of  $W_n$(rival) / $W_n$(Kelly)  ≥ K ), 40,000 paths each",
          note="The adaptive policy doubles its fraction after each loss (capped at 0.5); the ratio is a martingale, so Ville's inequality bounds it.")
    fig.tight_layout(rect=(0, 0.08, 0.93, 0.85))
    fig.savefig(os.path.join(OUT, "fig10_competitive.png")); plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
    f()
print("ok")
