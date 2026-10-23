"""Figures for the realized return of buying a home to live in, over the full period, by
sub-window, with a mortgage, and as a cumulative return over time. Styled after The
Economist's charts: red rule and tag, left-aligned bold title and subtitle, light
gridlines in one direction only, value labels on the axis edge, short source line.
Reads ../data/history_*.csv and ../data/benchmarks.csv (run history_pipeline.py
first). Outputs -> ../outputs."""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))

# ---- Economist-style palette and frame ----
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
SOURCE = ("Sources: Zillow; BLS; Census Bureau; Freddie Mac; S&P Dow Jones Indices; "
          "Kenneth French; author's calculations")

d = pd.read_csv(os.path.join(DATA, "history_units.csv"))
w = pd.read_csv(os.path.join(DATA, "history_windows.csv"))
bench = pd.read_csv(os.path.join(DATA, "benchmarks.csv"))
b = bench[bench.window == "Overall"].set_index("benchmark").cagr
WINDOWS = list(dict.fromkeys(w.window))
SP, TB, CPI = b["S&P 500 (total return)"], b["1-month T-bills"], b["CPI inflation"]
pct = lambda s: np.asarray(s) * 100
START_P = pd.Period(d.start.iloc[0], "M"); END_P = pd.Period(d.end.iloc[0], "M")
SPAN = f"{START_P.strftime('%b %Y')}–{END_P.strftime('%b %Y')}"
MORT = d.mortgage_rate.iloc[0] * 100
RUN = os.path.join(DATA, "history_running.csv")
RUN_LATE = os.path.join(DATA, "history_running_2023.csv")
short = lambda m: m.rsplit(", ", 1)[0]


def frame(fig, title, subtitle, note="", top=0.97, legend=None, legend_cols=3):
    """Economist header (red rule + tag, title, subtitle, optional legend row) and
    footer (note + sources). Returns the figure-coordinate top of the plot area."""
    fig.add_artist(Line2D([0.0, 1.0], [top, top], color=EC["red"], lw=1.3, transform=fig.transFigure))
    fig.add_artist(Rectangle((0.0, top - 0.022), 0.045, 0.022, color=EC["red"], lw=0,
                             transform=fig.transFigure))
    fig.text(0.0, top - 0.045, title, fontsize=14, weight="bold", va="top", ha="left")
    fig.text(0.0, top - 0.092, subtitle, fontsize=10.5, va="top", ha="left")
    y = top - 0.14
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, y + 0.01), ncol=legend_cols,
                   fontsize=9, handlelength=1.0, handleheight=0.9, columnspacing=1.4, borderaxespad=0)
        y -= 0.045
    foot = (note + "\n" if note else "") + SOURCE
    fig.text(0.0, 0.0, foot, fontsize=7.8, color=EC["grey"], va="bottom", ha="left")
    return y


def strip(ax, keep_bottom=False):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)


def bar_axes(ax, labels):
    """Horizontal bar conventions: names left, % axis on top, vertical grid only."""
    y = np.arange(len(labels))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(-0.6, len(labels) + 0.5)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.grid(axis="y", visible=False)
    strip(ax)
    ax.axvline(0, color=EC["ink"], lw=0.9, zorder=3)
    return y


def benchmark_marks(ax, n, rates=None, which=("sp", "cpi", "tb"), fs=8.5):
    """Benchmark verticals, labeled in the headroom above the top bar."""
    sp, tb, cpi = rates or (SP, TB, CPI)
    spec = {"sp": (sp, "S&P 500", "-", EC["red"], 1.8), "cpi": (cpi, "Inflation", ":", EC["grey"], 1.2),
            "tb": (tb, "T-bills", "--", EC["grey"], 1.2)}
    for k in which:
        v, name, ls, col, lw = spec[k]
        ax.axvline(v * 100, color=col, lw=lw, ls=ls, zorder=4)
        ax.text(v * 100 + 0.15, n - 0.25, f"{name}\n{v*100:.1f}%", color=col,
                fontsize=fs, va="bottom", ha="left",
                weight="bold" if k == "sp" else "normal", zorder=6)


def stacked_bars(ax, s, total, parts, colors, names, show_parts=True):
    """Bars to `total`; the last segments are the attribution columns `parts`."""
    y = np.arange(len(s))
    tot = pct(s[total])
    prts = [pct(s[c].fillna(0).clip(lower=0)) for c in parts]
    left = tot - sum(prts)
    ax.barh(y, left, height=0.66, color=colors[0], lw=0, zorder=2)
    for prt, col in zip(prts, colors[1:]):
        ax.barh(y, prt, left=left, height=0.66, color=col, lw=0, zorder=2)
        left = left + prt
    for i, t in enumerate(tot):
        extra = ", ".join(f"{nm} +{p_[i]:.1f}" for nm, p_ in zip(names, prts) if p_[i] >= 0.05)
        txt = f"{t:.1f}" + (f"  ({extra})" if extra and show_parts else "")
        ax.text(t + 0.25, i, txt, va="center", fontsize=8.5, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))


# ============================================================ FIG 1 ==========
def fig1():
    """Each metro's all-cash IRR vs benchmarks, with the part due to an assessment cap."""
    s = d.sort_values("irr_cash").reset_index(drop=True)
    fig = plt.figure(figsize=(8.6, 7.0))
    leg = [Patch(color=EC["blue"], label="All-cash IRR"),
           Patch(color=EC["yellow"], label="Of which: property-tax assessment cap")]
    n_beat = int((s.irr_cash > SP).sum())
    top = frame(fig, f"Bought with cash, {n_beat} of {len(s)} homes beat the S&P 500",
                f"Typical home bought in {START_P.strftime('%B %Y')}, lived in, sold in "
                f"{END_P.strftime('%B %Y')}; IRR, % a year, after tax",
                note="Note: return = rent saved less property tax, upkeep and insurance; 3% buying and 6% "
                     "selling costs. Tax: 40% on income, 25% on long-term gains and dividends.", legend=leg, legend_cols=2)
    ax = fig.add_axes([0.16, 0.07, 0.8, top - 0.12])
    bar_axes(ax, s.metro.map(short))
    stacked_bars(ax, s, "irr_cash", ["cap_cash"], [EC["blue"], EC["yellow"]], ["cap"])
    benchmark_marks(ax, len(s))
    ax.set_xlim(0, max(SP, s.irr_cash.max()) * 100 + 3)
    fig.savefig(f"{OUT}/fig1_irr_vs_benchmarks.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ============================================================ FIG 2 ==========
def fig2():
    """Where the all-cash return came from: price growth, rent not paid, assessment cap."""
    s = d.sort_values("irr_cash").reset_index(drop=True)
    cap = s.cap_cash.fillna(0).clip(lower=0)
    s = s.assign(rent_part=s.irr_cash - s.irr_price_only - cap, cap_part=cap)
    n_rent = int((s.rent_part + s.cap_part > s.irr_price_only).sum())
    fig = plt.figure(figsize=(8.6, 7.0))
    leg = [Patch(color=EC["blue"], label="Price appreciation"),
           Patch(color=EC["cyan"], label="Rent not paid, net of costs"),
           Patch(color=EC["yellow"], label="Assessment cap")]
    top = frame(fig, f"Rent saved out-earned price gains in {n_rent} of {len(s)} cities",
                f"Sources of the all-cash IRR, {SPAN}, % a year, after tax",
                note="Note: price appreciation is after buying and selling costs and tax on any gain above "
                     "the $500k exclusion.", legend=leg)
    ax = fig.add_axes([0.16, 0.07, 0.8, top - 0.12])
    bar_axes(ax, s.metro.map(short))
    stacked_bars(ax, s, "irr_cash", ["rent_part", "cap_part"], [EC["blue"], EC["cyan"], EC["yellow"]],
                 ["rent", "cap"], show_parts=False)
    benchmark_marks(ax, len(s))
    ax.set_xlim(0, max(SP, s.irr_cash.max()) * 100 + 2)
    fig.savefig(f"{OUT}/fig2_return_sources.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ============================================================ FIG 3 ==========
def fig3():
    """The same purchase in four market regimes: all-cash IRR per metro vs benchmarks."""
    order = d.sort_values("irr_cash").metro.tolist()
    fig = plt.figure(figsize=(10, 9.0))
    leg = [Line2D([0], [0], marker="o", ls="", color=EC["blue"], markersize=7, label="All-cash IRR"),
           Line2D([0], [0], color=EC["red"], lw=1.8, label="S&P 500"),
           Line2D([0], [0], color=EC["grey"], lw=1.2, ls="--", label="T-bills")]
    top = frame(fig, "Same homes, four markets",
                "All-cash purchase at the start of each period, sold at its end; IRR, % a year, after tax",
                note="Note: no buying or selling costs. Cities in the same order in every panel "
                     f"(overall {START_P.year}–{END_P.year} rank).", legend=leg)
    lo = min(w.irr_cash.min(), bench.cagr.min()) * 100 - 2
    hi = max(w.irr_cash.max(), bench.cagr.max()) * 100 + 3
    gs = fig.add_gridspec(2, 2, left=0.11, right=0.98, bottom=0.07, top=top - 0.07, hspace=0.42, wspace=0.08)
    for k, name in enumerate(WINDOWS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        s = w[w.window == name].set_index("metro").loc[order].reset_index()
        bb = bench[bench.window == name].set_index("benchmark").cagr
        labels = s.metro.map(short) if k % 2 == 0 else [""] * len(s)
        y = bar_axes(ax, labels)
        ax.set_ylim(-0.6, len(s) - 0.4)
        ax.xaxis.tick_bottom()                       # panel titles sit on top
        for v, col, ls, lw in [(bb["S&P 500 (total return)"], EC["red"], "-", 1.8),
                               (bb["1-month T-bills"], EC["grey"], "--", 1.2)]:
            ax.axvline(v * 100, color=col, ls=ls, lw=lw, zorder=4)
        ax.scatter(pct(s.irr_cash), y, s=40, color=EC["blue"], zorder=5, lw=0)
        ax.set_xlim(lo, hi)
        n_beat = int((s.irr_cash > bb["S&P 500 (total return)"]).sum())
        st, en = pd.Period(s.start.iloc[0], "M"), pd.Period(s.end.iloc[0], "M")
        ax.text(0, 1.075, f"{name}, {st.strftime('%b %Y')}–{en.strftime('%b %Y')}", transform=ax.transAxes,
                fontsize=10.5, weight="bold", ha="left", va="bottom")
        ax.text(0, 1.015, f"S&P 500 {bb['S&P 500 (total return)']*100:.1f}%; {n_beat} of {len(s)} homes "
                "beat it", transform=ax.transAxes, fontsize=9, ha="left", va="bottom", color=EC["grey"])
    fig.savefig(f"{OUT}/fig3_windows.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ============================================================ FIG 4 ==========
def fig4():
    """Mortgage IRR split exactly into price rise, rent saved (net of the mortgage payment
    and carrying costs), the mortgage-interest deduction and the assessment cap. Positive
    parts stack right of zero, negative parts left of it; a tick marks the net IRR."""
    s = d.sort_values("irr_lev").reset_index(drop=True)
    parts = [("wf_price_lev", EC["blue"], "Price rise"),
             ("wf_rent_lev", EC["cyan"], "Rent saved, net of mortgage and costs"),
             ("wf_int_lev", EC["green"], "Mortgage-interest deduction"),
             ("wf_cap_lev", EC["yellow"], "Assessment cap")]
    fig = plt.figure(figsize=(8.8, 7.2))
    leg = [Patch(color=c, label=lab) for _, c, lab in parts]
    leg.append(Line2D([0], [0], marker="|", ls="", color=EC["ink"], markersize=11, markeredgewidth=2,
                      label="IRR"))
    n_beat = int((s.irr_lev > SP).sum())
    top = frame(fig, f"With a mortgage, {n_beat} of {len(s)} homes beat the S&P 500",
                f"Bought with 30% down at {MORT:.1f}% in {START_P.strftime('%B %Y')}, sold in "
                f"{END_P.strftime('%B %Y')}; IRR and its sources, % a year, after tax",
                note="Note: parts added in order (price rise; monthly rent saved less mortgage payment, tax, "
                     "insurance, upkeep; interest deduction; cap), so they sum to the IRR.\nInterest fully "
                     "deductible before 2018 (debt up to $1m); from 2018 only above the standard deduction, "
                     "SALT capped at $10k (debt up to $750k).", legend=leg, legend_cols=3)
    ax = fig.add_axes([0.16, 0.08, 0.8, top - 0.16])
    y = bar_axes(ax, s.metro.map(short))
    pos, neg = np.zeros(len(s)), np.zeros(len(s))
    for col, color, _ in parts:
        v = pct(s[col].fillna(0))
        ax.barh(y, np.where(v > 0, v, 0), left=pos, height=0.66, color=color, lw=0, zorder=2)
        ax.barh(y, np.where(v < 0, v, 0), left=neg, height=0.66, color=color, lw=0, zorder=2)
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    tot = pct(s.irr_lev)
    ax.scatter(tot, y, marker="|", s=260, color=EC["ink"], linewidths=2.2, zorder=6)
    for i, (t, p_end) in enumerate(zip(tot, pos)):
        ax.text(p_end + 0.3, i, f"{t:.1f}", va="center", fontsize=8.5, weight="bold", zorder=6)
    benchmark_marks(ax, len(s))
    ax.set_xlim(min(neg.min(), 0) - 0.8, max(SP * 100, pos.max()) + 2.5)
    fig.savefig(f"{OUT}/fig4_mortgage_irr.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ============================================================ FIG 5 ==========
def fig5(run_path=None, fname="fig5_running_irr.png"):
    """Cumulative return of one buyer who bought with a mortgage at the start and sold in
    any later month (buy/sell costs, loan payoff, gain tax), against the S&P 500 and
    T-bills bought the same month, with the market regimes shaded. Cumulative return =
    (1 + IRR) ** years - 1; log scale on the growth multiple."""
    r = pd.read_csv(run_path or RUN)
    metros = [c for c in r.columns if c not in ("month", "years", "mortgage_rate", "S&P 500", "T-bills")]
    p0 = pd.Period(r.month.iloc[0], "M") - int(round(r.years.iloc[0] * 12))     # purchase month
    t = pd.DatetimeIndex([p0.to_timestamp()]).append(pd.PeriodIndex(r.month, freq="M").to_timestamp())
    cum = {c: np.r_[1.0, ((1 + r[c]) ** r.years).values] for c in metros + ["S&P 500", "T-bills"]}
    med = np.nanmedian(np.vstack([cum[c] for c in metros]), axis=0)
    ahead = t[1:][(med[1:] > cum["S&P 500"][1:])]
    lo_v = min(0.45, np.nanmin([np.nanmin(v) for v in cum.values()]) * 0.95)
    hi_v = np.nanmax([np.nanmax(v) for v in cum.values()]) * 1.35

    fig = plt.figure(figsize=(9.4, 7.0))
    hi = {"Dallas, TX": EC["green"], "San Jose, CA": EC["purple"], "New York, NY": EC["cyan"]}
    leg = [Line2D([0], [0], color=EC["red"], lw=2.4, label="S&P 500"),
           Line2D([0], [0], color=EC["blue"], lw=2.4, label="Median of 12 cities"),
           Line2D([0], [0], color=EC["line"], lw=1.2, label="Other cities"),
           Line2D([0], [0], color=EC["grey"], lw=1.2, ls="--", label="T-bills")]
    rate = r.mortgage_rate.iloc[0] * 100
    if len(ahead):
        title = f"A home bought in {p0.year} overtook the index by {ahead[0].year}"
    else:
        title = (f"Bought in {p0.year}, the median home is {(med[-1] - 1) * 100:+.0f}%; "
                 f"the S&P 500 is {(cum['S&P 500'][-1] - 1) * 100:+.0f}%").replace("-", "\u2212")
    top = frame(fig, title,
                f"Cumulative return if bought with 30% down at {rate:.1f}% in {p0.strftime('%B %Y')} and sold "
                "in a given month, after tax, %", legend=leg, legend_cols=4,
                note="Note: 3% buying and 6% selling costs, loan payoff and gain tax at sale. S&P 500 and T-bills "
                     "bought the same month, no trading costs. Log scale.")
    ax = fig.add_axes([0.0, 0.08, 0.84, top - 0.11])
    for i, (name, a, b_) in enumerate(w[["window", "start", "end"]].drop_duplicates().itertuples(index=False)):
        x0, x1 = max(pd.Period(a, "M").to_timestamp(), t[0]), pd.Period(b_, "M").to_timestamp()
        if x1 <= t[0]:
            continue
        if i % 2 == 0:
            ax.axvspan(x0, x1, color=EC["shade"], zorder=0, lw=0)
        ax.text(x0 + pd.Timedelta(days=40), hi_v * (0.85 if i % 2 == 0 else 0.62), name, ha="left", va="top",
                fontsize=8.5, color=EC["grey"], weight="bold")
    for c in metros:
        if c not in hi:
            ax.plot(t, cum[c], color=EC["line"], lw=1.0, zorder=2)
    for c, col in hi.items():
        ax.plot(t, cum[c], color=col, lw=1.6, zorder=3)
    ax.plot(t, med, color=EC["blue"], lw=2.6, zorder=4)
    ax.plot(t, cum["S&P 500"], color=EC["red"], lw=2.4, zorder=5)
    ax.plot(t, cum["T-bills"], color=EC["grey"], lw=1.2, ls="--", zorder=5)
    ax.axhline(1.0, color=EC["ink"], lw=0.9, zorder=3)
    ax.set_yscale("log")
    grid = ([0.5, 1, 1.5, 2, 3, 5, 10, 20, 30] if hi_v / lo_v > 6 else
            [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2])
    ticks = [v for v in grid if lo_v <= v <= hi_v]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{(v - 1) * 100:+,.0f}" if v != 1 else "0" for v in ticks])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.tick_right()
    ax.set_ylim(lo_v, hi_v); ax.set_xlim(t[0], t[-1])
    span_years = (t[-1] - t[0]).days / 365.25
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2 if span_years > 8 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", visible=False)
    strip(ax, keep_bottom=True)
    # direct labels at the right end, nudged apart in log space
    ends = [(short(c), cum[c][-1], col) for c, col in hi.items()]
    ends += [("Median", med[-1], EC["blue"]), ("S&P 500", cum["S&P 500"][-1], EC["red"])]
    ends.sort(key=lambda e: e[1])
    last = -9
    for name, v, col in ends:
        yv = max(np.log10(v), last + 0.07); last = yv
        ax.text(t[-1] + (t[-1] - t[0]) * 0.065, 10 ** yv, f"{name} {(v - 1) * 100:+,.0f}%", fontsize=8.5,
                color=col, weight="bold", va="center", ha="left", clip_on=False)
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    fig5(RUN_LATE, "fig6_running_irr_2023.png")
    cols = ["size_rank", "metro", "price_start", "price_end", "rent_start", "rent_end",
            "gross_yield_start", "price_cagr", "rent_cagr", "net_rent_saved_yield_avg",
            "irr_price_only", "irr_cash_pretax", "irr_cash", "irr_lev_pretax", "irr_lev", "irr_lev_refi",
            "irr_cash_conserv", "irr_lev_conserv", "assess_cap", "cap_cash", "cap_lev", "irr_lev_noint",
            "int_saving_lev", "wf_price_lev", "wf_rent_lev", "wf_int_lev", "wf_cap_lev"]
    d[cols].sort_values("irr_cash", ascending=False).to_csv(os.path.join(OUT, "history_summary.csv"), index=False)
    print("history charts + history_summary.csv written to", OUT)
