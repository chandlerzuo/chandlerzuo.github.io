"""Simulation studies for sizing a positive-edge bet: Kelly and friends, and stopping rules.

The game: every period one bet, win +1 per unit staked w.p. p, lose 1 w.p. 1-p (b = 1).
Kelly fraction f* = 2p - 1. All studies use common random numbers across strategies.

  A. known edge      p = 0.53 for everyone, the trader knows it.
  B. uncertain edge  trader believes p = 0.53 but the truth is p ~ N(0.52, 0.02^2) per
                     trader (a backtest that is a bit optimistic, and some have no edge).
  C. theory checks   (i) P(ever fall to x of starting capital) vs the exponent formula,
                     (ii) two-barrier stopping: P(hit floor first) and E[stopping time].

Writes CSVs to ../data. Run make_charts.py afterwards."""
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)

P_BELIEF = 0.53            # win probability the trader believes in
KELLY = 2 * P_BELIEF - 1   # = 0.06
N = 1260                   # 5 years of one decision a day
YEARS = 5
NP = 20000                 # simulated traders
W0 = 1.0


# ---------------------------------------------------------------- strategies
def _dd(W, peak):
    return 1.0 - W / peak


class Strategy:
    """stake(W, peak, wins, nobs) -> amount to stake; quit(W, peak) -> mask of traders who
    stop trading for good (they keep their money, earning nothing)."""
    def __init__(self, name, stake, quit=None, group=""):
        self.name, self.stake, self.group = name, stake, group
        self.quit = quit or (lambda W, peak: np.zeros_like(W, dtype=bool))


def frac(c, f_star=KELLY):
    return lambda W, peak, wins, nobs: c * f_star * W


def bayes_kelly(c=0.5, a0=200.0):
    """Beta(a0, a0) prior on p (mean 0.5: no edge until the data say so); stake c times the
    Kelly fraction of the posterior mean, zero when the posterior mean is below 1/2."""
    def stake(W, peak, wins, nobs):
        ph = (a0 + wins) / (2 * a0 + nobs)
        return c * np.maximum(2 * ph - 1, 0.0) * W
    return stake


STRATS = [
    Strategy("Fixed stake, 5% of start", lambda W, peak, wins, nobs: np.full_like(W, 0.05 * W0), group="fixed"),
    Strategy("1% of equity", frac(1 / 6), group="frac"),
    Strategy("Quarter Kelly", frac(0.25), group="frac"),
    Strategy("Half Kelly", frac(0.5), group="frac"),
    Strategy("Full Kelly", frac(1.0), group="frac"),
    Strategy("Double Kelly", frac(2.0), group="frac"),
    Strategy("Half Kelly, drawdown throttle",
             lambda W, peak, wins, nobs: 0.5 * KELLY * W * np.clip(1 - _dd(W, peak) / 0.30, 0, 1), group="rule"),
    Strategy("Half Kelly, quit at 25% drawdown", frac(0.5), quit=lambda W, peak: _dd(W, peak) >= 0.25, group="stop"),
    Strategy("Half Kelly, quit at -30% or +100%", frac(0.5),
             quit=lambda W, peak: (W <= 0.7 * W0) | (W >= 2.0 * W0), group="stop"),
    Strategy("Bayesian half Kelly", bayes_kelly(0.5), group="rule"),
    Strategy("Bayesian half Kelly, quit at 25% drawdown", bayes_kelly(0.5),
             quit=lambda W, peak: _dd(W, peak) >= 0.25, group="stop"),
]


def simulate(strat, X, keep=0, bands=False):
    """X: (paths, steps) int8 of +1/-1. Returns per-path summary (+ first `keep` wealth paths)."""
    n_p, n = X.shape
    W = np.full(n_p, W0); peak = W.copy(); mdd = np.zeros(n_p); minW = W.copy()
    active = np.ones(n_p, bool); wins = np.zeros(n_p); nobs = np.zeros(n_p)
    quit_t = np.full(n_p, -1)
    traj = np.empty((n + 1, keep)) if keep else None
    band = [(0, np.full(5, W0))] if bands else None
    if keep:
        traj[0] = W[:keep]
    for t in range(n):
        s = np.where(active, np.minimum(strat.stake(W, peak, wins, nobs), W), 0.0)
        x = X[:, t]
        W = W + s * x
        wins += active & (x > 0); nobs += active
        np.maximum(peak, W, out=peak); np.minimum(minW, W, out=minW)
        np.maximum(mdd, _dd(W, peak), out=mdd)
        stop = active & (strat.quit(W, peak) | (W <= 1e-9))
        quit_t[stop] = t
        active &= ~stop
        if keep:
            traj[t + 1] = W[:keep]
        if bands and (t + 1) % 10 == 0:
            band.append((t + 1, np.percentile(W, [5, 25, 50, 75, 95])))
    return dict(W=W, mdd=mdd, minW=minW, quit_t=quit_t, traj=traj, band=band)


def summarize(name, group, r, p_true=None):
    W = r["W"]
    row = dict(strategy=name, group=group,
               median_multiple=np.median(W), median_cagr=np.median(W) ** (1 / YEARS) - 1,
               mean_multiple=W.mean(), p5_multiple=np.percentile(W, 5), p95_multiple=np.percentile(W, 95),
               p_lose_money=(W < W0).mean(), p_lose_half=(W < 0.5 * W0).mean(),
               p_mdd30=(r["mdd"] >= 0.30).mean(), p_mdd50=(r["mdd"] >= 0.50).mean(),
               p_wiped=(r["minW"] <= 0.01 * W0).mean(), p_quit=(r["quit_t"] >= 0).mean(),
               median_mdd=np.median(r["mdd"]))
    return row


def draw(rng, p):
    return np.where(rng.random((NP, N)) < np.asarray(p).reshape(-1, 1), 1, -1).astype(np.int8)


# ---------------------------------------------------------------- A and B
def scenarios():
    rng = np.random.default_rng(20261115)
    p_A = np.full(NP, P_BELIEF)
    p_B = np.clip(rng.normal(0.52, 0.02, NP), 0.40, 0.60)
    out = {}
    for tag, p in (("A", p_A), ("B", p_B)):
        X = draw(np.random.default_rng(7 if tag == "A" else 8), p)
        rows = []; paths = {}
        for s in STRATS:
            r = simulate(s, X, keep=1 if tag == "A" else 0)
            rows.append(summarize(s.name, s.group, r))
            if tag == "B":
                for lab, m in (("edge", p > 0.5), ("no_edge", p <= 0.5)):
                    rr = dict(W=r["W"][m], mdd=r["mdd"][m], minW=r["minW"][m], quit_t=r["quit_t"][m])
                    row = summarize(s.name, s.group, rr); row["segment"] = lab; row["share"] = m.mean()
                    rows.append(row)
            if tag == "A":
                paths[s.name] = r["traj"][:, 0]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(DATA, f"results_{tag}.csv"), index=False)
        out[tag] = (df, p, X)
        if tag == "A":
            pd.DataFrame(paths).to_csv(os.path.join(DATA, "sample_paths.csv"), index=False)
    return out


def fan_bands(out):
    """5/25/50/75/95th percentile of wealth through time, scenario A, for the fan charts."""
    _, _, X = out["A"]
    rows = []
    for s in STRATS:
        if s.name in ("Fixed stake, 5% of start", "Quarter Kelly", "Half Kelly", "Full Kelly", "Double Kelly"):
            for t, q in simulate(s, X, bands=True)["band"]:
                rows.append(dict(strategy=s.name, t=t, **dict(zip(("p5", "p25", "p50", "p75", "p95"), q))))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fan_bands.csv"), index=False)


# ---------------------------------------------------------------- frontier
def frontier(out):
    """Sweep the Kelly multiple c (and the quit-at-drawdown threshold) in both scenarios."""
    rows = []
    for tag in "AB":
        _, p, X = out[tag]
        for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0):
            r = simulate(Strategy("", frac(c)), X)
            rows.append(dict(scenario=tag, family="size", param=c, **summarize("", "", r)))
        for q in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            for c in (0.5, 1.0):
                r = simulate(Strategy("", frac(c), quit=lambda W, peak, q=q: _dd(W, peak) >= q), X)
                rows.append(dict(scenario=tag, family=f"quit_c{c}", param=q, **summarize("", "", r)))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "frontier.csv"), index=False)


# ---------------------------------------------------------------- theory
def log_moments(f, p=P_BELIEF):
    """Mean m and variance s2 of the log-wealth increment when staking fraction f."""
    up, dn = np.log1p(f), np.log1p(-f)
    m = p * up + (1 - p) * dn
    return m, p * (1 - p) * (up - dn) ** 2


def exact_exponent(f, p=P_BELIEF):
    """Cramer-Lundberg exponent: P(log-wealth ever falls b below its start) ~ exp(-lam b)."""
    g = lambda lam: p * (1 + f) ** (-lam) + (1 - p) * (1 - f) ** (-lam) - 1
    return brentq(g, 1e-6, 200.0)


def hit_check():
    rows = []
    n, n_p = 12000, 5000
    X = draw_small(np.random.default_rng(3), n_p, n)
    for c in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        f = c * KELLY
        logw = np.zeros(n_p); lo = np.zeros(n_p)
        step = {1: np.log1p(f), -1: np.log1p(-f)}
        up, dn = np.log1p(f), np.log1p(-f)
        for t in range(n):
            logw += np.where(X[:, t] > 0, up, dn)
            np.minimum(lo, logw, out=lo)
        m, s2 = log_moments(f)
        lam_diff = 2 * m / s2
        lam_exact = exact_exponent(f)
        for x in (0.5, 0.7):
            rows.append(dict(c=c, x=x, sim=(lo <= np.log(x)).mean(),
                             diffusion=x ** lam_diff, exact=x ** lam_exact,
                             kelly_formula=x ** ((2 - c) / c)))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "theory_hit.csv"), index=False)


def draw_small(rng, n_p, n):
    return np.where(rng.random((n_p, n)) < P_BELIEF, 1, -1).astype(np.int8)


def barrier_check():
    """Half Kelly, floor at 0.7 of start, target u of start (u = inf: never stop on gains)."""
    f = 0.5 * KELLY
    m, s2 = log_moments(f); th = 2 * m / s2
    b = np.log(1 / 0.7)
    targets = [1.1, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf]
    n, n_p = 40000, 20000
    rng = np.random.default_rng(11)
    up, dn = np.log1p(f), np.log1p(-f)
    logw = np.zeros(n_p); alive = np.ones(n_p, bool)
    t_low = np.full(n_p, np.inf); t_up = {u: np.full(n_p, np.inf) for u in targets}
    for t in range(n):
        x = rng.random(n_p) < P_BELIEF
        logw = np.where(alive, logw + np.where(x, up, dn), logw)
        hit = alive & (logw <= -b)
        t_low[hit] = t + 1; alive &= ~hit
        for u in targets[:-1]:
            h = alive & (logw >= np.log(u)) & np.isinf(t_up[u])
            t_up[u][h] = t + 1
        if not alive.any():
            break
    rows = []
    for u in targets:
        a = np.log(u) if np.isfinite(u) else np.inf
        tu = t_up[u] if np.isfinite(u) else np.full(n_p, np.inf)
        low_first = t_low < tu
        stop_t = np.minimum(t_low, tu)
        if np.isfinite(a):
            p_low = (1 - np.exp(-th * a)) / (np.exp(th * b) - np.exp(-th * a))
            et = ((1 - p_low) * a - p_low * b) / m
        else:
            p_low = np.exp(-th * b); et = np.inf
        rows.append(dict(target=u, sim_p_floor=low_first.mean(), theory_p_floor=p_low,
                         sim_et=np.nanmean(np.where(np.isfinite(stop_t), stop_t, n)) if np.isfinite(a) else np.nan,
                         theory_et=et, theta=th, m=m, s2=s2))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "theory_barrier.csv"), index=False)


def horizon_mdd():
    """Median max drawdown of half / full Kelly as the horizon grows (growth is log-like)."""
    rows = []
    X = draw_small(np.random.default_rng(5), 5000, 2520 * 4)
    for c in (0.25, 0.5, 1.0):
        f = c * KELLY; up, dn = np.log1p(f), np.log1p(-f)
        logw = np.zeros(5000); lpeak = np.zeros(5000); mdd = np.zeros(5000)
        for t in range(X.shape[1]):
            logw += np.where(X[:, t] > 0, up, dn)
            np.maximum(lpeak, logw, out=lpeak)
            np.maximum(mdd, 1 - np.exp(logw - lpeak), out=mdd)
            if (t + 1) in (252, 1260, 2520, 5040, 10080):
                rows.append(dict(c=c, years=(t + 1) / 252, median_mdd=np.median(mdd),
                                 p_mdd50=(mdd >= .5).mean()))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "horizon_mdd.csv"), index=False)


if __name__ == "__main__":
    out = scenarios()
    for tag in "AB":
        print(f"--- scenario {tag}")
        print(out[tag][0].query("segment != segment" if tag == "B" else "index == index").round(4).to_string())
    fan_bands(out)
    frontier(out)
    hit_check(); barrier_check(); horizon_mdd()
    for f in ("theory_hit", "theory_barrier", "horizon_mdd"):
        print(f); print(pd.read_csv(os.path.join(DATA, f + ".csv")).round(4).to_string())
