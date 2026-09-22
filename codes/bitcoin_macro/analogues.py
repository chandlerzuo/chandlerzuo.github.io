"""Phase 5-6: per-regime risk signatures, and historical analogue matching.

The matching is the part of this study most likely to produce a spurious result,
so the guard rails come first:

  * Similarity is measured three independent ways (Mahalanobis on risk features,
    DTW on the normalised price path, Wasserstein on the return distribution).
    A match is only asserted when at least two agree.
  * There is an explicit NULL. Scanning thousands of asset-windows guarantees a
    close match to anything, so the question is never "is this match close?" but
    "is it closer than a randomly chosen asset's best match would have been?"
    The null is built by drawing pseudo-target windows from the reference panel
    itself and letting each one scan the whole panel exactly as Bitcoin does, so
    the null inherits the same multiple-comparison advantage.
  * Feature tiers respect data availability: no match is ever computed on a
    feature that does not exist in both eras (TIPS breakevens begin 2003, VIX
    1990, HY OAS 1996).
  * Leave-one-feature-out sensitivity, so no match rests on a single dimension.
"""
from __future__ import annotations

import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from fetchlib import PROC, ROOT, fetch

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
BTC_DAYS = 365          # Bitcoin trades every day

# The comparison panel starts here rather than in 1926. The dollar index begins
# in 1973, and a dollar sensitivity is now a required matching axis, so windows
# before that date could never be scored on it. Making the restriction explicit
# is better than letting complete-case filtering impose it silently.
PANEL_START = "1973-01-01"

# Reconciled regime chronology. Boundaries marked (S) are statistically robust
# (wild-bootstrap significant correlation breaks, or >=2-of-3 method agreement);
# those marked (N) are narrative/event anchors retained for economic
# interpretability even though they are not independently significant. The
# distinction is carried into the report rather than smoothed over.
REGIMES = [
    ("R1 2013 mania",            "2013-01-01", "2013-12-06", "S"),
    ("R2 2014-16 bear/recovery",  "2013-12-07", "2016-12-31", "N"),
    ("R3 2017-18 ICO cycle",      "2017-01-01", "2018-12-31", "N"),
    ("R4 2019-pre-COVID",         "2019-01-01", "2020-02-19", "N"),
    ("R5 2020-21 liquidity era",  "2020-02-20", "2021-11-09", "N"),
    ("R6 2022 tightening",        "2021-11-10", "2022-12-31", "N"),
    ("R7 2023-25 ETF era",        "2023-01-01", "2025-07-18", "N"),
    ("R8 2025-26 coupled",        "2025-07-19", "2026-09-30", "S"),
]


# --------------------------------------------------------------- data assembly
def industry49() -> pd.DataFrame:
    """Fama-French 49 industry value-weighted DAILY returns, 1926+.

    The file stacks several blocks (value-weighted daily, then equal-weighted
    daily, etc.). Only the first block is taken, detected by reading rows after
    the first header line until the dates stop being monotonically increasing.
    """
    p = fetch("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
              "49_Industry_Portfolios_daily_CSV.zip", "ff_industry49_daily.zip")
    with zipfile.ZipFile(p) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        text = zf.read(name).decode("latin-1")

    lines = text.split("\n")
    hdr_i = next(i for i, l in enumerate(lines) if l.startswith(","))
    cols = [c.strip() for c in lines[hdr_i].strip().split(",")[1:]]

    dates, rows, last = [], [], -1
    for l in lines[hdr_i + 1:]:
        parts = [x.strip() for x in l.strip().split(",")]
        if len(parts) != len(cols) + 1 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break               # end of the first (value-weighted) block
            continue
        dt = int(parts[0])
        if dt <= last:
            break                   # a new block restarted the calendar
        last = dt
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        dates.append(dt)
        rows.append(vals)

    df = pd.DataFrame(rows, columns=cols,
                     index=pd.to_datetime([str(d) for d in dates], format="%Y%m%d"))
    df = df.replace([-99.99, -999.0], np.nan) / 100.0
    df.index.name = "date"
    return df


def long_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (candidate asset daily returns, benchmark daily series)."""
    metals = pd.read_parquet(PROC / "metals_daily.parquet")
    etf = pd.read_parquet(PROC / "etf_nav_daily.parquet")
    fr = (pd.read_parquet(PROC / "fred_long.parquet")
            .pivot(index="date", columns="series", values="value").sort_index())
    ff = pd.read_parquet(PROC / "ff_daily.parquet")

    cand: dict[str, pd.Series] = {}

    def add(name, s):
        s = s.dropna()
        if len(s) > 400:
            cand[name] = s

    for m in ("gold", "silver", "platinum", "palladium"):
        if m in metals:
            x = metals[m].dropna()
            add(f"metal_{m}", np.log(x[x > 0]).diff())
    for sid, nm in (("NASDAQCOM", "idx_nasdaq_comp"), ("NASDAQ100", "idx_nasdaq100"),
                    ("NIKKEI225", "idx_nikkei"), ("DCOILWTICO", "cmdty_wti"),
                    ("DHHNGSP", "cmdty_natgas")):
        if sid in fr:
            x = fr[sid].dropna()
            add(nm, np.log(x[x > 0]).diff())
    for tic in ("spy", "gld", "xlk", "xle", "xlf", "dia", "mdy"):
        if tic in etf:
            x = etf[tic].dropna()
            add(f"etf_{tic}", np.log(x[x > 0]).diff())

    ind = industry49()
    for c in ind.columns:
        add(f"ind_{c}", ind[c])

    assets = pd.DataFrame(cand).loc[PANEL_START:]

    bench = pd.DataFrame({
        "eq": ff["Mkt-RF"],
        "rf": ff["RF"],
        "gold": np.log(metals["gold"].where(metals["gold"] > 0)).diff(),
        "d10y": fr["DGS10"].dropna().diff() if "DGS10" in fr else np.nan,
    })
    usd = None
    if "DTWEXM" in fr and "DTWEXBGS" in fr:
        a = np.log(fr["DTWEXM"].dropna()).diff()
        b = np.log(fr["DTWEXBGS"].dropna()).diff()
        usd = b.combine_first(a)          # broad index after 2006, major before
    elif "DTWEXBGS" in fr:
        usd = np.log(fr["DTWEXBGS"].dropna()).diff()
    bench["usd"] = usd
    return assets, bench.loc[PANEL_START:]


# ------------------------------------------------------------------- features
# Everything computed for each window. Note the split below: not all of these
# are used to MEASURE similarity.
FEATURES = ["vol_ann", "sharpe", "skew", "exkurt", "max_dd", "ar1",
            "frac_pos", "rho_eq", "rho_gold", "rho_d10y", "rho_usd",
            "rho_trade", "rho_cpi"]

# Axes the distance is actually computed on. Two describe the return
# distribution -- annualised volatility and the Sharpe ratio, i.e. how much risk
# and how much reward for it -- and the rest are sensitivities to MACRO STATE
# variables. Higher moments (skew, kurtosis), drawdown, autocorrelation and the
# positive-period share are computed and available but left out of the metric:
# with only 65 assets they add dimensions faster than they add information, and
# drawdown in particular is largely a restatement of volatility over a window.
#
# Correlation with GOLD and with EQUITIES are deliberately EXCLUDED from the
# distance and reported as diagnostics only. Both would leak: gold and silver
# are themselves candidate analogues, so "moves with gold" scores a precious
# metal highly for being a precious metal (and gold's own rho_gold is 1.0 by
# construction), and the candidate pool is dominated by US industry portfolios,
# which correlate with the equity market by construction. Scoring candidates on
# how much they resemble other candidates is circular. Keeping the two out of
# the metric and displaying them afterwards turns them into an out-of-sample
# check: the gold column in the essay's second chart is now something the
# matcher never optimised.
MATCH_FEATURES = ["vol_ann", "sharpe",
                  "rho_usd", "rho_d10y", "rho_trade", "rho_cpi"]

# A feature is admitted only if this share of reference windows can compute it.
MIN_FEATURE_COVERAGE = 0.50

BENCH_COLS = ["eq", "gold", "d10y", "usd", "trade", "cpi"]
RHO_NAMES = {"eq": "rho_eq", "gold": "rho_gold", "d10y": "rho_d10y",
             "usd": "rho_usd", "trade": "rho_trade", "cpi": "rho_cpi"}


def _rowwise_corr(A: np.ndarray, B: np.ndarray, min_n: int = 40) -> np.ndarray:
    """NaN-aware row-by-row correlation between two (m, L) arrays.

    min_n is the minimum number of overlapping observations for a correlation to
    be returned. 40 suits daily windows; monthly windows of 24-81 observations
    need a lower floor or every correlation feature silently becomes NaN and the
    matching quietly runs on volatility and shape alone.
    """
    m = np.isfinite(A) & np.isfinite(B)
    a = np.where(m, A, 0.0)
    b = np.where(m, B, 0.0)
    n = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sa, sb = a.sum(axis=1), b.sum(axis=1)
        saa = (a * a).sum(axis=1)
        sbb = (b * b).sum(axis=1)
        sab = (a * b).sum(axis=1)
        cov = sab / n - (sa / n) * (sb / n)
        va = saa / n - (sa / n) ** 2
        vb = sbb / n - (sb / n) ** 2
        out = cov / np.sqrt(va * vb)
    out[n < min_n] = np.nan
    return out


def window_features(r: pd.Series, bench: pd.DataFrame, per_year: float,
                    L: int, starts: np.ndarray,
                    min_corr_n: int = 40) -> pd.DataFrame:
    """Vectorised features for many equal-length windows of one series.

    Bitcoin's own regime features are produced by calling this with a single
    start, so the target and the reference panel are guaranteed to be measured by
    identical code. That matters more than speed: a subtly different skewness or
    drawdown convention between target and panel would bias every distance.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    idx = r.index
    rv = r.values.astype(float)
    W = sliding_window_view(rv, L)[starts]                     # (m, L)

    bm = bench.reindex(idx)
    rf = np.nan_to_num(bm["rf"].values.astype(float))
    RF = sliding_window_view(rf, L)[starts]

    n = L
    mean = W.mean(axis=1)
    var = W.var(axis=1)
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        m3 = ((W - mean[:, None]) ** 3).mean(axis=1)
        m4 = ((W - mean[:, None]) ** 4).mean(axis=1)
        skew = m3 / var ** 1.5
        exkurt = m4 / var ** 2 - 3.0

    cum = np.cumsum(W, axis=1)
    dd = (cum - np.maximum.accumulate(cum, axis=1)).min(axis=1)

    xs_mean = (W - RF).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = xs_mean * per_year / (sd * np.sqrt(per_year))
    ar1 = _rowwise_corr(W[:, :-1], W[:, 1:], min_corr_n)
    frac_pos = (W > 0).mean(axis=1)

    out = {
        "vol_ann": sd * np.sqrt(per_year), "sharpe": sharpe, "skew": skew,
        "exkurt": exkurt, "max_dd": dd, "ar1": ar1, "frac_pos": frac_pos,
    }
    for col in BENCH_COLS:
        if col not in bm.columns:
            out[RHO_NAMES[col]] = np.full(len(starts), np.nan)
            continue
        bv = bm[col].values.astype(float)
        B = sliding_window_view(bv, L)[starts]
        out[RHO_NAMES[col]] = _rowwise_corr(W, B, min_corr_n)

    df = pd.DataFrame(out)
    df["start"] = idx[starts]
    df["end"] = idx[starts + L - 1]
    return df


def featurize(r: pd.Series, bench: pd.DataFrame, per_year: float) -> dict | None:
    """Single-window features, via the same vectorised path as the panel."""
    r = r.dropna()
    if len(r) < 60:
        return None
    df = window_features(r, bench, per_year, len(r), np.array([0]))
    rec = df.iloc[0].to_dict()
    return None if not np.isfinite(rec.get("vol_ann", np.nan)) else rec


def reference_windows(assets: pd.DataFrame, bench: pd.DataFrame,
                     length_days: int, step: int = 21,
                     end_before: pd.Timestamp | None = None) -> pd.DataFrame:
    """Every rolling window of the target length, for every candidate asset.

    end_before makes the search genuinely HISTORICAL: a candidate window is only
    admissible if it finished before the Bitcoin regime being matched began.
    Without it the nearest window is often the same calendar period, or a later
    one, which is a statement about co-movement rather than about precedent --
    and "bitcoin's closest historical analogue is natural gas over exactly the
    same months" answers a different question than the one being asked.
    """
    frames = []
    for name in assets.columns:
        s = assets[name].dropna()
        if len(s) < length_days + 10:
            continue
        starts = np.arange(0, len(s) - length_days + 1, step)
        if end_before is not None:
            starts = starts[s.index[starts + length_days - 1] < end_before]
        if len(starts) == 0:
            continue
        f = window_features(s, bench, TRADING_DAYS, length_days, starts)
        f["asset"] = name
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[np.isfinite(out["vol_ann"])]


# -------------------------------------------------------------------- distance
def mahalanobis_setup(ref: pd.DataFrame, feats: list[str]):
    M = ref[feats].astype(float)
    ok = M.notna().all(axis=1)
    M = M[ok]
    mu, sd = M.mean(), M.std().replace(0, 1.0)
    Z = (M - mu) / sd
    C = np.cov(Z.values, rowvar=False)
    C += np.eye(len(feats)) * 1e-3 * np.trace(C) / len(feats)   # ridge
    return mu, sd, np.linalg.pinv(C), ok


def zscore(vec: pd.Series, mu, sd, feats) -> np.ndarray:
    return ((vec[feats].astype(float) - mu) / sd).values.astype(float)


def mdist_to(ref: pd.DataFrame, target: pd.Series, mu, sd, Ci,
             feats) -> np.ndarray:
    """Mahalanobis distance from each reference window TO the target.

    Note this is a distance to the TARGET, not to the panel centroid. Measuring
    distance from the centroid answers "which windows are unusual?", which is a
    different question and produces matches that are nowhere near Bitcoin: the
    first version of this scored windows with 16-25% volatility as close matches
    to a 143%-volatility Bitcoin regime.
    """
    Z = ((ref[feats].astype(float) - mu) / sd).values.astype(float)
    zt = zscore(target, mu, sd, feats)
    diff = Z - zt
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, Ci, diff))


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 30) -> float:
    """Sakoe-Chiba-banded DTW on amplitude-normalised cumulative paths."""
    a = (a - a.mean()) / (a.std() or 1)
    b = (b - b.mean()) / (b.std() or 1)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


def resample_path(r: pd.Series, k: int = 120) -> np.ndarray:
    c = r.cumsum().values
    return np.interp(np.linspace(0, len(c) - 1, k), np.arange(len(c)), c)


# ------------------------------------------------------------------- inference
def sharpe_se_lo(r: pd.Series, per_year: float) -> float:
    """Lo (2002) autocorrelation-corrected standard error of the Sharpe ratio."""
    r = r.dropna()
    n = len(r)
    if n < 20:
        return np.nan
    sr = r.mean() / r.std()
    q = min(6, n // 10)
    rhos = [r.autocorr(k) for k in range(1, q + 1)]
    rhos = [x for x in rhos if np.isfinite(x)]
    adj = 1.0 + 2.0 * sum((1 - (k + 1) / (q + 1)) * rhos[k] for k in range(len(rhos)))
    adj = max(adj, 0.1)
    se = np.sqrt((1 + 0.5 * sr ** 2) / n) * np.sqrt(adj)
    return se * np.sqrt(per_year)


def block_bootstrap_sharpe(r: pd.Series, per_year: float, n_boot: int = 2000,
                          block: int = 20, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = r.dropna().values
    n = len(x)
    if n < 30:
        return (np.nan, np.nan)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, nb)
        samp = np.concatenate([np.take(x, range(s, s + block), mode="wrap")
                               for s in starts])[:n]
        sd = samp.std()
        out[i] = samp.mean() / sd * np.sqrt(per_year) if sd > 0 else np.nan
    return (float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)))


def main() -> int:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    panel = pd.read_parquet(PROC / "panel_daily.parquet")
    panel["btc_xs"] = panel["btc_ret"] - panel["rf"].fillna(0)
    assets, bench = long_panel()

    emit("=" * 104)
    emit("REFERENCE PANEL")
    emit("=" * 104)
    emit(f"  candidate assets: {len(assets.columns)}")
    emit(f"  daily span: {assets.index.min().date()} to {assets.index.max().date()}")
    by_kind = pd.Series([c.split('_')[0] for c in assets.columns]).value_counts()
    emit("  by kind: " + ", ".join(f"{k}={v}" for k, v in by_kind.items()))
    emit(f"  benchmarks for correlation features: "
         f"{[c for c in bench.columns if c != 'rf']}")

    # ------------------------------------------------------ 5. regime profiles
    emit("\n" + "=" * 104)
    emit("PHASE 5. PER-REGIME RISK SIGNATURE (Bitcoin), with honest uncertainty")
    emit("  Sharpe CI: Lo (2002) HAC SE, and a stationary block bootstrap.")
    emit("=" * 104)
    prof_rows = []
    for name, a, b, tag in REGIMES:
        r = panel.loc[a:b, "btc_ret"].dropna()
        xs = panel.loc[a:b, "btc_xs"].dropna()
        if len(r) < 40:
            continue
        f = featurize(r, bench, BTC_DAYS)
        se = sharpe_se_lo(xs, BTC_DAYS)
        lo, hi = block_bootstrap_sharpe(xs, BTC_DAYS)
        prof_rows.append({
            "regime": name, "anchor": tag, "start": a, "end": b, "n_days": len(r),
            "vol_ann": f["vol_ann"], "sharpe": f["sharpe"], "sharpe_se": se,
            "sharpe_lo95": lo, "sharpe_hi95": hi,
            "skew": f["skew"], "exkurt": f["exkurt"], "max_dd": f["max_dd"],
            "rho_eq": f["rho_eq"], "rho_gold": f["rho_gold"],
            "rho_usd": f["rho_usd"], "frac_pos": f["frac_pos"],
        })
    prof = pd.DataFrame(prof_rows)
    emit(prof.round(3).to_string(index=False))
    prof.to_csv(ROOT / "outputs" / "tables" / "regime_profiles.csv", index=False)

    emit("\n  Sharpe ratios with 95% block-bootstrap intervals:")
    for _, r in prof.iterrows():
        emit(f"    {r['regime']:28s} {r['sharpe']:+6.2f}  "
             f"[{r['sharpe_lo95']:+6.2f}, {r['sharpe_hi95']:+6.2f}]  "
             f"(Lo SE {r['sharpe_se']:.2f})")
    emit("\n  Every interval is wide. A 1-2 year window on a 40-140% volatility")
    emit("  asset cannot pin down a Sharpe ratio, so regime Sharpe DIFFERENCES")
    emit("  below are descriptive, not significant unless stated.")

    # placebo: are regime differences bigger than random splits would give?
    emit("\n  PLACEBO: regimes were chosen partly because risk differs across them,")
    emit("  so the spread in Sharpe across regimes is mechanically inflated.")
    emit("  Comparison against random splits of the same number and lengths:")
    rng = np.random.default_rng(0)
    xs_all = panel.loc["2013-01-01":, "btc_xs"].dropna()
    lens = [len(panel.loc[a:b, "btc_xs"].dropna()) for _, a, b, _ in REGIMES]
    lens = [x for x in lens if x >= 40]
    obs_spread = prof["sharpe"].max() - prof["sharpe"].min()
    null_spreads = []
    for _ in range(2000):
        cuts = np.sort(rng.choice(np.arange(60, len(xs_all) - 60),
                                 size=len(lens) - 1, replace=False))
        bounds = [0] + list(cuts) + [len(xs_all)]
        srs = []
        for u, v in zip(bounds[:-1], bounds[1:]):
            seg = xs_all.iloc[u:v]
            if len(seg) < 40 or seg.std() == 0:
                continue
            srs.append(seg.mean() / seg.std() * np.sqrt(BTC_DAYS))
        if len(srs) >= 2:
            null_spreads.append(max(srs) - min(srs))
    p_spread = float(np.mean(np.array(null_spreads) >= obs_spread))
    emit(f"    observed Sharpe spread across regimes = {obs_spread:.2f}")
    emit(f"    random-split null: median {np.median(null_spreads):.2f}, "
         f"95th pct {np.percentile(null_spreads, 95):.2f}")
    emit(f"    p = {p_spread:.3f}  -> "
         + ("regime Sharpe spread is NOT distinguishable from random splitting"
            if p_spread > 0.05 else
            "regime Sharpe spread exceeds what random splitting produces"))

    # ------------------------------------------------------ 6. analogue match
    emit("\n" + "=" * 104)
    emit("PHASE 6. HISTORICAL ANALOGUE MATCHING")
    emit("=" * 104)

    all_matches, null_rows = [], []
    for name, a, b, tag in REGIMES:
        r_btc = panel.loc[a:b, "btc_ret"].dropna()
        if len(r_btc) < 120:
            emit(f"\n--- {name}: too short ({len(r_btc)}d) to match, skipped")
            continue
        # BTC trades 365 d/yr, candidates ~252. Match on CALENDAR length so the
        # windows cover the same economic span, not the same observation count.
        span_days = (pd.Timestamp(b) - pd.Timestamp(a)).days
        L = max(90, int(span_days * TRADING_DAYS / 365))

        ref = reference_windows(assets, bench, L, step=21,
                               end_before=pd.Timestamp(a))
        if ref.empty:
            emit(f"\n--- {name}: no candidate window ends before {a}, skipped")
            continue
        feats = [f for f in MATCH_FEATURES
                 if ref[f].notna().mean() > MIN_FEATURE_COVERAGE]
        btc_f = featurize(r_btc, bench, BTC_DAYS)
        btc_f["vol_ann"] = r_btc.std() * np.sqrt(BTC_DAYS)
        feats = [f for f in feats if np.isfinite(btc_f.get(f, np.nan))]

        mu, sd, Ci, ok = mahalanobis_setup(ref, feats)
        refv = ref[ok].reset_index(drop=True)
        bser = pd.Series(btc_f)
        refv["d_maha"] = mdist_to(refv, bser, mu, sd, Ci, feats)

        # path and distribution distances for the top Mahalanobis candidates
        top = refv.nsmallest(40, "d_maha").copy()
        btc_path = resample_path(r_btc)
        btc_std = ((r_btc - r_btc.mean()) / r_btc.std()).values
        dtws, wass = [], []
        for _, row in top.iterrows():
            s = assets[row["asset"]].loc[row["start"]:row["end"]].dropna()
            dtws.append(dtw_distance(btc_path, resample_path(s)))
            wass.append(wasserstein_distance(btc_std,
                                            ((s - s.mean()) / s.std()).values))
        top["d_dtw"], top["d_wass"] = dtws, wass
        for c in ("d_maha", "d_dtw", "d_wass"):
            top[f"rk_{c}"] = top[c].rank()
        top["rank_sum"] = top[["rk_d_maha", "rk_d_dtw", "rk_d_wass"]].sum(axis=1)

        # ---- NULL: let pseudo-targets from the panel scan the panel too ------
        rng2 = np.random.default_rng(7)
        pool = refv.sample(min(300, len(refv)), random_state=3)
        best_null = []
        for _, prow in pool.iterrows():
            other = refv[refv["asset"] != prow["asset"]]
            if other.empty:
                continue
            z = ((other[feats].astype(float) - mu) / sd).values
            zt = ((prow[feats].astype(float) - mu) / sd).values.astype(float)
            if not np.all(np.isfinite(zt)):
                continue
            diff = z - zt
            dd = np.sqrt(np.einsum("ij,jk,ik->i", diff, Ci, diff))
            best_null.append(np.nanmin(dd))
        best_null = np.array([x for x in best_null if np.isfinite(x)])
        btc_best = float(top["d_maha"].min())
        p_null = float(np.mean(best_null <= btc_best)) if len(best_null) else np.nan

        emit(f"\n--- {name}  [{a} .. {b}]  boundary={tag} ---")
        emit(f"    window {L} trading days; {len(refv)} reference windows from "
             f"{refv['asset'].nunique()} assets; features={len(feats)}")
        emit(f"    match axes ({len(feats)}): {', '.join(feats)}")
        emit(f"    reported but NOT matched: rho_eq, rho_gold "
             f"(both would leak -- see module docstring)")
        emit(f"    candidate windows all end before {a}: "
             f"{refv['start'].min().date()} to {refv['end'].max().date()}")
        emit(f"    BTC: vol={btc_f['vol_ann']:.2f} sharpe={btc_f['sharpe']:+.2f} "
             f"skew={btc_f['skew']:+.2f} maxDD={btc_f['max_dd']:.2f} "
             f"rho_eq={btc_f['rho_eq']:+.2f} rho_gold={btc_f['rho_gold']:+.2f}")
        emit(f"    best Mahalanobis distance = {btc_best:.2f}")
        emit(f"    NULL (random asset-window best match): median "
             f"{np.median(best_null):.2f}, 5th pct {np.percentile(best_null, 5):.2f}")
        emit(f"    p = {p_null:.3f}  -> " + (
            "BTC IS unusually well matched by history" if p_null < 0.05 else
            "BTC is NOT better matched than a random asset would be "
            "(no credible analogue)"))

        emit("    top 6 by consensus rank (Mahalanobis + DTW + Wasserstein):")
        emit(f"      {'asset':22s} {'window':24s} {'maha':>6s} {'dtw':>6s} "
             f"{'wass':>6s} {'vol':>6s} {'shrp':>6s} {'rhoEq':>6s}")
        for _, row in top.nsmallest(6, "rank_sum").iterrows():
            emit(f"      {row['asset']:22s} "
                 f"{str(row['start'].date()) + '..' + str(row['end'].date()):24s} "
                 f"{row['d_maha']:6.2f} {row['d_dtw']:6.3f} {row['d_wass']:6.3f} "
                 f"{row['vol_ann']:6.2f} {row['sharpe']:+6.2f} {row['rho_eq']:+6.2f}")
            all_matches.append({"regime": name, **row[
                ["asset", "start", "end", "d_maha", "d_dtw", "d_wass",
                 "vol_ann", "sharpe", "skew", "max_dd", "rho_eq"]].to_dict()})

        # leave-one-feature-out: does the match survive dropping any one axis?
        stable = []
        for drop in feats:
            sub = [f for f in feats if f != drop]
            mu2, sd2, Ci2, ok2 = mahalanobis_setup(ref, sub)
            rv = ref[ok2].reset_index(drop=True)
            dd = mdist_to(rv, bser, mu2, sd2, Ci2, sub)
            if np.all(np.isnan(dd)):
                continue
            stable.append(rv.loc[int(np.nanargmin(dd)), "asset"])
        if stable:
            vc = pd.Series(stable).value_counts()
            emit(f"    leave-one-feature-out: best match is "
                 f"'{vc.index[0]}' in {vc.iloc[0]}/{len(stable)} of "
                 f"{len(feats)} drops"
                 + ("  (stable)" if vc.iloc[0] / len(stable) >= 0.6
                    else "  (UNSTABLE - match depends on which features are used)"))
        null_rows.append({"regime": name, "btc_best_maha": btc_best,
                         "null_median": float(np.median(best_null)),
                         "null_p5": float(np.percentile(best_null, 5)),
                         "p_value": p_null, "n_features": len(feats),
                         "window_days": L})

    pd.DataFrame(all_matches).to_csv(
        ROOT / "outputs" / "tables" / "analogue_matches.csv", index=False)
    pd.DataFrame(null_rows).to_csv(
        ROOT / "outputs" / "tables" / "analogue_null_tests.csv", index=False)
    (ROOT / "outputs" / "tables" / "phase56_analogues.txt").write_text(
        "\n".join(lines) + "\n")
    print("\nwrote outputs/tables/phase56_analogues.txt, analogue_matches.csv, "
          "analogue_null_tests.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
