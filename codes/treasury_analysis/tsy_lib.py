"""
Shared library: Nelson-Siegel curve fitting, bond pricing, IRR.

Conventions
-----------
* Yields and coupons are handled as DECIMALS internally (e.g. 0.045), while the
  raw FRED data is in percent -- callers convert.
* Maturities are in YEARS.
* Bonds are semiannual-coupon, face = 100.
* A par bond issued at par yield y has coupon rate = y and price 100.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, brentq

# ----------------------------------------------------------------------------
# Nelson-Siegel
# ----------------------------------------------------------------------------
# y(tau) = b0 + b1 * (1-exp(-tau/lam))/(tau/lam)
#             + b2 * ((1-exp(-tau/lam))/(tau/lam) - exp(-tau/lam))
# b0 = long-run level, b1 = -slope (short-end), b2 = curvature.
# lam (lambda) controls where the curvature loads; fixed for a linear factor fit
# (Diebold-Li use a fixed lambda so the factors are OLS-recoverable).

DL_LAMBDA = 1.37  # years; Diebold-Li fix loading peak near 30 months (2.5y)

# canonical tenor label -> maturity in years (matches build_panel.py column names)
TENORS = {
    "1Mo": 1/12, "1.5Mo": 1.5/12, "2Mo": 2/12, "3Mo": 0.25, "4Mo": 4/12,
    "6Mo": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10,
    "20Y": 20, "30Y": 30,
}


def _ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """Return (n,3) matrix of NS factor loadings for maturities tau."""
    tau = np.asarray(tau, dtype=float)
    x = tau / lam
    # limit of (1-exp(-x))/x as x->0 is 1
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(x > 1e-8, (1.0 - np.exp(-x)) / x, 1.0)
    load0 = np.ones_like(tau)
    load1 = term
    load2 = term - np.exp(-x)
    return np.column_stack([load0, load1, load2])


def ns_yield(tau, betas, lam: float = DL_LAMBDA):
    """Evaluate NS yield(s) at maturity/maturities tau given betas=(b0,b1,b2)."""
    L = _ns_loadings(np.atleast_1d(tau), lam)
    y = L @ np.asarray(betas, dtype=float)
    return y[0] if np.isscalar(tau) or np.ndim(tau) == 0 else y


def fit_ns_ols(taus, yields, lam: float = DL_LAMBDA):
    """OLS fit of NS betas for one date (fixed lambda). yields in decimals.
    Returns (betas, rmse). NaNs are dropped."""
    taus = np.asarray(taus, float)
    yields = np.asarray(yields, float)
    m = np.isfinite(taus) & np.isfinite(yields)
    if m.sum() < 3:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    L = _ns_loadings(taus[m], lam)
    betas, *_ = np.linalg.lstsq(L, yields[m], rcond=None)
    resid = L @ betas - yields[m]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return betas, rmse


def fit_ns_nls(taus, yields, lam0: float = DL_LAMBDA):
    """Nonlinear fit that also estimates lambda. Returns (betas, lam, rmse)."""
    taus = np.asarray(taus, float)
    yields = np.asarray(yields, float)
    m = np.isfinite(taus) & np.isfinite(yields)
    taus, yields = taus[m], yields[m]
    if len(taus) < 4:
        b, r = fit_ns_ols(taus, yields, lam0)
        return b, lam0, r

    def resid(p):
        b0, b1, b2, lam = p
        return _ns_loadings(taus, max(lam, 0.05)) @ np.array([b0, b1, b2]) - yields

    b0, r0 = fit_ns_ols(taus, yields, lam0)
    p0 = np.array([b0[0], b0[1], b0[2], lam0])
    sol = least_squares(resid, p0, method="lm", max_nfev=2000)
    b = sol.x[:3]
    lam = float(sol.x[3])
    rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    return b, lam, rmse


# ----------------------------------------------------------------------------
# Bond pricing
# ----------------------------------------------------------------------------
def bond_price(coupon_rate: float, ytm: float, maturity_years: float,
               face: float = 100.0, freq: int = 2) -> float:
    """Price a semiannual-coupon bond.

    coupon_rate, ytm : annual, decimals
    maturity_years   : remaining years to maturity (>0)
    Uses standard actuarial discounting on the coupon grid measured back from
    maturity (the near-maturity fractional stub, if any, sits at the first cf).
    """
    if maturity_years <= 1e-9:
        return face
    c = coupon_rate * face / freq
    per = ytm / freq
    # coupon times (years from now): maturity, maturity-0.5, ... > 0
    n_full = int(np.floor(maturity_years * freq + 1e-9))
    times = [maturity_years - k / freq for k in range(n_full)]
    times = [t for t in times if t > 1e-9]
    times = sorted(times)
    pv = 0.0
    for t in times:
        pv += c / (1 + per) ** (t * freq)
    pv += face / (1 + per) ** (maturity_years * freq)
    return pv


# ----------------------------------------------------------------------------
# IRR
# ----------------------------------------------------------------------------
def xnpv(rate: float, times: np.ndarray, cfs: np.ndarray) -> float:
    """NPV with continuous year-fraction times and annual compounding."""
    return float(np.sum(cfs / (1.0 + rate) ** times))


def irr_annual(times, cfs) -> float:
    """Solve annualized IRR for cash flows at year-fraction `times`.
    Returns NaN if no sign change / not solvable in a sane range."""
    times = np.asarray(times, float)
    cfs = np.asarray(cfs, float)
    if not (np.any(cfs > 0) and np.any(cfs < 0)):
        return np.nan
    f = lambda r: xnpv(r, times, cfs)
    lo, hi = -0.9999, 5.0
    try:
        flo, fhi = f(lo), f(hi)
        if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
            # widen / scan for a bracket
            grid = np.linspace(lo, hi, 400)
            vals = np.array([f(r) for r in grid])
            sign = np.sign(vals)
            idx = np.where(np.diff(sign) != 0)[0]
            if len(idx) == 0:
                return np.nan
            a, b = grid[idx[0]], grid[idx[0] + 1]
            return float(brentq(f, a, b, maxiter=200))
        return float(brentq(f, lo, hi, maxiter=200))
    except Exception:  # noqa
        return np.nan


def par_bond_cashflows(coupon_rate: float, horizon_years: float,
                       sale_price: float, face: float = 100.0, freq: int = 2):
    """Build (times, cfs) for buying a par bond at 100 and selling at horizon.

    Coupons paid at 0.5, 1.0, ... up to horizon (cash, per user's choice).
    At `horizon_years` the investor receives the final coupon (if on grid) plus
    `sale_price`.
    """
    c = coupon_rate * face / freq
    times, cfs = [0.0], [-face]
    n = int(round(horizon_years * freq))
    for k in range(1, n + 1):
        t = k / freq
        cf = c
        if k == n:
            cf += sale_price
        times.append(t)
        cfs.append(cf)
    return np.array(times), np.array(cfs)
