"""
Shared assumptions and finance helpers (mortgage payment and balance, NPV).
history_pipeline.py uses the operating-cost, transaction and financing constants
and mortgage_payment(); underwrite() is the forward-looking annual model kept for
reference (see archive/12_metro_forward).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. GLOBAL INVESTMENT ASSUMPTIONS  (edit these to run scenarios)
# ----------------------------------------------------------------------------
HOLD_YEARS        = 10       # holding period
APPRECIATION      = 0.03     # annual property price appreciation
RENT_GROWTH       = 0.03     # annual rent growth
VACANCY           = 0.05     # vacancy + credit loss (fraction of gross rent)
MAINT_RATE        = 0.005    # annual maintenance reserve, % of value (HOA covers building)
INSURANCE_RATE    = 0.0035   # annual insurance, % of value
MGMT_RATE         = 0.00     # property-management fee, % of collected rent (0 = self-managed)
SELL_COST         = 0.06     # selling costs at exit (commission + transfer), % of sale price
BUY_CLOSING       = 0.03     # purchase closing costs, % of price (added to initial outlay)

# Financing (levered scenario)
DOWN_PAYMENT      = 0.30     # down payment
MORTGAGE_RATE     = 0.0725   # 30-yr investment-property rate (~2025)
LOAN_TERM_YEARS   = 30

DISCOUNT_RATE     = 0.08     # required return used for NPV (investor hurdle)


# ----------------------------------------------------------------------------
# 2. FINANCE HELPERS
# ----------------------------------------------------------------------------
def npv(rate, cashflows):
    """cashflows[0] at t=0."""
    t = np.arange(len(cashflows))
    return np.sum(np.asarray(cashflows) / (1 + rate) ** t)


def irr(cashflows, lo=-0.99, hi=1.0):
    """Robust IRR via bisection on NPV sign change. Returns np.nan if none in range."""
    f_lo, f_hi = npv(lo, cashflows), npv(hi, cashflows)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        # try to widen upper bound once
        hi = 5.0
        f_hi = npv(hi, cashflows)
        if f_lo * f_hi > 0:
            return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def mortgage_payment(principal, annual_rate, term_years):
    r = annual_rate / 12
    n = term_years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def loan_balance(principal, annual_rate, term_years, months_elapsed):
    r = annual_rate / 12
    n = term_years * 12
    pay = mortgage_payment(principal, annual_rate, term_years)
    if r == 0:
        return principal - pay * months_elapsed
    return principal * (1 + r) ** months_elapsed - pay * (((1 + r) ** months_elapsed - 1) / r)


# ----------------------------------------------------------------------------
# 3. PER-UNIT UNDERWRITING  ->  cash flows, IRR, NPV
# ----------------------------------------------------------------------------
def insurance_rate(row):
    """Per-unit insurance override (e.g. PUMS reported premium / value); fall back to global."""
    try:
        v = row["insurance_rate"]
        if pd.notna(v):
            return float(v)
    except (KeyError, TypeError):
        pass
    return INSURANCE_RATE


def underwrite(row, levered):
    price = row["price"]
    # per-unit appreciation override (real historical CAGR); fall back to global
    appr = APPRECIATION
    try:
        v = row["appreciation"]
        if pd.notna(v):
            appr = float(v)
    except (KeyError, TypeError):
        pass
    ins_rate = insurance_rate(row)
    # ---- Year-1 operating figures ----
    gross_rent_yr = row["monthly_rent"] * 12
    hoa_yr        = row["hoa_monthly"] * 12
    tax_yr        = price * row["tax_rate"]

    # Initial outlay
    if levered:
        loan   = price * (1 - DOWN_PAYMENT)
        equity = price * DOWN_PAYMENT
        pmt_m  = mortgage_payment(loan, MORTGAGE_RATE, LOAN_TERM_YEARS)
        debt_service_yr = pmt_m * 12
    else:
        loan = 0.0
        equity = price
        debt_service_yr = 0.0

    initial_outlay = equity + price * BUY_CLOSING
    cashflows = [-initial_outlay]

    noi_year1 = None
    for yr in range(1, HOLD_YEARS + 1):
        g = 1 + RENT_GROWTH
        a = 1 + appr
        gross_rent = gross_rent_yr * g ** (yr - 1)
        egi        = gross_rent * (1 - VACANCY)                 # effective gross income
        mgmt       = egi * MGMT_RATE
        value_yr   = price * a ** (yr - 1)
        taxes      = tax_yr * a ** (yr - 1)                     # taxes track value
        hoa        = hoa_yr * g ** (yr - 1)                     # HOA drifts with inflation
        maint      = value_yr * MAINT_RATE
        ins        = value_yr * ins_rate

        noi = egi - mgmt - taxes - hoa - maint - ins            # unlevered NOI
        if yr == 1:
            noi_year1 = noi
        cf = noi - debt_service_yr                              # levered cash flow
        cashflows.append(cf)

    # ---- Terminal sale at end of HOLD_YEARS ----
    sale_price = price * (1 + appr) ** HOLD_YEARS
    sale_net   = sale_price * (1 - SELL_COST)
    payoff     = loan_balance(loan, MORTGAGE_RATE, LOAN_TERM_YEARS, HOLD_YEARS * 12) if levered else 0.0
    cashflows[-1] += sale_net - payoff

    return dict(
        equity_invested=initial_outlay,
        noi_year1=noi_year1,
        cap_rate=noi_year1 / price,
        gross_yield=gross_rent_yr / price,
        annual_cf_year1=cashflows[1],
        coc_year1=cashflows[1] / initial_outlay,   # cash-on-cash, yr1
        irr=irr(cashflows),
        npv=npv(DISCOUNT_RATE, cashflows),
    )

