"""
REALIZED return of BUYING A HOME TO LIVE IN, in each metro, Aug 2010 -> Aug 2026,
against the S&P 500 and T-bills over the same window, plus the same calculation in
four sub-windows with different market regimes (WINDOWS).

The owner buys the metro's typical single-family home at the start of the window,
lives in it, pays the costs of ownership, and sells at the end of the window at that
month's actual price. There is no rental income. The return to living in the house is the
rent the owner did NOT pay for an equivalent home: each month's market rent, net of
property tax, insurance and maintenance, counts as a cash inflow. The IRR is
therefore the buy-vs-rent return: the rate at which buying beats renting the same
house, comparable to what a renter could earn by investing the cash instead.
IRR is solved on the monthly cash flows and annualized.

Everything is AFTER TAX: a 40% marginal rate on ordinary income (MARGINAL_TAX) and a
25% long-term capital-gains rate on gains realized at sale (LTCG_TAX), applied the
same way to every investment:
  home     rent saved is untaxed. Before 2018 a 40%-bracket household itemizes anyway
           (state income tax alone exceeds the standard deduction), so mortgage interest
           (acquisition debt up to $1M) and property tax are fully deductible. From 2018
           the $10k SALT cap is used up by state income tax and the standard deduction
           roughly doubled, so the benefit is only what interest adds above the standard
           deduction: T * (max(std, SALT cap + interest) - std), per calendar year, on
           acquisition debt up to $750k. Sale gain above the $500k primary-residence
           exclusion (married couple) taxed at the LTCG rate; basis = price + closing
           costs, amount realized net of selling cost.
  S&P 500  dividends treated as qualified: taxed at the LTCG rate as received and
           reinvested (DIVIDEND_TAX); remaining gain taxed at the LTCG rate when sold at
           the end of the window.
  T-bills  interest taxed at 40% as earned.
Pre-tax results are kept in *_pretax columns.

Property-tax assessment caps on owner-occupied homes (ASSESS_CAP), applied by
re-assessing every January: assessed value = min(market value, last year's assessed
value grown by the cap). CA Prop 13: +2%/yr; FL Save Our Homes: +min(3%, CPI)/yr;
TX homestead: +10%/yr; NY (NYC and Nassau class 1): +6%/yr and +20% per 5 years.
Capped states are taxed at the rate recent buyers pay (BUYER_TAX_RATE), because the
ACS all-owner rate already embeds long-time owners' cap savings. Every capped metro is
also run with no cap, i.e. tax on full market value at the same rate (*_nocap); the
difference (cap_*) is the return attributable to the cap.

Inputs (all public):
  price     Zillow ZHVI, single-family, middle tier, monthly (metro_pipeline)
  rent      (avoided) Zillow ZORI single-family, monthly, from 2015-01; before that, chained
            back with the BLS CPI "rent of primary residence" index for the same metro
  tax       ACS 2024 effective rate x that month's value; in capped states (CA, FL, TX,
            NY) the recent-buyer rate x the capped assessed value (see below)
  mortgage  30% down; Freddie Mac PMMS 30-yr fixed (owner-occupant), average of the
            window's first month
  S&P 500   S&P 500 total-return (^SP500TR) and price (^GSPC) indexes, month-end;
            their ratio gives the monthly dividend for the after-tax calculation
  T-bills   1-month Treasury bill return (Ken French data library, RF), compounded
  CPI       CPI-U all items, U.S. city average (BLS), for the inflation benchmark

Outputs: ../data/history_units.csv (overall window, one row per metro),
         ../data/history_windows.csv (sub-windows, one row per window x metro),
         ../data/benchmarks.csv (overall + sub-windows),
         ../data/history_cashflows.csv (monthly flows behind each overall IRR).
"""
import argparse, datetime, io, json, os, zipfile, re, xml.etree.ElementTree as ET
import numpy as np, pandas as pd
import re_model as m
import metro_pipeline as mp

HIST = os.path.join(mp.RAW, "history")
START, END = "2010-08", "2026-08"   # starts after the 2008-09 collapse
# Sub-windows, split at the turning points of the 12-metro average home price and of
# mortgage rates. Each is a fresh purchase at the window's start, at that month's
# mortgage rate, WITHOUT buy/sell costs (a 9% round trip would swamp 2-4 year windows),
# so the windows show what the market paid while the home was held.
WINDOWS = [("Bottoming out", "2010-08", "2012-03"),   # last leg down to the March 2012 trough
           ("Recovery", "2012-03", "2020-02"),        # cheap money, steady price gains
           ("Pandemic boom", "2020-02", "2022-06"),   # record-low rates, prices +15-17%/yr
           ("High rates", "2022-06", "2026-08")]      # 6.5-7% mortgages, flat prices

# download.bls.gov answers scripts with 403, so CPI comes from the keyless public API
# (v1: up to 25 series and 10 years per request).
BLS_API = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
CPI_FIRST_YEAR = 1997
PMMS_URL = "https://www.freddiemac.com/pmms/docs/historicalweeklydata.xlsx"
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_CSV.zip")
SPTR_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/%5ESP500TR"
            "?period1=1151712000&period2=1790000000&interval=1mo")
SPX_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
           "?period1=1151712000&period2=1790000000&interval=1mo")

# BLS CPI area code for each metro's "rent of primary residence" (SEHA) series.
CPI_AREA = {"New York, NY": "S12A", "Los Angeles, CA": "S49A", "Chicago, IL": "S23A",
            "Dallas, TX": "S37A", "Houston, TX": "S37B", "Washington, DC": "S35A",
            "Philadelphia, PA": "S12B", "Miami, FL": "S35B", "Atlanta, GA": "S35C",
            "Boston, MA": "S11A", "San Jose, CA": "S49B", "Seattle, WA": "S49D"}
# BLS publishes no rent index for San Jose; the San Francisco-Oakland-Hayward index (S49B)
# stands in for the Bay Area before Zillow's San Jose rent series starts in 2015.
CPI_ALL = "CUUR0000SA0"

RUN_LATE = "2023-08"        # second running-IRR buyer (figure 6): bought in the high-rate era
REFI_MONTH = "2012-12"     # sensitivity: one refinance near the post-crisis rate low
REFI_COST = 0.01           # refinance costs, % of balance, paid in cash
CONSERV = dict(maint=0.01) # sensitivity: double the maintenance reserve

MARGINAL_TAX = 0.40                 # ordinary income: interest, deductions
# Long-term capital gains for a household in a 40% ordinary bracket: 15% federal
# (the 32-35% brackets) + 3.8% net investment income tax + ~6% state.
LTCG_TAX = 0.25
# Nearly all S&P 500 dividends are qualified, so they are taxed at the long-term rate.
DIVIDEND_TAX = LTCG_TAX
SEC121_EXCLUSION = 500_000          # primary-residence gain exclusion, married couple
MORTGAGE_CAP_PRE2018, MORTGAGE_CAP = 1_000_000, 750_000   # deductible acquisition debt
SALT_CAP_START = pd.Period("2018-01", "M")  # TCJA: SALT cap + doubled standard deduction
SALT_CAP = 10_000
# Standard deduction, married filing jointly (IRS); only matters from 2018 on.
# Annual growth cap on a homestead's assessed value. "CPI3" = min(3%, prior-year CPI).
ASSESS_CAP = {"CA": dict(name="Prop 13", annual=0.02),
              "FL": dict(name="Save Our Homes", annual="CPI3"),
              "TX": dict(name="TX homestead cap", annual=0.10),
              "NY": dict(name="NYC class 1 cap", annual=0.06, five_year=0.20)}
# Effective tax rate paid by owners who bought 1-4 years before the 2024 ACS PUMS
# survey (archive/12_metro_forward, pums_pipeline.py); CA is the statutory 1.15%.
BUYER_TAX_RATE = {"Miami, FL": 0.009375, "Dallas, TX": 0.015455, "Houston, TX": 0.015470,
                  "New York, NY": 0.017360}
STD_DEDUCTION = {2018: 24_000, 2019: 24_400, 2020: 24_800, 2021: 25_100, 2022: 25_900,
                 2023: 27_700, 2024: 29_200, 2025: 31_500, 2026: 32_200}


# ---------------------------------------------------------------- fetch ------
def fetch(url, name, refresh=False):
    os.makedirs(HIST, exist_ok=True)
    path = os.path.join(HIST, name)
    if refresh or not os.path.exists(path):
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        with open(path, "wb") as f:
            f.write(body)
    return path


def monthly_index(dates):
    return pd.PeriodIndex(pd.to_datetime(dates), freq="M")


def load_cpi(refresh):
    """BLS CPI series needed here, monthly, cached as cpi_subset.txt in the layout
    of the BLS flat files (series_id, year, period, value)."""
    sub = os.path.join(HIST, "cpi_subset.txt")
    ids = sorted({f"CUUR{a}SEHA" for a in CPI_AREA.values()} | {CPI_ALL})
    if refresh or not os.path.exists(sub):
        import urllib.request
        os.makedirs(HIST, exist_ok=True)
        rows = []
        for y0 in range(CPI_FIRST_YEAR, datetime.date.today().year + 1, 10):
            body = json.dumps(dict(seriesid=ids, startyear=str(y0), endyear=str(y0 + 9))).encode()
            req = urllib.request.Request(BLS_API, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.load(r)
            if j["status"] != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API {y0}-{y0 + 9}: {j['status']} {j.get('message')}")
            rows += [(s["seriesID"], o["year"], o["period"], o["value"])
                     for s in j["Results"]["series"] for o in s["data"]]
        with open(sub, "w") as out:
            out.writelines("\t".join(r) + "\n" for r in sorted(rows))
    d = pd.read_csv(sub, sep="\t", header=None, usecols=[0, 1, 2, 3],
                    names=["series_id", "year", "period", "value"])
    d["series_id"] = d.series_id.str.strip()
    d = d[d.period.str.match(r"M(0[1-9]|1[0-2])$")]
    d["month"] = pd.PeriodIndex.from_fields(year=d.year, month=d.period.str[1:].astype(int), freq="M")
    d["value"] = pd.to_numeric(d.value, errors="coerce")       # BLS marks missing months "-"
    return d.pivot(index="month", columns="series_id", values="value")


def read_xlsx(path):
    """Minimal first-sheet xlsx reader (stdlib only)."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    ss = [''.join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("a:si", ns)]
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).find("a:sheetData", ns):
        row = {}
        for c in r:
            v = c.find("a:v", ns)
            v = None if v is None else (ss[int(v.text)] if c.get("t") == "s" else v.text)
            row[re.sub(r"\d", "", c.get("r"))] = v
        rows.append(row)
    return rows


def load_pmms(refresh):
    """Freddie Mac 30-yr fixed rate, monthly average of weekly readings (decimal)."""
    out = {}
    for r in read_xlsx(fetch(PMMS_URL, "historicalweeklydata.xlsx", refresh)):
        try:
            day = datetime.date(1899, 12, 30) + datetime.timedelta(days=float(r["A"]))
            out[day] = float(r["B"]) / 100
        except (TypeError, ValueError, KeyError):
            continue
    s = pd.Series(out)
    s.index = monthly_index(s.index)
    return s.groupby(level=0).mean()


def load_tbill(refresh):
    """1-month T-bill monthly return (decimal), Ken French RF."""
    z = zipfile.ZipFile(fetch(FF_URL, "F-F_Research_Data_Factors_CSV.zip", refresh))
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    rows = [l.split(",") for l in txt if re.match(r"^\s*\d{6}\s*,", l)]
    s = pd.Series({pd.Period(f"{r[0].strip()[:4]}-{r[0].strip()[4:]}", "M"): float(r[-1]) / 100
                   for r in rows})
    return s.sort_index()


def load_yahoo(url, name, refresh):
    """Yahoo Finance monthly chart, month-end close."""
    j = json.load(open(fetch(url, name, refresh)))["chart"]["result"][0]
    s = pd.Series(j["indicators"]["adjclose"][0]["adjclose"],
                  index=monthly_index(pd.to_datetime(j["timestamp"], unit="s")))
    return s.groupby(level=0).last()


def zillow_monthly(url, region_ids, refresh):
    z = pd.read_csv(mp.fetch(url, refresh)).set_index("RegionID")
    cols = [c for c in z.columns if c[:2] in ("19", "20")]
    t = z.loc[region_ids, cols].T
    t.index = monthly_index(t.index)
    return t


# ---------------------------------------------------------------- model ------
def irr_annual(cf, lo=-0.05, hi=0.05, tol=1e-10):
    """Annualized IRR of monthly cash flows, by bisection on the monthly rate.
    The bracket (about -46% to +80% a year) keeps (1+r)^240 finite; re_model.irr's
    -99% lower bound overflows at monthly frequency. NaN if no sign change."""
    t = np.arange(len(cf))
    f = lambda r: np.sum(cf / (1 + r) ** t)
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < tol or hi - lo < tol:
            break
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return (1 + mid) ** 12 - 1


def simulate(price, rent, tax, mort_rate, levered, maint=None, refi_rate=None, refi_at=None,
             costs=True, after_tax=True, interest_deduction=True):
    """Monthly cash flows for one owner-occupied home. price/rent/tax are monthly
    Series over the holding window (rent = market rent avoided, tax = monthly
    property-tax dollars). No vacancy or management: the owner lives in it.
    after_tax applies MARGINAL_TAX to deductions and LTCG_TAX to the sale gain;
    interest_deduction=False drops the mortgage-interest deduction only (for its
    attribution). Returns (cashflows, detail)."""
    maint = m.MAINT_RATE if maint is None else maint
    T = MARGINAL_TAX if after_tax else 0.0
    n = len(price) - 1
    p0 = price.iloc[0]
    loan = p0 * (1 - m.DOWN_PAYMENT) if levered else 0.0
    equity = p0 - loan
    closing = p0 * (m.BUY_CLOSING if costs else 0.0)
    cf = np.zeros(n + 1)
    cf[0] = -(equity + closing)
    pay = m.mortgage_payment(loan, mort_rate, m.LOAN_TERM_YEARS) if levered else 0.0
    bal, rate = loan, mort_rate
    cap = MORTGAGE_CAP_PRE2018 if price.index[0] < SALT_CAP_START else MORTGAGE_CAP
    deductible = min(1.0, cap / loan) if loan > 0 else 0.0     # share of interest deductible
    noi = np.zeros(n + 1)
    interest = np.zeros(n + 1)
    for t in range(1, n + 1):
        v = price.iloc[t - 1]                      # value at start of month for %-of-value costs
        # rent saved, net of the costs an owner pays and a renter does not
        noi[t] = rent.iloc[t] - tax.iloc[t] - v * (maint + m.INSURANCE_RATE) / 12
        interest[t] = bal * rate / 12
        bal = max(bal - (pay - interest[t]), 0.0) if levered else 0.0
        cf[t] = noi[t] - pay
        if levered and refi_at is not None and price.index[t] == refi_at:
            cf[t] -= bal * REFI_COST
            rate = refi_rate
            pay = m.mortgage_payment(bal, rate, m.LOAN_TERM_YEARS)
    # income-tax saving from owning, by calendar year, credited to that year's months
    tax_saved = np.zeros(n + 1)
    if T > 0:
        ded_int = interest * deductible if interest_deduction else np.zeros(n + 1)
        years_ = np.array([p.year for p in price.index])
        for y in np.unique(years_[1:]):
            idx = np.where((years_ == y) & (np.arange(n + 1) >= 1))[0]
            if y < SALT_CAP_START.year:            # itemizing anyway: interest + property tax
                tax_saved[idx] = T * (ded_int[idx] + tax.iloc[idx].values)
            else:                                  # only the excess over the standard deduction
                std, I = STD_DEDUCTION[y] * len(idx) / 12, ded_int[idx].sum()
                excess = max(SALT_CAP * len(idx) / 12 + I, std) - std
                if I > 0:
                    tax_saved[idx] = T * excess * ded_int[idx] / I
        cf[1:] += tax_saved[1:]
    sale = price.iloc[-1] * (1 - (m.SELL_COST if costs else 0.0))
    gain = sale - (p0 + closing)
    gain_tax = (LTCG_TAX if after_tax else 0.0) * max(gain - SEC121_EXCLUSION, 0.0)
    cf[-1] += sale - bal - gain_tax
    return cf, dict(noi=noi, sale_net=sale, payoff=bal, outlay=-cf[0], gain=gain,
                    gain_tax=gain_tax, tax_saved=tax_saved)


def years(start, end):
    return (pd.Period(end, "M") - pd.Period(start, "M")).n / 12


def load_inputs(refresh=False):
    """Monthly price, rent and CPI for every metro over the full START..END span."""
    metros, dropped, end, _ = mp.build(mp.TOP_N, refresh, as_of=END)
    assert end[:7] == END, f"latest Zillow month is {end[:7]}, expected {END}"
    months = pd.period_range(START, END, freq="M")
    ids = metros.region_id.tolist()
    zhvi = zillow_monthly(mp.ZHVI_URL, ids, refresh).reindex(months)
    zori = zillow_monthly(mp.ZORI_URL, ids, refresh).reindex(months)
    cpi_full = load_cpi(refresh).interpolate(limit_area="inside")   # BLS skipped Oct 2025
    cpi = cpi_full.reindex(months)
    series = {}
    for _, r in metros.iterrows():
        price = zhvi[r.region_id].interpolate(limit=1, limit_area="inside")  # Zillow skips Sep 2022 for San Jose
        z = zori[r.region_id]
        c = cpi[f"CUUR{CPI_AREA[r.metro]}SEHA"]
        first = z.first_valid_index()
        # market rent: ZORI where it exists, chained back with the metro's CPI rent index
        rent = z.copy()
        rent[rent.index < first] = z[first] * c[c.index < first] / c[first]
        assert price.notna().all() and rent.notna().all(), r.metro
        series[r.metro] = (price, rent, str(first))
    return dict(metros=metros, series=series, cpi=cpi, cpi_full=cpi_full, pmms=load_pmms(refresh),
                sp=load_yahoo(SPTR_URL, "sp500tr_monthly.json", refresh),
                spx=load_yahoo(SPX_URL, "sp500_price_monthly.json", refresh),
                tb=load_tbill(refresh), dropped=dropped)


def property_tax(r, price, cpi, capped=True):
    """Monthly property-tax dollars for a home bought at price.index[0]. In capped
    states the home is reassessed every January to min(market, capped growth of last
    year's assessment); capped=False taxes full market value at the same rate."""
    cap = ASSESS_CAP.get(r.state)
    if cap is None:
        return r.acs_tax_rate * price / 12
    rate = mp.PROP13_RATE if r.state == "CA" else BUYER_TAX_RATE[r.metro]
    if not capped:
        return rate * price / 12
    assessed = np.empty(len(price))
    history = {price.index[0].year: price.iloc[0]}       # assessed value by year
    a = price.iloc[0]
    for i, mo in enumerate(price.index):
        if i > 0 and mo.month == 1:
            y = mo.year
            if cap["annual"] == "CPI3":
                dec = cpi[pd.Period(f"{y - 1}-12", "M")] / cpi[pd.Period(f"{y - 2}-12", "M")] - 1
                g = min(0.03, max(dec, 0.0))
            else:
                g = cap["annual"]
            limit = a * (1 + g)
            if "five_year" in cap and (y - 5) in history:
                limit = min(limit, history[y - 5] * (1 + cap["five_year"]))
            a = min(price.iloc[i], limit)
            history[y] = a
        assessed[i] = a
    return pd.Series(rate * assessed / 12, index=price.index)


def sp500_after_tax(tr, px, T=DIVIDEND_TAX, T_gain=LTCG_TAX):
    """After-tax value of $1 in the S&P 500 over tr.index: monthly dividends (total
    return over price return) taxed at T and reinvested; remaining gain taxed at the
    long-term capital-gains rate T_gain on sale."""
    value, basis = 1.0, 1.0
    for t in range(1, len(tr)):
        price_ret = px.iloc[t] / px.iloc[t - 1]
        div = value * (tr.iloc[t] / tr.iloc[t - 1] - price_ret)
        value = value * price_ret + div * (1 - T)
        basis += div * (1 - T)
    return value - T_gain * max(value - basis, 0.0)


def benchmarks(inp, start, end):
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    n = years(start, end)
    sp, spx, tb, cpi = inp["sp"], inp["spx"], inp["tb"], inp["cpi"][CPI_ALL]
    tb_m = tb[s + 1: e]
    sp_at = sp500_after_tax(sp[s:e], spx[s:e])
    return [dict(benchmark="S&P 500 (total return)", cagr=sp_at ** (1 / n) - 1,
                 cagr_pretax=(sp[e] / sp[s]) ** (1 / n) - 1, start=sp[s], end=sp[e]),
            dict(benchmark="1-month T-bills", cagr=float(np.prod(1 + tb_m * (1 - MARGINAL_TAX))) ** (1 / n) - 1,
                 cagr_pretax=float(np.prod(1 + tb_m)) ** (1 / n) - 1, start=1.0, end=float(np.prod(1 + tb_m))),
            dict(benchmark="CPI inflation", cagr=(cpi[e] / cpi[s]) ** (1 / n) - 1,
                 cagr_pretax=(cpi[e] / cpi[s]) ** (1 / n) - 1, start=cpi[s], end=cpi[e])]


def evaluate(inp, start, end, costs=True, sensitivities=True):
    """One row per metro for a purchase at `start` and sale at `end`."""
    sl = slice(pd.Period(start, "M"), pd.Period(end, "M"))
    n = years(start, end)
    mort = inp["pmms"][pd.Period(start, "M")]
    refi_at = pd.Period(REFI_MONTH, "M")
    refi = inp["pmms"][refi_at]
    rows, flows = [], {}
    for _, r in inp["metros"].iterrows():
        price_all, rent_all, zori_start = inp["series"][r.metro]
        price, rent = price_all[sl], rent_all[sl]
        tax = property_tax(r, price, inp["cpi_full"][CPI_ALL])
        res = dict(size_rank=r.size_rank, metro=r.metro, state=r.state,
                   census_region=r.census_region, start=start, end=end, years=n,
                   price_start=price.iloc[0], price_end=price.iloc[-1],
                   rent_start=rent.iloc[0], rent_end=rent.iloc[-1], zori_start=zori_start,
                   gross_yield_start=rent.iloc[0] * 12 / price.iloc[0],
                   gross_yield_end=rent.iloc[-1] * 12 / price.iloc[-1],
                   price_cagr=(price.iloc[-1] / price.iloc[0]) ** (1 / n) - 1,
                   rent_cagr=(rent.iloc[-1] / rent.iloc[0]) ** (1 / n) - 1,
                   tax_rate=(mp.PROP13_RATE if r.state == "CA" else
                             BUYER_TAX_RATE.get(r.metro, r.acs_tax_rate)),
                   mortgage_rate=mort)
        runs = [("cash", False, {}), ("lev", True, {})]
        if sensitivities:
            runs += [("cash_conserv", False, CONSERV), ("lev_conserv", True, CONSERV)]
            if sl.start < refi_at < sl.stop:
                runs += [("lev_refi", True, dict(refi_rate=refi, refi_at=refi_at))]
        for key, lev, kw in runs:
            cf, det = simulate(price, rent, tax, mort, lev, costs=costs, **kw)
            res[f"irr_{key}"] = irr_annual(cf)
            if key in ("cash", "lev"):
                flows[(r.metro, key)] = cf
                res[f"gain_tax_{key}"] = det["gain_tax"]
                cfp, _ = simulate(price, rent, tax, mort, lev, costs=costs, after_tax=False, **kw)
                res[f"irr_{key}_pretax"] = irr_annual(cfp)
                if lev:                    # exact waterfall: price -> + rent saved -> + interest -> + cap
                    capped = r.state in ASSESS_CAP
                    tax_mkt = (property_tax(r, price, inp["cpi_full"][CPI_ALL], capped=False)
                               if capped else tax)
                    cf_p = np.zeros_like(cf)
                    cf_p[0] = cf[0]
                    cf_p[-1] = det["sale_net"] - det["payoff"] - det["gain_tax"]
                    w_price = irr_annual(cf_p)
                    c1, _ = simulate(price, rent, tax_mkt, mort, lev, costs=costs, interest_deduction=False, **kw)
                    c2, _ = simulate(price, rent, tax_mkt, mort, lev, costs=costs, **kw)
                    w1, w2 = irr_annual(c1), irr_annual(c2)
                    res.update(wf_price_lev=w_price, wf_rent_lev=w1 - w_price, wf_int_lev=w2 - w1,
                               wf_cap_lev=res["irr_lev"] - w2)
                if lev:                    # interest attribution: same loan, no interest deduction
                    cfi, _ = simulate(price, rent, tax, mort, lev, costs=costs,
                                      interest_deduction=False, **kw)
                    res["irr_lev_noint"] = irr_annual(cfi)
                    res["int_saving_lev"] = res["irr_lev"] - res["irr_lev_noint"]
                if r.state in ASSESS_CAP:  # cap attribution: same purchase, taxed on full market value
                    cfn, _ = simulate(price, rent, property_tax(r, price, inp["cpi_full"][CPI_ALL],
                                                                capped=False), mort, lev, costs=costs, **kw)
                    res[f"irr_{key}_nocap"] = irr_annual(cfn)
                    res[f"cap_{key}"] = res[f"irr_{key}"] - res[f"irr_{key}_nocap"]
                    res["assess_cap"] = ASSESS_CAP[r.state]["name"]
        # price-only return: buy and sell (with the same transaction costs), ignoring the
        # rent saved and ownership costs. irr_cash - irr_price_only is what living rent-free added.
        buy, sell = (m.BUY_CLOSING, m.SELL_COST) if costs else (0.0, 0.0)
        basis, amount = price.iloc[0] * (1 + buy), price.iloc[-1] * (1 - sell)
        net = amount - LTCG_TAX * max(amount - basis - SEC121_EXCLUSION, 0.0)
        res["irr_price_only"] = (net / basis) ** (1 / n) - 1
        cf, det = simulate(price, rent, tax, mort, False, costs=costs)
        res["net_rent_saved_yield_avg"] = (det["noi"][1:] * 12 / price.iloc[:-1].values).mean()
        rows.append(res)
    cfl = pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in flows.items()},
                       index=pd.period_range(start, end, freq="M").astype(str))
    return pd.DataFrame(rows), cfl, dict(mortgage=mort, refi=refi)


def running(inp, start=START, end=END, min_months=1, levered=True):
    """Running after-tax IRR of one purchase made at `start` and sold at every later
    month t: 3% closing cost at purchase; at t, 6% selling cost, loan payoff and gain
    tax. Next to it, the S&P 500 and T-bills bought at `start` and sold at t (no
    trading costs). NaN where the sale does not repay the loan (all flows negative).
    One row per month from start + min_months, with `years` held, so the cumulative
    return is (1 + IRR) ** years - 1."""
    s0 = pd.Period(start, "M")
    months = pd.period_range(s0 + min_months, pd.Period(end, "M"), freq="M")
    out = {"month": months.astype(str), "years": [(t - s0).n / 12 for t in months],
           "mortgage_rate": inp["pmms"][s0]}
    mort = inp["pmms"][s0]
    for _, r in inp["metros"].iterrows():
        price_all, rent_all, _ = inp["series"][r.metro]
        price, rent = price_all[s0:], rent_all[s0:]
        tax = property_tax(r, price, inp["cpi_full"][CPI_ALL])
        vals = []
        for t in months:
            cf, _ = simulate(price[:t], rent[:t], tax[:t], mort, levered, costs=True)
            vals.append(irr_annual(cf, lo=-0.9))      # short holds: very negative monthly rates
        out[r.metro] = vals
    sp, spx, tb = inp["sp"], inp["spx"], inp["tb"]
    out["S&P 500"] = [sp500_after_tax(sp[s0:t], spx[s0:t]) ** (12 / (t - s0).n) - 1 for t in months]
    out["T-bills"] = [float(np.prod(1 + tb[s0 + 1:t] * (1 - MARGINAL_TAX))) ** (12 / (t - s0).n) - 1
                      for t in months]
    return pd.DataFrame(out)


def build(refresh=False):
    inp = load_inputs(refresh)
    df, cfl, info = evaluate(inp, START, END, costs=True)
    bench = [dict(window="Overall", **b) for b in benchmarks(inp, START, END)]
    wins = []
    for name, s, e in WINDOWS:
        w, _, wi = evaluate(inp, s, e, costs=False, sensitivities=False)
        wins.append(w.assign(window=name))
        bench += [dict(window=name, **b) for b in benchmarks(inp, s, e)]
    run_irr = running(inp)
    run_irr_late = running(inp, start=RUN_LATE)
    run_irr_late.to_csv(os.path.join(mp.DATA, f"history_running_{RUN_LATE[:4]}.csv"), index=False)
    return df, pd.concat(wins, ignore_index=True), pd.DataFrame(bench), cfl, info, run_irr


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download raw data")
    args = ap.parse_args()
    df, win, bench, cfl, info, run_irr = build(args.refresh)
    run_irr.to_csv(os.path.join(mp.DATA, "history_running.csv"), index=False)
    df.to_csv(os.path.join(mp.DATA, "history_units.csv"), index=False)
    win.to_csv(os.path.join(mp.DATA, "history_windows.csv"), index=False)
    bench.to_csv(os.path.join(mp.DATA, "benchmarks.csv"), index=False)
    cfl.to_csv(os.path.join(mp.DATA, "history_cashflows.csv"), index_label="month")

    P = lambda x: f"{x*100:5.1f}"
    bo = bench[bench.window == "Overall"].set_index("benchmark").cagr
    print("=" * 112)
    print(f"  REALIZED RETURNS  {START} -> {END} ({years(START, END):.0f} yrs) | mortgage "
          f"{info['mortgage']*100:.2f}% (PMMS) | refi {REFI_MONTH} at {info['refi']*100:.2f}%")
    print("=" * 112)
    for k, v in bo.items():
        print(f"  {k:24s} {P(v)}%/yr")
    print("-" * 112)
    print(f"  (pre-tax: S&P {P(bench[bench.window == 'Overall'].cagr_pretax.iloc[0])}%, "
          f"T-bills {P(bench[bench.window == 'Overall'].cagr_pretax.iloc[1])}%; all figures below after "
          f"{MARGINAL_TAX*100:.0f}% ordinary / {LTCG_TAX*100:.0f}% long-term gains tax)")
    cols = ["metro", "price_cagr", "gross_yield_start", "irr_price_only", "irr_cash_pretax", "irr_cash",
            "irr_lev_pretax", "irr_lev", "irr_lev_refi", "irr_cash_conserv", "irr_lev_conserv",
            "int_saving_lev", "gain_tax_cash"]
    f = df[cols].sort_values("irr_cash", ascending=False).copy()
    for c in cols[1:-1]:
        f[c] = f[c].map(P)
    f["gain_tax_cash"] = f.gain_tax_cash.map(lambda v: f"{v:,.0f}")
    print(f.to_string(index=False))
    for _, r in df[df.state.isin(list(ASSESS_CAP))].iterrows():
        print(f"  {r.assess_cap:17s} {r.metro:18s}: cash {P(r.irr_cash)} vs no cap {P(r.irr_cash_nocap)} "
              f"(+{P(r.cap_cash)} pts); levered {P(r.irr_lev)} vs {P(r.irr_lev_nocap)} (+{P(r.cap_lev)} pts)")
    sp = bo["S&P 500 (total return)"]; tb = bo["1-month T-bills"]
    print(f"\n  All-cash IRR beats S&P 500 in {(df.irr_cash > sp).sum()}/{len(df)}; "
          f"levered in {(df.irr_lev > sp).sum()}; levered+refi in {(df.irr_lev_refi > sp).sum()}. "
          f"Beats T-bills (all-cash) in {(df.irr_cash > tb).sum()}/{len(df)}.")

    print("\n  SUB-WINDOWS (fresh purchase at window start, no buy/sell costs) — median across metros")
    for name, s, e in WINDOWS:
        w = win[win.window == name]
        b = bench[bench.window == name].set_index("benchmark").cagr
        print(f"  {name:14s} {s}->{e}  mort {w.mortgage_rate.iloc[0]*100:4.2f}%  "
              f"price {P(w.price_cagr.median())}  cash {P(w.irr_cash.median())} "
              f"[{P(w.irr_cash.min())},{P(w.irr_cash.max())}]  lev {P(w.irr_lev.median())}  | "
              f"S&P {P(b['S&P 500 (total return)'])}  T-bill {P(b['1-month T-bills'])}  "
              f"CPI {P(b['CPI inflation'])}  | cash>S&P {(w.irr_cash > b['S&P 500 (total return)']).sum()}/12  "
              f"| cap cash " + " ".join(f"{a[:2]}{P(x)}" for a, x in zip(w[w.cap_cash.notna()].metro,
                                                                  w[w.cap_cash.notna()].cap_cash)))
    print("  Saved -> data/history_units.csv, data/history_windows.csv, data/benchmarks.csv, "
          "data/history_cashflows.csv, data/history_running.csv")


if __name__ == "__main__":
    run()
