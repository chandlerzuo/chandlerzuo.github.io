"""
Metro selection and metro-level inputs, shared by history_pipeline.py.

Selection: the 10 most populous U.S. metropolitan areas (Zillow SizeRank) plus
San Jose (Silicon Valley: the San Jose-Sunnyvale-Santa Clara MSA) and Seattle.

Inputs (all free, all public, all metro-level):
  price        = Zillow ZHVI, single-family, middle tier (33rd-67th pct), monthly
  rent         = Zillow ZORI, single-family rentals, monthly (from 2015-01)
  property tax = Census ACS 1-yr (2024): median real-estate taxes paid (B25103)
                 / median owner-occupied value (B25077). California is handled
                 separately (Prop 13) by the caller.
  appreciation = ZHVI single-family CAGR, 20-yr and 10-yr windows ending at the latest month

Raw downloads are cached in ../data/raw/ (use --refresh in the caller to re-download).
"""
import os, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RAW = os.path.join(DATA, "raw")
TOP_N = 10                                       # metros with SizeRank 1..TOP_N
EXTRA_METROS = ("San Jose, CA", "Seattle, WA")       # kept regardless of rank (San Jose = Silicon Valley)

# Zillow revises its whole history every month, so the files are pinned to the release
# with data through Aug 2026, as archived by the Wayback Machine (byte-identical to the
# files used for the post). Drop the WAYBACK prefix to use the latest release instead.
ZILLOW = "https://files.zillowstatic.com/research/public_csvs/"
WAYBACK = "https://web.archive.org/web/{}id_/"
ZHVI_URL = (WAYBACK.format("20260928214703") + ZILLOW +
            "zhvi/Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv")
ZORI_URL = (WAYBACK.format("20260928214717") + ZILLOW +
            "zori/Metro_zori_uc_sfr_sm_month.csv")
ACS_YEAR = 2024
ACS_BASE = (f"https://www2.census.gov/programs-surveys/acs/summary_file/{ACS_YEAR}/"
            f"table-based-SF/")
ACS_TAX_URL = ACS_BASE + f"data/1YRData/acsdt1y{ACS_YEAR}-b25103.dat"
ACS_VAL_URL = ACS_BASE + f"data/1YRData/acsdt1y{ACS_YEAR}-b25077.dat"
ACS_GEO_URL = ACS_BASE + f"documentation/Geos{ACS_YEAR}1YR.txt"

# California reassesses at purchase to ~1.15% of price, then caps assessed growth
# at 2%/yr (Prop 13), so the resident median tax rate does not apply to a buyer.
PROP13_RATE, PROP13_CAP = 0.0115, 0.02

CENSUS_REGION = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest":   "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South":     "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West":      "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {s: r for r, ss in CENSUS_REGION.items() for s in ss.split()}


# ---------------------------------------------------------------- fetch ------
def fetch(url, refresh=False):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, os.path.basename(url))
    if refresh or not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=120) as r:
            body = r.read()
        with open(path, "wb") as f:
            f.write(body)
    return path


def _key(name, state):
    """Join key: first principal city + first state ('New York-Newark-..., NY-NJ')."""
    return name.split("-")[0].split("/")[0].strip().lower() + "|" + state.split("-")[0]


def load_acs(refresh):
    geo = pd.read_csv(fetch(ACS_GEO_URL, refresh), sep="|", dtype=str, encoding="utf-8-sig")
    geo = geo[geo.GEO_ID.str.startswith("310M") & geo.NAME.str.endswith("Metro Area")]
    tax = pd.read_csv(fetch(ACS_TAX_URL, refresh), sep="|")[["GEO_ID", "B25103_E001"]]
    val = pd.read_csv(fetch(ACS_VAL_URL, refresh), sep="|")[["GEO_ID", "B25077_E001"]]
    df = geo[["GEO_ID", "NAME"]].merge(tax, on="GEO_ID").merge(val, on="GEO_ID")
    df["cbsa"] = df.GEO_ID.str[-5:]
    nm = df.NAME.str.replace(" Metro Area", "", regex=False).str.rsplit(", ", n=1)
    df["key"] = [_key(c, s) for c, s in nm]
    df["acs_tax_rate"] = pd.to_numeric(df.B25103_E001, errors="coerce") / \
                         pd.to_numeric(df.B25077_E001, errors="coerce")
    return df.rename(columns={"NAME": "cbsa_name", "B25103_E001": "acs_med_tax",
                              "B25077_E001": "acs_med_value"})[
        ["key", "cbsa", "cbsa_name", "acs_med_tax", "acs_med_value", "acs_tax_rate"]]


def cagr(series, end, years):
    start = f"{int(end[:4]) - years}{end[4:7]}"
    col = next((c for c in series.index if c.startswith(start)), None)
    if col is None or pd.isna(series[col]) or pd.isna(series[end]):
        return np.nan
    return (series[end] / series[col]) ** (1 / years) - 1


# ---------------------------------------------------------------- build ------
def build(top_n=TOP_N, refresh=False, extra=EXTRA_METROS, as_of=None):
    """as_of ("YYYY-MM") caps the snapshot month, so newer Zillow releases still
    reproduce a fixed study window."""
    zhvi = pd.read_csv(fetch(ZHVI_URL, refresh))
    zori = pd.read_csv(fetch(ZORI_URL, refresh)).set_index("RegionID")
    acs = load_acs(refresh)

    months = [c for c in zhvi.columns if c[:2] in ("19", "20")]
    end = next(c for c in reversed(months) if (as_of is None or c[:7] <= as_of)
               and c in zori.columns
               and zhvi[c].notna().any() and zori[c].notna().any())
    us_price = float(zhvi.loc[zhvi.RegionType == "country", end].iloc[0])

    msa = zhvi[zhvi.RegionType == "msa"]
    # select on true rank (not row count) so a metro missing from the SFR file is
    # reported as absent rather than silently replaced by rank top_n+1
    metros = msa[(msa.SizeRank <= top_n) | msa.RegionName.isin(extra)].sort_values("SizeRank")
    rows, dropped = [], []
    for _, z in metros.iterrows():
        city, st = z.RegionName.rsplit(", ", 1)
        a = acs[acs.key == _key(city, st)]
        rent = zori[end].get(z.RegionID, np.nan)
        rec = dict(size_rank=int(z.SizeRank), metro=z.RegionName, state=st,
                   census_region=STATE_REGION.get(st), price=z[end], monthly_rent=rent,
                   appr_20y=cagr(z, end, 20), appr_10y=cagr(z, end, 10),
                   acs_tax_rate=a.acs_tax_rate.iloc[0] if len(a) else np.nan,
                   cbsa=a.cbsa.iloc[0] if len(a) else None,
                   cbsa_name=a.cbsa_name.iloc[0] if len(a) else None)
        missing = [k for k in ("price", "monthly_rent", "appr_20y", "acs_tax_rate")
                   if pd.isna(rec[k])]
        if missing:
            dropped.append((z.RegionName, missing)); continue
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["region_id"] = metros.set_index("RegionName").loc[df.metro, "RegionID"].values
    df["as_of"] = end[:7]
    return df, dropped, end, us_price

