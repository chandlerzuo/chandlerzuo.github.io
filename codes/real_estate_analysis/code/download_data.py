"""Download every public input the analysis uses into ../data/raw/.

No API keys. Files already present are kept unless --refresh is given.
history_pipeline.py calls the same fetchers, so running this first is optional,
but it lets the download be done (and checked) separately from the simulation.

  Zillow   ZHVI single-family middle tier and ZORI single-family, metro, monthly
  Census   ACS 2024 1-yr summary file: B25103 (property tax), B25077 (home value), geos
  BLS      CPI-U all items and each metro's rent of primary residence (public API v1)
  Freddie  PMMS 30-yr fixed mortgage rate, weekly
  French   1-month T-bill (RF), Fama-French factors file
  Yahoo    ^SP500TR and ^GSPC month-end closes
"""
import argparse, os
import metro_pipeline as mp
import history_pipeline as hp

METRO = [mp.ZHVI_URL, mp.ZORI_URL, mp.ACS_TAX_URL, mp.ACS_VAL_URL, mp.ACS_GEO_URL]
HISTORY = [(hp.PMMS_URL, "historicalweeklydata.xlsx"),
           (hp.FF_URL, "F-F_Research_Data_Factors_CSV.zip"),
           (hp.SPTR_URL, "sp500tr_monthly.json"),
           (hp.SPX_URL, "sp500_price_monthly.json")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    for url in METRO:
        print("fetch", os.path.relpath(mp.fetch(url, args.refresh), mp.DATA))
    for url, name in HISTORY:
        print("fetch", os.path.relpath(hp.fetch(url, name, args.refresh), mp.DATA))
    # BLS: the series used, via the public API (raw/history/cpi_subset.txt)
    cpi = hp.load_cpi(args.refresh)
    print(f"fetch raw/history/cpi_subset.txt ({cpi.shape[1]} series, "
          f"{cpi.index.min()}..{cpi.index.max()})")


if __name__ == "__main__":
    main()
