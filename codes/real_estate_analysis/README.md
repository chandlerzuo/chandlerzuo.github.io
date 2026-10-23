# Buy or Rent — Realized IRR of Owning a Home vs the S&P 500, 2010–2026

What did buying a typical single-family house to live in, in August 2010, and selling
in August 2026 actually return? (2010 is chosen to leave out the 2008–2010 collapse.) The same calculation is repeated in four sub-windows
with different market regimes (bottoming out 2010–12, recovery 2012–20, pandemic boom
2020–22, high rates 2022–26), as fresh purchases without buy/sell costs. All results are after tax
(40% on ordinary income, 25% on long-term gains and qualified dividends) on every investment (pre-tax kept as `*_pretax`), and the return
attributable to property-tax assessment caps (CA Prop 13, FL Save Our Homes, TX homestead,
NYC class 1) is broken out (`cap_*`). Cash flows are the net effect of buying the house and
no longer renting an equivalent one, so the rent saved each month (actual rent history)
is an inflow. One IRR per metro, for the ten most
populous U.S. metros (New York, Los Angeles, Chicago, Dallas, Houston, Washington,
Philadelphia, Miami, Atlanta, Boston) plus San Jose (Silicon Valley) and Seattle. Each is compared
with the S&P 500 (total return), 1-month T-bills and CPI inflation over the same window.

## Layout

```
real_estate_analysis/
├── run_all.sh                 # download -> simulate -> figures
├── code/
│   ├── download_data.py       # fetches every public input into data/raw/ (cached)
│   ├── metro_pipeline.py      # metro selection + Zillow ZHVI/ZORI + Census ACS tax inputs
│   ├── history_pipeline.py    # monthly simulation per metro, overall + sub-windows, benchmarks
│   ├── make_history_charts.py # fig1-6 (Economist-style) + history_summary.csv
│   └── re_model.py            # shared assumptions + mortgage helpers
├── data/                      # NOT committed; created by run_all.sh
│   ├── raw/                   # downloads (Zillow, Census in raw/, the rest in raw/history/)
│   ├── history_units.csv      # 2010-2026, one row per metro: inputs, IRRs (cash, mortgage, refi, 1% maintenance)
│   ├── history_windows.csv    # sub-windows, one row per window x metro
│   ├── benchmarks.csv         # S&P 500 TR, T-bills, CPI per window: start/end levels and CAGR
│   ├── history_cashflows.csv  # monthly cash flows behind each all-cash and levered IRR
│   ├── history_running.csv    # mortgage buyer of Aug 2010 selling in each later month (fig 5)
│   └── history_running_2023.csv # same for a buyer in Aug 2023 (fig 6)
└── outputs/                   # fig1_irr_vs_benchmarks, fig2_return_sources, fig3_windows,
                               # fig4_mortgage_irr, fig5_running_irr, fig6_running_irr_2023 (PNG,
                               # committed for the post), history_summary.csv (not committed)
```

The write-up is the blog post `blog/_posts/2026-09-16-real_estate.md`.

## Data sources (all public, no API keys)

| Input | Source |
|---|---|
| Price | Zillow ZHVI, single-family, middle tier, monthly |
| Rent saved | Zillow ZORI single-family, monthly from 2015-01; chained back with BLS CPI "rent of primary residence" for the same metro |
| Property tax | Census ACS 2024 1-yr effective rate (B25103 / B25077) × value; capped states (CA, FL, TX, NY): recent-buyer rate × capped assessed value |
| Mortgage | 30% down; Freddie Mac PMMS 30-yr fixed, average of the purchase month (owner-occupant rate) |
| S&P 500 | ^SP500TR total-return and ^GSPC price indexes, month-end (Yahoo Finance chart API) |
| T-bills | Ken French data library, RF (1-month T-bill) |
| Inflation | BLS CPI-U all items, U.S. city average |

Zillow revises its entire history with every monthly release, so the two Zillow files
are pinned to the release with data through August 2026, as archived by the Wayback
Machine (byte-identical to the files used for the post). The other sources only append
new months, and the study window is fixed at 2010-08 to 2026-08. BLS CPI comes from the
public API v1 (no key), since download.bls.gov rejects scripted requests.

## Reproduce

```bash
pip install numpy pandas matplotlib
./run_all.sh             # downloads to data/raw (cached), then simulates and draws fig1-6
./run_all.sh --refresh   # re-download everything
```

A clean run reproduces every CSV in `data/` and `outputs/history_summary.csv` exactly.

*Illustrative analysis, not investment advice. See the post's limitations.*
