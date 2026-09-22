# Bitcoin's macro betas, regimes and historical analogues

Code for the blog post *Bitcoin's Quiet Conversion*.

**No data is committed.** Every input is downloaded by the scripts here from
public sources that need **no API key and no registration**, then transformed
into the analysis panel. Running `./run_all.sh` on a clean checkout reproduces
`data/` and everything in `outputs/` from scratch.

## Quick start

```bash
./run_all.sh          # creates its own venv if none exists; ~35 min end to end
```

Or step by step:

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy statsmodels \
    matplotlib scikit-learn arch hmmlearn pyarrow openpyxl
.venv/bin/python fetch_fred.py        # 88 FRED series          -> data/raw/
.venv/bin/python fetch_factors.py     # Fama-French factors     -> data/raw/
.venv/bin/python fetch_btc.py         # BTC, 3 sources + QA     -> data/raw/
.venv/bin/python fetch_assets.py      # metals, ETFs, OECD, JST -> data/raw/
.venv/bin/python fetch_crypto.py      # altcoins, funding       -> data/raw/
.venv/bin/python build_panel.py       # clean + align           -> data/processed/
```

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| **Fetch** | `fetch_fred.py` | 88 macro series, one request each, into `data/raw/` |
| | `fetch_factors.py` | Fama-French 5 factors + momentum, daily and monthly |
| | `fetch_btc.py` | Bitcoin daily from CoinMetrics, Bitstamp and Coinbase, with a cross-source disagreement report |
| | `fetch_assets.py` | LBMA metals (1968+), SPDR ETF NAVs (2003+), OECD national equity indices (1957+), Jordà-Schularick-Taylor macrohistory |
| | `fetch_crypto.py` | 13 altcoins, stablecoin supply, Binance perpetual funding |
| **Build** | `build_panel.py` | `data/processed/panel_daily.parquet`, `panel_weekly.parquet`, and a generated `DATA_DICTIONARY.md` (written on first run) |
| **Analyse** | `breaks_correlation.py` | **The headline result**: structural breaks in the equity/gold/dollar correlation and in beta, by partial-change sup-F with a wild bootstrap |
| | `betas_static.py` | Static and split-sample factor models; thesis horse race; Dimson lead-lag betas |
| | `betas_tvp.py` | Rolling, DCC-GARCH and Kalman time-varying betas; the ρ vs σ-ratio decomposition |
| | `regimes.py` | Bai-Perron breaks, Markov-switching betas, Gaussian HMM regimes; partial-break sup-F with wild bootstrap |
| | `analogues.py` | Per-regime risk profiles; daily analogue matching against 65 assets, 1926-2026 |
| | `validate_analogues.py` | Out-of-sample test of whether matched analogues predict anything |
| | `crypto_controls.py` | Do macro loadings survive crypto-native controls? |
| | `analogues_em.py` | Monthly analogue matching including 12 national stockmarkets |
| **Figures** | `figures.py`, `fig_essay.py`, `fig_essay_analogues.py` | `outputs/figures/` |

`fetchlib.py` is the shared fetch layer. Every download is cached in `data/raw/`
and logged to `data/provenance.jsonl` with its URL, retrieval timestamp, byte
count and SHA-256, so a rerun is incremental and each figure is traceable to a
specific payload. Delete `data/raw/` to force a clean re-download.

## Data sources (all keyless)

| Source | Used for | Coverage |
|---|---|---|
| [CoinMetrics community](https://docs.coinmetrics.io/) | BTC + 13 altcoins, on-chain metrics | 2010-07 → |
| [FRED](https://fred.stlouisfed.org/) | rates, real rates, dollar, credit, liquidity, activity | 1919 → |
| [Kenneth French library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | FF5 + momentum; 49 industry portfolios | 1926 → |
| [LBMA](https://www.lbma.org.uk/prices-and-data/precious-metal-prices) | gold, silver, platinum, palladium fixes | 1968 → |
| SSGA / SPDR | ETF NAV history (SPY, GLD, XLK, …) | 2003 → |
| FRED / OECD | national share-price indices, 12 countries | 1957 → |
| [Bitstamp](https://www.bitstamp.net/api/) | BTC cross-check, hourly OHLC | 2011-09 → |
| Binance USD-M futures | BTC perpetual funding rate | 2019-09 → |
| [Jordà-Schularick-Taylor](https://www.macrohistory.net/database/) | long-run cross-country asset returns | 1870-2020 |

Two licensing traps are worth knowing if you extend this: on FRED's public CSV
endpoint every ICE BofA credit series is truncated to a rolling three years, and
`SP500`/`DJIA` to ten. Moody's `BAA10Y` and the Fama-French market factor are
used instead for the long history. Stooq, which would otherwise be convenient,
now sits behind a proof-of-work bot wall and is not used.

## Reproducibility notes

- The Fama-French files lag roughly two months behind the present (CRSP lag), so
  factor-based results end earlier than the Bitcoin sample.
- Re-running on a later date extends every series; regime dates and *p*-values
  will shift slightly, and the July 2025 break in particular is estimated on a
  short sample and should be expected to move.
- `analogues.py` and `validate_analogues.py` each take roughly ten minutes; they
  evaluate ~40,000 rolling windows.
- Figures use a brand-neutral palette in `figures.py` and an Economist-style one
  in the two `fig_essay*.py` scripts.

## Outputs

Only the three figures the blog post embeds are committed, under
`outputs/figures/`. Every other table and figure is regenerable and appears in
`outputs/` after a run; `.gitignore` keeps them untracked so the repository does
not accumulate derived files.

[`LITERATURE.md`](LITERATURE.md) is the annotated bibliography of verified
references behind the write-up. `DATA_DICTIONARY.md` documents every panel column
and is generated by `build_panel.py` on first run.

## A correction worth knowing about

An earlier version of this analysis standardised returns by computing the ratio
`r / sigma` and then back-filling the 26-week burn-in. That fills the start of the
sample with a single repeated value, and a constant block is indistinguishable
from a regime: the break test reported a highly significant "break" in July 2013
that was pure artefact, and in consequence mis-dated the real one. Dropping the
burn-in (or back-filling the volatility rather than the ratio) gives August 2019,
and the two choices agree. See the docstring of `ewma_vol` in
`breaks_correlation.py`.
