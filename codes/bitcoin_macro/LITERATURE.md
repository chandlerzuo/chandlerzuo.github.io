# Literature: Bitcoin's Macro Betas, Regimes, and Analogues

Curated synthesis. The full verified scan — ~60 references with sample periods,
exact claims, DOIs, and an explicit list of things that could **not** be
verified — is in [`LITERATURE_FULL.md`](LITERATURE_FULL.md).

All citations below were retrieved live from OpenAlex, Crossref, arXiv, NBER,
BIS, or the publisher. Items that could not be verified are listed as such in
the full file and are **not** relied on here.

---

## 1. The baseline finding: early Bitcoin had no macro betas

**Liu & Tsyvinski**, "Risks and Returns of Cryptocurrency," *RFS* 34(6), 2021
(NBER w24877) — sample BTC 2011-01-01 to 2018-05-31. They test FF3, Carhart-4,
FF5, FF6, FF30 industries, **155 additional anomaly series**, five currencies,
three precious metals, four real-activity macro factors (non-durable and durable
consumption growth, IP growth, personal income growth), corporate bonds, and
Case-Shiller housing. Verbatim: *"Cryptocurrencies have no exposure to most
common stock market and macroeconomic factors. They also have no exposure to the
returns of currencies and commodities."* CAPM betas are sizable but alphas
remain large and significant. What *does* price crypto: network/adoption
factors, time-series momentum, investor attention.

**Liu, Tsyvinski & Wu**, "Common Risk Factors in Cryptocurrency," *JF* 77(2),
2022 (NBER w25882) — 1,707 coins, 2014–2018 weekly. A crypto-native three-factor
model (market, size, momentum) captures the cross-section, and stock-market
factor models *"do not account for the cross-section of cryptocurrency returns."*

**Baur, Hong & Lee** (*JIFMIM* 2018) and **Corbet et al.** (*Economics Letters*
2018) independently establish the same pre-2018 "isolated asset" baseline.

> **Implication for this study.** These are the null hypothesis, not a rival
> claim. Our Phase 2 early-subsample regression reproduces them almost exactly
> (2013–2019: market beta 0.20, *t* = 0.6, R² = 0.001), which validates the data
> pipeline before any time-varying machinery runs.

## 2. The claim that Bitcoin became macro-sensitive after 2020

- **Iyer (IMF GFSN 2022/01)**: spillovers from Bitcoin *volatility* to the S&P 500
  and MSCI EM rose **12–16pp since the onset of COVID**, and from returns
  **8–10pp**; Bitcoin explains 14–18% of equity volatility variation.
- **Iyer & Popescu (IMF WP 2023/213)**: spillovers rise in turbulence and peak
  during COVID; increased correlation in risk-off episodes implies crypto behaves
  as a risk asset, not a diversifier.
- **Wątorek, Kwapień & Drożdż** (*Entropy* 25(2):377, 2023): 10-second data,
  Jan 2020–Oct 2022, *q*-dependent detrended cross-correlation. The cleanest
  published statement of the structural change — explicit *"bitcoin and ethereum
  coupling to the US tech stocks"* in the 2022 bear phase, plus newly emergent
  reaction to CPI releases. Their own 2019 paper (*Future Internet* 11(7):154)
  documents the *pre*-2020 decoupling, making a clean before/after pair.
- **Wu** (arXiv:2501.09911, 2025): BTC–Nasdaq-100 rolling correlation **peaking
  at 0.87 in 2024**; regime-dependent.

## 3. The live contradiction this study is positioned to resolve

**BlackRock**, "Bitcoin: A Unique Diversifier" (Sept 2025) reports a 10-year
**weekly** BTC–S&P 500 correlation of **0.2** (gold 0.04), June 2015–June 2025,
and argues Bitcoin *"reflects little fundamental exposure to other macro
variables,"* with correlation spikes that are *"short-term in nature"* and
concentrated around *"episodes of sudden shifts in U.S. dollar real interest
rates or liquidity."* They explicitly reject a risk-on/risk-off framing.

That is the opposite of the IMF and Wątorek conclusions. **Both are defensible,
because they are measured at different frequencies over different windows.** An
unconditional 10-year correlation averages over a structural break; a rolling
2024 daily window does not.

This is the central well-posed question, and it is directly answerable:
*is Bitcoin's macro exposure a stable low number with transient spikes, or a
sequence of persistent regimes?* Our Phase 2 split already indicates the latter
(market beta 0.20 → 0.94; R² 0.001 → 0.092; weekly alpha 1.66%/wk significant →
0.33%/wk insignificant across 2019/2020), so the regime framing is not assumed —
it is the empirically indicated structure.

## 4. Monetary policy: the result that breaks the "levered Nasdaq" story

**Karau**, "Monetary policy and Bitcoin," *JIMF* 137:102880, 2023 —
high-frequency identification plus a weekly proxy VAR. Verbatim: *"a
disinflationary monetary tightening by the **ECB lowers valuations** — consistent
with the notion of Bitcoin as a digital gold —, whereas a **Fed tightening
increases Bitcoin prices**."*

This sign asymmetry is flatly inconsistent with a simple "crypto = high-beta
Nasdaq" model, and it has not been re-tested on post-2021 data. Practical
consequence adopted here: **never pool central banks**, and treat the sign of the
policy beta as an object of study rather than a nuisance parameter.

Supporting: **Corbet et al.** (*JFS* 2020) FOMC event study with heterogeneity by
blockchain-stack position; **Elsayed & Sousa** (*EJF* 2022) TVP-VAR on shadow
policy rates, 2013–2019, finding the largest spillovers when shadow rates were
negative; **Nguyen et al.** (*RIBAF* 2019) asymmetric response to contractionary
vs expansionary policy.

## 5. Digital gold, inflation hedging, safe haven — mostly negative results

**Baur & Lucey** (*Financial Review* 2010) is the methodological source: *hedge* =
uncorrelated on average; *safe haven* = uncorrelated **in a crash**. Every crypto
paper below inherits that design.

The weight of evidence rejects the strong digital-gold claim:
- **Baur, Dimpfl & Kuck** (*FRL* 2018) replicate and overturn **Dyhrberg**'s
  (2016) "bitcoin sits between gold and the dollar" result.
- **Bouri et al.** (*FRL* 2017): diversifier, **not** a reliable hedge or safe haven.
- **Klein, Pham Thu & Walther** (*IRFA* 2018): Bitcoin's conditional correlation
  with equities behaves **oppositely to gold's** in downturns.
- **Conlon & McGee** (*FRL* 2020) and **Conlon, Corbet & McGee** (*RIBAF* 2020):
  fails as a safe haven in the COVID bear market.
- Inflation: **Conlon, Corbet & McGee** (*Economics Letters* 2021) find the
  positive link is confined to a brief window around COVID onset;
  **Smales** (*Accounting & Finance* 2023) finds returns are **lower on CPI
  release days** and respond **negatively to CPI surprises**;
  **Liu & Valcarcel** (*JFS* 2024) give the first evidence of **time variation**,
  with the response turning negative between the Luna crash and FTX;
  **Choi & Shin** (*FRL* 2022): an inflation hedge but **not** a safe haven.
- **Baur & Dimpfl** (*Economics Letters* 2018): crypto volatility asymmetry is
  **inverted** relative to equities — positive shocks raise volatility more than
  negative ones. A necessary control in any GARCH/regime specification, and a
  reason not to import equity-calibrated vol models unmodified.

## 6. Regime methods used in this literature (and the gap)

Well-established for **volatility**: MSGARCH with Bayesian estimation
(**Ardia, Bluteau & Rüede**, *FRL* 2019); >1,000 GARCH specifications with Model
Confidence Set selection (**Caporale & Zekokh**, *RIBAF* 2019); MRS-MIDAS with
jump-driven time-varying transition probabilities (**Ma et al.**, *J. Forecasting*
2020); Bayesian change-point analysis (**Thies & Molnár**, *FRL* 2018);
LPPLS bubble-regime detection with Lagrange regularization (**Gerlach, Demos &
Sornette**, *R. Soc. Open Sci.* 2019; **Shu, Song & Zhu**, *Stats* 2021);
right-tailed recursive unit-root tests, SADF/GSADF (**Demmler & Fernández** 2022).

Closest existing templates for regime-switching *betas*: **Shaikh** (*Borsa
Istanbul Review* 2020) pairs Markov regime-switching with quantile regression on
EPU/MPU; **Bianchi, Guidolin & Pedio** (WP 2020) estimate time-varying exposures
where the model specification itself changes over time.

> **The gap, stated by the scan:** there is little published work applying
> Markov-switching or change-point methods **directly to the Bitcoin–equity beta**
> rather than to Bitcoin volatility. *"A formal 2-state Markov-switching beta
> model on BTC vs Nasdaq/SPX with macro state variables appears to be an open
> slot."* That is Phase 4 of this study.
>
> **Second gap:** *"Fed balance sheet / global liquidity as a factor is
> essentially untested in the refereed literature."* Given that liquidity is the
> mechanism BlackRock names, this is the clearest white space. That is why
> `net_liquidity` (Fed assets − TGA − reverse repo) is built into the panel.

## 7. Historical analogues — the thinnest theme, and therefore original

Only one broad published head-to-head exists: **Náñez Alonso et al.**,
*Humanities and Social Sciences Communications* 11, 2024 — 9,967 records,
comparing SDs, growth rates, and a bubble index; finds the 2011/2013/2017/2021
Bitcoin bubbles resemble the **tulip (1634–37)** and **Mississippi (1719–20)**
bubbles. **Cheah & Fry** (*Economics Letters* 2015) is the foundational
"substantial bubble component" result. **Fan** (arXiv:2206.14130, 2022) is the
right methodological bridge: it applies GSADF/BSADF date-stamping to individual
1990s Nasdaq stocks, so the identical procedure can be run on Bitcoin and on a
historical analogue.

> **Explicitly not found in the indexed literature** (searched OpenAlex,
> Crossref, arXiv): no peer-reviewed paper comparing Bitcoin specifically to
> **silver in the Hunt-brothers era**, **gold post-1971**, **ARKK**, or an
> explicit **emerging-market-equity** risk/return analogue.
>
> So Phase 6 is an original contribution rather than a replication — which raises,
> not lowers, the burden of proof. It is precisely because nobody has done this
> that the bootstrap null, the trial-count disclosure, and the out-of-sample test
> in Phase 7 are non-negotiable. An unconstrained search over thousands of
> asset-windows will always find a "striking" analogue.

## 8. Design decisions this literature forces

1. **Frequency must be reported with every beta.** Liu & Tsyvinski get Sharpe
   0.09 daily, 0.23 weekly, and ~equity-comparable monthly *from the same data*.
   BlackRock's 0.2 is weekly/10-year; Wu's 0.87 is rolling daily in 2024. A bare
   "Bitcoin's beta is X" is meaningless without (frequency, window, window type).
   → We report weekly primary, daily Dimson-corrected, and state the window.
2. **Scale dependence is real.** **Bouri et al.** (*QREF* 2020) show safe-haven
   properties appear at some wavelet scales and not others; Wątorek et al. show
   dependence on both time scale and fluctuation magnitude. → Treat the
   frequency choice as a result, not a setup detail.
3. **Crypto-native controls are mandatory.** Without CMKT/CSIZE/CMOM, network and
   adoption factors, attention, and carry/convenience yield, macro betas are
   overstated because FTX/Luna/Mt. Gox shocks load onto whatever macro variable
   happens to co-move. → Layer C of our factor taxonomy.
4. **Gold is the conventional control asset** and is what makes "digital gold"
   falsifiable (BlackRock's 0.2 vs 0.04 is exactly this comparison).
   → LBMA gold and silver, daily from 1968, are in the panel.
5. **Document and justify the sample start.** Liu & Tsyvinski start at 2011
   citing illiquidity; Liu-Tsyvinski-Wu start at 2014 because volume data begins
   late 2013. → We start the main sample in 2013 and justify it *empirically*:
   median disagreement between independent price feeds is 7.0% in 2011 and 1.1%
   in 2012, but ≤0.3% from 2013 onward.
6. **Report the full distribution, not just Sharpe.** Bitcoin's positive skewness
   *rises* with horizon (0.80 daily → 1.76 weekly), the opposite of equities.
   That asymmetry is the quantitative heart of the venture-style-payoff claim and
   is invisible in a Sharpe ratio.
7. **Choose models out-of-sample.** Caporale-Zekokh and Maciel select via VaR/ES
   backtesting and Model Confidence Sets, not in-sample fit.

## 9. Citation hygiene

Two heavily cited papers are flagged **RETRACTED** in OpenAlex and are not relied
on: Corbet, Lucey, Urquhart & Yarovaya (*IRFA* 62, 2018) and Corbet, Lucey &
Yarovaya (*FRL* 26, 2018).

The scan also could not verify several things that are often asserted, and they
are therefore absent from this study's framing: any Fed Board FEDS Note or IFDP
on Bitcoin macro betas (the 2022 and 2023 indexes were checked directly — the
verified Fed-system contribution is the **Cleveland Fed** paper by Divakaruni &
Zimmerman, 2021); any ECB working paper on crypto macro betas; and a
"Benigno & Rossi" crypto paper, which does not appear to exist (the real
Benigno-Rossi paper is on asymmetries in monetary policy and is unrelated).
