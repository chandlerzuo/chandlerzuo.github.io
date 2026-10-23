---
layout: post
title: Buy or Rent? What a Home Earned in America's Largest Metros, 2010–2026, Against the S&P 500
author: <a href="http://chandlerzuo.github.io/">Chandler</a>
topic: finance
---

In August 2010 you had a choice. The worst of the housing crash was behind you, though
prices had not quite hit bottom. You could buy a typical single-family house in one of
America's big metros and live in it. Or you could rent the same kind of house and put
the money you would have spent on buying into an S&P 500 index fund, or simply roll it in
Treasury bills. It is now August 2026. Sixteen years have passed, and every number that
decided the outcome is on the public record: the price you paid, every month's rent, the
price you sold at. So instead of forecasting what buying a home *might* earn, this post
measures what it *did* earn, after tax — over the whole sixteen years, and over four
shorter windows in which the housing market behaved very differently.

I ran the experiment in the twelve metros most buy-versus-rent debates are about: the
ten most populous (New York, Los Angeles, Chicago, Dallas, Houston, Washington,
Philadelphia, Miami, Atlanta, Boston) plus Silicon Valley (the San Jose metro) and Seattle.
The answer is
one number per metro, the internal rate of return on buying, set against the S&P 500
and T-bills over exactly the same window. All of it is after tax: 40% on ordinary income
and 25% on long-term capital gains.

**What a home earns.** A house you live in pays no rental income. What it pays you is
the rent you no longer owe. Each month an owner skips the rent a tenant would pay for
the same house, and in exchange pays property tax, insurance and upkeep that a tenant
does not. The difference is the owner's monthly return. At the end the owner also gets
the house back, at whatever it is then worth.

**The experiment.** At the end of August 2010 you buy the metro's typical
single-family home (Zillow's home-value index for the middle third of homes) and pay
3% in closing costs. At the end of August 2026 you sell at that month's index value and
pay 6% to sell. The internal rate of return is the rate $$r$$ that makes the monthly
cash flows net to zero,

$$\sum_{t=0}^{T} \frac{C_t}{(1 + r)^{t/12}} = 0,$$

where every cash flow is the net effect of (a) buying the house and (b) no longer
renting an equivalent one:

$$C_0 = -(\text{down payment} + \text{closing costs}),$$

$$C_t = R_t - (M_t + \text{tax}_t + \text{insurance}_t + \text{maintenance}_t) + S_t, \quad 1 \le t < T,$$

$$C_T = C_t + P_T(1 - s) - B_T - G_T.$$

$$R_t$$ is the rent saved that month. It is a cash flow, and it rises over the years
with the metro's actual rent history: from 3.3% a year in Houston to 5.3% in Seattle
over 2010–2026. $$M_t$$ is the mortgage payment (zero for a cash purchase), $$S_t$$
the income tax the owner saves through deductions, $$P_T$$ the sale price, $$s$$ the
6% selling cost, $$B_T$$ the loan balance repaid at the sale and $$G_T$$ the tax on
any gain above the primary-residence exclusion. This is the buy-versus-rent comparison
stated as a rate. Picture a renter who invests the down payment, and each month invests
whatever owning would have cost beyond the rent (or draws out whatever owning would
have saved). If the IRR on buying beats the after-tax rate that renter earns, buying
came out ahead.

**Taxes, applied evenly.** Every investment faces the same two rates: 40% on ordinary
income, and 25% on long-term capital gains. The 25% is the 15% federal long-term rate
for a household in the 32–35% brackets, plus the 3.8% net investment income tax, plus
about 6% state tax.
- *S&P 500.* Nearly all S&P 500 dividends are "qualified", so they are taxed at the 25%
  long-term rate as received and reinvested, and the remaining gain is taxed at the same
  rate when the position is sold at the end.
- *T-bills.* Interest is taxed at 40% as earned.
- *The home.* The rent an owner saves is not taxable income. On sale, a married couple
  excludes $500,000 of gain, and anything above that is taxed at the 25% long-term rate.
- *Deductions before 2018.* A household in the 40% bracket itemizes anyway, because its
  state income tax alone exceeds the standard deduction. So mortgage interest (on up to
  $1 million of debt) and property tax are fully deductible.
- *Deductions from 2018.* State and local taxes are capped at $10,000, which state
  income tax already uses up, and the standard deduction roughly doubled. Mortgage
  interest saves tax only to the extent it lifts itemized deductions above the standard
  deduction, on up to $750,000 of debt.

From August 2010 the S&P 500 returned 15.4% a year before tax and **13.2% after**:
$100,000 grows to about $727,000 instead of $983,000. T-bills fall from 1.5% to 0.9%,
below the 2.7% rate of inflation. The home is taxed lightly: only San Jose (about
$128,000) and Los Angeles (about $3,700) owe any tax on the sale.

**The sixteen-year result.**

<img src="https://raw.githubusercontent.com/chandlerzuo/chandlerzuo.github.io/master/codes/real_estate_analysis/outputs/fig2_return_sources.png" width="600" style="display:block; margin:1.5em auto;" />

Bought with cash, one home beat the after-tax S&P 500: Miami (13.6%). Dallas (12.1%),
Houston (11.9%) and Atlanta (11.5%) came close. The median metro earned 9.3%, and Silicon Valley (San Jose) 9.5%. The bottom of the table is
Chicago (8.9%), Washington (8.2%), Philadelphia (8.0%) and New York (6.9%). Every home comfortably beat T-bills and inflation. Each
bar splits the return into price appreciation (after buying and selling costs and any
gain tax), the rent not paid net of the costs of owning, and the part that came from a
property-tax assessment cap, discussed below.

**With a mortgage.**

<img src="https://raw.githubusercontent.com/chandlerzuo/chandlerzuo.github.io/master/codes/real_estate_analysis/outputs/fig4_mortgage_irr.png" width="600" style="display:block; margin:1.5em auto;" />

The August 2010 mortgage rate was only 4.4%, below what nearly every home earned, so
borrowing raised the return everywhere. With 30% down, eleven of twelve metros beat the
index. Miami (24.6%), Houston (23.5%), Dallas (22.5%) and Atlanta (20.0%) led, San Jose earned
15.0%, and Philadelphia (13.3%) was the lowest of the eleven. Only New York (10.5%) fell short. The
median levered return was 15.2%.

The chart splits each levered return into four parts, added in order so they sum
exactly to the IRR:
- *Price rise.* Buying with 30% down and selling at the 2026 price, after selling costs,
  the loan balance and gain tax, ignoring all monthly flows. This is worth 8.6 (Philadelphia)
  to 12.9 points (Miami) a year, and is the largest part everywhere except Houston.
  San Jose's prices grew fastest of all twelve, 7.3% a year, worth 12.6 points. On its own
  the price rise would roughly match the index in the best city and trail it everywhere else.
- *Rent saved.* The monthly rent saved minus the mortgage payment, tax, insurance and
  upkeep. It is what separates the cities: +11.3 points in Houston and +9.0 in Dallas and
  Miami, but about zero in San Jose and slightly negative in New York (−0.3), where the
  rent saved did not quite cover carrying the loan. The winners are the metros where rent
  was highest relative to price at the start. A Miami house cost about $195,000 in 2010
  when an equivalent rental went for $1,530 a month, so the rent avoided came to 9.4% of
  the price. San Jose's $544,000 house would have rented for $2,540, 5.6% of the price,
  barely enough to carry a 4.4% loan. Except in New York, the mortgage, tax, insurance and
  upkeep cost less than the rent in every single month. Even in New York owning was
  cheaper in the first year, and it cost more than renting in only 39 of 191 months.
- *Interest deduction.* Worth 1.3 to 2.5 points. Without it, eight metros would still
  beat the index; Washington, Philadelphia and Boston would join New York below it.
- *Assessment cap.* Worth up to 1.1 points (San Jose), about twice the cash case,
  because the cap savings accrue to a smaller equity stake.

Together, the price rise and the rent saved make up 84% to 91% of every levered return.

**What property-tax caps are worth.** California's Prop 13 sets a home's taxable value
at its purchase price and lets it rise at most 2% a year, however fast the market climbs.
Three other states in the sample cap a homestead's assessment more loosely:
- *Florida* ("Save Our Homes"): the smaller of 3% and inflation each year.
- *Texas*: 10% a year.
- *New York City* (one- to three-family homes): 6% a year and 20% over five years.

The first two charts show each cap's contribution. To measure it, I re-ran the same
purchase with the property tax charged on full market value at the same rate, as if the
home were reassessed every year. In these states the starting rate is the one recent
buyers pay, because the all-owner average already includes long-time owners' cap
savings. The difference between the two runs is the return the cap delivered:

| Metro | Cap | All-cash | With mortgage |
|---|---|---|---|
| San Jose | Prop 13 | +0.6 pts | +1.1 pts |
| Los Angeles | Prop 13 | +0.3 pts | +0.6 pts |
| Miami | Save Our Homes | +0.3 pts | +0.6 pts |
| New York | NYC class 1 | +0.1 pts | +0.2 pts |
| Dallas, Houston | 10% homestead cap | under 0.1 pts | about 0.1 pts |

The caps were worth most in the 2012–2020 recovery, when prices outran them (San Jose
+0.4 points, Los Angeles and Miami +0.2 for a cash buyer in that window), and nothing in
a falling or flat market. They are real money, but a supporting actor, not the lead.

**Four windows, four different markets.** A sixteen-year average hides how unevenly it
was earned. The clearest way to see it is to follow one buyer who took out a mortgage in
August 2010, and ask for every later month: how much would that buyer's money have grown
by selling then, after the 6% selling cost, the loan payoff and any gain tax? The chart
shows that cumulative return, $$(1+\text{IRR})^{\text{years}} - 1$$, next to the S&P 500
and T-bills bought the same month.

<img src="https://raw.githubusercontent.com/chandlerzuo/chandlerzuo.github.io/master/codes/real_estate_analysis/outputs/fig5_running_irr.png" width="600" style="display:block; margin:1.5em auto;" />

A home buyer starts in a hole. Selling a month after buying would have cost about 28% of
the money put in, because the 9% round-trip cost lands on a 30% stake. Prices then kept
sliding, so the median buyer was down 38% at the December 2011 low, while the S&P 500
was up 17%. Then the recovery did its work. The median home was back to even in
April 2013 and passed the index in December 2014, staying ahead ever since. By the end of
the pandemic boom in June 2022 it had grown the buyer's money by 652%, against 258% for
the index. In the high-rate years the index closed part of that gap: by August 2026 the
median home stood at +862% and the S&P 500 at +627%. Dallas's buyer, the best of the
twelve, was up 2,475%. New York's did not break even until September 2014 and finished at
+393%, the only city behind the index.

**A recent buyer: August 2023.** The 2010 buyer bought near the bottom with cheap money.
The opposite case is a buyer in the high-rate era. Take someone who bought in August 2023
with 30% down at 7.1% and could sell in any month since:

<img src="https://raw.githubusercontent.com/chandlerzuo/chandlerzuo.github.io/master/codes/real_estate_analysis/outputs/fig6_running_irr_2023.png" width="600" style="display:block; margin:1.5em auto;" />

Three years on, that buyer is still in the hole. Selling a month after buying would have
cost about 26%, the 9% round-trip cost on a 30% stake. With prices flat and the mortgage
costing more than the rent saved in many cities, the median buyer has only climbed back
to −10% by August 2026. Over the same three years the S&P 500 returned +58% after tax and
T-bills +8%.

Only four cities are above water: Chicago (+24%), New York (+11%), Philadelphia (+7%)
and Boston (+2%). Only Chicago and New York beat T-bills, and none comes near the index.
The former stars are the worst. In Dallas (−42%), Atlanta (−29%) and Houston (−27%) prices
have fallen since 2023. In Seattle (−21%) and San Jose (−18%) they rose (3% and 11%), but too
little to cover a 7% mortgage and the 9% round trip. This is the high-rate window seen through
a buyer's eyes, with the transaction cost added back. Three years is too short to spread
a 9% round trip, and a 7% loan earns more than these homes did. The comparison is also
unfinished: a 2023 buyer who stays will be judged by what comes next, not by a sale
today.

To isolate each regime, I split the period at the turning points of home prices and
mortgage rates. For each window I modeled a fresh all-cash purchase at the window's
start, sold at the window's end and taxed the same way. I left out buying and selling
costs, which would swamp windows this short.

<img src="https://raw.githubusercontent.com/chandlerzuo/chandlerzuo.github.io/master/codes/real_estate_analysis/outputs/fig3_windows.png" width="600" style="display:block; margin:1.5em auto;" />

*Bottoming out: August 2010 to March 2012.* Prices kept falling in every metro, about 4.8%
a year in the median, to a common bottom in late 2011 and early 2012. The median cash
purchase roughly broke even, and six of twelve lost money; Seattle (−5.7%) did
worst. Houston (5.4%) and Dallas (5.2%) earned the most. Meanwhile the S&P 500 returned
17.4% after tax, so no home came close.

*The recovery: March 2012 to February 2020.* Buying at the bottom, with mortgages near
4%, was the best trade of the period. Prices rose about 7% a year. The median cash
purchase returned 13.6% against the index's after-tax 9.7%, and nine of twelve metros
beat it; New York (7.9%), Philadelphia (9.0%) and Washington (9.6%) fell short. Miami (17.4%), Atlanta
(16.3%) and Dallas (15.9%) led. With 30% down, the median buyer earned 27% a year.

*The pandemic boom: February 2020 to June 2022.* Mortgage rates fell to a record 2.7%
and prices rose 14% a year in the median metro. Every home beat the S&P 500's after-tax
9.9%. The median cash purchase returned 17.6%, with Atlanta at 25.5% and Dallas and
Miami near 24%; even New York earned 13.2%.

*High rates: June 2022 to August 2026.* Mortgage rates jumped to 5.5% at the window's
start and stayed at 6.5–7%. Home prices stalled at about 1.4% a year, falling in four
metros, while the S&P 500 returned 16.0% after tax and T-bills 2.6%. No home came close
to the index. The median cash purchase returned 5.4%, and San Jose (1.4%) and
Seattle (2.0%) fell below even T-bills. The leaders reversed: Chicago (9.6%), Philadelphia
(8.8%), Miami (7.9%) and New York (7.6%) led, while San Jose, Seattle and Dallas
(3.0%) trailed. In five metros borrowing lowered the return.

Read together, the windows say what the sixteen-year number cannot. A home's return
relative to stocks depended far more on *when* you bought than on *where*. Bought near
the 2012 bottom with cheap money, a house beat the S&P 500 nearly everywhere. Bought into
the last leg of the decline or into the rate shock, it lagged everywhere. The metros that
led in one window were not reliably the ones that led in the next.

**Is the comparison fair?** These are the places it can mislead:

- *Starting point.* Starting in 2010 leaves out the 2008–2010 crash, so these results
  are kinder to buying than a window that includes it. A buyer in August 2008 earned
  about two points less with cash, and the levered losses of the crash erased many
  down payments.
- *Dividend tax.* S&P 500 dividends are treated as qualified and taxed at the 25%
  long-term rate. Taxing them instead at the 40% ordinary rate would cut the index's
  after-tax return by 0.3 points, to 12.9%, and move no metro across it over the full
  sixteen years.
- *Upkeep.* The figures reserve 0.5% of the home's value a year for repairs. Doubling it
  to 1% lowers the median all-cash return from 9.3% to 8.8%. The number of metros where
  a mortgage beats the index falls from eleven to nine.
- *Risk.* The S&P 500 holds five hundred companies; a home is one house in one
  neighborhood, and actual houses scatter widely around the index-level home used here.
  Buying with 30% down puts more than three times your cash into that one asset, which
  is why the levered buyer's early years were so deeply negative.
- *Refinancing.* A 2010 buyer who refinanced in December 2012 (3.35%) would have gained
  only a little more, a median of 15.6%, because the 2010 rate was already low.
- *Hindsight in the windows.* The four windows start and end at turning points that
  were only clear afterwards, so they show the contrast between regimes at its sharpest.
  No buyer could have timed them, or avoided the 9% round-trip cost the windows leave
  out.

**The moral.** After tax, from 2010 to 2026, owning your home in America's largest metros
paid a strong real return: 6.9% to 13.6% a year bought with cash, far above T-bills and
inflation. Bought with cash, it beat renting and investing in the S&P 500 only in Miami.
Bought with a 4.4% mortgage, whose interest the tax code subsidized, it beat the index in
eleven of twelve. New York, the most expensive market relative to rent, was the only
exception. Most of a home's return is the rent you stop paying; appreciation matters most
in the priciest metros, and tax-base caps like Prop 13 play supporting roles. Inside those
sixteen years, the answer to "buy or rent?" flipped more than once. Whether a home beats
the stock market depends on what you paid for that rent, and on the price and the mortgage
rate in the year you bought.

---

**TECHNICAL DETAILS**

*Sample.* The ten largest metropolitan statistical areas by Zillow `SizeRank`
(population) plus San Jose (rank 36; the San Jose-Sunnyvale-Santa Clara MSA, i.e. Santa
Clara and San Benito counties, which is Silicon Valley proper) and Seattle (rank 15). San
Mateo County (Menlo Park, Redwood City) and Fremont belong to other metros and are not
included. One representative
single-family home per metro, owner-occupied. Overall holding period: end of August
2010 to end of August 2026, 192 months, chosen to leave out the 2008–2010 collapse.

*Windows.* Bottoming out: Aug 2010 – Mar 2012, the trough of the 12-metro average home
price (individual metros bottomed between November 2011 and December 2012). Recovery:
Mar 2012 – Feb 2020. Pandemic boom: Feb 2020 – Jun 2022, ending as mortgage rates passed
5.5% and price growth turned down. High rates: Jun 2022 – Aug 2026. Each window is a
fresh all-cash purchase at its first month (levered results in the data files use that
month's PMMS rate: 4.43%, 3.95%, 3.46%, 5.52%), taxed as below, with no buy or sell costs.
The overall result and the running return since 2010 include both.

*Price.* Zillow Home Value Index (ZHVI), single-family, middle tier (33rd–67th
percentile), smoothed and seasonally adjusted, monthly.

*Rent avoided.* The market rent of an equivalent single-family house: Zillow Observed
Rent Index (ZORI), single-family rentals, monthly, from January 2015. Before that the
series is chained back with the BLS CPI "rent of primary residence" index for the same
metro (not seasonally adjusted):
$$R_t = \text{ZORI}_{2015\text{-}01} \times \text{CPI}_t / \text{CPI}_{2015\text{-}01}$$.
The CPI index tracks rents on all leases, not only new ones, and so moves more slowly
than market rent. BLS publishes no rent index for San Jose, so the San Francisco-Oakland-
Hayward index stands in for it before 2015. BLS did not publish October 2025, and Zillow
has no September 2022 price for San Jose; both are interpolated. No vacancy or
management cost applies: the owner lives in the house.

*Ownership costs.* Maintenance reserve 0.5% and homeowner's insurance 0.35% of current
value a year. Sensitivity: maintenance 1%.

*Property tax.*
- *Uncapped metros.* Atlanta, Boston, Chicago, Philadelphia, Seattle and Washington use
  the Census ACS 2024 effective rate (median real-estate taxes paid ÷ median
  owner-occupied value, metro), applied to each month's value.
- *Capped metros.* The rate is applied to an assessed value reset every January to
  min(market value, last year's assessment grown by the cap).
  - CA: 2%; rate 1.15%.
  - FL: min(3%, prior-year December CPI-U change).
  - TX: 10%.
  - NY: 6% a year and 20% over five years.
  - The rate for FL, TX and NY is the median tax/value among ACS 2024 PUMS owners who
    moved in one to four years earlier: Miami 0.94%, Dallas 1.55%, Houston 1.55%, New
    York 1.74%. The all-owner rate already includes long-time owners' cap savings.
  - The NYC class-1 rule also covers Nassau County, but not New Jersey or the other
    suburbs, so it overstates the cap for the metro as a whole.
  - Homestead exemptions are ignored.
- *Cap attribution.* The same purchase taxed on full market value at the same rate.

*Income tax.* 40% on ordinary income; 25% on long-term capital gains (15% federal +
3.8% NIIT + ~6% state).
- *Owner's deductions.* Before 2018, mortgage interest (acquisition debt up to $1M) and
  property tax are fully deductible. From 2018, the saving is
  0.4 × [max(standard deduction, $10,000 + interest) − standard deduction] per calendar
  year, on debt up to $750,000. The standard deduction for married filing jointly runs
  from $24,000 in 2018 to $32,200 in 2026.
- *Sale.* Gain is the amount realized (net of selling cost) minus the price plus
  closing costs. Gain above the $500,000 exclusion is taxed at 25%.
- *Interest attribution.* The same levered purchase with no mortgage-interest deduction
  (`irr_lev_noint`); the difference is `int_saving_lev`.
- *S&P 500.* Monthly dividend = total-return index growth minus price-index growth. It
  is taxed at 25% as a qualified dividend and reinvested, and the gain above basis is
  taxed at 25% at the end.
- *T-bills.* Monthly interest taxed at 40%.
- *Pre-tax results.* Kept in the data files as `*_pretax`.

*Transactions.* 3% closing costs at purchase; 6% selling cost (overall result and
running returns).

*Financing.* 30% down, 30-year fixed at the Freddie Mac PMMS 30-year fixed rate averaged
over the purchase month (4.43% for August 2010). The balance remaining at sale is repaid
from the proceeds. Refinance sensitivity (overall only): the balance is refinanced in
December 2012 into a new 30-year loan at that month's PMMS rate (3.35%), with 1% costs
paid in cash.

*IRR.* Solved on the monthly cash flows and annualized, $$(1+r_m)^{12} - 1$$; checked
against an independent solver. Price-only return:
$$\big[(P_T(1-0.06) - G_T) / (P_0(1+0.03))\big]^{1/16} - 1$$. In the first chart, the rent
contribution is the all-cash IRR minus this and minus the cap contribution. The mortgage chart is an
exact waterfall of the levered IRR, with columns `wf_*_lev`:
- *Price rise.* The IRR of the down payment and closing costs against the sale proceeds
  net of selling costs, loan payoff and gain tax.
- *Rent saved.* Plus all monthly flows, with property tax on full market value and no
  interest deduction.
- *Interest deduction.* Plus the mortgage-interest deduction.
- *Assessment cap.* Plus the cap, which gives the actual IRR.

*Benchmarks.* S&P 500 total-return (^SP500TR) and price (^GSPC) indexes, month-end
close (total return 1,752.55 on 31 Aug 2010 and 17,219.94 on 31 Aug 2026). One-month
Treasury bill returns from the Ken French data library (RF). Inflation: CPI-U all
items, U.S. city average.

*Limits.* One index-level home per metro, so the results describe the typical house,
not any actual one. No capital-expenditure shocks beyond the maintenance reserve. The
ACS tax rates are 2024 rates held fixed throughout, and the rent of the typical rental
house is assumed equal to the rent the typical owned house would fetch.

*Reproducibility.* Full source code is available at
[github.com/chandlerzuo/chandlerzuo.github.io/tree/master/codes/real_estate_analysis](https://github.com/chandlerzuo/chandlerzuo.github.io/tree/master/codes/real_estate_analysis);
`run_all.sh` downloads every public input (`download_data.py`) and rebuilds all the
numbers and figures in this post. `metro_pipeline.py` selects the metros and loads the
Zillow and Census inputs. `history_pipeline.py` downloads the rent, mortgage, S&P 500
and T-bill series and simulates every metro over the full period and each window,
before and after tax and with and without assessment caps. It writes
`data/history_units.csv`, `data/history_windows.csv`, `data/benchmarks.csv` and the
monthly flows in `data/history_cashflows.csv`. It also writes the running IRR
(`data/history_running.csv`): a mortgage purchase in August 2010 sold in each later
month, with 3% closing and 6% selling costs, loan payoff and gain tax, after tax. The chart of the
2010 buyer plots its cumulative return, $$(1+\text{IRR})^{\text{years}} - 1$$, on a log scale.
`data/history_running_2023.csv` is the same for a buyer in August 2023 (7.07%), drawn in
the chart that follows it. `make_history_charts.py` builds the six figures and `outputs/history_summary.csv`.

*(c)2017-2026 CHANDLER ZUO ALL RIGHTS PRESERVED*
