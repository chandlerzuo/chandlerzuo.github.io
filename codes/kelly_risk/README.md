# Kelly sizing, drawdowns and stopping rules

Simulation studies behind the blog post on how to size a positive-edge bet (a poker session,
a day trade) so that the risk of ruin and of deep drawdowns stays small. Every period one bet
wins +1 per unit staked with probability 0.53 and loses 1 otherwise (Kelly fraction 6%,
annualised Sharpe about 0.95 at 252 bets a year), for 1,260 bets (5 years) and 20,000 simulated
traders per scenario, with the same random outcomes across strategies.

* **A. known edge**: every trader knows p = 0.53.
* **B. uncertain edge**: traders size to p = 0.53 but the true p ~ N(0.52, 0.02²) per trader.

Strategies: fixed stake; 1% of equity; quarter, half, full and double Kelly; half Kelly with a
drawdown throttle (stake scaled by 1 − DD/30%); half Kelly with a permanent quit at a 25%
drawdown, or at −30% / +100% from the start; Bayesian half Kelly (Beta(200,200) prior on p),
with and without the quit rule. Theory checks: the probability that the account ever falls to a
fraction x of its starting value (exponent 2m/s²), and the two-barrier stopping formulas.

```
kelly_risk/
├── run_all.sh        # simulate -> figures
├── code/
│   ├── sim.py        # all simulations -> data/*.csv
│   └── make_charts.py  # fig1-5 -> outputs/
├── data/             # NOT committed; created by run_all.sh
└── outputs/          # fig*.png are committed (the blog post embeds them)
```
