# Kelly fractions, first passage and stopping times

Simulations behind the blog post on controlling a multiplicative random walk
`W_n = W_{n-1} (1 + f_n X_n)` with i.i.d. steps `X = ±1`, `P(X = +1) = p = 0.53`
(Kelly fraction `f* = 2p - 1 = 0.06`). Every study uses 20,000 paths of 1,260 steps
(5 units of 252) with the same random steps for every policy.

* **A. known drift**: every policy knows p = 0.53.
* **B. uncertain drift**: policies assume p = 0.53 but the true p ~ N(0.52, 0.02²) per path.

Policies: fixed bet size; fixed fraction 1%; quarter, half, full and double Kelly; half Kelly with a
depth throttle; half Kelly stopped at 25% depth below the running maximum or at an exit band
(0.7 W0 / 2 W0); Bayesian half Kelly (Beta(200,200) prior), with and without the depth stop.

Theory checks in `sim.py`: P(ever fall to x W0) = x^λ, the two-barrier exit probability and
expected exit time, the median maximum depth below the running maximum, the law of large numbers
for S_n / n, and the competitive bound P(sup W_n / W*_n ≥ K) ≤ 1/K.

```
kelly_risk/
├── run_all.sh          # simulate -> figures
├── code/
│   ├── sim.py          # all simulations -> data/*.csv
│   └── make_charts.py  # fig1-10 -> outputs/
├── data/               # NOT committed; created by run_all.sh
└── outputs/            # fig*.png are committed (the blog post embeds them)
```
