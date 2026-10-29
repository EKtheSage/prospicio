# Reserving methods: which one when

## Use this guide when

You have a loss triangle and must select a reserving method in prospicio. The
question is: which method answers the question that you must answer, with the
data that you have? Typical questions are:

- a best estimate for the balance sheet
- a standard error
- a full distribution, for capital or to price a cover on the reserve
- a one-year view, for a solvency standard

This guide connects each situation to a method. It shows the call and tells
you where the method fails.

Terms in this guide:

- An **accident year** (the **origin**) is the year in which the losses
  occurred.
- The **development age** is the number of months from the start of the origin
  year. Thus age 12 on accident year 2021 has the valuation date December 2021.
- A **triangle** holds one cumulative value for each origin and age. You observe
  only the upper-left part.
- The **reserve** is the **ultimate** (the final cost of the origin) minus the
  amount paid to date.
- A **link ratio** is the growth of an origin from one age to the next. An
  **age-to-age factor** is an average of the link ratios across origins.
- A **tail factor** gives the development after the oldest age.
- An **increment** is the amount paid in one cell (the cumulative value minus
  the value before it).

Abbreviations and names:

- **Bornhuetter-Ferguson (BF)**, **Benktander** and **Cape Cod** mix the data
  with an **a priori** (a prior). The prior is an expected loss ratio that you
  set before you look at the triangle.
- **Over-dispersed Poisson (ODP)**: the model in which the variance of each
  increment is a scale times its mean.
- **Coefficient of variation (CV)**: the standard error divided by the mean.
- **Claims development result (CDR)**: the opening ultimate minus the ultimate
  one year later. A positive CDR is favourable development, and a negative CDR
  is adverse development.
- **Adverse development cover (ADC)**: reinsurance that pays when the reserve
  is more than an attachment. **Loss portfolio transfer (LPT)**: reinsurance
  that takes the reserve itself.
- **NUTS** (the No-U-Turn sampler) samples a Bayesian posterior. **R-hat** is
  its convergence check (a value near 1 is good).
- **CRC** and **CSR**: the correlated chain ladder and changing settlement
  rate models of Meyers.
- **Spearman** correlation is a correlation of ranks. **Iman-Conover** is the
  reordering that puts a Spearman correlation on simulations.

## Start here

1. **Best estimate.** Run the chain ladder. For the youngest origins, which
   have small paid amounts, compare it with BF or Cape Cod (section 4). First,
   check the data (section 1).
2. **Uncertainty.** For a standard error, use Mack (section 3). For a quantile,
   a cover or capital, use a bootstrap (section 6).
3. **Horizon.** Run-off to the last payment is the lifetime view. The next
   year only, as a solvency standard asks, is the one-year view (section 7).

## The decision table

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| Mature book, stable pattern, you need a best estimate | Chain ladder | `ChainLadder()` | Uses only the triangle. Assumes that the pattern is stable |
| You also need a standard error, with no distribution assumed | Mack | `Mack()` | Chain ladder plus Mack's formula, split into process and parameter risk |
| Young origins whose few payments give little information | Bornhuetter-Ferguson | `BornhuetterFerguson(apriori)` | Reserves only the share of a prior that is still to develop |
| A plan loss ratio exists, but you want the data to have more weight | Benktander | `Benktander(apriori, n_iters=2)` | Repeats Bornhuetter-Ferguson and moves toward the chain ladder |
| No plan loss ratio that you trust, but you have premium | Cape Cod | `CapeCod(trend, decay)` | Estimates the loss ratio from the triangle |
| An origin with no paid amount yet | Expected loss | `ExpectedLoss(apriori)` | The prior alone, ignores the data |
| Few ages, you need a smooth pattern to compare with the chain ladder | Clark growth curve | `ClarkLdf(curve, max_age)`, `ClarkCapeCod` | One curve for all cells. Set `max_age` where development stops |
| Development continues after the oldest age | Tail factor | `tail=` on `ChainLadder`, `Mack`, the expected-loss methods and the two bootstraps | A constant, or a curve fitted to the factors |
| A distribution of the reserve, ODP assumptions | ODP bootstrap | `OdpBootstrap().fit` | Resamples residuals, draws gamma process error |
| A distribution that agrees with the standard error of Mack | Mack bootstrap | `MackBootstrap().fit` | Same variance structure as Mack, simulated |
| A posterior with a prior on the development or origin effects | Bayesian ODP | `BayesOdp().fit` | Posterior predictive reserve by NUTS |
| Premium, 10 or more years, positive cells, a prior on the loss ratio | Meyers CRC or CSR | `Meyers(model).fit` | Log-normal ultimates with a prior on the loss ratio |
| One-year risk of the chain ladder, no simulation | Merz-Wuthrich | `Mack().fit(...).claims_development_result()` | Closed form. Volume-weighted factors, no tail, one segment |
| One-year risk of chain ladder, BF, Benktander, Cape Cod or expected loss, with a tail or another grain | Simulated one-year view | `MackBootstrap().one_year`, `OdpBootstrap().one_year` | Re-reserves each simulation |
| Several lines, capital for the total | Dependence between lines | `dependence=` (and `spearman=`) on the two bootstraps | Keeps the joint behaviour of the lines |
| Check a method on your own history | Diagonal backtest | the loop in section 9 | Refit without the newest diagonal, forecast it |
| Reinsurance on the reserve (ADC, LPT) | Aggregate cover | `Tower.apply_aggregate` | Applies a layer to each simulated total |

## 1. The data, and the rules about ages and exposure

The example is a synthetic paid triangle. It has seven accident years, annual
ages 12 to 84 months, and an earned premium for each origin. Money is in
thousands of a synthetic currency.

```python
from prospicio.reserving import (
    BayesOdp, Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ClarkCapeCod,
    ClarkLdf, ExpectedLoss, Mack, MackBootstrap, Meyers, OdpBootstrap,
    TailBondy, TailConstant, TailCurve, TailLogLinear, Triangle,
)

paid = [  # cumulative paid by accident year, one row per year, ages 12, 24, ... months
    [203, 374, 508, 583, 644, 656, 667],
    [232, 432, 555, 659, 711, 754],
    [219, 407, 550, 664, 720],
    [246, 434, 601, 678],
    [267, 470, 632],
    [305, 539],
    [323],
]
premium = [1000, 1050, 1100, 1160, 1220, 1280, 1350]  # earned premium by accident year
years = list(range(2018, 2025))

origin = [y for y, row in zip(years, paid) for _ in row]
age = [12 * (k + 1) for row in paid for k in range(len(row))]
tri = Triangle.from_long(
    origin,
    age,
    {
        "paid": [v for row in paid for v in row],
        "premium": [premium[i] for i, row in enumerate(paid) for _ in row],
    },
)
print(tri.origins[0], "to", tri.origins[-1], "| ages", tri.development, "| valued", tri.valuation)
print(tri)
```

```text
2018 to 2024 | ages [12, 24, 36, 48, 60, 72, 84] | valued 2024-12-31
Triangle: 1 segment x 2 columns (cumulative, valuation 2024-12)
column   n_origins  first_origin  last_origin  valuation  latest
paid             7          2018         2024    2024-12   4,313
premium          7          2018         2024    2024-12   8,160
```

`Triangle.from_long` takes one row for each origin and age. Three rules apply:

- **Ages are months.** Annual data has ages 12, 24, 36. A quarterly triangle
  has ages 3, 6, 9 and needs `development_grain="Q"`. If your data gives the
  valuation date of each cell instead of an age, use
  `development_is_valuation=True`. The library then converts the dates.
- **Cumulative is the default.** Use `cumulative=False` for incremental
  payments. An unobserved cell is `nan`. An observed zero stays zero.
- **A segment is a key.** Use `keys={"line": [...]}` to hold several lines in
  one triangle. Each fit then runs on each segment separately.

For data in a pandas frame, `Triangle.from_frame(df, origin="year",
development="age", columns=["paid", "premium"], keys="line")` does the same.

**Exposure columns.** Bornhuetter-Ferguson, Benktander, Cape Cod, expected loss
and Clark's Cape Cod take a second column name, the exposure (for example
premium): `fit(tri, "paid", "premium")`. The library reads the **latest
observed cumulative value** of that column for each origin. For a cumulative
triangle, repeat the premium on each cell of the origin, as above. For an
incremental triangle, the library adds the column over the ages. Thus give the
premium at the first age only and set the later cells to zero. If you repeat it
on each cell, the library counts it once for each age.

### Before you choose: check the data

Do two checks before you use a method. Examine the link ratios, and compare
the results from paid and incurred data.

```python
lr = tri.link_ratios().to_long()
by_age = {}
for dev, f in zip(lr["development"], lr["paid"]):
    by_age.setdefault(dev, []).append(f)
for dev, fs in by_age.items():  # one ratio per origin, oldest first
    print(f"{dev:2d} to {dev + 12:<3d}", " ".join(f"{f:.3f}" for f in fs))
```

```text
12 to 24  1.842 1.862 1.858 1.764 1.760 1.767
24 to 36  1.358 1.285 1.351 1.385 1.345
36 to 48  1.148 1.187 1.207 1.128
48 to 60  1.105 1.079 1.084
60 to 72  1.019 1.060
72 to 84  1.017
```

No ratio is an outlier, but the first column has a pattern. The three newest
12 to 24 ratios (1.764, 1.760, 1.767) are below the three oldest (1.842 to
1.862). Faster reporting or a change in payment practice can cause this. The
volume-weighted factor mixes the two regimes. Find the cause before you accept
either regime. The late factors each use only one or two ratios, and there a
tail or a benchmark helps.

When one link ratio is an outlier, compare the three averages (section 2). When
one large loss causes it, remove the loss from the triangle and reserve it
separately.

**Paid or incurred.** Incurred (paid plus case reserves) develops faster. Thus
it gives more information about young origins and long-tailed lines. But it
also includes each change in how the claims department sets case reserves. Run
the same calls on the two columns and compare the results. The block below
makes a synthetic incurred triangle from the paid one. It makes it once with
stable case reserving. It makes it again with the case reserves on the latest
diagonal increased by 25%.

```python
case = [0.80, 0.45, 0.25, 0.13, 0.06, 0.03, 0.015]  # case reserve over paid, by age

def incurred_triangle(strengthen):
    vals = [round(v * (1 + case[k] * (strengthen if k == len(row) - 1 else 1.0)))
            for row in paid for k, v in enumerate(row)]
    return Triangle.from_long(origin, age, {"incurred": vals})

for label, strengthen in [("paid", None), ("incurred", 1.0), ("incurred, case +25%", 1.25)]:
    if strengthen is None:
        fit = ChainLadder().fit(tri, "paid")
    else:
        fit = ChainLadder().fit(incurred_triangle(strengthen), "incurred")
    print(f"{label:20s}", " ".join(f"{u:5.0f}" for u in fit.ultimate), f" total {sum(fit.ultimate):5.0f}")
```

```text
paid                   667   767   762   781   850   974  1053  total  5853
incurred               677   778   773   792   862   989  1068  total  5939
incurred, case +25%    680   787   790   826   924  1099  1245  total  6350
```

With stable case reserving, the two agree within the case reserve that is
still open at 84 months (5,939 against 5,853). This difference shows that paid
possibly needs a tail. After the increase, the incurred chain ladder applies
factors from the old reserving to the stronger cells. The ultimate of the
youngest origin increases from 1,068 to 1,245. The paid triangle does not show
the change. When paid and incurred give different results like this, find the
cause in the claims department before you select either one.

## 2. Chain ladder: the default

**What it is.** Each age-to-age factor is an average of the link ratios at
that age. The product of the factors from the age of an origin to the end is
its age-to-ultimate factor (`cdf`). The ultimate is the latest value times
this factor.

**Use it when** the book is mature and stable, payment patterns did not
change, and each origin has a sufficient paid amount.

**Do not use it alone when** the youngest origins have small paid amounts.
Their ultimate is the latest cell times the largest factor. Thus a noisy first
cell goes directly into the reserve.

**Data it needs.** Only the triangle.

```python
cl = ChainLadder().fit(tri, "paid")
print("age-to-age", [round(f, 3) for f in cl.ldf])
print("age-to-ultimate", [round(f, 3) for f in cl.cdf])
print("ultimate", [round(u) for u in cl.ultimate])
print("reserve", [round(r) for r in cl.reserve], "total", round(cl.total_reserve))
for average in ["volume", "simple", "regression"]:
    print(average, round(ChainLadder(average=average).fit(tri, "paid").total_reserve, 1))
```

```text
age-to-age [1.804, 1.344, 1.167, 1.089, 1.041, 1.017]
age-to-ultimate [3.261, 1.807, 1.344, 1.152, 1.058, 1.017, 1.0]
ultimate [667, 767, 762, 781, 850, 974, 1053]
reserve [0, 13, 42, 103, 218, 435, 730] total 1540
volume 1540.4
simple 1542.7
regression 1538.0
```

`average` sets how the link ratios become a factor:

- `"volume"` (the default) weights by size.
- `"simple"` is the plain mean.
- `"regression"` is least squares through the origin.

On this triangle they differ by less than 5 in 1,540, so the choice is not
important here. It is important when the link ratio of one origin is an
outlier. Then compare all three.

**Pitfalls.** The chain ladder assumes that the next diagonal develops like the
last ones. A change in claims handling or case reserving makes this assumption
false, and the chain ladder does not show it. Use the diagonal backtest in
section 9 to find such a change.

### Tail factors

The oldest age here is 84 months, and paid still increases between ages (the
last factor is more than 1). Without a tail, the model assumes that
development stops. The `tail=` argument adds development after the oldest age.
`ChainLadder`, `Mack`, the expected-loss methods of section 4 and the two
bootstraps take it. Clark uses `max_age` instead. The Bayesian models do not
take a tail.

| Argument | What it does | Use it when |
|---|---|---|
| `tail=1.02` or `TailConstant(1.02)` | A given factor from the oldest age to ultimate | You have a benchmark |
| `TailCurve()` | Regresses `ln(f - 1)` on the development index 1, 2, 3, ... (`"exponential"`) or on its log (`"inverse_power"`), over the factors above 1, and extrapolates | The factors decay smoothly |
| `TailBondy()` | Each log factor is a constant times the one before it. By default this repeats the last factor | A simple rule when you do not trust a curve |
| `TailLogLinear()` | R ChainLadder's `tail = TRUE` rule | You must match an R `MackChainLadder(tail = TRUE)` |

```python
for name, tail in [
    ("none", None),
    ("TailConstant(1.02)", TailConstant(1.02)),
    ("TailCurve()", TailCurve()),
    ("TailBondy()", TailBondy()),
    ("TailLogLinear()", TailLogLinear()),
]:
    fit = ChainLadder(tail=tail).fit(tri, "paid")
    print(f"{name:20s} tail {fit.tail:.4f}  reserve {fit.total_reserve:7.1f}")
```

```text
none                 tail 1.0000  reserve  1540.4
TailConstant(1.02)   tail 1.0200  reserve  1657.5
TailCurve()          tail 1.0161  reserve  1634.5
TailBondy()          tail 1.0168  reserve  1638.6
TailLogLinear()      tail 1.0161  reserve  1634.5
```

`TailCurve()` and `TailLogLinear()` give the same factor here, because both
regress `ln(f - 1)` on the development index. Their guards are different.
`TailLogLinear` is 1 unless the third-last and second-last factors multiply to
more than 1.0001. It also resets a tail above 2 to 1, as R does. A tail of 1.02
adds 117 to the reserve, and a fitted curve adds 94. This is approximately 6%
to 8% of the reserve, so a tail is always an important assumption.

## 3. Mack: the chain ladder with a standard error

**What it is.** The distribution-free model of Mack has the same reserve as the
chain ladder. It adds the standard error of the reserve, in two parts:

- **Process risk** is the randomness of the payments.
- **Parameter risk** is the error in the estimated factors.

**Use it when** you must show the uncertainty of the chain ladder, and a mean
and a standard deviation are sufficient. It gives the numbers for a reserve
range or a coefficient of variation.

**Do not use it when** you need a quantile of the reserve. A standard error is
not a distribution. For the 99.5% reserve, use a bootstrap (section 6).

```python
mack = Mack().fit(tri, "paid")
print(f"reserve {mack.total_reserve:.1f}, standard error {mack.total_standard_error:.1f}, cv {mack.total_cv:.3f}")
print(f"process {mack.total_process_risk:.1f}, parameter {mack.total_parameter_risk:.1f}")
print("by origin", [round(s, 1) for s in mack.standard_error])
tailed = Mack(tail=TailCurve()).fit(tri, "paid")
print(f"with a fitted tail: reserve {tailed.total_reserve:.1f}, standard error {tailed.total_standard_error:.1f}")
```

```text
reserve 1540.4, standard error 173.6, cv 0.113
process 90.8, parameter 148.0
by origin [0.0, 20.8, 33.2, 35.5, 47.1, 58.2, 67.4]
with a fitted tail: reserve 1634.5, standard error 204.0
```

The coefficient of variation is approximately 11% here. Parameter risk (148.0)
is larger than process risk (90.8). A tail changes the standard error more, in
percentage, than it changes the reserve. The standard error goes from 173.6 to
204.0 (+17%), and the reserve increases by 6%. The cause is that the tail has
its own sigma and standard error. By default, Mack extrapolates both from the
factors. `tail_sigma=` and `tail_std_err=` replace them.

## 4. Methods that use an a priori: BF, Benktander, Cape Cod, expected loss

The chain ladder gives full weight to the latest cell. The methods in this
section mix the cell with a **prior** ultimate: the expected loss ratio times
the exposure.

| Method | Ultimate | Prior | Trust the data |
|---|---|---|---|
| `ExpectedLoss(apriori)` | prior | given | not at all |
| `BornhuetterFerguson(apriori)` | latest + prior x (1 - 1 / cdf) | given | in what is already paid |
| `Benktander(apriori, n_iters)` | BF repeated `n_iters` times | given | grows with `n_iters` |
| `CapeCod(trend, decay)` | BF with a prior estimated from the triangle | estimated | in what is already paid |

```python
methods = {
    "chain ladder": ChainLadder().fit(tri, "paid"),
    "expected loss": ExpectedLoss(apriori=0.70).fit(tri, "paid", "premium"),
    "Bornhuetter-Ferguson": BornhuetterFerguson(apriori=0.70).fit(tri, "paid", "premium"),
    "Benktander (2 steps)": Benktander(apriori=0.70, n_iters=2).fit(tri, "paid", "premium"),
    "Cape Cod": CapeCod().fit(tri, "paid", "premium"),
}
print(f"{'method':22s}" + "".join(f"{y:>7d}" for y in years) + "   total")
for name, fit in methods.items():
    print(f"{name:22s}" + "".join(f"{r:7.0f}" for r in fit.reserve) + f"  {fit.total_reserve:6.0f}")
```

```text
method                   2018   2019   2020   2021   2022   2023   2024   total
chain ladder                0     13     42    103    218    435    730    1540
expected loss              33    -19     50    134    222    357    622    1399
Bornhuetter-Ferguson        0     12     42    107    219    400    655    1436
Benktander (2 steps)        0     13     42    103    218    420    678    1474
Cape Cod                    0     12     43    108    221    404    661    1448
```

Read the table by origin:

- In 2018 there is no development left. All methods agree, except expected
  loss. Expected loss ignores the 667 already paid and puts a reserve of 33 on
  a closed year.
- In 2019, expected loss gives a negative reserve of -19. The prior of 70% of
  premium is less than the amount already paid.
- The youngest origin, 2024, shows the real difference. The chain ladder gives
  730, the methods with a prior give 655 to 678, and expected loss gives 622.
  This difference is the actuarial judgment that you must make.

**Bornhuetter-Ferguson (BF).** The reserve is the prior ultimate times the
share still to develop. The paid amount to date does not change the reserve
(only the pattern does). This protects against an unusual first cell, but it
also ignores a good or bad first cell.

**When to use BF.** The origin is young, or the pattern is volatile. Also, the
prior comes from a source other than the triangle (a plan, pricing, an industry
benchmark). **When not to use BF.** The prior is old, or the people under
evaluation set it. The method then repeats the error.

**Benktander** uses the ultimate of BF as the new prior and repeats the
calculation. With `n_iters=0` it is the expected loss method, with `1` it is
BF, and with many steps it is the chain ladder:

```python
for n in [0, 1, 2, 5, 100]:
    print(n, round(Benktander(apriori=0.70, n_iters=n).fit(tri, "paid", "premium").total_reserve, 1))
```

```text
0 1399.0
1 1435.6
2 1473.6
5 1521.7
100 1540.4
```

Thus `n_iters` moves the result between the prior (1,399) and the data
(1,540). The default is 1, which is Bornhuetter-Ferguson. Each step moves the
reserve toward the chain ladder, and at 100 steps the two are equal. Use
`n_iters=2`, the usual choice. It is the choice of Benktander and is near the
credibility-optimal mix of BF and the chain ladder (Mack 2000). If you want
more steps, you want a result near the chain ladder, so use the chain ladder.

**Cape Cod** does not need a plan loss ratio. It estimates the loss ratio from
the triangle: the paid amount to date divided by the exposure that has
developed.

```python
cc = CapeCod().fit(tri, "paid", "premium")
print("Cape Cod loss ratio", [round(a, 3) for a in cc.apriori][:2], "...")
cc2 = CapeCod(trend=0.03, decay=0.7).fit(tri, "paid", "premium")
print("with 3% trend and decay 0.7", [round(a, 3) for a in cc2.apriori])
```

```text
Cape Cod loss ratio [0.706, 0.706] ...
with 3% trend and decay 0.7 [0.665, 0.684, 0.695, 0.706, 0.724, 0.749, 0.772]
```

With the defaults, all origins have one loss ratio of 0.706. `trend` changes
the ratio of each year by an annual rate to the valuation date. A `decay` below
1 gives more weight to nearby years. With a 3% trend and a decay of 0.7, the
priors increase from 0.665 to 0.772. Use `decay=1` and `trend=0` only if the
years are exchangeable. In a changing market, they are usually not
exchangeable.

**A different prior for each year.** The plan loss ratio is frequently
different by year. Give `apriori` as a dict keyed by accident year, with one
loss ratio for each origin. The premium stays in the exposure column. Here the
plan expects the loss ratio to decrease from 70% to 65% after rate increases:

```python
elr = [0.70, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65]
plan = dict(zip(years, elr))
bf = BornhuetterFerguson(apriori=plan).fit(tri, "paid", "premium")
print("apriori", bf.apriori[:2], "...", bf.apriori[-1])
print("reserve", [round(r) for r in bf.reserve], "total", round(bf.total_reserve))
print("flat 70% prior, total", round(methods["Bornhuetter-Ferguson"].total_reserve))
```

```text
apriori [0.7, 0.7] ... 0.65
reserve [0, 12, 42, 104, 209, 377, 608] total 1353
flat 70% prior, total 1436
```

Each origin now has its own expected ultimate, `apriori[o] * exposure[o]`. The
decreasing plan changes the BF reserve from 1,436 (a flat 70%) to 1,353. Most of
the change is in the two youngest origins. This works for `ExpectedLoss`,
`BornhuetterFerguson` and `Benktander`. A list or array in origin order, oldest
first, also works, but use the keys when you can. A list can go to the wrong
origins without an error if your rows are not in the order you expect. A
pandas Series is read by its index, so `df.set_index("origin")["elr"]` lines up
in any row order. In R, give a vector named by origin: `apriori = c("2018" = 0.70, ...)`.

Before this form existed, the workaround was to put the expected ultimate in a
column, use it as the exposure and set `apriori=1.0`. It gives the same reserve.
It is still the way to give Cape Cod (and Clark's Cape Cod) a column of
relativities between years, because they estimate the level again. Here the
data increase the plan by a factor of 1.034, to a reserve of 1,399:

```python
expected = [r * p for r, p in zip(elr, premium)]
tri_prior = Triangle.from_long(
    origin, age, {"paid": [v for row in paid for v in row], "prior": [expected[years.index(y)] for y in origin]}
)
bf_col = BornhuetterFerguson(apriori=1.0).fit(tri_prior, "paid", "prior")
print("exposure column form, total", round(bf_col.total_reserve), "same as the dict form:",
      abs(bf_col.total_reserve - bf.total_reserve) < 1e-9)
cc_prior = CapeCod().fit(tri_prior, "paid", "prior")
print(f"Cape Cod on the same column: apriori {cc_prior.apriori[0]:.3f}, total {cc_prior.total_reserve:.0f}")
```

```text
exposure column form, total 1353 same as the dict form: True
Cape Cod on the same column: apriori 1.034, total 1399
```

**Is the result reasonable?** Before you book a method, compare the ultimate
loss ratio of each origin under each method with the plan:

```python
print(f"{'method':22s}" + "".join(f"{y:>7d}" for y in years))
for name, fit in methods.items():
    print(f"{name:22s}" + "".join(f"{u / p:7.3f}" for u, p in zip(fit.ultimate, premium)))
print(f"{'plan':22s}" + "".join(f"{r:7.3f}" for r in elr))
```

```text
method                   2018   2019   2020   2021   2022   2023   2024
chain ladder            0.667  0.730  0.693  0.673  0.696  0.761  0.780
expected loss           0.700  0.700  0.700  0.700  0.700  0.700  0.700
Bornhuetter-Ferguson    0.667  0.730  0.693  0.677  0.697  0.734  0.725
Benktander (2 steps)    0.667  0.730  0.693  0.674  0.697  0.749  0.742
Cape Cod                0.667  0.730  0.693  0.678  0.699  0.736  0.729
plan                    0.700  0.700  0.690  0.680  0.670  0.660  0.650
```

The mature years are near 0.67 to 0.73. The chain ladder then puts 2023 and
2024 at 0.761 and 0.780. These values are above all earlier years and much
above a plan of 0.66 and 0.65. One of two causes applies. Either the business
became worse, or the first cells of the young origins are high and the chain
ladder multiplies them by its largest factors. This question, not the
arithmetic, decides between the chain ladder and BF for those two years.

## 5. Clark: a growth curve instead of factors

**What it is.** Clark fits a parametric growth curve (`"loglogistic"` or
`"weibull"`) and an expected ultimate for each origin. The fit uses the
incremental losses and over-dispersed Poisson maximum likelihood.
`ClarkCapeCod` uses one expected loss ratio times the exposure instead of one
ultimate for each origin.

**Use it when** there are few ages, the triangle is noisy from cell to cell,
or you want a smooth pattern with few parameters. The fit gives process and
parameter standard errors (`total_standard_error`).

**Do not use it without checks.** The curve sets the tail. By default the
curve develops to infinity, and a log-logistic curve has a heavy tail:

```python
for label, model in [
    ("loglogistic, to infinity", ClarkLdf()),
    ("loglogistic, stop at 120", ClarkLdf(max_age=120)),
    ("weibull, to infinity", ClarkLdf("weibull")),
    ("weibull, stop at 120", ClarkLdf("weibull", max_age=120)),
]:
    fit = model.fit(tri, "paid")
    print(f"{label:26s} reserve {fit.total_reserve:6.0f}  standard error {fit.total_standard_error:5.0f}")
print("Clark Cape Cod, weibull, 120:", round(ClarkCapeCod("weibull", 120).fit(tri, "paid", "premium").total_reserve))
```

```text
loglogistic, to infinity   reserve   5267  standard error  1425
loglogistic, stop at 120   reserve   2407  standard error   309
weibull, to infinity       reserve   2588  standard error   538
weibull, stop at 120       reserve   2128  standard error   289
Clark Cape Cod, weibull, 120: 1985
```

The same data give a reserve of 5,267, 2,407, 2,588 or 2,128, from the curve
and the cut-off. Clark's Cape Cod gives 1,985. The chain ladder gives 1,540
with no tail and 1,635 with a fitted tail. Before you use any of these values,
check the fit. The curve gives an age-to-age factor between any two ages,
`fit.growth(a + 12) / fit.growth(a)`. Compare it with the factor of the chain
ladder.

```python
print(f"{'chain ladder':18s}", [round(f, 3) for f in cl.ldf])
for label, model in [("loglogistic, inf", ClarkLdf()), ("weibull, 120", ClarkLdf("weibull", max_age=120))]:
    f = model.fit(tri, "paid")
    end = f.max_age if f.max_age is not None else float("inf")
    print(f"{label:18s}", [round(f.growth(a + 12) / f.growth(a), 3) for a in range(12, 84, 12)],
          f"developed at 84: {f.growth(84) / f.growth(end):.0%}")
```

```text
chain ladder       [1.804, 1.344, 1.167, 1.089, 1.041, 1.017]
loglogistic, inf   [1.903, 1.276, 1.148, 1.096, 1.069, 1.052] developed at 84: 63%
weibull, 120       [1.923, 1.282, 1.147, 1.091, 1.061, 1.043] developed at 84: 93%
```

The curves do not fit the last two factors. The log-logistic curve gives 1.069
and 1.052, but the triangle shows 1.041 and 1.017. The curves also miss the
first factor in the other direction. A two-parameter curve cannot bend
sufficiently for this pattern. Thus it puts too much development after 84
months: the log-logistic curve says that only 63% is paid at 84 months.

Here the chain ladder fits the data and Clark does not. On a triangle this
mature, use Clark as a comparison, not as the booked method. Use it when the
triangle has few ages and the factors are too noisy to read one by one. Set
`max_age` to the age at which you think that development stops.

## 6. When you need a distribution, not a number

The methods above give a mean and at most a standard error. A 99.5% capital
figure, an adverse development cover, a risk margin and a reserve range need a
distribution. prospicio has four methods for this: the three in the table
below, and `Meyers` at the end of this section. Select one by the variance
structure that you think is correct.

| | `OdpBootstrap` | `MackBootstrap` | `BayesOdp` |
|---|---|---|---|
| Model | Over-dispersed Poisson on increments | Mack's model on link ratios | Over-dispersed Poisson with priors |
| Variance | scale x mean increment | Mack's: sigma squared per age, times the cumulative value | scale x mean increment |
| Reproduces | R `BootChainLadder` | Mack's standard error, Merz-Wuthrich one-year | The ODP model with priors (England and Verrall) |
| Negative cells | allowed | not allowed | allowed |
| Tail | yes | yes | no |

### The two bootstraps

```python
odp = OdpBootstrap(n_sims=10_000, seed=1).fit(tri, "paid")
mb = MackBootstrap(n_sims=10_000, seed=1).fit(tri, "paid")
print(f"{'':16s}{'mean':>8s}{'std dev':>9s}{'99.5%':>8s}")
for name, fit in [("ODP bootstrap", odp), ("Mack bootstrap", mb)]:
    r = fit.reserves
    print(f"{name:16s}{r.mean():8.0f}{r.variance() ** 0.5:9.0f}{r.quantile(0.995):8.0f}")
print(f"{'Mack formula':16s}{mack.total_reserve:8.0f}{mack.total_standard_error:9.0f}")
```

```text
                    mean  std dev   99.5%
ODP bootstrap       1544      125    1881
Mack bootstrap      1541      174    1981
Mack formula        1540      174
```

The mean and standard deviation of the Mack bootstrap agree with Mack's
formula. This is the reconciliation to check. The ODP bootstrap has the same
mean and a smaller standard deviation (125 against 174). The ODP sets the
process variance in proportion to the increment. Mack's formula sets it in
proportion to the cumulative value. These are different assumptions, and
neither is wrong. The knowledge note
[`mack-bootstrap-lifetime-vs-mack.md`](../../knowledge/findings/mack-bootstrap-lifetime-vs-mack.md)
records the reconciliation. At the 99.5% level, which a capital model reads,
the two differ by 100 (1,881 against 1,981).

Use `MackBootstrap` when your reserving process already reports the standard
error of Mack and the distribution must agree with it. Use `OdpBootstrap` when
you want agreement with R `BootChainLadder`, or when a triangle has a negative
increment. By default, both use the volume-weighted chain ladder. Both give
`fit.reserves`, a joint distribution by origin with `mean()`, `quantile(p)`,
`variance()`, `tvar(p)` and `total()`.

**Read the tail carefully.** A 99.5% quantile from 10,000 simulations uses the
worst 50 of them. Run again with different seeds to see the simulation error:

```python
for seed in [1, 2, 3]:
    o = OdpBootstrap(n_sims=10_000, seed=seed).fit(tri, "paid").reserves
    m = MackBootstrap(n_sims=10_000, seed=seed).fit(tri, "paid").reserves
    print(f"seed {seed}  ODP 99.5% {o.quantile(0.995):.0f}  Mack 99.5% {m.quantile(0.995):.0f}")
```

```text
seed 1  ODP 99.5% 1881  Mack 99.5% 1981
seed 2  ODP 99.5% 1884  Mack 99.5% 1987
seed 3  ODP 99.5% 1890  Mack 99.5% 1984
```

The seeds move the 99.5% by less than 10. This is much less than the 100
between the two models. Thus simulation error is the small part. The large
parts are not in any bootstrap. Seven years of residuals are a small sample
from which to learn a tail shape. The bootstrap also assumes the method, the
tail and a stable calendar-year environment. Model error (the choice of method
and tail, an inflation shock) also adds to this. Thus treat a 99.5% from a
triangle of this size as a central estimate of a very uncertain number.

**Tails.** Use `tail=` and, for a constant, `tail_std_err=`. The bootstrap
fits an estimated tail (`TailCurve`, `TailBondy`, `TailLogLinear`) again on
each simulation, which adds its own parameter error. `OdpBootstrap` does not
vary a constant tail unless you give `tail_std_err`. If you do not give it,
`MackBootstrap` extrapolates the standard error of Mack.

```python
tail = TailConstant(1.02)
boot = MackBootstrap(n_sims=10_000, seed=1, tail=tail, tail_std_err=0.01)
lifetime = boot.fit(tri, "paid")
one_year = boot.one_year(tri, "paid", ChainLadder(tail=tail))
print(f"lifetime  mean {lifetime.reserves.mean():.0f}  std dev {lifetime.reserves.variance() ** 0.5:.0f}")
print(f"one-year  mean {one_year.cdr.mean():.0f}  std dev {one_year.cdr.variance() ** 0.5:.0f}")
```

```text
lifetime  mean 1660  std dev 191
one-year  mean 2  std dev 130
```

A constant tail of 1.02 with a standard error of 0.01 increases the lifetime
mean to 1,660. The plug-in chain ladder with that tail gives 1,657.5. The tail
increases the standard deviation to 191. The one-year standard deviation is
130, against 127.2 without a tail (section 7), and its mean is approximately
zero.

Pitfall: an estimated tail that the ODP fits again on its pseudo factors is
sometimes exactly 1. Thus the ODP bootstrap with `TailLogLinear` is less stable
than a constant with a standard error (the finding
[`bootstrap-tail-parameter-error.md`](../../knowledge/findings/bootstrap-tail-parameter-error.md)).

### Bayesian models

**`BayesOdp`** puts normal priors on the log-scale origin and development
effects of the same ODP model. It samples the posterior with NUTS and gives
the posterior predictive distribution of the reserve. The priors are on the
log-scale effects, not on an ultimate or a loss ratio. `intercept_sd`,
`origin_sd` and `development_sd` set their widths (10 by default). Thus it is
difficult to state these priors from actuarial judgment. It takes a cumulative
triangle with no tail and no exposure column.

```python
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bayes = BayesOdp(n_sims=10_000, seed=1).fit(tri, "paid")
d = bayes.diagnostics
print("warnings", len(caught), "| max R-hat below 1.01:", d["max_rhat"] < 1.01, "| divergences", d["divergences"])
print(f"posterior predictive mean / chain ladder reserve {bayes.reserves.mean() / cl.total_reserve:.2f}")
print(f"posterior predictive std dev / ODP bootstrap std dev {bayes.reserves.variance() ** 0.5 / odp.reserves.variance() ** 0.5:.1f}")
```

```text
warnings 0 | max R-hat below 1.01: True | divergences 0
posterior predictive mean / chain ladder reserve 1.00
posterior predictive std dev / ODP bootstrap std dev 1.0
```

The default priors are wide. With them, the posterior predictive mean is the
mean of the chain ladder (ratio 1.00). Its standard deviation is that of the
ODP bootstrap (ratio 1.0). This is the check: a Bayesian fit with weak
priors must give the frequentist result. The finding
[`bayes-odp-vs-odp-bootstrap.md`](../../knowledge/findings/bayes-odp-vs-odp-bootstrap.md)
compares them with Stan.

Always read `fit.diagnostics` (R-hat, effective sample sizes, divergences).
When the sampler fails, the library does not stop. It gives one `UserWarning`,
and the first line above counts these warnings (0 here). Do not report a
posterior if you did not read its warning.

Thus, with its default priors, `BayesOdp` gives the same result as the ODP
bootstrap by a different method. It runs in less than a second on this
triangle, so cost is not a reason to omit it, but it adds nothing here. Use it
when you want to put a belief about the origin levels or the development
pattern into the model. An example is a smaller `origin_sd`. It moves the
level of each origin toward the level of the first origin, because the prior of
each effect has its centre at 0. For a prior on the loss ratio, which is the prior
that an actuary usually has, use `Meyers`.

**`Meyers`** fits the log-normal models of Meyers (CAS Monograph 8): `"crc"`
(correlated chain ladder, without the correlation between origins) and `"csr"`
(changing settlement rate). It takes the premium as a base
(`fit(tri, "paid", "premium")`) and puts a prior on the log loss ratio. The
standard deviation of the log value decreases with age. All observed cells must
be positive. Its ultimate is the value at the oldest age, so there is no tail.

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the sampler warns when its diagnostics fail
    meyers = Meyers("crc", n_sims=5000, seed=1).fit(tri, "paid", "premium")
print("Meyers std dev more than twice the ODP bootstrap's:", meyers.reserves.variance() ** 0.5 > 2 * odp.reserves.variance() ** 0.5)
```

```text
Meyers std dev more than twice the ODP bootstrap's: True
```

It needs more data than this example has. On this 7 by 7 triangle, the
log-normal ultimates are very uncertain. Thus the predictive standard deviation
is more than twice that of the ODP bootstrap, and the sampler can warn that its
diagnostics fail. Use it on triangles like the triangles of Meyers (10 by 10
with premium), and read the diagnostics before the reserve. The knowledge note
[`meyers-crc-csr-parity.md`](../../knowledge/findings/meyers-crc-csr-parity.md)
gives the parity checks.

## 7. Lifetime or one-year?

The bootstraps above give the **lifetime** (run-off) view: the uncertainty in
the reserve until the last claim is paid. A solvency standard asks a different
question. The **one-year view** is the uncertainty in the reserve over the
next year. It is the **claims development result (CDR)**: the opening ultimate
minus the ultimate after one more diagonal. It is smaller than the lifetime
risk, because it does not include the part that occurs after next year.

| Question | View | Call |
|---|---|---|
| Reserve range, ADC or LPT pricing, IFRS 17 run-off | Lifetime | `fit()` |
| Solvency II reserve risk, a 99.5% one-year loss | One-year | `claims_development_result()` or `one_year()` |

```python
mw = mack.claims_development_result()
mack_1y = MackBootstrap(n_sims=10_000, seed=1).one_year(tri, "paid", ChainLadder())
odp_1y = OdpBootstrap(n_sims=10_000, seed=1).one_year(tri, "paid", ChainLadder())
bf_1y = MackBootstrap(n_sims=10_000, seed=1).one_year(tri, "paid", BornhuetterFerguson(0.70), exposure="premium")
print(f"Mack lifetime (formula)         {mack.total_standard_error:6.1f}")
print(f"Merz-Wuthrich one-year          {mw.total_one_year_standard_error:6.1f}")
print(f"Mack bootstrap one-year         {mack_1y.cdr.variance() ** 0.5:6.1f}   mean CDR {mack_1y.cdr.mean():.1f}")
print(f"ODP bootstrap one-year          {odp_1y.cdr.variance() ** 0.5:6.1f}")
print(f"Bornhuetter-Ferguson one-year   {bf_1y.cdr.variance() ** 0.5:6.1f}   mean CDR {bf_1y.cdr.mean():.1f}")
print("one-year 99.5% loss:", round(-mack_1y.cdr.quantile(0.005)))
```

```text
Mack lifetime (formula)          173.6
Merz-Wuthrich one-year           127.1
Mack bootstrap one-year          127.2   mean CDR -1.2
ODP bootstrap one-year           100.5
Bornhuetter-Ferguson one-year    102.5   mean CDR -38.8
one-year 99.5% loss: 326
```

How to read the results:

- The one-year standard error of the chain ladder is 127.1 by the
  Merz-Wuthrich formula, against 173.6 for the lifetime.
  `MackBootstrap().one_year` gives 127.2 with a mean CDR of -1.2
  (approximately zero). Thus the simulation reproduces the formula.
- The same re-reserving under the ODP gives 100.5. The difference comes from
  the process model of the ODP, not from the re-reserving. The ratio to
  Merz-Wuthrich is different for each triangle (the finding
  [`one-year-bootstrap-vs-merz-wuthrich.md`](../../knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md)).
  Thus use `MackBootstrap` when the figure must agree with Merz-Wuthrich.
- `one_year` accepts `ChainLadder`, `ExpectedLoss`, `BornhuetterFerguson`,
  `Benktander` or `CapeCod` as `method`, with any averaging, tail or
  development grain. The closed form does not. For the expected-loss methods,
  give `exposure=`.
- The simulated next diagonal always follows the process of the bootstrap
  (here the chain ladder of Mack). The method only estimates again on it. Thus
  the Bornhuetter-Ferguson row is the risk of a BF estimate in a chain ladder
  world. Its mean CDR of -38.8 is not zero, because BF opens below the chain
  ladder (1,436 against 1,540). One more diagonal of data moves it toward the
  chain ladder. Do not read its 102.5 against the 127.2 of the chain ladder as
  "BF has less reserve risk". Do not give a CDR with a non-zero mean to a
  capital figure without a statement of the offset.
- The CDR is positive when development is favourable. Thus a one-year loss is a
  negative CDR. `-cdr.quantile(0.005)` is the one-year 99.5% loss, here 326.
- With a bootstrap `tail`, an origin whose tail step is in the next year takes
  all of the tail in that year. Thus its one-year risk is an upper bound
  (section 6, tails).

## 8. Several lines and dependence

A triangle with `keys` holds several segments. A bootstrap fit on all segments
gives one joint distribution, and `dependence` sets how the lines move
together.

| `dependence` | What it does | Use it when |
|---|---|---|
| `"independent"` | Each line resamples its own residuals | The lines are not related, or as a lower bound when you expect no negative correlation |
| `"synchronized"` | All lines take the residuals of the same origins and ages | All lines have the same shape of data, and you think that the residuals correlate |
| `"rank_correlation"` with `spearman=` | Iman-Conover reorders the simulations of the lines to your Spearman matrix | You have a correlation from a different source: the matrix of a regulator, or your own study of the calendar-year results of the lines |

Synchronized resampling needs the same origins, ages and observed cells in all
lines. Its correlation is approximately the correlation of the residuals of the
lines, decreased by process error. In the example, the second line is half of
the first. Thus their residuals are almost identical, and the correlation is
larger than in a real pair of lines.

```python
vals = [v for row in paid for v in row]
two = Triangle.from_long(
    origin * 2,
    age * 2,
    {"paid": vals + [round(0.5 * v) for v in vals]},
    keys={"line": ["property"] * len(vals) + ["liability"] * len(vals)},
)
for dependence, extra in [("independent", {}), ("synchronized", {}), ("rank_correlation", {"spearman": [[1, 0.5], [0.5, 1]]})]:
    r = OdpBootstrap(n_sims=5000, seed=2, dependence=dependence, **extra).fit(two, "paid").reserves
    by_line = r.aggregate(["line"])
    lines = " ".join(f"{k} {by_line.marginal((k,)).variance() ** 0.5:5.1f}" for k in ["property", "liability"])
    print(f"{dependence:17s} total {r.variance() ** 0.5:6.1f}  99.5% {r.quantile(0.995):5.0f} | {lines}")
```

```text
independent       total  138.8  99.5%  2695 | property 125.0 liability  60.3
synchronized      total  176.6  99.5%  2779 | property 126.0 liability  60.3
rank_correlation  total  164.5  99.5%  2751 | property 125.0 liability  60.3
```

After the bar are the standard deviations of each line. Dependence does not
change them (125.0 or 126.0 for property, 60.3 for liability), but it makes the
total wider. The standard deviation of the total reserve goes from 138.8 to
176.6 (synchronized) or 164.5 (a Spearman correlation of 0.5). The 99.5% goes
from 2,695 to 2,779 or 2,751. The 0.5 is an input, and the triangle does not
estimate it. Take it from the matrix of a regulator or from your own study.
Then test how much the total changes with it. `fit.reserves.aggregate(["line"])`
gives the reserve of each line and keeps the dependence between them.

## 9. Which method is right? Backtest it

No method is right in general. One method is right for your book. A **diagonal
backtest** fits the triangle again without its newest calendar diagonal. It
then forecasts that diagonal and compares. An actual-over-expected ratio near
1, with no drift, supports the assumptions of the method.

The loop below is a chain ladder backtest on an annual triangle. It keeps the
cells with a valuation before a calendar year and fits the chain ladder. Then
it compares the next diagonal with the forecast `latest x age-to-age factor`.

```python
long = tri.to_long()
valuation_year = [o.year + dev // 12 - 1 for o, dev in zip(long["origin"], long["development"])]
actual = {(o.year, dev): v for o, dev, v in zip(long["origin"], long["development"], long["paid"])}

def backtest(year):
    keep = [i for i, y in enumerate(valuation_year) if y < year]
    train = Triangle.from_long(
        [long["origin"][i] for i in keep],
        [long["development"][i] for i in keep],
        {"paid": [long["paid"][i] for i in keep]},
    )
    ldf = ChainLadder().fit(train, "paid").ldf
    pairs = {}  # origin: (observed, forecast)
    for oy in range(2018, year):
        k = year - oy
        if k - 1 < len(ldf):
            pairs[oy] = (actual[(oy, 12 * (k + 1))], actual[(oy, 12 * k)] * ldf[k - 1])
    return pairs

for year in [2022, 2023, 2024]:  # actual over expected, in total and by origin
    pairs = backtest(year)
    total = sum(a for a, _ in pairs.values()) / sum(f for _, f in pairs.values())
    print(year, f"total {total:.3f} |", " ".join(f"{oy}: {a / f:.3f}" for oy, (a, f) in pairs.items()))
```

```text
2022 total 1.008 | 2019: 1.035 2020: 1.025 2021: 0.951
2023 total 1.004 | 2019: 0.977 2020: 1.033 2021: 1.041 2022: 0.962
2024 total 0.994 | 2019: 1.041 2020: 0.994 2021: 0.955 2022: 1.000 2023: 0.974
```

The totals are within 1% of 1, but the origins show more. In each year, the
youngest origin (the origin that the 12 to 24 factor forecasts) was below its
forecast (0.951, 0.962, 0.974). This is the decrease in the newest 12 to 24
ratios of section 1. The first factor of the chain ladder is high, and this
also explains the high 2023 and 2024 loss ratios of section 4. Three diagonals
alone are weak evidence. With the other two checks, they support BF or Cape
Cod on the two youngest origins.

The loop above applies to the chain ladder only. A BF backtest needs its own
loop. It forecasts the next increment of each origin as the prior ultimate
times the share of the pattern in the year:
`prior x (1 / cdf(k + 1) - 1 / cdf(k))`. Here the age-to-ultimate factors come
from the training fit, and `cdf(k)` is the factor at the current age of the
origin.

## 10. From reserve distribution to reinsurance and capital

A reserve distribution is a `PredictiveDistribution`, the same type that the
pricing guides use. Thus it goes directly into a reinsurance structure.

- **Capital.** `reserves.quantile(0.995)` is the 99.5% reserve, not the
  capital. The capital is the amount by which that quantile is more than the
  mean (or more than the booked reserve). `reserves.tvar(0.995)` is the tail
  value at risk. For a one-year standard, the capital is
  `-cdr.quantile(0.005)`, because a negative CDR is adverse development.
- **Adverse development cover (ADC).** A stop-loss layer on the total reserve.
  `Tower.apply_aggregate` treats the total of each simulation as one loss and
  applies the layer to it.
- **Loss portfolio transfer (LPT).** A quota share of the full reserve.

```python
from prospicio.reinsurance import Layer, Tower

adc = Tower([Layer.stop_loss("ADC", 300.0, 1700.0)]).apply_aggregate(mb.reserves)
by_kind = adc.aggregate(["kind"])
gross, ceded, net = (by_kind.marginal((k,)) for k in ["gross", "ceded", "net"])
print(f"gross mean {gross.mean():.0f}, 99.5% {gross.quantile(0.995):.0f}")
print(f"ADC pays on average {ceded.mean():.1f}, attaches in {1 - ceded.cdf(0.0):.1%} of simulations")
print(f"net mean {net.mean():.0f}, 99.5% {net.quantile(0.995):.0f}")
lpt = Tower([Layer.quota_share("LPT", 1.0)]).apply_aggregate(mb.reserves)
print(f"gross 99.5% tail value at risk {mb.reserves.tvar(0.995):.0f}")
print("LPT cedes", round(lpt.aggregate(["kind"]).marginal(("ceded",)).mean()))
gross_cap, net_cap = gross.quantile(0.995) - gross.mean(), net.quantile(0.995) - net.mean()
print(f"lifetime capital, 99.5% less mean: gross {gross_cap:.0f}, net {net_cap:.0f}")
print(f"one-year capital, -cdr.quantile(0.005): {-mack_1y.cdr.quantile(0.005):.0f}")
```

```text
gross mean 1541, 99.5% 1981
ADC pays on average 17.7, attaches in 18.7% of simulations
net mean 1524, 99.5% 1700
gross 99.5% tail value at risk 2019
LPT cedes 1541
lifetime capital, 99.5% less mean: gross 440, net 176
one-year capital, -cdr.quantile(0.005): 326
```

The ADC of 300 xs 1,700 attaches in 18.7% of simulations and pays 17.7 on
average. This is its expected loss, before a risk load. The net mean is 1,524,
and the ADC caps the net 99.5% at 1,700, down from 1,981 gross. On the lifetime
view, the capital decreases from 440 (1,981 minus the mean of 1,541) to 176
net. The one-year capital of the same reserve is 326 (section 7). These figures
do not include discounting or unallocated loss adjustment expenses. An IFRS 17
measure or the price of a cover adds them.

The LPT cedes all 1,541 on average. Its price starts from that mean, adds a
load for the risk and subtracts a discount for the time to pay. Use
`prospicio.pricing` for these two parts:

1. `price(ceded, Distortion.tvar(0.995), cost_of_capital=...)` on the ceded
   marginal of the LPT gives a one-period price (`Distortion` is in
   `prospicio.risk`).
2. `price_over_payout(that_price, pattern, rate=..., cost_of_capital=...)`
   gives the same price over a payment pattern. It discounts the losses. It
   charges the cost of capital (a spread over the discount rate) on the
   capital. The capital decreases with the share of the loss that is still
   unpaid.
   Give it the same cost of capital or distortion as the one-period price.

The pattern of a reserve on the books is its run-off from the valuation date.
It is not the pattern of a new origin from its start. Make it from the paid
chain ladder as follows:

1. For each calendar year `j` after the valuation, add the expected payments of
   all origins: `ultimate x (1 / cdf(k + j) - 1 / cdf(k + j - 1))`. Here
   `cdf(k)` is the factor at the current age of the origin.
2. Divide these sums by the total reserve.
3. Give the times `1, 2, ...` (years after the valuation) and those shares to
   `PaymentPattern(times, shares)`.

`PaymentPattern.from_development([age / 12 for age in tri.development], cl.cdf)`
gives the pattern of a new accident year from its start (the last factor must
be 1). This pattern is for a prospective cover, not for the booked reserve.
[`reinsurance-price.md`](reinsurance-price.md) gives an example on a layer.
The time value bullet in
[`risk-loads-and-distortions.md`](risk-loads-and-distortions.md) explains the
calculation over the payout.

To price the cover, add a load to the expected loss for the risk that the
reinsurer takes. The guide
[`risk-loads-and-distortions.md`](risk-loads-and-distortions.md) shows how, and
[`reinsurance-price.md`](reinsurance-price.md) shows the parts of a price. The
expected loss of an ADC is sensitive to the tail of the reserve distribution.
Thus use the tail that you think is correct. The `tail` of the bootstrap and
the choice between the process of Mack and of the ODP change it.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): how to select from all the
  methods in prospicio
- [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md): a load on
  an expected loss for risk
- [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md):
  pricing from loss history and from exposure
- [`aggregate-distributions.md`](aggregate-distributions.md): the distribution
  of the annual total
- [`severity-and-frequency.md`](severity-and-frequency.md): fitting the size
  and the number of losses
- [`reinsurance-price.md`](reinsurance-price.md): what a reinsurance price is
