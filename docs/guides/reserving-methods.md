# Reserving methods: which one when

<!-- site: execute -->

## Use this guide when

You have a loss triangle and must select a reserving method in prospicio. Which
method answers your question with the data that you have? Typical questions:

- a best estimate for the balance sheet
- a standard error
- a full distribution, for capital or to price a cover on the reserve
- a one-year view, for a solvency standard

Terms:

- The **origin** (accident year) is the year in which the losses occurred.
- The **development age** is the number of months from the start of the origin
  year. Age 12 on accident year 2021 is the valuation date December 2021.
- A **triangle** holds one cumulative value for each origin and age. You observe
  only the upper-left part.
- The **reserve** is the **ultimate** (the final cost of the origin) minus the
  amount paid to date.
- A **link ratio** is the growth of an origin from one age to the next. An
  **age-to-age factor** is an average of the link ratios across origins. A
  **tail factor** gives the development after the oldest age.
- An **increment** is the amount paid in one cell.
- **Bornhuetter-Ferguson (BF)**, **Benktander** and **Cape Cod** mix the data
  with an **a priori**: an expected loss ratio that you set before you look at
  the triangle.
- **Over-dispersed Poisson (ODP)**: the variance of each increment is a scale
  times its mean. **CV**: standard error over mean.
- **Claims development result (CDR)**: the opening ultimate minus the ultimate
  one year later. Positive is favourable, negative is adverse.
- **NUTS** samples a Bayesian posterior. **R-hat** is its convergence check (a
  value near 1 is good).

## Start here

1. **Best estimate.** Check the data (section 1). Run the chain ladder. For
   young origins with small paid amounts, compare it with BF or Cape Cod
   (section 4).
2. **Uncertainty.** For a standard error, use Mack (section 3). For a quantile,
   a cover or capital, use a bootstrap (section 6).
3. **Horizon.** Run-off to the last payment is the lifetime view. The next year
   only, as a solvency standard asks, is the one-year view (section 7).

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

The example is a synthetic paid triangle: seven accident years, annual ages 12
to 84 months and an earned premium for each origin. Money is in thousands of a
synthetic currency.

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

`Triangle.from_long` takes one row for each origin and age. Rules:

- **Ages are months.** Annual data has ages 12, 24, 36. A quarterly triangle
  has ages 3, 6, 9 and needs `development_grain="Q"`. If your data gives the
  valuation date of each cell, use `development_is_valuation=True`.
- **Cumulative is the default.** Use `cumulative=False` for incremental
  payments. An unobserved cell is `nan`. An observed zero stays zero.
- **A segment is a key.** `keys={"line": [...]}` holds several lines in one
  triangle. Each fit then runs on each segment separately.

For data in a pandas frame, `Triangle.from_frame(df, origin="year",
development="age", columns=["paid", "premium"], keys="line")` does the same.

**Exposure columns.** Bornhuetter-Ferguson, Benktander, Cape Cod, expected loss
and Clark's Cape Cod take a second column name, the exposure (for example
premium): `fit(tri, "paid", "premium")`. The library reads the **latest
observed cumulative value** of that column for each origin. For a cumulative
triangle, repeat the premium on each cell of the origin, as above. For an
incremental triangle, the library adds the column over the ages: give the
premium at the first age only and set the later cells to zero, or it counts the
premium once for each age.

### Before you choose: check the data

Examine the link ratios, and compare paid with incurred.

```python
import pandas as pd
from great_tables import GT

lr = tri.link_ratios().to_long()
by_age = {}
for dev, f in zip(lr["development"], lr["paid"]):
    by_age.setdefault(dev, []).append(f)  # one ratio per origin, oldest first
cols = [str(y) for y in years]
ratios = pd.DataFrame(
    [fs + [float("nan")] * (6 - len(fs)) for fs in by_age.values()],
    index=[f"{dev} to {dev + 12}" for dev in by_age],
    columns=cols[:6],
).rename_axis("ages").reset_index()
(GT(ratios, rowname_col="ages")
 .tab_header("Link ratios of the paid triangle", "One column for each origin")
 .fmt_number(cols[:6], decimals=3)
 .sub_missing(missing_text=""))
```

```text
           2018   2019   2020   2021   2022   2023
12 to 24  1.842  1.862  1.858  1.764  1.760  1.767
24 to 36  1.358  1.285  1.351  1.385  1.345       
36 to 48  1.148  1.187  1.207  1.128              
48 to 60  1.105  1.079  1.084                     
60 to 72  1.019  1.060                            
72 to 84  1.017
```

No ratio is an outlier, but the three newest 12 to 24 ratios are below the
three oldest. Faster reporting or a change in payment practice can cause this.
The volume-weighted factor mixes the two regimes: find the cause before you
accept either. The late factors each use one or two ratios; a tail or a
benchmark helps there.

For an outlying link ratio, compare the three averages (section 2). If one large
loss causes it, remove the loss from the triangle and reserve it separately.

**Paid or incurred.** Incurred (paid plus case reserves) develops faster, so it
says more about young origins and long-tailed lines. But it also contains each
change in how the claims department sets case reserves. Run the same calls on
both columns. Here a synthetic incurred triangle has stable case reserving, or
case reserves on the latest diagonal increased by 25%.

```python
case = [0.80, 0.45, 0.25, 0.13, 0.06, 0.03, 0.015]  # case reserve over paid, by age

def incurred_triangle(strengthen):
    vals = [round(v * (1 + case[k] * (strengthen if k == len(row) - 1 else 1.0)))
            for row in paid for k, v in enumerate(row)]
    return Triangle.from_long(origin, age, {"incurred": vals})

def wide(rows, stub, columns):
    """One row for each key of rows (the stub), with the given columns."""
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns).rename_axis(stub).reset_index()

ultimates = {}
for label, strengthen in [("paid", None), ("incurred", 1.0), ("incurred, case +25%", 1.25)]:
    if strengthen is None:
        fit = ChainLadder().fit(tri, "paid")
    else:
        fit = ChainLadder().fit(incurred_triangle(strengthen), "incurred")
    ultimates[label] = list(fit.ultimate) + [sum(fit.ultimate)]
(GT(wide(ultimates, "triangle", cols + ["total"]), rowname_col="triangle")
 .tab_header("Chain ladder ultimate by origin", "Thousands of a synthetic currency")
 .fmt_number(cols + ["total"], decimals=0))
```

```text
                     2018  2019  2020  2021  2022   2023   2024  total
paid                  667   767   762   781   850    974  1,053  5,853
incurred              677   778   773   792   862    989  1,068  5,939
incurred, case +25%   680   787   790   826   924  1,099  1,245  6,350
```

With stable case reserving, the two agree within the case reserve still open at
84 months (5,939 against 5,853), so paid needs a tail. After the increase, the
incurred chain ladder applies the old factors to the stronger cells, and the
youngest ultimate rises from 1,068 to 1,245. The paid triangle does not show the
change. When paid and incurred disagree like this, ask the claims department
before you select either.

## 2. Chain ladder: the default

**What it is.** Each age-to-age factor is an average of the link ratios at that
age. The product of the factors from the age of an origin to the end is its
age-to-ultimate factor (`cdf`). The ultimate is the latest value times it.

**Use it when** the book is mature and stable, payment patterns did not change,
and each origin has a sufficient paid amount.

**Do not use it alone when** the youngest origins have small paid amounts. Their
ultimate is the latest cell times the largest factor, so a noisy first cell goes
directly into the reserve.

**Data it needs.** Only the triangle.

```python
cl = ChainLadder().fit(tri, "paid")
n = len(years)
cl_rows = pd.DataFrame(
    {
        "age": [12 * (n - i) for i in range(n)] + [None],
        "next": [cl.ldf[n - 1 - i] if n - 1 - i < len(cl.ldf) else None for i in range(n)] + [None],
        "cdf": [cl.cdf[n - 1 - i] for i in range(n)] + [None],
        "ultimate": list(cl.ultimate) + [sum(cl.ultimate)],
        "reserve": list(cl.reserve) + [cl.total_reserve],
    },
    index=cols + ["total"],
).rename_axis("origin").reset_index()
(GT(cl_rows, rowname_col="origin")
 .tab_header("Chain ladder, volume-weighted", "Thousands of a synthetic currency")
 .cols_label(age="latest age", next="next factor", cdf="to ultimate")
 .fmt_integer("age")
 .fmt_number(["next", "cdf"], decimals=3)
 .fmt_number(["ultimate", "reserve"], decimals=0)
 .sub_missing(missing_text=""))
```

```text
       latest age  next factor  to ultimate  ultimate  reserve
2018           84                     1.000       667        0
2019           72        1.017        1.017       767       13
2020           60        1.041        1.058       762       42
2021           48        1.089        1.152       781      103
2022           36        1.167        1.344       850      218
2023           24        1.344        1.807       974      435
2024           12        1.804        3.261     1,053      730
total                                           5,853    1,540
```

`average` sets how the link ratios become a factor: `"volume"` (the default)
weights by size, `"simple"` is the plain mean and `"regression"` is least
squares through the origin.

```python
for average in ["volume", "simple", "regression"]:
    print(average, round(ChainLadder(average=average).fit(tri, "paid").total_reserve, 1))
```

```text
volume 1540.4
simple 1542.7
regression 1538.0
```

The three differ by less than 5 in 1,540. They differ more when one origin has
an outlying link ratio: then compare all three.

**Pitfalls.** The chain ladder assumes that the next diagonal develops like the
last ones. A change in claims handling or case reserving makes this false, and
the chain ladder does not show it. The diagonal backtest (section 9) can.

### Tail factors

The oldest age is 84 months and paid still grows (the last factor is more than
1). Without a tail, the model assumes that development stops. `tail=` adds
development after the oldest age. `ChainLadder`, `Mack`, the expected-loss
methods of section 4 and the two bootstraps take it. Clark uses `max_age`; the
Bayesian models take no tail.

| Argument | What it does | Use it when |
|---|---|---|
| `tail=1.02` or `TailConstant(1.02)` | A given factor from the oldest age to ultimate | You have a benchmark |
| `TailCurve()` | Regresses `ln(f - 1)` on the development index 1, 2, 3, ... (`"exponential"`) or on its log (`"inverse_power"`), over the factors above 1, and extrapolates | The factors decay smoothly |
| `TailBondy()` | Each log factor is a constant times the one before it. By default this repeats the last factor | A simple rule when you do not trust a curve |
| `TailLogLinear()` | R ChainLadder's `tail = TRUE` rule | You must match an R `MackChainLadder(tail = TRUE)` |

```python
tails = []
for name, tail in [
    ("none", None),
    ("TailConstant(1.02)", TailConstant(1.02)),
    ("TailCurve()", TailCurve()),
    ("TailBondy()", TailBondy()),
    ("TailLogLinear()", TailLogLinear()),
]:
    fit = ChainLadder(tail=tail).fit(tri, "paid")
    tails.append((name, fit.tail, fit.total_reserve))
(GT(pd.DataFrame(tails, columns=["rule", "tail", "reserve"]), rowname_col="rule")
 .tab_header("Chain ladder with a tail", "Thousands of a synthetic currency")
 .fmt_number("tail", decimals=4)
 .fmt_number("reserve", decimals=1))
```

```text
                      tail  reserve
none                1.0000  1,540.4
TailConstant(1.02)  1.0200  1,657.5
TailCurve()         1.0161  1,634.5
TailBondy()         1.0168  1,638.6
TailLogLinear()     1.0161  1,634.5
```

`TailCurve()` and `TailLogLinear()` agree here because both regress
`ln(f - 1)` on the development index. Their guards differ. `TailLogLinear` is 1
unless the third-last and second-last factors multiply to more than 1.0001, and
it resets a tail above 2 to 1, as R does. A tail adds 94 to 117, about 6% to 8%
of the reserve, so a tail is always an important assumption.

## 3. Mack: the chain ladder with a standard error

**What it is.** Mack's distribution-free model has the chain ladder's reserve
and adds its standard error, in two parts:

- **Process risk** is the randomness of the payments.
- **Parameter risk** is the error in the estimated factors.

**Use it when** a mean and a standard deviation are sufficient (a reserve
range, a CV).

**Do not use it when** you need a quantile of the reserve. A standard error is
not a distribution. For the 99.5% reserve, use a bootstrap (section 6).

```python
mack = Mack().fit(tri, "paid")
tailed = Mack(tail=TailCurve()).fit(tri, "paid")
mack_rows = pd.DataFrame(
    {
        "reserve": list(mack.reserve) + [mack.total_reserve],
        "standard error": list(mack.standard_error) + [mack.total_standard_error],
    },
    index=cols + ["total"],
).rename_axis("origin").reset_index()
(GT(mack_rows, rowname_col="origin")
 .tab_header("Mack, no tail", "Thousands of a synthetic currency")
 .fmt_number(["reserve", "standard error"], decimals=1))
```

```text
       reserve  standard error
2018       0.0             0.0
2019      12.6            20.8
2020      41.8            33.2
2021     103.0            35.5
2022     217.6            47.1
2023     435.1            58.2
2024     730.3            67.4
total  1,540.4           173.6
```

```python
summary = {
    "no tail": mack,
    "fitted tail": tailed,
}
mack_totals = pd.DataFrame(
    [(k, m.total_reserve, m.total_standard_error, m.total_cv, m.total_process_risk, m.total_parameter_risk)
     for k, m in summary.items()],
    columns=["total", "reserve", "standard error", "cv", "process", "parameter"],
)
(GT(mack_totals, rowname_col="total")
 .tab_header("Mack, total reserve", "Thousands of a synthetic currency")
 .fmt_number(["reserve", "standard error", "process", "parameter"], decimals=1)
 .fmt_number("cv", decimals=3))
```

```text
             reserve  standard error     cv  process  parameter
no tail      1,540.4           173.6  0.113     90.8      148.0
fitted tail  1,634.5           204.0  0.125    100.1      177.8
```

Parameter risk exceeds process risk. A tail changes the standard error more
(+17%) than the reserve (+6%), because the tail has its own sigma and standard
error. By default, Mack extrapolates both from the factors; `tail_sigma=` and
`tail_std_err=` replace them.

## 4. Methods that use an a priori: BF, Benktander, Cape Cod, expected loss

The chain ladder gives full weight to the latest cell. These methods mix it with
a **prior** ultimate: the expected loss ratio times the exposure.

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
reserves = {name: list(fit.reserve) + [fit.total_reserve] for name, fit in methods.items()}
(GT(wide(reserves, "method", cols + ["total"]), rowname_col="method")
 .tab_header("Reserve by method and origin", "Thousands of a synthetic currency")
 .fmt_number(cols + ["total"], decimals=0))
```

```text
                      2018  2019  2020  2021  2022  2023  2024  total
chain ladder             0    13    42   103   218   435   730  1,540
expected loss           33   -19    50   134   222   357   622  1,399
Bornhuetter-Ferguson     0    12    42   107   219   400   655  1,436
Benktander (2 steps)     0    13    42   103   218   420   678  1,474
Cape Cod                 0    12    43   108   221   404   661  1,448
```

Expected loss ignores what is paid: it puts a reserve of 33 on the closed year
2018 and a negative reserve on 2019 (70% of premium is less than the amount
already paid). The real difference is the youngest origin, 2024: 730 for the
chain ladder, 655 to 678 with a prior, 622 for expected loss. That is the
actuarial judgment you must make.

**Bornhuetter-Ferguson (BF).** The reserve is the prior ultimate times the
share still to develop. The paid amount to date does not change the reserve
(only the pattern does). This protects against an unusual first cell, but also
ignores a good or bad one.

**Use BF when** the origin is young or the pattern is volatile, and the prior
comes from outside the triangle (a plan, pricing, a benchmark). **Do not use BF
when** the prior is old, or the people under evaluation set it: the method then
repeats the error.

**Benktander** uses the ultimate of BF as the new prior and repeats the
calculation. `n_iters=0` is the expected loss method, `1` is BF, and many steps
give the chain ladder:

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

`n_iters` moves the result between the prior and the data. The default is 1
(BF). Use `n_iters=2`, Benktander's own choice, which is near the
credibility-optimal mix of BF and the chain ladder (Mack 2000). If you want more
steps, use the chain ladder.

**Cape Cod** needs no plan loss ratio. It estimates the loss ratio from the
triangle: the paid amount to date divided by the exposure that has developed.
`trend` adjusts each year's ratio by an annual rate to the valuation date. A
`decay` below 1 gives more weight to nearby years. Use `decay=1` and `trend=0`
only if the years are exchangeable, which they rarely are in a changing market.

```python
cc = CapeCod().fit(tri, "paid", "premium")
cc2 = CapeCod(trend=0.03, decay=0.7).fit(tri, "paid", "premium")
priors = pd.DataFrame(
    {"defaults": cc.apriori, "3% trend, decay 0.7": cc2.apriori}, index=cols
).rename_axis("origin").reset_index()
(GT(priors, rowname_col="origin")
 .tab_header("Cape Cod loss ratio by origin")
 .fmt_number(["defaults", "3% trend, decay 0.7"], decimals=3))
```

```text
      defaults  3% trend, decay 0.7
2018     0.706                0.665
2019     0.706                0.684
2020     0.706                0.695
2021     0.706                0.706
2022     0.706                0.724
2023     0.706                0.749
2024     0.706                0.772
```

**A different prior for each year.** The plan loss ratio often differs by year.
Give `apriori` as a dict keyed by accident year. The premium stays in the
exposure column. Here the plan expects the loss ratio to fall from 70% to 65%
after rate increases:

```python
elr = [0.70, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65]
plan = dict(zip(years, elr))
bf = BornhuetterFerguson(apriori=plan).fit(tri, "paid", "premium")
flat = methods["Bornhuetter-Ferguson"]
plan_rows = pd.DataFrame(
    {
        "plan": elr + [None],
        "plan reserve": list(bf.reserve) + [bf.total_reserve],
        "flat 70% reserve": list(flat.reserve) + [flat.total_reserve],
    },
    index=cols + ["total"],
).rename_axis("origin").reset_index()
(GT(plan_rows, rowname_col="origin")
 .tab_header("Bornhuetter-Ferguson with a prior for each year", "Reserves in thousands of a synthetic currency")
 .cols_label(plan="plan loss ratio")
 .fmt_number("plan", decimals=2)
 .fmt_number(["plan reserve", "flat 70% reserve"], decimals=0)
 .sub_missing(missing_text=""))
```

```text
       plan loss ratio  plan reserve  flat 70% reserve
2018              0.70             0                 0
2019              0.70            12                12
2020              0.69            42                42
2021              0.68           104               107
2022              0.67           209               219
2023              0.66           377               400
2024              0.65           608               655
total                          1,353             1,436
```

Each origin has its own expected ultimate, `apriori[o] * exposure[o]`, for
`ExpectedLoss`, `BornhuetterFerguson` and `Benktander`. A list or array in
origin order (oldest first) also works, but use the keys when you can: a list
can go to the wrong origins without an error. A pandas Series is read by its
index, so `df.set_index("origin")["elr"]` lines up in any row order. In R, give
a vector named by origin: `apriori = c("2018" = 0.70, ...)`.

The older form puts the expected ultimate in a column, uses it as the exposure
and sets `apriori=1.0`. It gives the same reserve. It is still how you give Cape
Cod (and Clark's Cape Cod) a column of relativities between years, because they
estimate the level again. Here the data raise the plan by 1.034, to a reserve of
1,399:

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

**Is the result reasonable?** Before you book a method, compare each origin's
ultimate loss ratio with the plan:

```python
ratios = {name: [u / p for u, p in zip(fit.ultimate, premium)] for name, fit in methods.items()}
ratios["plan"] = elr
(GT(wide(ratios, "method", cols), rowname_col="method")
 .tab_header("Ultimate loss ratio by method and origin")
 .fmt_number(cols, decimals=3))
```

```text
                       2018   2019   2020   2021   2022   2023   2024
chain ladder          0.667  0.730  0.693  0.673  0.696  0.761  0.780
expected loss         0.700  0.700  0.700  0.700  0.700  0.700  0.700
Bornhuetter-Ferguson  0.667  0.730  0.693  0.677  0.697  0.734  0.725
Benktander (2 steps)  0.667  0.730  0.693  0.674  0.697  0.749  0.742
Cape Cod              0.667  0.730  0.693  0.678  0.699  0.736  0.729
plan                  0.700  0.700  0.690  0.680  0.670  0.660  0.650
```

The chain ladder puts 2023 and 2024 at 0.761 and 0.780, above all earlier years
(0.67 to 0.73) and far above the plan. Either the business became worse, or the
first cells of the young origins are high and the chain ladder multiplies them
by its largest factors. This question, not the arithmetic, decides between the
chain ladder and BF for those two years.

## 5. Clark: a growth curve instead of factors

**What it is.** Clark fits a parametric growth curve (`"loglogistic"` or
`"weibull"`) and an expected ultimate for each origin, to the incremental losses
by over-dispersed Poisson maximum likelihood. `ClarkCapeCod` uses one expected
loss ratio times the exposure instead of one ultimate for each origin.

**Use it when** there are few ages, the triangle is noisy from cell to cell, or
you want a smooth pattern with few parameters. The fit gives a standard error
(`total_standard_error`).

**Do not use it without checks.** The curve sets the tail. By default the curve
develops to infinity, and a log-logistic curve has a heavy tail:

```python
clark = []
for label, model in [
    ("loglogistic, to infinity", ClarkLdf()),
    ("loglogistic, stop at 120", ClarkLdf(max_age=120)),
    ("weibull, to infinity", ClarkLdf("weibull")),
    ("weibull, stop at 120", ClarkLdf("weibull", max_age=120)),
]:
    fit = model.fit(tri, "paid")
    clark.append((label, fit.total_reserve, fit.total_standard_error))
clark.append(("Clark Cape Cod, weibull, 120",
              ClarkCapeCod("weibull", 120).fit(tri, "paid", "premium").total_reserve, float("nan")))
(GT(pd.DataFrame(clark, columns=["curve", "reserve", "standard error"]), rowname_col="curve")
 .tab_header("Clark's reserve by curve", "Thousands of a synthetic currency")
 .fmt_number(["reserve", "standard error"], decimals=0)
 .sub_missing(missing_text=""))
```

```text
                              reserve  standard error
loglogistic, to infinity        5,267           1,425
loglogistic, stop at 120        2,407             309
weibull, to infinity            2,588             538
weibull, stop at 120            2,128             289
Clark Cape Cod, weibull, 120    1,985
```

The curve and the cut-off move the reserve from 2,128 to 5,267. The chain ladder
gives 1,540 with no tail and 1,635 with a fitted tail. Check the fit before you
use any of these: the curve gives an age-to-age factor between any two ages,
`fit.growth(a + 12) / fit.growth(a)`, to compare with the chain ladder's.

```python
steps = [f"{a} to {a + 12}" for a in range(12, 84, 12)]
factors = {"chain ladder": list(cl.ldf) + [None]}
for label, model in [("loglogistic, inf", ClarkLdf()), ("weibull, 120", ClarkLdf("weibull", max_age=120))]:
    f = model.fit(tri, "paid")
    end = f.max_age if f.max_age is not None else float("inf")
    factors[label] = [f.growth(a + 12) / f.growth(a) for a in range(12, 84, 12)] + [f.growth(84) / f.growth(end)]
(GT(wide(factors, "curve", steps + ["developed at 84"]), rowname_col="curve")
 .tab_header("Age-to-age factors: chain ladder and Clark's curves")
 .fmt_number(steps, decimals=3)
 .fmt_percent("developed at 84", decimals=0)
 .sub_missing(missing_text=""))
```

```text
                  12 to 24  24 to 36  36 to 48  48 to 60  60 to 72  72 to 84  developed at 84
chain ladder         1.804     1.344     1.167     1.089     1.041     1.017                 
loglogistic, inf     1.903     1.276     1.148     1.096     1.069     1.052              63%
weibull, 120         1.923     1.282     1.147     1.091     1.061     1.043              93%
```

The curves miss the last two factors (log-logistic 1.069 and 1.052, triangle
1.041 and 1.017) and, in the other direction, the first. A two-parameter curve
cannot bend enough for this pattern, so it puts too much development after 84
months (the log-logistic curve says that only 63% is paid at 84 months).

On a triangle this mature, use Clark as a comparison, not as the booked method.
Use it when the triangle has few ages and the factors are too noisy to read one
by one. Set `max_age` to the age at which you think that development stops.

## 6. When you need a distribution, not a number

The methods above give a mean and at most a standard error. A 99.5% capital
figure, an adverse development cover, a risk margin and a reserve range need a
distribution. prospicio has four methods for this: the three below and `Meyers`.
Select one by the variance structure that you think is correct.

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
compare = pd.DataFrame({
    "method": ["ODP bootstrap", "Mack bootstrap", "Mack formula"],
    "mean": [odp.reserves.mean(), mb.reserves.mean(), mack.total_reserve],
    "std dev": [odp.reserves.variance() ** 0.5, mb.reserves.variance() ** 0.5, mack.total_standard_error],
    "99.5%": [odp.reserves.quantile(0.995), mb.reserves.quantile(0.995), None],
})
(GT(compare, rowname_col="method")
 .tab_header("Total reserve by method", "Thousands of a synthetic currency")
 .fmt_number(["mean", "std dev", "99.5%"], decimals=0)
 .sub_missing(missing_text=""))
```

```text
                 mean  std dev  99.5%
ODP bootstrap   1,544      125  1,881
Mack bootstrap  1,541      174  1,981
Mack formula    1,540      174
```

The Mack bootstrap agrees with Mack's formula in mean and standard deviation.
The ODP bootstrap has the same mean and a smaller standard deviation, because
the ODP sets the process variance in proportion to the increment and Mack's
formula in proportion to the cumulative value. These are different assumptions,
and neither is wrong. The knowledge note
[`mack-bootstrap-lifetime-vs-mack.md`](../../knowledge/findings/mack-bootstrap-lifetime-vs-mack.md)
records the reconciliation. At the 99.5% level that a capital model reads, the
two differ by 100.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
bins = list(range(1200, 2300, 25))
ax.hist(odp.reserves.total().draws, bins=bins, histtype="step", linewidth=2, label="ODP bootstrap")
ax.hist(mb.reserves.total().draws, bins=bins, histtype="step", linewidth=2, label="Mack bootstrap")
ax.axvline(mack.total_reserve, color="0.75", linestyle=":")
ax.set_xlabel("total reserve (thousands; dotted: the Mack reserve)")
ax.set_ylabel("simulations out of 10,000")
ax.set_title("Total reserve: the two bootstraps")
ax.legend()
```

Use `MackBootstrap` when the distribution must agree with the standard error of
Mack. Use `OdpBootstrap` for agreement with R `BootChainLadder`, or when a
triangle has a negative increment. Both use the volume-weighted chain ladder by
default and give `fit.reserves`, a joint distribution by origin with `mean()`,
`quantile(p)`, `variance()`, `tvar(p)` and `total()`.

**Read the tail carefully.** A 99.5% quantile from 10,000 simulations uses the
worst 50 of them. Run again with other seeds to see the simulation error:

```python
seeds = []
for seed in [1, 2, 3]:
    o = OdpBootstrap(n_sims=10_000, seed=seed).fit(tri, "paid").reserves
    m = MackBootstrap(n_sims=10_000, seed=seed).fit(tri, "paid").reserves
    seeds.append((str(seed), o.quantile(0.995), m.quantile(0.995)))
(GT(pd.DataFrame(seeds, columns=["seed", "ODP", "Mack"]), rowname_col="seed")
 .tab_header("99.5% of the total reserve by seed", "Thousands of a synthetic currency")
 .fmt_number(["ODP", "Mack"], decimals=0))
```

```text
     ODP   Mack
1  1,881  1,981
2  1,884  1,987
3  1,890  1,984
```

The seeds move the 99.5% by less than 10, far less than the 100 between the two
models. The large errors are in no bootstrap: seven years of residuals are a
small sample from which to learn a tail shape, and model error (the choice of
method and tail, an inflation shock) adds to it. Treat a 99.5% from a triangle
of this size as a central estimate of a very uncertain number.

**Tails.** Use `tail=` and, for a constant, `tail_std_err=`. The bootstrap
refits an estimated tail (`TailCurve`, `TailBondy`, `TailLogLinear`) on each
simulation, which adds its own parameter error. A constant tail varies in
`OdpBootstrap` only with `tail_std_err`; `MackBootstrap` without it extrapolates
the standard error of Mack.

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

The plug-in chain ladder with that tail gives 1,657.5. The one-year standard
deviation is 130, against 127.2 without a tail (section 7).

Pitfall: an estimated tail that the ODP refits on its pseudo factors is
sometimes exactly 1. The ODP bootstrap with `TailLogLinear` is therefore less
stable than a constant with a standard error (the finding
[`bootstrap-tail-parameter-error.md`](../../knowledge/findings/bootstrap-tail-parameter-error.md)).

### Bayesian models

**`BayesOdp`** puts normal priors on the log-scale origin and development
effects of the same ODP model. It samples the posterior with NUTS and gives the
posterior predictive distribution of the reserve. The priors are on the
log-scale effects, not on an ultimate or a loss ratio, so they are hard to state
from actuarial judgment. `intercept_sd`, `origin_sd` and `development_sd` set
their widths (10 by default). It takes a cumulative triangle with no tail and no
exposure column.

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

With the wide default priors, the posterior predictive mean and standard
deviation equal the chain ladder's and the ODP bootstrap's, as a Bayesian fit
with weak priors must give. The finding
[`bayes-odp-vs-odp-bootstrap.md`](../../knowledge/findings/bayes-odp-vs-odp-bootstrap.md)
compares them with Stan.

Always read `fit.diagnostics` (R-hat, effective sample sizes, divergences). When
the sampler fails, the library does not stop: it gives one `UserWarning` (0
here). Do not report a posterior if you did not read its warning.

With its default priors, `BayesOdp` repeats the ODP bootstrap by a different
method. Use it to put a belief about the origin levels or the development
pattern into the model. A smaller `origin_sd`, for example, moves each origin's
level toward the first origin's, because each prior is centred at 0. For a prior
on the loss ratio, which an actuary usually has, use `Meyers`.

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

It needs more data than this example has. On this 7 by 7 triangle the log-normal
ultimates are very uncertain, and the sampler can warn that its diagnostics
fail. Use it on triangles like those of Meyers (10 by 10 with premium), and read
the diagnostics before the reserve. The knowledge note
[`meyers-crc-csr-parity.md`](../../knowledge/findings/meyers-crc-csr-parity.md)
gives the parity checks.

## 7. Lifetime or one-year?

The bootstraps above give the **lifetime** (run-off) view: the uncertainty in
the reserve until the last claim is paid. A solvency standard asks for the
**one-year view**: the uncertainty over the next year, the **claims development
result (CDR)**. It is smaller because it excludes what occurs after next year.

| Question | View | Call |
|---|---|---|
| Reserve range, ADC or LPT pricing, IFRS 17 run-off | Lifetime | `fit()` |
| Solvency II reserve risk, a 99.5% one-year loss | One-year | `claims_development_result()` or `one_year()` |

```python
mw = mack.claims_development_result()
mack_1y = MackBootstrap(n_sims=10_000, seed=1).one_year(tri, "paid", ChainLadder())
odp_1y = OdpBootstrap(n_sims=10_000, seed=1).one_year(tri, "paid", ChainLadder())
bf_1y = MackBootstrap(n_sims=10_000, seed=1).one_year(tri, "paid", BornhuetterFerguson(0.70), exposure="premium")
one_year_rows = pd.DataFrame(
    [
        ("Mack lifetime (formula)", mack.total_standard_error, None),
        ("Merz-Wuthrich one-year", mw.total_one_year_standard_error, None),
        ("Mack bootstrap one-year", mack_1y.cdr.variance() ** 0.5, mack_1y.cdr.mean()),
        ("ODP bootstrap one-year", odp_1y.cdr.variance() ** 0.5, odp_1y.cdr.mean()),
        ("Bornhuetter-Ferguson one-year", bf_1y.cdr.variance() ** 0.5, bf_1y.cdr.mean()),
    ],
    columns=["view", "std dev", "mean CDR"],
)
(GT(one_year_rows, rowname_col="view")
 .tab_header("Reserve risk: lifetime and one-year", "Thousands of a synthetic currency")
 .fmt_number(["std dev", "mean CDR"], decimals=1)
 .sub_missing(missing_text=""))
```

```text
                               std dev  mean CDR
Mack lifetime (formula)          173.6          
Merz-Wuthrich one-year           127.1          
Mack bootstrap one-year          127.2      -1.2
ODP bootstrap one-year           100.5      -1.6
Bornhuetter-Ferguson one-year    102.5     -38.8
```

```python
print("one-year 99.5% loss:", round(-mack_1y.cdr.quantile(0.005)))
```

```text
one-year 99.5% loss: 326
```

How to read the results:

- The one-year standard error of the chain ladder is 127.1 by Merz-Wuthrich,
  against 173.6 for the lifetime. `MackBootstrap().one_year` reproduces it
  (127.2, with a mean CDR of -1.2, about zero).
- The ODP gives 100.5 with the same re-reserving, from its process model. The
  ratio to Merz-Wuthrich differs by triangle (the finding
  [`one-year-bootstrap-vs-merz-wuthrich.md`](../../knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md)).
  Use `MackBootstrap` when the figure must agree with Merz-Wuthrich.
- `one_year` accepts `ChainLadder`, `ExpectedLoss`, `BornhuetterFerguson`,
  `Benktander` or `CapeCod` as `method`, with any averaging, tail or
  development grain. The closed form does not. For the expected-loss methods,
  give `exposure=`.
- The simulated next diagonal always follows the bootstrap's process (here
  Mack's); the method only re-estimates on it. The Bornhuetter-Ferguson row is
  thus the risk of a BF estimate in a chain ladder world. Its mean CDR of -38.8
  is not zero, because BF opens below the chain ladder (1,436 against 1,540) and
  one more diagonal moves it toward the chain ladder. Do not read its 102.5
  against 127.2 as "BF has less reserve risk", and state the offset when a CDR
  with a non-zero mean goes into a capital figure.
- A one-year loss is a negative CDR: `-cdr.quantile(0.005)` is the one-year
  99.5% loss, here 326.
- With a bootstrap `tail`, an origin whose tail step is in the next year takes
  all of the tail in that year, so its one-year risk is an upper bound.

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
lines. Its correlation is that of the lines' residuals, reduced by process
error. Here the second line is half of the first, so the residuals are almost
identical and the correlation is larger than in a real pair of lines.

```python
vals = [v for row in paid for v in row]
two = Triangle.from_long(
    origin * 2,
    age * 2,
    {"paid": vals + [round(0.5 * v) for v in vals]},
    keys={"line": ["property"] * len(vals) + ["liability"] * len(vals)},
)
dep_rows = []
for dependence, extra in [("independent", {}), ("synchronized", {}), ("rank_correlation", {"spearman": [[1, 0.5], [0.5, 1]]})]:
    r = OdpBootstrap(n_sims=5000, seed=2, dependence=dependence, **extra).fit(two, "paid").reserves
    by_line = r.aggregate(["line"])
    dep_rows.append((dependence, r.variance() ** 0.5, r.quantile(0.995),
                     *(by_line.marginal((k,)).variance() ** 0.5 for k in ["property", "liability"])))
(GT(pd.DataFrame(dep_rows, columns=["dependence", "total", "total 99.5%", "property", "liability"]),
    rowname_col="dependence")
 .tab_header("Reserve risk of two lines", "Standard deviation, except the 99.5% of the total")
 .fmt_number(["total", "property", "liability"], decimals=1)
 .fmt_number("total 99.5%", decimals=0))
```

```text
                  total  total 99.5%  property  liability
independent       138.8        2,695     125.0       60.3
synchronized      176.6        2,779     126.0       60.3
rank_correlation  164.5        2,751     125.0       60.3
```

Dependence does not change the standard deviation of a line, but it widens the
total. The triangle does not estimate the Spearman correlation of 0.5: it is an
input. Take it from a regulator's matrix or your own study, then test how much
the total changes with it. `fit.reserves.aggregate(["line"])` gives the reserve
of each line and keeps the dependence between them.

## 9. Which method is right? Backtest it

No method is right in general. One method is right for your book. A **diagonal
backtest** refits the triangle without its newest calendar diagonal, forecasts
that diagonal and compares. An actual-over-expected ratio near 1, with no drift,
supports the assumptions of the method.

The loop below is a chain ladder backtest on an annual triangle. It fits the
cells with a valuation before a calendar year, then compares the next diagonal
with the forecast `latest x age-to-age factor`.

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

ae = {}  # actual over expected, in total and by origin
for year in [2022, 2023, 2024]:
    pairs = backtest(year)
    ae[str(year)] = {"total": sum(a for a, _ in pairs.values()) / sum(f for _, f in pairs.values())}
    ae[str(year)].update({str(oy): a / f for oy, (a, f) in pairs.items()})
ae_cols = ["total"] + cols[1:-1]
(GT(wide({k: [v.get(c, float("nan")) for c in ae_cols] for k, v in ae.items()}, "diagonal", ae_cols),
    rowname_col="diagonal")
 .tab_header("Backtest: actual over expected paid", "By diagonal and accident year")
 .fmt_number(ae_cols, decimals=3)
 .sub_missing(missing_text=""))
```

```text
      total   2019   2020   2021   2022   2023
2022  1.008  1.035  1.025  0.951              
2023  1.004  0.977  1.033  1.041  0.962       
2024  0.994  1.041  0.994  0.955  1.000  0.974
```

```python
fig, ax = plt.subplots()
ax.axhline(1.0, color="0.75")
for year in [2022, 2023, 2024]:
    pairs = backtest(year)
    ax.plot(list(pairs), [a / f for a, f in pairs.values()], marker="o", label=f"diagonal {year}")
ax.set_xticks(years[1:-1])
ax.set_xlabel("accident year")
ax.set_ylabel("actual over expected")
ax.set_title("Backtest: actual over expected paid, by origin")
ax.legend()
```

The totals are within 1% of 1, but the origins show more. In each year the
youngest origin (forecast by the 12 to 24 factor) was below its forecast. This
is the fall in the newest 12 to 24 ratios of section 1: the first factor is
high, which also explains the high 2023 and 2024 loss ratios of section 4. Three
diagonals alone are weak evidence, but with the other two checks they support BF
or Cape Cod on the two youngest origins.

A BF backtest needs its own loop: it forecasts the next increment of each origin
as `prior x (1 / cdf(k + 1) - 1 / cdf(k))`, with the age-to-ultimate factors
`cdf` from the training fit and `k` the current age of the origin.

## 10. From reserve distribution to reinsurance and capital

A reserve distribution is a `PredictiveDistribution`, the type that the pricing
guides use, so it goes directly into a reinsurance structure.

- **Capital.** `reserves.quantile(0.995)` is the 99.5% reserve, not the
  capital: the capital is the amount by which it exceeds the mean (or the booked
  reserve). `reserves.tvar(0.995)` is the tail value at risk. For a one-year
  standard, the capital is `-cdr.quantile(0.005)`.
- **Adverse development cover (ADC).** A stop-loss layer on the total reserve.
  `Tower.apply_aggregate` treats the total of each simulation as one loss and
  applies the layer to it.
- **Loss portfolio transfer (LPT).** A quota share of the full reserve.

```python
from prospicio.reinsurance import Layer, Tower

adc = Tower([Layer.stop_loss("ADC", 300.0, 1700.0)]).apply_aggregate(mb.reserves)
by_kind = adc.aggregate(["kind"])
gross, ceded, net = (by_kind.marginal((k,)) for k in ["gross", "ceded", "net"])
print(f"ADC pays on average {ceded.mean():.1f}, attaches in {1 - ceded.cdf(0.0):.1%} of simulations")
lpt = Tower([Layer.quota_share("LPT", 1.0)]).apply_aggregate(mb.reserves)
print(f"gross 99.5% tail value at risk {mb.reserves.tvar(0.995):.0f}")
print("LPT cedes", round(lpt.aggregate(["kind"]).marginal(("ceded",)).mean()))
print(f"one-year capital, -cdr.quantile(0.005): {-mack_1y.cdr.quantile(0.005):.0f}")
```

```text
ADC pays on average 17.7, attaches in 18.7% of simulations
gross 99.5% tail value at risk 2019
LPT cedes 1541
one-year capital, -cdr.quantile(0.005): 326
```

```python
views = pd.DataFrame(
    [(k, d.mean(), d.quantile(0.995), d.quantile(0.995) - d.mean() if k != "ceded" else None)
     for k, d in [("gross", gross), ("ceded", ceded), ("net", net)]],
    columns=["view", "mean", "99.5%", "capital"],
)
(GT(views, rowname_col="view")
 .tab_header("Lifetime reserve with the ADC 300 xs 1,700", "Capital: the 99.5% less the mean")
 .fmt_number("mean", decimals=1)
 .fmt_number(["99.5%", "capital"], decimals=0)
 .sub_missing(missing_text=""))
```

```text
          mean  99.5%  capital
gross  1,541.3  1,981      440
ceded     17.7    281         
net    1,523.7  1,700      176
```

The ADC caps the net 99.5% at 1,700 and cuts the lifetime capital from 440 to
176. The one-year capital of the same reserve is 326 (section 7). These figures
exclude discounting and unallocated loss adjustment expenses; an IFRS 17 measure
or the price of a cover adds them.

The LPT cedes all 1,541 on average. Its price starts from that mean, adds a load
for the risk and subtracts a discount for the time to pay. Use
`prospicio.pricing` for both:

1. `price(ceded, Distortion.tvar(0.995), cost_of_capital=...)` on the ceded
   marginal of the LPT gives a one-period price (`Distortion` is in
   `prospicio.risk`).
2. `price_over_payout(that_price, pattern, rate=..., cost_of_capital=...)`
   gives the price over a payment pattern: it discounts the losses and charges
   the cost of capital (a spread over the discount rate) on the capital, which
   falls with the share of the loss still unpaid. Give it the same cost of
   capital or distortion as step 1.

A reserve on the books needs its run-off pattern from the valuation date, not
the pattern of a new origin. Make it from the paid chain ladder: for each
calendar year `j` after the valuation, add the expected payments of all origins,
`ultimate x (1 / cdf(k + j) - 1 / cdf(k + j - 1))` (`cdf(k)` is the factor at
the current age of the origin), divide by the total reserve, and give the times
`1, 2, ...` and these shares to `PaymentPattern(times, shares)`.
`PaymentPattern.from_development([age / 12 for age in tri.development], cl.cdf)`
gives instead the pattern of a new accident year from its start (the last factor
must be 1), for a prospective cover. [`reinsurance-price.md`](reinsurance-price.md)
gives an example on a layer, and the time value bullet in
[`risk-loads-and-distortions.md`](risk-loads-and-distortions.md) explains the
calculation over the payout.

To price the cover, add a load for the reinsurer's risk
([`risk-loads-and-distortions.md`](risk-loads-and-distortions.md);
[`reinsurance-price.md`](reinsurance-price.md) shows the parts of a price). The
expected loss of an ADC is sensitive to the tail of the reserve distribution:
the bootstrap `tail` and the choice of Mack's or the ODP's process change it.

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
