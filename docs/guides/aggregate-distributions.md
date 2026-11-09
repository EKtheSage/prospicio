# Aggregate distributions: FFT, Panjer, simulation or moments

<!-- site: execute -->

Use this guide to compute the distribution of the total loss of a year (the
**aggregate loss**), or of what one reinsurance contract pays out of it.
prospicio has four engines: closed-form **moments**, **FFT** (fast Fourier
transform on a grid), **Panjer**'s recursion and **Monte Carlo simulation**.
Each fails in its own way. The guide says which engine to use, how to size and
check it, and the exact calls. It is for an actuary who knows frequency and
severity from primary pricing and is new to reinsurance pricing. Money is in
millions. CI runs the code and checks every printed number.

## The decision table

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| Mean, standard deviation and skewness of the year, or a sanity check | Moments | `compound_moments` | Closed form, no grid, no sampling error. Its fitted distributions are rough in the tail |
| Quantiles, stop-loss prices or any tail number for one book | FFT, grid sized for you | `fft_auto` | Exact for the discretized problem, error reported |
| The same, with a bucket you must use (a policy limit, a market convention) | FFT on your grid | `Grid.local_moment` then `fft` | You control the step. `local_moment` keeps the mean |
| An independent exact check of an FFT result | Panjer | `panjer` | Same answer by a different algorithm. No aliasing |
| A large claim count (a Poisson mean of several hundred or more) | FFT | `fft_auto` | Panjer fails once the probability of a zero year underflows: a Poisson mean of 745 runs, 800 does not (section 4) |
| Claims that cluster (contagion, or an uncertain claim rate) | FFT with a mixed Poisson count | `fft_auto(Count.mixed_poisson(...), sev)` | A more variable count gives a fatter annual tail. Panjer refuses it (section 3.6) |
| A policy limit or deductible on each claim | FFT on a mapped grid | `Grid.local_moment`, `Grid.map`, then `fft` | `fft_auto` sizes for the severity it gets (section 3.7) |
| A count that is not Poisson, negative binomial or binomial | FFT | `fft` with a `Count` | Panjer needs the count families of the recursion |
| Heavy tail with infinite variance | FFT, grid sized by one big claim | `fft_auto` | Moments are undefined. The sizing does not need them |
| What an excess layer, quota share, stop loss, reinstatements or a corridor pay | FFT on the tower | `Tower.on_grid` | Exact, no sampling error |
| A quota share or stop loss (with its commissions) on an annual loss already on a grid, such as a loss ratio times premium | The tower on the annual grid | `Tower.on_annual_grid` | The total of each year is one loss. No claim count is needed, and the expected commissions are exact |
| Reinstatement premium pro rata as to time, a surplus treaty, annual terms that inure to a later layer | Simulation | `Tower.apply` on an `EventSet` | They need event order, dates or sums insured. A compound grid has none |
| Several layers or lines together (joint distribution, dependence) | Simulation | `Tower.apply`, `PredictiveDistribution` | A grid is one marginal distribution |
| The event loss table of a catastrophe model | Simulation | `events, rows = EventLossTable(...).simulate(n_years, seed)` | Returns the years and the row in the table of each loss. Draws the loss of each event from its secondary uncertainty. Exact OEP and PML from `EventLossTable.oep` and `EventLossTable.occurrence_pml` |
| Your own simulated years (a year loss table), a non-cat set beside a cat set, an hours clause | Simulation | `EventSet.from_long` or `EventSet.from_years`, `EventSet.merge`, `EventSet.grouped(window_hours)` | The events are the model. `merge` needs the same number of years in both sets. `grouped` needs loss times (fractions of a year) and drops sums insured |
| The largest single loss of the year (occurrence PML) | Closed form | `occurrence_pml`, `occurrence_exceedance` | For per-occurrence covers the annual total is the wrong curve |
| How often a layer is hit, its loss on line, its count and severity moments | Closed form | `occurrence_layer_statistics` | One call |

One rule covers most cases: **use FFT unless a contract term forces simulation,
and read the report either way.**

## 1. The running book

The running book is synthetic: six claims a year on average (Poisson) and
lognormal claim sizes with mean 1 and coefficient of variation 2. We want the
annual distribution and the price of a per-occurrence layer of 5 xs 5 (a limit
of 5 above an attachment of 5).

```python
from prospicio.aggregate import (
    compound_moments, fft, fft_auto, occurrence_exceedance, occurrence_pml,
    panjer, recommend_grid, round_bucket, simulate_events,
)
from prospicio.distributions import Count, Grid, Lognormal, Pareto, Poisson
from prospicio.pricing import CollectiveModel, Mbbefd, RiskProfile
from prospicio.reinsurance import Layer, Tower, occurrence_layer_statistics

freq = Poisson(6.0)
sev = Lognormal.from_mean_cv(1.0, 2.0)

annual, report, size = fft_auto(freq, sev, 12)
print(f"step {size.step}, points {size.points}, sized by {size.method}")
print(f"mean {annual.mean():.4f}, sd {annual.variance() ** 0.5:.4f}")
print(f"99%: {annual.quantile(0.99):.2f}, 99.5%: {annual.quantile(0.995):.2f}")
```

```text
step 0.0625, points 4096, sized by single_big_jump
mean 5.9999, sd 5.4683
99%: 25.81, 99.5%: 31.56
```

The `12` caps the grid at `2**12` points (default `2**16`, a finer step). The 99%
value is the annual loss exceeded with probability 1%.

For a Poisson count the expected annual loss to a layer is the expected claim
count times the layer mean of the severity. `CollectiveModel` gives this closed
form, and later sections compare every engine with it.

```python
model = CollectiveModel(freq, sev)
exact = model.layer_mean(5.0, 5.0)  # limit, then attachment
print(f"expected annual loss to 5 xs 5: {exact:.5f}")
print(f"standard deviation: {model.layer_std(5.0, 5.0):.4f}")
```

```text
expected annual loss to 5 xs 5: 0.43812
standard deviation: 1.3018
```

## 2. Moments: when closed form is enough

`compound_moments` gives the mean, variance, skewness and third cumulant of the
year in closed form, with no grid and no sampling error. `.approximate(kind)`
fits a distribution to them.

```python
m = compound_moments(freq, sev)
print(f"mean {m.mean:.3f}, cv {m.cv:.4f}, skew {m.skew:.4f}")
```

```text
mean 6.000, cv 0.9129, skew 4.5644
```

```python
import pandas as pd
from great_tables import GT

rows = []
for kind in ["normal", "gamma", "lognormal", "shifted_gamma", "shifted_lognormal"]:
    fit = m.approximate(kind)
    rows.append({"fit": kind, "q99": fit.quantile(0.99), "q995": fit.quantile(0.995),
                 "err99": fit.quantile(0.99) / annual.quantile(0.99) - 1,
                 "err995": fit.quantile(0.995) / annual.quantile(0.995) - 1})
rows.append({"fit": "FFT", "q99": annual.quantile(0.99), "q995": annual.quantile(0.995),
             "err99": 0.0, "err995": 0.0})
(GT(pd.DataFrame(rows), rowname_col="fit")
 .tab_header("Moment fits against the FFT", "Annual loss at two quantiles, running book")
 .tab_spanner("annual loss (money units)", ["q99", "q995"])
 .tab_spanner("error against the FFT", ["err99", "err995"])
 .cols_label(q99="99%", q995="99.5%", err99="99%", err995="99.5%")
 .fmt_number(["q99", "q995"], decimals=2)
 .fmt_percent(["err99", "err995"], decimals=1))
```

```text
                     99%  99.5%     99%   99.5%
normal             18.74  20.11  -27.4%  -36.3%
gamma              25.24  28.82   -2.2%   -8.7%
lognormal          27.11  32.92    5.0%    4.3%
shifted_gamma      30.63  37.50   18.6%   18.8%
shifted_lognormal  27.59  34.16    6.9%    8.2%
FFT                25.81  31.56    0.0%    0.0%
```

**Use moments** for the mean and spread of a year: a quick load, a check on a
grid or a simulation, or the input to a rule that needs only a standard
deviation. **Do not use the fits for capital or layer prices.** A fit matches
three numbers and says nothing else about the shape, and no family is right at
both quantiles.

**Data needed**: any count, and a severity with a finite third moment (a heavy
tail has none, section 3.5). The shifted fits also need positive skewness.

## 3. FFT: the default engine

An FFT turns the convolution of `N` severities into a product in the Fourier
domain, so the whole distribution costs about `n log n` for an `n`-point grid.
It works for any count with a probability generating function and for large
counts. It is **exact for the discretized problem**; the approximations are
the discretization, the finite grid and aliasing.

### 3.1 Let the library size the grid

`recommend_grid` sizes the grid as the Python package `aggregate` does, and
`fft_auto` sizes it so, then runs the FFT. The step is on a ladder of values
(`round_bucket`).

```python
print(size)
print(f"extent {size.extent:.1f}, grid reaches {size.step * size.points:.1f}")
print([round_bucket(x) for x in (0.3, 0.7, 3.4, 6.0, 30.0)])
```

```text
GridSize(step=0.0625, points=4096, method="single_big_jump", tail_estimate=1.6733116333700584e-6)
extent 167.9, grid reaches 256.0
[0.5, 1.0, 4.0, 8.0, 40.0]
```

- `extent` is how far the grid must reach. `step` is `extent` over the points,
  rounded up the ladder.
- `method` says what set the extent: `"moments"` (a lognormal or gamma fitted to
  the mean and variance of the year) or `"single_big_jump"` (one big claim on
  top of a typical year, which governs heavy tails).
- `tail_estimate` is roughly the probability beyond the grid from one claim.
  If it is not tiny, increase `log2`.

### 3.2 Read the report every time

`fft` and `panjer` also return a `CompoundReport`. Read it in this order:

| Field | Meaning | Act when |
|---|---|---|
| `aliasing_error` | Largest change in any probability when the FFT buffer doubles | Above about `1e-9` (a rule of thumb, not a bound): the grid is unreliable, `tail_mass` included. Lengthen the grid |
| `tail_mass` | Probability beyond the last point, put on that point | Not small beside the tail you price (a 99.5% quantile needs it well under 0.5%) |
| `mean_error()` | Grid mean minus `E[N] E[X]` on the severity grid | Not tiny: lost tail or aliasing |
| `points` | Length of the result | Shorter than you meant |

A grid that is too short shows at once (same book and step):

```python
def tiny(x):
    """Print a residual, or '<1e-10' when it is rounding noise that differs by platform."""
    return f"{x:.1e}" if x >= 1e-10 else "<1e-10"


rows = []
for n in (128, 256, 512, 1024, 4096):
    g, disc = Grid.local_moment(sev, 0.0625, n)
    a, r = fft(freq, g, n)
    rows.append({"points": f"{n:,}", "tail_mass": f"{r.tail_mass:.1e}", "aliasing": tiny(r.aliasing_error),
                 "mean_error": f"{r.mean_error():+.1e}", "grid_mean": a.mean()})
(GT(pd.DataFrame(rows), rowname_col="points")
 .tab_header("The report of a growing grid", "Same book and step 0.0625; the exact mean is 6")
 .cols_label(tail_mass="tail mass", aliasing="aliasing", mean_error="mean error", grid_mean="grid mean")
 .fmt_number("grid_mean", decimals=3))
```

```text
       tail mass  aliasing  mean error  grid mean
128      2.4e-01   4.5e-04    -8.4e-01      4.763
256      4.6e-02   1.3e-05    -1.9e-01      5.674
512      4.8e-03   1.1e-07    -2.5e-02      5.940
1,024    4.0e-04   6.7e-10    -2.3e-03      5.991
4,096    1.9e-06    <1e-10    -1.1e-05      6.000
```

Read `aliasing_error` first: when it is large, wrapped mass sits inside the
grid and `tail_mass` understates the truth. Passing it is not enough: the
1024-point row has aliasing 6.7e-10 and still 4.0e-4 of probability beyond the
grid, too much for a 99.9% quantile. Both tests must pass.

### 3.3 Choosing the discretizer

When you choose the grid, three discretizers turn a continuous severity into
grid probabilities. Each returns the grid and a `DiscretizationReport`.

| Call | What it does | Use for |
|---|---|---|
| `Grid.local_moment(sev, h, n)` | Splits the mass of each cell between its two neighbouring points, so it keeps the mean | Pricing. Means and layer means are right even at a coarse step |
| `Grid.rounding(sev, h, n)` | Moves each loss to the nearest point | Matching the reports of `aggregate`. Fine at a small step. `fft_auto` uses it |
| `Grid.lower(sev, h, n)` | Moves the mass of each cell to its left end | A stochastic lower bound: it understates every loss, layer and quantile, so never use it for a price |

The same book at three steps (exact mean 6, exact 5 xs 5 value 0.43812):

```python
tower = Tower([Layer("5x5", 5.0, 5.0)])
rows = []
for step, n in ((1.0, 128), (0.25, 512), (0.0625, 2048)):
    for name in ("rounding", "lower", "local_moment"):
        g, _ = getattr(Grid, name)(sev, step, n)
        grids = tower.on_grid(freq, g, n)
        rows.append({"step": f"{step}", "discretizer": name, "gross": grids.gross.mean(),
                     "layer": grids.ceded[0].mean(), "q99": grids.gross.quantile(0.99)})
(GT(pd.DataFrame(rows), rowname_col="step", groupname_col=None)
 .tab_header("Three discretizers at three steps", "Gross mean, 5 xs 5 layer mean and 99% quantile, money units")
 .cols_label(discretizer="discretizer", gross="gross mean", layer="5x5 layer", q99="99% quantile")
 .fmt_number(["gross", "layer"], decimals=4)
 .fmt_number("q99", decimals=2))
```

```text
         discretizer  gross mean  5x5 layer  99% quantile
1.0         rounding      5.5738     0.4360         25.00
1.0            lower      3.7553     0.3783         23.00
1.0     local_moment      5.9987     0.4381         26.00
0.25        rounding      5.9775     0.4380         25.75
0.25           lower      5.2739     0.4224         24.75
0.25    local_moment      5.9987     0.4381         25.75
0.0625      rounding      5.9987     0.4381         25.81
0.0625         lower      5.8111     0.4341         25.56
0.0625  local_moment      5.9987     0.4381         25.81
```

`local_moment` has the mean right at every step; its shortfall below 6 is the
tail beyond the last grid point. `lower` is a true lower bound and converges
slowly. A grid quantile is a grid point, so quantiles move with the step.

`local_moment` keeps the mean, not the variance: splitting mass between two
points adds spread. At a coarse step its quantiles and stop-loss premiums are a
little high (26.00 at step 1.0 against 25.81 at the finest). Before you price a
stop loss or quote a tail number, pass the halving check below.

### 3.4 Is the step fine enough? Halve it

The library reports only the discretization error of the mean. To check the
step, halve it, double the points and see if your number moves.

```python
rows, prev = [], None
for step, n in ((0.25, 512), (0.125, 1024), (0.0625, 2048), (0.03125, 4096)):
    g, _ = Grid.local_moment(sev, step, n)
    a, r = fft(freq, g, n)
    q = a.quantile(0.995)
    rows.append({"step": f"{step}", "q995": q, "change": None if prev is None else q - prev,
                 "stop_loss": a.stop_loss(30.0)})
    prev = q
(GT(pd.DataFrame(rows), rowname_col="step")
 .tab_header("Halving the step", "99.5% quantile and the stop-loss premium at 30, money units")
 .cols_label(q995="99.5%", change="change", stop_loss="stop loss at 30")
 .fmt_number("q995", decimals=4)
 .fmt_number("change", decimals=3, force_sign=True)
 .fmt_number("stop_loss", decimals=5)
 .sub_missing(missing_text=""))
```

```text
           99.5%  change  stop loss at 30
0.25     31.5000                  0.06879
0.125    31.6250  +0.125          0.06877
0.0625   31.5625  -0.062          0.06877
0.03125  31.5938  +0.031          0.06877
```

A quantile moves one step at a time, so also judge a smooth number (a stop-loss
premium, a layer mean). Here the quantile stays within one step, and the
premium is stable to four digits from the first halving.

### 3.5 Heavy tails

A Pareto with `alpha` 1.5 has a finite mean and an infinite variance. Moments
refuse it, and one big claim sets the size of the grid:

```python
heavy = Pareto(0.5, 1.5)  # threshold 0.5, alpha 1.5
try:
    compound_moments(freq, heavy)
except ValueError as e:
    print(e)
print(recommend_grid(freq, heavy, 12))
heavy_annual, heavy_report, _ = fft_auto(freq, heavy, 12)
print(f"tail mass {heavy_report.tail_mass:.1e}, aliasing {tiny(heavy_report.aliasing_error)}")
```

```text
invalid data: the severity's first three moments must be finite
GridSize(step=1.0, points=4096, method="single_big_jump", tail_estimate=8.092194914091655e-6)
tail mass 8.1e-06, aliasing <1e-10
```

A power-law tail never ends, so some mass is always beyond the grid. If the
report shows too much, increase `log2` or truncate the severity (`Pareto` takes
a truncation point). With `alpha` below 1 the mean is infinite and the sizing
refuses:

```python
try:
    fft_auto(freq, Pareto(0.5, 0.9), 12)
except ValueError as e:
    print(e)
```

```text
invalid parameter severity = inf: the aggregate mean must be finite and positive; choose the step by hand
```

A real book with such a tail has a policy limit (section 3.7).

### 3.6 Claims that cluster: a mixed Poisson count

One event can cause many claims (contagion), or the claim rate can be uncertain
from year to year, so the count is more variable than a Poisson. A **mixed
Poisson** count multiplies the rate by a random factor of mean 1; with the
default gamma mixing, `Count.mixed_poisson(mean, cv)` is the negative binomial.
FFT takes it directly:

```python
mixed = Count.mixed_poisson(6.0, 0.5)  # rate factor with cv 0.5
mixed_annual, mixed_report, _ = fft_auto(mixed, sev, 12)
print(f"tail mass {mixed_report.tail_mass:.1e}, aliasing {tiny(mixed_report.aliasing_error)}")
```

```text
tail mass 1.9e-06, aliasing <1e-10
```

```python
rows = [{"count": label, "var": c.variance(), "mean": a.mean(),
         "q99": a.quantile(0.99), "q995": a.quantile(0.995)}
        for label, c, a in (("Poisson", freq, annual), ("mixed Poisson", mixed, mixed_annual))]
(GT(pd.DataFrame(rows), rowname_col="count")
 .tab_header("Poisson against a mixed Poisson count", "Same mean count of 6; annual loss in money units")
 .cols_label(var="count variance", mean="annual mean", q99="99%", q995="99.5%")
 .fmt_number("var", decimals=1)
 .fmt_number(["mean"], decimals=3)
 .fmt_number(["q99", "q995"], decimals=2))
```

```text
               count variance  annual mean    99%  99.5%
Poisson                   6.0        6.000  25.81  31.56
mixed Poisson            15.0        6.000  28.56  34.44
```

The mean is the same, and the 99.5% is higher. Claim sizes are still
independent of each other and of the count. Dependence between size and number
needs simulation or a scenario.

### 3.7 A policy limit or deductible on each claim

`fft_auto` sizes its grid for the severity it gets. To cap each claim (or take
off a deductible), discretize the unlimited severity yourself, then apply the
term with `Grid.map`, which keeps the mean exactly. Here the cap is 10:

```python
g, _ = Grid.local_moment(sev, 0.0625, 4096)
capped, on_points = g.map(lambda x: min(x, 10.0))
limited, limited_report = fft(freq, capped, 4096)
print(f"limited claim mean: grid {capped.mean():.4f}, exact {sev.lev(10.0):.4f}, on points {on_points}")
print(f"annual mean {limited.mean():.4f}, 99.5% {limited.quantile(0.995):.2f},"
      f" tail mass {tiny(limited_report.tail_mass)}")
```

```text
limited claim mean: grid 0.9527, exact 0.9527, on points True
annual mean 5.7162, 99.5% 21.81, tail mass <1e-10
```

The grid's limited mean equals `sev.lev(10.0)`, and the cap pulls the 99.5% from 31.56 to 21.81.
Put the limit on a grid point (here 10 is a multiple of the step): then `on_points` is `True` and the shape,
not only the mean, is exact. A deductible is `lambda x: max(x - d, 0.0)`.

## 4. Panjer: the exact check

Panjer's recursion computes the aggregate probabilities one point at a time. It
is exact for the discretized problem and has no aliasing. Its work grows with
the square of the grid length, against `n log n` for the FFT.

```python
g, _ = Grid.local_moment(sev, 0.0625, 4096)
a_panjer, r_panjer = panjer(freq, g, 4096)
a_fft, r_fft = fft(freq, g, 4096)
gap = max(abs(x - y) for x, y in zip(a_panjer.probs, a_fft.probs))
print(f"largest difference between Panjer and FFT: {tiny(gap)}")
print(f"aliasing error: Panjer {r_panjer.aliasing_error}, FFT {tiny(r_fft.aliasing_error)}")
```

```text
largest difference between Panjer and FFT: <1e-10
aliasing error: Panjer 0.0, FFT <1e-10
```

**Use Panjer** as an independent check of an FFT result on a small problem.

**Do not use it** for large counts or long grids. It needs a count from the
families of the recursion: Poisson, negative binomial, binomial, and their
zero-modified and zero-truncated versions (`Count.zero_truncated`). It refuses
mixed Poisson counts and the Neyman type A. It fails for a large Poisson mean
because the probability of a zero year, `exp(-mean)`, underflows:

```python
panjer(Poisson(745.0), g, 4096)
print("Poisson 745: Panjer runs")
big = Poisson(800.0)
try:
    panjer(big, g, 4096)
except ValueError as e:
    print(e)
pig = Count.mixed_poisson(10.0, 0.5, mixing="inverse_gaussian")
try:
    panjer(pig, g, 4096)
except ValueError as e:
    print(e)
_, big_report = fft(big, g, 4096)
print(f"FFT on the same grid: tail_mass {big_report.tail_mass:.3f}")
big_annual, big_report, big_size = fft_auto(big, sev, 16)
print(f"fft_auto: step {big_size.step}, points {big_size.points}, mean {big_annual.mean():.1f},"
      f" tail mass {big_report.tail_mass:.1e}")
```

```text
Poisson 745: Panjer runs
invalid parameter frequency = 800: P(S = 0) underflows for this claim count; use FFT
invalid data: Panjer's recursion needs a count in the (a, b, 1) class; use FFT for this one
FFT on the same grid: tail_mass 0.998
fft_auto: step 0.03125, points 65536, mean 800.0, tail mass 1.7e-07
```

With 800 claims a year the annual mean is 800, far past a grid that reaches 256.
`fft` ran with no error: a `tail_mass` near 1 is the only warning (section 3.2).
`fft_auto` sizes the grid for this book.

## 5. Simulation

`simulate_events` draws many years (a claim count, then that many sizes) and
keeps the individual losses in an `EventSet`, so per-loss terms (layers, sums
insured, dates) apply afterwards.

```python
events = simulate_events(freq, sev, 20_000, 1)
totals = events.totals()
print(events)
print(f"mean {totals.mean():.3f} (exact 6), 99.5% quantile {totals.quantile(0.995):.2f}"
      f" (FFT {annual.quantile(0.995):.2f})")
```

```text
EventSet(n_sims=20000, seed=1)
mean 5.960 (exact 6), 99.5% quantile 30.57 (FFT 31.56)
```

**Use simulation** when the answer depends on the order, timing or identity of
losses, or for several results at once (section 6), or when no grid can hold
the problem: a catastrophe model's event loss table (`EventLossTable`) or a
severity that exists only as simulated losses. **Do not use it** for a tail
quantile of a single book when FFT applies: the error is large and shrinks
slowly.

### 5.1 Sampling error and seeds

A simulated number is random, and its error shrinks with the square root of the
years. Ten seeds at two sizes, as average and standard deviation across seeds:

```python
import statistics

rows = []
for n in (10_000, 100_000):
    runs = [simulate_events(freq, sev, n, seed).totals() for seed in range(1, 11)]
    qs = [t.quantile(0.995) for t in runs]
    ms = [t.mean() for t in runs]
    rows.append({"years": f"{n:,}", "q": statistics.mean(qs), "q_sd": statistics.stdev(qs),
                 "m": statistics.mean(ms), "m_sd": statistics.stdev(ms)})
(GT(pd.DataFrame(rows), rowname_col="years")
 .tab_header("Sampling error over ten seeds", f"The FFT gives {annual.quantile(0.995):.2f} for the 99.5% quantile and 6 for the mean")
 .tab_spanner("99.5% quantile", ["q", "q_sd"])
 .tab_spanner("mean", ["m", "m_sd"])
 .cols_label(q="average", q_sd="sd", m="average", m_sd="sd")
 .fmt_number(["q", "q_sd"], decimals=2)
 .fmt_number(["m", "m_sd"], decimals=3))
```

```text
         average    sd  average     sd
10,000     31.55  1.42    6.001  0.047
100,000    31.68  0.51    6.004  0.016
```

Ten times the years cut the error about threefold, not tenfold. A tail quantile
is the noisiest number there is.

```python
import matplotlib.pyplot as plt

draws = [row[0] for row in runs[0].draw_matrix()]
xs = [annual.step * i for i in range(len(annual.probs))]
density = [p / annual.step for p in annual.probs]
fig, ax = plt.subplots()
ax.hist(draws, bins=[0.5 * i for i in range(81)], density=True, color="0.75",
        label="simulation, 100,000 years")
ax.plot(xs, density, label="FFT")
ax.set_xlim(0, 40)
ax.set_xlabel("annual loss (money units)")
ax.set_ylabel("density")
ax.set_title("Annual loss of the running book: FFT against simulation")
ax.legend()
```

A standard error from the sample variance (section 6) assumes a finite
variance. For the Pareto of section 3.5 the variance is infinite, the error bar
understates the noise and the mean settles slowly. Prefer the grid there.

The **seed** fixes the draws, for any number of threads
(`docs/design/aggregate.md`, tested in the Rust crate). Each year has its own
stream, so you can replay year `i` alone. Record the seed with every result.

```python
a = simulate_events(freq, sev, 1000, 5)
b = simulate_events(freq, sev, 1000, 5)
c = simulate_events(freq, sev, 1000, 6)
print(a.totals().mean() == b.totals().mean(), a.totals().mean() == c.totals().mean())
print(a.events(0) == b.events(0), a.seed)
```

```text
True False
True 5
```

`CollectiveModel(freq, sev).simulate(n, seed)` gives the same years from the
pricing object. Other sources of years:

- `RiskProfile.simulate` draws years from a property risk profile (section 6.2).
- `EventLossTable.simulate` draws years from a catastrophe model's event loss
  table and returns the row of each loss.
- `EventSet.from_years` loads a list of losses for each year.
- `EventSet.from_long` loads one row for each loss with its year, the form of a
  year loss table.

A year loss table lists only the years with a loss. Give `from_long` the full
`n_years`, including the years with no loss, or every frequency and loss cost
comes out too high.

## 6. What each treaty pays

Two engines compute what a contract pays: `Tower.on_grid` (exact FFT) and
`Tower.apply` (simulation).

### 6.1 Exactly on the grid

`Tower.on_grid(freq, severity_grid, points)` applies each layer's
per-occurrence recovery to the severity grid, compounds the result with the
same count, then applies the annual terms (annual deductible and limit,
reinstatements, corridor) and the share to the annual total. It returns gross,
each layer's ceded loss and, where it is one compound total, net.  There is no sampling error.

```python
program = Tower([
    Layer("5x5", 5.0, 5.0, reinstatements=1),
    Layer("10x10", 10.0, 10.0),
    Layer.quota_share("QS", 0.25),
])
g, _ = Grid.local_moment(sev, 0.0625, 4096)
grids = program.on_grid(freq, g, 4096)
rows = [{"layer": name, "mean": ceded.mean(), "q99": ceded.quantile(0.99)}
        for name, ceded in zip(program.layer_names, grids.ceded)]
(GT(pd.DataFrame(rows), rowname_col="layer")
 .tab_header("Ceded loss by layer, on the grid", "Mean and 99% quantile of the annual ceded loss, money units")
 .cols_label(mean="mean", q99="99%")
 .fmt_number("mean", decimals=4)
 .fmt_number("q99", decimals=3))
```

```text
         mean    99%
5x5    0.4377  5.000
10x10  0.1956  8.562
QS     1.5000  6.453
```

```python
print("net available:", grids.net is not None, " on points:", grids.on_points)
plain = Tower([Layer("5x5", 5.0, 5.0)]).on_grid(freq, g, 4096)
stop_loss_last = Tower.inuring([[Layer("5x5", 5.0, 5.0)], [Layer.stop_loss("SL", 10.0, 10.0)]])
print("net without annual terms:", plain.net is not None,
      " net with a stop loss last:", stop_loss_last.on_grid(freq, g, 4096).net is not None)
```

```text
net available: False  on points: True
net without annual terms: True  net with a stop loss last: True
```

`net` is `None` because the 5x5 has an annual limit (from the reinstatement),
so net would depend on several annual totals at once. Net exists when no layer
has annual terms, or when the last stage is a single aggregate cover such as a
stop loss. `on_points` says whether every value fell on a grid point; if not,
means are still exact but shapes smear by up to one step.

The same tower by simulation:

```python
sim = program.apply(simulate_events(freq, sev, 100_000, 3))
rows = []
for name, ceded in zip(program.layer_names, grids.ceded):
    s = sim.marginal(("ceded", name))
    rows.append({"layer": name, "grid": ceded.mean(), "sim": s.mean(),
                 "se": s.variance() ** 0.5 / len(s) ** 0.5})
(GT(pd.DataFrame(rows), rowname_col="layer")
 .tab_header("Expected ceded loss: grid against simulation", "Money units, 100,000 simulated years")
 .cols_label(grid="grid", sim="simulated", se="standard error")
 .fmt_number(["grid", "sim", "se"], decimals=4))
```

```text
         grid  simulated  standard error
5x5    0.4377     0.4360          0.0041
10x10  0.1956     0.1952          0.0038
QS     1.5000     1.4985          0.0044
```

Publish the grid answer, which has no noise. Simulation is the check.

### 6.2 What forces simulation

A compound distribution does not record which loss came first, when it
happened or how big the insured risk was, so terms that need those facts cannot
run on the grid. `on_grid` refuses the first three "No" rows with an error.

| Contract term | On the grid? | Why |
|---|---|---|
| Per-occurrence layer, share, quota share, stop loss | Yes | A function of each loss or of the annual total |
| Annual deductible and limit, reinstatements pro rata as to amount | Yes | Functions of the annual layer loss |
| Loss corridor | Yes | A function of the annual layer loss |
| Premium, ceding commission, sliding scale, profit commission, swing rating | Yes | Their expectations come from the ceded grid |
| Inuring stages without annual terms | Yes | Each stage is a function of each loss |
| Annual terms in a layer that inures to a later one | No | The later layer sees each loss net of the earlier one, which depends on event order |
| Reinstatement premium pro rata as to time | No | Needs the date of each loss |
| Surplus treaty | No | Cedes a share set by the sum insured of each risk |
| Joint distribution across layers or lines | No (a limit, not an error) | A grid is one marginal distribution |

```python
timed = Layer("5x5", 5.0, 5.0, premium=1.0, reinstatement_rates=[1.0], pro_rata_time=True)
surplus = Layer.surplus("surplus", 1.0, 4.0)
inuring = Tower.inuring([[Layer("5x5", 5.0, 5.0, reinstatements=1)], [Layer("10x10", 10.0, 10.0)]])
for label, tw in (("pro rata as to time", Tower([timed])),
                  ("surplus", Tower([surplus])),
                  ("annual terms inuring", inuring)):
    try:
        tw.on_grid(freq, g, 4096)
    except ValueError as e:
        print(f"{label}: {str(e).split(': use')[0]}")
```

```text
pro rata as to time: invalid data: layer "5x5" has reinstatements pro rata as to time; a compound distribution has no event times
surplus: invalid data: layer "surplus" is a surplus treaty, which works risk by risk; a compound distribution has no sums insured
annual terms inuring: invalid parameter layers = 0: annual terms in a stage that inures to a later one depend on event order; use Tower::apply on simulated events
```

**Dates (pro rata as to time).** This reinstatement premium is lower when the
layer uses its limit late in the year, so each loss needs a date.
`with_uniform_times()` draws sorted uniform times; `with_seasonal_times(weights)`
follows a season such as a hurricane year.

```python
events = simulate_events(freq, sev, 20_000, 3).with_uniform_times()
by_amount = Layer("5x5", 5.0, 5.0, premium=1.0, reinstatement_rates=[1.0])
rows = []
for label, layer in (("as to amount, simulated", by_amount), ("as to time, simulated", timed)):
    result = Tower([layer]).apply(events)
    paid = result.marginal(("reinstatement_premium", "5x5"))
    rows.append({"basis": label, "premium": paid.mean(), "se": paid.variance() ** 0.5 / len(paid) ** 0.5})
grid_amount = Tower([by_amount]).on_grid(freq, g, 4096).expected_reinstatement_premium[0]
rows.append({"basis": "as to amount, on the grid", "premium": grid_amount, "se": None})
(GT(pd.DataFrame(rows), rowname_col="basis")
 .tab_header("Expected reinstatement premium", "Money units; layer 5 xs 5 with one reinstatement at 100%")
 .cols_label(premium="premium", se="standard error")
 .fmt_number(["premium", "se"], decimals=4)
 .sub_missing(missing_text=""))
```

```text
                           premium  standard error
as to amount, simulated     0.0870          0.0018
as to time, simulated       0.0440          0.0010
as to amount, on the grid   0.0844                
```

Pro rata as to time roughly halves the premium, because an exhausting loss
arrives on average half way through the year. The grid value is within two
standard errors of the simulated by-amount premium.

**Sums insured (surplus).** A surplus treaty cedes a share set by the sum
insured of the risk hit, so each event must carry one. `RiskProfile.simulate`
draws losses from a property risk profile (bands of sum insured, each with an
exposure curve) and attaches the sums insured. Here the surplus inures to a
per-risk excess of loss with a reinstatement, so two reasons apply at once.

```python
profile = RiskProfile([1e6, 5e6, 20e6], [800, 120, 20], Mbbefd.swiss_re(3.0),
                      premiums=[3e6, 2.5e6, 1.5e6], loss_ratio=0.6)
treaty = Tower.inuring([
    [Layer.surplus("surplus", 2e6, 4.0)],
    [Layer("3x2", 3e6, 2e6, reinstatements=1)],
])
result = treaty.apply(profile.simulate(20_000, 5))
rows = [{"layer": name, "sim": result.marginal(("ceded", name)).mean(), "exact": exact}
        for name, exact in (("surplus", profile.expected_surplus_loss(2e6, 4.0)),
                            ("3x2", profile.expected_layer_loss(3e6, 2e6, 2e6, 4.0)))]
(GT(pd.DataFrame(rows), rowname_col="layer")
 .tab_header("Expected ceded loss: simulation against exposure rating", "Money units")
 .cols_label(sim="simulated", exact="exposure-rated")
 .fmt_integer(["sim", "exact"]))
```

```text
         simulated  exposure-rated
surplus  1,259,913       1,260,000
3x2        118,875         117,525
```

The profile's exposure curves give an independent expected value (see
[Experience and exposure rating](experience-and-exposure-rating.md)).

**Joint results.** `Tower.apply` returns one joint distribution, so gross, every
layer and net agree year by year (gross = ceded + net). The 99% of the sum of
two layers, or dependence between lines, needs joint draws. To tie lines
together with a copula or a rank correlation, see section 8 of
[Choosing a method](choosing-a-method.md#8-capital-risk-measures-and-allocation)
(`GaussianCopula`, `iman_conover`, `PredictiveDistribution`). Reserves across
lines use `SegmentDependence` ([Reserving methods](reserving-methods.md)).

## 7. The occurrence view

Many covers pay per occurrence: a catastrophe layer pays on each event, not on
the sum of the year. Their curve is the **occurrence exceedance
probability** (OEP): the probability that the *largest* loss of the year
exceeds `x`. It is `1 - G_N(F(x))`, with `G_N` the generating function of the
count. The **annual aggregate** curve is the probability that the *total*
exceeds `x`. The **occurrence PML** (probable maximum loss) at annual
probability `p` is the loss whose OEP is `p`.

```python
rows = []
for p in (0.1, 0.01):
    o = occurrence_pml(freq, sev, p)
    rows.append({"p": f"{p}", "pml": o.loss, "rp": o.annual_return_period,
                 "agg": annual.quantile(1 - p)})
(GT(pd.DataFrame(rows), rowname_col="p")
 .tab_header("Occurrence PML against the annual aggregate", "Money units, at annual exceedance probability p")
 .cols_label(pml="occurrence PML", rp="return period (years)", agg="annual aggregate")
 .fmt_number(["pml", "agg"], decimals=2)
 .fmt_integer("rp"))
```

```text
      occurrence PML  return period (years)  annual aggregate
0.1             6.48                     10             11.94
0.01           18.49                    100             25.81
```

```python
print(f"occurrence return period of the 1% loss: {o.occurrence_return_period:.1f} years")
print(f"P(largest claim > 5): {occurrence_exceedance(freq, sev, 5.0):.4f},"
      f" with the mixed Poisson count of 4.6: {occurrence_exceedance(mixed, sev, 5.0):.4f}")
print(f"simulated 1% occurrence PML: {simulate_events(freq, sev, 100_000, 3).maxima().quantile(0.99):.2f}")
```

```text
occurrence return period of the 1% loss: 99.5 years
P(largest claim > 5): 0.1573, with the mixed Poisson count of 4.6: 0.1543
simulated 1% occurrence PML: 18.54
```

The 1-in-100 annual aggregate is well above the 1-in-100 occurrence, because
several claims add up. Use the occurrence curve for per-occurrence covers
(catastrophe, per-risk layers), the aggregate curve for stop loss, aggregate
limits and capital. The occurrence return period is the average time between
occurrences above the loss. It is not 100 because a year can
hold more than one occurrence.

For a Poisson count the OEP is `1 - exp(-6 (1 - F(x)))`, with `F` the severity
cdf:

```python
import math

grid = [0.25 * i for i in range(1, 161)]
oep = [1 - math.exp(-6.0 * (1 - sev.cdf(x))) for x in grid]
aep = [1 - annual.cdf(x) for x in grid]
fig, ax = plt.subplots()
ax.plot(grid, aep, label="annual aggregate (total of the year)")
ax.plot(grid, oep, label="occurrence (largest claim of the year)")
ax.set_yscale("log")
ax.set_ylim(1e-3, 1)
ax.set_xlabel("loss (money units)")
ax.set_ylabel("probability of exceeding the loss")
ax.set_title("Exceedance probability: occurrence against annual aggregate")
ax.legend()
```

The OEP is closed form for every count prospicio accepts. A more variable count
makes a year with at least one claim above 5 slightly *less* likely (0.1543
against 0.1573 for the same mean), because more years have few or no claims,
yet it makes the aggregate tail fatter (section 3.6). A simulation gives the
same curve from `EventSet.maxima()` (the largest loss of each year).

`occurrence_layer_statistics` answers the layer questions in one call:

```python
st = occurrence_layer_statistics(freq, g, 1.0, 5.0, 5.0)  # share, limit, attachment
rows = [("P(claim reaches layer)", st.pr_attach, 4), ("P(claim exhausts layer)", st.pr_detach, 4),
        ("P(loss in a year)", st.pr_loss, 4), ("loss on line", st.loss_on_line, 4),
        ("expected excess claims", st.count.mean, 4), ("mean layer loss per excess claim", st.severity.mean, 3),
        ("annual ceded: mean", st.aggregate.mean, 4), ("annual ceded: cv", st.aggregate.cv, 3),
        ("annual ceded: skew", st.aggregate.skew, 3)]
df = pd.DataFrame({"statistic": [r[0] for r in rows], "value": [r[1] for r in rows]})
tbl = GT(df, rowname_col="statistic").tab_header("The 5 xs 5 layer in one call", "occurrence_layer_statistics")
for d in (3, 4):
    tbl = tbl.fmt_number("value", rows=[i for i, r in enumerate(rows) if r[2] == d], decimals=d)
tbl
```

```text
                                   value
P(claim reaches layer)            0.0282
P(claim exhausts layer)           0.0072
P(loss in a year)                 0.1557
loss on line                      0.0876
expected excess claims            0.1692
mean layer loss per excess claim   2.589
annual ceded: mean                0.4381
annual ceded: cv                   2.972
annual ceded: skew                 3.383
```

The annual mean matches the `CollectiveModel` value in section 1. `pr_loss` is
the OEP at the attachment: 0.1573 from `occurrence_exceedance` on the continuous
severity, 0.1557 on the grid.

## 8. Which method for which job

| Job | Use | Notes |
|---|---|---|
| Layer pricing (expected loss, loss on line, a risk load) | `CollectiveModel.layer_mean`, `occurrence_layer_statistics`, `Tower.on_grid` | Closed form or FFT. Use `local_moment` so means are right. A distortion price needs the grid (`TowerGrids.price(distortion)` for the stand-alone price of each layer, or `price(grid, Distortion.tvar(p), cost_of_capital=r)` for any grid at a chosen capital standard). See [What a reinsurance price is](reinsurance-price.md) |
| Capital on one book (99% or 99.5% VaR or TVaR) | `fft_auto`, then the halving check | The grid is exact where simulation is noisy in the tail. Report `tail_mass` and `aliasing_error` |
| Capital with dependence between lines or across treaties | Simulation, `PredictiveDistribution`, a copula | Needs joint draws ([Choosing a method](choosing-a-method.md#8-capital-risk-measures-and-allocation)). Fix the seed, use many years, report the standard error (section 5.1) |
| Reporting a result others must reproduce | FFT, with its report stored beside it | Deterministic. If you simulate, record the seed and `n_sims` |
| Sizing a catastrophe or per-risk limit | `occurrence_pml`, `occurrence_layer_statistics` | The occurrence curve, not the annual total |

A grid gives both capital measures. TVaR at `p` is the mean of the worst
`1 - p` of years: the `p` quantile plus `E[(S - q)+] / (1 - p)`. Solvency II
asks for the 99.5% VaR, the Swiss Solvency Test for the 99% TVaR:

```python
print(f"99.5% VaR {annual.quantile(0.995):.2f}, 99.5% TVaR {annual.tvar(0.995):.2f},"
      f" 99% TVaR {annual.tvar(0.99):.2f}")
```

```text
99.5% VaR 31.56, 99.5% TVaR 43.83, 99% TVaR 36.06
```

TVaR reaches further into the tail than a VaR, so `tail_mass` must be small
beside `1 - p`. Run the halving check (section 3.4) on the TVaR itself.

## 9. Pitfalls

- **Reading `tail_mass` before `aliasing_error`.** Large aliasing hides wrapped
  mass in the grid, so `tail_mass` looks small.
- **A grid too short for the book, with no error.** `fft` computes on the grid
  you give it (section 4, 800 claims). Use `fft_auto` or check the report.
- **A `lower` grid for a price.** It is a lower bound by construction.
- **Trusting a quantile to the step.** A grid quantile is a grid point. Halve the
  step and judge a smooth number as well.
- **Moment fits in the tail.** They match three numbers and no more (section 2).
- **Comparing simulations on different seeds.** Keep the seed fixed when you
  compare two versions of a treaty, so the years differ only by your change.
- **Pricing a per-occurrence layer on the OEP.** The OEP gives the probability of
  a loss to the layer. The annual loss comes from the compound.
- **Adding layer grids.** You can add means, but not get the distribution of the
  sum, because the layers share the same losses. Use `Tower.apply`.

## See also

- [Choosing a method](choosing-a-method.md): which method fits which reinsurance question.
- [Risk loads and distortions](risk-loads-and-distortions.md): pricing the distributions this guide produces.
- [Experience and exposure rating](experience-and-exposure-rating.md): where the severity and the risk profile come from.
- [Severity and frequency](severity-and-frequency.md): choosing and fitting the inputs.
- [Reserving methods](reserving-methods.md): the other side of the actuary's work.
- [What a reinsurance price is](reinsurance-price.md): the price built from these distributions.
- [The aggregate design note](../design/aggregate.md) and [the `aggregate` parity reference](../../knowledge/references/aggregate-mildenhall.md): the design and the parity checks.

## Appendix: terms

- **Frequency** is the number of claims in a year, **severity** the size of one
  claim. The **aggregate loss** is their sum `S = X_1 + ... + X_N`, with claim
  sizes independent of each other and of `N`.
- **Per-occurrence** terms apply to each claim (an excess layer pays the part
  between its attachment and attachment plus limit). **Annual** terms apply to
  the total of the year (an annual deductible or limit, a stop loss).
- **Grid**: a distribution held as probabilities at `0, h, 2h, ...`, with
  **step** (bucket) `h`. FFT and Panjer **discretize** a continuous severity
  onto a grid, and that is their approximation.
- **Aliasing**: an FFT is circular, so probability beyond the end of its buffer
  wraps onto the start. **Tail mass** is the probability beyond the last grid
  point, which the grid puts on that point.
- **Seed**: the number that fixes the random draws of a simulation, so you can
  repeat a run.
