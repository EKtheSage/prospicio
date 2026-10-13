# Aggregate distributions: FFT, Panjer, simulation or moments

Use this guide when you need the distribution of a year's total loss (the
**aggregate loss**), or of what one reinsurance contract pays out of it, and
must choose how to compute it. prospicio has four engines: closed-form
**moments**, **FFT** (fast Fourier transform on a grid), **Panjer**'s
recursion and **Monte Carlo simulation**. They answer overlapping questions at
different costs and fail in different ways. This guide says which to reach for,
how to size and check it, and the exact calls.

It is written for an actuary who knows frequency and severity from primary
pricing and is new to reinsurance pricing. Money is in millions. Every number
below is printed by the code next to it, and CI re-runs the code.

## The decision table

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| Mean, standard deviation and skewness of the year, or a sanity check | Moments | `compound_moments` | Closed form, no grid, no sampling error. Its fitted distributions are rough in the tail |
| Quantiles, stop-loss prices or any tail number for one book | FFT, grid sized for you | `fft_auto` | Exact for the discretized problem, error reported |
| The same, with a bucket you must use (a policy limit, a market convention) | FFT on your grid | `Grid.local_moment` then `fft` | You control the step; `local_moment` keeps the mean |
| An independent exact check of an FFT result | Panjer | `panjer` | Same answer by a different algorithm; no aliasing |
| A large claim count (a Poisson mean of several hundred or more) | FFT | `fft_auto` | Panjer fails once the chance of a zero year underflows: a Poisson mean of 745 runs, 800 does not (section 5) |
| Claims that cluster (contagion, or an uncertain claim rate) | FFT with a mixed Poisson count | `fft_auto(Count.mixed_poisson(...), sev)` | A more variable count gives a fatter annual tail; Panjer refuses it (section 4.6) |
| A policy limit or deductible on each claim | FFT on a mapped grid | `Grid.local_moment`, `Grid.map`, then `fft` | `fft_auto` sizes for the severity it is given (section 4.7) |
| A count that is not Poisson, negative binomial or binomial | FFT | `fft` with a `Count` | Panjer needs the recursion's count families |
| Heavy tail with infinite variance | FFT, grid sized by one big claim | `fft_auto` | Moments are undefined; the sizing does not need them |
| What an excess layer, quota share, stop loss, reinstatements or a corridor pay | FFT on the tower | `Tower.on_grid` | Exact, no sampling error |
| A quota share or stop loss (with its commissions) on an annual loss already on a grid, such as a loss ratio times premium | The tower on the annual grid | `Tower.on_annual_grid` | Each year's total is one loss; no claim count needed, expected commissions exact |
| Reinstatement premium pro rata as to time, a surplus treaty, annual terms that inure to a later layer | Simulation | `Tower.apply` on an `EventSet` | They need event order, dates or sums insured; a compound grid has none |
| Several layers or lines together (joint distribution, dependence) | Simulation | `Tower.apply`, `PredictiveDistribution` | A grid is one marginal distribution |
| A catastrophe model's event loss table | Simulation | `events, rows = EventLossTable(...).simulate(n_years, seed)` | Returns the years and each loss's row in the table; draws each event's loss from its secondary uncertainty; exact OEP and PML from `EventLossTable.oep` and `EventLossTable.occurrence_pml` |
| Your own simulated years (a year loss table), a non-cat set beside a cat set, an hours clause | Simulation | `EventSet.from_long` or `EventSet.from_years`; `EventSet.merge`; `EventSet.grouped(window_hours)` | The events are the model. `merge` needs the same number of years in both sets; `grouped` needs loss times (fractions of a year) and drops sums insured |
| The largest single loss of the year (occurrence PML) | Closed form | `occurrence_pml`, `occurrence_exceedance` | For per-occurrence covers the annual total is the wrong curve |
| How often a layer is hit, its loss on line, its count and severity moments | Closed form | `occurrence_layer_statistics` | One call |

One rule covers most days: **use FFT unless a contract term forces simulation,
and read the report either way.**

## 1. Words you need first

- **Frequency** is the number of claims in a year, **severity** the size of one
  claim, **aggregate loss** their sum `S = X_1 + ... + X_N`. Claim sizes are
  independent of each other and of `N`.
- **Per-occurrence** terms apply to each claim (an excess layer pays the part
  of each claim between its attachment and attachment plus limit). **Annual**
  terms apply to the year's total (an annual deductible, an annual limit, a
  stop loss).
- **Grid**: a distribution held as probabilities at `0, h, 2h, ...`, with
  **step** (bucket) `h`. **Discretizing** a continuous severity onto a grid is
  the approximation FFT and Panjer make.
- **Aliasing**: an FFT is circular, so probability beyond the end of its buffer
  wraps onto the start. **Tail mass** is the probability beyond the last grid
  point, which the grid lumps onto that point. The library measures both.
- **Seed**: the number that fixes a simulation's random draws, so a run can be
  repeated.

## 2. The running book

One synthetic book: six claims a year on average (Poisson), lognormal claim
sizes with mean 1 and coefficient of variation 2. We want the annual
distribution and the price of a per-occurrence layer of 5 xs 5 (a limit of 5
above an attachment of 5). The first call sizes a grid and runs the FFT.

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

`fft_auto(freq, sev, 12)` allows at most `2**12` points. The default is `2**16`,
which gives a finer step; we use 12 to keep the examples quick. The 99% value is
the annual loss exceeded with probability 1%.

The layer has a closed form: for a Poisson count, the expected annual loss is
the expected claim count times the severity's layer mean. `CollectiveModel`
gives it, and every engine below is compared with it.

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

## 3. Moments: when closed form is enough

`compound_moments` returns the mean, variance, skewness and third cumulant of
the year from the count's cumulants and the severity's first three moments. It
has no grid and no sampling error, and `.approximate(kind)` fits a distribution
to those moments.

```python
m = compound_moments(freq, sev)
print(f"mean {m.mean:.3f}, cv {m.cv:.4f}, skew {m.skew:.4f}")
for kind in ["normal", "gamma", "lognormal", "shifted_gamma", "shifted_lognormal"]:
    fit = m.approximate(kind)
    print(f"{kind:18s} 99%: {fit.quantile(0.99):6.2f}   99.5%: {fit.quantile(0.995):6.2f}")
print(f"{'FFT':18s} 99%: {annual.quantile(0.99):6.2f}   99.5%: {annual.quantile(0.995):6.2f}")
```

```text
mean 6.000, cv 0.9129, skew 4.5644
normal             99%:  18.74   99.5%:  20.11
gamma              99%:  25.24   99.5%:  28.82
lognormal          99%:  27.11   99.5%:  32.92
shifted_gamma      99%:  30.63   99.5%:  37.50
shifted_lognormal  99%:  27.59   99.5%:  34.16
FFT                99%:  25.81   99.5%:  31.56
```

**Use moments** for the mean and spread of a year: a quick load, a check on a
grid or a simulation, the input to a rule that wants only a standard deviation.

**Do not use the fitted distributions for capital or layer prices.** A fit
matches three numbers and says nothing else about the shape. At 99% the normal
is far too light (18.74 against the FFT's 25.81), the gamma a little light, and
the lognormal, shifted lognormal and shifted gamma too heavy, the last by almost
a fifth. No family is right at both quantiles.

**Data needed**: any count, and a severity with a finite third moment. The shifted
fits also need positive skewness. A heavy tail has no third moment and the call
refuses (section 4.5).

## 4. FFT: the default engine

An FFT turns the convolution of `N` severities into a product in the Fourier
domain, so the whole distribution of `S` costs about `n log n` for an `n`-point
grid. It works for any count with a probability generating function (Poisson,
negative binomial, binomial and every `Count` family) and for large counts.

It is **exact for the discretized problem**. The approximations are the
discretization of the severity, the grid's finite length, and aliasing. This
section is about measuring them.

### 4.1 Let the library size the grid

`recommend_grid` picks the step and number of points from the count and
severity, as the Python package `aggregate` does. `fft_auto` sizes the grid the
same way and then runs the FFT. The step is on a ladder (`round_bucket`).

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
  rounded up the ladder, so the grid reaches at least that far.
- `method` says what set the extent: `"moments"` (a lognormal or gamma fitted to
  the year's mean and variance) or `"single_big_jump"` (one big claim on top of
  a typical year, which governs heavy tails).
- `tail_estimate` is roughly the probability beyond the grid from one claim
  alone. It should be tiny. If it is not, raise `log2`.

### 4.2 Read the report every time

`fft` and `panjer` return the grid and a `CompoundReport`. Read it in this
order:

| Field | Meaning | Act when |
|---|---|---|
| `aliasing_error` | Largest change in any probability when the FFT buffer doubles | Above about `1e-9` (a rule of thumb, not a bound): the grid is unreliable, `tail_mass` included. Lengthen the grid |
| `tail_mass` | Probability beyond the last point, lumped onto it | Not small beside the tail you price (a 99.5% quantile needs it well under 0.5%) |
| `mean_error()` | Grid mean minus `E[N] E[X]` on the severity grid | Not tiny: lost tail or aliasing |
| `points` | Length of the result | Shorter than you meant |

A grid that is too short shows at once. The same book at the same step, on 128
to 4096 points:

```python
def tiny(x):
    """Print a residual, or '<1e-10' when it is rounding noise that differs by platform."""
    return f"{x:.1e}" if x >= 1e-10 else "<1e-10"


for n in (128, 256, 512, 1024, 4096):
    g, disc = Grid.local_moment(sev, 0.0625, n)
    a, r = fft(freq, g, n)
    print(f"{n:5d} points  tail_mass {r.tail_mass:.1e}  aliasing {tiny(r.aliasing_error)}"
          f"  mean error {r.mean_error():+.1e}  grid mean {a.mean():.3f}")
```

```text
  128 points  tail_mass 2.4e-01  aliasing 4.5e-04  mean error -8.4e-01  grid mean 4.763
  256 points  tail_mass 4.6e-02  aliasing 1.3e-05  mean error -1.9e-01  grid mean 5.674
  512 points  tail_mass 4.8e-03  aliasing 1.1e-07  mean error -2.5e-02  grid mean 5.940
 1024 points  tail_mass 4.0e-04  aliasing 6.7e-10  mean error -2.3e-03  grid mean 5.991
 4096 points  tail_mass 1.9e-06  aliasing <1e-10  mean error -1.1e-05  grid mean 6.000
```

At 128 points the grid reaches only 8, nearly a quarter of the probability lies
beyond it and the grid mean is far under 6. Aliasing and tail mass both fall as
the grid grows; look at both. Do not read `tail_mass` alone: when aliasing is
large, wrapped mass sits inside the grid and `tail_mass` understates the truth.
`aliasing_error` first. Passing the aliasing rule is not enough either: the
1024-point row has aliasing 6.7e-10 and still 4.0e-4 of probability beyond the
grid, too much for a 99.9% quantile. Both tests must pass for the tail you need.

### 4.3 Choosing the discretizer

When you choose the grid yourself, three discretizers turn a continuous
severity into grid probabilities. Each returns the grid and a
`DiscretizationReport`.

| Call | What it does | Use for |
|---|---|---|
| `Grid.local_moment(sev, h, n)` | Splits each cell's mass between its two neighbouring points so the mean is kept | Pricing. Means and layer means are right even at a coarse step |
| `Grid.rounding(sev, h, n)` | Moves each loss to the nearest point | Matching `aggregate`'s reports; fine at a small step. `fft_auto` uses it |
| `Grid.lower(sev, h, n)` | Moves each cell's mass to its left end | A stochastic lower bound: every loss, layer and quantile is understated, so never a price |

The same book at three steps (the 5 xs 5 exact value is 0.43812):

```python
tower = Tower([Layer("5x5", 5.0, 5.0)])
print("step   discretizer   gross mean   5x5 layer   99% quantile")
for step, n in ((1.0, 128), (0.25, 512), (0.0625, 2048)):
    for name in ("rounding", "lower", "local_moment"):
        g, _ = getattr(Grid, name)(sev, step, n)
        grids = tower.on_grid(freq, g, n)
        print(f"{step:<6} {name:<13} {grids.gross.mean():10.4f} {grids.ceded[0].mean():11.4f}"
              f" {grids.gross.quantile(0.99):12.2f}")
```

```text
step   discretizer   gross mean   5x5 layer   99% quantile
1.0    rounding          5.5738      0.4360        25.00
1.0    lower             3.7553      0.3783        23.00
1.0    local_moment      5.9987      0.4381        26.00
0.25   rounding          5.9775      0.4380        25.75
0.25   lower             5.2739      0.4224        24.75
0.25   local_moment      5.9987      0.4381        25.75
0.0625 rounding          5.9987      0.4381        25.81
0.0625 lower             5.8111      0.4341        25.56
0.0625 local_moment      5.9987      0.4381        25.81
```

`local_moment` has the mean right at every step (its shortfall below 6 is the
tail beyond the grid's last point). `lower` is a true lower bound and converges
slowly. Quantiles move with the step because a grid quantile is a grid point.

`local_moment` keeps the mean, not the variance: splitting mass between two
points adds a little spread, so at a coarse step its quantiles and stop-loss
premiums sit a little high (26.00 at step 1.0 against 25.81 at the finest).
Before you price a stop loss or quote a tail number, make the step small enough
that the halving check below is stable.

### 4.4 Is the step fine enough? Halve it

The only discretization error the library reports is the mean's. The practical
check is to halve the step, double the points, and see whether your number moves:

```python
prev = None
for step, n in ((0.25, 512), (0.125, 1024), (0.0625, 2048), (0.03125, 4096)):
    g, _ = Grid.local_moment(sev, step, n)
    a, r = fft(freq, g, n)
    q = a.quantile(0.995)
    tag = "" if prev is None else f"  change {q - prev:+.3f}"
    print(f"step {step:<8} 99.5% {q:7.4f}  stop-loss at 30: {a.stop_loss(30.0):.5f}{tag}")
    prev = q
```

```text
step 0.25     99.5% 31.5000  stop-loss at 30: 0.06879
step 0.125    99.5% 31.6250  stop-loss at 30: 0.06877  change +0.125
step 0.0625   99.5% 31.5625  stop-loss at 30: 0.06877  change -0.062
step 0.03125  99.5% 31.5938  stop-loss at 30: 0.06877  change +0.031
```

A quantile moves a step at a time, so also judge a smooth number (a stop-loss
premium, a layer mean). Here the quantile stays within one step of its
neighbours and the stop-loss premium is settled to four digits from the first
halving.

### 4.5 Heavy tails

A Pareto with `alpha` 1.5 has a finite mean and an infinite variance. Moments
refuse it, and the sizing is by one big claim:

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

A power-law tail never ends, so some mass is always beyond the grid. The report
says how much; if it is too much for the quantile you need, raise `log2` or
truncate the severity (`Pareto` takes a truncation point). With `alpha` below 1
the mean is infinite and the sizing refuses:

```python
try:
    fft_auto(freq, Pareto(0.5, 0.9), 12)
except ValueError as e:
    print(e)
```

```text
invalid parameter severity = inf: the aggregate mean must be finite and positive; choose the step by hand
```

A real book with such a tail has a policy limit; apply it as in section 4.7.

### 4.6 Claims that cluster: a mixed Poisson count

The running book assumes claims arrive independently at a known rate. When one
event causes many claims (contagion), or the rate itself is uncertain from year
to year, the count is more variable than a Poisson. A **mixed Poisson** count
multiplies the rate by a random factor of mean 1; `Count.mixed_poisson(mean, cv)`
with the default gamma mixing is the negative binomial. FFT handles it
directly:

```python
mixed = Count.mixed_poisson(6.0, 0.5)  # rate factor with cv 0.5
mixed_annual, mixed_report, _ = fft_auto(mixed, sev, 12)
print(f"count variance: Poisson {freq.variance():.1f}, mixed {mixed.variance():.1f}")
for label, a in (("Poisson", annual), ("mixed Poisson", mixed_annual)):
    print(f"{label:14s} mean {a.mean():.3f}  99%: {a.quantile(0.99):6.2f}  99.5%: {a.quantile(0.995):6.2f}")
print(f"tail mass {mixed_report.tail_mass:.1e}, aliasing {tiny(mixed_report.aliasing_error)}")
```

```text
count variance: Poisson 6.0, mixed 15.0
Poisson        mean 6.000  99%:  25.81  99.5%:  31.56
mixed Poisson  mean 6.000  99%:  28.56  99.5%:  34.44
tail mass 1.9e-06, aliasing <1e-10
```

Same mean, and a 99.5% of 34.44 against 31.56. If your claim counts vary more
from year to year than a Poisson allows, this is the cheapest way to say so.
Claim sizes are still independent of each other and of the count; dependence
between the size and the number of claims needs simulation or a scenario.

### 4.7 A policy limit or deductible on each claim

`fft_auto` sizes its grid for the severity it is given. To cap each claim at a
policy limit (or take off a deductible), discretize the unlimited severity
yourself and apply the term with `Grid.map`, which keeps the mean exactly. Here
each claim is capped at 10:

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

The grid's limited mean equals the severity's limited expected value
(`sev.lev(10.0)`), and the limit pulls the 99.5% down from 31.56 to 21.81. Put
the limit on a grid point (here 10 is a multiple of the step) so `on_points` is
`True` and the shape, not only the mean, is exact. A deductible is
`lambda x: max(x - d, 0.0)`.

## 5. Panjer: the exact check

Panjer's recursion computes the aggregate probabilities one point at a time. It
is exact for the discretized problem and has no aliasing. Each point sums over
all the points below it, so its work grows with the square of the grid length,
against `n log n` for the FFT.

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
recursion's families: Poisson, negative binomial, binomial, and their
zero-modified and zero-truncated versions (`Count.zero_truncated`). Mixed
Poisson counts and the Neyman type A are refused and point to FFT. It also fails
for a large Poisson mean, because the chance of a zero year, `exp(-mean)`,
underflows:

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

The fourth line repeats section 4.2's lesson. With 800 claims a year the annual
mean is 800, far past a grid that reaches 256, and `fft` ran without complaint;
a `tail_mass` near 1 is the warning. `fft_auto` sizes the grid for this book.

## 6. Simulation

Monte Carlo draws many years: a claim count, then that many claim sizes.
`simulate_events` keeps every year's individual losses in an `EventSet`, so
per-loss terms (layers, sums insured, dates) can be applied afterwards.

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
losses, or on several results at once (section 7), or when no grid can hold the
problem (a catastrophe model's event loss table, which `EventLossTable` turns
into simulated years, or a severity that is only simulated losses).

**Do not use it** for a tail quantile of a single book when FFT applies: the
error is large and shrinks slowly.

### 6.1 Sampling error and seeds

A simulated number is random, and its error shrinks with the square root of the
number of years. Ten seeds at two sizes show the spread of the 99.5% quantile
and of the mean (each as the average over the seeds, plus or minus the standard
deviation across them):

```python
import statistics

for n in (10_000, 100_000):
    runs = [simulate_events(freq, sev, n, seed).totals() for seed in range(1, 11)]
    qs = [t.quantile(0.995) for t in runs]
    ms = [t.mean() for t in runs]
    print(f"{n:7d} years: 99.5% {statistics.mean(qs):.2f} +/- {statistics.stdev(qs):.2f},"
          f"  mean {statistics.mean(ms):.3f} +/- {statistics.stdev(ms):.3f}")
print(f"FFT: {annual.quantile(0.995):.2f}")
```

```text
  10000 years: 99.5% 31.55 +/- 1.42,  mean 6.001 +/- 0.047
 100000 years: 99.5% 31.68 +/- 0.51,  mean 6.004 +/- 0.016
FFT: 31.56
```

Ten times the years cut the error about threefold, not tenfold. A tail quantile
is the noisiest number there is: at 10,000 years its spread (1.42) is several
percent of its value, the mean's (0.047) under one percent.

A standard error from the sample variance, as in section 7, assumes the
simulated quantity has a finite variance. For a severity like section 4.5's
Pareto with `alpha` 1.5 it does not: the error bar understates the noise and the
mean settles slowly. Prefer the grid there.

The **seed** fixes the draws. By the library's design (`docs/design/aggregate.md`,
tested in the Rust crate), the same seed gives the same years for any number of
threads, and each year has its own stream, so year `i` can be replayed alone.
Record the seed with every result.

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
pricing object, `RiskProfile.simulate` draws from a property risk profile
(section 7.2), `EventLossTable.simulate` draws years from a catastrophe model's
event loss table (it returns the years with each loss's row in the table), and
`EventSet.from_years` (a list of losses per year) or `EventSet.from_long` (one
row per loss with its year, as a year loss table comes) loads years from
elsewhere. A year loss table lists only the years with a loss, so give
`from_long` the `n_years` you simulated, counting the years with no loss;
otherwise every frequency and loss cost comes out too high.

## 7. What each treaty pays

A contract pays out of the year's losses. Two engines compute the payout
distribution: `Tower.on_grid` (exact FFT) and `Tower.apply` (simulation).

### 7.1 Exactly on the grid

`Tower.on_grid(freq, severity_grid, points)` applies each layer's per-occurrence
recovery to the severity grid, compounds it with the same count, then applies
the annual terms (annual deductible and limit, reinstatements, corridor) and the
share to the annual total. It returns gross, each layer's ceded loss and, where
it is one compound total, net. There is no sampling error.

```python
program = Tower([
    Layer("5x5", 5.0, 5.0, reinstatements=1),
    Layer("10x10", 10.0, 10.0),
    Layer.quota_share("QS", 0.25),
])
g, _ = Grid.local_moment(sev, 0.0625, 4096)
grids = program.on_grid(freq, g, 4096)
for name, ceded in zip(program.layer_names, grids.ceded):
    print(f"{name:6s} mean {ceded.mean():.4f}  99% {ceded.quantile(0.99):.3f}")
print("net available:", grids.net is not None, " on points:", grids.on_points)
plain = Tower([Layer("5x5", 5.0, 5.0)]).on_grid(freq, g, 4096)
stop_loss_last = Tower.inuring([[Layer("5x5", 5.0, 5.0)], [Layer.stop_loss("SL", 10.0, 10.0)]])
print("net without annual terms:", plain.net is not None,
      " net with a stop loss last:", stop_loss_last.on_grid(freq, g, 4096).net is not None)
```

```text
5x5    mean 0.4377  99% 5.000
10x10  mean 0.1956  99% 8.562
QS     mean 1.5000  99% 6.453
net available: False  on points: True
net without annual terms: True  net with a stop loss last: True
```

`net` is `None` because the 5x5 has an annual limit (from the reinstatement), and
net would then depend on several annual totals at once. Net exists when no layer
has annual terms, or when the last stage is a single aggregate cover such as a
stop loss, as the last line shows. `on_points` says whether every value fell on a grid point; when it is
`False`, means are still exact but shapes smear by up to a step.

The same tower by simulation, to see the two agree:

```python
sim = program.apply(simulate_events(freq, sev, 100_000, 3))
for name, ceded in zip(program.layer_names, grids.ceded):
    s = sim.marginal(("ceded", name))
    print(f"{name:6s} grid {ceded.mean():.4f}   simulated {s.mean():.4f}"
          f" +/- {s.variance() ** 0.5 / len(s) ** 0.5:.4f}")
```

```text
5x5    grid 0.4377   simulated 0.4360 +/- 0.0041
10x10  grid 0.1956   simulated 0.1952 +/- 0.0038
QS     grid 1.5000   simulated 1.4985 +/- 0.0044
```

Publish the grid answer, which has no noise. Simulation is the check.

### 7.2 What forces simulation

A compound distribution has no memory of which loss came first, when it
happened, or how big the insured risk was. Terms that need those cannot run on
the grid. `on_grid` refuses the first three "No" rows below with an error; the
last is a limit of the method, not an error.

| Contract term | On the grid? | Why |
|---|---|---|
| Per-occurrence layer, share, quota share, stop loss | Yes | A function of each loss or of the annual total |
| Annual deductible and limit, reinstatements pro rata as to amount | Yes | Functions of the annual layer loss |
| Loss corridor | Yes | A function of the annual layer loss |
| Premium, ceding commission, sliding scale, profit commission, swing rating | Yes | Their expectations are reported from the ceded grid |
| Inuring stages without annual terms | Yes | Each stage is a function of each loss |
| Annual terms in a layer that inures to a later one | No | The later layer sees each loss net of the earlier one, which depends on event order |
| Reinstatement premium pro rata as to time | No | Needs each loss's date |
| Surplus treaty | No | Cedes a share set by each risk's sum insured |
| Joint distribution across layers or lines | No (a limit, not an error) | A grid is one marginal distribution |

The refusals, in the library's words:

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

Three recipes, one per reason.

**Dates (pro rata as to time).** This reinstatement premium charges less when
the limit is used late in the year, so each loss needs a date.
`with_uniform_times()` draws sorted uniform times; `with_seasonal_times(weights)`
follows a season such as a hurricane year.

```python
events = simulate_events(freq, sev, 20_000, 3).with_uniform_times()
by_amount = Layer("5x5", 5.0, 5.0, premium=1.0, reinstatement_rates=[1.0])
for label, layer in (("pro rata as to amount", by_amount), ("pro rata as to time", timed)):
    result = Tower([layer]).apply(events)
    paid = result.marginal(("reinstatement_premium", "5x5"))
    se = paid.variance() ** 0.5 / len(paid) ** 0.5
    print(f"{label:22s} expected reinstatement premium {paid.mean():.4f} +/- {se:.4f}")
grid_amount = Tower([by_amount]).on_grid(freq, g, 4096).expected_reinstatement_premium[0]
print(f"as to amount on the grid: {grid_amount:.4f}")
```

```text
pro rata as to amount  expected reinstatement premium 0.0870 +/- 0.0018
pro rata as to time    expected reinstatement premium 0.0440 +/- 0.0010
as to amount on the grid: 0.0844
```

Charging by time roughly halves the premium here, since an exhausting loss
arrives on average half way through the year. The grid's by-amount premium is
within two standard errors of the simulated one.

**Sums insured (surplus).** A surplus treaty cedes a share of each loss set by
the sum insured of the risk hit, so each event must carry one.
`RiskProfile.simulate` draws losses from a property risk profile (bands of sum
insured, each with an exposure curve) and attaches them. Here the surplus
inures to a per-risk excess of loss that has a reinstatement, so both reasons
apply at once.

```python
profile = RiskProfile([1e6, 5e6, 20e6], [800, 120, 20], Mbbefd.swiss_re(3.0),
                      premiums=[3e6, 2.5e6, 1.5e6], loss_ratio=0.6)
treaty = Tower.inuring([
    [Layer.surplus("surplus", 2e6, 4.0)],
    [Layer("3x2", 3e6, 2e6, reinstatements=1)],
])
result = treaty.apply(profile.simulate(20_000, 5))
for name, exact in (("surplus", profile.expected_surplus_loss(2e6, 4.0)),
                    ("3x2", profile.expected_layer_loss(3e6, 2e6, 2e6, 4.0))):
    print(f"{name:8s} simulated {result.marginal(('ceded', name)).mean():12,.0f}"
          f"   exposure-rated {exact:12,.0f}")
```

```text
surplus  simulated    1,259,913   exposure-rated    1,260,000
3x2      simulated      118,875   exposure-rated      117,525
```

The profile's exposure curves give an independent expected value to check the
simulation against (see the guide on experience and exposure rating).

**Joint results.** `Tower.apply` returns one joint distribution, so gross, every
layer and net are consistent year by year (gross = ceded + net in every
simulated year). If you need the 99% of the sum of two layers, or dependence
between lines, you need joint draws, so simulate. To tie lines together with a
copula or a rank correlation, see section 8 of
[Choosing a method](choosing-a-method.md#8-capital-risk-measures-and-allocation)
(`GaussianCopula`, `iman_conover`, `PredictiveDistribution`); reserves across
lines use `SegmentDependence` ([Reserving methods](reserving-methods.md)).

## 8. The occurrence view

Many covers pay per occurrence: a catastrophe layer pays on each event, not on
the year's sum. For them the relevant curve is the **occurrence exceedance
probability** (OEP): the chance that the year's *largest* loss exceeds `x`,
`1 - G_N(F(x))`, with `G_N` the count's generating function. The **annual
aggregate** curve is the chance that the *total* does. The **occurrence PML**
(probable maximum loss) at annual probability `p` is the loss whose OEP is `p`.

```python
for p in (0.1, 0.01):
    o = occurrence_pml(freq, sev, p)
    print(f"p {p}: occurrence PML {o.loss:6.2f} (return period {o.annual_return_period:.0f} years),"
          f"  annual aggregate {annual.quantile(1 - p):6.2f}")
print(f"occurrence return period of the 1% loss: {o.occurrence_return_period:.1f} years")
print(f"P(largest claim > 5): {occurrence_exceedance(freq, sev, 5.0):.4f},"
      f" with the mixed Poisson count of 4.6: {occurrence_exceedance(mixed, sev, 5.0):.4f}")
print(f"simulated 1% occurrence PML: {simulate_events(freq, sev, 100_000, 3).maxima().quantile(0.99):.2f}")
```

```text
p 0.1: occurrence PML   6.48 (return period 10 years),  annual aggregate  11.94
p 0.01: occurrence PML  18.49 (return period 100 years),  annual aggregate  25.81
occurrence return period of the 1% loss: 99.5 years
P(largest claim > 5): 0.1573, with the mixed Poisson count of 4.6: 0.1543
simulated 1% occurrence PML: 18.54
```

The 1-in-100 annual aggregate is well above the 1-in-100 occurrence because
several claims add up. Use the occurrence curve for per-occurrence covers
(catastrophe, per-risk layers) and for sizing a single loss; use the aggregate
curve for stop loss, aggregate limits and capital on the year's total. The
occurrence return period is the average time between occurrences above the
loss, which differs from the annual return period (100) because a year can hold
more than one.

The OEP is closed form for the counts prospicio accepts (Poisson, negative
binomial, binomial and a `Count`), not only the Poisson. A more variable count
makes a year with at least one claim above 5 slightly *less* likely (0.1543
against 0.1573 for the same mean), because more years have few or no claims,
while it fattens the aggregate tail (section 4.6). A simulation gives the same
curve from `EventSet.maxima()` (each year's largest loss), as the last line
shows.

`occurrence_layer_statistics` answers the layer questions in one call: how often
a claim reaches the layer, how often it exhausts it, the chance of any loss to
it in a year, its loss on line, and the moments of the excess claim count, the
layer severity given a loss and the annual ceded loss.

```python
st = occurrence_layer_statistics(freq, g, 1.0, 5.0, 5.0)  # share, limit, attachment
print(f"P(claim reaches layer) {st.pr_attach:.4f}, P(exhausts) {st.pr_detach:.4f},"
      f" P(loss in a year) {st.pr_loss:.4f}")
print(f"loss on line {st.loss_on_line:.4f}, expected excess claims {st.count.mean:.4f}")
print(f"mean layer loss per loss to the layer {st.severity.mean:.3f}")
print(f"annual ceded: mean {st.aggregate.mean:.4f}, cv {st.aggregate.cv:.3f}, skew {st.aggregate.skew:.3f}")
```

```text
P(claim reaches layer) 0.0282, P(exhausts) 0.0072, P(loss in a year) 0.1557
loss on line 0.0876, expected excess claims 0.1692
mean layer loss per loss to the layer 2.589
annual ceded: mean 0.4381, cv 2.972, skew 3.383
```

The annual mean matches the `CollectiveModel` value in section 2. `pr_loss` is
the OEP at the attachment (0.1573 from `occurrence_exceedance` on the continuous
severity, 0.1557 on the grid): the answer to a cedant's "how often will it hit?".

## 9. Which method for which job

| Job | Use | Notes |
|---|---|---|
| Layer pricing (expected loss, loss on line, a risk load) | `CollectiveModel.layer_mean`, `occurrence_layer_statistics`, `Tower.on_grid` | Closed form or FFT. Use `local_moment` so means are right. A distortion price needs the grid (`TowerGrids.price(distortion)` for each layer's stand-alone price, or `price(grid, Distortion.tvar(p), cost_of_capital=r)` for any grid at a chosen capital standard; see [What a reinsurance price is](reinsurance-price.md)) |
| Capital on one book (99% or 99.5% VaR or TVaR) | `fft_auto`, then the halving check | The grid is exact where simulation is noisy in the tail. Report `tail_mass` and `aliasing_error` |
| Capital with dependence between lines or across treaties | Simulation, `PredictiveDistribution`, a copula | Needs joint draws ([Choosing a method](choosing-a-method.md#8-capital-risk-measures-and-allocation)). Fix the seed, use many years, report the standard error (section 6.1) |
| Reporting a result others must reproduce | FFT, with its report stored beside it | Deterministic. If you simulate, record the seed and `n_sims` |
| Sizing a catastrophe or per-risk limit | `occurrence_pml`, `occurrence_layer_statistics` | The occurrence curve, not the annual total |

A grid gives both capital measures directly. TVaR at `p` is the mean of the
worst `1 - p` of years: the `p` quantile plus `E[(S - q)+] / (1 - p)`. Solvency
II asks for the 99.5% VaR; the Swiss Solvency Test asks for the 99% TVaR
(expected shortfall):

```python
print(f"99.5% VaR {annual.quantile(0.995):.2f}, 99.5% TVaR {annual.tvar(0.995):.2f},"
      f" 99% TVaR {annual.tvar(0.99):.2f}")
```

```text
99.5% VaR 31.56, 99.5% TVaR 43.83, 99% TVaR 36.06
```

TVaR reaches further into the tail than a VaR at the same level, so the grid's
`tail_mass` must be small beside `1 - p` with room to spare; run the halving
check (section 4.4) on the TVaR itself.

## 10. Pitfalls

- **Reading `tail_mass` before `aliasing_error`.** With large aliasing the
  wrapped mass hides in the grid and `tail_mass` looks small.
- **A grid too short for the book, with no error.** `fft` returns what it is
  given (section 5, 800 claims). Use `fft_auto` or check the report.
- **A `lower` grid for a price.** It is a lower bound by construction.
- **Trusting a quantile to the step.** A grid quantile is a grid point. Halve the
  step and judge a smooth number as well.
- **Moment fits in the tail.** They match three numbers and no more (section 3).
- **Comparing simulations on different seeds.** One run is one draw from a noisy
  estimator. Keep the seed fixed when you compare two versions of a treaty, so
  the years differ only by the change you made.
- **Pricing a per-occurrence layer on the OEP.** The OEP gives the chance of
  hitting the layer. The annual loss comes from the compound.
- **Adding layer grids.** Adding means is fine. Adding grids does not give the
  distribution of the sum, because the layers share the same losses. Use
  `Tower.apply`.

## See also

- [Choosing a method](choosing-a-method.md): the map of which method fits which reinsurance question.
- [Risk loads and distortions](risk-loads-and-distortions.md): pricing the distributions this guide produces.
- [Experience and exposure rating](experience-and-exposure-rating.md): where the severity and the risk profile come from.
- [Severity and frequency](severity-and-frequency.md): choosing and fitting the inputs.
- [Reserving methods](reserving-methods.md): the other side of the actuary's work.
- [What a reinsurance price is](reinsurance-price.md): the price built from these distributions.
- [The aggregate design note](../design/aggregate.md) and [the `aggregate` parity reference](../../knowledge/references/aggregate-mildenhall.md): the design and the parity checks.
