# prospicio for aggregate users: the twenty-minute tour

Use this guide if you already work in Stephen Mildenhall's Python package
[`aggregate`](https://aggregate.readthedocs.io/) and want to know what the same
work looks like in prospicio, or if you are new to both and want the
collective risk model by running it. It follows the package's own
"`aggregate` in Twenty Minutes" session (release 1.2.0), step by step, and
prints prospicio's numbers beside the ones that documentation publishes, so you
can see where the two agree to the last digit and where they differ by
convention.

Money here is in the units of that session (think thousands). Every number
below is printed by the code next to it, and CI re-runs the code.

## 1. The idea, and what is different

Both libraries compute the distribution of a year's total loss
`S = X_1 + ... + X_N` by discretizing the severity onto a grid, applying the
count's probability generating function with a fast Fourier transform (FFT),
and reading everything off the resulting probabilities. They differ in how you
ask:

| | `aggregate` | prospicio |
|---|---|---|
| Model written as | a DecL program (a string): `agg Casualty 250 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson` | Python objects: a count, a severity, a grid |
| Grid | chosen for you (`bs`, `log2`), overridable with `hints{}` | `fft_auto` sizes it for an unlimited severity; with a limit, a layer or a portfolio you give the step and the number of points, and `audit_grid` re-runs the build around it as `sharpen` does |
| Result | an `Aggregate` object holding DataFrames | a `Grid` (the distribution) and a `CompoundReport` (the checks) |
| Reinsurance | clauses in the program (`occurrence net of ...`) | `Layer` and `Tower` objects applied to the grid |
| Library of named examples | the recipe base (`build.recipes`) | none; the guides and notebooks are the examples |

Two habits carry over unchanged. **Validate the gross distribution before you
reinsure it**, because the check compares the grid with a closed form that
disappears once a cover is applied. And **an FFT answer is deterministic**: the
same inputs give the same digits every run.

The rest of this guide is the tour. Section 12 collects the DecL clauses in one
table, and section 13 lists what prospicio does not have.

## 2. Exact and discrete: the dice

`aggregate`'s first example is `agg Eg1 dfreq [1:5] dsev [1:3]`, a count of 1 to
5 and a severity of 1 to 3, all equally likely. In prospicio an empirical count
is `Count.empirical` (probabilities for 0, 1, 2, ...) and an empirical severity
is a `Grid` with step 1.

```python
from prospicio.aggregate import compound_moments, fft, panjer, simulate_events
from prospicio.distributions import Count, Grid, Lognormal, Poisson
from prospicio.pricing import Pentagon, Portfolio
from prospicio.reinsurance import Layer, Tower
from prospicio.risk import Distortion

count = Count.empirical([0.0] + [1 / 5] * 5)
severity = Grid(1.0, [0.0] + [1 / 3] * 3)
eg1, _ = fft(count, severity, 16)
print(f"mean {eg1.mean():.4f}, sd {eg1.variance() ** 0.5:.4f}")
print(f"pmf(1) {eg1.probs[1]:.6f} = 1/15, pmf(15) {eg1.probs[15]:.8f} = 1/1215")
print(f"cdf(5) {eg1.cdf(5.0):.6f}")
```

```text
mean 6.0000, sd 3.1623
pmf(1) 0.066667 = 1/15, pmf(15) 0.00082305 = 1/1215
cdf(5) 0.465021
```

The probability of the smallest outcome is one claim of size one,
`1/5 * 1/3`, and of the largest five claims of size three,
`1/5 * (1/3)^5`; the grid returns both exactly. For a discrete severity on a
grid the method is exact, so there is nothing to validate.

## 3. The casualty book

The session's main example is a book of 250 claims with lognormal severity of
mean 100 and coefficient of variation (CV) 1.5, limited to 1,000 per claim:

```text
agg Casualty 250 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson
```

prospicio has no string to parse; the same statement is three lines. The limit
is a mapping on the discretized severity (`Grid.map` keeps the mean exactly),
the count is a `Poisson`, and `fft` compounds them. The step of 1 and
`2**16` points are what `aggregate` reports using for this book.

```python
STEP, POINTS = 1.0, 2**16


def limited_severity(severity, limit, step=STEP, points=POINTS):
    """The severity capped at limit, on a grid: `limit xs 0`."""
    grid, _ = Grid.local_moment(severity, step, points)
    capped, _ = grid.map(lambda x: min(x, limit))
    return capped


claim = limited_severity(Lognormal.from_mean_cv(100.0, 1.5), 1000.0)
casualty, report = fft(Poisson(250.0), claim, POINTS)
print(f"claim mean {claim.mean():.3f}")
print(f"annual mean {casualty.mean():.1f}, sd {casualty.variance() ** 0.5:.1f}")
quantiles = {p: casualty.quantile(p) for p in (0.01, 0.5, 0.99, 0.999)}
print({p: round(q) for p, q in quantiles.items()})
```

```text
claim mean 98.137
annual mean 24534.1, sd 2561.7
{0.01: 18954, 0.5: 24449, 0.99: 30862, 0.999: 33170}
```

For comparison, `aggregate` 1.2.0 prints a mean claim of 98.137, an annual mean
of 24,534, a standard deviation of 2,561.7, a 1% quantile of 18,954, a median of
24,449 and a 99% quantile of 30,862, and the numbers above should agree to the
digits shown. The 99.9% quantile that the session's last section calls 33,169
is compared in section 9.

## 4. A vocabulary of severities

`aggregate` takes any `scipy.stats` family by mean and CV, and mixtures by
weights. prospicio's continuous families are in `prospicio.distributions`
(`Lognormal`, `Gamma`, `Weibull`, `Pareto`, `Burr` and others, many with a
`from_mean_cv`), and a mixture of limited severities is the weighted sum of
their grids. The session's three-component book,

```text
agg Mix 100 claims 2000 xs 0 sev lognorm [50 100 200] cv [1 1.5 2] wts [.5 .3 .2] poisson
```

is a weighted average of three limited, discretized lognormals:

```python
def mixed_severity(components, limit, step=STEP, points=POINTS):
    """A mixture of limited severities: components are (weight, severity) pairs."""
    grids = [limited_severity(sev, limit, step, points) for _, sev in components]
    probs = [0.0] * points
    for (weight, _), grid in zip(components, grids):
        for i, p in enumerate(grid.probs):
            probs[i] += weight * p
    return Grid(step, probs)


mix_claim = mixed_severity(
    [
        (0.5, Lognormal.from_mean_cv(50.0, 1.0)),
        (0.3, Lognormal.from_mean_cv(100.0, 1.5)),
        (0.2, Lognormal.from_mean_cv(200.0, 2.0)),
    ],
    2000.0,
)
mix, _ = fft(Poisson(100.0), mix_claim, POINTS)
print(f"claim mean {mix_claim.mean():.3f}")
print(f"annual mean {mix.mean():.1f}, sd {mix.variance() ** 0.5:.1f}")
print(f"1%: {mix.quantile(0.01):.0f}, median {mix.quantile(0.5):.0f}, 99%: {mix.quantile(0.99):.0f}")
```

```text
claim mean 92.996
annual mean 9299.6, sd 1894.5
1%: 5638, median 9130, 99%: 14423
```

`aggregate` prints a mean claim of 92.996, an annual mean of 9,299.6, a standard
deviation of 1,894.5, and 5,638, 9,130 and 14,423 for the 1%, median and 99%.

Tail shape is a modelling choice with a visible price. Two books with the same
mean and CV, one with a lognormal severity and one with a gamma, differ at the
99.9% level; the quickest way to see by how much is to run both:

```python
from prospicio.distributions import Gamma

for name, family in (
    ("lognormal", Lognormal.from_mean_cv(100.0, 2.0)),
    ("gamma", Gamma.from_mean_cv(100.0, 2.0)),
):
    book, _ = fft(Poisson(50.0), limited_severity(family, 20000.0), POINTS)
    print(f"{name:10s} mean {book.mean():.0f}  99.9% {book.quantile(0.999):,.0f}")
```

```text
lognormal  mean 5000  99.9% 13,979
gamma      mean 5000  99.9% 11,166
```

The session publishes 13,980 for the lognormal and 11,166 for the gamma, at a
CV of 2 and 50 claims. The digits here can differ by the cap, but the ordering
is the point: the lognormal's 99.9% sits well above the gamma's.

## 5. Frequency: Poisson is an opinion

A Poisson count says years differ only by chance. If they also differ for
reasons shared across claims (weather, inflation, the legal climate) the count
is over-dispersed. `mixed gamma 0.4` in DecL multiplies the Poisson rate by a
gamma factor of mean 1 and CV 0.4. In prospicio:

```python
book = limited_severity(Lognormal.from_mean_cv(100.0, 1.5), 1000.0)
poisson, _ = fft(Poisson(100.0), book, POINTS)
mixed, _ = fft(Count.mixed_poisson(100.0, 0.4), book, POINTS)
print(f"{'':12s} {'Poisson':>10s} {'mixed gamma':>12s}")
print(f"{'agg CV':12s} {poisson.variance() ** 0.5 / poisson.mean():10.3f} {mixed.variance() ** 0.5 / mixed.mean():12.3f}")
print(f"{'99.5% VaR':12s} {poisson.quantile(0.995):10,.0f} {mixed.quantile(0.995):12,.0f}")
```

```text
                Poisson  mixed gamma
agg CV            0.165        0.433
99.5% VaR        14,456       23,914
```

`aggregate` prints a CV of 0.165 against 0.433 and a 99.5% of 14,456 against
23,914. The mean is the same, and the capital is about two-thirds higher. Other
mixing families, zero-modified and zero-truncated counts, the logarithmic and
Neyman type A are `Count` constructors (`Count.zero_truncated`,
`Count.zero_modified`, `Count.logarithmic`, and `Count.mixed_poisson` with
`mixing="inverse_gaussian"`); see
[Severity and frequency](severity-and-frequency.md).

## 6. Trust, but verify

`aggregate` recommends a cycle: specify, update, validate, adjust, use
(Mildenhall's SUVA-use). Its validation compares moments computed exactly from
the specification with moments estimated from the FFT output and reports "not
unreasonable". prospicio's `fft` returns the same audit in a `CompoundReport`,
and the closed-form side is a few lines:

```python
n = 250.0
exact_mean = n * claim.mean()
exact_var = n * (claim.variance() + claim.mean() ** 2)  # Poisson: n * E[X^2]
print(f"mean: exact {exact_mean:.4f}, grid {casualty.mean():.4f}")
print(f"variance: exact {exact_var:.1f}, grid {casualty.variance():.1f}")
print(f"aliasing error below 1e-15: {report.aliasing_error < 1e-15}, tail mass {report.tail_mass:.1e}")
```

```text
mean: exact 24534.1289, grid 24534.1289
variance: exact 6562423.0, grid 6562423.0
aliasing error below 1e-15: True, tail mass 0.0e+00
```

For a continuous severity the gap is discretization error, controlled by the
step. The aliasing error says whether the FFT buffer was long enough, and the
tail mass is the probability that fell off the end of the grid. Read both every
time, as [Aggregate distributions](aggregate-distributions.md) explains.

## 7. Limits, attachments and increased limits factors

Policy terms are part of the language in DecL. Here they are a mapping on the
grid, and an increased-limits-factor table is a loop over the limit. The
session's loop, `100 claims lim xs 0 sev lognorm 100 cv 1.5`:

```python
base_grid, _ = Grid.local_moment(Lognormal.from_mean_cv(100.0, 1.5), STEP, POINTS)
base = None
for limit in (500.0, 1000.0, 2000.0, 5000.0):
    capped, _ = base_grid.map(lambda x, cap=limit: min(x, cap))
    year, _ = fft(Poisson(100.0), capped, POINTS)
    base = base or year.mean()
    print(f"limit {limit:6,.0f}   E[S] {year.mean():9,.0f}   ILF {year.mean() / base:.3f}")
```

```text
limit    500   E[S]     9,334   ILF 1.000
limit  1,000   E[S]     9,814   ILF 1.051
limit  2,000   E[S]     9,963   ILF 1.067
limit  5,000   E[S]     9,997   ILF 1.071
```

`aggregate` prints 9,334, 9,814, 9,963 and 9,997, with factors 1.000, 1.051,
1.067 and 1.071.

A layer that attaches above zero, `300 xs 200`, has two readings in DecL. The
default is the severity conditional on exceeding the attachment, with the
claim count thinned by the probability of exceeding it. A trailing `!` keeps
every claim, most of them paying nothing. For a Poisson count the two give the
same annual distribution, because thinning a Poisson count gives a Poisson
count with a smaller mean. In prospicio you take the second route and keep the
full count; the check below builds the first route by hand:

```python
full, _ = Grid.local_moment(Lognormal.from_mean_cv(100.0, 1.5), STEP, POINTS)
layered, _ = full.map(lambda x: min(max(x - 200.0, 0.0), 300.0))  # 300 xs 200
p0 = layered.probs[0]
conditional = Grid(STEP, [0.0] + [p / (1.0 - p0) for p in layered.probs[1:]])
unconditional, _ = fft(Poisson(40.0), layered, POINTS)
thinned, _ = fft(Poisson(40.0 * (1.0 - p0)), conditional, POINTS)
print(f"claims reaching the layer: {40.0 * (1.0 - p0):.3f} of 40")
print(f"mean {unconditional.mean():.4f} vs {thinned.mean():.4f}")
print(f"99% {unconditional.quantile(0.99):.0f} vs {thinned.quantile(0.99):.0f}")
```

```text
claims reaching the layer: 4.732 of 40
mean 631.2392 vs 631.2392
99% 1663 vs 1663
```

With a count that is not closed under thinning (a binomial or a fixed count)
the two readings differ, and the conditional one needs the thinned count.

## 8. Reinsurance, gross to net

The session's `Book`, ten claims limited to 1,000 with lognormal severity of
mean 100 and CV 2, is reinsured per occurrence by half of 300 xs 200 and all of
500 xs 500:

```text
agg Book 10 claims 1000 xs 0 sev lognorm 100 cv 2
    occurrence net of 50% po 300 xs 200 and 100% po 500 xs 500 poisson
```

A `Layer` is a limit, an attachment and a placed share; a `Tower` applies its
layers to the grid and returns the ceded loss of each and, when there are no
annual terms, the net.

```python
book_claim = limited_severity(Lognormal.from_mean_cv(100.0, 2.0), 1000.0, STEP, 2**14)
programme = Tower(
    [
        Layer("half of 300 xs 200", 300.0, 200.0, share=0.5),
        Layer("500 xs 500", 500.0, 500.0),
    ]
)
cover = programme.on_grid(Poisson(10.0), book_claim, 2**14)
gross, _ = fft(Poisson(10.0), book_claim, 2**14)
ceded_mean = sum(c.mean() for c in cover.ceded)
print(f"gross mean {gross.mean():.2f}, ceded {ceded_mean:.2f}, net {cover.net.mean():.2f}")
print(f"net sd {cover.net.variance() ** 0.5:.2f}, net 99% {cover.net.quantile(0.99):.1f}")
for layer, ceded in zip(programme.layer_names, cover.ceded):
    print(f"{layer:20s} mean {ceded.mean():7.3f}  sd {ceded.variance() ** 0.5:7.3f}")
```

```text
gross mean 952.70, ceded 161.52, net 791.18
net sd 369.93, net 99% 1816.0
half of 300 xs 200   mean  88.497  sd 100.878
500 xs 500           mean  73.019  sd 168.066
```

`aggregate` prints a gross mean of 952.7, a ceded mean of 161.52, and a net mean
of 791.18 with a standard deviation of 369.93. The means are exact on the grid;
`cover.on_points` says whether every recovery landed on a grid point, and when
it is `False` the shapes can smear by up to one step while the means stay exact.

An annual cover is a `Layer.stop_loss`, applied to the year's total. The
session's stop loss attaches at 30,000 on a book averaging 9,814, which never
pays; an attachment inside the distribution makes the point:

```python
year, _ = fft(Poisson(100.0), book, POINTS)
stop = Tower([Layer.stop_loss("5,000 xs 10,000", 5000.0, 10000.0)]).on_grid(
    Poisson(100.0), book, POINTS
)
print(f"gross mean {year.mean():.1f}, 99% {year.quantile(0.99):.0f}")
print(f"stop loss pays {stop.ceded[0].mean():.2f} on average")
print(f"net mean {stop.net.mean():.1f}, net 99% {stop.net.quantile(0.99):.0f}")
```

```text
gross mean 9813.7, 99% 13949
stop loss pays 558.87 on average
net mean 9254.8, net 99% 10000
```

Per-layer statistics of every kind (how often a layer is hit, loss on line,
ceded and net moments by view) and the contract terms are in
[What a reinsurance price is](reinsurance-price.md) and
[Experience and exposure rating](experience-and-exposure-rating.md).

## 9. The head to head: FFT against Monte Carlo

The session's strongest argument is a comparison. Simulate the casualty book
five times at 20,000 years each and read the 99.9% quantile:

```python
exact = casualty.quantile(0.999)
print(f"FFT 99.9%: {exact:,.0f}")
for seed in (4817, 2531, 9026, 311, 7742):
    totals = simulate_events(Poisson(250.0), claim, 20_000, seed).totals()
    estimate = totals.quantile(0.999)
    print(f"simulation, seed {seed}: {estimate:,.0f}  ({estimate / exact - 1:+.2%})")
```

```text
FFT 99.9%: 33,170
simulation, seed 4817: 33,145  (-0.08%)
simulation, seed 2531: 33,146  (-0.07%)
simulation, seed 9026: 33,214  (+0.13%)
simulation, seed 311: 32,909  (-0.79%)
simulation, seed 7742: 32,882  (-0.87%)
```

`aggregate` prints 33,169 for the FFT. The unit of difference is the quantile
convention where the distribution function crosses 0.999 inside a bucket; at a
step of 1 it is the resolution of the answer. Each simulated estimate rests on
the twenty largest of 20,000 years, so it moves by a percent or so from run to
run, and that matters when it sets capital. The FFT value is the same every run,
and refreshing it after a change to a limit or a treaty costs the same.
Simulation is still the tool when events need an order, a date or a sum insured,
or when dependence cannot be put on a grid (see
[Aggregate distributions](aggregate-distributions.md)).

## 10. Portfolios, capital and a first price

A `port` in DecL convolves independent units on one grid. prospicio's
equivalent is `Portfolio.from_independent`, which takes units on grids of the
same step. The session's book has a property and a casualty unit, the casualty
count mixed by a gamma of CV 0.4:

```text
port Book
    agg Property 80 claims 500 xs 0 sev lognorm 50 cv 1.2 poisson
    agg Casualty 20 claims 2000 xs 0 sev lognorm 250 cv 2.0 mixed gamma 0.4
```

```python
property_unit, _ = fft(
    Poisson(80.0), limited_severity(Lognormal.from_mean_cv(50.0, 1.2), 500.0), POINTS
)
casualty_unit, _ = fft(
    Count.mixed_poisson(20.0, 0.4),
    limited_severity(Lognormal.from_mean_cv(250.0, 2.0), 2000.0),
    POINTS,
)
book_port = Portfolio.from_independent(
    ["Property", "Casualty"], [property_unit, casualty_unit]
)
stand_alone = property_unit.quantile(0.996) + casualty_unit.quantile(0.996)
print(f"99.6% VaR, units alone summed {stand_alone:,.0f}")
print(f"99.6% VaR, portfolio         {book_port.assets(0.996):,.0f}")
```

```text
99.6% VaR, units alone summed 19,989
99.6% VaR, portfolio         18,137
```

`aggregate` prints 19,989 and 18,137. The gap is the diversification benefit.
The session then asks for a price that backs capital with the 99.6% VaR and
targets a 10% return on it (`price_pentagon`). The **pentagon** is loss `L`,
margin `M`, premium `P = L + M`, capital `Q` and assets `a = P + Q`, with the
ratios between them. A constant cost of capital of 10% on assets at the 99.6%
quantile prices the book, and `Pentagon.solve` completes the identity from
three of its quantities:

```python
priced = book_port.price(Distortion.ccoc(0.10), p=0.996)
total = priced.total
pentagon = Pentagon.solve(
    loss=total.loss, assets=total.assets, return_on_capital=0.10
)
print(
    f"L {pentagon.loss:.1f}  M {pentagon.margin:.2f}  P {pentagon.premium:.1f}  "
    f"Q {pentagon.capital:.1f}  a {pentagon.assets:,.0f}"
)
print(f"loss ratio {pentagon.loss_ratio:.5f}, premium to capital {pentagon.premium_to_capital:.4f}")
```

```text
L 8639.7  M 863.39  P 9503.1  Q 8633.9  a 18,137
loss ratio 0.90915, premium to capital 1.1007
```

`aggregate` prints `L` 8,639.7, `M` 863.39, `P` 9,503.1, `Q` 8,633.9, assets
18,137, a loss ratio of 0.90915 and a premium to capital of 1.1007. Note that
`L` is the expected loss **capped at the assets** (limited liability), a little
under the uncapped mean:

```python
print(f"uncapped mean {property_unit.mean() + casualty_unit.mean():.1f}, capped at assets {total.loss:.1f}")
for name, unit in zip(priced.units, priced.allocated):
    print(f"{name:9s} loss {unit.loss:8.1f}  premium {unit.premium:8.1f}  capital {unit.capital:8.1f}")
```

```text
uncapped mean 8646.1, capped at assets 8639.7
Property  loss   3972.3  premium   3969.3  capital    -30.3
Casualty  loss   4667.4  premium   5533.8  capital   8664.2
```

The natural allocation splits premium and capital between the units. Which
distortion to choose, how to calibrate it to a quote, and how to allocate are in
[Risk loads and distortions](risk-loads-and-distortions.md).

## 11. Everything is an object you can read

`aggregate` returns pandas DataFrames and matplotlib figures. prospicio returns
Python objects with methods; the grid's probabilities are a list, so a frame or
a plot is one line away in the library you prefer:

```python
print([round(p, 6) for p in casualty.probs[20_000:20_003]])
print(f"F(20,000) {casualty.cdf(20_000.0):.6f}")
```

```text
[3.2e-05, 3.2e-05, 3.2e-05]
F(20,000) 0.031989
```

`aggregate` prints a probability of 0.000032 per unit near 20,000 and
`F(20,000)` of 0.031989.

## 12. DecL clauses and their prospicio equivalents

| DecL | What it says | prospicio |
|---|---|---|
| `250 claims` | expected claim count | the count's mean: `Poisson(250.0)` |
| `500 loss` | expected loss, count derived | `Poisson(500.0 / claim.mean())` on the limited severity's mean |
| `1000 prem at .65 lr` | premium and loss ratio | expected loss `1000 * 0.65`, then as `500 loss` |
| `1000 xs 0` | per-claim limit | `grid.map(lambda x: min(x, 1000.0))` |
| `130 xs 20` | a layer, claims thinned | `grid.map(lambda x: min(max(x - 20.0, 0.0), 130.0))` with the full count (section 7) |
| `[10 20 30] claims [100 200 75] xs [0 50 75]` | a limit profile | one limited grid per limit, then `Portfolio.from_independent` or a weighted `Grid` |
| `sev lognorm 100 cv 1.5` | severity by mean and CV | `Lognormal.from_mean_cv(100.0, 1.5)`; `Gamma.from_mean_cv` and others |
| `sev lognorm [50 100] cv [1 2] wts [.5 .5]` | a mixture | a weighted average of grids (section 4), or `Mixture` for a continuous severity |
| `dsev [1 2 10] [.4 .4 .2]`, `dfreq [1:5]` | discrete outcomes | `Grid(step, probs)`, `Discrete`, `Count.empirical` |
| `poisson`, `negbin`, `binomial`, `geometric` | count families | `Poisson`, `NegativeBinomial`, `Binomial`, `Count` |
| `mixed gamma 0.4` | mixed Poisson | `Count.mixed_poisson(mean, 0.4)`; `mixing="inverse_gaussian"` |
| `zt`, `zm .5` | zero-truncated and zero-modified counts | `Count.zero_truncated(base)`, `Count.zero_modified(base, 0.5)` |
| `occurrence net of 50% po 300 xs 200` | per-claim cover, net | `Layer(name, 300.0, 200.0, share=0.5)` in a `Tower`, `.on_grid(...)`, `.net` |
| `aggregate net of 5000 xs 10000` | annual cover | `Layer.stop_loss(name, 5000.0, 10000.0)` |
| `occurrence ceded to tower [10:50:10]` | a tower of layers | a `Tower` of `Layer`s |
| `reinstatements 1 free and 1 at 50%` | reinstatements | `Layer(..., reinstatements=1)`, `reinstatement_rates=[...]` |
| `swing`, `slide`, `pc`, `corridor`, `deposit`, `rol`, `rate`, `cede` | contract terms | the `Layer.with_*` methods (`with_swing_rating`, `with_sliding_scale`, `with_profit_commission`, `with_loss_corridor`, `with_deposit_premium`, `with_rate_on_line`, `with_ceding_commission`) |
| `approximate sgamma` | moment-matched aggregate | `compound_moments(freq, sev).approximate("shifted_gamma")` |
| `tweedie p mean dispersion` | a Tweedie aggregate | `Tweedie` in `prospicio.distributions` |
| `port ... agg ... agg ...` | independent units | `Portfolio.from_independent(names, grids)` |
| `bivariate`, `clash` | two lines, one count | `fft_bivariate`, `Clash` |
| `pnl`, `xpnl` | premium less loss | `Ledger`, `Book`, `pnl_program` in `prospicio.aggregate` |
| `distortion D ph 0.9` | a distortion | `Distortion.proportional_hazard(0.9)`, `.wang`, `.ccoc`, `.tvar` and others |
| `port.price(p, ROE)`, `price_pentagon` | capital-loaded price | `Portfolio.price(distortion, p=...)`, `Pentagon.solve` |
| `hints{bs=1/64; log2=16}` | the grid | the `step` and `points` arguments |

## 13. Speed, and what prospicio does not have

On the workloads in this tour a release build of prospicio ran 1.1 to 2.1 times
faster than `aggregate` 0.30.1 end to end (a 2^16 casualty book 38 ms against
77 ms; a 2^20 grid 0.66 s against 1.0 s; a two-layer tower on a 2^14 grid about
equal). The FFT itself is numpy's in `aggregate`, so the gain is modest, and a
debug build (`maturin develop` without `--release`) is about ten times slower
than `aggregate`. These timings are one Windows machine and are not checked by CI.

Checked against the 1.2.0 documentation, prospicio lacks:


- **A DecL front end and recipe base.** There is no string language, `build`
  or `%%agg` cell magic, and no library of named programs to browse. Models
  are Python code.
- **Automatic grids for a limited severity or a portfolio.** `fft_auto` sizes
  the grid for an unlimited severity; with a limit or a portfolio you pick the
  step, and read `aliasing_error` and `tail_mass` (section 6). `audit_grid`
  (and `audit_portfolio_grid`) plays the part of `sharpen`: it builds the
  compound at your step and at doubled and halved ones, scores each by moment
  errors against the exact moments, and returns the best sound cell as a
  `GridAudit`. It audits a grid you give it; it does not choose one for you.
- **Shifted and scaled severities in a distribution.** `10 * lognorm 1 + 20`
  is a transformation on a grid here (`Grid.map`), which keeps the mean and
  loses the continuous form.
- **Exact rational compounds** (`nttagg`), the `cantor` severity, and the
  exhibit and chart documents.

A shared mixing variable is not on that list. In `aggregate` 1.2.0 it is shared
across the severity components of one `Aggregate` (one mixed count on the
claim-weighted mixture) and across the two units of a bivariate, never across
the units of a `Portfolio`, whose total is the product of their transforms.
prospicio has both: `fft` of a mixed Poisson count on a `Mixture`, and
`fft_bivariate` with a mixed count. `Portfolio.from_independent` treats units as
independent, as `aggregate`'s portfolios do.

Where to go next: [Aggregate distributions](aggregate-distributions.md) for
sizing grids and the engines' failure modes,
[What a reinsurance price is](reinsurance-price.md) for the contract terms and
the technical premium, and
[Risk loads and distortions](risk-loads-and-distortions.md) for the pentagon
and the pricing theory behind it.
