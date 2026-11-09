# prospicio for aggregate users: the twenty-minute tour

<!-- site: execute -->

Use this guide if you work with Stephen Mildenhall's package
[`aggregate`](https://aggregate.readthedocs.io/) and want the same work in
prospicio. It also teaches the collective risk model to readers new to both. It
follows the "`aggregate` in Twenty Minutes" session (release 1.2.0) step by
step, with the published numbers beside prospicio's. Where the two differ, the
cause is a convention.

Money is in the units of that session (think thousands).

## 1. The idea, and what is different

Both libraries compute the distribution of the total loss of a year,
`S = X_1 + ... + X_N`. They discretize the severity onto a grid, apply the
probability generating function of the count with a fast Fourier transform
(FFT), and read everything off the resulting probabilities. They differ in how
you ask:

| | `aggregate` | prospicio |
|---|---|---|
| Model written as | a DecL program (a string): `agg Casualty 250 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson` | Python objects: a count, a severity, a grid |
| Grid | chosen for you (`bs`, `log2`), overridable with `hints{}` | `fft_auto` sizes it for an unlimited severity. With a limit, a layer or a portfolio you give the step and the number of points. `audit_grid` re-runs the build around it as `sharpen` does |
| Result | an `Aggregate` object holding DataFrames | a `Grid` (the distribution) and a `CompoundReport` (the checks) |
| Reinsurance | clauses in the program (`occurrence net of ...`) | `Layer` and `Tower` objects applied to the grid |

Two habits carry over unchanged. **Validate the gross distribution before you
reinsure it**: the closed form you validate against disappears once you apply a
cover. And **an FFT answer is deterministic**: the same inputs give the same
digits every run.



## 2. Exact and discrete: the dice

`aggregate`'s first example is `agg Eg1 dfreq [1:5] dsev [1:3]`: a count of 1 to
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

Every bar is an exact probability.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(range(1, 16), eg1.probs[1:16])
ax.set_xlabel("total of the year (claim units)")
ax.set_ylabel("probability")
ax.set_title("Eg1: the total of 1 to 5 claims of size 1 to 3")
```

The smallest outcome is one claim of size one, `1/5 * 1/3`. The largest is five
claims of size three, `1/5 * (1/3)^5`. The grid returns both exactly. The method
is exact for a discrete severity on a grid, so there is nothing to validate.

## 3. The casualty book

The main example of the session is a book of 250 claims. The severity is
lognormal with mean 100 and coefficient of variation (CV) 1.5, and each claim
has a limit of 1,000:

```text
agg Casualty 250 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson
```

In prospicio this is three lines. The limit is a mapping on the discretized
severity (`Grid.map` keeps the mean exactly). The count is a `Poisson`, and
`fft` compounds them. The step of 1 and `2**16` points are what `aggregate`
uses for this book.

```python
import pandas as pd
from great_tables import GT

STEP, POINTS = 1.0, 2**16


def limited_severity(severity, limit, step=STEP, points=POINTS):
    """The severity capped at limit, on a grid: `limit xs 0`."""
    grid, _ = Grid.local_moment(severity, step, points)
    capped, _ = grid.map(lambda x: min(x, limit))
    return capped


def compare(rows, title, subtitle):
    """A table of (label, prospicio, book, decimals[, book decimals]) rows; a book value of None stays blank."""
    frame = pd.DataFrame([r[:3] for r in rows], columns=["measure", "prospicio", "book"])
    table = GT(frame, rowname_col="measure")
    for i, row in enumerate(rows):
        table = table.fmt_number("prospicio", rows=[i], decimals=row[3])
        table = table.fmt_number("book", rows=[i], decimals=row[4] if len(row) > 4 else row[3])
    return table.sub_missing(missing_text="").tab_header(title, subtitle)


claim = limited_severity(Lognormal.from_mean_cv(100.0, 1.5), 1000.0)
casualty, report = fft(Poisson(250.0), claim, POINTS)
quantiles = {p: casualty.quantile(p) for p in (0.01, 0.5, 0.99, 0.999)}
compare(
    [
        ("claim mean", claim.mean(), 98.137, 3),
        ("annual mean", casualty.mean(), 24534, 1, 0),
        ("annual sd", casualty.variance() ** 0.5, 2561.7, 1),
        ("1% quantile", quantiles[0.01], 18954, 0),
        ("median", quantiles[0.5], 24449, 0),
        ("99% quantile", quantiles[0.99], 30862, 0),
        ("99.9% quantile", quantiles[0.999], 33169, 0),
    ],
    "The casualty book against aggregate 1.2.0",
    "Claim units",
)
```

```text
                prospicio     book
claim mean         98.137   98.137
annual mean      24,534.1   24,534
annual sd         2,561.7  2,561.7
1% quantile        18,954   18,954
median             24,449   24,449
99% quantile       30,862   30,862
99.9% quantile     33,170   33,169
```

The two agree to the digits shown, except the 99.9% quantile (section 9).

```python
fig, ax = plt.subplots()
probs = casualty.probs
ax.plot(range(15_000, 36_001), probs[15_000:36_001])
for p, q in quantiles.items():
    ax.axvline(q, color="0.5", linestyle=":")
    height = 0.05 if p == 0.5 else 0.35
    ax.text(q, height, f" {p:.1%}", transform=ax.get_xaxis_transform(), color="0.4")
ax.xaxis.set_major_formatter("{x:,.0f}")
ax.set_xlabel("annual loss")
ax.set_ylabel("probability per unit of loss")
ax.set_title("The casualty book: distribution of the annual total")
```

## 4. A vocabulary of severities

`aggregate` takes any `scipy.stats` family by mean and CV, and it takes mixtures
by weights. The continuous families of prospicio are in `prospicio.distributions`
(`Lognormal`, `Gamma`, `Weibull`, `Pareto`, `Burr` and others, many with a
`from_mean_cv`). A mixture of limited severities is the weighted sum of their
grids. The session's three-component book,

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
compare(
    [
        ("claim mean", mix_claim.mean(), 92.996, 3),
        ("annual mean", mix.mean(), 9299.6, 1),
        ("annual sd", mix.variance() ** 0.5, 1894.5, 1),
        ("1% quantile", mix.quantile(0.01), 5638, 0),
        ("median", mix.quantile(0.5), 9130, 0),
        ("99% quantile", mix.quantile(0.99), 14423, 0),
    ],
    "The three-component book against aggregate 1.2.0",
    "Claim units",
)
```

```text
              prospicio     book
claim mean       92.996   92.996
annual mean     9,299.6  9,299.6
annual sd       1,894.5  1,894.5
1% quantile       5,638    5,638
median            9,130    9,130
99% quantile     14,423   14,423
```

Tail shape has a visible price. Two books with the same mean and CV differ at the 99.9% level:

```python
from prospicio.distributions import Gamma

tails = []
for name, family, published in (
    ("lognormal", Lognormal.from_mean_cv(100.0, 2.0), 13980),
    ("gamma", Gamma.from_mean_cv(100.0, 2.0), 11166),
):
    book, _ = fft(Poisson(50.0), limited_severity(family, 20000.0), POINTS)
    tails.append((f"{name}, mean", book.mean(), None, 0))
    tails.append((f"{name}, 99.9%", book.quantile(0.999), published, 0))
compare(tails, "Same mean and CV, different tail", "50 claims, CV 2; claim units")
```

```text
                  prospicio    book
lognormal, mean       5,000
lognormal, 99.9%     13,979  13,980
gamma, mean           5,000
gamma, 99.9%         11,166  11,166
```

The digits can differ by the cap, but the ordering is the point: the
lognormal's 99.9% sits well above the gamma's.

## 5. Frequency: Poisson is an opinion

A Poisson count says that years differ only by chance. Years can also differ for
reasons that all claims share (weather, inflation, the legal climate). Then the
count is over-dispersed. `mixed gamma 0.4` in DecL multiplies the Poisson rate
by a gamma factor of mean 1 and CV 0.4. In prospicio:

```python
book = limited_severity(Lognormal.from_mean_cv(100.0, 1.5), 1000.0)
poisson, _ = fft(Poisson(100.0), book, POINTS)
mixed, _ = fft(Count.mixed_poisson(100.0, 0.4), book, POINTS)


def agg_cv(grid):
    return grid.variance() ** 0.5 / grid.mean()


frame = pd.DataFrame(
    {
        "measure": ["agg CV", "agg CV, book", "99.5% VaR", "99.5% VaR, book"],
        "Poisson": [agg_cv(poisson), 0.165, poisson.quantile(0.995), 14456],
        "mixed gamma": [agg_cv(mixed), 0.433, mixed.quantile(0.995), 23914],
    }
)
(
    GT(frame, rowname_col="measure")
    .fmt_number(["Poisson", "mixed gamma"], rows=[0, 1], decimals=3)
    .fmt_number(["Poisson", "mixed gamma"], rows=[2, 3], decimals=0)
    .tab_header("Poisson against mixed gamma count", "Annual loss in claim units")
)
```

```text
                 Poisson  mixed gamma
agg CV             0.165        0.433
agg CV, book       0.165        0.433
99.5% VaR         14,456       23,914
99.5% VaR, book   14,456       23,914
```

Same mean, capital about two-thirds higher.

```python
fig, ax = plt.subplots()
xs = range(0, 40_001)
ax.plot(xs, poisson.probs[:40_001], label="Poisson")
ax.plot(xs, mixed.probs[:40_001], label="mixed gamma")
ax.xaxis.set_major_formatter("{x:,.0f}")
ax.set_xlabel("annual loss")
ax.set_ylabel("probability per unit of loss")
ax.set_title("Poisson against mixed gamma count: same mean, longer tail")
ax.legend()
```

`Count` also builds other mixing families, zero-modified and zero-truncated
counts, the logarithmic and Neyman type A (`Count.zero_truncated`,
`Count.zero_modified`, `Count.logarithmic`, and `Count.mixed_poisson` with
`mixing="inverse_gaussian"`). See [Severity and frequency](severity-and-frequency.md).

## 6. Trust, but verify

`aggregate` validates (the V in Mildenhall's SUVA-use cycle) by comparing moments computed exactly from the specification with moments estimated from the FFT output. The `fft` of prospicio returns the same audit in a `CompoundReport`. The closed form:

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

For a continuous severity, the gap is discretization error, which the step
controls. The aliasing error shows whether the FFT buffer was long enough. The
tail mass is the probability that fell off the end of the grid. Read both every
time, as [Aggregate distributions](aggregate-distributions.md) explains.

## 7. Limits, attachments and increased limits factors

Policy terms are a mapping on the grid, and increased limits factors are a loop over the limit (the session's `100 claims lim xs 0 sev lognorm 100 cv 1.5`):

```python
base_grid, _ = Grid.local_moment(Lognormal.from_mean_cv(100.0, 1.5), STEP, POINTS)
limits = [500.0, 1000.0, 2000.0, 5000.0]
means = []
for limit in limits:
    capped, _ = base_grid.map(lambda x, cap=limit: min(x, cap))
    year, _ = fft(Poisson(100.0), capped, POINTS)
    means.append(year.mean())
ilf = pd.DataFrame(
    {
        "limit": [f"{limit:,.0f}" for limit in limits],
        "E[S]": means,
        "E[S], book": [9334, 9814, 9963, 9997],
        "ILF": [m / means[0] for m in means],
        "ILF, book": [1.000, 1.051, 1.067, 1.071],
    }
)
(
    GT(ilf, rowname_col="limit")
    .fmt_number(["E[S]", "E[S], book"], decimals=0)
    .fmt_number(["ILF", "ILF, book"], decimals=3)
    .tab_header("Increased limits factors", "100 claims, lognormal severity, mean 100 and CV 1.5")
)
```

```text
        E[S]  E[S], book    ILF  ILF, book
500    9,334       9,334  1.000      1.000
1,000  9,814       9,814  1.051      1.051
2,000  9,963       9,963  1.067      1.067
5,000  9,997       9,997  1.071      1.071
```

A layer that attaches above zero, `300 xs 200`, has two readings in DecL. The
default is the severity conditional on exceeding the attachment, with the claim
count thinned by the probability of exceeding it. A trailing `!` keeps every
claim, most of them paying nothing. For a Poisson count the two give the same
annual distribution, because thinning a Poisson count gives a Poisson count with
a smaller mean. prospicio takes the second route and keeps the full count. The
check builds the first route by hand:

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

A binomial or fixed count is not closed under thinning. With such a count the
two readings differ, and the conditional one needs the thinned count.

## 8. Reinsurance, gross to net

The `Book` of the session has ten claims limited to 1,000, with lognormal
severity of mean 100 and CV 2. A reinsurer takes half of 300 xs 200 and all of
500 xs 500 per occurrence:

```text
agg Book 10 claims 1000 xs 0 sev lognorm 100 cv 2
    occurrence net of 50% po 300 xs 200 and 100% po 500 xs 500 poisson
```

A `Layer` is a limit, an attachment and a placed share. A `Tower` applies its
layers to the grid. It returns the ceded loss of each layer and, when there are
no annual terms, the net.

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
layers = [(c.mean(), c.variance() ** 0.5) for c in cover.ceded]
nan = float("nan")
tower = pd.DataFrame(
    {
        "view": ["gross", *programme.layer_names, "ceded, both layers", "net"],
        "mean": [gross.mean(), *[m for m, _ in layers], ceded_mean, cover.net.mean()],
        "mean, book": [952.7, nan, nan, 161.52, 791.18],
        "sd": [nan, *[s for _, s in layers], nan, cover.net.variance() ** 0.5],
        "sd, book": [nan, nan, nan, nan, 369.93],
        "99%": [nan, nan, nan, nan, cover.net.quantile(0.99)],
    }
)
(
    GT(tower, rowname_col="view")
    .fmt_number("mean", rows=[0, 3, 4], decimals=2)
    .fmt_number("mean", rows=[1, 2], decimals=3)
    .fmt_number("mean, book", decimals=2, drop_trailing_zeros=True)
    .fmt_number("sd", rows=[1, 2], decimals=3)
    .fmt_number("sd", rows=[4], decimals=2)
    .fmt_number("sd, book", decimals=2)
    .fmt_number("99%", decimals=1)
    .sub_missing(missing_text="")
    .tab_header("The Book, gross to net", "Per occurrence; claim units")
)
```

```text
                      mean  mean, book       sd  sd, book      99%
gross               952.70       952.7
half of 300 xs 200  88.497              100.878
500 xs 500          73.019              168.066
ceded, both layers  161.52      161.52
net                 791.18      791.18   369.93    369.93  1,816.0
```

The means are exact on the grid. `cover.on_points` says whether every recovery
landed on a grid point. When it is `False`, the shapes can smear by up to one
step while the means stay exact.

An annual cover is a `Layer.stop_loss`, applied to the total of the year. The
stop loss of the session attaches at 30,000 on a book that averages 9,814, so it
never pays. An attachment inside the distribution makes the point:

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

Per-layer statistics (how often a layer is hit, loss on line, ceded and net
moments by view) and the contract terms are in
[What a reinsurance price is](reinsurance-price.md) and
[Experience and exposure rating](experience-and-exposure-rating.md).

## 9. The head to head: FFT against Monte Carlo

Simulate the casualty book five times at 20,000 years each and read the 99.9% quantile:

```python
exact = casualty.quantile(0.999)
seeds = (4817, 2531, 9026, 311, 7742)
estimates = [simulate_events(Poisson(250.0), claim, 20_000, s).totals().quantile(0.999) for s in seeds]
runs = pd.DataFrame(
    {
        "run": ["FFT", "FFT, book", *[f"simulation, seed {s}" for s in seeds]],
        "99.9%": [exact, 33169, *estimates],
        "against FFT": [None, None, *[e / exact - 1 for e in estimates]],
    }
)
(
    GT(runs, rowname_col="run")
    .fmt_number("99.9%", decimals=0)
    .fmt_percent("against FFT", decimals=2, force_sign=True)
    .sub_missing(missing_text="")
    .tab_header("The casualty book's 99.9% quantile", "FFT against 20,000-year simulations")
)
```

```text
                        99.9%  against FFT
FFT                    33,170
FFT, book              33,169
simulation, seed 4817  33,145       -0.08%
simulation, seed 2531  33,146       -0.07%
simulation, seed 9026  33,214       +0.13%
simulation, seed 311   32,909       -0.79%
simulation, seed 7742  32,882       -0.87%
```

The 1-unit gap to `aggregate`'s FFT is the quantile convention where the
distribution function crosses 0.999 inside a bucket. At a step of 1 it is the
resolution of the answer. Each simulated estimate rests on the twenty largest of
20,000 years, so it moves by a percent or so from run to run. That matters when it sets capital. The FFT value is the same every run. Simulation is still the tool when
events need an order, a date or a sum insured, or when dependence cannot be put
on a grid (see [Aggregate distributions](aggregate-distributions.md)).

## 10. Portfolios, capital and a first price

A `port` in DecL convolves independent units on one grid. In prospicio that is
`Portfolio.from_independent`, which takes units on grids of the same step. The
book of the session has a property unit and a casualty unit. A gamma of CV 0.4
mixes the casualty count:

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
print(f"99.6% VaR, units alone summed {stand_alone:,.0f} (aggregate: 19,989)")
print(f"99.6% VaR, portfolio         {book_port.assets(0.996):,.0f} (aggregate: 18,137)")
```

```text
99.6% VaR, units alone summed 19,989 (aggregate: 19,989)
99.6% VaR, portfolio         18,137 (aggregate: 18,137)
```

The gap is the diversification benefit. The session then asks for a price that
backs capital with the 99.6% VaR and targets a 10% return on it
(`price_pentagon`). The **pentagon** is loss `L`, margin `M`, premium
`P = L + M`, capital `Q` and assets `a = P + Q`, with the ratios between them. A
constant cost of capital of 10% on assets at the 99.6% quantile prices the book.
`Pentagon.solve` completes the identity from three of its quantities:

```python
priced = book_port.price(Distortion.ccoc(0.10), p=0.996)
total = priced.total
pentagon = Pentagon.solve(
    loss=total.loss, assets=total.assets, return_on_capital=0.10
)
compare(
    [
        ("loss L", pentagon.loss, 8639.7, 1),
        ("margin M", pentagon.margin, 863.39, 2),
        ("premium P", pentagon.premium, 9503.1, 1),
        ("capital Q", pentagon.capital, 8633.9, 1),
        ("assets a", pentagon.assets, 18137, 0),
        ("loss ratio", pentagon.loss_ratio, 0.90915, 5),
        ("premium to capital", pentagon.premium_to_capital, 1.1007, 4),
    ],
    "The pentagon of the book against aggregate 1.2.0",
    "CCoC 10% at the 99.6% quantile; claim units",
)
```

```text
                    prospicio     book
loss L                8,639.7  8,639.7
margin M               863.39   863.39
premium P             9,503.1  9,503.1
capital Q             8,633.9  8,633.9
assets a               18,137   18,137
loss ratio            0.90915  0.90915
premium to capital     1.1007   1.1007
```

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


[Risk loads and distortions](risk-loads-and-distortions.md) explains which
distortion to choose, how to calibrate it to a quote, and how to allocate.

## 11. Everything is an object you can read

The probabilities of a grid are a plain list, so a frame or a plot is one line away:

```python
print([round(p, 6) for p in casualty.probs[20_000:20_003]])
print(f"F(20,000) {casualty.cdf(20_000.0):.6f}")
```

```text
[3.2e-05, 3.2e-05, 3.2e-05]
F(20,000) 0.031989
```

`aggregate` prints the same: 0.000032 per unit near 20,000 and `F(20,000)` of
0.031989.

## 12. DecL clauses and their prospicio equivalents

| DecL | What it says | prospicio |
|---|---|---|
| `250 claims` | expected claim count | the count's mean: `Poisson(250.0)` |
| `500 loss` | expected loss, count derived | `Poisson(500.0 / claim.mean())` on the limited severity's mean |
| `1000 prem at .65 lr` | premium and loss ratio | expected loss `1000 * 0.65`, then as `500 loss` |
| `1000 xs 0` | per-claim limit | `grid.map(lambda x: min(x, 1000.0))` |
| `130 xs 20` | a layer, claims thinned | `grid.map(lambda x: min(max(x - 20.0, 0.0), 130.0))` with the full count (section 7) |
| `[10 20 30] claims [100 200 75] xs [0 50 75]` | a limit profile | one limited grid per limit, then `Portfolio.from_independent` or a weighted `Grid` |
| `sev lognorm 100 cv 1.5` | severity by mean and CV | `Lognormal.from_mean_cv(100.0, 1.5)`, `Gamma.from_mean_cv` and others |
| `sev lognorm [50 100] cv [1 2] wts [.5 .5]` | a mixture | a weighted average of grids (section 4), or `Mixture` for a continuous severity |
| `dsev [1 2 10] [.4 .4 .2]`, `dfreq [1:5]` | discrete outcomes | `Grid(step, probs)`, `Discrete`, `Count.empirical` |
| `poisson`, `negbin`, `binomial`, `geometric` | count families | `Poisson`, `NegativeBinomial`, `Binomial`, `Count` |
| `mixed gamma 0.4` | mixed Poisson | `Count.mixed_poisson(mean, 0.4)`, with `mixing="inverse_gaussian"` as an option |
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

## 13. Speed and the grid audit

On the workloads in this tour, a release build of prospicio ran 1.1 to 2.1 times
faster than `aggregate` 0.30.1 end to end. A 2^16 casualty book took 38 ms
against 77 ms. A 2^20 grid took 0.66 s against 1.0 s. A two-layer tower on a
2^14 grid took about the same time. `aggregate` uses numpy's FFT, so the gain is
modest. A debug build (`maturin develop` without `--release`) is about ten times
slower than `aggregate`. The timings come from one Windows machine, and CI does
not check them.

`fft_auto` sizes the grid for an unlimited severity. With a limit or a
portfolio, you pick the step and read `aliasing_error` and `tail_mass` (section
6). `audit_grid` (and `audit_portfolio_grid`) does the work of `sharpen`. It
builds the compound at your step and at doubled and halved steps, scores each
build by its moment errors against the exact moments, and returns the best sound
cell as a `GridAudit`. It audits the grid that you give it.

`aggregate` 1.2.0 shares a mixing variable across the severity components of one
`Aggregate` (one mixed count on the claim-weighted mixture) and across the two
units of a bivariate. It never shares it across the units of a `Portfolio`, whose
total is the product of their transforms. prospicio supports both: `fft` of a
mixed Poisson count on a `Mixture`, and `fft_bivariate` with a mixed count.
`Portfolio.from_independent` treats units as independent, as the portfolios of
`aggregate` do.



- [Aggregate distributions](aggregate-distributions.md): sizing grids and the
  failure modes of the engines.
- [What a reinsurance price is](reinsurance-price.md): the contract terms and the
  technical premium.
- [Risk loads and distortions](risk-loads-and-distortions.md): the pentagon and
  the pricing theory behind it.
