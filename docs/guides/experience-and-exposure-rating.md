# Experience rating, exposure rating and blending them

An excess of loss layer has two independent sources of evidence for its expected loss: what the cedant's own
large losses would have cost the layer in the past (**experience rating**), and what the cedant's book of risks
should cost it under a standard loss shape (**exposure rating**). Most layers need both, weighted by how far
each can be trusted. This guide says which to use when, how prospicio does each, how to weight them, and how to
reach layers the data never touched.

**Use this guide when** you have a layer to price and must decide where its expected loss comes from: how many
years of large losses the cedant gave you, whether you have a profile of sums insured or policy limits, how
often the layer has been hit, and what to do above the highest loss ever seen. It is written for an actuary who
knows rating plans, increased limits factors and credibility, and is new to reinsurance pricing. It stops at the
expected loss. The risk load is in [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md), the annual
total in [`aggregate-distributions.md`](aggregate-distributions.md), and fitting a severity in depth in
[`severity-and-frequency.md`](severity-and-frequency.md). Primary ratemaking (on-level premium, the overall rate
indication) is in section 1 of [`choosing-a-method.md`](choosing-a-method.md).

The layers here are plain per-risk or per-loss layers with no reinstatements, annual aggregate limit or
aggregate deductible, and no clash cover (several risks hit by one event). Those terms change the expected loss
and are in [`reinsurance-price.md`](reinsurance-price.md) and
[`aggregate-distributions.md`](aggregate-distributions.md); `Layer.ceded` takes them for a burning cost.

Money is in millions of a synthetic currency. Every number in the text is printed by the code in the guide,
which is run and checked in CI.

## The decision in one table

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| A working layer (many losses pierce it), several years of as-if large losses | Burning cost (a rate on on-level subject premium when the book's size changed), or a Pareto fitted to the losses | `Layer.ceded`, `RateHistory`, `Pareto.fit`, `CollectiveModel.layer_mean` | The cedant's own losses are the best evidence when there are enough |
| A higher layer, pierced by few or no losses | Exposure rating | `RiskProfile.expected_layer_loss` | Experience is silent; the profile and a curve are not |
| Property per risk, profile by sum insured | MBBEFD (Swiss Re) curves | `Mbbefd.swiss_re(c)`, `RiskProfile` | One parameter `c` sets how losses spread over the sum insured |
| A published or in-house curve held as a table | Tabulated curve | `TabulatedCurve` | Same use as MBBEFD, from the table you hold |
| Surplus treaty, or a per-risk layer on what a surplus leaves | Exposure, treaty applied risk by risk | `expected_surplus_loss`, `expected_layer_loss(..., retention, lines)` | The cession depends on each risk's sum insured |
| A band covering a wide range of sums insured | Spread the band between bounds | `RiskProfile(..., lower, upper, spread)` | One sum insured per band misprices layers inside the range |
| Casualty, profile by policy limit | ILFs and a severity curve per limit, the severity trended to the treaty period | `ilf`, `loss_elimination_ratio`, `severity_exposure_curve`, `TabulatedCurve` | The shape comes from a ground-up severity and changes with the limit |
| Experience and exposure both exist | Weight by the credibility of the experience in that layer | `LimitedFluctuation`, `credibility_weighted` | Credibility follows the number of claims in the layer, so it falls as the layer rises |
| Loss ratios of many cedants or years, with premiums | Bühlmann–Straub | `buhlmann_straub` | A complement from similar cedants, with the variance between them |
| Layers above the data, two credible layers below | Pareto extrapolation | `alpha_between_layers`, `pareto_extrapolation` | The slope between known layers carries upward, with judgement |
| A tower of layer losses or prices | One model for the whole tower | `match_tower` | Layer losses, frequency and tail become one consistent severity |
| References from several sources (experience, exposure, a frequency) | One model through all of them | `fit_references` | Reproduces each reference exactly, so check them first |
| A PML or return-period curve | A severity through the points | `fit_pml_curve` | The curve is the only information on size |
| A handful of losses | Exposure first, experience as a check, an alpha from outside | `Pareto.fit` (`censored`), `fit_pml_curve` | An alpha fitted to few losses moves a lot |

## 1. Words you need first

- **Large loss**: a claim above the **reporting threshold** (here 1), listed individually by the cedant.
- **As-if**: a historical loss restated to the prospective year: trended to the future price level, developed to
  ultimate, and measured against the contract being priced (its retention and limit, with any indexation). This
  guide starts from as-if losses. Producing them is not yet in prospicio (section 8).
- **Burning cost**: the average annual loss a layer would have paid on the as-if history, the layer's pure
  premium from experience.
- **Pierce**: a loss pierces a layer when it exceeds the attachment. The losses that pierce a layer are the only
  claims that tell you about it.
- **Exposure curve** `G(x)`: for a risk with maximum possible loss (**MPL**, the sum insured or policy limit),
  the share of its expected loss below the fraction `x` of the MPL. A loss divided by the MPL is its
  **destruction rate**. The share of a risk's expected loss in a layer is `G(top / MPL) - G(attachment / MPL)`.
- **ILF and LER**: the increased limits factor is the limited expected value (LEV) of a severity at one limit
  over the LEV at a basic limit. The loss elimination ratio is the LEV at a deductible over the mean.
- **Local Pareto alpha**: how fast the number of losses above a threshold falls as the threshold rises. A higher
  alpha is a lighter tail.
- **Working layer**: one that is hit often; a **higher layer** is hit rarely.

## 2. Experience rating

### What it is

Take each large loss of the as-if history, apply the layer, and average over the years: that is the burning
cost. The second form replaces the raw history with a fitted model, a Poisson count of large losses and a Pareto
severity, which gives the expected loss of any layer in closed form, including layers the history never reached.

The running example is ten years of as-if losses above 1 from a synthetic property cedant.

```python
import math

from prospicio.credibility import LimitedFluctuation, buhlmann_straub
from prospicio.distributions import Lognormal, Pareto, Poisson
from prospicio.pricing import (
    CollectiveModel, Mbbefd, RiskProfile, TabulatedCurve, alpha_between_layers,
    credibility_weighted, fit_pml_curve, fit_references, ilf, loss_elimination_ratio,
    match_tower, pareto_extrapolation, severity_exposure_curve, trend_factor,
)
from prospicio.reinsurance import Layer

HISTORY = {
    2016: [1.35, 1.39, 1.03, 1.03], 2017: [1.37, 1.53, 2.03, 1.04, 1.66, 1.22],
    2018: [3.58, 1.17], 2019: [2.16, 1.00], 2020: [1.28, 2.79], 2021: [2.12],
    2022: [2.73, 1.56, 1.55], 2023: [5.43], 2024: [6.59, 1.38],
    2025: [1.48, 1.17, 1.90, 1.07, 2.09],
}
LOSSES = [x for year in HISTORY.values() for x in year]
YEARS = len(HISTORY)
print(YEARS, len(LOSSES), max(LOSSES))
```

```text
10 28 6.59
```

Ten years, 28 losses, the largest 6.59. A layer is written limit xs attachment: `1 xs 1` pays the part of a loss
between 1 and 2, `5 xs 5` the part between 5 and 10.

```python
LAYERS = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (10.0, 10.0)]   # (limit, attachment)


def burning_cost(limit, attachment):
    layer = Layer("xl", limit, attachment)
    return sum(layer.ceded(year) for year in HISTORY.values()) / YEARS


def pierced(attachment):
    return sum(x > attachment for x in LOSSES)


for limit, att in LAYERS:
    print(f"{limit:g} xs {att:g}: losses above {pierced(att):2d}, burning cost {burning_cost(limit, att):.3f}")
```

```text
1 xs 1: losses above 27, burning cost 1.518
2 xs 2: losses above  9, burning cost 0.750
5 xs 5: losses above  2, burning cost 0.202
10 xs 10: losses above  0, burning cost 0.000
```

`Layer.ceded` takes the losses of one year and returns what the layer pays on them, so the sum over the years
divided by the number of years is the burning cost. It is exact for the history, which is its limit: no loss
reached 10 xs 10, so its burning cost is zero, and zero is not a price.

The fitted form uses the same losses to estimate a frequency and an alpha.

```python
alpha_hat = Pareto.fit(LOSSES, 1.0).alpha
severity = Pareto(1.0, alpha_hat)
fitted = CollectiveModel(Poisson(len(LOSSES) / YEARS), severity)
drawn_from = CollectiveModel(Poisson(3.0), Pareto(1.0, 1.6))   # the model that drew this history
print(f"alpha {alpha_hat:.3f}, {len(LOSSES) / YEARS:.1f} losses above 1 a year")
for limit, att in LAYERS:
    print(f"{limit:g} xs {att:g}: burning {burning_cost(limit, att):.3f}, "
          f"fitted {fitted.layer_mean(limit, att):.3f}, drawn from {drawn_from.layer_mean(limit, att):.3f}")
```

```text
alpha 1.880, 2.8 losses above 1 a year
1 xs 1: burning 1.518, fitted 1.453, drawn from 1.701
2 xs 2: burning 0.750, fitted 0.789, drawn from 1.122
5 xs 5: burning 0.202, fitted 0.352, drawn from 0.648
10 xs 10: burning 0.000, fitted 0.191, drawn from 0.427
```

`Pareto.fit(losses, t)` is the maximum likelihood alpha for losses at or above the threshold `t`.
`CollectiveModel(frequency, severity)` gives the expected loss of any layer (`layer_mean(limit, attachment)`).
The last column is the synthetic model that drew these ten years (three losses a year, alpha 1.6), which in
practice you would not know. These ten years were lighter than their source, so both estimates sit below it in
every layer, and the gap in ratio grows with height (1.453 against 1.701 at 1 xs 1, 0.191 against 0.427 at 10 xs
10). A short history can easily miss the tail.

The fit counts the loss of exactly 1.00 in 2019, so it sees 28 losses while 1 xs 1 is pierced by 27. A Poisson
count and one Pareto alpha above the threshold are assumptions to check before you price on them: the variance
of the yearly counts against their mean (a negative binomial if it is larger), the choice of threshold, and a mean
excess plot. When one alpha does not fit, `PiecewisePareto.fit(losses, thresholds)` fits one alpha per band of
size, and the `truncation` argument of `Pareto.fit` and `alpha_between_layers` caps the tail at a largest
possible loss. [`severity-and-frequency.md`](severity-and-frequency.md) covers these checks.

**Use it when** the layer is pierced by enough losses for the burning cost to be stable (section 5 turns that
into a number), the portfolio is stable or restated to today's, and the as-if work is done: trend, development
of open claims, indexation of retention and limit.

**Do not rely on it when** the layer has had few or no hits (use the fitted model for a frequency and an alpha
from the lower layers, never as the sole price), the book has changed (past losses then describe another
portfolio), or the losses are not as-if (a burning cost on untrended, undeveloped losses is biased low, and
prospicio will not warn you).

**Data needed**: individual losses above a reporting threshold, by year, on the prospective basis, and the
cedant's subject premium (or another exposure base) for each year. The threshold `t` is a modelling choice, and
the Pareto must fit above it. If sources report from different thresholds or policy limits capped some losses,
say so (section 7). Three steps of the as-if work are easy to miss:

- **Exposure volume.** The per-year average above assumes the book was the same size every year, as this
  synthetic one was. When it was not, quote the burning cost as a rate: the total as-if layer loss over the total
  subject premium of the same years, each year's premium brought to today's rate level (`RateHistory`, section 1
  of [`choosing-a-method.md`](choosing-a-method.md)), and multiply the rate by the prospective subject premium.
  Summing before dividing weights each year by its premium.
- **The threshold after trend.** Trend raises losses across the threshold: a loss just under 1 at its own date
  may be over 1 at today's cost. Ask for losses from a lower threshold (1 divided by the largest trend factor),
  trend them, and only then keep those above 1, or the frequency above 1 is understated.
- **Legal cost (ALAE).** State whether it is in the loss (and so erodes the layer), shared pro rata with the
  indemnity, or excluded, and use the same basis for the loss ratio of the exposure rating.

## 3. Exposure rating: property

### What it is

Exposure rating needs no loss history. The cedant gives a **risk profile**: bands of sum insured with the number
of risks and premium in each. Each band gets an expected loss (premium times a loss ratio) and an exposure
curve, and the curve says what share of each risk's expected loss lies in the layer. It plays the part that a
rating plan's limits factors play for a primary policy.

The premium is the cedant's gross premium for the perils the treaty covers, at the prospective rate level, before
any commission. The loss ratio is the expected ground-up loss ratio to that premium for the treaty year,
excluding catastrophe losses and on the same ALAE basis as the layer. A loss ratio net of commission, or one that
includes catastrophe years, misstates every layer in proportion. Per-risk exposure rating prices a layer that
each risk triggers alone; it does not price a catastrophe (per-event) layer, where one event hits many risks at
once. That needs a catastrophe model or the PML route of section 6.

In prospicio the profile is `RiskProfile` and the curve is `Mbbefd` or `TabulatedCurve`. Here are four bands
with a 55% loss ratio, and the four layers of section 2.

```python
SUMS_INSURED = [2.0, 5.0, 10.0, 20.0]
RISKS = [1500, 400, 120, 30]
PREMIUM = [7.2, 4.8, 3.6, 2.4]
LOSS_RATIO = 0.55


def profile(curve):
    return RiskProfile(SUMS_INSURED, RISKS, curve, premiums=PREMIUM, loss_ratio=LOSS_RATIO)


print(f"expected loss, all sizes {profile(Mbbefd.swiss_re(3.0)).expected_loss():.3f}")
for c in (2.0, 3.0, 4.0):
    p = profile(Mbbefd.swiss_re(c))
    row = [p.expected_layer_loss(limit, att) for limit, att in LAYERS]
    print(f"c = {c:g}: " + "  ".join(f"{x:.3f}" for x in row) + f"   total-loss probability {Mbbefd.swiss_re(c).total_loss_probability():.4f}")
```

```text
expected loss, all sizes 9.900
c = 2: 2.193  1.305  0.913  0.419   total-loss probability 0.1300
c = 3: 1.771  1.060  0.675  0.295   total-loss probability 0.0327
c = 4: 1.324  0.799  0.453  0.183   total-loss probability 0.0065
```

The expected loss of the whole book does not move with `c`: the premium and loss ratio fix it. `c` moves loss
between small partial losses and total losses. A higher `c` means fewer total losses (a probability of 0.1300 at
`c = 2`, 0.0065 at `c = 4`) and so cheaper higher layers.

### Choosing the curve

`Mbbefd.swiss_re(c)` is Bernegger's one-parameter family: `c` of 1.5, 2, 3 and 4 are the Swiss Re curves and 5
is Lloyd's. By the usual market reading, lower values suit small, vulnerable risks where a fire tends to destroy
the whole value, and higher values suit large industrial risks where protection makes partial losses the norm.
That is a convention, not something the library decides. This guide takes `c = 3` by judgement, a middle curve
for a commercial book of mixed occupancy, and then checks it. Two checks on a choice:

1. **Total loss probability.** `total_loss_probability()` is the chance that a loss destroys the whole sum
   insured. If the cedant's claims show a very different rate of total losses, the curve is wrong.
2. **The working layers.** Their burning cost should sit near their exposure-rated value. The scan below picks
   the `c` that minimizes the squared log error on the two lowest layers.

```python
def log_error(c):
    p = profile(Mbbefd.swiss_re(c))
    return sum(math.log(p.expected_layer_loss(l, a) / burning_cost(l, a)) ** 2 for l, a in LAYERS[:2])


grid = [1.5 + 0.25 * i for i in range(15)]
best = min(grid, key=log_error)
print(f"best c on the working layers: {best:g}")
print("squared log error: " + ", ".join(f"c = {c:g} {log_error(c):.3f}" for c in (2.0, 3.0, 4.0, 5.0)))
p_best = profile(Mbbefd.swiss_re(best))
for limit, att in LAYERS:
    print(f"{limit:g} xs {att:g}: burning {burning_cost(limit, att):.3f}, exposure {p_best.expected_layer_loss(limit, att):.3f}")
for limit, att in LAYERS[:2]:
    paid = [min(max(x - att, 0.0), limit) for x in LOSSES if x > att]   # what each piercing loss pays
    m1, m2 = sum(paid) / len(paid), sum(v * v for v in paid) / len(paid)
    gap = math.log(profile(Mbbefd.swiss_re(3.0)).expected_layer_loss(limit, att) / burning_cost(limit, att))
    print(f"{limit:g} xs {att:g}: relative standard error of the burning cost {math.sqrt(m2 / m1**2 / len(paid)):.2f}, "
          f"log gap at c = 3 {gap:.2f}")
```

```text
best c on the working layers: 4
squared log error: c = 2 0.442, c = 3 0.144, c = 4 0.023, c = 5 0.321
1 xs 1: burning 1.518, exposure 1.324
2 xs 2: burning 0.750, exposure 0.799
5 xs 5: burning 0.202, exposure 0.453
10 xs 10: burning 0.000, exposure 0.183
1 xs 1: relative standard error of the burning cost 0.23, log gap at c = 3 0.15
2 xs 2: relative standard error of the burning cost 0.46, log gap at c = 3 0.35
```

The scan picks `c = 4` (squared log error 0.023 against 0.144 for `c = 3`). The last two lines say whether that
difference means anything. The burning cost of 1 xs 1 rests on 27 claims and has a relative standard error of
about 0.23; that of 2 xs 2 rests on 9 and has about 0.46. The judged `c = 3` misses them by 0.15 and 0.35 in
log, each within one standard error, so the experience does not reject it. Use the scan as this kind of
diagnostic, to see whether experience and curve disagree by more than the noise allows, and keep a range of
plausible `c` (here 3 to 4). The rest of this guide uses `c = 3`. Because the history is synthetic, we can also
see that `c = 3` happens to sit near the process that drew it (its layers in the table above, 1.771, 1.060,
0.675 and 0.295, against 1.701, 1.122, 0.648 and 0.427), which a real pricing never knows.

The curve also needs the profile to be right. If a band covers a wide range of sums insured, give its bounds.

```python
c3 = Mbbefd.swiss_re(3.0)
# 400 risks from 1 to 5 with a total sum insured of 800, a mean of 2 each
for label, kw in [("one sum insured of 2", {}),
                  ("uniform between 1 and 5", dict(lower=[1.0], upper=[5.0])),
                  ("tilted to a mean of 2", dict(lower=[1.0], upper=[5.0], spread="tilted"))]:
    band = RiskProfile([2.0], [400], c3, expected_losses=[1.2], **kw)
    print(f"{label:25s} 3 xs 2: {band.expected_layer_loss(3.0, 2.0):.3f}, surplus: {band.expected_surplus_loss(2.0, 4.0):.3f}")
```

```text
one sum insured of 2      3 xs 2: 0.000, surplus: 0.000
uniform between 1 and 5   3 xs 2: 0.200, surplus: 0.450
tilted to a mean of 2     3 xs 2: 0.090, surplus: 0.210
```

With every risk set at the band's mean of 2, nothing exceeds 2 and a layer attaching at 2 is worth nothing. A
uniform spread between the bounds puts the mean at 3, which overstates the band's total sum insured.
`spread="tilted"` keeps the mean at 2 and leans toward the small risks, which is what a total of 800 over 400
risks says: 0.090 for the layer against 0.200. The default spread is uniform, so give `spread="tilted"` when you
know the band's total.

### Surplus treaties

A surplus treaty cedes a share of each risk that depends on the risk's sum insured, so it needs the profile risk
by risk. The cedant keeps the **retention** (one **line**) of every risk, and the treaty takes up to `lines`
further lines, so its **capacity** is `retention * lines` and the most it covers on a risk is
`retention * (1 + lines)`. A risk of sum insured `SI` cedes the share `min(SI - retention, capacity) / SI` of
every loss, and anything above the capacity stays with the cedant (or goes to facultative cover, which this
calculation does not see). `expected_surplus_loss(retention, lines)` is the expected ceded loss. To rate a
per-risk layer that inures to the surplus, pass the retention and lines to `expected_layer_loss`: the layer then
applies to the part of each risk the cedant keeps.

```python
# Surplus treaty and a per-risk layer on what is left of each risk
p3 = profile(Mbbefd.swiss_re(3.0))
retention, lines = 2.0, 4.0
for si in SUMS_INSURED:
    ceded = min(max(si - retention, 0.0), retention * lines)
    print(f"sum insured {si:g}: ceded {ceded:g}, kept {si - ceded:g}")
print(f"surplus, retention 2 and 4 lines: {p3.expected_surplus_loss(2.0, 4.0):.3f}")
print(f"5 xs 5 gross {p3.expected_layer_loss(5.0, 5.0):.3f}, net of the surplus {p3.expected_layer_loss(5.0, 5.0, 2.0, 4.0):.3f}")
```

```text
sum insured 2: ceded 0, kept 2
sum insured 5: ceded 3, kept 2
sum insured 10: ceded 8, kept 2
sum insured 20: ceded 8, kept 12
surplus, retention 2 and 4 lines: 3.696
5 xs 5 gross 0.675, net of the surplus 0.164
```

With a retention of 2 and 4 lines the treaty covers risks up to 10 in full. The risks of 20 cede 8 and keep 12,
so the cedant keeps more of them than the retention. The surplus takes 3.696 of the 9.900 expected loss. Net of
it, every risk but the largest keeps 2, so only the band of 20 still reaches 5 xs 5: 0.164 net against 0.675
gross.

### A published table

`TabulatedCurve(x, g)` takes a table of `(x, G(x))` from `(0, 0)` to `(1, 1)`, concave, and interpolates
linearly. The table must be fine near zero, because the mean destruction rate is the first chord `x1 / G(x1)`.
Its layer shares are piecewise linear approximations (section 4 measures the difference).

**Pitfalls**: a curve is a shape and knows nothing of the cedant's occupancy, construction or protection. The
expected loss is the premium you give times the loss ratio you give, and an inadequate premium gives a low price
without a warning. `expected_claims()` counts losses of every size, mostly tiny, not the ones that pierce a
layer. A band's sum insured is its MPL, so use the MPL where a risk cannot lose its whole value.

## 4. Exposure rating: casualty

### What it is

For liability the cost of a layer follows the **policy limit**, and the shape of loss within a limit comes from
a **ground-up severity** (the claim before any limit or deductible). A primary actuary knows this as increased
limits: ILFs and loss elimination ratios, both ratios of limited expected values.

```python
# Casualty: a ground-up severity, ILFs and limited expected values
ground_up = Lognormal(math.log(0.05), 1.8)
heavy = Pareto(ground_up.mean() / 3.0, 1.5)   # alpha 1.5, scaled to the same mean
print(f"mean {ground_up.mean():.3f} (lognormal), {heavy.mean():.3f} (Pareto)")
for limit in (1.0, 2.0, 5.0, 10.0):
    print(f"limit {limit:g}: ILF {ilf(ground_up, limit, 0.25):.3f} / {ilf(heavy, limit, 0.25):.3f}, "
          f"LER {loss_elimination_ratio(ground_up, limit):.3f} / {loss_elimination_ratio(heavy, limit):.3f}")
```

```text
mean 0.253 (lognormal), 0.253 (Pareto)
limit 1: ILF 1.737 / 1.316, LER 0.636 / 0.807
limit 2: ILF 2.071 / 1.408, LER 0.758 / 0.863
limit 5: ILF 2.403 / 1.490, LER 0.880 / 0.913
limit 10: ILF 2.561 / 1.531, LER 0.938 / 0.939
```

`ilf(severity, limit, basic_limit)` is the LEV at `limit` over the LEV at `basic_limit` (here 0.25), and
`loss_elimination_ratio(severity, deductible)` is the LEV at the deductible over the mean. Each line shows the
lognormal and then a Pareto scaled to the same mean, so the two differ only in shape. They differ at every
limit: the ILF from 0.25 to 10 is 2.561 for the lognormal and 1.531 for the Pareto, and the lognormal keeps
0.636 of its mean below 1 where the Pareto keeps 0.807. The shape of the severity, not the arithmetic, drives
the answer, and the tail beyond the largest limit you sell is the part the data does not contain.

To rate a layer over a profile of policy limits, make one exposure curve per limit with
`severity_exposure_curve(severity, mpl, x)`, which returns `LEV(x * mpl) / LEV(mpl)` at each `x`, put it in a
`TabulatedCurve`, and give the profile a list of curves, one per band.

```python
LIMITS = [1.0, 2.0, 5.0, 10.0]
POLICIES = [800, 300, 100, 40]
CASUALTY_PREMIUM = [8.0, 6.0, 4.0, 3.0]
xs = [0.0] + [0.0005 * 1.25**k for k in range(60) if 0.0005 * 1.25**k < 1.0] + [1.0]
curves = [TabulatedCurve(xs, severity_exposure_curve(ground_up, limit, xs)) for limit in LIMITS]
casualty = RiskProfile(LIMITS, POLICIES, curves, premiums=CASUALTY_PREMIUM, loss_ratio=0.65)
exact = (ground_up.lev(10.0) - ground_up.lev(5.0)) / ground_up.lev(10.0)
print(f"{len(xs)} points; share of a 10m policy in 5 xs 5: table {curves[3].layer_share(5.0, 5.0, 10.0):.4f}, exact {exact:.4f}")
print(f"expected loss {casualty.expected_loss():.3f}")
for limit, att in [(1.0, 1.0), (4.0, 1.0), (5.0, 5.0), (5.0, 2.0)]:
    print(f"{limit:g} xs {att:g}: {casualty.expected_layer_loss(limit, att):.3f}")
```

```text
37 points; share of a 10m policy in 5 xs 5: table 0.0619, exact 0.0617
expected loss 13.650
1 xs 1: 1.248
4 xs 1: 1.862
5 xs 5: 0.121
5 xs 2: 0.677
```

The 37 points start at 0.0005 of the limit and grow by a quarter at each step. The table reproduces the exact
share of a policy with a limit of 10 in 5 xs 5 to three decimals (0.0619 against 0.0617). A policy with a limit
of 1 has no loss above 1, so 1 xs 1 comes from the other three bands. 5 xs 5 is small (0.121) because only the
policies with a limit of 10 reach it.

The severity must be at the cost level of the treaty period, and trend does not reach a layer one for one.
Severity trend lifts every claim, so more of them cross the attachment and those that did already go further
into the layer: an excess layer's loss rises faster than the ground-up trend, and a limit at the bottom rises
slower. This is **leveraged trend**.

```python
# Five per cent a year of severity trend for three years, on the lognormal
f = trend_factor(0.05, 3.0)
trended = Lognormal(math.log(0.05 * f), 1.8)
print(f"ground up: x {trended.mean() / ground_up.mean():.3f}, {(trended.mean() / ground_up.mean()) ** (1 / 3) - 1:.1%} a year")
for limit, att in [(1.0, 0.0), (1.0, 1.0), (5.0, 5.0)]:
    ratio = (trended.lev(att + limit) - trended.lev(att)) / (ground_up.lev(att + limit) - ground_up.lev(att))
    print(f"{limit:g} xs {att:g}: x {ratio:.3f}, {ratio ** (1 / 3) - 1:.1%} a year")
```

```text
ground up: x 1.158, 5.0% a year
1 xs 0: x 1.106, 3.4% a year
1 xs 1: x 1.197, 6.2% a year
5 xs 5: x 1.276, 8.5% a year
```

Five per cent a year on every claim is 3.4% a year on the first 1 of each claim, 6.2% on 1 xs 1 and 8.5% on
5 xs 5. A layer rated on today's severity, or a burning cost on untrended losses, misses that leverage, and it
compounds with each year between the data and the treaty period.

The severity must also match the policies' basis. Claims-made policies and occurrence policies report the same
losses in different years, and a per-claim limit caps each claimant while a per-occurrence limit caps the sum
over all claimants in one event, so the severity behind a per-occurrence limit is the occurrence's total. A
long-tail excess layer also develops for many years after its accident year (#192 in section 8).

**Use it when** a casualty or professional liability program has policy limits and premiums by limit and
little or no large-loss history, or when a layer cuts through a limit (5 xs 5 over policies of 5 and 10 sees only
the 10s). **Do not rely on it when** the cedant's own large claims contradict the assumed severity, or the
policies share limits or aggregate across coverages (each policy's loss is treated alone, so there is no clash).
Legal cost (ALAE), trend and jurisdiction are choices you put into the severity before this call. The ground-up
severity moves the answer more than any other input (the ILF above is 2.561 or 1.531), so run two or three and
report the range.

## 5. Credibility of the experience, layer by layer

### What it is

Experience is worth more in a low layer than in a high one, because more losses pierce it. **Classical (limited
fluctuation) credibility** says how many claims the experience needs to stand alone, the **full credibility
standard**. With fewer claims the credibility is `Z = sqrt(n / standard)`, the square root rule of primary
indications. For layers, `n` is the number of claims **in the layer**, the losses that pierce it, not the number
in the book.

```python
lf = LimitedFluctuation(0.90, 0.10)
print(f"frequency standard {lf.frequency_standard():.0f} claims; at k = 0.05 it is {LimitedFluctuation(0.90, 0.05).frequency_standard():.0f}")
rows = []
for limit, att in LAYERS:
    s = 1.0 - severity.cdf(att)
    m1 = severity.layer(limit, att) / s
    m2 = severity.layer_second_moment(limit, att) / s
    cv = math.sqrt(m2 / m1**2 - 1.0)
    standard = lf.pure_premium_standard(cv)
    z = lf.credibility(pierced(att), standard)
    rows.append(z)
    print(f"{limit:g} xs {att:g}: claims {pierced(att):2d}, cv {cv:.3f}, standard {standard:.0f}, Z {z:.3f}")
```

```text
frequency standard 271 claims; at k = 0.05 it is 1082
1 xs 1: claims 27, cv 0.717, standard 410, Z 0.257
2 xs 2: claims  9, cv 0.717, standard 410, Z 0.148
5 xs 5: claims  2, cv 0.717, standard 410, Z 0.070
10 xs 10: claims  0, cv 0.717, standard 410, Z 0.000
```

`LimitedFluctuation(p, k)` asks that the observed value lie within `k` of its mean with probability `p`; here
10% with probability 90%. The default `k = 0.05` quadruples every standard (1082 claims for the frequency alone
against 271). `p` and `k` are judgements to document. The pure premium standard needs the coefficient of
variation `cv` of the loss to the layer per claim that pierces it, which the block computes from the layer
moments of the fitted Pareto. These four layers, each with limit equal to attachment, share one `cv` by the
scale invariance of the Pareto, so only the number of claims differs.

The lowest layer has 27 claims and `Z` of 0.257, the next 9 and 0.148, then 2 and 0.070, and the top none and 0.
In this example even the working layer is far from full credibility, so exposure carries most of the weight.
Credibility rises with the cedant's size (a large cedant's working layer can have hundreds of claims) and with a
looser standard: `p` and `k` are the levers, and the result depends on them. Classical credibility also treats
the complement as exact and ignores the uncertainty of the exposure estimate itself; Bühlmann-type weights
(below) or a Bayesian model that weighs both sources by their variances are the alternatives.

### Blending

`credibility_weighted(estimate, credibility, complement)` is `Z * estimate + (1 - Z) * complement`: the burning
cost is the estimate and the exposure-rated loss the complement.

```python
for (limit, att), z in zip(LAYERS, rows):
    exp_ = burning_cost(limit, att)
    expo = p3.expected_layer_loss(limit, att)
    print(f"{limit:g} xs {att:g}: burning {exp_:.3f}, exposure {expo:.3f}, Z {z:.3f}, blend {credibility_weighted(exp_, z, expo):.3f}")
```

```text
1 xs 1: burning 1.518, exposure 1.771, Z 0.257, blend 1.706
2 xs 2: burning 0.750, exposure 1.060, Z 0.148, blend 1.014
5 xs 5: burning 0.202, exposure 0.675, Z 0.070, blend 0.642
10 xs 10: burning 0.000, exposure 0.295, Z 0.000, blend 0.295
```

In 1 xs 1 the blend, 1.706, lies between the burning cost (1.518) and the exposure (1.771), closer to the
exposure. In 5 xs 5, with two hits, it is 0.642 against an exposure of 0.675, and in 10 xs 10 it is the exposure
alone. The higher the layer, the more the price is the profile and the curve.

Three cautions. The standard is a choice, and a looser `k` gives experience more weight, so show the
sensitivity. `Z` supports judgement and does not replace it: move the weights for a one-off event or a changed
book. The complement is treated as correct, so if the curve is the weak part, blend against a range of `c`.
Frequency and severity can also be weighted apart, with `frequency_standard()` and `severity_standard(cv)`.

### A complement from a collective of cedants

When you hold loss ratios for many similar cedants, or many years of one, **Bühlmann–Straub** credibility takes
the weights from the data instead of a standard. `buhlmann_straub(ratios, weights)` takes one row per group (a
cedant) and one column per period, with a weight (premium) per cell. The fit has the collective mean, the
variance components and each group's credibility and premium.

```python
ratios = [[0.50, 0.60, 0.55, 0.70, 0.65, 0.50], [0.90, 0.80, 1.10, 0.70, 0.95, 0.85],
          [0.60, 0.70, 0.50, 0.65, 0.60, 0.70], [1.20, 0.40, 0.90, 1.50, 0.60, 0.80]]
premiums = [[10, 11, 12, 12, 13, 14], [20, 22, 21, 23, 24, 25], [5, 5, 6, 6, 7, 7], [3, 3, 4, 4, 4, 5]]
fit = buhlmann_straub(ratios, premiums)
print(f"collective {fit.collective_mean:.3f}, k {fit.k:.2f}")
for i in range(4):
    print(f"cedant {i + 1}: mean {fit.group_means[i]:.3f}, weight {fit.group_weights[i]:.0f}, Z {fit.credibility[i]:.3f}, premium {fit.premiums[i]:.3f}")
own = RiskProfile(SUMS_INSURED, RISKS, c3, premiums=PREMIUM, loss_ratio=fit.premiums[0])
print(f"cedant 1's profile at its credibility loss ratio: 5 xs 5 {own.expected_layer_loss(5.0, 5.0):.3f}")
```

```text
collective 0.745, k 10.50
cedant 1: mean 0.584, weight 72, Z 0.873, premium 0.605
cedant 2: mean 0.880, weight 135, Z 0.928, premium 0.871
cedant 3: mean 0.625, weight 36, Z 0.774, premium 0.652
cedant 4: mean 0.904, weight 23, Z 0.687, premium 0.855
cedant 1's profile at its credibility loss ratio: 5 xs 5 0.742
```

A cedant's credibility is its total weight over the weight plus `k`, where `k = 10.50` is the within-group
variance over the variance between groups. The two largest cedants (weights 135 and 72) have `Z` of 0.928 and
0.873. The smallest (weight 23), with the most volatile ratios, has 0.687, and its premium, 0.855, is pulled
from its own mean of 0.904 toward the collective's 0.745. Use it for a quantity per period with a natural
weight, such as a loss ratio or a rate on line by treaty year. Its output here is the loss ratio that exposure
rating needs: if the profile of section 3 were cedant 1's, its credibility loss ratio of 0.605 in place of the
assumed 55% prices 5 xs 5 at 0.742 against 0.675. It is not a layer tool: it needs a panel, and one
cedant's large losses are too sparse to feed it, which makes section 5's limited fluctuation weights a common
choice for the weight of a layer's own experience. Hierarchical and Bayesian credibility are planned, not available.

## 6. Reaching layers above the data

### What it is

The highest layers of a program may never have been touched. The lower layers say something about the slope of
the tail: given the expected losses of two layers, one Pareto alpha reproduces both, and the same alpha prices
the next layer.

```python
alpha = alpha_between_layers((1.0, 1.0, burning_cost(1.0, 1.0)), (2.0, 2.0, burning_cost(2.0, 2.0)))
step = pareto_extrapolation((2.0, 2.0), (5.0, 5.0), alpha)
print(f"alpha between 1 xs 1 and 2 xs 2: {alpha:.3f}")
print(f"5 xs 5 is {step:.3f} of 2 xs 2, so {step * burning_cost(2.0, 2.0):.3f}")
print(f"10 xs 10 is {pareto_extrapolation((2.0, 2.0), (10.0, 10.0), alpha) * burning_cost(2.0, 2.0):.3f}")
alpha_expo = alpha_between_layers((1.0, 1.0, p3.expected_layer_loss(1.0, 1.0)), (2.0, 2.0, p3.expected_layer_loss(2.0, 2.0)))
print(f"alpha implied by the exposure curve between the same layers: {alpha_expo:.3f}")
```

```text
alpha between 1 xs 1 and 2 xs 2: 2.017
5 xs 5 is 0.394 of 2 xs 2, so 0.295
10 xs 10 is 0.146
alpha implied by the exposure curve between the same layers: 1.740
```

`alpha_between_layers((limit, attachment, expected_loss), (limit, attachment, expected_loss))` returns the alpha
at which a Pareto reproduces both losses; one layer must lie above the other. `pareto_extrapolation((limit,
attachment), (limit, attachment), alpha)` returns the expected loss of the second layer per unit of the first's,
so multiplying by the first layer's loss prices the second.

The extrapolated 5 xs 5 is 0.295, below the blend (0.642) and the exposure value (0.675). The two layers came
from the light ten years, and their alpha of 2.017 is lighter than the 1.6 that drew them. The extrapolation
takes it as the truth for everything above. It does what it says and cannot know the sample was kind. Use it:

between layers that both have credible experience, to fill a gap; upward from the highest credible layer with an
alpha you choose in place of one from two noisy points; and as a check on the exposure curve, whose implied alpha
should not differ implausibly from the experience's. The last line is that check: the `c = 3` curve implies
1.740 between the same two layers, against 2.017 from the experience. prospicio ships no market benchmark
alphas, so the outside alpha is either your own benchmark for the line or one implied by the exposure curve, as
here.

### A whole tower, references and PML curves

`match_tower(attachments, layer_losses, frequencies)` fits one model to a tower of contiguous layers, the last
unlimited (Riegel 2018): a frequency and a piecewise Pareto severity that reproduce every layer loss.
`fit_references` takes expected layer losses from any source (they may overlap or leave gaps) and excess
frequencies, and fits a model through all of them, which suits one reference from experience, one from exposure
and a known frequency. `fit_pml_curve(return_periods, amounts, tail_alpha)` fits a severity through a PML curve:
`amounts[j]` is exceeded once in `return_periods[j]` years.

```python
ATT = [1.0, 2.0, 5.0, 10.0]
tower_losses = [p3.expected_layer_loss(ATT[i + 1] - ATT[i], ATT[i]) for i in range(3)] + [p3.expected_layer_loss(math.inf, 10.0)]
tower = match_tower(ATT, tower_losses, [len(LOSSES) / YEARS, None, None, None])
print([round(x, 3) for x in tower_losses])
print(f"frequency above 1: {tower.frequency:.2f}; above 5: {tower.excess_frequency(5.0):.3f} (experience {pierced(5.0) / YEARS:.2f})")
print(f"15 xs 15: {tower.layer_loss(15.0, 15.0):.3f}")
refs = fit_references([(1.0, 1.0, credibility_weighted(burning_cost(1.0, 1.0), rows[0], p3.expected_layer_loss(1.0, 1.0))),
                       (5.0, 5.0, p3.expected_layer_loss(5.0, 5.0)),
                       (math.inf, 10.0, p3.expected_layer_loss(math.inf, 10.0))],
                      [(1.0, len(LOSSES) / YEARS)], default_alpha=2.0)
print("references:", ", ".join(f"{refs.layer_loss(l, a):.3f}" for l, a in [(1.0, 1.0), (5.0, 5.0), (math.inf, 10.0)]))
print("between and above:", ", ".join(f"{refs.layer_loss(l, a):.3f}" for l, a in [(2.0, 2.0), (10.0, 10.0), (15.0, 15.0)]))
print(f"smallest alpha in the fit {min(refs.severity.alpha):.2f}")
pml = fit_pml_curve([2, 5, 10, 25], [2.5, 4.0, 6.0, 10.0], tail_alpha=1.8)
print(f"PML curve: excess frequency of 4 is {pml.excess_frequency(4.0):.2f}; 5 xs 5 {pml.layer_loss(5.0, 5.0):.3f}")
```

```text
[1.771, 1.456, 0.675, 0.295]
frequency above 1: 2.80; above 5: 0.238 (experience 0.20)
15 xs 15: 0.102
references: 1.706, 0.675, 0.295
between and above: 2.117, 0.234, 0.093
smallest alpha in the fit 0.55
PML curve: excess frequency of 4 is 0.20; 5 xs 5 0.369
```

The tower is pinned to the experienced 2.8 losses a year above 1. The model it returns has 0.238 losses a year
above 5, against 0.20 seen (two in ten years), so experience and curve agree within the noise of two events. A
`TowerModel` has `frequency` and `severity` as properties, and `layer_loss(limit, attachment)` and
`excess_frequency(x)` as methods. The unlimited top layer fixes the tail, and the tower then prices a layer
above it (15 xs 15: 0.102).

`fit_references` was given the blended 1 xs 1, the exposure 5 xs 5 and the exposure unlimited layer above 10,
plus the frequency, and reproduces all three. Three cautions for this family of fits:

- **They reproduce the inputs exactly, noise included.** Alphas can jump between pieces (the smallest in this
  fit is 0.55). Read the layer losses and frequencies, not the alphas piece by piece.
- **They leave a gap between references to the fit.** 2 xs 2 comes out at 2.117 here, against 1.060 by exposure
  and 1.014 in the blend. Add a reference for any layer you care about and compare with the other evidence.
- **The tail above the highest reference is an assumption**: the alpha of the unlimited layer, or
  `default_alpha` (here 2.0).

## 7. When data are thin

With a handful of large losses, in this order:

1. **Start from exposure** if there is any profile, and state the range of `c`.
2. **Use the experience as a check and a small adjustment**, with the credibility of section 5. A low `Z` is a
   fact about the data.
3. **Take the alpha from outside the sample**: your own benchmark for the line (prospicio ships none), the
   lower layers, or the alpha implied by the exposure curve or tower (section 6).
4. **Use everything the losses say.** A loss capped by a policy limit says only that it was at least that large
   (`Pareto.fit(..., censored=...)`); losses reported above different thresholds go in `reporting_thresholds`.
5. **For catastrophe or scenario business, start from the PML curve** with `fit_pml_curve` and a tail alpha you
   choose.

```python
capped = [min(x, 5.0) for x in LOSSES]
flags = [x >= 5.0 for x in LOSSES]
print(f"alpha ignoring a 5m policy limit {Pareto.fit(capped, 1.0).alpha:.3f}, with censoring {Pareto.fit(capped, 1.0, censored=flags).alpha:.3f}")
few = LOSSES[-8:]
alpha_few = Pareto.fit(few, 1.0).alpha
print(f"alpha from the last {len(few)} losses {alpha_few:.3f} against {alpha_hat:.3f} from all")
print(f"standard error, about alpha / sqrt(n): {alpha_few / math.sqrt(len(few)):.3f} with {len(few)} losses, "
      f"{alpha_hat / math.sqrt(len(LOSSES)):.3f} with {len(LOSSES)}")
```

```text
alpha ignoring a 5m policy limit 1.927, with censoring 1.789
alpha from the last 8 losses 1.357 against 1.880 from all
standard error, about alpha / sqrt(n): 0.480 with 8 losses, 0.355 with 28
```

The first line pretends a 5 policy limit capped the two largest losses. Treating the capped amounts as exact
gives 1.927 and marking them censored 1.789, so ignoring a limit makes the tail look lighter. The second line
fits the last eight losses, a subset that happens to hold the two largest claims, and gets 1.357 where all 28
give 1.880. It is not a random sample, but the third line shows why any eight losses leave the alpha loose: the
maximum likelihood alpha has a standard error of about `alpha / sqrt(n)`, 0.480 with eight losses and still
0.355 with all 28. Either range changes the price of a high layer a great deal.

## 8. What prospicio does not do yet

This guide starts from as-if losses because preparing them is not in the library. The gaps are filed for the
lanes that own the code:

- **#190, prepare individual large losses**: trend each loss from its accident date, develop known large losses
  to ultimate (IBNER) and large-loss counts for late reports, apply an indexation or stability clause, convert
  currencies, and give the as-if layer loss by year. Today you write the loop, as `burning_cost` above does,
  around `trend_factor` and `Layer.ceded`.
- **#191, as-if history under past program structures**: restate each past year under its own actual structure
  (retentions, limits, inuring quota shares) next to the proposed one, so the history can be reconciled with the
  cedant's reported recoveries.
- **#192, excess-layer loss development**: excess development factors from a ground-up triangle and a severity
  curve, credibility blending of a layer's own development with benchmark patterns, and a Bornhuetter–Ferguson
  prior from the Pareto layer share. Until then, development of layer losses is unstable with many zero cells.

Smaller awkward spots met while writing this guide: no helper turns a severity into a `TabulatedCurve` (section
4 builds the grid by hand); nothing calibrates `c` to experience (section 3 scans by hand); the `cv` of the loss
to a layer per claim is computed by hand from the layer moments (section 5); the burning cost as a rate on
on-level subject premium is a loop you write (section 2); no benchmark alphas or curves by line ship with the
library (section 7); and `fit_references` and `match_tower` accept references that give jagged alphas without a
warning.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): the map of the guides, and which to read first.
- [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md): after the expected loss, the risk load.
- [`aggregate-distributions.md`](aggregate-distributions.md): the annual distribution of what the layers pay,
  from the frequency and severity used here.
- [`severity-and-frequency.md`](severity-and-frequency.md): fitting and choosing the severity and the count.
- [`reserving-methods.md`](reserving-methods.md): development of the losses before they are used as experience.
- [`reinsurance-price.md`](reinsurance-price.md): what the price of a contract is, from the expected loss to the
  technical premium.
- `docs/design/pareto.md`, `docs/design/credibility.md` and `docs/design/aggregate.md`: the design behind each
  method.
