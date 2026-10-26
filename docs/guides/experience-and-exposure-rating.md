# Experience rating, exposure rating and blending them

An excess of loss layer has two independent sources of evidence for its expected loss:

- **Experience rating**: what the large losses of the cedant cost the layer in the past.
- **Exposure rating**: what the book of risks of the cedant costs the layer under a standard loss shape.

Most layers need both, with a weight for each that shows how much you can trust it. This guide tells which
method to use, how prospicio does each method, how to weight them, and how to price layers above the data.

**Use this guide when** you must find the expected loss of a layer. These inputs decide the method:

- the number of years of large losses.
- a profile of sums insured or policy limits.
- how often losses hit the layer.
- the largest loss in the data.

The guide is for an actuary who knows rating plans, increased limits factors
and credibility, and who is new to reinsurance pricing. It stops at the expected loss. These guides give the
next steps:

- the risk load: [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md).
- the annual total: [`aggregate-distributions.md`](aggregate-distributions.md).
- a full severity fit: [`severity-and-frequency.md`](severity-and-frequency.md).
- primary ratemaking (on-level premium, the overall rate indication): section 1 of
  [`choosing-a-method.md`](choosing-a-method.md).

The layers in this guide are plain per-risk or per-loss layers. They have no reinstatements, no annual aggregate
limit or deductible, and no clash cover (one event that hits several risks). Those terms change the expected
loss. [`reinsurance-price.md`](reinsurance-price.md) and [`aggregate-distributions.md`](aggregate-distributions.md)
show them. `PreparedLosses.burning_cost` accepts towers with those terms for a burning cost, but not together
with an index clause.

Money is in millions of a synthetic currency. The code in the guide prints each number in the text, and CI runs
and checks the code.

## The decision in one table

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| A working layer (many losses pierce it), several years of as-if large losses | Burning cost (a rate on on-level subject premium when the size of the book changed), or a Pareto fitted to the losses | `LargeLossExperience`, `PreparedLosses.burning_cost`, `RateHistory`, `Pareto.fit`, `CollectiveModel.layer_mean` | The own losses of the cedant are the best evidence when there are sufficient losses |
| A higher layer, pierced by few or no losses | Exposure rating | `RiskProfile.expected_layer_loss` | The experience has no information, but the profile and a curve have |
| Property per risk, profile by sum insured | MBBEFD (Swiss Re) curves | `Mbbefd.swiss_re(c)`, `RiskProfile` | One parameter `c` sets how losses spread over the sum insured |
| A published or in-house curve held as a table | Tabulated curve | `TabulatedCurve` | Same use as MBBEFD, from the table that you have |
| Surplus treaty, or a per-risk layer on the part that a surplus leaves | Exposure, treaty applied risk by risk | `expected_surplus_loss`, `expected_layer_loss(..., retention, lines)` | The cession depends on the sum insured of each risk |
| A band that covers a wide range of sums insured | Spread the band between bounds | `RiskProfile(..., lower, upper, spread)` | One sum insured for each band misprices layers inside the range |
| Casualty, profile by policy limit | ILFs and a severity curve for each limit, the severity trended to the treaty period | `ilf`, `loss_elimination_ratio`, `severity_exposure_curve`, `TabulatedCurve` | The shape comes from a ground-up severity and changes with the limit |
| Experience and exposure both exist | Weight by the credibility of the experience in that layer | `LimitedFluctuation`, `credibility_weighted` | Credibility follows the number of claims in the layer, so it decreases as the layer goes up |
| Loss ratios of many cedants or years, with premiums | Bühlmann–Straub | `buhlmann_straub` | A complement from similar cedants, with the variance between them |
| Layers above the data, two credible layers below | Pareto extrapolation | `alpha_between_layers`, `pareto_extrapolation` | The slope between known layers continues upward, with judgement |
| A tower of layer losses or prices | One model for the full tower | `match_tower` | Layer losses, frequency and tail become one consistent severity |
| References from several sources (experience, exposure, a frequency) | One model through all of them | `fit_references` | It reproduces each reference exactly, so check them first |
| A PML or return-period curve | A severity through the points | `fit_pml_curve` | The curve is the only information on size |
| A small number of losses | Exposure first, experience as a check, an alpha from outside | `Pareto.fit` (`censored`), `fit_pml_curve` | An alpha fitted to few losses changes much |

## 1. Words you need first

- **Large loss**: a claim above the **reporting threshold** (here 1). The cedant lists each large loss.
- **As-if**: a historical loss restated to the prospective year. You trend it to the future price level and
  develop it to ultimate. Then you measure it against the contract that you price (its retention and limit, with
  any indexation). This guide starts from as-if losses. `LargeLossExperience` and `PreparedLosses` make them
  (section 2, Preparing raw losses). `prepare` applies currency, development (IBNER) and trend.
  `burning_cost` applies the index clause.
- **Burning cost**: the average annual loss that a layer pays on the as-if history. It is the pure premium of
  the layer from experience.
- **Pierce**: a loss pierces a layer when it is larger than the attachment. Only the losses that pierce a layer
  give information about it.
- **Exposure curve** `G(x)`: the share of the expected loss of a risk below the fraction `x` of its maximum
  possible loss (**MPL**, the sum insured or policy limit). A loss divided by the MPL is its
  **destruction rate**. The share of the expected loss of a risk in a layer is
  `G(top / MPL) - G(attachment / MPL)`.
- **ILF and LER**: the ILF (increased limits factor) is the limited expected value (LEV) of a severity at a
  limit, divided by the LEV at a basic limit. The loss elimination ratio is the LEV at a deductible divided by the mean.
- **Local Pareto alpha**: how fast the number of losses above a threshold decreases as the threshold increases.
  A higher alpha is a lighter tail.
- **Working layer**: a layer that losses hit often. A **higher layer** is a layer that losses hit seldom.

## 2. Experience rating

### What it is

Take each large loss of the as-if history, apply the layer, and calculate the average over the years. The
result is the burning cost. The second form replaces the raw history with a fitted model: a Poisson count of
large losses and a Pareto severity. This model gives the expected loss of any layer in closed form, also for
layers that the history did not reach.

The example uses ten years of as-if losses above 1 from a synthetic property cedant.

```python
import math

from prospicio.credibility import LimitedFluctuation, buhlmann_straub
from prospicio.distributions import Lognormal, Pareto, Poisson
from prospicio.pricing import (
    CollectiveModel, DevelopmentPattern, IndexClause, LargeLossExperience, Mbbefd, RiskProfile,
    TabulatedCurve, alpha_between_layers, credibility_weighted, fit_pml_curve, fit_references, ilf,
    layer_cv, loss_elimination_ratio, match_tower, pareto_extrapolation, severity_exposure_curve, trend_factor,
)
from prospicio.reinsurance import Layer, Tower

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

There are ten years and 28 losses, and the largest loss is 6.59. A layer is written limit xs attachment. The
layer `1 xs 1` pays the part of a loss between 1 and 2, and `5 xs 5` pays the part between 5 and 10.

```python
LAYERS = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (10.0, 10.0)]   # (limit, attachment)
ids = [f"{year}-{i}" for year, losses in HISTORY.items() for i in range(len(losses))]
accident = [year + 0.5 for year, losses in HISTORY.items() for _ in losses]
experience = LargeLossExperience(ids, accident, LOSSES, threshold=1.0)
prepared = experience.prepare(rate=0.0, to=2026.5)   # already as-if, so no trend


def burning_cost(limit, attachment):
    tower = Tower([Layer("xl", limit, attachment)])
    cost = prepared.burning_cost(tower, list(HISTORY), premium=[1.0] * YEARS, development=[1.0] * YEARS)
    return cost.mean_loss_cost[0]


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

`LargeLossExperience(ids, accident, amount, threshold=...)` is the list of large losses of the cedant. Each loss
has an identifier, an accident date as a decimal year (mid-year here) and an amount. `prepare` restates the
losses to the prospective period. The rate is 0 because this history is already as-if (the next subsection
prepares raw losses).

`burning_cost(tower, years, premium=..., development=...)` applies the tower to the losses of each year. It
develops the layer total of each year by its factor and divides it by its premium. With a premium of 1.0 and a
factor of 1.0 for each year, `mean_loss_cost` is the average annual layer loss, the burning cost. It has one
entry for each layer of the tower. The burning cost is exact for the history, and that is its limit. No loss
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
The last column is the synthetic model that drew these ten years (three losses a year, alpha 1.6). In practice,
you do not know this model. These ten years were lighter than their source. Thus both estimates are below it in
all layers. The ratio of the source to the estimate increases with height (1.453 against 1.701 at 1 xs 1, 0.191 against 0.427
at 10 xs 10). A short history can easily miss the tail.

The fit counts the loss of exactly 1.00 in 2019. Thus it sees 28 losses, but 27 losses pierce 1 xs 1. A Poisson
count and one Pareto alpha above the threshold are assumptions. Before you price with them, do these checks:

1. Compare the variance of the yearly counts with their mean. If the variance is larger, use a negative binomial.
2. Examine the choice of threshold.
3. Make a mean excess plot.

When one alpha does not fit, `PiecewisePareto.fit(losses, thresholds)` fits one alpha for each band of size.
The `truncation` argument of `Pareto.fit` and `alpha_between_layers` caps the tail at a largest possible loss.
[`severity-and-frequency.md`](severity-and-frequency.md) shows these checks.

**Use it when** these conditions are true:

- The layer has sufficient piercing losses for a stable burning cost (section 5 gives a number for this).
- The portfolio is stable, or you restated it to the portfolio of today.
- The as-if work is complete: trend, development of open claims, and indexation of retention and limit.
  `PreparedLosses` does those steps (the next subsection), from the rate, pattern and clause that you give it.

**Do not use it alone when** one of these conditions is true:

- The layer had few or no hits. Use the fitted model for a frequency and an alpha from the lower layers, but
  never as the only price.
- The book changed. The past losses then describe a different portfolio.
- The losses are not as-if. A burning cost on untrended, undeveloped losses is too low. prospicio does not warn
  you: it accepts `prepare(rate=0.0, to=...)` with no development, as above. Only you know whether the losses
  were already as-if.

**Data needed**: individual losses above a reporting threshold. Each loss needs its accident date, its amount at
a stated valuation date and its currency. (Here the losses are already on the prospective basis.) You also need
the subject premium of the cedant (or a different exposure base) for each year. Raw losses also need these
inputs (the next subsection):

- a trend rate.
- a development pattern for known claims.
- count IBNR factors.
- for an index clause, the index ratios.

The threshold `t` is a modelling choice, and the Pareto must fit above it. Sources can report from different
thresholds, or policy limits can cap some losses. If so, tell the fit (section 7). Three steps of the as-if work
are easy to miss:

- **Exposure volume.** The average for each year above assumes a book of the same size each year, as this
  synthetic book is. When the size changed, give the burning cost as a rate on subject premium. Bring the
  premium of each year to the rate level of today (`RateHistory`, section 1 of
  [`choosing-a-method.md`](choosing-a-method.md)), and pass it as `premium=` to `burning_cost`. Its `loss_cost`
  is the sum of the developed layer losses divided by the sum of the premiums. This weights each year by its
  premium. `mean_loss_cost` is the average of the yearly rates. Multiply the rate by the prospective subject
  premium.
- **The threshold after trend.** Trend moves losses across the threshold. A loss just below 1 at its own date
  can be above 1 at the cost of today, and it is not in the list. Pass the reporting threshold as `threshold=`
  to `LargeLossExperience`. Then `prepared.complete_above` is the threshold times the largest trend factor. Above
  this size, each year is complete. The size is in the currency of the reported losses, before development.
  The test compares it with the reported amount of each loss times its own trend factor. It does not use the
  as-if amount, which also includes the exchange rate and IBNER. The burning cost leaves out the losses below
  `complete_above`, so each year counts the same losses. A layer should attach well above `complete_above`
  converted to the pricing currency (times `fx`, with an allowance for development). If it does not, ask the
  cedant for losses from a lower threshold.
- **Legal cost (ALAE).** State whether it is in the loss (and thus erodes the layer), shared pro rata with the
  indemnity, or excluded. Use the same basis for the loss ratio of the exposure rating.

### Preparing raw losses

The listing of a cedant is seldom as-if. Here is a small listing: six losses reported above 1, in a currency
worth 1.1 of the pricing currency. The valuation date is the start of 2026, and the average accident date of the
treaty is mid-2026. The known claims still develop. Thus the first step estimates that development (IBNER) from
older claims valued at several ages.

```python
# Older large claims valued at ages 0.5 to 3.5 years, to estimate the development of known claims (IBNER)
dev_ids = ["p1"] * 4 + ["p2"] * 4 + ["p3"] * 3 + ["p4"] * 3 + ["p5"] * 2
dev_age = [0.5, 1.5, 2.5, 3.5] * 2 + [0.5, 1.5, 2.5] * 2 + [0.5, 1.5]
dev_amount = [1.0, 1.3, 1.4, 1.4, 2.0, 2.4, 2.6, 2.7, 1.5, 1.8, 1.9, 0.8, 1.0, 1.1, 1.2, 1.5]
ibner = DevelopmentPattern.from_history(dev_ids, dev_age, dev_amount, ages=[0.5, 1.5, 2.5, 3.5])
print("IBNER to ultimate at ages", ibner.ages, ":", ", ".join(f"{f:.3f}" for f in ibner.factors))

raw = LargeLossExperience(
    ["r1", "r2", "r3", "r4", "r5", "r6"],
    [2022.3, 2022.8, 2023.4, 2024.1, 2024.6, 2025.2],
    [2.10, 1.05, 3.40, 1.02, 2.60, 1.30],
    threshold=1.0, valuation=2026.0, fx=1.1,
)
prepared_raw = raw.prepare(rate=0.05, to=2026.5, development=ibner)
print(f"complete above {prepared_raw.complete_above:.3f}")
table = prepared_raw.to_dict()
for row in zip(table["id"], table["year"], table["amount"], table["development"], table["factor"], table["trended"]):
    print("{} {}: reported {:.2f}, development {:.3f}, trend {:.3f}, as-if {:.3f}".format(*row))
```

```text
IBNER to ultimate at ages [0.5, 1.5, 2.5, 3.5] : 1.359, 1.104, 1.025, 1.000
complete above 1.227
r1 2022: reported 2.10, development 1.000, trend 1.227, as-if 2.835
r2 2022: reported 1.05, development 1.007, trend 1.198, as-if 1.394
r3 2023: reported 3.40, development 1.023, trend 1.163, as-if 4.449
r4 2024: reported 1.02, development 1.072, trend 1.124, as-if 1.353
r5 2024: reported 2.60, development 1.129, trend 1.097, as-if 3.544
r6 2025: reported 1.30, development 1.282, trend 1.065, as-if 1.954
```

`DevelopmentPattern.from_history(ids, age, amount, ages=...)` is a volume-weighted chain ladder on the claims.
The factor from one age to the next is the sum of the later amounts divided by the sum of the earlier amounts.
It uses the claims valued at both ages with a positive earlier amount. The result is a factor to ultimate at
each age (1.359 at half a year, 1.000 at three and a half years). A claim first seen later is IBNR, not IBNER,
and the pattern does not include it.

The pattern does not correct for how the cedant selected the listing. A list cut at the reporting threshold on
the latest incurred does not have the small claims that grew later. This biases the factors, so you must judge
the pattern. `DevelopmentPattern(ages, factors)` takes a pattern that you select.

`prepare(rate=0.05, to=2026.5, development=ibner)` does three steps for each loss:

1. It converts the loss at `fx`.
2. It develops the loss by the pattern at its age at the valuation date (interpolated between ages, constant
   past the ends).
3. It trends the loss by 5% a year from its accident date to `to`.

The as-if amount is the product: for r1, 2.10 times 1.1 times 1.000 times 1.227 is 2.835. In place of a rate,
you can give a trend factor for each loss (`factors=`). In place of a pattern, you can give a development factor
for each loss. `complete_above` is the threshold times the largest trend factor, 1.227, in the reported currency.

The test uses the reported amount, not the as-if amount. The as-if amount of r4, 1.353, is above 1.227. But its
reported 1.02 times its trend factor of 1.124 is below 1.227. Thus the burning cost leaves out r4. A loss of the
same trended size from the oldest year was below the threshold at its own date, and is not in the list.

Then apply the treaty by year. Each loss is already developed. Thus the development by year is only the count
IBNR factor (claims not yet reported). It is not an excess factor that already includes IBNER. In place of a
list, you can give a count `DevelopmentPattern`, for example from a `ChainLadder` on a triangle of claim counts.
The function reads it at the average age of each year.

```python
XL = Tower([Layer("2 xs 2", 2.0, 2.0)])
RAW_YEARS = [2022, 2023, 2024, 2025]
ON_LEVEL = [9.0, 9.5, 10.0, 10.5]          # subject premium at today's rate level
COUNT_IBNR = [1.0, 1.0, 1.02, 1.10]
plain = prepared_raw.burning_cost(XL, RAW_YEARS, premium=ON_LEVEL, development=COUNT_IBNR)
indexed = prepared_raw.burning_cost(XL, RAW_YEARS, premium=ON_LEVEL, development=COUNT_IBNR,
                                    index=IndexClause(0.10, kind="severe"),
                                    ratios=[1.08, 1.04, 1.15, 1.04, 1.12, 1.06])
p, q = plain.to_dict(), indexed.to_dict()
for year, n, a, b in zip(p["year"], p["losses"], p["developed"], q["developed"]):
    print(f"{year}: losses {n}, developed layer loss {a[0]:.3f}, with the index clause {b[0]:.3f}")
print(f"rate on premium {plain.loss_cost[0]:.4f}, with the index clause {indexed.loss_cost[0]:.4f}")
print(f"on a prospective premium of 11: {11 * plain.loss_cost[0]:.3f}, {11 * indexed.loss_cost[0]:.3f}")
history = prepared_raw.as_if([Tower([Layer("1.5 xs 1.5", 1.5, 1.5)])] * 2 + [XL] * 2, RAW_YEARS, XL)
h = history.to_dict()
for year, layers, n, was, now in zip(h["year"], h["historical_layers"], h["reported_losses"], h["historical"], h["proposed"]):
    print(f"{year}: {n} reported, under {layers[0]} {was[0]:.3f}; as-if under 2 xs 2 {now[0]:.3f}")
```

```text
2022: losses 2, developed layer loss 0.835, with the index clause 0.835
2023: losses 1, developed layer loss 2.000, with the index clause 2.091
2024: losses 1, developed layer loss 1.574, with the index clause 1.537
2025: losses 1, developed layer loss 0.000, with the index clause 0.000
rate on premium 0.1131, with the index clause 0.1145
on a prospective premium of 11: 1.244, 1.259
2022: 2 reported, under 1.5 xs 1.5 0.600; as-if under 2 xs 2 0.835
2023: 1 reported, under 1.5 xs 1.5 1.500; as-if under 2 xs 2 2.000
2024: 2 reported, under 2 xs 2 0.600; as-if under 2 xs 2 1.544
2025: 1 reported, under 2 xs 2 0.000; as-if under 2 xs 2 0.000
```

The 2024 layer loss, 1.574, is the 1.544 of r5 in the layer times the count factor 1.02. The rate on premium is
the sum of the developed layer losses divided by the sum of the premiums, 0.1131. Thus a prospective subject
premium of 11 gives 1.244.

`IndexClause(franchise, kind)` moves the retention and limit of each loss by its index ratio. The index ratio is
the index at the expected payment date of the loss, divided by the index at the base date of the treaty. The
base date is frequently the inception. Larger claims settle later, so their ratios are higher. The two kinds work as follows:

- A severe inflation clause (`kind="severe"`) moves the terms only by the increase above the franchise, and
  never down.
- `kind="full"` moves them by the full ratio, up or down, when the ratio passes the franchise.

The clause decreases the 2024 layer loss (1.537 against 1.574) but increases the 2023 layer loss (2.091 against
2.000). The loss r3 exhausts the layer, and the clause increases the limit with the retention. The clause takes
only per-occurrence terms. With an index clause, a tower with an annual aggregate limit or deductible, or with
limited reinstatements, gives an error.

`as_if(historical, years, proposed)` shows two views of each year side by side:

- The historical side runs each loss as reported through the program of that year. In 2024 there are two losses,
  r4 included, in their own currency, untrended and undeveloped. The program here is 1.5 xs 1.5 until 2023. The
  values 0.600 and 1.500 are the amounts that the cedant should have recovered. Compare them with the reported
  recoveries to check the listing before you trend anything.
- The proposed side is the as-if layer loss before development by year (1.544 in 2024 against 1.574
  developed), without the index clause.

## 3. Exposure rating: property

### What it is

Exposure rating needs no loss history. The cedant gives a **risk profile**: bands of sum insured, with the
number of risks and the premium in each band. Each band gets an expected loss (premium times a loss ratio) and
an exposure curve. The curve gives the share of the expected loss of each risk that is in the layer. It has the
same function as the limits factors of a rating plan for a primary policy.

The premium is the gross premium of the cedant for the perils that the treaty covers. It is at the prospective
rate level, before commission. The loss ratio is the expected ground-up loss ratio to that premium for the
treaty year. It excludes catastrophe losses and uses the same ALAE basis as the layer. A loss ratio net of
commission, or one that includes catastrophe years, gives an incorrect value for each layer in proportion.

Per-risk exposure rating prices a layer that each risk triggers alone. It does not price a catastrophe
(per-event) layer, where one event hits many risks at the same time. For that layer, use a catastrophe model or
the PML method of section 6.

In prospicio the profile is `RiskProfile` and the curve is `Mbbefd` or `TabulatedCurve`. This example has four
bands with a 55% loss ratio, and the four layers of section 2.

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

The expected loss of the full book does not change with `c`, because the premium and loss ratio set it. The
parameter `c` moves loss between small partial losses and total losses. A higher `c` gives fewer total losses
(a probability of 0.1300 at `c = 2`, 0.0065 at `c = 4`), and thus cheaper higher layers.

### Choosing the curve

`Mbbefd.swiss_re(c)` is the one-parameter family of Bernegger. The values 1.5, 2, 3 and 4 of `c` are the Swiss
Re curves, and 5 is the Lloyd's curve. By the usual market reading, lower values are for small, vulnerable
risks, where a fire usually destroys the full value. Higher values are for large industrial risks, where
protection makes partial losses usual. This guide selects `c = 3` by judgement, a middle curve for a commercial
book of mixed occupancy, and then checks it. There are two checks on a choice:

1. **Total loss probability.** `total_loss_probability()` is the probability that a loss destroys the full sum
   insured. If the claims of the cedant show a very different rate of total losses, the curve is wrong.
2. **The working layers.** Their burning cost should be near their exposure-rated value. The scan below selects
   the `c` that gives the smallest squared log error on the two lowest layers.

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

The scan selects `c = 4` (squared log error 0.023 against 0.144 for `c = 3`). The last two lines show whether
that difference is important:

- The burning cost of 1 xs 1 comes from 27 claims and has a relative standard error of approximately 0.23.
- The burning cost of 2 xs 2 comes from 9 claims and has a relative standard error of approximately 0.46.

The judged `c = 3` misses them by 0.15 and 0.35 in log, each within one standard error. Thus the experience does
not reject `c = 3`. Use the scan as this type of diagnostic: it shows whether experience and curve disagree by
more than the noise. Keep a range of plausible `c` (here 3 to 4). The remainder of this guide uses `c = 3`.

The history is synthetic, so we can also compare `c = 3` with the process that drew it. Its layers in the table
above (1.771, 1.060, 0.675 and 0.295) are near the values of that process (1.701, 1.122, 0.648 and 0.427). A real
pricing never knows the process.

Two calibrations find `c` directly:

- `Mbbefd.swiss_re_for_mean_rate(rate)` takes a mean destruction rate. This is the mean loss as a share of
  the sum insured, from the claims of the cedant or a market study.
- `Mbbefd.swiss_re_for_layer_share(limit, attachment, mpl, share)` takes the share of the loss of one risk that
  a layer should carry.

Each calibration returns the curve and its `c`:

```python
curve, c = Mbbefd.swiss_re_for_mean_rate(0.10)
print(f"mean destruction rate 0.10: c = {c:.3f}, total-loss probability {curve.total_loss_probability():.4f}")
share = Mbbefd.swiss_re(3.0).layer_share(1.0, 1.0, 4.0)
curve, c = Mbbefd.swiss_re_for_layer_share(1.0, 1.0, 4.0, share)
print(f"a share of {share:.4f} for 1 xs 1 on a risk of 4: c = {c:.3f}")
```

```text
mean destruction rate 0.10: c = 2.861, total-loss probability 0.0402
a share of 0.1767 for 1 xs 1 on a risk of 4: c = 3.000
```

The second call gives back the `c = 3` that made the share.

The curve also needs a correct profile. If a band covers a wide range of sums insured, give its bounds.

```python
c3 = Mbbefd.swiss_re(3.0)
# 400 risks from 1 to 5 with a total sum insured of 800, a mean of 2 each
for label, kw in [("one sum insured of 2", {}),
                  ("uniform between 1 and 5", dict(lower=[1.0], upper=[5.0], spread="uniform")),
                  ("tilted to a mean of 2", dict(lower=[1.0], upper=[5.0], spread="tilted"))]:
    band = RiskProfile([2.0], [400], c3, expected_losses=[1.2], **kw)
    print(f"{label:25s} 3 xs 2: {band.expected_layer_loss(3.0, 2.0):.3f}, surplus: {band.expected_surplus_loss(2.0, 4.0):.3f}")
```

```text
one sum insured of 2      3 xs 2: 0.000, surplus: 0.000
uniform between 1 and 5   3 xs 2: 0.200, surplus: 0.450
tilted to a mean of 2     3 xs 2: 0.090, surplus: 0.210
```

When all risks have the mean of the band, 2, no risk is larger than 2. Thus a layer that attaches at 2 has no
value. A uniform spread between the bounds puts the mean at 3, which overstates the total sum insured of the
band. `spread="tilted"` keeps the mean at 2 and moves weight to the small risks. This agrees with a total of 800
over 400 risks: 0.090 for the layer against 0.200.

Give the spread that you intend. If you do not give it, the spread is uniform, and `RiskProfile` warns for each
band whose mean it moves. `spread="auto"` tilts each band whose sum insured is between its bounds.

### Surplus treaties

A surplus treaty cedes a share of each risk that depends on the sum insured of the risk. Thus it needs the
profile risk by risk. These are the terms:

- The cedant keeps the **retention** (one **line**) of each risk.
- The treaty takes up to `lines` more lines. Thus its **capacity** is `retention * lines`, and the maximum it
  covers on a risk is `retention * (1 + lines)`.
- A risk of sum insured `SI` cedes the share `min(SI - retention, capacity) / SI` of each loss.
- Any amount above the capacity stays with the cedant (or goes to facultative cover).

`expected_surplus_loss(retention, lines)` is the expected ceded loss. To rate a per-risk layer that inures to
the surplus, give the retention and lines to `expected_layer_loss`. The layer then applies to the part of each
risk that the cedant keeps.

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

With a retention of 2 and 4 lines, the treaty covers risks up to 10 in full. The risks of 20 cede 8 and keep 12.
Thus the cedant keeps more of them than the retention. The surplus takes 3.696 of the 9.900 expected loss. Net
of the surplus, all risks except the largest keep 2. Thus only the band of 20 still reaches 5 xs 5: 0.164 net
against 0.675 gross.

### A published table

`TabulatedCurve(x, g)` takes a table of `(x, G(x))` from `(0, 0)` to `(1, 1)`. The table must be concave, and the
curve interpolates linearly. The table must have small steps near zero, because the mean destruction rate is
the first chord `x1 / G(x1)`. Its layer shares are piecewise linear approximations (section 4 measures the
difference).

**Pitfalls**:

- A curve is a shape. It has no information about the occupancy, construction or protection of the cedant.
- The expected loss is the premium that you give times the loss ratio that you give. An inadequate premium gives
  a low price without a warning.
- `expected_claims()` counts losses of all sizes, most of them very small. It does not count only the losses
  that pierce a layer.
- The sum insured of a band is its MPL. Thus use the MPL when a risk cannot lose its full value.

## 4. Exposure rating: casualty

### What it is

For liability, the cost of a layer follows the **policy limit**. The shape of loss within a limit comes from a
**ground-up severity** (the claim before any limit or deductible). A primary actuary knows this as increased
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

`ilf(severity, limit, basic_limit)` is the LEV at `limit` divided by the LEV at `basic_limit` (here 0.25).
`loss_elimination_ratio(severity, deductible)` is the LEV at the deductible divided by the mean. Each line shows
the lognormal and then a Pareto scaled to the same mean. Thus the two severities differ only in shape.

They differ at each limit. The ILF from 0.25 to 10 is 2.561 for the lognormal and 1.531 for the Pareto. The
lognormal keeps 0.636 of its mean below 1, and the Pareto keeps 0.807. The shape of the severity, not the
arithmetic, controls the answer. The data does not contain the tail above the largest limit that you sell.

To rate a layer over a profile of policy limits, make one exposure curve for each limit with
`TabulatedCurve.from_severity(severity, mpl)`. It tabulates `G(x) = LEV(x * mpl) / LEV(mpl)` at 200 points.
`severity_exposure_curve(severity, mpl, x)` gives the same values at the `x` that you select. Then give the
profile a list of curves, one for each band.

```python
LIMITS = [1.0, 2.0, 5.0, 10.0]
POLICIES = [800, 300, 100, 40]
CASUALTY_PREMIUM = [8.0, 6.0, 4.0, 3.0]
curves = [TabulatedCurve.from_severity(ground_up, limit) for limit in LIMITS]
casualty = RiskProfile(LIMITS, POLICIES, curves, premiums=CASUALTY_PREMIUM, loss_ratio=0.65)
exact = (ground_up.lev(10.0) - ground_up.lev(5.0)) / ground_up.lev(10.0)
print(f"{len(curves[3].x)} points; share of a 10m policy in 5 xs 5: table {curves[3].layer_share(5.0, 5.0, 10.0):.4f}, exact {exact:.4f}")
print(f"expected loss {casualty.expected_loss():.3f}")
for limit, att in [(1.0, 1.0), (4.0, 1.0), (5.0, 5.0), (5.0, 2.0)]:
    print(f"{limit:g} xs {att:g}: {casualty.expected_layer_loss(limit, att):.3f}")
```

```text
201 points; share of a 10m policy in 5 xs 5: table 0.0617, exact 0.0617
expected loss 13.650
1 xs 1: 1.245
4 xs 1: 1.857
5 xs 5: 0.120
5 xs 2: 0.677
```

Half of the 201 points are geometric in `x` from 1e-6 of the limit, where the curve is steepest. The other half
are evenly spaced in `G`. The table gives the exact share of a policy with a limit of 10 in 5 xs 5 to four
decimals. A coarser grid that you make by hand misprices layers: 37 geometric points gave 0.0619 here. A policy
with a limit of 1 has no loss above 1, so 1 xs 1 comes from the other three bands. The value of 5 xs 5 is small
(0.120), because only the policies with a limit of 10 reach it.

The severity must be at the cost level of the treaty period, and trend does not reach a layer one for one.
Severity trend increases each claim. Thus more claims cross the attachment, and the claims that crossed it
already go further into the layer. The loss of an excess layer increases faster than the ground-up trend, and a
limit at the bottom increases slower. This is **leveraged trend**.

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

Five per cent a year on each claim gives 3.4% a year on the first 1 of each claim. It gives 6.2% on 1 xs 1 and
8.5% on 5 xs 5. A layer rated on the severity of today, or a burning cost on untrended losses, does not include that
leverage. The error increases with each year between the data and the treaty period.

The severity must also agree with the basis of the policies:

- Claims-made policies and occurrence policies report the same losses in different years.
- A per-claim limit caps each claimant. A per-occurrence limit caps the sum over all claimants in one event.
  Thus the severity behind a per-occurrence limit is the total of the occurrence.

A long-tail excess layer also develops for many years after its accident year.
[`reserving-excess-layers.md`](reserving-excess-layers.md) calculates its factors from the ground-up triangle
and a severity.

**Use it when** a casualty or professional liability program has policy limits and premiums by limit, and
little or no large-loss history. Also use it when a layer cuts through a limit: 5 xs 5 over policies of 5 and
10 sees only the policies of 10.

**Do not use it when** one of these conditions is true:

- The large claims of the cedant contradict the assumed severity.
- The policies share limits or aggregate across coverages. The method treats the loss of each policy alone, so
  there is no clash.

Legal cost (ALAE), trend and jurisdiction are choices that you put into the severity before this call. The
ground-up severity changes the answer more than any other input (the ILF above is 2.561 or 1.531). Thus run two
or three severities and report the range.

## 5. Credibility of the experience, layer by layer

### What it is

Experience is worth more in a low layer than in a high layer, because more losses pierce it. **Classical
(limited fluctuation) credibility** gives the number of claims that the experience needs to stand alone. This is
the **full credibility standard**. With fewer claims, the credibility is `Z = sqrt(n / standard)`, the square
root rule of primary indications. For layers, `n` is the number of claims **in the layer**, the losses that
pierce it, not the number in the book.

```python
lf = LimitedFluctuation(0.90, 0.10)
print(f"frequency standard {lf.frequency_standard():.0f} claims; at k = 0.05 it is {LimitedFluctuation(0.90, 0.05).frequency_standard():.0f}")
rows = []
for limit, att in LAYERS:
    cv = layer_cv(severity, limit, att)
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

`LimitedFluctuation(p, k)` asks that the observed value is within `k` of its mean with probability `p`. Here it
is 10% with probability 90%. The default `k = 0.05` makes each standard four times larger (1082 claims for the
frequency alone, against 271). The values of `p` and `k` are judgements, and you must record them.

The pure premium standard needs the coefficient of variation `cv` of the loss to the layer for each claim that
pierces it. `layer_cv` calculates it from the layer moments of the fitted Pareto. Each of these four layers has
a limit equal to its attachment. Thus they have the same `cv` by the scale invariance of the Pareto, and only
the number of claims is different.

The lowest layer has 27 claims and a `Z` of 0.257. The next layer has 9 claims and 0.148, then 2 claims and
0.070. The top layer has no claims and a `Z` of 0. In this example even the working layer is far from full
credibility. Thus exposure gets most of the weight.

Credibility increases with the size of the cedant (the working layer of a large cedant can have hundreds of
claims). It also increases with a looser standard. The values `p` and `k` control the result. Classical
credibility also treats the complement as exact, and it does not include the uncertainty of the exposure
estimate. Bühlmann-type weights (below) or a Bayesian model that weights both sources by their variances are
the alternatives.

### Blending

`credibility_weighted(estimate, credibility, complement)` is `Z * estimate + (1 - Z) * complement`. The burning
cost is the estimate, and the exposure-rated loss is the complement.

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

In 1 xs 1, the blend (1.706) is between the burning cost (1.518) and the exposure (1.771), nearer to the
exposure. In 5 xs 5, with two hits, the blend is 0.642 against an exposure of 0.675. In 10 xs 10, the blend is
the exposure alone. The higher the layer, the more the price comes from the profile and the curve.

There are three cautions:

1. The standard is a choice, and a looser `k` gives experience more weight. Thus show the sensitivity.
2. `Z` helps judgement, but does not replace it. Change the weights for an unusual single event or a changed
   book.
3. The method treats the complement as correct. If the curve is the weak part, blend against a range of `c`.

You can also weight frequency and severity separately, with `frequency_standard()` and
`severity_standard(cv)`.

### A complement from a collective of cedants

When you have loss ratios for many similar cedants or for many years of one cedant, use **Bühlmann–Straub**
credibility. It takes the weights from the data, not from a standard. `buhlmann_straub(ratios, weights)` takes one row for each
group (a cedant) and one column for each period, with a weight (premium) for each cell. The fit gives the
collective mean, the variance components, and the credibility and premium of each group.

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

The credibility of a cedant is its total weight divided by the sum of that weight and `k`. Here `k = 10.50` is
the variance within groups divided by the variance between groups. The two largest cedants (weights 135 and 72)
have a `Z` of 0.928 and 0.873. The smallest cedant (weight 23) has the most variable ratios and a `Z` of 0.687.
Its premium, 0.855, moves from its own mean of 0.904 toward the collective mean of 0.745.

Use it for a quantity with a natural weight in each period, for example a loss ratio or a rate on line by
treaty year. Here its output is the loss ratio that exposure rating needs. Assume that the profile of section
3 is the profile of cedant 1. Then its credibility loss ratio of 0.605, in place of the assumed 55%, prices
5 xs 5 at 0.742 against 0.675.

It is not a layer tool, because it needs a panel. The large losses of one cedant are too few to supply it. Thus
the limited fluctuation weights of section 5 are a usual choice for the weight of the own experience of a layer.

## 6. Reaching layers above the data

### What it is

Losses may never have hit the highest layers of a program. The lower layers give information about the slope of
the tail. When you know the expected losses of two layers, one Pareto alpha gives both. The same alpha prices
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
at which a Pareto gives both losses. One layer must be above the other. `pareto_extrapolation((limit,
attachment), (limit, attachment), alpha)` returns the expected loss of the second layer for each unit of the
expected loss of the first layer. Thus the product with the loss of the first layer prices the second layer.

The extrapolated 5 xs 5 is 0.295, below the blend (0.642) and the exposure value (0.675). The two layers come
from the light ten years. Their alpha of 2.017 is lighter than the 1.6 that drew them. The extrapolation uses
that alpha as the truth for all layers above. It cannot know that the sample was light.

Use the extrapolation in these three ways:

1. Between two layers that both have credible experience, to fill a gap.
2. Upward from the highest credible layer, with an alpha that you select. Do not use an alpha from two noisy
   points.
3. As a check on the exposure curve. Its implied alpha should not be very different from the alpha of the
   experience.

The last line of the output is that check. The `c = 3` curve gives 1.740 between the same two layers, against
2.017 from the experience. Take the outside alpha from your own benchmark for the line, or from the exposure
curve, as here.

### A whole tower, references and PML curves

These functions fit one model through several values:

- `match_tower(attachments, layer_losses, frequencies)` fits one model to a tower of contiguous layers, with the
  last layer unlimited (Riegel 2018). The model is a frequency and a piecewise Pareto severity that give each
  layer loss.
- `fit_references` takes expected layer losses from any source (they can overlap or leave gaps) and excess
  frequencies. It fits a model through all of them. It is good for one reference from experience, one from
  exposure and a known frequency.
- `fit_pml_curve(return_periods, amounts, tail_alpha)` fits a severity through a PML curve. The value
  `amounts[j]` occurs once in `return_periods[j]` years.

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
diag = refs.diagnostics()
print(f"alphas {diag['min_alpha']:.2f} to {diag['max_alpha']:.2f}; {len(diag['warnings'])} warnings")
pml = fit_pml_curve([2, 5, 10, 25], [2.5, 4.0, 6.0, 10.0], tail_alpha=1.8)
print(f"PML curve: excess frequency of 4 is {pml.excess_frequency(4.0):.2f}; 5 xs 5 {pml.layer_loss(5.0, 5.0):.3f}")
```

```text
[1.771, 1.456, 0.675, 0.295]
frequency above 1: 2.80; above 5: 0.238 (experience 0.20)
15 xs 15: 0.102
references: 1.706, 0.675, 0.295
between and above: 2.117, 0.234, 0.093
alphas 0.55 to 7.28; 7 warnings
PML curve: excess frequency of 4 is 0.20; 5 xs 5 0.369
```

The tower uses the experience frequency of 2.8 losses a year above 1. Its model has 0.238 losses a year above 5,
against 0.20 in the data (two in ten years). Thus experience and curve agree within the noise of two events. A
`TowerModel` has `frequency` and `severity` as properties, and `layer_loss(limit, attachment)` and
`excess_frequency(x)` as methods. The unlimited top layer sets the tail. Then the tower prices a layer above it
(15 xs 15: 0.102).

The inputs to `fit_references` were the blended 1 xs 1, the exposure 5 xs 5, the exposure unlimited layer above
10, and the frequency. The fit gives all three layer losses again. There are three cautions for this family of
fits:

- **They give the inputs exactly, noise included.** The alphas can jump between pieces (the smallest in this
  fit is 0.55). `diagnostics()` lists each alpha below 1, and each pair of adjacent alphas with a ratio of more
  than two. Both thresholds are arguments. Here there are two pieces below 1 and five jumps.
  `fit_references(..., alpha_bounds=(lo, hi))` keeps the alphas in a range. Read the layer losses and
  frequencies, not the alphas piece by piece.
- **The fit fills the gaps between references.** Here 2 xs 2 is 2.117, against 1.060 by exposure and 1.014 in
  the blend. Add a reference for each layer that is important to you, and compare it with the other evidence.
- **The tail above the highest reference is an assumption**: the alpha of the unlimited layer, or
  `default_alpha` (here 2.0).

## 7. When data are thin

With a small number of large losses, do these steps in this order:

1. **Start from exposure** if there is a profile, and give the range of `c`.
2. **Use the experience as a check and a small adjustment**, with the credibility of section 5. A low `Z` is a
   fact about the data.
3. **Take the alpha from outside the sample.** Use your own benchmark for the line, the lower layers, or the
   alpha from the exposure curve or tower (section 6).
4. **Use all the information in the losses.** A loss that a policy limit caps tells you only that it was at
   least that large (`Pareto.fit(..., censored=...)`). Losses reported above different thresholds go in
   `reporting_thresholds`.
5. **For catastrophe or scenario business, start from the PML curve** with `fit_pml_curve` and a tail alpha that
   you select.

```python
capped = [min(x, 5.0) for x in LOSSES]
flags = [x >= 5.0 for x in LOSSES]
print(f"alpha ignoring a 5m policy limit {Pareto.fit(capped, 1.0).alpha:.3f}, with censoring {Pareto.fit(capped, 1.0, censored=flags).alpha:.3f}")
few = LOSSES[-8:]
fit_few, se_few = Pareto.fit_with_standard_error(few, 1.0)
fit_all, se_all = Pareto.fit_with_standard_error(LOSSES, 1.0)
print(f"alpha from the last {len(few)} losses {fit_few.alpha:.3f} against {fit_all.alpha:.3f} from all")
print(f"standard error, about alpha / sqrt(n): {se_few:.3f} with {len(few)} losses, {se_all:.3f} with {len(LOSSES)}")
```

```text
alpha ignoring a 5m policy limit 1.927, with censoring 1.789
alpha from the last 8 losses 1.357 against 1.880 from all
standard error, about alpha / sqrt(n): 0.480 with 8 losses, 0.355 with 28
```

The first line assumes that a policy limit of 5 capped the two largest losses. When the fit treats the capped
amounts as exact, it gives 1.927. When it marks them as censored, it gives 1.789. Thus a fit that ignores a
limit makes the tail look lighter.

The second line fits the last eight losses, and gets 1.357. All 28 losses give 1.880. This subset contains the
two largest claims, so it is not a random sample. But the third line shows why any eight losses give an
uncertain alpha. The maximum likelihood alpha has a standard error of approximately `alpha / sqrt(n)`: 0.480
with eight losses, and still 0.355 with all 28. Each of these ranges changes the price of a high layer by a
large amount.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): the map of the guides, and which guide to read first.
- [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md): the risk load, after the expected loss.
- [`aggregate-distributions.md`](aggregate-distributions.md): the annual distribution of what the layers pay,
  from the frequency and severity of this guide.
- [`severity-and-frequency.md`](severity-and-frequency.md): how to fit and select the severity and the count.
- [`reserving-methods.md`](reserving-methods.md): the development of the losses before you use them as
  experience.
- [`reinsurance-price.md`](reinsurance-price.md): the price of a contract, from the expected loss to the
  technical premium.
- [`reserving-excess-layers.md`](reserving-excess-layers.md): development factors for an excess layer.
- `docs/design/pareto.md`, `docs/design/credibility.md` and `docs/design/aggregate.md`: the design of each
  method.
