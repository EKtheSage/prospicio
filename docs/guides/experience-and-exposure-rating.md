# Experience rating, exposure rating and blending them

<!-- site: execute -->

An excess of loss layer has two independent sources of evidence for its expected loss:

- **Experience rating**: what the large losses of the cedant cost the layer in the past.
- **Exposure rating**: what the book of risks of the cedant costs the layer under a standard loss shape.

Most layers need both, weighted by how far you can trust each.

**Use this guide when** you must find the expected loss of a layer. The years of large losses, a profile of
sums insured or policy limits, how often losses hit the layer and the largest loss in the data decide the
method. It is for an actuary who knows rating plans, increased limits factors and credibility, and who is new
to reinsurance pricing. It stops at the expected loss. Next: the risk load
([`risk-loads-and-distortions.md`](risk-loads-and-distortions.md)), the annual total
([`aggregate-distributions.md`](aggregate-distributions.md)), a full severity fit
([`severity-and-frequency.md`](severity-and-frequency.md)), primary ratemaking (section 1 of
[`choosing-a-method.md`](choosing-a-method.md)).

The layers here are plain per-risk or per-loss layers: no reinstatements, no annual aggregate limit or
deductible, no clash cover (one event that hits several risks). Those terms change the expected loss
([`reinsurance-price.md`](reinsurance-price.md)). `PreparedLosses.burning_cost` accepts towers with those terms, but not together with an index clause.

Money is in millions of a synthetic currency. CI runs and checks the code and every table.

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

## 1. Experience rating

### What it is

Apply the layer to each large loss of the as-if history and average over the years: the burning cost. The
second form fits a Poisson count of large losses and a Pareto severity, which gives the expected loss of any
layer in closed form, also one that the history did not reach. The example uses ten years of as-if losses above
1 from a synthetic property cedant.

```python
import math

import pandas as pd
from great_tables import GT

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

A layer is written limit xs attachment: `1 xs 1` pays the part of a loss between 1 and 2.

```python
LAYERS = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (10.0, 10.0)]   # (limit, attachment)
NAMES = [f"{limit:g} xs {att:g}" for limit, att in LAYERS]
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


burning = pd.DataFrame({
    "layer": NAMES,
    "losses above": [pierced(att) for _, att in LAYERS],
    "burning cost": [burning_cost(limit, att) for limit, att in LAYERS],
})
(GT(burning, rowname_col="layer")
 .tab_header("Burning cost of the as-if history", "Average annual layer loss over ten years, millions")
 .fmt_integer("losses above")
 .fmt_number("burning cost", decimals=3))
```

```text
          losses above  burning cost
1 xs 1              27         1.518
2 xs 2               9         0.750
5 xs 5               2         0.202
10 xs 10             0         0.000
```

`LargeLossExperience(ids, accident, amount, threshold=...)` holds the large losses (identifier, accident date
as a decimal year, amount). `prepare` restates them to the prospective period; the rate is 0 because this
history is already as-if. `burning_cost(tower, years, premium=..., development=...)` applies the tower to each
year, develops the layer total and divides by the premium. With a premium and a factor of 1.0,
`mean_loss_cost` is the average annual layer loss, one entry for each layer. The burning cost is exact for the
history, and that is its limit: no loss reached 10 xs 10, so its cost is zero, and zero is not a price.

```python
alpha_hat = Pareto.fit(LOSSES, 1.0).alpha
severity = Pareto(1.0, alpha_hat)
fitted = CollectiveModel(Poisson(len(LOSSES) / YEARS), severity)
drawn_from = CollectiveModel(Poisson(3.0), Pareto(1.0, 1.6))   # the model that drew this history
print(f"alpha {alpha_hat:.3f}, {len(LOSSES) / YEARS:.1f} losses above 1 a year")
```

```text
alpha 1.880, 2.8 losses above 1 a year
```

```python
(GT(pd.DataFrame({
    "layer": NAMES,
    "burning": [burning_cost(limit, att) for limit, att in LAYERS],
    "fitted": [fitted.layer_mean(limit, att) for limit, att in LAYERS],
    "drawn from": [drawn_from.layer_mean(limit, att) for limit, att in LAYERS],
}), rowname_col="layer")
 .tab_header("Expected annual layer loss", "Burning cost, fitted model and the model that drew the history, millions")
 .fmt_number(["burning", "fitted", "drawn from"], decimals=3))
```

```text
          burning  fitted  drawn from
1 xs 1      1.518   1.453       1.701
2 xs 2      0.750   0.789       1.122
5 xs 5      0.202   0.352       0.648
10 xs 10    0.000   0.191       0.427
```

`Pareto.fit(losses, t)` is the maximum likelihood alpha above the threshold `t`. `CollectiveModel(frequency,
severity).layer_mean(limit, attachment)` is the expected loss of any layer. The last column is the synthetic
model that drew these ten years (three losses a year, alpha 1.6); in practice you do not know it. These ten
years were lighter than their source, so both estimates are below it, and the gap grows with height.
The fit counts the loss of exactly 1.00 in 2019, so it sees 28 losses, but 27 pierce 1 xs 1.

A Poisson count and one Pareto alpha are assumptions. Before you price with them, compare the variance of the
yearly counts with their mean (if larger, use a negative binomial), examine the threshold and make a mean
excess plot ([`severity-and-frequency.md`](severity-and-frequency.md)). When one alpha does not fit,
`PiecewisePareto.fit(losses, thresholds)` fits one alpha for each band of size. The `truncation` argument of
`Pareto.fit` and `alpha_between_layers` caps the tail at a largest possible loss.

**Use it when** the layer has enough piercing losses for a stable burning cost (section 4 gives a number),
the portfolio is stable or restated to today's, and the as-if work is complete: trend, development of open
claims and indexation of retention and limit (`PreparedLosses`, the next subsection).

**Do not use it alone when**:

- The layer had few or no hits. The fitted model, with an alpha from the lower layers, is never the only
  price.
- The book changed. The past losses then describe a different portfolio.
- The losses are not as-if. A burning cost on untrended, undeveloped losses is too low. prospicio does not warn
  you: it accepts `prepare(rate=0.0, to=...)` with no development, as above.

**Data needed**: individual losses above a reporting threshold, each with its accident date, its amount at a
stated valuation date and its currency, and the subject premium of the cedant (or another exposure base) for
each year. Raw losses also need a trend rate, a development pattern for known claims, count IBNR factors and,
for an index clause, the index ratios.

The Pareto must fit above the threshold `t`. If sources report from different thresholds, or policy limits cap
some losses, tell the fit (section 6). Three steps of the as-if work are easy to miss:

- **Exposure volume.** The average above assumes a book of the same size each year. When the size changed, bring
  the premium of each year to today's rate level (`RateHistory`, section 1 of
  [`choosing-a-method.md`](choosing-a-method.md)) and pass it as `premium=`. `loss_cost` is then the sum of the
  developed layer losses over the sum of the premiums, which weights each year by its premium;
  `mean_loss_cost` is the average of the yearly rates. Multiply the rate by the prospective subject premium.
- **The threshold after trend.** Trend moves losses across the threshold: a loss just below 1 at its own date
  can be above 1 at today's cost, and it is not in the list. Pass the reporting threshold as `threshold=`.
  Then `prepared.complete_above` is the threshold times the largest trend factor, in the reported currency,
  before development. Each year is complete above this size. The test compares it with each loss's reported
  amount times its own trend factor, not with the as-if amount (which also includes the exchange rate and
  IBNER). The burning cost leaves out the losses below `complete_above`. A layer should attach well above
  `complete_above` in the pricing currency (times `fx`, with an allowance for development). If it does not,
  ask the cedant for losses from a lower threshold.
- **Legal cost (ALAE).** State whether it is in the loss (and thus erodes the layer), shared pro rata with the
  indemnity, or excluded. Use the same basis for the loss ratio of the exposure rating.

### Preparing raw losses

The listing of a cedant is seldom as-if. This one has six losses reported above 1, in a currency worth 1.1 of
the pricing currency. The valuation date is the start of 2026, and the average accident date of the treaty is
mid-2026. Known claims still develop, so first estimate that development (IBNER) from older claims valued at
several ages.

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
```

```text
IBNER to ultimate at ages [0.5, 1.5, 2.5, 3.5] : 1.359, 1.104, 1.025, 1.000
complete above 1.227
```

```python
table = prepared_raw.to_dict()
as_if = pd.DataFrame({
    "loss": table["id"],
    "year": [str(year) for year in table["year"]],
    "reported": table["amount"],
    "development": table["development"],
    "trend": table["factor"],
    "as-if": table["trended"],
})
(GT(as_if, rowname_col="loss")
 .tab_header("Raw losses restated as-if", "Reported amount, IBNER development, trend to mid-2026, millions")
 .fmt_number("reported", decimals=2)
 .fmt_number(["development", "trend", "as-if"], decimals=3))
```

```text
    year  reported  development  trend  as-if
r1  2022      2.10        1.000  1.227  2.835
r2  2022      1.05        1.007  1.198  1.394
r3  2023      3.40        1.023  1.163  4.449
r4  2024      1.02        1.072  1.124  1.353
r5  2024      2.60        1.129  1.097  3.544
r6  2025      1.30        1.282  1.065  1.954
```

`DevelopmentPattern.from_history(ids, age, amount, ages=...)` is a volume-weighted chain ladder on the claims:
the factor from one age to the next is the sum of the later amounts over the sum of the earlier amounts, for the
claims valued at both ages with a positive earlier amount. A claim first seen later is IBNR, not IBNER, and the
pattern leaves it out. A list cut at the reporting threshold on the latest incurred lacks the small claims that
grew later, which biases the factors, so judge the pattern. `DevelopmentPattern(ages, factors)` takes a pattern
that you select.

`prepare(rate=0.05, to=2026.5, development=ibner)` converts each loss at `fx`, develops it by the pattern at its
age at the valuation date (interpolated between ages, constant past the ends) and trends it by 5% a year from
its accident date to `to`. The as-if amount is the product: for r1, 2.10 times 1.1 times 1.000 times 1.227 is
2.835. You can give a trend factor (`factors=`) or a development factor for each loss in place of a rate or a
pattern.

The as-if amount of r4, 1.353, is above `complete_above` (1.227), but its reported 1.02 times its trend factor
of 1.124 is below. So the burning cost leaves out r4: a loss of the same trended size from the oldest year was
below the threshold at its own date and is not in the list.

Then apply the treaty by year. Each loss is already developed, so the development by year is only the count
IBNR factor (claims not yet reported). You can give a count `DevelopmentPattern`, for example from a
`ChainLadder` on claim counts; the function reads it at the average age of each year.

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
by_year = pd.DataFrame({
    "year": [str(year) for year in p["year"]],
    "losses": p["losses"],
    "developed layer loss": [a[0] for a in p["developed"]],
    "with the index clause": [b[0] for b in q["developed"]],
})
(GT(by_year, rowname_col="year")
 .tab_header("2 xs 2 by year", "Layer loss developed by the count IBNR factor, millions")
 .fmt_integer("losses")
 .fmt_number(["developed layer loss", "with the index clause"], decimals=3))
```

```text
      losses  developed layer loss  with the index clause
2022       2                 0.835                  0.835
2023       1                 2.000                  2.091
2024       1                 1.574                  1.537
2025       1                 0.000                  0.000
```

```python
print(f"rate on premium {plain.loss_cost[0]:.4f}, with the index clause {indexed.loss_cost[0]:.4f}")
print(f"on a prospective premium of 11: {11 * plain.loss_cost[0]:.3f}, {11 * indexed.loss_cost[0]:.3f}")
```

```text
rate on premium 0.1131, with the index clause 0.1145
on a prospective premium of 11: 1.244, 1.259
```

The 2024 layer loss, 1.574, is the 1.544 of r5 in the layer times the count factor 1.02.

`IndexClause(franchise, kind)` moves the retention and limit of each loss by its index ratio: the index at the
expected payment date over the index at the base date of the treaty (often the inception). Larger claims
settle later, so their ratios are higher. The two kinds:

- A severe inflation clause (`kind="severe"`) moves the terms only by the increase above the franchise, and
  never down.
- `kind="full"` moves them by the full ratio, up or down, when the ratio passes the franchise.

The clause lowers the 2024 layer loss but raises the 2023 one: r3 exhausts the layer, and the clause raises
the limit with the retention. The clause takes only per-occurrence terms: a tower with an annual aggregate
limit or deductible, or with limited reinstatements, gives an error.

`as_if(historical, years, proposed)` shows two views of each year. The historical side runs each loss as
reported through that year's program (here 1.5 xs 1.5 until 2023), untrended and undeveloped, in its own
currency. Compare these amounts with the cedant's reported recoveries to check the listing before you trend
anything. The proposed side is the as-if layer loss before development, without the index clause.

```python
history = prepared_raw.as_if([Tower([Layer("1.5 xs 1.5", 1.5, 1.5)])] * 2 + [XL] * 2, RAW_YEARS, XL)
h = history.to_dict()
(GT(pd.DataFrame({
    "year": [str(year) for year in h["year"]],
    "reported": h["reported_losses"],
    "program": [layers[0] for layers in h["historical_layers"]],
    "as reported": [was[0] for was in h["historical"]],
    "as-if, 2 xs 2": [now[0] for now in h["proposed"]],
}), rowname_col="year")
 .tab_header("Each year under its own program and as-if", "Layer loss before development, millions")
 .fmt_integer("reported")
 .fmt_number(["as reported", "as-if, 2 xs 2"], decimals=3))
```

```text
      reported     program  as reported  as-if, 2 xs 2
2022         2  1.5 xs 1.5        0.600          0.835
2023         1  1.5 xs 1.5        1.500          2.000
2024         2      2 xs 2        0.600          1.544
2025         1      2 xs 2        0.000          0.000
```

## 2. Exposure rating: property

### What it is

Exposure rating needs no loss history. The cedant gives a **risk profile**: bands of sum insured, with the
number of risks and the premium in each band. Each band gets an expected loss (premium times a loss ratio) and
an exposure curve, which gives the share of each risk's expected loss that is in the layer, as the limits
factors of a primary rating plan do.

The premium is the gross premium for the perils that the treaty covers, at the prospective rate level, before
commission. The loss ratio is the expected ground-up loss ratio to that premium for the treaty year, without
catastrophe losses and on the same ALAE basis as the layer. A loss ratio net of commission, or one that
includes catastrophe years, misstates each layer in proportion.

This method prices a layer that each risk triggers alone. For a catastrophe (per-event) layer, where one event
hits many risks, use a catastrophe model or the PML method of section 5.

In prospicio the profile is `RiskProfile` and the curve is `Mbbefd` or `TabulatedCurve`. This example has four
bands with a 55% loss ratio, and the four layers of section 1.

```python
SUMS_INSURED = [2.0, 5.0, 10.0, 20.0]
RISKS = [1500, 400, 120, 30]
PREMIUM = [7.2, 4.8, 3.6, 2.4]
LOSS_RATIO = 0.55


def profile(curve):
    return RiskProfile(SUMS_INSURED, RISKS, curve, premiums=PREMIUM, loss_ratio=LOSS_RATIO)


print(f"expected loss, all sizes {profile(Mbbefd.swiss_re(3.0)).expected_loss():.3f}")
```

```text
expected loss, all sizes 9.900
```

```python
by_c = []
for c in (2.0, 3.0, 4.0):
    curve = Mbbefd.swiss_re(c)
    by_c.append({
        "curve": f"c = {c:g}",
        **{name: profile(curve).expected_layer_loss(limit, att) for name, (limit, att) in zip(NAMES, LAYERS)},
        "total-loss probability": curve.total_loss_probability(),
    })
(GT(pd.DataFrame(by_c), rowname_col="curve")
 .tab_header("Expected annual layer loss by Swiss Re curve", "Millions; the book's expected loss is 9.900 for every c")
 .fmt_number(NAMES, decimals=3)
 .fmt_number("total-loss probability", decimals=4))
```

```text
       1 xs 1  2 xs 2  5 xs 5  10 xs 10  total-loss probability
c = 2   2.193   1.305   0.913     0.419                  0.1300
c = 3   1.771   1.060   0.675     0.295                  0.0327
c = 4   1.324   0.799   0.453     0.183                  0.0065
```

The parameter `c` moves loss between partial losses and total losses. A higher `c` gives fewer total losses
and cheaper higher layers. In the figure, each curve is the share of a risk's expected loss that lies below a
given loss, for a sum insured of 4. A high `c` rises fast (partial losses carry the loss); a low `c` rises
slowly (total losses carry it).

```python
import matplotlib.pyplot as plt

x = [4.0 * i / 40 for i in range(1, 41)]
fig, ax = plt.subplots()
for c in (1.5, 2.5, 3.5, 5.0):
    ax.plot(x, [Mbbefd.swiss_re(c).layer_share(v, 0.0, 4.0) for v in x], color="0.75")
for c in (2.0, 3.0, 4.0):
    ax.plot(x, [Mbbefd.swiss_re(c).layer_share(v, 0.0, 4.0) for v in x], label=f"c = {c:g}")
ax.set_title("Swiss Re exposure curves (grey: c = 1.5, 2.5, 3.5 and 5)")
ax.set_xlabel("loss limit (sum insured of the risk = 4)")
ax.set_ylabel("share of the expected loss")
ax.legend()
```

### Choosing the curve

`Mbbefd.swiss_re(c)` is Bernegger's one-parameter family. The values 1.5, 2, 3 and 4 of `c` are the Swiss Re
curves, and 5 is the Lloyd's curve. By the usual market reading, lower values suit small, vulnerable risks,
where a fire usually destroys the full value; higher values suit large industrial risks, where protection makes
partial losses usual. This guide selects `c = 3` by judgement (a middle curve for a commercial book of mixed
occupancy) and checks it twice:

1. **Total loss probability.** `total_loss_probability()` is the probability that a loss destroys the full sum
   insured. If the cedant's claims show a very different rate, the curve is wrong.
2. **The working layers.** Their burning cost should be near their exposure-rated value. The scan below picks
   the `c` with the smallest squared log error on the two lowest layers.

```python
def log_error(c):
    p = profile(Mbbefd.swiss_re(c))
    return sum(math.log(p.expected_layer_loss(l, a) / burning_cost(l, a)) ** 2 for l, a in LAYERS[:2])


grid = [1.5 + 0.25 * i for i in range(15)]
best = min(grid, key=log_error)
print(f"best c on the working layers: {best:g}")
print("squared log error: " + ", ".join(f"c = {c:g} {log_error(c):.3f}" for c in (2.0, 3.0, 4.0, 5.0)))
p_best = profile(Mbbefd.swiss_re(best))
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
1 xs 1: relative standard error of the burning cost 0.23, log gap at c = 3 0.15
2 xs 2: relative standard error of the burning cost 0.46, log gap at c = 3 0.35
```

The burning cost of each layer against the exposure value at `c = 3` and `c = 4`:

```python
import numpy as np

by_layer = pd.DataFrame({
    "layer": NAMES,
    "burning cost": [burning_cost(l, a) for l, a in LAYERS],
    "exposure, c = 3": [profile(Mbbefd.swiss_re(3.0)).expected_layer_loss(l, a) for l, a in LAYERS],
    "exposure, c = 4": [p_best.expected_layer_loss(l, a) for l, a in LAYERS],
})
pos = np.arange(len(NAMES))
fig, ax = plt.subplots()
for k, col in enumerate(by_layer.columns[1:]):
    ax.bar(pos + (k - 1) * 0.27, by_layer[col], width=0.27, label=col)
ax.set_xticks(pos, NAMES)
ax.set_title("Burning cost against exposure rating, by layer")
ax.set_xlabel("layer (limit xs attachment)")
ax.set_ylabel("expected annual loss (monetary units)")
ax.legend()
```

```python
(GT(by_layer, rowname_col="layer")
 .tab_header("Expected annual loss of each layer", "Burning cost of ten years against the exposure curve, millions")
 .fmt_number(["burning cost", "exposure, c = 3", "exposure, c = 4"], decimals=3))
```

```text
          burning cost  exposure, c = 3  exposure, c = 4
1 xs 1           1.518            1.771            1.324
2 xs 2           0.750            1.060            0.799
5 xs 5           0.202            0.675            0.453
10 xs 10         0.000            0.295            0.183
```

The scan selects `c = 4`. The last two lines of its output show whether the gap to `c = 3` matters: the
burning cost of 1 xs 1 (27 claims) and of 2 xs 2 (9 claims) has a relative standard error of 0.23 and 0.46,
and `c = 3` misses them by 0.15 and 0.35 in log, each within one standard error. So the experience does not
reject `c = 3`. Use the scan to see whether experience and curve disagree by more than the noise, and keep a
range of plausible `c` (here 3 to 4). The rest of this guide uses `c = 3`, which is also near the "drawn from"
column of the first layer table (a real pricing never knows it).

Two calibrations find `c` directly, and each returns the curve and its `c`:

- `Mbbefd.swiss_re_for_mean_rate(rate)` takes a mean destruction rate: the mean loss as a share of the sum
  insured, from the claims of the cedant or a market study.
- `Mbbefd.swiss_re_for_layer_share(limit, attachment, mpl, share)` takes the share of the loss of one risk that
  a layer should carry. Here it gives back the `c = 3` that made the share.

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

If a band covers a wide range of sums insured, give its bounds. Here 400 risks from 1 to 5 have a total sum
insured of 800, a mean of 2 each.

```python
c3 = Mbbefd.swiss_re(3.0)
spreads = []
for label, kw in [("one sum insured of 2", {}),
                  ("uniform between 1 and 5", dict(lower=[1.0], upper=[5.0], spread="uniform")),
                  ("tilted to a mean of 2", dict(lower=[1.0], upper=[5.0], spread="tilted"))]:
    band = RiskProfile([2.0], [400], c3, expected_losses=[1.2], **kw)
    spreads.append((label, band.expected_layer_loss(3.0, 2.0), band.expected_surplus_loss(2.0, 4.0)))
(GT(pd.DataFrame(spreads, columns=["spread", "3 xs 2", "surplus"]), rowname_col="spread")
 .tab_header("One band, three spreads", "Expected loss in a layer and in a surplus of 4 lines on a retention of 2, millions")
 .fmt_number(["3 xs 2", "surplus"], decimals=3))
```

```text
                         3 xs 2  surplus
one sum insured of 2      0.000    0.000
uniform between 1 and 5   0.200    0.450
tilted to a mean of 2     0.090    0.210
```

If all risks sit at the band mean, 2, no risk exceeds 2 and a layer that attaches at 2 has no value. A uniform
spread puts the mean at 3, which overstates the band's total sum insured. `spread="tilted"` keeps the mean at 2
and moves weight to the small risks, which agrees with 800 over 400 risks. The default is uniform, and
`RiskProfile` warns for each band whose mean it moves. `spread="auto"` tilts each band whose sum insured is
between its bounds.

### Surplus treaties

A surplus treaty cedes a share that depends on each risk's sum insured, so it needs the profile risk by risk:

- The cedant keeps the **retention** (one **line**) of each risk.
- The treaty takes up to `lines` more lines. Its **capacity** is `retention * lines`.
- A risk of sum insured `SI` cedes the share `min(SI - retention, capacity) / SI` of each loss. Any amount above
  the capacity stays with the cedant (or goes to facultative cover).

`expected_surplus_loss(retention, lines)` is the expected ceded loss. To rate a per-risk layer that inures to
the surplus, give the retention and lines to `expected_layer_loss`: the layer then applies to the part that
the cedant keeps.

```python
# Surplus treaty and a per-risk layer on what is left of each risk
p3 = profile(Mbbefd.swiss_re(3.0))
retention, lines = 2.0, 4.0
split = []
for si in SUMS_INSURED:
    ceded = min(max(si - retention, 0.0), retention * lines)
    split.append((f"{si:g}", ceded, si - ceded))
(GT(pd.DataFrame(split, columns=["sum insured", "ceded", "kept"]), rowname_col="sum insured")
 .tab_header("Surplus treaty, retention 2 and 4 lines", "Part of each risk ceded and kept")
 .fmt_number(["ceded", "kept"], decimals=0))
```

```text
    ceded  kept
2       0     2
5       3     2
10      8     2
20      8    12
```

```python
print(f"surplus, retention 2 and 4 lines: {p3.expected_surplus_loss(2.0, 4.0):.3f}")
print(f"5 xs 5 gross {p3.expected_layer_loss(5.0, 5.0):.3f}, net of the surplus {p3.expected_layer_loss(5.0, 5.0, 2.0, 4.0):.3f}")
```

```text
surplus, retention 2 and 4 lines: 3.696
5 xs 5 gross 0.675, net of the surplus 0.164
```

The treaty cedes all above the retention for risks up to 10. The risks of 20 keep 12: the retention and the 10 above the
capacity of 8. Net of the surplus, only the band of 20 still reaches 5 xs 5.

### A published table

`TabulatedCurve(x, g)` takes a table of `(x, G(x))` from `(0, 0)` to `(1, 1)`. The table must be concave, and the
curve interpolates linearly. The table must have small steps near zero, because the mean destruction rate is
the first chord `x1 / G(x1)`. Its layer shares are piecewise linear approximations (section 3 measures the
difference).

**Pitfalls**:

- A curve is a shape. It has no information about the occupancy, construction or protection of the cedant.
- The expected loss is the premium that you give times the loss ratio that you give. An inadequate premium gives
  a low price without a warning.
- `expected_claims()` counts losses of all sizes, most of them very small, not only those that pierce a layer.
- The sum insured of a band is its MPL. Use the MPL when a risk cannot lose its full value.

## 3. Exposure rating: casualty

### What it is

For liability, the cost of a layer follows the **policy limit**. The shape of loss within a limit comes from a
**ground-up severity** (the claim before any limit or deductible). ILFs and loss elimination ratios are both
ratios of limited expected values.

```python
# Casualty: a ground-up severity, ILFs and limited expected values
ground_up = Lognormal(math.log(0.05), 1.8)
heavy = Pareto(ground_up.mean() / 3.0, 1.5)   # alpha 1.5, scaled to the same mean
print(f"mean {ground_up.mean():.3f} (lognormal), {heavy.mean():.3f} (Pareto)")
```

```text
mean 0.253 (lognormal), 0.253 (Pareto)
```

```python
limits = (1.0, 2.0, 5.0, 10.0)
(GT(pd.DataFrame({
    "limit": [f"{limit:g}" for limit in limits],
    "ILF, lognormal": [ilf(ground_up, limit, 0.25) for limit in limits],
    "ILF, Pareto": [ilf(heavy, limit, 0.25) for limit in limits],
    "LER, lognormal": [loss_elimination_ratio(ground_up, limit) for limit in limits],
    "LER, Pareto": [loss_elimination_ratio(heavy, limit) for limit in limits],
}), rowname_col="limit")
 .tab_header("Increased limits factors and loss elimination ratios", "Basic limit 0.25; the two severities have the same mean")
 .fmt_number(["ILF, lognormal", "ILF, Pareto", "LER, lognormal", "LER, Pareto"], decimals=3))
```

```text
    ILF, lognormal  ILF, Pareto  LER, lognormal  LER, Pareto
1            1.737        1.316           0.636        0.807
2            2.071        1.408           0.758        0.863
5            2.403        1.490           0.880        0.913
10           2.561        1.531           0.938        0.939
```

`ilf(severity, limit, basic_limit)` is the LEV at `limit` over the LEV at `basic_limit` (here 0.25).
`loss_elimination_ratio(severity, deductible)` is the LEV at the deductible over the mean. The two severities
have the same mean and differ only in shape, yet they differ at each limit. The shape controls the answer, and
the data does not contain the tail above the largest limit that you sell.

To rate a layer over a profile of policy limits, make one exposure curve for each limit with
`TabulatedCurve.from_severity(severity, mpl)`. It tabulates `G(x) = LEV(x * mpl) / LEV(mpl)` at 200 points
(`severity_exposure_curve(severity, mpl, x)` gives the same values at the `x` that you select). Give the
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

Half of the 201 points are geometric in `x` from 1e-6 of the limit, where the curve is steepest, and half are
evenly spaced in `G`. A coarser hand-made grid misprices layers (37 geometric points gave 0.0619). A policy
with a limit of 1 has no loss above 1, so 1 xs 1 comes from the other three bands, and only the policies with
a limit of 10 reach 5 xs 5.

The severity must be at the cost level of the treaty period, and trend does not reach a layer one for one.
Severity trend raises each claim, so more claims cross the attachment, and the claims that crossed it go
further into the layer. An excess layer grows faster than the ground-up trend, and a bottom layer slower. This
is **leveraged trend**.

```python
# Five per cent a year of severity trend for three years, on the lognormal
f = trend_factor(0.05, 3.0)
trended = Lognormal(math.log(0.05 * f), 1.8)
ratio_up = trended.mean() / ground_up.mean()
trend_rows = [("ground up", ratio_up, ratio_up ** (1 / 3) - 1)]
for limit, att in [(1.0, 0.0), (1.0, 1.0), (5.0, 5.0)]:
    ratio = (trended.lev(att + limit) - trended.lev(att)) / (ground_up.lev(att + limit) - ground_up.lev(att))
    trend_rows.append((f"{limit:g} xs {att:g}", ratio, ratio ** (1 / 3) - 1))
(GT(pd.DataFrame(trend_rows, columns=["layer", "three-year factor", "a year"]), rowname_col="layer")
 .tab_header("Leveraged trend", "Five per cent a year of severity trend for three years")
 .fmt_number("three-year factor", decimals=3)
 .fmt_percent("a year", decimals=1))
```

```text
           three-year factor  a year
ground up              1.158    5.0%
1 xs 0                 1.106    3.4%
1 xs 1                 1.197    6.2%
5 xs 5                 1.276    8.5%
```

A layer rated on today's severity, or a burning cost on untrended losses, misses that leverage, and the error
grows with each year between the data and the treaty period.

The severity must also agree with the basis of the policies. Claims-made and occurrence policies report the
same losses in different years. A per-claim limit caps each claimant; a per-occurrence limit caps the sum over
all claimants in one event, so its severity is the total of the occurrence. A long-tail excess layer also
develops for many years ([`reserving-excess-layers.md`](reserving-excess-layers.md)).

**Use it when** a casualty or professional liability program has policy limits and premiums by limit, and
little or no large-loss history, or when a layer cuts through a limit: 5 xs 5 over policies of 5 and 10 sees
only the policies of 10.

**Do not use it when** the large claims of the cedant contradict the assumed severity, or the policies share
limits or aggregate across coverages (the method treats each policy alone, so there is no clash).

Put legal cost (ALAE), trend and jurisdiction into the severity before this call. The ground-up severity
changes the answer more than any other input, so run two or three severities and report the range.

## 4. Credibility of the experience, layer by layer

### What it is

Experience is worth more in a low layer, because more losses pierce it. **Classical (limited fluctuation)
credibility** gives the number of claims that the experience needs to stand alone: the **full credibility
standard**. With fewer claims, `Z = sqrt(n / standard)`. For layers, `n` is the number of claims **in the
layer**, the losses that pierce it, not the number in the book.

```python
lf = LimitedFluctuation(0.90, 0.10)
print(f"frequency standard {lf.frequency_standard():.0f} claims; at k = 0.05 it is {LimitedFluctuation(0.90, 0.05).frequency_standard():.0f}")
```

```text
frequency standard 271 claims; at k = 0.05 it is 1082
```

```python
rows, z_rows = [], []
for limit, att in LAYERS:
    cv = layer_cv(severity, limit, att)
    standard = lf.pure_premium_standard(cv)
    z = lf.credibility(pierced(att), standard)
    rows.append(z)
    z_rows.append((f"{limit:g} xs {att:g}", pierced(att), cv, standard, z))
(GT(pd.DataFrame(z_rows, columns=["layer", "claims", "cv", "standard", "Z"]), rowname_col="layer")
 .tab_header("Credibility of the burning cost", "Limited fluctuation, within 10% with probability 90%")
 .fmt_integer("claims")
 .fmt_number("cv", decimals=3)
 .fmt_integer("standard")
 .fmt_number("Z", decimals=3))
```

```text
          claims     cv  standard      Z
1 xs 1        27  0.717       410  0.257
2 xs 2         9  0.717       410  0.148
5 xs 5         2  0.717       410  0.070
10 xs 10       0  0.717       410  0.000
```

`LimitedFluctuation(p, k)` asks that the observed value is within `k` of its mean with probability `p`. The
default `k = 0.05` makes each standard four times larger. Record your `p` and `k`: they are judgements. The
pure premium standard needs the coefficient of variation `cv` of the loss to the layer per piercing claim;
`layer_cv` takes it from the layer moments of the fitted Pareto. These four layers have a limit equal to their
attachment, so by the scale invariance of the Pareto they share one `cv`.

Even the working layer is far from full credibility, so exposure gets most of the weight. Credibility grows
with the size of the cedant (a large cedant's working layer can have hundreds of claims) and with a looser
standard. Classical credibility treats the complement as exact. Bühlmann-type weights (below) or a Bayesian
model that weights both sources by their variances are the alternatives.

### Blending

`credibility_weighted(estimate, credibility, complement)` is `Z * estimate + (1 - Z) * complement`. The burning
cost is the estimate, and the exposure-rated loss is the complement.

```python
blend_rows = []
for (limit, att), z in zip(LAYERS, rows):
    exp_ = burning_cost(limit, att)
    expo = p3.expected_layer_loss(limit, att)
    blend_rows.append((f"{limit:g} xs {att:g}", exp_, expo, z, credibility_weighted(exp_, z, expo)))
(GT(pd.DataFrame(blend_rows, columns=["layer", "burning", "exposure", "Z", "blend"]), rowname_col="layer")
 .tab_header("Experience blended with exposure", "Expected annual layer loss at c = 3, millions")
 .fmt_number(["burning", "exposure", "Z", "blend"], decimals=3))
```

```text
          burning  exposure      Z  blend
1 xs 1      1.518     1.771  0.257  1.706
2 xs 2      0.750     1.060  0.148  1.014
5 xs 5      0.202     0.675  0.070  0.642
10 xs 10    0.000     0.295  0.000  0.295
```

The higher the layer, the more the price comes from the profile and the curve; in 10 xs 10 it is the exposure
alone. Three cautions: the standard is a choice (a looser `k` gives experience more weight, so show the
sensitivity); `Z` does not replace judgement (change the weights for an unusual single event or a changed
book); and the complement is taken as correct (if the curve is the weak part, blend against a range of `c`).
You can also weight frequency and severity separately, with `frequency_standard()` and `severity_standard(cv)`.

### A complement from a collective of cedants

With loss ratios for many similar cedants, or many years of one cedant, use **Bühlmann–Straub** credibility.
It takes the weights from the data, not from a standard. `buhlmann_straub(ratios, weights)` takes one row for
each group (a cedant), one column for each period and a weight (premium) for each cell. The fit gives the
collective mean, the variance components, and each group's credibility and premium.

```python
ratios = [[0.50, 0.60, 0.55, 0.70, 0.65, 0.50], [0.90, 0.80, 1.10, 0.70, 0.95, 0.85],
          [0.60, 0.70, 0.50, 0.65, 0.60, 0.70], [1.20, 0.40, 0.90, 1.50, 0.60, 0.80]]
premiums = [[10, 11, 12, 12, 13, 14], [20, 22, 21, 23, 24, 25], [5, 5, 6, 6, 7, 7], [3, 3, 4, 4, 4, 5]]
fit = buhlmann_straub(ratios, premiums)
print(f"collective {fit.collective_mean:.3f}, k {fit.k:.2f}")
```

```text
collective 0.745, k 10.50
```

```python
(GT(pd.DataFrame({
    "cedant": [f"cedant {i + 1}" for i in range(4)],
    "mean": fit.group_means,
    "weight": fit.group_weights,
    "Z": fit.credibility,
    "premium": fit.premiums,
}), rowname_col="cedant")
 .tab_header("Bühlmann–Straub credibility", "Loss ratio of each cedant; the collective mean is 0.745")
 .fmt_number(["mean", "Z", "premium"], decimals=3)
 .fmt_number("weight", decimals=0))
```

```text
           mean  weight      Z  premium
cedant 1  0.584      72  0.873    0.605
cedant 2  0.880     135  0.928    0.871
cedant 3  0.625      36  0.774    0.652
cedant 4  0.904      23  0.687    0.855
```

The credibility of a cedant is its total weight over that weight plus `k`, where `k` (10.50) is the variance
within groups over the variance between groups. The smallest cedant has the most variable ratios, so its
premium moves furthest toward the collective mean.

Use it for a quantity with a natural weight in each period, for example a loss ratio or a rate on line by
treaty year. Its output is the loss ratio that exposure rating needs. If the property profile above is
cedant 1's, its credibility loss ratio of 0.605 replaces the assumed 55%:

```python
own = RiskProfile(SUMS_INSURED, RISKS, c3, premiums=PREMIUM, loss_ratio=fit.premiums[0])
print(f"cedant 1's profile at its credibility loss ratio: 5 xs 5 {own.expected_layer_loss(5.0, 5.0):.3f}")
```

```text
cedant 1's profile at its credibility loss ratio: 5 xs 5 0.742
```

It is not a layer tool, because it needs a panel. One cedant's large losses are too few, so use the limited
fluctuation weights above for the own experience of a layer.

## 5. Reaching layers above the data

### What it is

Losses may never have hit the highest layers of a program, but the lower layers show the slope of the tail.
Two layers' expected losses fix one Pareto alpha, which prices the next layer.

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
at which a Pareto gives both losses (one layer above the other). `pareto_extrapolation((limit, attachment),
(limit, attachment), alpha)` returns the second layer's expected loss per unit of the first layer's.

The extrapolated 5 xs 5 is 0.295, below the blend (0.642) and the exposure value (0.675). Both layers come from
the light ten years: their alpha of 2.017 is lighter than the 1.6 that drew them, and the extrapolation takes
it as the truth for all layers above. Use it:

1. Between two layers that both have credible experience, to fill a gap.
2. Upward from the highest credible layer, with an alpha that you select. Do not use an alpha from two noisy
   points.
3. As a check on the exposure curve. The last line is that check: the `c = 3` curve implies 1.740 between the
   same two layers, against 2.017 from the experience. Take the outside alpha from your benchmark for the
   line or from the exposure curve, as here.

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
against 0.20 in the data (two in ten years): experience and curve agree within the noise of two events. A
`TowerModel` has `frequency` and `severity` as properties, and `layer_loss(limit, attachment)` and
`excess_frequency(x)` as methods. The unlimited top layer sets the tail, so the tower prices 15 xs 15.

The inputs to `fit_references` were the blended 1 xs 1, the exposure 5 xs 5, the exposure unlimited layer above
10 and the frequency. Three cautions for this family of fits:

- **They give the inputs exactly, noise included.** The alphas can jump between pieces (the smallest in this
  fit is 0.55). `diagnostics()` lists each alpha below 1, and each pair of adjacent alphas with a ratio of more
  than two (both thresholds are arguments). Here there are two pieces below 1 and five jumps.
  `fit_references(..., alpha_bounds=(lo, hi))` keeps the alphas in a range. Read the layer losses and
  frequencies, not the alphas.
- **The fit fills the gaps between references.** Here 2 xs 2 is 2.117, against 1.060 by exposure and 1.014 in
  the blend. Add a reference for each layer that matters, and compare it with the other evidence.
- **The tail above the highest reference is an assumption**: the alpha of the unlimited layer, or
  `default_alpha` (here 2.0).

## 6. When data are thin

With a small number of large losses, do these steps in this order:

1. **Start from exposure** if there is a profile, and give the range of `c`.
2. **Use the experience as a check and a small adjustment**, with the credibility of section 4. A low `Z` is a
   fact about the data.
3. **Take the alpha from outside the sample**: your benchmark for the line, the lower layers, or the exposure
   curve or tower (section 5).
4. **Use all the information in the losses.** A loss that a policy limit caps was at least that large
   (`Pareto.fit(..., censored=...)`). Losses reported above different thresholds go in
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

The first line assumes that a policy limit of 5 capped the two largest losses: a fit that treats the capped
amounts as exact makes the tail look lighter. The second line fits the last eight losses (not a random sample,
since they hold the two largest claims). The third line shows the uncertainty: the standard error is about
`alpha / sqrt(n)`, still 0.355 with all 28. Each of these ranges changes the price of a high layer by a large
amount.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): the map of the guides.
- [`reserving-methods.md`](reserving-methods.md): the development of the losses before you use them as
  experience.
- [`reinsurance-price.md`](reinsurance-price.md): from the expected loss to the technical premium.
- `docs/design/pareto.md`, `docs/design/credibility.md` and `docs/design/aggregate.md`: the design of each
  method.

## Appendix: terms

- **Large loss**: a claim above the **reporting threshold** (here 1). The cedant lists each large loss.
- **As-if**: a historical loss restated to the prospective year: trended to the future price level, developed
  to ultimate and measured against the contract that you price (its retention and limit, with any
  indexation). `LargeLossExperience` and `PreparedLosses` make them (section 1, Preparing raw losses):
  `prepare` applies currency, development (IBNER) and trend, `burning_cost` the index clause.
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
