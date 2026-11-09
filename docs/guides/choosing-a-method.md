# Choosing a method: which prospicio tool for which job

<!-- site: execute -->

## Use this guide when

You price primary insurance (rating plans, GLMs, loss ratio indications) and
now do reinsurance work. This guide is the map of the guide set. For each
situation it gives the usual data, the method and why, the calls, the pitfalls
and the guide with more detail. It does not teach the methods.

Money is in millions of a synthetic currency. The numbers show how to read the
output, not what a market pays. Each code block runs alone, except where it
says that it uses an earlier block. A layer is limit xs attachment: "4 xs 6"
pays the part of a loss between 6 and 10. Calls take the limit first:
`Layer("4x6", 4.0, 6.0)`, `expected_layer_loss(4.0, 6.0)`.

## The decision table

| Situation | Method | prospicio entry point | Why | Guide |
|---|---|---|---|---|
| Primary rates, overall rate indication | Loss ratio or pure premium method, with on-level premium, trend, development and credibility. Class relativities from a GLM before that | `RateHistory`, `LossRatioExperience`, `PurePremiumExperience`, `Provisions`, `LimitedFluctuation`. `prospicio.models.Glm` for relativities | Standard ratemaking. Premium and losses go to the same rate level and cost level | [Experience and exposure rating](experience-and-exposure-rating.md) |
| Per-risk property excess of loss, you have sums insured | Exposure rating with a curve | `RiskProfile.expected_layer_loss`, `Mbbefd` | There are too few large losses to rate a layer, and you know the profile of limits | [Experience and exposure rating](experience-and-exposure-rating.md) |
| Per-risk property or casualty excess of loss, you have large losses | Experience rating: prepare the losses, then fit a severity above a threshold and apply it to the layer, or burn the layer | `LargeLossExperience`, `.prepare`, `PreparedLosses.burning_cost`. `Pareto.fit`, `CollectiveModel.layer_mean` | The history of the cedant is the best evidence of its tail, when the history is sufficient. First trend the losses to the treaty period and develop them to ultimate | [Experience and exposure rating](experience-and-exposure-rating.md), [Severity and frequency](severity-and-frequency.md) |
| Casualty excess of loss, you have a profile of policy limits and premium | Exposure rating with increased limit factors: the premium of each limit times an expected loss ratio, split into the layer by ILFs | `ilf` in a loop over the limits | There are too few excess losses, and you know the limits written | This guide, section 3 |
| Casualty excess of loss, you have a price for one layer and need it for another | Increased limit factors and Pareto extrapolation, or match a tower | `ilf`, `pareto_extrapolation`, `match_tower` | The layers that you can see give the price of the layers above the data | [Experience and exposure rating](experience-and-exposure-rating.md) |
| Catastrophe excess of loss | Use the output of the cat model: an event loss table, a year loss table, or return period points | `EventLossTable` (events with secondary uncertainty), `EventSet.from_years`, `EventSet.from_long`, `EventSet.grouped`, `fit_pml_curve`, `Tower.apply` | The model gives frequency and severity. You apply the terms and price the risk | [What a reinsurance price is](reinsurance-price.md) |
| Quota share, surplus | Technical commission from the loss ratio distribution, with its centre at the indicated loss ratio of the cedant (first row) | `Layer.quota_share`, `Layer.surplus`, `with_sliding_scale`, `Tower.on_grid`. `Tower.on_annual_grid` on a loss-ratio grid. `price(grid, Distortion.tvar(0.99), cost_of_capital=r)` for the margin | The price is the commission. The full loss ratio is at risk, not only large losses | [What a reinsurance price is](reinsurance-price.md) |
| Aggregate stop loss, adverse development or loss portfolio cover | Distribution of the aggregate, then the cover on it. For reserves, a reserve bootstrap | `Layer.stop_loss`, `Tower.on_grid`. `OdpBootstrap` or `MackBootstrap`, then `Tower.apply_aggregate` | The cover pays on the total of the year, so you need the distribution of the total, not one loss | [Aggregate distributions](aggregate-distributions.md) |
| Reserve and its uncertainty, lifetime or one year | Chain ladder, Mack, Bornhuetter-Ferguson. Bootstraps for a distribution | `Triangle`, `ChainLadder`, `Mack`, `OdpBootstrap`, `MackBootstrap` | Triangles show development. Bootstraps give a distribution that you can combine | [Reserving methods](reserving-methods.md) |
| Capital, risk measures, allocation | Distortion risk measures. Euler allocation | `Distortion`, `capital`, `allocate` | One coherent family prices and allocates | [Risk loads and distortions](risk-loads-and-distortions.md) |
| A contract in an existing portfolio | Price on the portfolio, allocate by co-measure | `Portfolio.from_independent`, `price_portfolio` | Diversification changes what a contract needs | [Risk loads and distortions](risk-loads-and-distortions.md) |

Terms in the table:

- The **cedant** is the insurer that buys the reinsurance.
- An **on-level** premium is past premium at the current rates.
- **ILF** is an increased limit factor (section 3).
- **OEP** is the occurrence exceedance curve of a catastrophe model
  (section 4).

Two questions select the row:

1. What do you know about the cedant? A loss history points to experience
   rating. A list of sums insured points to exposure rating. The output of a
   catastrophe model points to the use of that output.
2. On what does the cover pay? One loss (an excess of loss layer), the total
   of the year (stop loss), a share of all losses (quota share) or an event
   (catastrophe). The answer selects how you represent losses.

## Three ways to represent losses

Almost all methods below need a model of the losses of the year. Where the
three ways overlap, they agree within grid and Monte Carlo error.

| Way | You give | You get | Use when |
|---|---|---|---|
| Grid (exact) | a claim count and a severity | the annual loss as a distribution on a grid of steps, with no simulation error | the cover depends only on amounts: excess of loss, quota share, stop loss, aggregate limits, reinstatements pro rata as to amount |
| Simulation | a claim count and a severity, a risk profile or a year loss table | simulated years of events, with Monte Carlo error | the cover needs event times (reinstatements pro rata as to time) or sums insured (surplus), or you want to join with other sources |
| Formula | a severity and a layer | expected loss, variance | a quick look at one layer under a Pareto or other severity, with no annual terms, reinstatements or commissions |

A compound distribution has no event times, so the grid refuses reinstatements
pro rata as to time:

```python
from prospicio.distributions import Grid, Pareto, Poisson
from prospicio.reinsurance import Layer, Tower

severity, _ = Grid.rounding(Pareto(1.0, 1.8), 0.1, 201)
timed = Layer("5x5", 5.0, 5.0, premium=1.0, reinstatement_rates=[1.0], pro_rata_time=True)
try:
    Tower([timed]).on_grid(Poisson(4.0), severity, 2**12)
except ValueError as e:
    print("refused:", str(e).split(";")[0])
annual_cap = Layer("5x5", 5.0, 5.0, aggregate_limit=10.0)
print("net grid with an annual limit:", Tower([annual_cap]).on_grid(Poisson(4.0), severity, 2**12).net)
```

```text
refused: invalid data: layer "5x5" has reinstatements pro rata as to time
net grid with an annual limit: None
```

Pitfall: a layer with annual terms (an aggregate limit or reinstatements) has
no `net` grid. `net` exists when no layer has annual terms, or when the last
stage is one aggregate cover such as a stop loss. Otherwise make the net
yourself or use simulation (`simulate_events`, `RiskProfile.simulate`,
`Tower.apply`).

## 1. Primary ratemaking and the overall rate indication

**What it is.** The indicated rate change that makes premium equal to expected
losses and expenses. The **loss ratio method** compares the projected loss
ratio with the permissible loss ratio. The **pure premium method** calculates a
rate from the losses per exposure unit. Both need:

- premium at the **current rate level** (the **on-level factor**, by the
  parallelogram method)
- losses **developed** to ultimate and **trended** to the policy period
- a **credibility** weight against a complement

**Data you usually have.** Written and earned premium by year, a history of
rate changes, losses with development factors, trend selections, and expense
and profit provisions.

**When to use it.** For the average rate level of a book that has a rating
plan. Not for one treaty, which has no rate history to put on-level.

**The calls.**

```python
from prospicio.credibility import LimitedFluctuation
from prospicio.pricing import LossRatioExperience, Period, Provisions, RateHistory, credibility_weighted

history = RateHistory([2022.5, 2023.5], [0.05, 0.03])   # +5% from mid 2022, +3% from mid 2023
years = [2021.0, 2022.0, 2023.0]
on_level = [history.rate_level(Period.calendar(y, y + 1.0), term=1.0).on_level_factor for y in years]
experience = LossRatioExperience(
    earned_premium=[100.0, 110.0, 120.0],
    losses=[62.0, 70.0, 66.0],
    on_level_factor=on_level,
    development=[1.00, 1.02, 1.10],
    loss_trend=[1.08, 1.05, 1.025],
    lae_factor=1.08,     # 1 plus the loss adjustment expense not already in the losses
)
provisions = Provisions(variable=0.25, profit=0.05)
indicated = provisions.loss_ratio_change(experience.loss_ratio)
lf = LimitedFluctuation(0.90, 0.05)
z = lf.credibility(900, lf.frequency_standard())    # 900 expected claims
selected = credibility_weighted(indicated, z, 0.02)  # complement: +2%
print("on-level factors", [round(f, 4) for f in on_level])
print(f"projected loss ratio {experience.loss_ratio:.1%}, permissible {provisions.permissible_loss_ratio():.1%}")
print(f"indicated change {indicated:+.1%}; credibility {z:.2f}; selected {selected:+.1%}")
```

```text
on-level factors [1.0815, 1.0748, 1.0323]
projected loss ratio 66.7%, permissible 70.0%
indicated change -4.7%; credibility 0.91; selected -4.1%
```

The 2021 year must catch up on the full 8.15% of cumulative rate increase, and
2023 only on a small part, so the factors decrease for recent years.
Credibility moves the indication toward the complement.

**Pitfalls.**

- `experience.loss_ratio`, `provisions.variable` and
  `RateLevel.on_level_factor` are properties: no parentheses.
- Dates are year fractions (`2022.5` is mid 2022). `year_fraction` converts a
  calendar date.
- The calculation assumes uniform writing and earning. For a policy term other
  than a year, or a risks-attaching treaty, use `Period.policy`, not
  `Period.calendar`.
- These classes are for primary ratemaking, with no layer or treaty year. For
  individual large losses use `LargeLossExperience(ids, accident, amount,
  threshold=, valuation=, fx=)` and its `.prepare(rate=, to=, development=)`
  (section 2).

**More detail.** [Experience and exposure rating](experience-and-exposure-rating.md)
(pure premium method, credibility), `docs/design/ratemaking.md` (definitions).

## 2. Per-risk property excess of loss

A **per-risk** cover pays on each loss to one risk (a building, a plant), above
an attachment and up to a limit. Use two methods together:

- **Exposure rating** prices the layer from the **risk profile** of the cedant
  (the sum insured in each band). An exposure curve gives the share of the
  expected loss of a risk below a given fraction of its maximum possible loss
  (MPL). `RiskProfile` takes the MPL as the sum insured. It needs no loss
  history.
- **Experience rating** fits the large losses of the cedant and applies the
  fitted severity to the layer. It shows what occurred, but a high layer may
  have had only two losses in five years.

**Data you usually have.** A table of bands (sum insured, number of risks,
premium) and a list of large losses with dates, possibly with development.

**Which to use.**

| You have | Use | Because |
|---|---|---|
| Profile and no history, or a new account | Exposure | There is nothing else |
| Profile and a thin history | Exposure, checked against experience | Experience on a high layer is mostly noise |
| Long, clean history on a low layer | Experience, with exposure as a check | The history is the cedant's own, and there is sufficient layer data |
| Both, and they do not agree | Blend with credibility, and find why they do not agree | A large difference shows that the curve, the profile or the loss data is wrong |

The example rates the layer 4 xs 6 (the part of each loss between 6 and 10).

```python
import pandas as pd
from great_tables import GT

from prospicio.distributions import Pareto, Poisson
from prospicio.pricing import CollectiveModel, Mbbefd, RiskProfile
from prospicio.reinsurance import Layer


def profile(c):
    return RiskProfile([1.0, 3.0, 6.0, 12.0], [400, 150, 60, 20], Mbbefd.swiss_re(c),
                       premiums=[8.0, 6.0, 5.0, 4.0], loss_ratio=0.55)


exposure = {c: profile(c).expected_layer_loss(4.0, 6.0) for c in (2.0, 3.0, 4.0)}
exposure_el = exposure[3.0]

large = [1.2, 1.5, 1.1, 2.4, 1.8, 1.3, 3.9, 1.6, 1.05, 2.2, 5.5, 1.4, 1.9, 1.15, 7.8, 1.25, 2.9, 1.7, 1.35, 4.4]
YEARS = 5
fit = Pareto.fit(large, 1.0)                                   # losses above 1, alpha by maximum likelihood
experience_el = CollectiveModel(Poisson(len(large) / YEARS), fit).layer_mean(4.0, 6.0)
burning = Layer("4x6", 4.0, 6.0).ceded(large) / YEARS         # what the layer would have paid
print(f"fitted alpha {fit.alpha:.2f}")

sources = pd.DataFrame({
    "source": ["exposure, Swiss Re c=2", "exposure, Swiss Re c=3", "exposure, Swiss Re c=4",
               "burning cost, 5 years", "fitted Pareto"],
    "loss": [exposure[2.0], exposure[3.0], exposure[4.0], burning, experience_el],
})
(GT(sources, rowname_col="source")
 .tab_header("Expected loss a year of the layer 4 xs 6", "Exposure rating at three curves, then experience rating")
 .cols_label(loss="expected loss")
 .fmt_number("loss", decimals=3))
```

```text
fitted alpha 1.48
                        expected loss
exposure, Swiss Re c=2          0.482
exposure, Swiss Re c=3          0.347
exposure, Swiss Re c=4          0.224
burning cost, 5 years           0.360
fitted Pareto                   0.774
```

The curve alone moves the expected loss by more than a factor of two, so test
it. The fitted Pareto is more than twice the exposure rating. The burning cost
is near it: the layer paid on one loss above 6 in five years. A layer with this
shape needs many more than 20 claims for full credibility. The next block (it
uses the one above) weighs the two:

```python
from prospicio.credibility import LimitedFluctuation

severity_cv = fit.layer_variance(4.0, 6.0) ** 0.5 / fit.layer(4.0, 6.0)    # spread of the payment per large loss
rows = []
for p, k in ((0.90, 0.10), (0.90, 0.05)):                      # two full-credibility standards
    lf = LimitedFluctuation(p, k)
    z = lf.credibility(len(large), lf.pure_premium_standard(severity_cv))
    rows.append((f"({p:.2f}, {k:.2f})", z, z * experience_el + (1 - z) * exposure_el))
standards = pd.DataFrame(rows, columns=["standard", "z", "blend"])
(GT(standards, rowname_col="standard")
 .tab_header("Experience weighted against exposure", "Limited fluctuation standards (probability, tolerance)")
 .cols_label(z="credibility", blend="blended loss")
 .fmt_number("z", decimals=2)
 .fmt_number("blend", decimals=3))
```

```text
              credibility  blended loss
(0.90, 0.10)         0.06         0.374
(0.90, 0.05)         0.03         0.361
```

A stricter standard (0.05 instead of 0.10) halves the weight. A Bühlmann-type
weight is also common. The blend is an illustration, not a rule.

**Pitfalls.**

- The curve is the model. `Mbbefd.swiss_re(c)` is a one-parameter family:
  c = 1.5, 2, 3, 4 are the Swiss Re curves and 5 is Lloyd's. Select the curve
  for each occupancy class and show the sensitivity, as above.
- Exposure curves use the fraction of the MPL. For large industrial risks the
  MPL can be much less than the sum insured, so the sum insured spreads each
  loss over too large a base. Put the MPL in `sums_insured` when you have it.
  A first-loss scale from a primary rating manual is a different curve on a
  different base: do not mix them.
- Prepare large losses before you fit. `LargeLossExperience` does each step:
  - trend by date at a rate (or a factor for each loss)
  - development of known losses by a `DevelopmentPattern` (or a factor for
    each loss)
  - an `fx` rate to one currency
  - an `IndexClause` given to `burning_cost(index=, ratios=)`
- After `prepare` develops the known losses, give
  `burning_cost(..., development=)` only the count IBNR (a factor for each year
  or a `DevelopmentPattern`). An excess factor that includes IBNER counts it
  twice.
- With a reporting threshold, the method omits losses below `complete_above()`
  (the threshold times the largest trend factor), so each year is complete on
  the same basis. `PreparedLosses.as_if(historical_towers, years, proposed)`
  shows the history under the program of each year next to the proposed one.
- `Pareto.fit` takes a threshold, optional reporting thresholds for each loss,
  flags for losses capped by a policy limit (`censored`) and a `truncation`.
  `Pareto.fit_with_standard_error` also gives the standard error of alpha. For
  the lognormal, gamma, Weibull and Burr, `fit_severity(family, losses,
  reporting_thresholds=, censored=)` fits the same data by maximum likelihood.
  `goodness_of_fit` compares the fit with the product-limit estimate.

**More detail.** [Experience and exposure rating](experience-and-exposure-rating.md),
[Severity and frequency](severity-and-frequency.md).

## 3. Casualty excess of loss

Casualty covers (general liability, motor liability, professional lines) attach
high and pay for many years. The data are few large losses, policy limits, and
slow development. Two ideas:

- The **increased limit factor** (ILF) is the ratio of the expected loss at one
  policy limit to the expected loss at a basic limit.
- **Pareto extrapolation**: if the severity above a threshold is Pareto with a
  given alpha, one priced layer gives the price of any layer.

**Data you usually have.** A limits profile (premium written by policy limit),
the expected loss ratio of the cedant, a severity curve (from a rating bureau
or your own) and a short list of large losses that still develop.

| You have | Use | Call |
|---|---|---|
| A limits profile and an expected loss ratio | Exposure rating: the expected loss of each limit, split into the layer by ILFs | `ilf` in a loop (below) |
| A severity curve from a primary rating bureau or your own data | ILF between limits | `ilf(severity, limit, basic_limit)` |
| The expected loss of one layer, and a view on alpha | Move up or down the tower | `pareto_extrapolation(from, to, alpha)` |
| Expected losses for some layers | Match them with a frequency and a severity | `match_tower`, `fit_references` |
| Excess frequencies at several thresholds | Alpha between them | `alpha_between_frequencies`, `alpha_between_layers` |
| A deductible to evaluate | Loss elimination ratio | `loss_elimination_ratio` |

The example exposure-rates the layer 1.5 xs 0.5 from a profile of three policy
limits. Then it moves between layers.

```python
from prospicio.distributions import Lognormal
from prospicio.pricing import ilf, match_tower, pareto_extrapolation

policy_severity = Lognormal.from_mean_cv(0.05, 4.0)      # losses to a casualty account
print(f"ILF at a limit of 1 against a basic limit of 0.1: {ilf(policy_severity, 1.0, 0.1):.2f}")
```

```text
ILF at a limit of 1 against a basic limit of 0.1: 1.68
```

```python
import pandas as pd
from great_tables import GT

LIMIT, ATTACHMENT, ELR, BASIC = 1.5, 0.5, 0.65, 0.1
PREMIUMS = [30.0, 20.0, 10.0]                                               # premium written at each limit
rows = []
for policy_limit, premium in zip([0.5, 1.0, 2.0], PREMIUMS):
    low, high = min(policy_limit, ATTACHMENT), min(policy_limit, ATTACHMENT + LIMIT)
    in_layer = (ilf(policy_severity, high, BASIC) - ilf(policy_severity, low, BASIC)) / ilf(policy_severity, policy_limit, BASIC)
    rows.append((f"{policy_limit}", premium, in_layer, premium * ELR * in_layer))
layer_el = sum(r[3] for r in rows)
rows.append(("all limits", sum(PREMIUMS), layer_el / (sum(PREMIUMS) * ELR), layer_el))
by_limit = pd.DataFrame(rows, columns=["limit", "premium", "share", "loss"])
(GT(by_limit, rowname_col="limit")
 .tab_header("Exposure rating of the layer 1.5 xs 0.5", "By policy limit; expected loss ratio 0.65")
 .cols_label(share="share of loss in layer", loss="layer loss")
 .fmt_number("premium", decimals=1)
 .fmt_number(["share", "loss"], decimals=3))
```

```text
            premium  share of loss in layer  layer loss
0.5            30.0                   0.000       0.000
1.0            20.0                   0.084       1.098
2.0            10.0                   0.129       0.837
all limits     60.0                   0.050       1.934
```

The policies with a limit of 0.5 put nothing in the layer. The layer loss is the
sum of premium times loss ratio times share. Compare it with a burning cost on
developed, trended large losses (`PreparedLosses.burning_cost`, section 2)
before you use either.

```python
for alpha in (1.5, 2.5):
    print(f"alpha {alpha}: layer 2x2 costs {pareto_extrapolation((1.0, 1.0), (2.0, 2.0), alpha):.1%} of layer 1x1")

tower = match_tower([1.0, 2.0, 5.0], [0.30, 0.20, 0.10])     # three layers; the last is unlimited
print(f"matched tower: {tower.frequency:.2f} losses a year above 1, alphas {[round(a, 2) for a in tower.severity.alpha]}")
print(f"2 xs 3 from the matched tower: {tower.layer_loss(2.0, 3.0):.4f}")
```

```text
alpha 1.5: layer 2x2 costs 70.7% of layer 1x1
alpha 2.5: layer 2x2 costs 35.4% of layer 1x1
matched tower: 0.58 losses a year above 1, alphas [1.86, 1.73, 2.39, 2.2]
2 xs 3 from the matched tower: 0.0890
```

**Pitfalls.**

- Prepare losses as in section 2 (`LargeLossExperience.prepare`).
  `ExcessPattern` gives the development of the excess layer by blending layer
  factors from the ground-up triangle and a severity with the factors of the
  layer ([`reserving-excess-layers.md`](reserving-excess-layers.md)).
- Decide how allocated loss adjustment expense (ALAE) goes into the layer: pro
  rata with the loss, or added to it within the limit. Treat the losses and the
  severity curve the same way.
- Pareto extrapolation assumes one alpha across the layers. In casualty this is
  usually false, so `match_tower` fits a piecewise Pareto with one alpha for
  each layer.
- `match_tower` and `fit_references` refuse inputs that cannot come from a
  decreasing, convex excess loss function (for example, rates on line that
  increase up the tower). A match gives its references exactly, so read
  `tower.diagnostics()`. Its `warnings` show an alpha below 1, or adjacent
  alphas more than twice apart. `fit_references(..., alpha_bounds=(lo, hi))`
  keeps the fitted alphas in a range that you set.
- `compound_moments` gives an error when a severity does not have three finite
  moments (a Pareto with alpha below 3), so its normal or gamma `approximate`
  is not available. Use the grid or simulation for heavy tails.
- Agree with the wording how these change the losses in the layer: ECO and XPL
  (extra-contractual obligations and losses in excess of policy limits), clash
  between policies on one occurrence, and claims-made or occurrence triggers.
- The prices in this section are one-period prices. For a long tail, price over
  the payout with a `PaymentPattern` (for the next accident year,
  `PaymentPattern.from_development(ages_in_years, paid_cdf)`) and
  `price_over_payout(price, pattern, rate=, cost_of_capital=)`. Discounting
  decreases the loss (`discount_factor` below 1). Capital held until the last
  payment increases the capital charge (`capital_years` above 1). The second
  effect is usually larger, so the payout premium is often above the
  one-period premium.

**More detail.** [Experience and exposure rating](experience-and-exposure-rating.md),
[Severity and frequency](severity-and-frequency.md).

## 4. Catastrophe excess of loss

A **catastrophe** cover pays on the total of an event (a hurricane, an
earthquake) across many risks. The cedant or broker gives the output of a
vendor model, and you do not model the hazard again. You get one of these:

- a **year loss table**: a simulated list of years, each with its events and
  loss amounts
- **return period points** on the occurrence exceedance probability (OEP)
  curve: the event loss that occurs once in 100 years, and so on

**Data you usually have.** The vendor's year loss table, or its OEP and AEP
(aggregate exceedance) points for the portfolio, the layers and reinstatement
terms of the program, and some historical events.

| You have | Use | Call |
|---|---|---|
| An event loss table (rate, mean, standard deviations and exposure for each event) | Simulate years from it, or read its OEP and PML | `EventLossTable(rate, mean, sd_independent, sd_correlated, exposure)`. `.simulate(n_years, seed)` gives `(EventSet, event ids)`. `.oep(x)`, `.occurrence_pml(p)` |
| A year loss table | Load it as events, apply the terms | `EventSet.from_years(years, sums_insured, seed, times)`, or `EventSet.from_long(years, losses, n_years)` for one row for each loss (years are indices from 0, and `n_years` is the full simulation count, including the years with no loss). Then `Tower.apply` |
| Return period points | Fit a frequency and a severity to them, then use the grid or simulation | `fit_pml_curve(return_periods, amounts, tail_alpha, truncation)` |

```python
import pandas as pd
from great_tables import GT

from prospicio.pricing import fit_pml_curve

RETURN_PERIODS, AMOUNTS = [10, 20, 50, 100, 250], [30.0, 55.0, 100.0, 140.0, 200.0]
oep = fit_pml_curve(RETURN_PERIODS, AMOUNTS, tail_alpha=2.0, truncation=500.0)
fitted = pd.DataFrame({
    "return period": [str(rp) for rp in RETURN_PERIODS],
    "amount": AMOUNTS,
    "fitted": [oep.excess_frequency(a) for a in AMOUNTS],
    "target": [1 / rp for rp in RETURN_PERIODS],
})
(GT(fitted, rowname_col="return period")
 .tab_header("OEP points refitted", "Events a year above each amount")
 .cols_label(amount="amount", fitted="fitted frequency", target="1 / return period")
 .fmt_number("amount", decimals=1)
 .fmt_number(["fitted", "target"], decimals=4))
```

```text
     amount  fitted frequency  1 / return period
10     30.0            0.1000             0.1000
20     55.0            0.0500             0.0500
50    100.0            0.0200             0.0200
100   140.0            0.0100             0.0100
250   200.0            0.0040             0.0040
```

The fit reproduces each point. It reads a return period of RP as an exceedance
frequency of 1/RP a year. A vendor OEP probability p is different: convert it
with -ln(1 - p) first.

```python
import math

from prospicio.aggregate import simulate_events
from prospicio.distributions import Grid, Poisson
from prospicio.reinsurance import Layer, Tower

print(f"an OEP probability of 0.10 is {-math.log(1 - 0.10):.4f} events a year")
frequency = Poisson(oep.frequency)
grid, _ = Grid.rounding(oep.severity, 1.0, 501)
cat_layers = [Layer("50x50", 50.0, 50.0, reinstatements=1), Layer("100x100", 100.0, 100.0, reinstatements=1)]
exact = Tower(cat_layers).on_grid(frequency, grid, 2**11)
sim = Tower(cat_layers).apply(simulate_events(frequency, oep.severity, 20_000, 1))
ceded = pd.DataFrame(
    [["exact on the grid", *[g.mean() for g in exact.ceded]],
     ["simulated events", *[sim.marginal(("ceded", l.name)).mean() for l in cat_layers]]],
    columns=["method", "50x50", "100x100"])
(GT(ceded, rowname_col="method")
 .tab_header("Expected ceded loss a year", "Two catastrophe layers with one reinstatement")
 .fmt_number(["50x50", "100x100"], decimals=3))
```

```text
an OEP probability of 0.10 is 0.1054 events a year
                   50x50  100x100
exact on the grid  1.672    0.948
simulated events   1.684    0.947
```

The two agree within simulation error. With event times
(`EventSet.with_uniform_times`, `with_seasonal_times`), the simulation also
applies reinstatements pro rata as to time.

**Pitfalls.**

- A fitted OEP curve describes events above its lowest point only. Here it
  models nothing below 30.
- Return period points give the loss of one event, not the total of the year,
  so the number of events matters for a cover with an annual limit or paid
  reinstatements. When the vendor's events cluster, a Poisson count fitted to
  OEP points gives too few years with several events. Use the year loss table
  (its AEP) when you have it, and the OEP fit only when you do not.
- In an `EventLossTable`, the secondary uncertainty of each event is a beta on
  [0, exposure] with the mean and standard deviation of the row.
- `EventSet.grouped(window_hours)` groups losses into occurrences under an hours
  clause (a year is 8,760 hours). It needs each loss's time as a fraction of a
  year (`from_years(..., times=)`, `with_uniform_times` or
  `with_seasonal_times`).
- `EventSet.merge` adds a non-cat set year by year.
- For clash across two lines (one event hits both), use `Clash` or
  `fft_bivariate` on a joint grid (`prospicio.aggregate`), with a copula between
  the lines in an event if needed.

**More detail.** [Aggregate distributions](aggregate-distributions.md) (grid and
simulation), [What a reinsurance price is](reinsurance-price.md) (the price of a
layer).

## 5. Quota share, surplus and other proportional treaties

A **quota share** cedes a constant share of the premium and losses of each
policy. A **surplus** treaty cedes, for each risk, the part of the sum insured
above the retention line of the cedant, up to a given number of lines, so the
ceded share differs by risk. The reinsurer pays the cedant a **ceding
commission**, and the commission is the price.

**What sets the price.** The loss ratio of the full book, not only the large
losses: you need its annual distribution and the terms that move the
commission with it:

- a **sliding scale**: the commission decreases as the loss ratio increases
- a **profit commission**: the reinsurer gives a share of its profit back to the
  cedant

**Data you usually have.** The subject premium (the premium that the treaty
shares), the loss ratios of the cedant by underwriting year with development,
its expense ratio, and the commission and brokerage terms offered. For a
surplus, also the risk profile by sum insured. The expected loss ratio is the
on-level, developed and trended one, as in section 1. The spread around it
comes from the claim count and severity, or from the variation of those loss
ratios from year to year.

| Situation | Method | Call |
|---|---|---|
| Quota share price | Technical commission: 1 minus the expected ceded loss ratio, expenses, brokerage and margin | `Tower.on_grid` for the ceded grid (or `Tower.on_annual_grid` on a loss-ratio grid), `price(grid, Distortion.tvar(0.99), cost_of_capital=r)` for the margin |
| Sliding scale or profit commission | Expected commission over the loss ratio distribution | `with_sliding_scale`, `with_profit_commission`, `TowerGrids.expected_ceding_commission` or `AnnualGrids.expected_ceding_commission` |
| Surplus | Simulated events with sums insured | `Layer.surplus`, `RiskProfile.simulate`, `Tower.apply` |
| Surplus inuring to a per-risk excess of loss | Stages in order | `Tower.inuring([[surplus], [xl]])` |

This example puts a quota share and an aggregate stop loss on one book, which
section 6 reuses: 400 claims a year (a spread wider than Poisson), 100 of
premium, an expected loss ratio of 60%.

```python
from prospicio.distributions import Grid, Lognormal, NegativeBinomial
from prospicio.pricing import price
from prospicio.reinsurance import Layer, Tower
from prospicio.risk import Distortion

SUBJECT = 100.0
claims = NegativeBinomial.from_mean_variance(400.0, 600.0)
claim_grid, _ = Grid.rounding(Lognormal.from_mean_cv(0.15, 2.0), 0.01, 2000)

qs = (Layer.quota_share("QS 30%", 0.30).with_premium_rate(1.0, SUBJECT)
      .with_sliding_scale([(0.35, 0.55), (0.30, 0.65), (0.25, 0.75)]))
stop = Layer.stop_loss("SL 20 xs 70", 20.0, 70.0).with_deposit_premium(2.0)
book = Tower([qs, stop]).on_grid(claims, claim_grid, 2**14)
qs_loss, sl_loss = (g.mean() for g in book.ceded)
qs_premium, qs_commission = book.expected_premium[0], book.expected_ceding_commission[0]
print(f"book loss ratio {book.gross.mean() / SUBJECT:.1%}, spread (sd) {book.gross.variance() ** 0.5 / SUBJECT:.1%} of premium")
print(f"QS: ceded premium {qs_premium:.1f}, ceded loss {qs_loss:.2f}, "
      f"expected commission {qs_commission:.2f} ({qs_commission / qs_premium:.1%})")
annual = Tower([qs]).on_annual_grid(book.gross)      # the same terms on the year's total alone, no claim count
print(f"QS on the annual total grid: expected commission {annual.expected_ceding_commission[0]:.2f}")

tvar99 = Distortion.tvar(0.99)
EXPENSES, BROKERAGE = 0.02, 0.015
# Flat commission: the reinsurer pays the ceded loss plus a fixed commission, so only the loss is at risk.
flat = price(book.ceded[0], tvar99, cost_of_capital=0.10)    # assets at 99% TVaR, 10% on capital
technical = 1 - qs_loss / qs_premium - EXPENSES - BROKERAGE - flat.margin / qs_premium
print(f"flat: tail above the mean {flat.assets - qs_loss:.2f}, capital {flat.capital:.2f}; technical commission {technical:.1%}")
# Sliding scale: the reinsurer pays the ceded loss plus a commission that falls as the loss rises.
outgo, _ = book.ceded[0].map(lambda loss: loss + qs.ceding_commission_for(loss))
scale = price(outgo, tvar99, cost_of_capital=0.10)
available = qs_premium * (1 - EXPENSES - BROKERAGE) - outgo.mean()
print(f"scale: tail above the mean {scale.assets - outgo.mean():.2f}, capital {scale.capital:.2f}; "
      f"margin needed {scale.margin:.2f}, margin in the terms {available:.2f}")
```

```text
book loss ratio 60.0%, spread (sd) 7.0% of premium
QS: ceded premium 30.0, ceded loss 18.00, expected commission 9.64 (32.1%)
QS on the annual total grid: expected commission 9.64
flat: tail above the mean 6.71, capital 6.10; technical commission 34.5%
scale: tail above the mean 4.57, capital 4.16; margin needed 0.42, margin in the terms 1.32
```

The third line puts the quota share on the total of the year alone, as when a
cedant gives a loss ratio distribution with no claim count.
`Tower.on_annual_grid` treats that total as one loss. The terms depend only on
the total, so the commission is the same. `price` sets the assets at the 99%
TVaR of each grid and charges 10% on the capital (the assets above the
premium).

The two contracts price different risks, so do not compare the scale's
expected commission (32.1%) with the flat technical commission (34.5%):

- With a flat commission, only the ceded loss is at risk.
- With the scale, the outgo of the reinsurer is the loss plus a commission that
  decreases as the loss increases. The scale gives part of the loss ratio
  spread back to the cedant, so the tail above the mean and the capital fall.

The scale leaves a margin of 1.32 against the 0.42 needed: its terms price
above the target.

A surplus needs events that have sums insured. `RiskProfile.simulate` makes
them from the profile of section 2. The exposure-rated `expected_surplus_loss`
checks the simulation:

```python
from prospicio.pricing import Mbbefd, RiskProfile
from prospicio.reinsurance import Layer, Tower

profile = RiskProfile([1.0, 3.0, 6.0, 12.0], [400, 150, 60, 20], Mbbefd.swiss_re(3.0),
                      premiums=[8.0, 6.0, 5.0, 4.0], loss_ratio=0.55)
surplus = Layer.surplus("S 9 lines", 1.0, 9.0)                   # retention 1, up to 9 lines
for size in (5.0, 20.0):
    print(f"a risk of {size:.0f} cedes {surplus.ceded_with_sums_insured([1.0], [size]):.0%} of its loss")
simulated = Tower([surplus]).apply(profile.simulate(20_000, 1)).marginal(("ceded", "S 9 lines")).mean()
print(f"ceded loss a year: simulated {simulated:.2f}, exposure rated {profile.expected_surplus_loss(1.0, 9.0):.2f}, "
      f"of a total {profile.expected_loss():.2f}")
```

```text
a risk of 5 cedes 80% of its loss
a risk of 20 cedes 45% of its loss
ceded loss a year: simulated 6.13, exposure rated 6.14, of a total 12.65
```

The cession goes through the same commission calculation as the quota share. A
per-risk excess of loss on the retained part is a second stage
(`Tower.inuring([[surplus], [xl]])`).

**Pitfalls.**

- A quota share depends on the loss ratio of the full book. Attritional
  losses held at their mean give too little capital: give them a spread.
- Set the premium (`with_premium_rate`, `with_deposit_premium` or
  `with_rate_on_line`) before `with_sliding_scale` or `with_profit_commission`,
  which need it for the loss ratio. A sliding scale cannot combine with swing
  rating or paid reinstatements.
- For a quota share or stop loss on an annual loss ratio grid (no claim count),
  use `Tower.on_annual_grid(grid)`: `.ceded`, `expected_ceding_commission` and
  `expected_profit_commission` are exact. Per-occurrence layers with a severity
  need `on_grid`.
- `price(losses, assets, cost_of_capital=r)` takes a grid, simulated losses
  (`Sampled`) or a `PredictiveDistribution`. `assets` is a distortion risk
  measure (here 99% TVaR). The result has `.margin`, `.capital` and `.assets`.

**More detail.** [What a reinsurance price is](reinsurance-price.md), section 4.

## 6. Aggregate stop loss, adverse development and loss portfolio covers

- A **stop loss** (aggregate excess of loss) pays when the total of the year is
  more than a retention, up to a limit.
- An **adverse development cover** (ADC) does the same on reserves. It pays
  when the reserve for past years becomes worse than a retention.
- A **loss portfolio transfer** (LPT) moves a block of reserves to the
  reinsurer for a premium.

All three pay on a total, so the work is to find the distribution of that
total.

| The total comes from | Method | Call |
|---|---|---|
| A claim count and a severity | Exact aggregate on the grid | `Tower.on_grid`, or `fft` and `fft_auto` for the distribution alone |
| Simulated events | Apply to each year | `Tower.apply` |
| A reserve bootstrap | Apply the cover to the simulated reserve | `Tower.apply_aggregate(boot.reserves)` |

**Data you usually have.** For a stop loss, the same data as for a quota share
(section 5). For an ADC or LPT, the reserve triangles of the covered years, the
booked reserves, and the payment pattern.

This block uses `book` from section 5, whose stop loss pays on the gross total.
Its deposit premium of 2.0 there is a placeholder. The block prices it as
section 5 priced the flat quota share.

```python
sl_grid = book.ceded[1]
sl_price = price(sl_grid, Distortion.tvar(0.99), cost_of_capital=0.10)
print(f"stop loss 20 xs 70: expected loss {sl_loss:.3f}, chance of any recovery {1 - sl_grid.cdf(0.0):.1%}")
print(f"at 99% TVaR assets and 10% on capital: premium {sl_price.premium:.2f} "
      f"(rate on line {sl_price.premium / 20.0:.1%}), capital {sl_price.capital:.2f}")
```

```text
stop loss 20 xs 70: expected loss 0.339, chance of any recovery 8.2%
at 99% TVaR assets and 10% on capital: premium 1.42 (rate on line 7.1%), capital 10.79
```

A stop loss pays rarely, and then a large amount, so most of its premium is the
margin on its capital. The capital standard and the cost of capital set the
price: state both with the quote. An ADC is the same call on a different total
(section 7).

**Pitfalls.**

- The retention and limit apply to the total that the cover sees. In a `Tower`
  with stages, that total is net of earlier stages (`Tower.inuring`).
- The total must have the correct tail. A normal or gamma approximation of the
  total (`compound_moments(...).approximate`) gives a heavy tail that is too
  small. The grid does not.
- For an ADC or LPT, the uncertainty is in the reserve, so use a bootstrap or
  a Bayesian model, not a point estimate. A bootstrap has only parameter and
  process error under its model's assumptions (here the chain ladder, with no
  tail unless you give one). It has no calendar-year inflation shock and no
  change in claims handling, which usually cause an ADC loss. Treat its result
  as a minimum, and show the layer under a second method or tail (section 7).
- Time value matters for an ADC or LPT, which have a long tail. The prices
  here are one-period prices: use `price_over_payout(price(...), pattern,
  rate=, cost_of_capital=)` and read its effects as in section 3.
- The pattern for a reserve cover is the calendar-year run-off of the reserve
  from the valuation date. It is not the development of one accident year. Make
  it from the paid chain ladder (the `cl` of section 7) as follows:
  1. Origin i, with its latest age at index k_i, pays
     `ult_i / cdf[j] - ult_i / cdf[j - 1]` at calendar time j - k_i, for each
     later age index j.
  2. Add these payments over the origins by calendar time.
  3. Give the times and the shares of the total to
     `PaymentPattern(times, shares)`.
- A layer that attaches above the mean reserve pays later, so the duration of
  that pattern is a lower bound for the layer. For a cover on a future accident
  year, use `PaymentPattern.from_development(ages_in_years, paid_cdf)` (with
  `.shifted`).
- `Layer.with_rate_on_line` needs a finite limit for each loss, so it refuses a
  stop loss. Set its premium with `with_deposit_premium` or
  `with_premium_rate`.

**More detail.** [Aggregate distributions](aggregate-distributions.md),
[Reserving methods](reserving-methods.md).

## 7. Reserving a book and its uncertainty

Reserving turns a **triangle** (losses by origin year and development age)
into an estimate of the amount still to be paid.

**Data you usually have.** Paid and incurred triangles by origin year, premium
or another exposure by origin year, and a prior loss ratio from pricing.

| Question | Method | Call |
|---|---|---|
| The best estimate of the reserve | Chain ladder. Bornhuetter-Ferguson, Cape Cod or expected loss when a year is immature. Clark for a growth curve | `ChainLadder`, `BornhuetterFerguson`, `CapeCod`, `ExpectedLoss`, `ClarkLdf` |
| Its uncertainty to ultimate (the **lifetime** view) | Mack's formula, or a bootstrap for a full distribution | `Mack`, `OdpBootstrap`, `MackBootstrap` |
| Its uncertainty over the next calendar year (the **one-year** view, for solvency capital) | Merz-Wuthrich from Mack, or re-reserving in a bootstrap | `Mack().fit(...).claims_development_result()`, `OdpBootstrap.one_year`, `MackBootstrap.one_year` |
| A distribution that includes prior information | Bayesian chain ladder | `BayesOdp`, `Meyers` |

The chain ladder reads development only from the triangle. For the latest
origin years it multiplies a small number by a large factor.
Bornhuetter-Ferguson mixes the chain ladder with a prior expected loss ratio
and is more stable there. Use it when you trust the prior and have an exposure
measure (premium) in the triangle.

```python
import csv
from pathlib import Path

from prospicio.reinsurance import Layer, Tower
from prospicio.reserving import BornhuetterFerguson, ChainLadder, Mack, OdpBootstrap, Triangle

rows = list(csv.DictReader(l for l in open(Path("../../validation/data/genins.csv")) if not l.startswith("#")))
tri = Triangle.from_long([int(r["origin"]) for r in rows], [int(r["development"]) for r in rows],
                         {"paid": [float(r["value"]) for r in rows], "premium": [8.0e6] * len(rows)})
cl = ChainLadder().fit(tri, "paid")
mack = Mack().fit(tri, "paid")
cdr = mack.claims_development_result()
print(f"chain ladder reserve {cl.total_reserve / 1e6:.2f}m; Mack standard error {mack.total_standard_error / 1e6:.2f}m "
      f"(next year only: {cdr.total_one_year_standard_error / 1e6:.2f}m)")
bf = BornhuetterFerguson(apriori=0.65).fit(tri, "paid", "premium")
print(f"Bornhuetter-Ferguson reserve {bf.total_reserve / 1e6:.2f}m")
```

```text
chain ladder reserve 18.68m; Mack standard error 2.44m (next year only: 1.77m)
Bornhuetter-Ferguson reserve 17.53m
```

Bornhuetter-Ferguson is more stable in the latest years:

```python
import pandas as pd
from great_tables import GT

latest = pd.DataFrame({
    "origin": ["latest - 2", "latest - 1", "latest"],
    "cl": [u / 1e6 for u in cl.ultimate[-3:]],
    "bf": [u / 1e6 for u in bf.ultimate[-3:]],
})
(GT(latest, rowname_col="origin")
 .tab_header("Ultimate of the latest three origin years", "Millions")
 .cols_label(cl="chain ladder", bf="Bornhuetter-Ferguson")
 .fmt_number(["cl", "bf"], decimals=2))
```

```text
            chain ladder  Bornhuetter-Ferguson
latest - 2          6.78                  5.87
latest - 1          5.64                  5.31
latest              4.97                  5.18
```

A bootstrap is a minimum for the uncertainty of an ADC. The triangle cannot
show whether a 5% tail is right, and this tail makes the expected recovery
almost three times larger. The cost of the layer depends on the model, so show
it under more than one:

```python
adc_layer = Tower([Layer.stop_loss("ADC", 5e6, 20e6)])
rows = []
for tail in (1.0, 1.05):                               # no tail, then a 5% tail past the oldest age
    boot = OdpBootstrap(n_sims=2000, seed=1, tail=tail).fit(tri, "paid")
    total = boot.reserves.total()
    recovery = adc_layer.apply_aggregate(boot.reserves).marginal(("ceded", "ADC")).mean()
    rows.append((f"{tail:.2f}", total.mean() / 1e6, total.quantile(0.995) / 1e6, recovery / 1e6))
boots = pd.DataFrame(rows, columns=["tail", "mean", "q995", "recovery"])
(GT(boots, rowname_col="tail")
 .tab_header("ODP bootstrap of the total reserve", "Millions; the ADC pays 5 xs 20")
 .cols_label(mean="mean", q995="99.5%", recovery="ADC expected recovery")
 .fmt_number(["mean", "q995", "recovery"], decimals=2))
```

```text
       mean  99.5%  ADC expected recovery
1.00  18.80  27.80                   0.65
1.05  21.47  30.77                   1.82
```

The data are GenIns, a public triangle of paid losses in `validation/data`. The
premium column is a constant 8.0m a year that only gives Bornhuetter-Ferguson an
exposure to scale. Use your own.

**Pitfalls.**

- Development ages are in months (12, 24, ...). The library refuses ages of 1,
  2, 3, because they are not on the development grid.
- Mack's formula gives a lifetime standard error. The one-year figure is
  smaller, and it is the right one for a one-year solvency capital: do not mix
  them. The ODP bootstrap's simulated one-year view differs from Merz-Wuthrich
  (`knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md` has the causes).
  `MackBootstrap` agrees with it.
- BF takes its exposure as a triangle column (the latest value for each origin)
  and `apriori` as one loss ratio for all origins. For a prior that changes by
  year, give `apriori` as a dict keyed by accident year (or a pandas Series
  indexed by year). Keys are safer than a list in origin order.
- The bootstraps give a `PredictiveDistribution` (each simulation of each
  origin), which section 8 needs. The mean and quantiles come from
  `boot.reserves.total()`.
- `cl.ultimate`, `cl.total_reserve` and `boot.reserves` are properties.
  `mack.claims_development_result()` is a method.

**More detail.** [Reserving methods](reserving-methods.md).

## 8. Capital, risk measures and allocation

Capital is an amount that you select from the distribution of the result of the
year. A **distortion risk measure** turns a distribution into one number and
gives more weight to the tail. **TVaR** at 99% is the average loss in the
worst 1% of years. The company selects the measure and the level.

**Data you usually have.** Simulated or gridded annual results by line or
contract (from the sections above, a reserve bootstrap, a cat model), their
dependence, and the capital standard and target return of the company.

| You want | Measure or method | Call |
|---|---|---|
| Capital standard, TVaR | Average of the worst `1 - p` of years | `Distortion.tvar(p).measure(dist)` |
| Capital standard, VaR (a Solvency II style 99.5%) | A quantile | `dist.quantile(p)` on a `PredictiveDistribution` or grid. `Portfolio.price(..., p=p)` sets the assets at it |
| A price that adds a margin for each unit of capital | Constant cost of capital | `Distortion.ccoc(r)`, `price(losses, assets, cost_of_capital=r)` |
| A price of risk that changes with the tail, fitted to a market quote | Proportional hazard, Wang, dual power | `Distortion.proportional_hazard`, `.wang`, `.dual_power`. `calibrate` |
| Capital split over lines or contracts | Euler (co-measure), covariance, proportional, Shapley | `capital(pd, distortion, method=...)`, `allocate` |
| Dependence between lines | Copula or rank correlation | `GaussianCopula`, `StudentTCopula`, `ArchimedeanCopula("clayton" or "gumbel", ...)`, `iman_conover`, `simulate` |

```python
from prospicio.distributions import Lognormal
from prospicio.risk import Distortion, GaussianCopula, capital, simulate

KEYS = [("reserve",), ("premium",), ("cat",)]
lines = simulate(
    GaussianCopula([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]),
    [Lognormal.from_mean_cv(50, 0.15), Lognormal.from_mean_cv(30, 0.25), Lognormal.from_mean_cv(10, 1.0)],
    20_000, 1, keys=KEYS, dims=["risk"],
)
means = [lines.marginal(k).mean() for k in KEYS]
print("expected loss  ", [round(m, 1) for m in means])
tvar99 = Distortion.tvar(0.99)
allocs = {method: capital(lines, tvar99, method=method) for method in ("euler", "covariance", "marginal")}
a = allocs["marginal"]
print(f"TVaR 99% of the total {a.total:.1f}; the lines alone sum to {sum(a.standalone):.1f}")
```

```text
expected loss   [50.0, 30.0, 10.0]
TVaR 99% of the total 159.1; the lines alone sum to 196.6
```

The lines are reserve risk, premium risk and a cat line, each a lognormal
joined by a Gaussian copula. The allocations split the TVaR of the total and
include the expected loss of each line. Capital in the usual sense is the
allocation minus the mean of the line.

```python
import pandas as pd
from great_tables import GT

rows = []
for method, a in allocs.items():
    rows.append([method, *a.allocated, sum(a.allocated), *[x - m for x, m in zip(a.allocated, means)]])
table = pd.DataFrame(rows, columns=["method", "reserve", "premium", "cat", "sum",
                                    "reserve_above", "premium_above", "cat_above"])
(GT(table, rowname_col="method")
 .tab_header("Allocation of the TVaR 99% of the total", "By method; the right half is the allocation minus the expected loss")
 .tab_spanner("allocated", ["reserve", "premium", "cat", "sum"])
 .tab_spanner("above the mean", ["reserve_above", "premium_above", "cat_above"])
 .cols_label(reserve_above="reserve", premium_above="premium", cat_above="cat")
 .fmt_number(["reserve", "premium", "cat", "sum", "reserve_above", "premium_above", "cat_above"], decimals=1))
```

```text
            reserve  premium   cat    sum  reserve  premium   cat
euler          58.6     40.7  59.8  159.1      8.6     10.7  49.9
covariance     44.8     48.5  65.7  159.1     -5.1     18.5  55.8
marginal       56.3     38.0  40.4  134.7      6.3      8.0  30.4
```

Euler and covariance add to the total but split it differently: covariance
gives the stable reserve line less than its mean. Under both, the cat line
(expected loss 10) takes most of the capital, because it drives the worst
years. The marginal method (the fall of the total without the line) does not
add to the total, so it is a diagnostic, not an allocation. The total is much
smaller than the lines alone: the diversification benefit.

**Pitfalls.**

- A 99% tail measure from 20,000 simulations uses 200 points, so the allocation
  to a small line with a heavy tail is noisy. Add simulations until the
  results are stable.
- A Gaussian copula has no tail dependence. If lines fail together in the worst
  years, use `StudentTCopula` or `ArchimedeanCopula` (`"gumbel"` joins the large
  values, `"clayton"` the small ones), or a scenario.
- `capital`, `price_portfolio` and `allocate` need components that add to the
  whole: lines or contracts, not gross, ceded and net together.
- `Tower.apply` gives a `PredictiveDistribution` with all kinds (gross, ceded,
  net, premiums, commissions). Its total adds them all, so it counts some
  amounts twice. Take one kind with `select("kind", ["ceded"])` before you join
  it to a book, as below.
- After `calibrate`, read a distortion with the properties `d.family` and
  `d.parameters` (for example `d.parameters["rho"]` for a proportional hazard).

The double count, and how to prevent it:

```python
from prospicio.aggregate import simulate_events
from prospicio.distributions import Pareto, Poisson
from prospicio.reinsurance import Layer, Tower

res = Tower([Layer("5x5", 5.0, 5.0)]).apply(simulate_events(Poisson(4.0), Pareto(1.0, 1.8), 2000, 1))
print("components:", res.components())
treaty = res.select("kind", ["ceded"])
print("selected:", treaty.components())
print(f"total of everything {res.total().mean():.2f}; the ceded loss alone {treaty.total().mean():.2f}")
```

```text
components: [('gross', 'ground_up'), ('ceded', '5x5'), ('net', 'retained')]
selected: [('ceded', '5x5')]
total of everything 17.76; the ceded loss alone 0.55
```

The first total is gross plus ceded plus net, twice the gross loss. Only the
second is the treaty, ready to join a book.

**More detail.** [Risk loads and distortions](risk-loads-and-distortions.md).

## 9. Pricing a contract inside an existing portfolio

A reinsurer prices a contract on top of the business that it already has. A
contract with losses in other years than the rest of the book needs little
additional capital. One with losses in the same years needs much. A portfolio
total has a price under a distortion, and **co-measure** allocates it back to
the parts by what each contributes in the bad years, so the parts add to the
total. This is the **natural allocation**.

**Data you usually have.** The annual loss distribution of the existing book
(from the capital model, or made as in sections 2 to 7), the terms and loss
model of the new contract, and the capital standard and cost of capital.

| The existing book and the contract are | Method | Call |
|---|---|---|
| Independent, both on grids with the same step | Portfolio from independent units | `Portfolio.from_independent(units, grids)`, then `.price(distortion, p=...)` |
| Dependent, or from simulation | Joint simulation, then price | `PredictiveDistribution`, `iman_conover` or a copula, `price_portfolio(pd, assets, cost_of_capital=r)` |

```python
import pandas as pd
from great_tables import GT

from prospicio.aggregate import fft
from prospicio.distributions import Grid, Pareto, Poisson
from prospicio.pricing import Portfolio
from prospicio.reinsurance import Layer, Tower
from prospicio.risk import Distortion

STEP, POINTS = 0.1, 2**14
large_losses, _ = Grid.rounding(Pareto(1.0, 1.8), STEP, 201)
contract = Tower([Layer("5x5", 5.0, 5.0, reinstatements=1)]).on_grid(Poisson(4.0), large_losses, POINTS).ceded[0]
rest_of_book, _ = fft(Poisson(3.0), large_losses, POINTS)

ccoc = Distortion.ccoc(0.10)
in_book = Portfolio.from_independent(["book", "contract"], [rest_of_book, contract]).price(ccoc, p=0.99)
alone = Portfolio.from_independent(["contract"], [contract]).price(ccoc, p=0.99).total
units = [(name, unit) for name, unit in zip(in_book.units, in_book.allocated)] + [("contract alone", alone)]
priced = pd.DataFrame([(name, u.loss, u.premium, u.capital, u.assets) for name, u in units],
                      columns=["unit", "loss", "premium", "capital", "assets"])
(GT(priced, rowname_col="unit")
 .tab_header("Pricing the contract", "Natural allocation of the 99% VaR, CCoC 10%")
 .cols_label(loss="expected loss")
 .fmt_number(["loss", "premium", "capital", "assets"], decimals=3))
```

```text
                expected loss  premium  capital  assets
book                    6.364    8.091   17.266  25.357
contract                0.584    0.635    0.508   1.143
contract alone          0.565    1.022    4.578   5.600
```

Both prices use assets at the 99% quantile (VaR) of the total and a 10% cost of
capital. In each row, assets are premium plus capital. In the book, the
contract is independent of the rest, so the bad years of the rest are not its
bad years, and it needs far less capital than alone. Its expected loss alone
is a little lower, because limited liability caps the payments at the assets.
This is the view of the reinsurer. A cedant sees only the price.

**Pitfalls.**

- Independence is an assumption. Add dependence with a copula or `iman_conover`
  on a `PredictiveDistribution` (section 8) and price it with
  `price_portfolio`.
- `Portfolio.from_independent` needs grids with the same step. The same
  contract costs less in a book that it diversifies.
- The calls give a risk-loaded price (loss plus a margin on allocated capital).
  `Quote.technical(price, expenses=, brokerage=, fixed=, limit=)` turns a
  `Price` into the premium that the cedant pays, with the rate
  on line, payback, multiple and loss ratio. `Quote.at(premium)` gives the
  return of the reinsurer at the quoted premium of the broker.
- `Quote.technical` takes a standalone `Price` from `price(...)` and refuses a
  unit of the `Portfolio.price` allocation. For a contract priced in the book,
  calculate (premium + fixed) / (1 - expenses - brokerage) yourself, or price
  the contract standalone first.

**More detail.** [Risk loads and distortions](risk-loads-and-distortions.md),
[What a reinsurance price is](reinsurance-price.md), section 6.

## See also

- [Choosing a method](choosing-a-method.md) (this guide)
- [Risk loads and distortions](risk-loads-and-distortions.md)
- [Experience and exposure rating](experience-and-exposure-rating.md)
- [Aggregate distributions](aggregate-distributions.md)
- [Severity and frequency](severity-and-frequency.md)
- [Reserving methods](reserving-methods.md)
- [What a reinsurance price is](reinsurance-price.md)
- The worked notebooks in [`examples/`](../../examples/README.md), and
  `knowledge/findings/program-evaluation-api-map.md`, which maps each step of
  the evaluation of a submission to the Python API.
