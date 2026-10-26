# Choosing a method: which prospicio tool for which job

## Use this guide when

You price primary insurance (rating plans, GLMs, loss ratio indications), and
now you must do reinsurance work or answer a wider set of actuarial questions.
This guide tells you which method fits each situation and which prospicio
calls run it. It is the map of the guide set. For each situation it gives:

- the data that you usually have
- the method to use, and why
- the calls
- the pitfalls
- the guide that gives more detail

It does not teach each method. It refers you to the guide that does.

Money in the examples is in millions of a synthetic currency, and all examples
are synthetic. The numbers show how to read the output, not what a market
pays. Each code block imports what it uses and runs alone, except where it
says that it uses an earlier block. This guide writes a layer as limit xs
attachment: "4 xs 6" pays the part of a loss between 6 and 10. Each prospicio
call takes the limit first: `Layer("4x6", 4.0, 6.0)`,
`expected_layer_loss(4.0, 6.0)`.

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
2. On what does the cover pay? It can pay on one loss (an excess of loss
   layer) or on the total of the year (stop loss). It can also pay on a share
   of all losses (quota share) or on an event (catastrophe). The answer selects how you represent losses, which
   is the subject of the next section.

## Three ways to represent losses

Almost all methods below need a model of the losses of the year. prospicio has
three ways to hold one. Where they overlap, they give the same result, within
the step of the grid and the Monte Carlo error of the simulation.

| Way | You give | You get | Use when |
|---|---|---|---|
| Grid (exact) | a claim count and a severity | the annual loss as a distribution on a grid of steps, with no simulation error | the cover depends only on amounts: excess of loss, quota share, stop loss, aggregate limits, reinstatements pro rata as to amount |
| Simulation | a claim count and a severity, a risk profile or a year loss table | simulated years of events, with Monte Carlo error | the cover needs event times (reinstatements pro rata as to time) or sums insured (surplus), or you want to join with other sources |
| Formula | a severity and a layer | expected loss, variance | a quick look at one layer under a Pareto or other severity, with no annual terms, reinstatements or commissions |

A layer with reinstatements pro rata as to time has no meaning on a grid,
because a compound distribution has no event times. The call refuses it:

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

The second line is a pitfall. A layer with annual terms (an aggregate limit or
reinstatements) has no `net` grid. Thus make the net yourself or use
simulation. `net` is available when no layer has annual terms, or when the last
stage is one aggregate cover such as a stop loss. When you need these features,
use simulation (`simulate_events`, `RiskProfile.simulate`, `Tower.apply`).

## 1. Primary ratemaking and the overall rate indication

**What it is.** The indicated rate change that makes premium equal to expected
losses and expenses. The **loss ratio method** compares the projected loss
ratio with the permissible loss ratio. The **pure premium method** calculates a
rate from the losses per exposure unit. The two methods need:

- premium at the **current rate level** (the **on-level factor**, by the
  parallelogram method)
- losses **developed** to ultimate and **trended** to the policy period
- a **credibility** weight against a complement

**Data you usually have.** Written and earned premium by year, a history of
rate changes, reported or paid losses with development factors, trend
selections, and expense and profit provisions.

**When to use it.** When the question is an average rate level for a book that
has a rating plan. It is not the tool for one treaty, which has no rate history
to put on-level.

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
2023 only on a small part. Thus the factors decrease for more recent years. The
indication compares the projected loss ratio with the permissible loss ratio.
Credibility moves it toward the complement.

**Pitfalls.**

- `experience.loss_ratio`, `provisions.variable` and
  `RateLevel.on_level_factor` are properties, not methods. Read them without
  parentheses.
- Dates are year fractions (`2022.5` is mid 2022). `year_fraction` converts a
  calendar date.
- The calculation assumes uniform writing and earning. For a policy term other
  than a year, or for a risks-attaching treaty, make sure that the period type
  (`Period.calendar` or `Period.policy`) is correct.
- The experience classes are for primary ratemaking. They do not use a layer or
  a treaty year. To trend, develop and convert individual large losses, use
  `LargeLossExperience(ids, accident, amount, threshold=, valuation=, fx=)` and
  its `.prepare(rate=, to=, development=)` (see
  [Experience and exposure rating](experience-and-exposure-rating.md)).

**More detail.** [Experience and exposure rating](experience-and-exposure-rating.md)
for the pure premium method and credibility. `docs/design/ratemaking.md` for
the definitions.

## 2. Per-risk property excess of loss

A **per-risk** cover pays on each loss to one risk (a building, a plant), above
an attachment and up to a limit. Two methods price it. In practice, use the two
methods together:

- **Exposure rating** prices the layer from the **risk profile** of the cedant.
  The profile gives the sum insured in each band. An exposure curve gives the
  share of the expected loss of a risk that is below a given fraction of its
  maximum possible loss (MPL). The MPL is usually the sum insured, and
  `RiskProfile` uses it so. This method needs no loss history.
- **Experience rating** fits the large losses of the cedant and applies the
  fitted severity to the layer. It shows what really occurred. But a high layer
  possibly had only two losses in five years.

**Data you usually have.** A table of bands (sum insured, number of risks,
premium) and a list of large losses with their dates, possibly with
development.

**Which to use.**

| You have | Use | Because |
|---|---|---|
| Profile and no history, or a new account | Exposure | There is nothing else |
| Profile and a thin history | Exposure, checked against experience | Experience on a high layer is mostly noise |
| Long, clean history on a low layer | Experience, with exposure as a check | The history is the cedant's own, and there is sufficient layer data |
| Both, and they do not agree | Blend with credibility, and find why they do not agree | A large difference shows that the curve, the profile or the loss data is wrong |

The example rates the layer 4 xs 6 (the part of each loss between 6 and 10).

```python
from prospicio.credibility import LimitedFluctuation
from prospicio.distributions import Pareto, Poisson
from prospicio.pricing import CollectiveModel, Mbbefd, RiskProfile
from prospicio.reinsurance import Layer


def profile(c):
    return RiskProfile([1.0, 3.0, 6.0, 12.0], [400, 150, 60, 20], Mbbefd.swiss_re(c),
                       premiums=[8.0, 6.0, 5.0, 4.0], loss_ratio=0.55)


for c in (2.0, 3.0, 4.0):
    print(f"Swiss Re c={c:.0f}: 4x6 expected loss {profile(c).expected_layer_loss(4.0, 6.0):.3f}")
exposure_el = profile(3.0).expected_layer_loss(4.0, 6.0)

large = [1.2, 1.5, 1.1, 2.4, 1.8, 1.3, 3.9, 1.6, 1.05, 2.2, 5.5, 1.4, 1.9, 1.15, 7.8, 1.25, 2.9, 1.7, 1.35, 4.4]
YEARS = 5
fit = Pareto.fit(large, 1.0)                                   # losses above 1, alpha by maximum likelihood
experience_el = CollectiveModel(Poisson(len(large) / YEARS), fit).layer_mean(4.0, 6.0)
burning = Layer("4x6", 4.0, 6.0).ceded(large) / YEARS         # what the layer would have paid
print(f"fitted alpha {fit.alpha:.2f}; burning cost {burning:.3f}; fitted Pareto {experience_el:.3f}; exposure {exposure_el:.3f}")

severity_cv = fit.layer_variance(4.0, 6.0) ** 0.5 / fit.layer(4.0, 6.0)    # spread of the payment per large loss
for p, k in ((0.90, 0.10), (0.90, 0.05)):                      # two full-credibility standards
    lf = LimitedFluctuation(p, k)
    z = lf.credibility(len(large), lf.pure_premium_standard(severity_cv))
    print(f"standard ({p}, {k}): credibility {z:.2f}; blend {z * experience_el + (1 - z) * exposure_el:.3f}")
```

```text
Swiss Re c=2: 4x6 expected loss 0.482
Swiss Re c=3: 4x6 expected loss 0.347
Swiss Re c=4: 4x6 expected loss 0.224
fitted alpha 1.48; burning cost 0.360; fitted Pareto 0.774; exposure 0.347
standard (0.9, 0.1): credibility 0.06; blend 0.374
standard (0.9, 0.05): credibility 0.03; blend 0.361
```

Read the output in three steps:

1. The curve alone changes the expected loss of the layer by more than a factor
   of two (0.482 against 0.224). Thus the curve is the main assumption of
   exposure rating. Test it. Do not accept it without a test.
2. The fitted Pareto (0.774) is more than twice the exposure rating (0.347).
   The actual burning cost (0.360) is near the exposure rating. The burning
   cost is what the layer paid in the five years, from the one loss above 6.
3. A layer with this shape needs many more than 20 claims for full credibility.
   Thus the experience gets a weight of 0.06, and the exposure rating has most
   of the weight.

The weight depends on the standard that you select. A stricter standard (0.05
instead of 0.10) decreases it by half, to 0.03. Limited fluctuation is one
choice. A Bühlmann-type weight between exposure and experience is also common.
The blend shows the steps. It is not a rule.

**Pitfalls.**

- The curve is the model. `Mbbefd.swiss_re(c)` is a one-parameter family. The
  class docstring gives c = 1.5, 2, 3, 4 as the Swiss Re curves and 5 as the
  Lloyd's curve. Select the curve for each occupancy class yourself, and show
  the sensitivity as above.
- Exposure curves use the fraction of the MPL. For large industrial risks, the
  MPL can be much less than the total sum insured. If you rate on the sum
  insured, each loss spreads over too large a base, and the layer loss changes
  much. Put the MPL in `sums_insured` when you have it. A first-loss scale from
  a primary rating manual is a different curve on a different base. Do not mix
  the two.
- Prepare large losses before you fit. Trend them to the treaty year, develop
  open claims, index them if the treaty has an inflation clause, and convert
  them to one currency. `LargeLossExperience` does each step:
  - trend by date at a rate (or a factor for each loss)
  - development of known losses by a `DevelopmentPattern` (or a factor for
    each loss)
  - an `fx` rate
  - an `IndexClause` given to `burning_cost(index=, ratios=)`
- After `prepare` develops the known losses, give
  `burning_cost(..., development=)` only the count IBNR (a factor for each year
  or a `DevelopmentPattern`). Do not give an excess factor that already
  includes IBNER, because the method then counts it twice.
- With a reporting threshold, the method omits losses below `complete_above()`
  (the threshold times the largest trend factor). Thus each year is complete on
  the same basis. To show history under the program of each year next to the
  proposed program, use `PreparedLosses.as_if(historical_towers, years,
  proposed)`.
- `Pareto.fit` takes a threshold, optional reporting thresholds for each loss,
  flags for losses capped by a policy limit (`censored`) and a `truncation`.
  `Pareto.fit_with_standard_error` also gives the standard error of alpha. For
  the lognormal, gamma, Weibull and Burr, `fit_severity(family, losses,
  reporting_thresholds=, censored=)` fits the same data by maximum likelihood.
  `goodness_of_fit` compares the fit with the product-limit estimate.
- A surplus treaty cedes from each risk a share of its sum insured above a
  retention. It needs simulated events that have sums insured. Section 5 runs
  one on this profile.

**More detail.** [Experience and exposure rating](experience-and-exposure-rating.md),
[Severity and frequency](severity-and-frequency.md).

## 3. Casualty excess of loss

Casualty covers (general liability, motor liability, professional lines) attach
high and pay for many years. The data are a small number of large losses,
policy limits, and slow development. The two main ideas are:

- The **increased limit factor** (ILF) is the ratio of the expected loss at one
  policy limit to the expected loss at a basic limit.
- **Pareto extrapolation**: if the severity above a threshold is Pareto with a
  given alpha, one priced layer gives the price of any layer.

**Data you usually have.** A limits profile (premium written
by policy limit) and the expected loss ratio of the cedant. You also have a
severity curve (from a rating bureau or your own) and a short list of large
losses that still develop.

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

LIMIT, ATTACHMENT, ELR, BASIC = 1.5, 0.5, 0.65, 0.1
layer_el = 0.0
for policy_limit, premium in zip([0.5, 1.0, 2.0], [30.0, 20.0, 10.0]):      # premium written at each limit
    low, high = min(policy_limit, ATTACHMENT), min(policy_limit, ATTACHMENT + LIMIT)
    in_layer = (ilf(policy_severity, high, BASIC) - ilf(policy_severity, low, BASIC)) / ilf(policy_severity, policy_limit, BASIC)
    layer_el += premium * ELR * in_layer
    print(f"policy limit {policy_limit}: share of its loss in the layer {in_layer:.3f}")
print(f"exposure-rated 1.5 xs 0.5: {layer_el:.3f}")

for alpha in (1.5, 2.5):
    print(f"alpha {alpha}: layer 2x2 costs {pareto_extrapolation((1.0, 1.0), (2.0, 2.0), alpha):.1%} of layer 1x1")

tower = match_tower([1.0, 2.0, 5.0], [0.30, 0.20, 0.10])     # three layers; the last is unlimited
print(f"matched tower: {tower.frequency:.2f} losses a year above 1, alphas {[round(a, 2) for a in tower.severity.alpha]}")
print(f"2 xs 3 from the matched tower: {tower.layer_loss(2.0, 3.0):.4f}")
```

```text
ILF at a limit of 1 against a basic limit of 0.1: 1.68
policy limit 0.5: share of its loss in the layer 0.000
policy limit 1.0: share of its loss in the layer 0.084
policy limit 2.0: share of its loss in the layer 0.129
exposure-rated 1.5 xs 0.5: 1.934
alpha 1.5: layer 2x2 costs 70.7% of layer 1x1
alpha 2.5: layer 2x2 costs 35.4% of layer 1x1
matched tower: 0.58 losses a year above 1, alphas [1.86, 1.73, 2.39, 2.2]
2 xs 3 from the matched tower: 0.0890
```

The policies with a limit of 0.5 put nothing in the layer. The higher limits
put a small share of their loss in it (0.084 and 0.129). The layer loss of the
profile, 1.934, is the sum of premium times loss ratio times share. Compare it
with a burning cost on developed, trended large losses
(`PreparedLosses.burning_cost`, the steps of section 2) before you use either.

**Pitfalls.**

- Prepare losses before you fit or burn them. Trend them to the treaty period
  and develop open claims one by one with `LargeLossExperience.prepare(rate=,
  to=, development=)`. `ExcessPattern` gives the development of the excess
  layer. It blends layer factors from the ground-up triangle and a severity
  with the factors of the layer
  ([`reserving-excess-layers.md`](reserving-excess-layers.md)).
- Decide how allocated loss adjustment expense (ALAE) goes into the layer: pro
  rata with the loss, or added to it within the limit. Treat the losses and the
  severity curve the same way.
- Pareto extrapolation assumes one alpha across the layers. In casualty this is
  usually not true. Thus `match_tower` fits a piecewise Pareto with one alpha
  for each layer.
- `match_tower` and `fit_references` refuse inputs that cannot come from a
  decreasing, convex excess loss function (for example, rates on line that
  increase up the tower). A match gives its references exactly. Thus read
  `tower.diagnostics()` after the match. Its `warnings` show an alpha below 1,
  or adjacent alphas that are more than twice apart. `fit_references(...,
  alpha_bounds=(lo, hi))` keeps the fitted alphas in a range that you set.
- `compound_moments` gives an error when a severity does not have three finite
  moments (a Pareto with alpha below 3). Thus you cannot use its normal or
  gamma `approximate` there. Use the grid or simulation for heavy tails.
- These items change which losses go into the layer. Agree them with the
  wording before you rate:
  - extra-contractual obligations and losses in excess of policy limits (ECO
    and XPL)
  - clash between policies on one occurrence
  - claims-made or occurrence triggers
- The prices in this section are one-period prices. For a long tail, calculate
  the price over its payout. Use a `PaymentPattern` (for the next accident
  year, `PaymentPattern.from_development(ages_in_years, paid_cdf)`) and
  `price_over_payout(price, pattern, rate=, cost_of_capital=)`. This has two
  effects. Discounting decreases the loss (`discount_factor` below 1). Capital
  held until the last payment increases the capital charge (`capital_years`
  above 1). With a cost of capital, the second effect is usually larger for a
  long tail. Thus the payout premium is frequently above the one-period
  premium.

**More detail.** [Experience and exposure rating](experience-and-exposure-rating.md),
[Severity and frequency](severity-and-frequency.md).

## 4. Catastrophe excess of loss

A **catastrophe** cover pays on the total of an event (a hurricane, an
earthquake) across many risks. The cedant or broker usually gives the output of
a vendor model, and you do not model the hazard again. You get one of these:

- a **year loss table**: a simulated list of years, each with its events and
  loss amounts
- **return period points** on the occurrence exceedance probability (OEP)
  curve: the event loss that occurs once in 100 years, and so on

**Data you usually have.** The year loss table of the vendor model, or its OEP
and AEP (aggregate exceedance) points for the portfolio of the cedant. Also the
layers and reinstatement terms of the program, and possibly some historical
events.

| You have | Use | Call |
|---|---|---|
| An event loss table (rate, mean, standard deviations and exposure for each event) | Simulate years from it, or read its OEP and PML | `EventLossTable(rate, mean, sd_independent, sd_correlated, exposure)`. `.simulate(n_years, seed)` gives `(EventSet, event ids)`. `.oep(x)`, `.occurrence_pml(p)` |
| A year loss table | Load it as events, apply the terms | `EventSet.from_years(years, sums_insured, seed, times)`, or `EventSet.from_long(years, losses, n_years)` for one row for each loss (years are indices from 0, and `n_years` is the full simulation count, including the years with no loss). Then `Tower.apply` |
| Return period points | Fit a frequency and a severity to them, then use the grid or simulation | `fit_pml_curve(return_periods, amounts, tail_alpha, truncation)` |

```python
import math

from prospicio.aggregate import simulate_events
from prospicio.distributions import Grid, Poisson
from prospicio.pricing import fit_pml_curve
from prospicio.reinsurance import Layer, Tower

RETURN_PERIODS, AMOUNTS = [10, 20, 50, 100, 250], [30.0, 55.0, 100.0, 140.0, 200.0]
oep = fit_pml_curve(RETURN_PERIODS, AMOUNTS, tail_alpha=2.0, truncation=500.0)
for rp, amount in zip(RETURN_PERIODS, AMOUNTS):
    print(f"return period {rp:3d}: events a year above {amount:5.1f} {oep.excess_frequency(amount):.4f}, 1/{rp} = {1 / rp:.4f}")
print(f"an OEP probability of 0.10 is {-math.log(1 - 0.10):.4f} events a year")
frequency = Poisson(oep.frequency)
grid, _ = Grid.rounding(oep.severity, 1.0, 501)
cat_layers = [Layer("50x50", 50.0, 50.0, reinstatements=1), Layer("100x100", 100.0, 100.0, reinstatements=1)]
exact = Tower(cat_layers).on_grid(frequency, grid, 2**11)
print("exact on the grid:", [round(g.mean(), 3) for g in exact.ceded])
sim = Tower(cat_layers).apply(simulate_events(frequency, oep.severity, 20_000, 1))
print("simulated events: ", [round(sim.marginal(("ceded", l.name)).mean(), 3) for l in cat_layers])
```

```text
return period  10: events a year above  30.0 0.1000, 1/10 = 0.1000
return period  20: events a year above  55.0 0.0500, 1/20 = 0.0500
return period  50: events a year above 100.0 0.0200, 1/50 = 0.0200
return period 100: events a year above 140.0 0.0100, 1/100 = 0.0100
return period 250: events a year above 200.0 0.0040, 1/250 = 0.0040
an OEP probability of 0.10 is 0.1054 events a year
exact on the grid: [1.672, 0.948]
simulated events:  [1.684, 0.947]
```

The fit gives each point again. It reads a return period of RP as an
exceedance frequency of 1/RP events a year. An OEP probability p from a vendor
is a different number. First convert it to a frequency with -ln(1 - p) (the
line after the five points). The two methods that apply the terms agree within
simulation error. The same code then prices reinstatements. When the events
have times (`EventSet.with_uniform_times`, `with_seasonal_times`), the
simulation also applies reinstatements pro rata as to time.

**Pitfalls.**

- A fitted OEP curve describes events above its lowest point only. Here it
  models nothing below 30.
- Return period points give the loss of one event, not the total of the year.
  For a cover with an annual limit or paid reinstatements, the number of events
  is important. When the events of the vendor cluster, a Poisson count fitted
  to OEP points gives too few years with several events. Thus use the year loss
  table, which has the multi-event years of the vendor (its AEP). Use the OEP
  fit only when you do not have the year loss table.
- In an event loss table (`EventLossTable`), the secondary uncertainty of each
  event is a beta on [0, exposure]. The beta has the mean and standard
  deviation of the row.
- `EventSet.grouped(window_hours)` groups losses into occurrences under an hours
  clause. It needs the time of each loss (`from_years(..., times=)`,
  `with_uniform_times` or `with_seasonal_times`) as a fraction of a year. The
  window is in hours (a year is 8,760 hours).
- `EventSet.merge` adds a non-cat set year by year.
- For clash across two lines (one event that hits both), use `Clash` or
  `fft_bivariate` on a joint grid (`prospicio.aggregate`). Add a copula between
  the lines in an event if you need one.

**More detail.** [Aggregate distributions](aggregate-distributions.md) for the
grid and simulation, [What a reinsurance price is](reinsurance-price.md) for
the price of a layer.

## 5. Quota share, surplus and other proportional treaties

A **quota share** cedes a constant share of the premium and losses of each policy.
For each risk, a **surplus** treaty cedes the part of the sum insured above
the retention line of the cedant. It cedes up to a given number of lines. Thus the ceded share
is different for each risk. In both, the reinsurer pays the cedant a **ceding
commission**, and the commission is the price.

**What sets the price.** The loss ratio of the full book, not only the large
losses. Thus you need the distribution of the annual loss ratio, and the terms
that change the commission with it:

- a **sliding scale**: the commission decreases as the loss ratio increases
- a **profit commission**: the reinsurer gives a share of its profit back to the
  cedant

**Data you usually have.** The subject premium (the premium
that the treaty shares) and the loss ratios of the cedant by underwriting year
with their development. You also have its expense ratio, and the commission and
brokerage terms offered.
For a surplus, also the risk profile by sum insured. The expected loss ratio is
the on-level, developed and trended loss ratio of the cedant, calculated as in
section 1. The spread around it comes from the claim count and severity, or
from the variation of those loss ratios from year to year.

| Situation | Method | Call |
|---|---|---|
| Quota share price | Technical commission: 1 minus the expected ceded loss ratio, expenses, brokerage and margin | `Tower.on_grid` for the ceded grid (or `Tower.on_annual_grid` on a loss-ratio grid), `price(grid, Distortion.tvar(0.99), cost_of_capital=r)` for the margin |
| Sliding scale or profit commission | Expected commission over the loss ratio distribution | `with_sliding_scale`, `with_profit_commission`, `TowerGrids.expected_ceding_commission` or `AnnualGrids.expected_ceding_commission` |
| Surplus | Simulated events with sums insured | `Layer.surplus`, `RiskProfile.simulate`, `Tower.apply` |
| Surplus inuring to a per-risk excess of loss | Stages in order | `Tower.inuring([[surplus], [xl]])` |

The next example puts a quota share and an aggregate stop loss on the same
book, so that section 6 can use it again. The book has 400 claims a year (with
a spread wider than Poisson) and writes 100 of premium. The expected loss ratio
of 60% is an example of a section 1 selection.

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

The third line puts the same quota share on the total of the year alone. A
cedant can give a loss ratio distribution with no claim count, and then this is
all you have. `Tower.on_annual_grid` treats the total of each year as one loss.
The expected commission is the same 9.64, because the terms of the quota share
depend only on that total. Then `price` sets the assets at the 99% TVaR of each
grid. It charges 10% on the capital, which is the assets above the premium.

The two contracts need separate calculations:

- With a flat commission, only the ceded loss is at risk. A 10% return on its
  capital allows a commission of 34.5%.
- With the scale, the outgo of the reinsurer is the loss plus a commission that
  decreases as the loss increases. Thus the scale gives part of the loss ratio
  spread back to the cedant. The tail above the mean decreases from 6.71 to
  4.57, and the capital from 6.10 to 4.16.

The terms of the scale leave a margin of 1.32 against the 0.42 needed. Thus
the price of this scale is above the target, and you can increase it until the two
are equal. Do not compare the expected commission of the scale (32.1%) with the
flat technical commission (34.5%). They price different risks.

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

The cession then goes through the same commission calculation as the quota
share. A per-risk excess of loss on the retained part is a second stage
(`Tower.inuring([[surplus], [xl]])`).

**Pitfalls.**

- A quota share depends on the loss ratio of the full book. If you hold the
  attritional losses at their mean, the capital is too low. Give them a spread.
- Set the premium (`with_premium_rate`, `with_deposit_premium` or
  `with_rate_on_line`) before `with_sliding_scale` or `with_profit_commission`.
  The two terms need the premium to calculate a loss ratio. A sliding scale
  cannot combine with swing rating or paid reinstatements.
- For a quota share or stop loss on an annual loss ratio grid (no claim count),
  use `Tower.on_annual_grid(grid)`. It gives the ceded grid (`.ceded`),
  `expected_ceding_commission` and `expected_profit_commission` exactly. It is
  not for per-occurrence layers with a severity. Use `on_grid` for those.
- `price(losses, assets, cost_of_capital=r)` takes a grid, simulated losses
  (`Sampled`) or a `PredictiveDistribution`. Its `assets` argument is a
  distortion risk measure (here 99% TVaR). The result has `.margin`, `.capital`
  and `.assets`, which the example above uses.

**More detail.** [What a reinsurance price is](reinsurance-price.md), section 5.

## 6. Aggregate stop loss, adverse development and loss portfolio covers

- A **stop loss** (aggregate excess of loss) pays when the total of the year is
  more than a retention, up to a limit.
- An **adverse development cover** (ADC) does the same on reserves. It pays
  when the reserve for past years becomes worse than a retention.
- A **loss portfolio transfer** (LPT) moves a block of reserves to the
  reinsurer for a premium.

All three pay on a total. Thus the work is to find the distribution of that
total.

| The total comes from | Method | Call |
|---|---|---|
| A claim count and a severity | Exact aggregate on the grid | `Tower.on_grid`, or `fft` and `fft_auto` for the distribution alone |
| Simulated events | Apply to each year | `Tower.apply` |
| A reserve bootstrap | Apply the cover to the simulated reserve | `Tower.apply_aggregate(boot.reserves)` |

**Data you usually have.** For a stop loss, the same data as for a quota share
(section 5): subject premium and the loss ratio history. For an ADC or LPT, the
reserve triangles of the covered years, the booked reserves, and the payment
pattern.

The stop loss on the book of section 5 is already in `book` (this block uses
the quota share block of section 5). It pays on the gross total. Its deposit
premium of 2.0 in that block is only a placeholder. This block prices it as
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

A stop loss pays rarely, and when it pays, it pays a large amount. Thus its
capital (10.79) is large against its expected loss (0.339), and most of its
premium (1.42) is the margin on that capital. The capital standard and the cost
of capital set the price, so state both with the quote. The ADC on a reserve
is the same call on a different total. The reserving example below shows it.

**Pitfalls.**

- The retention and limit apply to the total that the cover sees. In a `Tower`
  with stages, that total is net of earlier stages (`Tower.inuring`).
- The total must have the correct tail. A normal or gamma approximation of the
  total (`compound_moments(...).approximate`) gives a heavy tail that is too
  small. The grid does not.
- For an ADC or LPT, the uncertainty is in the reserve. Thus use a bootstrap or
  a Bayesian model, not a point estimate. A bootstrap has only parameter and
  process error under the assumptions of its model (here the chain ladder with
  no tail). Unless you give a tail, it has no tail after the triangle. It also
  has no calendar-year inflation shock and no change in claims handling, which
  are usually the causes of an ADC loss. Treat its result as a minimum, and
  show the layer under a second method or tail (section 7 does this).
- Time value is very important for an ADC or LPT. The prices here are
  one-period prices. `price_over_payout(price(...), pattern, rate=,
  cost_of_capital=)` gives a price over its payout, with two effects.
  Discounting decreases the loss (`discount_factor` below 1). Capital held
  until the last payment increases the capital charge (`capital_years` above
  1). For a long tail at a cost of capital, the second effect is usually
  larger. Thus the payout premium is frequently above the one-period premium.
- The pattern for a reserve cover is the calendar-year run-off of the reserve
  from the valuation date. It is not the development of one accident year. Make
  it from the paid chain ladder (the `cl` of section 7) as follows:
  1. Origin i, with its latest age at index k_i, pays
     `ult_i / cdf[j] - ult_i / cdf[j - 1]` at calendar time j - k_i, for each
     later age index j.
  2. Add these payments over the origins by calendar time.
  3. Give the times and the shares of the total to
     `PaymentPattern(times, shares)`.
- A layer that attaches above the mean reserve pays later. Thus treat the
  duration of that pattern as a lower bound for the layer. For a
  cover on a future accident year, use
  `PaymentPattern.from_development(ages_in_years, paid_cdf)` (with `.shifted`).
  Examples are a casualty excess of loss or a stop loss on the book of next
  year.
- `Layer.with_rate_on_line` needs a finite limit for each loss. A stop loss
  does not have one (its limit is annual), so the call refuses it. Set the
  premium of a stop loss with `with_deposit_premium` or `with_premium_rate`.

**More detail.** [Aggregate distributions](aggregate-distributions.md),
[Reserving methods](reserving-methods.md).

## 7. Reserving a book and its uncertainty

Reserving changes a **triangle** (losses by origin year and development age)
into an estimate of the amount still to be paid. There are three questions,
and each has a method.

**Data you usually have.** Paid and incurred triangles by origin year, premium
or a different exposure by origin year, and a prior loss ratio from pricing or
plans.

| Question | Method | Call |
|---|---|---|
| The best estimate of the reserve | Chain ladder. Bornhuetter-Ferguson, Cape Cod or expected loss when a year is immature. Clark for a growth curve | `ChainLadder`, `BornhuetterFerguson`, `CapeCod`, `ExpectedLoss`, `ClarkLdf` |
| Its uncertainty to ultimate (the **lifetime** view) | Mack's formula, or a bootstrap for a full distribution | `Mack`, `OdpBootstrap`, `MackBootstrap` |
| Its uncertainty over the next calendar year (the **one-year** view, for solvency capital) | Merz-Wuthrich from Mack, or re-reserving in a bootstrap | `Mack().fit(...).claims_development_result()`, `OdpBootstrap.one_year`, `MackBootstrap.one_year` |
| A distribution that includes prior information | Bayesian chain ladder | `BayesOdp`, `Meyers` |

To select the point estimate: the chain ladder reads development only from the
triangle. For the latest origin years, which have little development, it
multiplies a small number by a large factor. Bornhuetter-Ferguson mixes the
chain ladder with a prior expected loss ratio and is more stable there. Use it
when you have a prior that you trust, and an exposure measure (premium) in the
triangle.

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
print("ultimates of the latest three years, chain ladder:", [round(u / 1e6, 2) for u in cl.ultimate[-3:]],
      "Bornhuetter-Ferguson:", [round(u / 1e6, 2) for u in bf.ultimate[-3:]])
adc_layer = Tower([Layer.stop_loss("ADC", 5e6, 20e6)])
for tail in (1.0, 1.05):                               # no tail, then a 5% tail past the oldest age
    boot = OdpBootstrap(n_sims=2000, seed=1, tail=tail).fit(tri, "paid")
    total = boot.reserves.total()
    recovery = adc_layer.apply_aggregate(boot.reserves).marginal(("ceded", "ADC")).mean()
    print(f"ODP bootstrap, tail {tail}: mean {total.mean() / 1e6:.2f}m, 99.5% {total.quantile(0.995) / 1e6:.2f}m; "
          f"ADC 5m xs 20m expected recovery {recovery / 1e6:.2f}m")
```

```text
chain ladder reserve 18.68m; Mack standard error 2.44m (next year only: 1.77m)
Bornhuetter-Ferguson reserve 17.53m
ultimates of the latest three years, chain ladder: [6.78, 5.64, 4.97] Bornhuetter-Ferguson: [5.87, 5.31, 5.18]
ODP bootstrap, tail 1.0: mean 18.80m, 99.5% 27.80m; ADC 5m xs 20m expected recovery 0.65m
ODP bootstrap, tail 1.05: mean 21.47m, 99.5% 30.77m; ADC 5m xs 20m expected recovery 1.82m
```

For the latest three years, the ultimates of the chain ladder go from 6.78m to
4.97m. The ultimates of Bornhuetter-Ferguson stay between 5.18m and 5.87m. This
is what "more stable" means.

The ADC shows why a bootstrap is a minimum for its uncertainty. The triangle
cannot show if a 5% tail is right or wrong. But this tail makes the expected recovery
almost three times larger (0.65m to 1.82m). The cost of the layer depends on
the model that you select, so show it under more than one model.

The data are GenIns, a public triangle of general insurance paid losses in the
`validation/data` folder of the repository. The premium column here is a
constant 8.0m for each year. It only gives Bornhuetter-Ferguson an exposure to
scale. Use your own.

**Pitfalls.**

- Development ages are in months (12, 24, ...). The library refuses ages of 1,
  2, 3, because they are not on the development grid.
- Mack's formula gives a lifetime standard error. The one-year figure is
  smaller, and it is the correct figure for a one-year solvency capital. Do not
  mix them. The simulated one-year view of the ODP bootstrap is different from
  Merz-Wuthrich. `knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md`
  records the causes. `MackBootstrap` agrees with Merz-Wuthrich.
- BF takes its exposure as a triangle column (the latest value for each origin)
  and `apriori` as one number. For a prior that changes by year, put the prior
  loss ratio times premium in the exposure column and use `apriori=1.0`.
- The bootstraps give a `PredictiveDistribution` (each simulation of each
  origin), which the capital section needs. The mean and quantiles come from
  `boot.reserves.total()`.
- Some accessors are properties and some are methods. `cl.ultimate`,
  `cl.total_reserve` and `boot.reserves` are properties.
  `mack.claims_development_result()` is a method.

**More detail.** [Reserving methods](reserving-methods.md).

## 8. Capital, risk measures and allocation

Capital is an amount of money that you select from a distribution of the result
of the year. A **distortion risk measure** changes a distribution into one
number. It gives more weight to the tail. **TVaR** at 99% is the average loss
in the worst 1% of years. The company selects the measure and the level. The
library gives the measures and does the calculation.

**Data you usually have.** Simulated or gridded annual
results by line or contract (from the sections above, a reserve bootstrap, a
cat model). You also have a view of how the lines move together, and the
capital standard and target return of the company.

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
for method in ("euler", "covariance", "marginal"):
    a = capital(lines, tvar99, method=method)
    print(f"{method:11s}", [round(x, 1) for x in a.allocated], f"sum {sum(a.allocated):.1f};",
          "above the mean", [round(x - m, 1) for x, m in zip(a.allocated, means)])
print(f"TVaR 99% of the total {a.total:.1f}; the lines alone sum to {sum(a.standalone):.1f}")
```

```text
expected loss   [50.0, 30.0, 10.0]
euler       [58.6, 40.7, 59.8] sum 159.1; above the mean [8.6, 10.7, 49.9]
covariance  [44.8, 48.5, 65.7] sum 159.1; above the mean [-5.1, 18.5, 55.8]
marginal    [56.3, 38.0, 40.4] sum 134.7; above the mean [6.3, 8.0, 30.4]
TVaR 99% of the total 159.1; the lines alone sum to 196.6
```

The three lines are reserve risk, premium risk and a cat line. Each line is a
lognormal, and a Gaussian copula connects them. The allocated amounts are
shares of the TVaR of the total, which is the assets needed in the worst 1% of
years. They include the expected loss of each line. Capital in the usual sense
is the money held above the expected loss. It is the allocation minus the mean
of the line (the "above the mean" lists).

Euler and covariance split the total so that the parts add to it, but they do
not agree on how. Covariance even gives the stable reserve line less than its
mean (-5.1). With each method, the cat line (expected loss 10) takes most of
the capital: 49.9 under Euler, against 8.6 for the reserve line with 50. The
cause is that the cat line drives the worst years. The marginal method (how
much the total decreases without the line) does not add to the total (134.7
against 159.1). Thus it is a diagnostic, not an allocation. The total is much
smaller than the sum of the lines alone (196.6). This is the diversification
benefit.

**Pitfalls.**

- Tail measures at 99% from 20,000 simulations use 200 points. The allocation
  to a small line with a heavy tail is noisy. Increase the number of
  simulations until the results are stable.
- A Gaussian copula has no tail dependence. If lines fail together in the worst
  years, use `StudentTCopula` or `ArchimedeanCopula`, or a scenario. The family
  `"gumbel"` connects the large values, and `"clayton"` connects the small
  values.
- `capital`, `price_portfolio` and `allocate` need components that add to the
  whole: lines or contracts, not gross, ceded and net together.
- `Tower.apply` gives a `PredictiveDistribution` with all kinds (gross, ceded,
  net, premiums, commissions). Its total adds them all, and thus counts some
  amounts two times. Take one kind with `select("kind", ["ceded"])` before you
  join it to a book, as below.
- After `calibrate`, read the family and parameters of a distortion with
  `d.family` and `d.parameters` (properties). For example,
  `d.parameters["rho"]` is the parameter of a proportional hazard.

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

The first figure is gross plus ceded plus net, which is twice the gross loss.
Only the second figure is the treaty, ready to join a book.

**More detail.** [Risk loads and distortions](risk-loads-and-distortions.md).

## 9. Pricing a contract inside an existing portfolio

A reinsurer prices a contract on top of the business that it already has. A
contract with losses in different years from the rest of the book needs little
additional capital. A contract with losses in the same years needs much
capital. The tool is a portfolio whose total has a price under a distortion.
**Co-measure** allocates this price back to the parts. The share of each part
is what it contributes in the bad years, so the parts add to the total. This is
the **natural allocation**.

**Data you usually have.** The annual loss distribution of
the existing book (from the capital model of the company, or made as in
sections 2 to 7). You also have the terms and loss model of the new contract,
and the capital standard and cost of capital of the company.

| The existing book and the contract are | Method | Call |
|---|---|---|
| Independent, both on grids with the same step | Portfolio from independent units | `Portfolio.from_independent(units, grids)`, then `.price(distortion, p=...)` |
| Dependent, or from simulation | Joint simulation, then price | `PredictiveDistribution`, `iman_conover` or a copula, `price_portfolio(pd, assets, cost_of_capital=r)` |

```python
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
for name, unit in zip(in_book.units, in_book.allocated):
    print(f"{name:9s} loss {unit.loss:6.3f} premium {unit.premium:6.3f} capital {unit.capital:6.3f} assets {unit.assets:6.3f}")
print(f"contract alone loss {alone.loss:6.3f} premium {alone.premium:6.3f} capital {alone.capital:6.3f} assets {alone.assets:6.3f}")
```

```text
book      loss  6.364 premium  8.091 capital 17.266 assets 25.357
contract  loss  0.584 premium  0.635 capital  0.508 assets  1.143
contract alone loss  0.565 premium  1.022 capital  4.578 assets  5.600
```

The two prices use the same standard: assets at the 99% quantile (VaR) of the
total, and a 10% cost of capital. In each row, the assets are the premium plus
the capital.

- Alone, the contract needs assets of 5.600 (its own 99% quantile) and a
  premium of 1.022.
- In the book, the contract is independent of the rest. The natural allocation
  gives it assets of 1.143 and a premium of 0.635 on an expected loss of 0.584.
  The bad years of the rest of the book are not its bad years.

Its expected loss alone (0.565) is a little lower, because limited liability
caps the payments at the assets. This is the view of the reinsurer. A cedant
sees only the price.

**Pitfalls.**

- Independence is an assumption. Add dependence with a copula or
  `iman_conover` on a `PredictiveDistribution`, as in section 8. Then price it
  with `price_portfolio`.
- `Portfolio.from_independent` needs grids with the same step. The allocated
  figures depend on the existing book. The same contract costs less in a book
  that it diversifies.
- The calls give a risk-loaded price (loss plus a margin on allocated capital).
  `Quote.technical(price, expenses=, brokerage=, fixed=, limit=)` changes a
  risk-loaded `Price` (from `price`) into the premium that the cedant pays. It
  also gives the rate on line, payback, multiple and loss ratio.
  `Quote.at(premium)` gives the return of the reinsurer at the quoted premium
  of the broker.
- `Quote.technical` takes a standalone `Price` from `price(...)`. It refuses a
  unit of the `Portfolio.price` allocation. For a contract priced in the book,
  calculate the premium yourself, (premium + fixed) / (1 - expenses -
  brokerage). Or price the contract standalone with `price` first.

**More detail.** [Risk loads and distortions](risk-loads-and-distortions.md),
[What a reinsurance price is](reinsurance-price.md), section 7.

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
