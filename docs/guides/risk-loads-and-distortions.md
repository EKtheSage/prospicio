# Risk loads and distortions: when to use which

<!-- site: execute -->

Use this guide to turn the loss distribution of a cover (an excess of loss
layer, a quota share or a whole book) into a price with a load for risk. A
primary actuary loads by a percentage of the expected loss. A reinsurer loads
for the tail, so two covers with the same expected loss get different prices
when they differ in how often and how badly they go wrong.

The guide runs from a simple load to a **distortion** (a reweighting of
probabilities that prices a loss by its tail), the capital standard,
calibration to a target or to quotes, and the price of a portfolio and its
split. Each choice has a rule for when to use it. The guide also marks the
modelling choices that no calculation can settle.

It assumes the annual loss distribution of the cover ([aggregate
guide](aggregate-distributions.md); [experience and exposure
rating](experience-and-exposure-rating.md) give its source) and the idea of a
layer ([price guide](reinsurance-price.md)).

## Which method for which situation

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| A first number for a screen, nothing else known | A load on expected loss | `classical_premium("expected_value", x, t)` | Cheap, but blind to the tail. Use it only to compare covers of one shape |
| A stable, roughly symmetric result (a large quota share) where the load must follow volatility | Standard deviation or variance principle | `classical_premium("standard_deviation", x, t)`, `calibrate_classical` | Price rises with spread. Needs a finite variance, and layer prices do not add up (section 2) |
| "We need 10% on capital" | Constant cost of capital (CCoC) | `price(x, Distortion.tvar(p), cost_of_capital=r)`, `Pentagon.solve(..., assets=a)` or `price_portfolio(..., cost_of_capital=r)` for capital on a TVaR or EPD standard. `Distortion.ccoc(r)` for capital equal to the maximum loss | Every unit of capital earns the same return, the easiest rule to explain. Say which capital the 10% is on (section 3) |
| One quote or target, price the rest of a program consistently | One-parameter distortion (proportional hazard, Wang) calibrated to it | `calibrate("ph", x, premium)`, then `d.measure(y)` or `TowerGrids.price(d)` | Prices add up across layers, one number to defend |
| Remote layers (catastrophe) with a minimum rate on line | A distortion with a floor: LEP, capped linear, linear yield, CCoC | `calibrate("lep", x, premium, r0=0.03)` | A family without a floor prices a 1-in-1,000 layer at a fraction of a percent |
| Quotes on several layers | Fit two parameters, or take the concave hull of the quotes | `calibrate(..., r0=...)` with a search on `r0`, `Distortion.convex(points)` | One parameter cannot match both ends of a program |
| The capital standard is a tail measure | TVaR, or assets for a target expected policyholder deficit | `Distortion.tvar(p)`, `Portfolio.assets_for_epd(e)` | VaR ignores how bad the tail is beyond its level |
| A contract goes into a book | Price the portfolio, then allocate | `price_portfolio(pd, assets, cost_of_capital=r)`, `Portfolio.price(d, p=0.99)` | Capital depends on what else is in the book (section 6) |
| Split the capital of a book among its parts | Euler (for a distortion, the natural or co-measure allocation), marginal, Shapley, Bodoff | `capital(pd, d)`, `allocate(pd, d)`, `Portfolio.price(..., allocation=)`, `Portfolio.bodoff` | They answer different questions (section 6) |
| Heavy tail, variance infinite or unreliable | A distortion, never a variance-based, Esscher or exponential principle | `calibrate("ph", ...)` | A distortion needs only the weighted mean of the tail |

## 1. The running example

The book is the one in [the price guide](reinsurance-price.md), on a coarser
grid. Each year has a Poisson count of 4 large losses, each with a Pareto tail
(alpha 1.8) capped at a policy limit of 20. Each of the three layers has one
free reinstatement, so it pays at most twice its limit in a year. A distortion
prices the **annual** loss of the layer, the loss the reinsurer carries.

```python
import math

import pandas as pd
from great_tables import GT
from prospicio.aggregate import fft
from prospicio.distributions import Grid, Lognormal, Pareto, Poisson, PredictiveDistribution
from prospicio.pricing import Pentagon, Portfolio, calibrate_classical, classical_premium, price, price_portfolio
from prospicio.reinsurance import Layer, Tower
from prospicio.risk import Distortion, GaussianCopula, calibrate, capital

STEP, POINTS = 0.1, 2**12
severity, _ = Grid.rounding(Pareto(1.0, 1.8), STEP, 201)   # losses above 1, capped at 20
frequency = Poisson(4.0)
LAYERS = [Layer("2x3", limit=2.0, attachment=3.0, reinstatements=1),
          Layer("5x5", limit=5.0, attachment=5.0, reinstatements=1),
          Layer("10x10", limit=10.0, attachment=10.0, reinstatements=1)]
NAMES = [layer.name for layer in LAYERS]
tower_grids = Tower(LAYERS).on_grid(frequency, severity, POINTS)
annual = dict(zip(NAMES, tower_grids.ceded))   # each layer's annual loss, a Grid
TAIL = Distortion.tvar(0.99)                   # the capital standard used until section 4

layer_stats = pd.DataFrame(
    [(name, 1 - g.cdf(0.0), g.mean(), math.sqrt(g.variance()), g.quantile(0.99), TAIL.measure(g), layer.aggregate_limit)
     for layer, (name, g) in zip(LAYERS, annual.items())],
    columns=["layer", "p_loss", "mean", "sd", "var", "tvar", "max"])
(GT(layer_stats, rowname_col="layer")
 .tab_header("The annual loss of each layer", "Loss units per year")
 .fmt_percent("p_loss", decimals=1)
 .fmt_number("mean", decimals=3)
 .fmt_number(["sd", "var", "tvar"], decimals=2)
 .fmt_integer("max")
 .cols_label(p_loss="P(loss)", mean="exp loss", var="99% VaR", tvar="99% TVaR"))
```

```text
       P(loss)  exp loss    sd  99% VaR  99% TVaR  max
2x3      41.6%     0.686  1.05     4.00      4.00    4
5x5      19.5%     0.586  1.52     5.60      7.78   10
10x10     6.1%     0.337  1.64    10.00     10.49   20
```

The 10x10 layer has the smallest expected loss and the largest standard
deviation. The 99% TVaR of the 2x3 layer equals its maximum, because the layer
is full in more than 1% of years. `TAIL.measure(g)` is a method. A `Distortion`
measures a `Grid`, a `Sampled` (simulated draws) or a `PredictiveDistribution`
(its total).

Every method below starts from one target: a 10% return on 99% TVaR capital for
the 5x5 layer. The question is what each method then charges the other two.

```python
mid = annual["5x5"]
RETURN = 0.10
target = price(mid, TAIL, cost_of_capital=RETURN)
print(f"5x5 target: risk premium {target.premium:.3f} = loss {target.expected_loss:.3f} + margin {target.margin:.3f}, "
      f"capital {target.capital:.3f}, loss ratio {target.loss_ratio:.1%}")
same = Pentagon.solve(loss=mid.mean(), assets=TAIL.measure(mid), return_on_capital=RETURN)
print(f"the same pentagon from three of its sides: {math.isclose(same.premium, target.premium)}")
```

```text
5x5 target: risk premium 1.240 = loss 0.586 + margin 0.654, capital 6.538, loss ratio 47.3%
the same pentagon from three of its sides: True
```

`price(losses, assets, cost_of_capital=r)` takes the capital standard as a
distortion (here the 99% TVaR). The premium is (E[X] + r a) / (1 + r), so the
margin is r times the capital. `Pentagon.solve` takes any three independent
quantities by keyword (section 4). The results are properties (`target.premium`,
not `target.premium()`).

## 2. A simple load, or a premium principle

A premium principle is a formula that gives a premium from the loss
distribution and one loading t. `classical_premium(principle, x, loading)` has
these principles:

- `expected_value`: (1 + t) E[X].
- `standard_deviation`: E[X] + t sd(X).
- `variance`: E[X] + t Var(X).
- `semi_variance`: the same with the squared excess over the mean.
- `dutch` and `fischer`: the mean excess over the mean, and its power mean with
  power `q`.
- `exponential`: log E[exp(t X)] / t.
- `esscher`: E[X exp(t X)] / E[exp(t X)].
- `var`: the lower t quantile.

`calibrate_classical(principle, x, premium)` finds the loading for a target
premium. The table calibrates each principle to the 5x5 premium and prices all
three layers, as a multiple of expected loss:

```python
def bisect(f, lo, hi, want, steps=60):
    """The x in [lo, hi] where the increasing function f reaches want."""
    for _ in range(steps):
        middle = (lo + hi) / 2
        lo, hi = (middle, hi) if f(middle) < want else (lo, middle)
    return (lo + hi) / 2


PRINCIPLES = ["expected_value", "standard_deviation", "variance", "semi_variance", "dutch", "fischer", "esscher"]
loadings = {p: calibrate_classical(p, mid, target.premium) for p in PRINCIPLES}
loadings["exponential"] = bisect(lambda t: classical_premium("exponential", mid, t), 0.01, 3.0, target.premium)

rows = [(principle.replace("_", " "), t, *[classical_premium(principle, g, t) / g.mean() for g in annual.values()])
        for principle, t in loadings.items()]
(GT(pd.DataFrame(rows, columns=["principle", "loading", *NAMES]), rowname_col="principle")
 .tab_header("Price / expected loss", "Each principle calibrated to the 5x5 premium")
 .fmt_number("loading", decimals=3)
 .fmt_number(NAMES, decimals=2))
```

```text
                    loading   2x3   5x5  10x10
expected value        1.115  2.12  2.12   2.12
standard deviation    0.430  1.66  2.12   3.09
variance              0.283  1.45  2.12   3.25
semi variance         0.322  1.38  2.12   3.45
dutch                 1.362  1.86  2.12   2.28
fischer               0.458  1.60  2.12   3.18
esscher               0.185  1.34  2.12   4.80
exponential           0.333  1.31  2.12   5.66
```

`calibrate_classical("exponential", ...)` gives the same loading as `bisect`.
To check a loading, price again with it:

```python
for principle in ("exponential", "standard_deviation"):
    t = calibrate_classical(principle, mid, target.premium)
    print(f"{principle:>20}: loading {t:.3g}, gives the target premium: "
          f"{math.isclose(classical_premium(principle, mid, t), target.premium, rel_tol=1e-6)}")
```

```text
         exponential: loading 0.333, gives the target premium: True
  standard_deviation: loading 0.43, gives the target premium: True
```

The expected value principle cannot tell a working layer from a top layer. The
others charge the top layer more, because it is more volatile for its mean.
Esscher and exponential, which weight outcomes by exp(t x), go furthest.

**Use a principle when** you need a quick, transparent load, the covers have
about the same shape and the variance describes the risk (a large quota share, a
stop loss on a diversified book). A principle also checks a richer model.

**Do not use one when the tail is heavy.** The uncapped severity here has an
infinite variance, so variance-based principles depend on where you cap the
loss:

```python
print(f"uncapped Pareto variance {Pareto(1.0, 1.8).variance()}, "
      f"severity capped at 20: standard deviation {math.sqrt(severity.variance()):.2f}")
```

```text
uncapped Pareto variance inf, severity capped at 20: standard deviation 2.16
```

Esscher and exponential need E[exp(t X)], infinite for a Pareto tail at any
t > 0. They exist only for a capped loss, and the cap sets their price.

**Do not use one to price a program layer by layer.** The prices of the pieces
of one loss must add up to the price of the whole. The table splits the year's
total large loss at 10 and prices the whole and the two parts under each rule:

```python
gross, _ = fft(frequency, severity, POINTS)            # the year's total of all large losses
lower, _ = gross.map(lambda x: min(x, 10.0))           # the first 10 of the year's total
upper, _ = gross.map(lambda x: max(x - 10.0, 0.0))     # everything above 10
ph = calibrate("ph", mid, target.premium)              # section 3 explains this call
rules = {"expected value": lambda g: classical_premium("expected_value", g, loadings["expected_value"]),
         "standard deviation": lambda g: classical_premium("standard_deviation", g, loadings["standard_deviation"]),
         "variance": lambda g: classical_premium("variance", g, loadings["variance"]),
         "proportional hazard": ph.measure}
rows = [(name, rule(gross), rule(lower) + rule(upper)) for name, rule in rules.items()]
(GT(pd.DataFrame(rows, columns=["rule", "whole", "parts"]), rowname_col="rule")
 .tab_header("Premium of the year's large losses", "The whole, and the sum of its two parts")
 .fmt_number(["whole", "parts"], decimals=3)
 .cols_label(parts="two parts"))
```

```text
                      whole  two parts
expected value       18.066     18.066
standard deviation   11.150     11.592
variance             18.949     15.826
proportional hazard  11.241     11.241
```

The two parts move together (a bad year is bad in both), so a sound price of
the whole equals the sum of the parts. A distortion gives that result, and so
does the expected value principle, because it loads only the mean. Standard
deviation and variance do not, so a reinsurer that uses them can get one risk
offered in pieces at a different total.

For different risks (two lines of a book), a distortion price of the sum is at
most the sum of the prices (the diversification benefit of section 6).

A loading of 0.43 on a standard deviation and 0.185 on an Esscher exponent are
not comparable. State the principle with the loading, and calibrate again when
the cover changes.

## 3. Distortions

A distortion g is a concave function with g(0) = 0 and g(1) = 1. Because g is
concave, the price is at least the expected loss. The same g weights every size,
so the prices of the pieces add up (the test above). Each `Distortion`
constructor is a family. `calibrate(name, x, premium)` returns the member that
prices `x` at `premium`:

| Family | Constructor | `calibrate` name | Shape of the load |
|---|---|---|---|
| Constant cost of capital | `ccoc(r)` | `"ccoc"` | Flat return on capital. It jumps at zero, so every layer pays a floor |
| Proportional hazard | `proportional_hazard(rho)` | `"ph"` | g(s) = s to the power rho. The weight grows as rho falls |
| Wang | `wang(lam)` | `"wang"` | Normal shift: less weight far out than PH |
| Dual power | `dual_power(beta)` | `"dual"` | Weights the middle up, little extra far out |
| TVaR | `tvar(p)` | `"tvar"` | A multiple of the average of the worst 1 - p of outcomes |
| Exponential | `exponential(k)` | `"exp"` | Risk aversion that rises exponentially (not the classical `exponential` principle) |
| With a floor `r0` | `capped_linear(r0, slope)`, `capped_log_linear(r0, b)`, `lep(r0, r)`, `linear_yield(r0, r)` | `"clin"`, `"cll"`, `"lep"`, `"ly"` | A minimum rate on line |
| Your own shape | `beta`, `mixture`, `minimum`, `bitvar`, `weighted_tvar`, `convex(points)` | none | Build it |

The table calibrates the main families to the 5x5 target. g(s) is the rate on
line of a thin layer that a loss reaches once in 1/s years. The last column,
`d.mass` (a property), is the **floor**: the rate on line of the most remote
layer.

```python
import re

FAMILIES = {"ccoc": "ccoc", "ph": "ph", "wang": "wang", "dual": "dual", "tvar": "tvar", "exponential": "exp"}
fitted = {name: calibrate(family, mid, target.premium) for name, family in FAMILIES.items()}


def label(d, places=3):
    """A distortion's family and parameters, rounded (a floor r0 first, as the constructor takes it)."""
    values = [d.parameters[k] for k in sorted(d.parameters, key=lambda k: k != "r0")]
    return f"{d.family}(" + ", ".join(f"{v:.{places}f}" for v in values) + ")"


ODDS = (2, 10, 50, 100, 1000)
shapes = pd.DataFrame([(label(d), *[d.g(1 / n) for n in ODDS], d.mass) for d in fitted.values()],
                      columns=["family", *[f"1 in {n}" for n in ODDS], "floor"])
(GT(shapes, rowname_col="family")
 .tab_header("g(s) of six families", "Each priced to 5x5 = 1.24; the rate on line at each return period")
 .fmt_percent(shapes.columns[1:].tolist(), decimals=1))
```

```text
                            1 in 2  1 in 10  1 in 50  1 in 100  1 in 1000  floor
ccoc(0.075)                  53.5%    16.3%     8.8%      7.9%       7.0%   6.9%
proportional_hazard(0.686)   62.2%    20.6%     6.8%      4.3%       0.9%   0.0%
wang(0.482)                  68.5%    21.2%     5.8%      3.3%       0.5%   0.0%
dual_power(2.286)            79.5%    21.4%     4.5%      2.3%       0.2%   0.0%
tvar(0.527)                 100.0%    21.2%     4.2%      2.1%       0.2%   0.0%
exponential(2.088)           74.0%    21.5%     4.7%      2.4%       0.2%   0.0%
```

```python
import matplotlib.pyplot as plt
import numpy as np

odds = np.logspace(0, 3, 60)                    # 1 in 1 to 1 in 1000
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.0, 3.6))
for ax, names in zip(axes, [["ccoc", "ph", "wang"], ["dual", "tvar", "exponential"]]):
    for name in names:
        ax.plot(odds, [fitted[name].g(1 / n) for n in odds], label=label(fitted[name]))
    ax.set_xscale("log")
    ax.set_xlabel("return period of the loss (years)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("g(s), the rate on line")
fig.suptitle("g(s) of six families, each priced to 5x5 = 1.24")
fig.tight_layout()
```

Near the 5x5 layer (reached in 19.5% of years, close to one in ten) the families
agree within a few points, CCoC aside. Away from it they diverge. CCoC has a
floor because it charges every layer for its maximum loss as capital. The other
floors are zero.

The next table prices all three layers with each distortion, as a multiple of
expected loss and as the return on the layer's 99% TVaR capital.
`price(x, assets, distortion=d)` gives the premium `d.measure(x)` and the return
at the capital standard you choose (here `TAIL`). `tower_grids.price(d)` prices
every layer of a tower in one call and agrees with `d.measure(g)`; the code
checks this.

```python
for name, d in fitted.items():
    for p, g in zip(tower_grids.price(d).ceded, annual.values()):
        assert abs(p.premium - d.measure(g)) < 1e-9
rows = []
for name, d in fitted.items():
    priced = {n: price(g, TAIL, distortion=d) for n, g in annual.items()}
    rows.append((name, *[priced[n].premium / priced[n].expected_loss for n in NAMES],
                 *[priced[n].return_on_capital for n in NAMES]))
columns = ["family", *[f"m{n}" for n in NAMES], *[f"r{n}" for n in NAMES]]
(GT(pd.DataFrame(rows, columns=columns), rowname_col="family")
 .tab_header("Each distortion priced to the 5x5 layer", "Price / expected loss, and return on 99% TVaR capital")
 .tab_spanner("price / expected loss", columns[1:4])
 .tab_spanner("return on capital", columns[4:])
 .fmt_number(columns[1:4], decimals=2)
 .fmt_percent(columns[4:], decimals=1)
 .cols_label(**{c: n for c, n in zip(columns[1:], NAMES + NAMES)}))
```

```text
              2x3   5x5  10x10    2x3    5x5  10x10
ccoc         1.34  2.12   5.05   7.5%  10.0%  15.5%
ph           1.61  2.12   2.99  14.5%  10.0%   7.1%
wang         1.72  2.12   2.65  17.5%  10.0%   5.8%
dual         1.90  2.12   2.23  22.8%  10.0%   4.3%
tvar         2.12  2.12   2.12  30.0%  10.0%   3.8%
exponential  1.83  2.12   2.29  20.7%  10.0%   4.5%
```

How each distortion weights the tail:

- **TVaR** charges every layer the same multiple. Here `tvar(p)` is a pricing
  shape, not a capital level: g(s) = s / (1 - p) below 1 - p. The fitted level
  is 0.527 and every layer is reached with a probability below 47.3%, so only
  the straight part applies and the cover pays 1 / (1 - p) times its expected
  loss, with no charge for a remote tail. It is a poor choice for remote
  layers.
- **Dual power**, **exponential** and **Wang** weight the middle outcomes up and
  the remote ones down. They undercharge remote layers relative to a constant
  return.
- **Proportional hazard** sits between.
- **CCoC** prices capital at the full maximum, far above the TVaR of a remote
  layer, so the remote tail is the most expensive part.

The families agree only near the layer they are calibrated to. Below it (2x3)
and above it (10x10) the multiples diverge. Calibrate to a layer like the one
you price. Far from the calibration point the family sets the price, and that
needs a reason: a market that it matches (section 5) or a view of what remote
capital costs.

**Use a distortion when** you need prices for several layers, or for a book and
its parts, that agree with each other, or must defend one price of risk across a
program. **Do not use one** when there is one cover and one quote: a load on
expected loss gives the same answer with less work. **Pitfalls:**

- `calibrate` with no `assets` takes the maximum of the distribution as the
  capital for CCoC. Here that is the full annual limit, so the fitted `ccoc` is
  a return on full-limit capital; the same premium is a 10% return on 99% TVaR
  capital. State which capital a CCoC rate refers to.
- A premium must lie strictly between the (capped) mean and the maximum, and
  that is not sufficient for every family. A floor `r0` sets the smallest
  premium the family can reach. LEP cannot reach a premium near the maximum even
  with no floor. Outside its range `calibrate` raises `ValueError` ("no <family>
  distortion gives (or reaches) the premium", example below).
- Do not carry a parameter across books. The hazard parameter is 0.686 for this
  layer and 0.690 for the book of section 6. Each is the price of risk for one
  cover and one capital standard, not a property of the market.

```python
for r0, premium in [(0.0, target.premium), (0.10, target.premium), (0.0, 5.0)]:
    try:
        print(f"lep, r0 {r0:.2f}, premium {premium:.2f}: {label(calibrate('lep', mid, premium, r0=r0))}")
    except ValueError as error:
        print(f"lep, r0 {r0:.2f}, premium {premium:.2f}: ValueError, "
              + re.sub(r"\d+\.\d+", lambda m: f"{float(m.group()):.3f}", str(error)))
```

```text
lep, r0 0.00, premium 1.24: lep(0.000, 0.519)
lep, r0 0.10, premium 1.24: ValueError, invalid data: no lep distortion gives the premium 1.240 (closest 1.442)
lep, r0 0.00, premium 5.00: ValueError, invalid data: no lep distortion reaches the premium 5
```

**Other shapes.** The constructors without a `calibrate` name take their
parameters from you: `beta(a, b)` (two parameters, to match two quotes without
LEP's floor), `mixture(ds, weights)` (a CCoC floor with a smooth family) and
`minimum(ds)` (the lowest). Set parameters with a search as in section 5 and
price with `d.measure(x)`.

## 4. The capital standard

The price depends on how much the reinsurer must hold. The usual standards are:

- The **VaR(p)** of the annual loss. It ignores how large the loss is beyond
  that quantile.
- The **TVaR(p)**. It sees the whole tail and is coherent, so a diversified book
  never needs more than the sum of its parts.
- A target **expected policyholder deficit (EPD)** ratio: the expected loss that
  the assets do not cover, as a fraction of the expected loss. The assets are
  the smallest that meet the target.
- The **full limit**, which CCoC assumes without `assets`.

```python
pf = Portfolio.from_independent(["5x5"], [mid])
standards = [("99% VaR", pf.assets(0.99)), ("99.5% VaR", pf.assets(0.995)),
             ("99% TVaR", TAIL.measure(mid)), ("99.5% TVaR", Distortion.tvar(0.995).measure(mid)),
             ("EPD ratio 1%", pf.assets_for_epd(0.01)), ("full limit", 10.0)]
rows = [(name, assets, max(pf.epd(assets)[0], 0.0),
         Pentagon.solve(loss=mid.mean(), assets=assets, return_on_capital=RETURN).premium)
        for name, assets in standards]
(GT(pd.DataFrame(rows, columns=["standard", "assets", "epd", "premium"]), rowname_col="standard")
 .tab_header("The 5x5 risk premium by capital standard", "At a 10% return on capital")
 .fmt_number("assets", decimals=2)
 .fmt_percent("epd", decimals=2)
 .fmt_number("premium", decimals=3)
 .cols_label(epd="EPD ratio", premium="risk premium"))
```

```text
              assets  EPD ratio  risk premium
99% VaR         5.60      3.71%         1.042
99.5% VaR       7.50      1.40%         1.215
99% TVaR        7.78      1.18%         1.240
99.5% TVaR      9.14      0.35%         1.364
EPD ratio 1%    8.03      1.00%         1.263
full limit     10.00      0.00%         1.442
```

The capital standard alone moves the risk premium by more than a third. From the
99% VaR to the 99% TVaR it rises by about a fifth.

`Portfolio.assets(p)` is a VaR. For a TVaR, use `Distortion.tvar(p).measure(x)`.
`Portfolio.epd(a)` returns the total EPD ratio and the ratio by unit, and
`assets_for_epd(e)` inverts it.

**The choice is a modelling choice.** No level is the right one. Regulatory
regimes use both VaR and TVaR, at different levels. Use the standard the
reinsurer is held to and keep it fixed when you compare prices. Prefer TVaR for
a layer with a lump of outcomes at its limit, as here: a VaR jumps when the
probability of a full-limit loss crosses 1 - p, a TVaR does not.

## 5. Calibrating to a target return, to a price, or to several layers

`Portfolio.calibrate` accepts any one of three targets (`premium`,
`return_on_capital` or `loss_ratio`) at a capital standard (`assets=` or `p=`):

```python
A = TAIL.measure(mid)
targets = {"premium": dict(premium=target.premium),
           "return on capital": dict(return_on_capital=RETURN),
           "loss ratio": dict(loss_ratio=target.loss_ratio)}
rows = []
for how, kw in targets.items():
    d = pf.calibrate("ph", assets=A, **kw)
    p = pf.price(d, assets=A).total
    rows.append((how, label(d), p.premium, p.return_on_capital, p.loss_ratio))
(GT(pd.DataFrame(rows, columns=["target", "fit", "premium", "return", "loss_ratio"]), rowname_col="target")
 .tab_header("A proportional hazard fitted to each target", "The 5x5 layer at the 99% TVaR standard")
 .fmt_number("premium", decimals=3)
 .fmt_percent(["return", "loss_ratio"], decimals=1)
 .cols_label(loss_ratio="loss ratio"))
```

```text
                                          fit  premium  return  loss ratio
premium            proportional_hazard(0.671)    1.240   10.1%       46.7%
return on capital  proportional_hazard(0.673)    1.234   10.0%       47.0%
loss ratio         proportional_hazard(0.676)    1.225    9.9%       47.3%
```

The fits differ slightly from each other and from the 0.686 above, because
`Portfolio` prices the loss capped at the assets (the reinsurer cannot pay more
than it holds), so its expected loss is a little below the layer's. The free
function `calibrate(family, x, premium, assets=None)` takes only a premium.

Each target has its use:

- **A target return** sets a technical price.
- **A market price** shows the price of risk the market pays. Price other layers
  with it, or compare your own return.
- **A loss ratio** suits a quota share.

### More than one layer

One parameter matches one price. Suppose the market also quotes 1.30 (an
illustrative number) for the 10x10. A proportional hazard fitted to the 5x5
prices the 10x10 much lower. A family with a floor has a second parameter. To
match both quotes:

1. Fix the floor `r0`.
2. Let `calibrate` set the other parameter to hit the 5x5.
3. Search on `r0` until the price of the 10x10 matches its quote.

```python
top = annual["10x10"]
quote_top = 1.30      # an illustrative market quote for the 10x10 risk premium
one_parameter = calibrate("ph", mid, target.premium)
r0 = bisect(lambda r0: calibrate("lep", mid, target.premium, r0=r0).measure(top), 0.0, 0.05, quote_top)
two_parameter = calibrate("lep", mid, target.premium, r0=r0)
nan = float("nan")
rows = [("quote", nan, target.premium, quote_top, nan)]
rows += [(label(d, 4), *[d.measure(g) for g in annual.values()], d.mass) for d in (one_parameter, two_parameter)]
(GT(pd.DataFrame(rows, columns=["fit", *NAMES, "floor"]), rowname_col="fit")
 .tab_header("One and two parameters against two quotes", "Risk premium of each layer")
 .fmt_number(NAMES, decimals=3)
 .fmt_percent("floor", decimals=2)
 .sub_missing(missing_text=""))
```

```text
                               2x3    5x5  10x10  floor
quote                               1.240  1.300       
proportional_hazard(0.6855)  1.106  1.240  1.009  0.00%
lep(0.0297, 0.2962)          1.045  1.240  1.300  2.89%
```

The two-parameter fit hits both quotes. For LEP the floor `d.mass` is close to
`r0` (0.0297) but not equal to it. If the market also quotes the working layer,
its difference from the fit's 1.045 tests the family.

To fit more quotes than the family has parameters, minimise the squared
differences between `calibrate(...).measure(layer)` and the quotes over the free
parameter (for example `scipy.optimize.least_squares` on `r0`). The residuals
show whether the family fits the market.

For quotes on thin layers (as for catastrophe bonds), the concave hull of the
quotes is a distortion itself. Give `Distortion.convex` pairs
`(chance of being reached, rate on line)`:

```python
# Illustrative rates on line for thin layers of the year's total: (chance of being reached, rate on line).
quotes = [(0.20, 0.34), (0.10, 0.22), (0.05, 0.14), (0.02, 0.07), (0.01, 0.045), (0.004, 0.025)]
hull = Distortion.convex(quotes)
print(f"g at 1 in 500: {hull.g(1 / 500):.2%}, g at 1 in 5000: {hull.g(1 / 5000):.2%}, floor {hull.mass:.1%}")
print({name: round(hull.measure(g), 3) for name, g in annual.items()})
```

```text
g at 1 in 500: 1.25%, g at 1 in 5000: 0.12%, floor 0.0%
{'2x3': 1.094, '5x5': 1.297, '10x10': 1.034}
```

Between quotes the hull is a straight line. Below the last quote it runs to the
origin, so it has no floor unless you add a quote at zero. It needs no family
assumption but has no information outside the quotes. It drops a quote inside
the hull with no warning.

## 6. Stand-alone price, price in a portfolio, and allocation

A contract priced alone ignores the rest of the reinsurer's book. If its bad
years rarely coincide with the book's, it adds little to the book's tail and
needs less capital than alone. The difference is the **diversification
benefit**. The code builds a small book of three lines joined by a Gaussian
copula (see [the severity and frequency guide](severity-and-frequency.md)) as a
`PredictiveDistribution` of 20,000 simulated years, one column per line:

```python
LINES = ["property", "casualty", "cat"]
margins = [Lognormal.from_mean_cv(20.0, 0.3), Lognormal.from_mean_cv(30.0, 0.2), Lognormal.from_mean_cv(10.0, 1.0)]
copula = GaussianCopula([[1.0, 0.3, 0.5], [0.3, 1.0, 0.0], [0.5, 0.0, 1.0]])
rows = [[m.quantile(u) for m, u in zip(margins, row)] for row in copula.sample(20_000, 7)]
book = PredictiveDistribution(["line"], [(n,) for n in LINES], rows)

alone = capital(book, TAIL).standalone
rows = [("stand-alone", *alone, sum(alone), float("nan"))]
for method in ["euler", "covariance", "proportional", "marginal", "shapley"]:
    a = capital(book, TAIL, method)
    rows.append((method, *a.allocated, sum(a.allocated), a.total))
(GT(pd.DataFrame(rows, columns=["method", *LINES, "sum", "total"]), rowname_col="method")
 .tab_header("The 99% TVaR of the book split among its lines", "Assets: premium plus capital")
 .fmt_number([*LINES, "sum", "total"], decimals=2)
 .sub_missing(missing_text=""))
```

```text
              property  casualty    cat     sum   total
stand-alone      41.63     49.81  64.23  155.68        
euler            34.18     33.85  59.75  127.79  127.79
covariance       38.70     25.36  63.73  127.79  127.79
proportional     34.17     40.89  52.73  127.79  127.79
marginal         32.39     32.10  46.44  110.93  127.79
shapley          35.17     39.12  53.50  127.79  127.79
```

`capital(pd, distortion, method)` allocates the distortion measure of the total
(here the 99% TVaR) to the lines. The amounts are **assets** (premium plus
capital), because the capital standard sets the assets. The stand-alone amounts
sum to far more than the book needs; the difference is the diversification
benefit. The five methods answer different questions:

| Method | Answers | Adds up? |
|---|---|---|
| `euler` (the default, also `allocate(pd, d)`) | What does each line contribute in the bad years of the book? For TVaR, the average loss of each line in the worst 1% of the years of the book | Yes |
| `covariance` | Who moves with the total, over the whole distribution and not only the tail | Yes |
| `proportional` | Stand-alone measures scaled to the total. It ignores dependence | Yes |
| `marginal` | What would the book save without the line (Merton and Perold)? | No: it falls short, and the shortfall belongs to no line |
| `shapley` | Average marginal contribution over every order of joining (at most 12 lines) | Yes |

**Euler is the usual choice.** Only Euler is the gradient of the risk measure,
so it is consistent with small changes to the portfolio: it says whether growth
in a line makes the book's assets grow faster or slower than the line's
stand-alone assets. For a distortion it is the natural (co-measure) allocation
that `price_portfolio` and `Portfolio.price` use below. Casualty diversifies,
and cat drives the tail of the book.

Use `marginal` for the one-off question "what is this contract worth to the
book", not to split the book.

`price_portfolio` prices the lines, not only the capital. It applies a cost of
capital to the assets and allocates both by co-measure (the natural allocation,
the same split as `euler`), so the component prices sum to the portfolio price:

```python
priced = price_portfolio(book, TAIL, cost_of_capital=RETURN)
rows = [(name, s.expected_loss, s.premium, c.premium, s.capital, c.capital)
        for name, s, c in zip(LINES, priced.standalone, priced.allocated)]
rows.append(("book", priced.total.expected_loss, sum(s.premium for s in priced.standalone), priced.total.premium,
             sum(s.capital for s in priced.standalone), priced.total.capital))
(GT(pd.DataFrame(rows, columns=["line", "loss", "p_alone", "p_book", "c_alone", "c_book"]), rowname_col="line")
 .tab_header("Premium and capital of each line", "Priced alone and in the book, at a 10% return")
 .tab_spanner("premium", ["p_alone", "p_book"])
 .tab_spanner("capital", ["c_alone", "c_book"])
 .fmt_number(["loss", "p_alone", "p_book", "c_alone", "c_book"], decimals=2)
 .cols_label(loss="exp loss", p_alone="alone", p_book="in book", c_alone="alone", c_book="in book"))
```

```text
          exp loss  alone  in book  alone  in book
property     20.01  21.98    21.30  19.66    12.88
casualty     29.99  31.79    30.34  18.02     3.51
cat           9.92  14.86    14.45  49.38    45.30
book         59.92  68.63    66.09  87.05    61.70
```

```python
print(f"premium saved by writing the lines together: {priced.diversification():.2f}")
```

```text
premium saved by writing the lines together: 2.54
```

```python
shown = pd.DataFrame({"stand-alone": [x.capital for x in priced.standalone],
                      "in the book": [x.capital for x in priced.allocated]}, index=LINES)
ax = shown.plot.bar(rot=0)
ax.set_xlabel("line")
ax.set_ylabel("capital (99% TVaR standard)")
ax.set_title("Capital of each line, alone and in the book")
```

In the book, premium and capital add up to the Euler assets above
(66.09 + 61.70 = 127.79). Each line earns 10% on its capital, so cat carries the
most margin.

Capital falls by 25.35 in the book, but the premium by only 2.54, about a tenth
of that. `priced.diversification()` returns the premium saved. The components
must add up to the portfolio (segments or covers), never gross, ceded and net
side by side.

`price_portfolio` uses a constant cost of capital. To price a book with a
distortion calibrated to a return or a quote, use `Portfolio`: for each total it
keeps the average loss of each unit given that total. `Portfolio.price` prices
the loss capped at the assets (here the same 99% TVaR) and splits loss, margin,
premium, capital and assets among the units.

The part above the assets has two splits. `allocation="linear"` uses each unit's
expected share of the total loss, `"lifted"` its distorted share. Bodoff's
percentile layer method, `port.bodoff(p=0.99)`, is a third split of the assets:

```python
port = Portfolio.from_predictive(book)
A_book = TAIL.measure(book)                    # the book's 99% TVaR, as for price_portfolio
d_book = port.calibrate("ph", assets=A_book, return_on_capital=RETURN)
print(label(d_book), f"assets {A_book:.2f}")
linear = port.price(d_book, assets=A_book)
lifted = port.price(d_book, assets=A_book, allocation="lifted")
bodoff = port.bodoff(assets=A_book)
```

```text
proportional_hazard(0.690) assets 127.79
```

```python
rows = [(name, linear.allocated[i].premium, lifted.allocated[i].premium,
         linear.allocated[i].return_on_capital, lifted.allocated[i].return_on_capital,
         linear.allocated[i].assets, lifted.allocated[i].assets, bodoff[i])
        for i, name in enumerate(LINES)]
rows.append(("total", linear.total.premium, lifted.total.premium, linear.total.return_on_capital,
             lifted.total.return_on_capital, linear.total.assets, lifted.total.assets, sum(bodoff)))
columns = ["line", "p_lin", "p_lift", "r_lin", "r_lift", "a_lin", "a_lift", "a_bodoff"]
(GT(pd.DataFrame(rows, columns=columns), rowname_col="line")
 .tab_header("Three splits of the book", "Calibrated proportional hazard, 99% TVaR assets")
 .tab_spanner("premium", ["p_lin", "p_lift"])
 .tab_spanner("return", ["r_lin", "r_lift"])
 .tab_spanner("assets", ["a_lin", "a_lift", "a_bodoff"])
 .fmt_number(["p_lin", "p_lift", "a_lin", "a_lift", "a_bodoff"], decimals=2)
 .fmt_percent(["r_lin", "r_lift"], decimals=1)
 .cols_label(p_lin="linear", p_lift="lifted", r_lin="linear", r_lift="lifted",
             a_lin="linear", a_lift="lifted", a_bodoff="Bodoff"))
```

```text
          linear  lifted  linear  lifted  linear  lifted  Bodoff
property   21.71   21.68   12.0%   10.1%   36.00   38.45   40.10
casualty   30.88   30.85    9.6%    5.7%   40.27   46.18   52.90
cat        13.43   13.49    9.3%   12.2%   51.52   43.15   34.79
total      66.02   66.02   10.0%   10.0%  127.79  127.79  127.79
```

The book premium is 66.02 with the calibrated distortion and 66.09 with a
constant cost of capital in `price_portfolio`. The premium by line is almost the
same under both allocations, and the book earns exactly 10% under both. The
returns and assets by line differ.

Bodoff gives cat the smallest part of the assets and casualty the largest. No
split is the right one: each says who owns each slice of the asset stack in the
bad years, and that is a convention. Choose one, state it, and do not compare
returns by line across conventions. `port.price(...)` returns a `NaturalPrice`;
`.to_dict()` gives its table as columns for `pandas.DataFrame`.

**Use** `price_portfolio` for the technical price of a contract inside a book at
the return the reinsurer asks, `Portfolio.price` for a book priced with a
distortion calibrated to a quote, and `capital` (Euler) to split capital among
lines without a price.

**Pitfalls:** the diversification benefit depends on the dependence model. A
Gaussian copula has no tail dependence and can understate how often lines have
their worst years together; test with a t or Archimedean copula. An in-book
price is a price for this reinsurer's book, so a cedant cannot use it to compare
reinsurers.

## 7. Which family for which market

This table is judgment, not theorem. Check it against your own quotes.

| Market | What matters | Reasonable starting point | Why |
|---|---|---|---|
| Working layers, casualty: frequent losses | Most of the price is in the body | A smooth family (PH or Wang) or CCoC on a TVaR standard, calibrated to a layer of similar attachment to the one you price | The families agree only near the layer they are calibrated to (section 3). The family matters for any layer far from it |
| Top casualty layers, long tail | Parameter and reserve uncertainty dominates | PH or Wang calibrated to the quotes you have, with a loss model whose tail reflects the uncertainty | The family matters from one in fifty out, but model uncertainty is larger |
| Property per risk | A mix of both ends | PH or Wang calibrated to two layers | One parameter will miss one end |
| Catastrophe: remote layers | Rate on line has a floor. Prices come from quotes more than from models | A family with a floor (LEP, capped linear, CCoC) or `Distortion.convex` of the quotes | PH, Wang and dual power fall toward zero, but markets rarely quote below a minimum rate on line |
| Quota share, stop loss, whole-book total | A share of a roughly symmetric total | Standard deviation, or CCoC on a TVaR standard | Layer-by-layer consistency matters less. Volatility describes the risk |

### What is a modelling choice

These choices do not come from the data. Make them on purpose, write them down
and test how much each moves the price.

- The **capital standard** (VaR or TVaR, its level, or an EPD target). Section 4
  moved the premium of one layer by more than a third with this choice alone.
- The **return on capital**, and whether it is the same for every layer.
- The **distortion family**. It sets the price of every layer that you have no
  quote for, and it matters most for remote layers.
- The **dependence** between the contract and the book, and between lines.
- The **allocation method**, which changes the return by line, not the total.
- Pricing **stand-alone** or **in the book**. The stand-alone price is the
  reference. The in-book price is what this reinsurer can accept.
- **Time value**. The risk premium here is on undiscounted loss, with the
  capital held for one year. `PaymentPattern` and
  `price_over_payout(price, pattern, rate=..., cost_of_capital=...)` restate it
  over the payout; give `distortion=...`, the rule that made the price, in place
  of `cost_of_capital`. The cost of capital is then a spread over the discount
  rate, because the discount on the losses is the investment income. The capital
  runs off with the share of the loss still outstanding.
- **Model and parameter uncertainty**. A load for not knowing the distribution
  is different from a load for its spread. Build it into the loss model (a
  heavier tail, a mixed parameter) or load it separately, and say which.

These are not choices: a distortion price is at least the expected loss, the
prices of the pieces of one loss add up to the whole, and the premium and
capital of the units from `price_portfolio`, `Portfolio.price` and `capital`
(except `marginal`) add up to the total.

## 8. Pitfalls in one place

- **Properties and methods.** Properties (no parentheses): on a `Distortion`,
  `mass`, `family`, `parameters`; on a `Pentagon`, `loss`, `margin`, `premium`,
  `capital`, `assets`, `loss_ratio`, `return_on_capital` (a `Price` has the same,
  with the loss named `expected_loss`); on an `Allocation` and a
  `PortfolioPrice`, `allocated`, `standalone`, `total`; on a `Portfolio`,
  `totals`, `probs`, `units`. Methods: `g(s)`, `g_inv`, `g_dual`, `measure(x)`,
  `weights(n)`, `diversification_benefit()`, `diversification()`, `assets(p)`,
  `epd(a)`, `bodoff(...)`.
- **Price the annual loss of the cover**, after reinstatements and the annual
  limit, not a single loss. Pricing each occurrence and adding the prices counts
  the load twice.
- **Risk premium, not technical premium.** `Quote.technical(price, expenses=,
  brokerage=, limit=)` adds expenses and brokerage to the risk premium (see [the
  price guide](reinsurance-price.md)).
- **Few draws, noisy tail.** A 99% TVaR from 20,000 draws rests on 200 of them.
  Use a grid and `fft` where you can, or more draws and a fixed seed.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): the map across all methods.
- [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md): the
  source of the loss distribution.
- [`aggregate-distributions.md`](aggregate-distributions.md): the annual loss
  distribution.
- [`severity-and-frequency.md`](severity-and-frequency.md): the inputs, and
  dependence between them.
- [`reserving-methods.md`](reserving-methods.md): reserve risk in the capital
  standard.
- [`reinsurance-price.md`](reinsurance-price.md): the technical premium these
  methods feed.

Reference: Mildenhall and Major, *Pricing Insurance Risk* (Wiley, 2022), for
distortions, calibration and the natural allocation. See also the design note
`docs/design/risk.md`.

## Appendix: terms

- **Risk premium**: expected loss plus the margin for risk, before expenses and
  brokerage. This guide prices the risk premium.
- **Capital**, **assets**: capital is the money that the reinsurer holds behind
  the contract. Assets are premium plus capital. The **capital standard** sets
  the assets, for example "99% TVaR of the annual loss". **Return on capital**
  is margin divided by capital.
- **VaR** at level p is the p quantile of the loss. **TVaR** at level p is the
  average of the worst 1 - p of outcomes.
- **Distortion** g: turns a survival probability s (the probability that the
  loss exceeds a size) into a weight. The price of a loss is the integral of
  g(s(x)) over x. If g(s) = s, the price is the expected loss; a concave g above
  the diagonal weights the tail more than its probability. g(s) is the **rate on
  line** (premium over limit) of a thin layer that a loss reaches with
  probability s.
- **Pentagon**: loss, margin, premium, capital, assets, linked by
  premium = loss + margin and assets = premium + capital. Any three that are
  independent fix the other two.
