# Risk loads and distortions: when to use which

Use this guide when you have a loss distribution for a cover (an excess of loss
layer, a quota share, a whole book) and must turn it into a price with a load for
risk. A primary pricing actuary loads a premium by a percentage of the expected
loss. A reinsurer loads for the tail: the same expected loss deserves different
prices depending on how often and how badly the cover goes wrong. The guide
takes the choices in the order you meet them: a simple load, a premium
principle, a **distortion** (a reweighting of probabilities that prices a loss by
its tail), the capital standard behind it, calibration to a target or to market
quotes, and pricing and splitting a portfolio. It says which to use when, and
which parts are modelling choices that no calculation can settle.

It assumes you have the distribution of the cover's annual loss (see
[the aggregate guide](aggregate-distributions.md)) and know what a layer is (see
[the price guide](reinsurance-price.md)). Experience and exposure rating (see
[that guide](experience-and-exposure-rating.md)) give the expected loss and the
distribution; this guide adds the load for risk on top of them.

## Which method for which situation

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| A first number for a screen, nothing else known | A load on expected loss | `classical_premium("expected_value", x, t)` | Cheap, but blind to the tail. Fine only to compare covers of one shape |
| A stable, roughly symmetric result (a large quota share) where the load should follow volatility | Standard deviation or variance principle | `classical_premium("standard_deviation", x, t)`, `calibrate_classical` | Price rises with spread. Needs a finite variance, and layer prices do not add up (section 3) |
| "We need 10% on capital" | Constant cost of capital (CCoC) | `price(x, Distortion.tvar(p), cost_of_capital=r)`, `Pentagon.solve(..., assets=a)` or `price_portfolio(..., cost_of_capital=r)` for capital on a TVaR or EPD standard; `Distortion.ccoc(r)` for capital equal to the maximum loss | Every unit of capital earns the same return, the easiest rule to explain. Say which capital the 10% is on (section 4) |
| One quote or target, price the rest of a program consistently | One-parameter distortion (proportional hazard, Wang) calibrated to it | `calibrate("ph", x, premium)`, then `d.measure(y)` or `TowerGrids.price(d)` | Prices add up across layers, one number to defend |
| Remote layers (catastrophe) with a minimum rate on line | A distortion with a floor: LEP, capped linear, linear yield, CCoC | `calibrate("lep", x, premium, r0=0.03)` | A family without a floor prices a 1-in-1,000 layer at a fraction of a percent |
| Quotes on several layers | Fit two parameters, or take the concave hull of the quotes | `calibrate(..., r0=...)` with a search on `r0`, `Distortion.convex(points)` | One parameter cannot match both ends of a program |
| The capital standard is a tail measure | TVaR, or assets for a target expected policyholder deficit | `Distortion.tvar(p)`, `Portfolio.assets_for_epd(e)` | VaR ignores how bad the tail is beyond its level |
| A contract goes into a book | Price the portfolio, then allocate | `price_portfolio(pd, assets, cost_of_capital=r)`, `Portfolio.price(d, p=0.99)` | Capital depends on what else is in the book (section 7) |
| Split a book's capital among its parts | Euler, which for a distortion is the natural (co-measure) allocation; marginal, Shapley, Bodoff | `capital(pd, d)`, `allocate(pd, d)`, `Portfolio.price(..., allocation=)`, `Portfolio.bodoff` | They answer different questions (section 7) |
| Heavy tail, variance infinite or unreliable | A distortion, never a variance-based, Esscher or exponential principle | `calibrate("ph", ...)` | A distortion needs only the weighted mean of the tail |

## 1. Words you need first

- **Risk premium**: expected loss plus the margin for bearing risk, before
  expenses and brokerage. This guide prices the risk premium.
- **Capital**, **assets**: capital is the reinsurer's own money held behind the
  contract; assets are premium plus capital. The **capital standard** sets the
  assets, for example "99% TVaR of the annual loss". **Return on capital** is
  margin divided by capital.
- **VaR** at level p is the p quantile of the loss. **TVaR** at level p is the
  average of the worst 1 - p of outcomes.
- **Distortion** g: turns a survival probability s (the chance the loss exceeds
  a size) into a weight. The price of a loss is the integral of g(s(x)) over x.
  If g(s) = s the price is the expected loss; a concave g above the diagonal
  weights the tail more than its probability. Read g(s) as the **rate on line**
  (premium over limit) of a thin layer reached with probability s.
- **Pentagon**: loss, margin, premium, capital, assets, linked by
  premium = loss + margin and assets = premium + capital. Any three that are
  independent fix the other two.

## 2. The running example

The book is the one in [the price guide](reinsurance-price.md), on a coarser
grid: a Poisson count of 4 large losses a year, each with a Pareto tail (alpha
1.8) capped at a policy limit of 20. We price three layers, each with one free
reinstatement, so each pays at most twice its limit in a year. A distortion
prices the layer's **annual** loss, because that is what the reinsurer carries.

```python
import math

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
TAIL = Distortion.tvar(0.99)                   # the capital standard used until section 5

print(f"{'layer':>6} {'P(loss)':>8} {'exp loss':>9} {'sd':>6} {'99% VaR':>8} {'99% TVaR':>9} {'max':>5}")
for layer, (name, g) in zip(LAYERS, annual.items()):
    print(f"{name:>6} {1 - g.cdf(0.0):8.1%} {g.mean():9.3f} {math.sqrt(g.variance()):6.2f} "
          f"{g.quantile(0.99):8.2f} {TAIL.measure(g):9.2f} {layer.aggregate_limit:5.0f}")
```

```text
 layer  P(loss)  exp loss     sd  99% VaR  99% TVaR   max
   2x3    41.6%     0.686   1.05     4.00      4.00     4
   5x5    19.5%     0.586   1.52     5.60      7.78    10
 10x10     6.1%     0.337   1.64    10.00     10.49    20
```

The 10x10 layer pays in about one year in sixteen, has the smallest expected
loss and the largest standard deviation. For the 2x3 layer the 99% TVaR equals
the maximum, because that layer is full in more than 1% of years.
`TAIL.measure(g)` is a method: a `Distortion` measures a `Grid`, a `Sampled`
(simulated draws) or a `PredictiveDistribution` (its total).

Every method below does the same job. The reinsurer wants a 10% return on 99%
TVaR capital for the 5x5 layer. That fixes a risk premium for this layer, and the
question is what each method then charges the other two.

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
distortion (here the 99% TVaR) and charges the premium (E[X] + r a) / (1 + r),
so that the margin is r times the capital. It is the pentagon solved from the
loss, the assets and the return: `Pentagon.solve` takes any three independent
quantities by keyword, which section 5 needs when the assets are a VaR or an
EPD target rather than a distortion. On both, the results are properties
(`target.premium`, not `target.premium()`).

## 3. A simple load, or a premium principle

A premium principle is a formula for a premium from the loss distribution and
one loading t. `classical_premium(principle, x, loading)` offers `expected_value`
(1 + t) E[X]; `standard_deviation` E[X] + t sd(X); `variance` E[X] + t Var(X);
`semi_variance` (the same with the squared excess over the mean);
`dutch` and `fischer` (the mean excess over the mean, and its power mean, with
power `q`); `exponential` log E[exp(t X)] / t; `esscher` E[X exp(t X)] / E[exp(t X)];
and `var`, the lower t quantile. `calibrate_classical(principle, x, premium)`
finds the loading that gives a target premium. Calibrate each to the 5x5 premium
and read what it charges all three layers, as a multiple of expected loss:

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

print(f"{'principle':>20} {'loading':>8}" + "".join(f"{n:>8}" for n in NAMES) + "   (price / expected loss)")
for principle, t in loadings.items():
    multiples = [classical_premium(principle, g, t) / g.mean() for g in annual.values()]
    print(f"{principle:>20} {t:8.3f}" + "".join(f"{m:8.2f}" for m in multiples))
```

```text
           principle  loading     2x3     5x5   10x10   (price / expected loss)
      expected_value    1.115    2.12    2.12    2.12
  standard_deviation    0.430    1.66    2.12    3.09
            variance    0.283    1.45    2.12    3.25
       semi_variance    0.322    1.38    2.12    3.45
               dutch    1.362    1.86    2.12    2.28
             fischer    0.458    1.60    2.12    3.18
             esscher    0.185    1.34    2.12    4.80
         exponential    0.333    1.31    2.12    5.66
```

The exponential loading comes from the small `bisect` helper above, as a
workaround: `calibrate_classical("exponential", ...)` returns a loading of about
1e-31 for this layer, without an error, and that loading does not give the target
premium. Whatever the principle, check a calibrated loading by pricing with it
again:

```python
for principle in ("exponential", "standard_deviation"):
    t = calibrate_classical(principle, mid, target.premium)
    print(f"{principle:>20}: loading {t:.3g}, gives the target premium: "
          f"{math.isclose(classical_premium(principle, mid, t), target.premium, rel_tol=1e-6)}")
```

```text
         exponential: loading 4.25e-31, gives the target premium: False
  standard_deviation: loading 0.43, gives the target premium: True
```

The expected value principle charges every layer 2.12 times its
expected loss: it cannot tell a working layer from a top layer. The
variance-based principles charge the working layer less and the top layer more,
because the top layer is more volatile for its mean. Esscher and exponential go
furthest: they weight outcomes by exp(t x), so the top layer pays more than
three times as much per unit of expected loss as the working layer.

**Use a principle when** you need a quick, transparent load, every cover you
compare has about the same shape, and the loss is stable enough that its
variance describes its risk (a large quota share, a stop loss on a diversified
book). It is also a sanity check on a richer model.

**Do not use one when the tail is heavy.** The uncapped severity here has an
infinite variance, so every variance-based principle depends on where you cap
the loss:

```python
print(f"uncapped Pareto variance {Pareto(1.0, 1.8).variance()}, "
      f"severity capped at 20: standard deviation {math.sqrt(severity.variance()):.2f}")
```

```text
uncapped Pareto variance inf, severity capped at 20: standard deviation 2.16
```

Esscher and exponential are worse off: they need E[exp(t X)], which is infinite
for a Pareto tail at any positive t, so they exist only for a capped loss and
their price is set by the cap.

**Do not use one to price a program layer by layer.** A layer price is only
meaningful if the prices of the pieces of one loss (layers, or a retention and
the excess above it) add up to the price of the whole. Split
the year's total large loss at 10 into the part below and the part above, and
price the whole and the two parts under the same rule:

```python
gross, _ = fft(frequency, severity, POINTS)            # the year's total of all large losses
lower, _ = gross.map(lambda x: min(x, 10.0))           # the first 10 of the year's total
upper, _ = gross.map(lambda x: max(x - 10.0, 0.0))     # everything above 10
ph = calibrate("ph", mid, target.premium)              # section 4 explains this call
rules = {"expected value": lambda g: classical_premium("expected_value", g, loadings["expected_value"]),
         "standard deviation": lambda g: classical_premium("standard_deviation", g, loadings["standard_deviation"]),
         "variance": lambda g: classical_premium("variance", g, loadings["variance"]),
         "proportional hazard": ph.measure}
print(f"{'rule':>20} {'whole':>8} {'two parts':>10}")
for name, rule in rules.items():
    print(f"{name:>20} {rule(gross):8.3f} {rule(lower) + rule(upper):10.3f}")
```

```text
                rule    whole  two parts
      expected value   18.066     18.066
  standard deviation   11.150     11.592
            variance   18.949     15.826
 proportional hazard   11.241     11.241
```

The two parts move together (a bad year is bad in both), so a sound price of the
whole equals the sum. A distortion gives that: the proportional hazard prices of
the two parts add up to the whole to the last digit. The expected value principle
does too, because it loads the mean only. Standard deviation and variance do not,
so a reinsurer using them could be offered one risk in pieces and get a different
total. This holds for pieces of the same loss, which move together. For different
risks (two lines of a book) a distortion price of the sum is at most the sum of
the prices, and the shortfall is the diversification benefit of section 7. A loading's meaning also changes with the principle (0.43 of a standard
deviation and 0.185 on an Esscher exponent are not comparable), so state the
principle with the loading and recalibrate when the cover changes.

## 4. Distortions

A distortion g is a concave function with g(0) = 0 and g(1) = 1. Because it is
concave, the price is at least the expected loss; because the same g weights
every size, the price of a layer plus the price of the rest is the price of the
whole (the test above). Each `Distortion` constructor is a family, and
`calibrate(name, x, premium)` returns the member whose price of `x` is `premium`:

| Family | Constructor | `calibrate` name | Shape of the load |
|---|---|---|---|
| Constant cost of capital | `ccoc(r)` | `"ccoc"` | Flat return on capital; jumps at zero, so every layer pays a floor |
| Proportional hazard | `proportional_hazard(rho)` | `"ph"` | g(s) = s to the power rho; weight grows as rho falls |
| Wang | `wang(lam)` | `"wang"` | Normal shift: less weight far out than PH |
| Dual power | `dual_power(beta)` | `"dual"` | Weights the middle up, little extra far out |
| TVaR | `tvar(p)` | `"tvar"` | A multiple of the average of the worst 1 - p of outcomes |
| Exponential | `exponential(k)` | `"exp"` | Risk aversion rising exponentially (not the classical `exponential` principle) |
| With a floor `r0` | `capped_linear(r0, slope)`, `capped_log_linear(r0, b)`, `lep(r0, r)`, `linear_yield(r0, r)` | `"clin"`, `"cll"`, `"lep"`, `"ly"` | A minimum rate on line |
| Your own shape | `beta`, `mixture`, `minimum`, `bitvar`, `weighted_tvar`, `convex(points)` | none | Build it |

Calibrate the main families to the 5x5 target and read g(s) as the rate on line
of a thin layer reached once in 1/s years:

```python
import re

FAMILIES = {"ccoc": "ccoc", "ph": "ph", "wang": "wang", "dual": "dual", "tvar": "tvar", "exponential": "exp"}
fitted = {name: calibrate(family, mid, target.premium) for name, family in FAMILIES.items()}


def label(d, places=3):
    """A distortion's family and parameters, rounded (a floor r0 first, as the constructor takes it)."""
    values = [d.parameters[k] for k in sorted(d.parameters, key=lambda k: k != "r0")]
    return f"{d.family}(" + ", ".join(f"{v:.{places}f}" for v in values) + ")"


ODDS = (2, 10, 50, 100, 1000)
print(f"{'':>30}" + "".join(f"{'1 in ' + str(n):>10}" for n in ODDS) + f"{'floor':>9}")
for name, d in fitted.items():
    print(f"{label(d):>30}" + "".join(f"{d.g(1 / n):10.1%}" for n in ODDS) + f"{d.mass:9.1%}")
```

```text
                                  1 in 2   1 in 10   1 in 50  1 in 100 1 in 1000    floor
                   ccoc(0.075)     53.5%     16.3%      8.8%      7.9%      7.0%     6.9%
    proportional_hazard(0.686)     62.2%     20.6%      6.8%      4.3%      0.9%     0.0%
                   wang(0.482)     68.5%     21.2%      5.8%      3.3%      0.5%     0.0%
             dual_power(2.286)     79.5%     21.4%      4.5%      2.3%      0.2%     0.0%
                   tvar(0.527)    100.0%     21.2%      4.2%      2.1%      0.2%     0.0%
            exponential(2.088)     74.0%     21.5%      4.7%      2.4%      0.2%     0.0%
```

All six agree the 5x5 layer is worth 1.24, because that is what they were
calibrated to. At one in ten, near the 5x5 layer (reached in 19.5% of years), they
are within a few points of each other, CCoC aside. Away from it they part in both
directions: at one in two the rate on line runs from 53.5% to 100.0%, at one in a
hundred from 2.1% to 7.9%, and at one in a thousand from 0.2% to 7.0%. The last column, `d.mass` (a property, not a
method), is the **floor**: the rate on line of the remotest layer there is. CCoC
has one because it charges every layer for its maximum loss as capital; the
others fall to zero.

Now price all three layers with each distortion, as a multiple of expected loss,
and show the return that price earns on each layer's own 99% TVaR capital.
`price(x, assets, distortion=d)` gives the premium `d.measure(x)` and the return
it earns on the capital standard you choose, here `TAIL`.
(`tower_grids.price(d)` prices every layer of a tower in one call and agrees with
`d.measure(g)`; the code checks it.)

```python
for name, d in fitted.items():
    for p, g in zip(tower_grids.price(d).ceded, annual.values()):
        assert abs(p.premium - d.measure(g)) < 1e-9
print(f"{'':>12}" + "".join(f"{n:>8}" for n in NAMES) + "   return on 99% TVaR capital")
for name, d in fitted.items():
    priced = {n: price(g, TAIL, distortion=d) for n, g in annual.items()}
    print(f"{name:>12}" + "".join(f"{priced[n].premium / priced[n].expected_loss:8.2f}" for n in NAMES) + "   "
          + "  ".join(f"{priced[n].return_on_capital:6.1%}" for n in NAMES))
```

```text
                 2x3     5x5   10x10   return on 99% TVaR capital
        ccoc    1.34    2.12    5.05     7.5%   10.0%   15.5%
          ph    1.61    2.12    2.99    14.5%   10.0%    7.1%
        wang    1.72    2.12    2.65    17.5%   10.0%    5.8%
        dual    1.90    2.12    2.23    22.8%   10.0%    4.3%
        tvar    2.12    2.12    2.12    30.0%   10.0%    3.8%
 exponential    1.83    2.12    2.29    20.7%   10.0%    4.5%
```

**How each distortion weights the tail, and what that does to layer multiples.**

- **TVaR** charges every layer 2.12 times its expected loss, the same as the
  expected value principle. TVaR at level p is g(s) = s / (1 - p) while s is
  below 1 - p. The fitted level is 0.527, and every layer has a chance of any loss
  below 47.3% (section 2), so only the straight part of g is used. Here `tvar(p)`
  is a pricing shape, not a capital level like the 99% of section 2: a cover hit
  with probability below 1 - p pays 1 / (1 - p) times its expected loss. It is the
  expected value principle in disguise for such covers, with no extra charge for a
  remote tail, so it is a poor choice for pricing remote layers.
- **Dual power**, **exponential** and **Wang** weight the middle outcomes up and
  the remote ones less. For dual power the multiple climbs only from 1.90 to 2.23
  across the layers, and the return on each layer's own capital falls from 22.8%
  to 4.3%: they undercharge remote layers relative to a constant return.
- **Proportional hazard** sits between: a multiple from 1.61 to 2.99, and returns
  from 14.5% to 7.1%.
- **CCoC** charges the top layer 5.05 times its expected loss, and its return on
  99% TVaR capital rises from 7.5% on the working layer to 15.5% on the top,
  because it prices capital at the full maximum, which is far above the TVaR of a
  remote layer. It treats the remote tail as the most expensive part.

This is the main practical result. The families agree only close to the layer
they were calibrated to. Below it, on the working 2x3 layer, the multiple runs
from 1.34 to 2.12 and the return from 7.5% to 30.0%; above it, on the 10x10, from
2.12 to 5.05. So calibrate to a layer like the one you price, and for any layer
far from your calibration point the family decides the price and needs a reason:
a market it matches (section 6) or a view of what remote capital costs.

**Use a distortion when** you need prices for several layers, or for a book and
its parts, that are consistent with each other, or must defend one price of risk
across a program. **Do not** when there is one cover and one quote and nothing to
be consistent with: a load on expected loss gives the same answer with less
machinery. **Pitfalls:**

- `calibrate` with no `assets` takes the distribution's maximum as the capital
  for CCoC. Here that is the full annual limit, so the fitted `ccoc` is a return
  on full-limit capital. The same premium is a 10% return on 99% TVaR capital.
  State which capital a CCoC rate refers to.
- A premium must lie strictly between the (capped) mean and the maximum, but
  that is not enough for every family. A floor `r0` sets a smallest premium the
  family can reach, so a large floor fails for a modest target, and LEP cannot
  reach a premium near the maximum even with no floor. Outside its range
  `calibrate` raises `ValueError` with the message "no <family> distortion
  gives (or reaches) the premium" (example below).
- Do not carry a parameter across books. A hazard parameter of 0.686 for this
  layer (0.690 for the book of section 7) is the price of risk for that cover and
  capital standard, not a property of the market.

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
parameters from you. `beta(a, b)` is a flexible two-parameter shape, a candidate
when two quotes are to be matched and LEP's floor is not wanted; `mixture(ds,
weights)` blends distortions, for example a CCoC floor with a smooth family;
`minimum(ds)` takes the lower of several. Set their parameters with a search such
as the one in section 6, and price with `d.measure(x)` as for any distortion.

## 5. The capital standard

A reinsurer cannot pay more than it holds, so the price depends on how much it
must hold. The usual standards are the **VaR(p)** of the annual loss (it ignores
how large the loss is beyond that quantile); the **TVaR(p)** (it sees the whole
tail and is coherent, so a diversified book never needs more than the sum of its
parts); a target **expected policyholder deficit (EPD)** ratio (the expected loss
the assets would not cover, as a fraction of the expected loss, with the assets
the smallest that meet the target); and the **full limit**, which CCoC assumes
without `assets`.

```python
pf = Portfolio.from_independent(["5x5"], [mid])
print(f"{'capital standard':>18} {'assets':>7} {'EPD ratio':>10} {'risk premium at 10%':>20}")
standards = [("99% VaR", pf.assets(0.99)), ("99.5% VaR", pf.assets(0.995)),
             ("99% TVaR", TAIL.measure(mid)), ("99.5% TVaR", Distortion.tvar(0.995).measure(mid)),
             ("EPD ratio 1%", pf.assets_for_epd(0.01)), ("full limit", 10.0)]
for name, assets in standards:
    pent = Pentagon.solve(loss=mid.mean(), assets=assets, return_on_capital=RETURN)
    print(f"{name:>18} {assets:7.2f} {max(pf.epd(assets)[0], 0.0):10.2%} {pent.premium:20.3f}")
```

```text
  capital standard  assets  EPD ratio  risk premium at 10%
           99% VaR    5.60      3.71%                1.042
         99.5% VaR    7.50      1.40%                1.215
          99% TVaR    7.78      1.18%                1.240
        99.5% TVaR    9.14      0.35%                1.364
      EPD ratio 1%    8.03      1.00%                1.263
        full limit   10.00      0.00%                1.442
```

Same layer, same 10% return, and the risk premium runs from 1.042 to 1.442 with
the capital standard alone. A 99% VaR sets the assets at 5.60 and leaves 3.71% of
the expected loss unpaid in the bad years. The 99% TVaR asks for 7.78 and leaves
1.18%. Moving from VaR to TVaR at the same level raises the premium by about a
fifth. `Portfolio.assets(p)` is a VaR; for a TVaR use
`Distortion.tvar(p).measure(x)`; `Portfolio.epd(a)` returns the total EPD ratio
and the ratio by unit; `assets_for_epd(e)` inverts it.

**The choice is a modelling choice.** There is no correct level. Regulatory
regimes use both VaR and TVaR standards, at different levels. Pick the standard
the reinsurer is held to, and keep it fixed when you compare prices. Prefer TVaR
for a layer with a lump of outcomes at its limit, as here: a VaR changes
abruptly when the probability of a full-limit loss crosses 1 - p, a TVaR does not.

## 6. Calibrating to a target return, to a price, or to several layers

`Portfolio.calibrate` accepts any one of three targets (`premium`,
`return_on_capital` or `loss_ratio`) at a capital standard (`assets=` or `p=`):

```python
A = TAIL.measure(mid)
targets = {"premium": dict(premium=target.premium),
           "return on capital": dict(return_on_capital=RETURN),
           "loss ratio": dict(loss_ratio=target.loss_ratio)}
for how, kw in targets.items():
    d = pf.calibrate("ph", assets=A, **kw)
    p = pf.price(d, assets=A).total
    print(f"{how:>18}: {label(d):<30} premium {p.premium:.3f}, return {p.return_on_capital:.1%}, "
          f"loss ratio {p.loss_ratio:.1%}")
```

```text
           premium: proportional_hazard(0.671)     premium 1.240, return 10.1%, loss ratio 46.7%
 return on capital: proportional_hazard(0.673)     premium 1.234, return 10.0%, loss ratio 47.0%
        loss ratio: proportional_hazard(0.676)     premium 1.225, return 9.9%, loss ratio 47.3%
```

The three fits differ slightly from each other and from the 0.686 found above,
because `Portfolio` prices the loss capped at the assets (the reinsurer cannot pay
more than it holds), so its expected loss is a little below the layer's. The free
function `calibrate(family, x, premium, assets=None)` takes only a premium.
**Calibrate to a target return** to set a technical price. **Calibrate to a market
price** to read the price of risk the market pays, price other layers with it, or
see whether your own return is above or below it. **Calibrate to a loss ratio**
when the target is stated that way, as for a quota share.

### More than one layer

One parameter matches one price. Suppose the market also quotes 1.30 as the risk
premium of the 10x10 (an illustrative number). A proportional hazard fitted to
the 5x5 gives a much lower price for it. A family with a floor has a second
parameter: fix the floor `r0`, let `calibrate` set the other parameter to hit the
5x5, and search on `r0` until the 10x10 is right as well:

```python
top = annual["10x10"]
quote_top = 1.30      # an illustrative market quote for the 10x10 risk premium
one_parameter = calibrate("ph", mid, target.premium)
r0 = bisect(lambda r0: calibrate("lep", mid, target.premium, r0=r0).measure(top), 0.0, 0.05, quote_top)
two_parameter = calibrate("lep", mid, target.premium, r0=r0)
print(f"{'':>30}" + "".join(f"{n:>8}" for n in NAMES) + f"{'floor':>8}")
print(f"{'quote':>30}{'':>8}{target.premium:8.3f}{quote_top:8.3f}")
for d in (one_parameter, two_parameter):
    print(f"{label(d, 4):>30}" + "".join(f"{d.measure(g):8.3f}" for g in annual.values()) + f"{d.mass:8.2%}")
```

```text
                                   2x3     5x5   10x10   floor
                         quote           1.240   1.300
   proportional_hazard(0.6855)   1.106   1.240   1.009   0.00%
           lep(0.0297, 0.2962)   1.045   1.240   1.300   2.89%
```

The two-parameter fit hits both quotes, with a floor (`d.mass`) of 2.89% rate on
line; for LEP the floor is close to `r0` (0.0297) but not equal to it. It also
predicts the working layer at 1.045; if the market quotes that layer too, the
gap tests the family. prospicio has no fit to more quotes than a family has
parameters. For that, minimise the squared differences between
`calibrate(...).measure(layer)` and the quotes over the free parameter yourself
(for example with `scipy.optimize.least_squares` on `r0`); the residuals then say
whether the family fits the market.

When the quotes are for thin layers (as for catastrophe bonds), the concave hull
of the quotes is a distortion in itself. Give `Distortion.convex` pairs
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

Between quotes the hull is a straight line, and below the last it runs to the
origin, so it has no floor unless you add a quote at zero. It uses the quotes and
nothing else: no family assumption, but no information outside the quotes, and a
quote that lies inside the hull is silently dropped.

## 7. Stand-alone price, price in a portfolio, and allocation

A contract priced alone ignores the rest of the reinsurer's book. If its bad
years rarely coincide with the book's, adding it barely raises the book's tail and
it needs less capital than it would alone. The portfolio price is then lower, and
the difference is the **diversification benefit**. Build a small book of three
lines joined by a Gaussian copula (a way to give distributions a chosen
correlation; see [the severity and frequency guide](severity-and-frequency.md)) in
a `PredictiveDistribution` of 20,000 simulated years, one column per line:

```python
LINES = ["property", "casualty", "cat"]
margins = [Lognormal.from_mean_cv(20.0, 0.3), Lognormal.from_mean_cv(30.0, 0.2), Lognormal.from_mean_cv(10.0, 1.0)]
copula = GaussianCopula([[1.0, 0.3, 0.5], [0.3, 1.0, 0.0], [0.5, 0.0, 1.0]])
rows = [[m.quantile(u) for m, u in zip(margins, row)] for row in copula.sample(20_000, 7)]
book = PredictiveDistribution(["line"], [(n,) for n in LINES], rows)

print(f"{'method':>13} " + "".join(f"{n:>10}" for n in LINES) + f"{'sum':>9}{'total':>9}")
alone = capital(book, TAIL).standalone
print(f"{'stand-alone':>13} " + "".join(f"{x:10.2f}" for x in alone) + f"{sum(alone):9.2f}")
for method in ["euler", "covariance", "proportional", "marginal", "shapley"]:
    a = capital(book, TAIL, method)
    print(f"{method:>13} " + "".join(f"{x:10.2f}" for x in a.allocated) + f"{sum(a.allocated):9.2f}{a.total:9.2f}")
```

```text
       method   property  casualty       cat      sum    total
  stand-alone      41.63     49.81     64.23   155.68
        euler      34.18     33.85     59.75   127.79   127.79
   covariance      38.70     25.36     63.73   127.79   127.79
 proportional      34.17     40.89     52.73   127.79   127.79
     marginal      32.39     32.10     46.44   110.93   127.79
      shapley      35.17     39.12     53.50   127.79   127.79
```

`capital(pd, distortion, method)` allocates the distortion measure of the total
(here the 99% TVaR) to the lines. Despite the function's name, in the words of
section 1 these amounts are **assets** (premium plus capital), because the capital
standard sets the assets. The stand-alone amounts sum to far more than the book
needs; the difference is the diversification. The five methods split the
total differently because they answer different questions:

| Method | Answers | Adds up? |
|---|---|---|
| `euler` (default; also `allocate(pd, d)`) | What does each line contribute in the book's bad years? For TVaR, each line's average loss in the worst 1% of the book's years | Yes |
| `covariance` | Who moves with the total, over the whole distribution and not only the tail | Yes |
| `proportional` | Stand-alone measures scaled to the total; ignores dependence | Yes |
| `marginal` | What would the book save without the line (Merton and Perold)? | No: it falls short, and the shortfall belongs to no one |
| `shapley` | Average marginal contribution over every order of joining (at most 12 lines) | Yes |

**Euler is the usual choice.** Of these methods it is the only one that is the
gradient of the risk measure, so it is consistent with small changes to the
portfolio: it tells you whether growing a line makes the book's assets grow faster
or slower than the line's own stand-alone assets. For a distortion it is also the
natural (co-measure) allocation that `price_portfolio` and `Portfolio.price` use
below. Here the cat line gets 59.75 against 64.23 alone and the casualty line
33.85 against 49.81:
casualty diversifies, cat drives the book's tail. Use `marginal` for the one-off
question "what is this contract worth to the book", not to split the book.

To price the lines and not only split the capital, `price_portfolio` applies a
cost of capital to the assets and allocates both by co-measure (the natural
allocation, the same split as `euler` above), so component prices sum to the
portfolio price:

```python
priced = price_portfolio(book, TAIL, cost_of_capital=RETURN)
print(f"{'':>10} {'exp loss':>9} {'alone':>8} {'in book':>8} {'capital alone':>14} {'capital in book':>16}")
for name, s, c in zip(LINES, priced.standalone, priced.allocated):
    print(f"{name:>10} {s.expected_loss:9.2f} {s.premium:8.2f} {c.premium:8.2f} {s.capital:14.2f} {c.capital:16.2f}")
print(f"{'book':>10} {priced.total.expected_loss:9.2f} {sum(s.premium for s in priced.standalone):8.2f} "
      f"{priced.total.premium:8.2f} {sum(s.capital for s in priced.standalone):14.2f} {priced.total.capital:16.2f}")
print(f"premium saved by writing the lines together: {priced.diversification():.2f}")
```

```text
            exp loss    alone  in book  capital alone  capital in book
  property     20.01    21.98    21.30          19.66            12.88
  casualty     29.99    31.79    30.34          18.02             3.51
       cat      9.92    14.86    14.45          49.38            45.30
      book     59.92    68.63    66.09          87.05            61.70
premium saved by writing the lines together: 2.54
```

The premium and capital in the book add up to the Euler assets above: for
property 21.30 + 12.88 = 34.18, for the book 66.09 + 61.70 = 127.79. Each line
earns 10% on the capital allocated to it, so the cat line, which needs
the most capital, carries the most margin: a premium of 14.45 for an expected loss
of 9.92, against 30.34 for 29.99 for casualty. Capital falls from 87.05 stand-alone
to 61.70 in the book, but the premium falls only from 68.63 to 66.09, about a
tenth of the capital saved. `priced.diversification()` is a method returning the
premium saved. The components must add up to the portfolio (segments or covers),
never gross, ceded and net side by side.

`price_portfolio` uses a constant cost of capital. To price a book with a
distortion calibrated to a return or a quote, use `Portfolio`, which keeps for
each total the average loss of each unit given that total. `Portfolio.price`
prices the loss capped at the assets (here the same 99% TVaR as above) and splits loss,
margin, premium, capital and assets among the units. How the part above the
assets is shared has two forms: `allocation="linear"` uses each unit's expected
share of the total loss, `"lifted"` its distorted share. Bodoff's percentile
layer method, `port.bodoff(p=0.99)`, is a third split of the assets:

```python
port = Portfolio.from_predictive(book)
A_book = TAIL.measure(book)                    # the book's 99% TVaR, as for price_portfolio
d_book = port.calibrate("ph", assets=A_book, return_on_capital=RETURN)
print(label(d_book), f"assets {A_book:.2f}")
linear = port.price(d_book, assets=A_book)
lifted = port.price(d_book, assets=A_book, allocation="lifted")
bodoff = port.bodoff(assets=A_book)
print(f"{'':>10} {'premium lin':>12} {'lifted':>8} {'return lin':>11} {'lifted':>8} {'assets lin':>11} {'lifted':>8} {'Bodoff':>8}")
for i, name in enumerate(LINES):
    a, b = linear.allocated[i], lifted.allocated[i]
    print(f"{name:>10} {a.premium:12.2f} {b.premium:8.2f} {a.return_on_capital:11.1%} {b.return_on_capital:8.1%} "
          f"{a.assets:11.2f} {b.assets:8.2f} {bodoff[i]:8.2f}")
print(f"{'total':>10} {linear.total.premium:12.2f} {lifted.total.premium:8.2f} {linear.total.return_on_capital:11.1%} "
      f"{lifted.total.return_on_capital:8.1%} {linear.total.assets:11.2f} {lifted.total.assets:8.2f} {sum(bodoff):8.2f}")
```

```text
proportional_hazard(0.690) assets 127.79
            premium lin   lifted  return lin   lifted  assets lin   lifted   Bodoff
  property        21.71    21.68       12.0%    10.1%       36.00    38.45    40.10
  casualty        30.88    30.85        9.6%     5.7%       40.27    46.18    52.90
       cat        13.43    13.49        9.3%    12.2%       51.52    43.15    34.79
     total        66.02    66.02       10.0%    10.0%      127.79   127.79   127.79
```

On the same assets the book's premium is 66.02 with the calibrated distortion,
against 66.09 with a constant cost of capital in `price_portfolio`, but the split
by line differs: the distortion charges the cat line 13.43, not 14.45. The book
earns exactly 10% under both allocations, and the premium by line is almost the
same, but returns and assets by line are not: under the lifted allocation the
casualty line earns 5.7% and the cat line 12.2%, under the linear 9.6% and 9.3%.
Bodoff gives the cat line the least of the assets, 34.79, and casualty the most,
52.90.
None is the right one: they answer who owns each slice of the asset stack in the
bad years, and the answer is a convention. Choose one, state it, and do not
compare returns by line across conventions. `port.price(...)` returns a
`NaturalPrice`; `.to_dict()` gives its table as columns for `pandas.DataFrame`.

**Use** `price_portfolio` for a technical price of a contract inside a book at the
return the reinsurer asks; `Portfolio.price` for a book priced with a distortion
calibrated to a quote; `capital` (Euler) to split capital among lines without
setting a price. **Pitfalls:** the diversification benefit depends on the
dependence model, and a Gaussian copula has no tail dependence, so it can
understate how often lines have their worst years together (test with a t or
Archimedean copula). And an in-book price is a price for this reinsurer's book,
not one a cedant can use to compare reinsurers.

## 8. Which family for which market

What follows is practice and judgment, not theorem. Check each point against your
own quotes.

| Market | What matters | Reasonable starting point | Why |
|---|---|---|---|
| Working layers, casualty: frequent losses | Most of the price is in the body | A smooth family (PH or Wang) or CCoC on a TVaR standard, calibrated to a layer of similar attachment to the one you price | The families agree only near the layer they were calibrated to (section 4); the family matters for any layer far from it |
| Top casualty layers, long tail | Parameter and reserve uncertainty dominates | PH or Wang calibrated to the quotes you have, with a loss model whose tail reflects the uncertainty | The family matters from one in fifty out, but model uncertainty is larger |
| Property per risk | A mix of both ends | PH or Wang calibrated to two layers | One parameter will miss one end |
| Catastrophe: remote layers | Rate on line has a floor; prices are quoted more than modelled | A family with a floor (LEP, capped linear, CCoC) or `Distortion.convex` of the quotes | PH, Wang and dual power fall toward zero, but markets rarely quote below a minimum rate on line |
| Quota share, stop loss, whole-book total | A share of a roughly symmetric total | Standard deviation, or CCoC on a TVaR standard | Layer-by-layer consistency matters less; volatility describes the risk |

### What is a modelling choice

These cannot be derived from the data. Decide them on purpose, write them down and
test how much each moves the price.

- The **capital standard** (VaR or TVaR, its level, or an EPD target). Section 5
  moved one layer's premium from 1.042 to 1.442 with this alone.
- The **return on capital**, and whether it is the same for every layer.
- The **distortion family**. It sets the price of every layer you have no quote for,
  and matters most for remote layers.
- The **dependence** between the contract and the book, and between lines.
- The **allocation method**, which changes the return by line, not the total.
- Pricing **stand-alone** or **in the book**: the stand-alone price is the
  reference, the in-book price is what this reinsurer can accept.
- **Time value**: the risk premium here is on undiscounted loss, with the
  capital held for one year. `PaymentPattern` and
  `price_over_payout(price, pattern, rate=..., cost_of_capital=...)` (or
  `distortion=...`, the rule the price was made with) restate it over the
  payout. There the cost of capital is a spread over the discount rate, because
  discounting the losses is the investment income, and the capital runs off with
  the share of the loss still outstanding.
- **Model and parameter uncertainty**: a load for not knowing the distribution
  is a different thing from a load for its spread. Build it into the loss model
  (a heavier tail, a mixed parameter) or load it separately, and say which.

What is not a choice: a distortion's price is at least the expected loss, the
prices of pieces of one loss add up to the whole, and the units' premium and
capital from `price_portfolio`, `Portfolio.price` and `capital` (except
`marginal`) add up to the total.

## 9. Pitfalls in one place

- **Properties and methods.** On a `Distortion`, `mass`, `family` and
  `parameters` are properties, while `g(s)`, `g_inv`, `g_dual`, `measure(x)` and
  `weights(n)` are methods. On a `Pentagon`, `loss`, `margin`, `premium`,
  `capital`, `assets`, `loss_ratio` and `return_on_capital` are properties; a
  `Price` has the same, with the loss named `expected_loss`. On an `Allocation`
  and a `PortfolioPrice`, `allocated`, `standalone` and `total` are properties,
  while `diversification_benefit()` and `diversification()` are methods. On a
  `Portfolio`, `totals`, `probs` and `units` are properties, while `assets(p)`,
  `epd(a)` and `bodoff(...)` are methods.
- **Price the annual loss of the cover**, after reinstatements and the annual
  limit, not a single loss. Pricing per occurrence and summing double counts the
  load.
- **Risk premium, not technical premium.** Expenses and brokerage are grossed up
  afterwards with `Quote.technical(price, expenses=, brokerage=, limit=)` (see
  [the price guide](reinsurance-price.md)).
- **Few draws, noisy tail.** A 99% TVaR from 20,000 draws rests on 200 of them.
  Use a grid and `fft` where you can, or more draws and a fixed seed.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): the map across all the methods
  in these guides.
- [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md): where
  the loss distribution that these methods price comes from.
- [`aggregate-distributions.md`](aggregate-distributions.md): turning frequency and
  severity into the annual loss distribution.
- [`severity-and-frequency.md`](severity-and-frequency.md): choosing and fitting
  the inputs, and dependence between them.
- [`reserving-methods.md`](reserving-methods.md): the reserve risk that an
  insurance book carries into the capital standard.
- [`reinsurance-price.md`](reinsurance-price.md): what a reinsurance price is, and
  the technical premium these methods feed.

Reference: Mildenhall and Major, *Pricing Insurance Risk* (Wiley, 2022), for
distortions, calibration and the natural allocation; the design note
`docs/design/risk.md`.
