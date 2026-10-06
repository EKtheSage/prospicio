"""A one-year view of a small P&C insurer, end to end with actuarialrs.

Three risks, each modelled where it lives, then put together:

1. Reserve risk: an ODP bootstrap of the GenIns paid triangle.
2. Motor premium risk: a Poisson GLM for claim frequency on a policy file,
   a gamma GLM for severity, and a collective model with the GLM's
   parameter uncertainty in the claim count.
3. Property per-risk: a risk profile by sum-insured band, a surplus treaty
   inuring to a per-risk excess of loss with a paid reinstatement, pro
   rata as to time.

Then: price the per-risk XL from its simulated losses, join the three
risks with a rank correlation, and set and allocate capital at TVaR 99%.

Run from the repository root (needs the package: ``cd python && maturin
develop``):

    python examples/one_year_view.py
"""

import csv
import math
import os
from pathlib import Path

from actuarialrs.aggregate import simulate_events
from actuarialrs.distributions import Gamma, PredictiveDistribution, claim_count
from actuarialrs.models import Glm, Terms
from actuarialrs.pricing import Mbbefd, RiskProfile, price
from actuarialrs.reinsurance import Layer, Tower
from actuarialrs.reserving import OdpBootstrap, Triangle
from actuarialrs.risk import Distortion, capital, iman_conover

DATA = Path(__file__).resolve().parent.parent / "validation" / "data"
N_SIMS = int(os.environ.get("ACTUARIALRS_EXAMPLE_SIMS", "20000"))


def read_csv(name):
    with open(DATA / name) as f:
        rows = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(rows))


def one_component(label, sampled):
    """A single-line PredictiveDistribution from a component's draws."""
    return PredictiveDistribution(["line"], [(label,)], [[x] for x in sampled.draws])


def money(x):
    return f"{x / 1e6:8.2f}m"


# --- 1. Reserve risk --------------------------------------------------------
rows = read_csv("genins.csv")
tri = Triangle.from_frame(
    {
        "origin": [int(r["origin"]) for r in rows],
        "age": [int(r["development"]) for r in rows],
        "paid": [float(r["value"]) for r in rows],
    },
    "origin",
    "age",
    "paid",
)
boot = OdpBootstrap(n_sims=N_SIMS, seed=1).fit(tri, "paid")
reserve = boot.reserves.total()
print("1. Reserve risk (GenIns, ODP bootstrap)")
print(f"   mean reserve {money(reserve.mean())}, 99.5% {money(reserve.quantile(0.995))}")

# --- 2. Fleet motor premium risk -------------------------------------------
# 600 fleet policies; the file's amounts are in hundreds.
policies = read_csv("glm_policies.csv")
data = {
    "age": [float(p["age"]) for p in policies],
    "region": [p["region"] for p in policies],
}
exposure = [float(p["exposure"]) for p in policies]
claims = [float(p["claims"]) for p in policies]
coding = Terms().intercept().numeric("age").factor("region").fit(data)
frequency = Glm("poisson").fit(
    coding.design(data, offset=[math.log(e) for e in exposure]), claims
)
# Severity: a gamma GLM on each policy's average claim, weighted by its
# number of claims; its dispersion is a single claim's squared CV.
avg = [100.0 * float(p["avg_sev"]) for p in policies]
n_claims = [float(p["n_sev"]) for p in policies]
severity_fit = Glm("gamma", link="log").fit(
    Terms().intercept().fit(data).design(data, weights=n_claims), avg
)
severity = Gamma.from_mean_cv(
    math.exp(severity_fit.coefficients[0]), math.sqrt(severity_fit.dispersion)
)
# Next year every policy renews for a full year. The GLM's predictive
# distribution of the book's claim count carries parameter and process
# uncertainty; a negative binomial with its mean and variance carries it
# into a collective model.
counts = frequency.predict_distribution(
    coding.design(data, offset=[0.0] * len(policies)), N_SIMS, 2
).total()
count_model = claim_count(counts.mean(), counts.variance() / counts.mean())
motor = simulate_events(count_model, severity, N_SIMS, 3).totals().total()
print("2. Fleet motor (Poisson and gamma GLMs, collective model)")
print(f"   claims {counts.mean():.0f} (Var/E {counts.variance() / counts.mean():.2f}),"
      f" mean claim {severity.mean():,.0f}")
print(f"   mean loss {money(motor.mean())}, 99.5% {money(motor.quantile(0.995))}")

# --- 3. Property per-risk ---------------------------------------------------
# Bands of sum insured with bounds, expected loss and an exposure curve each.
profile = RiskProfile(
    [0.5e6, 3e6, 15e6],
    [4000, 600, 60],
    [Mbbefd.swiss_re(2.0), Mbbefd.swiss_re(3.0), Mbbefd.swiss_re(4.0)],
    expected_losses=[2.5e6, 2.0e6, 1.5e6],
    lower=[0.1e6, 1e6, 5e6],
    upper=[1e6, 5e6, 25e6],
)
retention, lines = 2e6, 4.0
xl_limit, xl_attachment = 3e6, 1e6
# Exposure-rated cost of the per-risk XL, net of the surplus; the upfront
# premium is that over a 70% loss ratio.
xl_cost = profile.expected_layer_loss(xl_limit, xl_attachment, retention, lines)
xl = Layer("per_risk_xl", xl_limit, xl_attachment, premium=xl_cost / 0.7,
           reinstatement_rates=[1.0], pro_rata_time=True)
tower = Tower.inuring([[Layer.surplus("surplus", retention, lines)], [xl]])
events = profile.simulate(N_SIMS, 4).with_uniform_times()
result = tower.apply(events)
gross = result.marginal(("gross", "ground_up"))
net = result.marginal(("net", "retained"))
xl_loss = result.marginal(("ceded", "per_risk_xl"))
reinstated = result.marginal(("reinstatement_premium", "per_risk_xl"))
print("3. Property per-risk (risk profile; surplus inuring to a per-risk XL)")
print(f"   gross {money(gross.mean())}, surplus cedes"
      f" {money(result.marginal(('ceded', 'surplus')).mean())}"
      f" (exposure rating {money(profile.expected_surplus_loss(retention, lines))})")
print(f"   per-risk XL {money(xl_loss.mean())} (exposure rating {money(xl_cost)}),"
      f" reinstatement premium {money(reinstated.mean())}")
print(f"   net mean {money(net.mean())}, 99.5% {money(net.quantile(0.995))}")

# The top band runs to 25m, past the surplus's 2m + 8m: its largest risks
# keep most of a total loss. More lines cut the net tail.
for more in [8.0, 12.0]:
    alt = Tower.inuring([[Layer.surplus("surplus", retention, more)], [xl]]).apply(events)
    print(f"   with {more:.0f} lines: net 99.5%"
          f" {money(alt.marginal(('net', 'retained')).quantile(0.995))}")

# Price the XL from its simulated losses: assets at TVaR 99%, 10% cost of
# capital, less the reinstatement premium it expects to collect.
xl_price = price(xl_loss, Distortion.tvar(0.99), cost_of_capital=0.10)
print(f"   XL risk-loaded premium {money(xl_price.premium)}"
      f" (capital {money(xl_price.capital)}),"
      f" net of reinstatements {money(xl_price.premium - reinstated.mean())}")

# --- 4. The company ---------------------------------------------------------
def company(property_line):
    parts = PredictiveDistribution.join(
        [
            ("reserve", one_component("reserve", reserve)),
            ("motor", one_component("motor", motor)),
            ("property", one_component("property", property_line)),
        ],
        "risk",
    ).aggregate(["risk"])
    # Rank correlation between the lines: reserve and motor share claims
    # inflation; property is nearly independent.
    corr = [[1.0, 0.4, 0.1], [0.4, 1.0, 0.1], [0.1, 0.1, 1.0]]
    return iman_conover(parts, corr, seed=5)


print("4. TVaR 99% of the year's losses, allocated by Euler (CoTVaR)")
tvar = Distortion.tvar(0.99)
allocations = {}
for label, line in [("gross of reinsurance", gross), ("net of reinsurance", net)]:
    pd = company(line)
    a = allocations[label] = capital(pd, tvar)
    shares = ", ".join(
        f"{k[0]} {money(x).strip()}" for k, x in zip(pd.components(), a.allocated)
    )
    print(f"   {label}: TVaR {money(a.total)} = {shares};"
          f" diversification {money(a.diversification_benefit())}")
saved = allocations["gross of reinsurance"].total - allocations["net of reinsurance"].total
print(f"   reinsurance cuts the company's TVaR by {money(saved).strip()}")
