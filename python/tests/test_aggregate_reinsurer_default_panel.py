"""Reinsurer default on a tower (prospicio.reinsurance.Panel and Reinsurer,
#980): the result's components and identities, exact values on a panel
whose defaults are certain, determinism by seed, the errors, and the
independent and factor cases of validation/reference/reinsurer_default.csv
on stratified lognormal years at a few standard errors."""

import math
from statistics import NormalDist

import pytest
from helpers import VALIDATION, read_csv

from prospicio.reinsurance import Layer, Panel, Reinsurer, Tower


def draws(pd, *key):
    return list(pd.marginal(list(key)).draws)


def near(a, b, rel=1e-12, abs_=1e-12):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == y or math.fabs(x - y) <= max(rel * math.fabs(y), abs_), (x, y)


def certain_panel():
    """The Rust doctest's panel: Re A always defaults, Re B never does."""
    return (
        Panel([Reinsurer("Re A", 1.0, 0.4), Reinsurer("Re B", 0.0, 0.4)])
        .with_line("Re A", "10x10", 0.5)
        .with_line("Re B", "10x10", 0.5)
    )


def test_reinsurer_fields_and_repr():
    r = Reinsurer("Re A", 0.002, 0.4, collateral=1e6, correlation=0.3)
    assert (r.name, r.probability, r.recovery, r.collateral, r.correlation) == (
        "Re A",
        0.002,
        0.4,
        1e6,
        0.3,
    )
    assert repr(r).startswith('Reinsurer("Re A"')
    plain = Reinsurer("Re B", 0.01, 0.5)
    assert (plain.collateral, plain.correlation) == (0.0, 0.0)


def test_certain_default_gives_the_rust_values():
    years = Tower([Layer("10x10", 10.0, 10.0)]).apply_annual([5.0, 15.0, 30.0])
    after = certain_panel().apply(years, seed=7)
    # The Rust doctest's values: 60% of half the ceded loss.
    assert draws(after, "credit_loss", "10x10") == [0.0, 1.5, 3.0]
    assert draws(after, "recovered", "10x10") == [0.0, 3.5, 7.0]
    assert draws(after, "net_after_default", "retained") == [5.0, 11.5, 23.0]
    # The input's components come first, unchanged.
    assert after.components()[: len(years.components())] == years.components()
    by = certain_panel().by_reinsurer(years, seed=7)
    assert draws(by, "default", "Re A") == [1.0, 1.0, 1.0]
    assert draws(by, "default", "Re B") == [0.0, 0.0, 0.0]
    assert draws(by, "exposure", "Re A") == [0.0, 2.5, 5.0]
    assert draws(by, "credit_loss", "Re A") == [0.0, 1.5, 3.0]


def test_the_identities_hold_on_random_defaults():
    gross = [float(g) for g in range(0, 80, 3)]
    tower = Tower([Layer("A", 20.0, 15.0), Layer("B", 40.0, 35.0), Layer("C", 10.0, 75.0)])
    years = tower.apply_annual(gross)
    panel = (
        Panel([Reinsurer("R1", 0.4, 0.3, collateral=5.0), Reinsurer("R2", 0.5, 0.6)])
        .with_line("R1", "A", 0.6)
        .with_line("R2", "A", 0.4)
        .with_line("R1", "B", 0.3)
        .with_line("R2", "B", 0.7)
    )
    after = panel.apply(years, seed=11)
    total_credit = [0.0] * len(gross)
    for layer in ["A", "B", "C"]:
        ceded = draws(years, "ceded", layer)
        credit = draws(after, "credit_loss", layer)
        near([c + r for c, r in zip(credit, draws(after, "recovered", layer))], ceded)
        total_credit = [t + c for t, c in zip(total_credit, credit)]
    # C is on no line: no credit risk.
    assert draws(after, "credit_loss", "C") == [0.0] * len(gross)
    net = draws(years, "net", "retained")
    near(draws(after, "net_after_default", "retained"), [n + c for n, c in zip(net, total_credit)])
    by = panel.by_reinsurer(years, seed=11)
    by_total = [a + b for a, b in zip(draws(by, "credit_loss", "R1"), draws(by, "credit_loss", "R2"))]
    near(by_total, total_credit)
    assert any(c > 0 for c in total_credit)
    assert panel.reinsurers[1].name == "R2"
    assert repr(panel) == 'Panel(["R1", "R2"])'


def test_the_draws_are_fixed_by_the_seed():
    years = Tower([Layer("A", 20.0, 15.0)]).apply_annual([float(g) for g in range(10, 60)])
    panel = Panel([Reinsurer("R1", 0.5, 0.2, correlation=0.4)]).with_line("R1", "A", 1.0)
    one = draws(panel.by_reinsurer(years, seed=3), "default", "R1")
    assert draws(panel.by_reinsurer(years, seed=3), "default", "R1") == one
    assert draws(panel.by_reinsurer(years, seed=4), "default", "R1") != one
    # The default seed is 0.
    assert draws(panel.by_reinsurer(years), "default", "R1") == draws(
        panel.by_reinsurer(years, seed=0), "default", "R1"
    )
    # A driver ranks the years instead of the gross loss.
    driven = panel.with_driver([float(-i) for i in range(50)]).apply(years, seed=3)
    assert len(draws(driven, "credit_loss", "A")) == 50


def test_the_errors():
    years = Tower([Layer("A", 20.0, 15.0)]).apply_annual([10.0, 40.0])
    with pytest.raises(ValueError):
        Reinsurer("R1", 1.5, 0.4)
    with pytest.raises(ValueError):
        Reinsurer("R1", 0.1, 0.4, correlation=1.0)
    with pytest.raises(ValueError):
        Reinsurer("R1", 0.1, 0.4, collateral=-1.0)
    with pytest.raises(ValueError, match="at least one reinsurer"):
        Panel([])
    with pytest.raises(ValueError, match="unique"):
        Panel([Reinsurer("R1", 0.1, 0.4), Reinsurer("R1", 0.2, 0.4)])
    panel = Panel([Reinsurer("R1", 0.1, 0.4), Reinsurer("R2", 0.1, 0.4)])
    with pytest.raises(ValueError, match="no reinsurer"):
        panel.with_line("R9", "A", 0.5)
    with pytest.raises(ValueError):
        panel.with_line("R1", "A", 1.5)
    with pytest.raises(ValueError, match="already has a line"):
        panel.with_line("R1", "A", 0.5).with_line("R1", "A", 0.5)
    with pytest.raises(ValueError):
        panel.with_driver([1.0, float("nan")])
    # Checked when the panel meets the tower.
    with pytest.raises(ValueError):
        panel.with_line("R1", "A", 0.5).apply(years)
    with pytest.raises(ValueError):
        panel.with_line("R1", "Z", 1.0).apply(years)
    full = panel.with_line("R1", "A", 0.5).with_line("R2", "A", 0.5)
    with pytest.raises(ValueError):
        full.with_driver([1.0, 2.0, 3.0]).apply(years)
    with pytest.raises(ValueError):
        full.apply(full.apply(years))


# The reference's cases on stratified years: the gross loss at the
# lognormal's quantiles (i + 1/2) / n, so the only sampling error is the
# default draws'.
N_YEARS = 100_000
ROWS = read_csv(VALIDATION / "reference" / "reinsurer_default.csv")


def params(row):
    return {k: float(v) for k, v in (p.split("=") for p in row["params"].split(";"))}


@pytest.fixture(scope="module")
def stratified():
    p = params(ROWS[0])
    z = NormalDist()
    gross = [math.exp(p["meanlog"] + p["sdlog"] * z.inv_cdf((i + 0.5) / N_YEARS)) for i in range(N_YEARS)]
    tower = Tower([Layer(l, p[f"limit_{l}"], p[f"attach_{l}"]) for l in "AB"])
    return tower.apply_annual(gross)


def panel_of(row):
    p = params(row)
    names = [f"R{k}" for k in range(1, int(p["reinsurers"]) + 1)]
    panel = Panel(
        [
            Reinsurer(r, p[f"p_{r}"], p[f"rec_{r}"], collateral=p[f"col_{r}"], correlation=p[f"rho_{r}"])
            for r in names
        ]
    )
    for r in names:
        for l in "AB":
            if p[f"line_{r}_{l}"] > 0:
                panel = panel.with_line(r, l, p[f"line_{r}_{l}"])
    return panel


QUANTITIES = {"credit_loss_mean", "credit_loss_mean_by_layer", "default_frequency"}
CASES = [r for r in ROWS if r["distribution"] != "gross" and r["quantity"] in QUANTITIES]


@pytest.mark.parametrize("row", CASES, ids=lambda r: f"{r['distribution']}-{r['quantity']}-{r['arg']}")
def test_the_reference_means_within_a_few_standard_errors(stratified, row):
    panel = panel_of(row)
    quantity, arg = row["quantity"], row["arg"]
    if quantity == "credit_loss_mean_by_layer":
        x = draws(panel.apply(stratified, seed=5), "credit_loss", arg)
    else:
        by = panel.by_reinsurer(stratified, seed=5)
        kind = "default" if quantity == "default_frequency" else "credit_loss"
        names = [r.name for r in panel.reinsurers]
        cols = [draws(by, kind, r) for r in (names if arg == "total" else [arg])]
        x = [sum(c) for c in zip(*cols)]
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    expected = float(row["expected"])
    assert math.fabs(m - expected) <= 5 * sd / math.sqrt(n), (m, expected, sd / math.sqrt(n))
