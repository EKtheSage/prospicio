"""Reinsurer default over a term (prospicio.reinsurance.Panel.apply_term and
by_reinsurer_term, #980): collateral used up year by year on a certain
default, a term of one year drawing as apply does, a default final for the
rest of the path at the rate 1 - (1 - p)**t, the identities and the
errors."""

import math

import pytest

from prospicio.aggregate import EventSet, Exposure, Timeline
from prospicio.distributions import PredictiveDistribution
from prospicio.reinsurance import Layer, Panel, Reinsurer, Tower


def draws(pd, *key):
    return list(pd.marginal(list(key)).draws)


def term_of(n_paths, n_years):
    """Paths of independent years, one loss a year from 0 to 59."""
    tower = Tower([Layer("A", 20.0, 15.0), Layer("B", 20.0, 35.0)])
    years = [
        EventSet.from_years([[float((7 * j + 13 * t * t + t) % 60)] for j in range(n_paths)])
        for t in range(n_years)
    ]
    return tower.apply_term(years)


def test_certain_default_uses_the_collateral_up():
    years = [EventSet.from_years([[15.0]]), EventSet.from_years([[25.0]])]
    term = Tower([Layer("10x10", 10.0, 10.0)]).apply_term(years)
    panel = Panel([Reinsurer("Re A", 1.0, 0.0, collateral=8.0)]).with_line("Re A", "10x10", 1.0)
    after = panel.apply_term(term, 2)
    # Year 1 cedes 5, covered by the collateral; year 2 cedes 10, of which
    # the 3 left cover 3.
    assert [draws(after, "credit_loss", "10x10", t) for t in (1, 2)] == [[0.0], [7.0]]
    assert [draws(after, "recovered", "10x10", t) for t in (1, 2)] == [[5.0], [3.0]]
    assert draws(after, "net_after_default", "retained", 2) == [15.0 + 7.0]
    by = panel.by_reinsurer_term(term, 2, seed=3)
    assert by.dims == ["kind", "reinsurer", "year"]
    assert [draws(by, "exposure", "Re A", t) for t in (1, 2)] == [[0.0], [7.0]]
    assert [draws(by, "default", "Re A", t) for t in (1, 2)] == [[1.0], [1.0]]
    # The input's components come first, their draws unchanged.
    assert after.components()[: len(term.components())] == term.components()
    for key in term.components():
        assert draws(after, *key) == draws(term, *key), key


def test_a_term_of_one_year_draws_as_apply_does():
    events = EventSet.from_years([[float(g)] for g in range(0, 80, 2)])
    tower = Tower([Layer("A", 20.0, 15.0)])
    panel = Panel([Reinsurer("R1", 0.4, 0.3, collateral=2.0, correlation=0.5)]).with_line("R1", "A", 1.0)
    one = panel.by_reinsurer(tower.apply(events), seed=9)
    term = panel.by_reinsurer_term(tower.apply_term([events]), 1, seed=9)
    for kind in ("credit_loss", "exposure", "default"):
        assert draws(term, kind, "R1", 1) == draws(one, kind, "R1")


def test_a_default_is_final_and_keeps_its_rate():
    n_paths, n_years, p = 4000, 4, 0.2
    term = term_of(n_paths, n_years)
    panel = (
        Panel([Reinsurer("R1", p, 0.3, collateral=4.0), Reinsurer("R2", 0.1, 0.5, correlation=0.4)])
        .with_line("R1", "A", 0.5)
        .with_line("R2", "A", 0.5)
        .with_line("R1", "B", 1.0)
    )
    by = panel.by_reinsurer_term(term, n_years, seed=5)
    after = panel.apply_term(term, n_years, seed=5)
    defaults = [draws(by, "default", "R1", t) for t in range(1, n_years + 1)]
    for t in range(1, n_years):
        assert all(b >= a for a, b in zip(defaults[t - 1], defaults[t]))
    for t in range(1, n_years + 1):
        q = 1 - (1 - p) ** t
        rate = sum(defaults[t - 1]) / n_paths
        assert math.fabs(rate - q) <= 5 * math.sqrt(q * (1 - q) / n_paths), (t, rate, q)
        # recovered = ceded - credit loss, and the net after default adds
        # the year's credit loss.
        credit = [0.0] * n_paths
        for layer in ("A", "B"):
            c = draws(after, "credit_loss", layer, t)
            ceded = draws(term, "ceded", layer, t)
            assert draws(after, "recovered", layer, t) == [x - y for x, y in zip(ceded, c)]
            credit = [a + b for a, b in zip(credit, c)]
        by_total = [a + b for a, b in zip(draws(by, "credit_loss", "R1", t), draws(by, "credit_loss", "R2", t))]
        assert all(math.fabs(a - b) <= 1e-9 * (1 + b) for a, b in zip(credit, by_total))
        net = draws(term, "net", "retained", t)
        after_net = draws(after, "net_after_default", "retained", t)
        assert all(math.fabs(a - (n + c)) <= 1e-9 * (1 + a) for a, n, c in zip(after_net, net, by_total))


def test_the_errors():
    term = term_of(30, 2)
    years = Tower([Layer("A", 20.0, 15.0), Layer("B", 20.0, 35.0)]).apply_annual([10.0, 40.0])
    panel = Panel([Reinsurer("R1", 0.1, 0.4)]).with_line("R1", "A", 1.0)
    with pytest.raises(ValueError, match="apply_term"):
        panel.apply(term)
    with pytest.raises(ValueError, match="term"):
        panel.apply_term(years, 1)
    for n_years in (0, 1, 3):
        with pytest.raises(ValueError):
            panel.apply_term(term, n_years)
    with pytest.raises(ValueError, match="path-year"):
        panel.with_driver([1.0] * 30).apply_term(term, 2)
    with pytest.raises(ValueError):
        panel.apply_term(panel.apply_term(term, 2), 2)
    for seed in (-1, 2**64):
        with pytest.raises(OverflowError):
            panel.apply_term(term, 2, seed=seed)
        with pytest.raises(OverflowError):
            panel.by_reinsurer_term(term, 2, seed=seed)
    with pytest.raises(OverflowError):
        panel.apply_term(term, -1)


def test_aggregate_and_timeline_terms():
    # apply_aggregate_term: one path ceding 5 then 10 on a 10x10 aggregate
    # cover; Re A certainly defaults with collateral 8.
    panel = Panel([Reinsurer("Re A", 1.0, 0.0, collateral=8.0)]).with_line("Re A", "XL", 1.0)
    totals = [PredictiveDistribution(["line"], [("a",)], [[x]]) for x in (15.0, 25.0)]
    agg = Tower([Layer("XL", 10.0, 10.0)]).apply_aggregate_term(totals)
    after = panel.apply_term(agg, 2)
    assert [draws(after, "credit_loss", "XL", t) for t in (1, 2)] == [[0.0], [7.0]]
    # apply_timeline, risks attaching: an event in month 14 of 24 split 1:3
    # over the two contract years cedes 5 and then 50.
    events = EventSet.from_years([[100.0]], times=[[13.5 / 24]])
    shares = [[0.0] * 24, [0.0] * 24]
    shares[0][13], shares[1][13] = 0.25, 0.75
    tower = Tower([Layer("XL", 50.0, 20.0)]).with_cover_basis("risks_attaching")
    rad = tower.apply_timeline(Timeline(events, 24), Exposure(shares))
    after = panel.apply_term(rad, 2)
    # Year 1's 5 is covered by the collateral; year 2 loses 50 less the 3 left.
    assert [draws(after, "credit_loss", "XL", t) for t in (1, 2)] == [[0.0], [47.0]]
