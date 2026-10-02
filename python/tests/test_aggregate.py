import pytest

import actuarialrs as ar
from actuarialrs.aggregate import Layer, Tower, fft, panjer, simulate_events
from actuarialrs.distributions import Grid, Lognormal, NegativeBinomial, Poisson

SEV = Grid(1.0, [0.1, 0.3, 0.25, 0.2, 0.1, 0.05])


def test_panjer_and_fft_agree_and_match_compound_moments():
    for n in (Poisson(3.0), NegativeBinomial(2.5, 1.5)):
        a, ra = panjer(n, SEV, 300)
        b, rb = fft(n, SEV, 300)
        assert ra.method == "panjer" and rb.method == "fft"
        assert max(abs(x - y) for x, y in zip(a.probs, b.probs)) < 1e-13
        assert rb.aliasing_error < 1e-12
        # E[S] = E[N] E[X]; Var[S] = E[N] Var[X] + Var[N] E[X]^2.
        assert a.mean() == pytest.approx(n.mean() * SEV.mean(), rel=1e-12)
        var = n.mean() * SEV.variance() + n.variance() * SEV.mean() ** 2
        assert a.variance() == pytest.approx(var, rel=1e-10)


def test_panjer_underflow_points_to_fft():
    unit = Grid(1.0, [0.0, 1.0])
    with pytest.raises(ValueError, match="FFT"):
        panjer(Poisson(800.0), unit, 10)
    with pytest.raises(TypeError):
        panjer(Lognormal(0.0, 1.0), unit, 10)


def test_events_are_reproducible_and_feed_a_tower():
    sev = Lognormal.from_mean_cv(3e6, 1.5)
    events = simulate_events(Poisson(2.0), sev, 20_000, 11)
    again = simulate_events(Poisson(2.0), sev, 20_000, 11)
    assert events.events(123) == again.events(123)
    assert events.n_sims == 20_000 and events.seed == 11
    assert sum(events.counts()) > 0
    with pytest.raises(IndexError):
        events.events(20_000)

    tower = Tower([Layer("5x5", 5e6, 5e6, reinstatements=1), Layer("15x10", 15e6, 10e6, share=0.6)])
    result = tower.apply(events)
    assert result.dims == ["kind", "layer"]
    for row in result.draw_matrix()[:500]:
        gross, ceded_a, ceded_b, net = row
        assert gross == pytest.approx(ceded_a + ceded_b + net, abs=1e-6)
    by_kind = result.aggregate(["kind"])
    assert [k[0] for k in by_kind.components()] == ["gross", "ceded", "net"]
    assert result.provenance()["seed"] == 11


def test_mean_ceded_matches_the_exact_layer_value():
    sev = Lognormal.from_mean_cv(3e6, 1.5)
    events = simulate_events(Poisson(2.0), sev, 100_000, 3)
    ceded = Tower([Layer("5x5", 5e6, 5e6)]).apply(events).marginal(("ceded", "5x5"))
    exact = 2.0 * sev.layer(5e6, 5e6)
    se = ceded.variance() ** 0.5 / len(ceded) ** 0.5
    assert abs(ceded.mean() - exact) < 4 * se


def test_layer_terms():
    layer = Layer("L", 10.0, 5.0, share=0.5, aggregate_deductible=4.0, aggregate_limit=15.0)
    assert layer.ceded([8.0, 20.0, 12.0]) == 7.5
    with pytest.raises(ValueError, match="not both"):
        Layer("L", 1.0, 0.0, aggregate_limit=2.0, reinstatements=1)
    with pytest.raises(ValueError):
        Layer("L", 1.0, 0.0, share=1.5)
    with pytest.raises(ValueError):
        Tower([Layer("L", 1.0, 0.0), Layer("L", 2.0, 1.0)])
