import math
import pickle

import pytest

import prospicio as ar

D = ar.distributions


def test_pareto_layers_and_truncation():
    p = D.Pareto(500.0, 2.0)
    assert p.layer(4000.0, 1000.0) == pytest.approx(200.0, rel=1e-12)
    assert p.mean() == pytest.approx(1000.0)
    assert p.survival(1000.0) == pytest.approx(0.25)
    assert math.isinf(D.Pareto(1.0, 0.9).mean())
    t = D.Pareto(500.0, 2.0, truncation=8000.0)
    assert t.truncation == 8000.0
    assert t.quantile(1.0) == pytest.approx(8000.0)
    assert t.lev(1e5) == pytest.approx(t.mean())
    assert t.layer_variance(1e4, 0.0) == pytest.approx(t.variance(), rel=1e-12)
    with pytest.raises(ValueError, match="alpha"):
        D.Pareto(1.0, 0.0)


def test_pareto_fit_closed_form():
    fit = D.Pareto.fit([1500.0, 2500.0, 4000.0, 10_000.0], 1000.0,
                       censored=[False, False, False, True])
    assert fit.alpha == pytest.approx(3.0 / math.log(150.0), rel=1e-12)
    assert isinstance(fit, D.Pareto)


def test_piecewise_pareto():
    pp = D.PiecewisePareto([1000.0, 2000.0], [1.0, 2.0])
    assert pp.survival(4000.0) == pytest.approx(0.125, rel=1e-12)
    assert pp.stop_loss(2000.0) == pytest.approx(1000.0, rel=1e-12)
    assert pp.t == [1000.0, 2000.0] and pp.alpha == [1.0, 2.0]
    lp = D.PiecewisePareto([1000.0, 2000.0], [1.0, 2.0], truncation=5000.0)
    wd = D.PiecewisePareto([1000.0, 2000.0], [1.0, 2.0], truncation=5000.0, truncation_type="wd")
    assert lp.truncation_type == "lp" and wd.truncation_type == "wd"
    assert lp.survival(1500.0) != wd.survival(1500.0)
    with pytest.raises(ValueError, match="truncation_type"):
        D.PiecewisePareto([1.0], [1.0], truncation=2.0, truncation_type="xx")
    fit = D.PiecewisePareto.fit([1200.0, 1500.0, 2500.0, 6000.0], [1000.0, 2000.0])
    want = 2.0 / (math.log(1.2) + math.log(1.5) + 2.0 * math.log(2.0))
    assert fit.alpha[0] == pytest.approx(want, rel=1e-12)


def test_log_affine_and_generalized_pareto():
    d = D.LogAffinePareto.from_delta(1e6, 1.5, 0.5)
    assert d.local_alpha(2e6) == pytest.approx(2.0)
    assert d.delta == pytest.approx(0.5)
    # gamma = 0 is the Pareto.
    flat = D.LogAffinePareto(1000.0, 2.5, 0.0)
    assert flat.layer(4000.0, 1000.0) == pytest.approx(D.Pareto(1000.0, 2.5).layer(4000.0, 1000.0))
    g = D.GeneralizedPareto.riegel(1000.0, 2.0, 1.5)
    assert g.survival(2000.0) == pytest.approx((7.0 / 3.0) ** -1.5, rel=1e-12)
    assert g.location == 1000.0
    assert g.lev(5000.0) + g.stop_loss(5000.0) == pytest.approx(g.mean(), rel=1e-12)


def test_counts_by_dispersion():
    assert isinstance(D.claim_count(4.0, 1.0), D.Poisson)
    assert isinstance(D.claim_count(4.0, 2.5), D.NegativeBinomial)
    b = D.claim_count(10.0, 0.3)
    assert isinstance(b, D.Binomial)
    assert (b.n, b.mean()) == (15, pytest.approx(10.0))
    assert D.Binomial(10, 0.3).pmf(11) == 0.0
    assert sum(D.Binomial(10, 0.3).pmf(k) for k in range(11)) == pytest.approx(1.0)


def test_pareto_family_feeds_aggregation():
    # Truncated inside the grid, so no severity or aggregate mass is lumped
    # into the last point (up to 20 claims of at most 50,000).
    sev = D.PiecewisePareto([1000.0, 3000.0], [1.2, 2.0], truncation=50_000.0)
    grid, _ = D.Grid.local_moment(sev, 100.0, 501)
    assert grid.mean() == pytest.approx(sev.mean(), rel=1e-12)
    agg, _ = ar.aggregate.panjer(D.Binomial(20, 0.1), grid, 10_001)
    assert agg.mean() == pytest.approx(2.0 * grid.mean(), rel=1e-9)


def test_pickle_round_trip():
    for d in (D.Pareto(500.0, 2.0, 8000.0),
              D.PiecewisePareto([1.0, 2.0], [1.0, 2.0], 5.0, "wd"),
              D.LogAffinePareto(1.0, 1.5, 0.3),
              D.GeneralizedPareto(0.5, 2.0, 1.0),
              D.Binomial(5, 0.2)):
        back = pickle.loads(pickle.dumps(d))
        assert repr(back) == repr(d)


def test_generalized_pareto_fit_and_whole_truncation():
    g = D.GeneralizedPareto.riegel(1000.0, 3.0, 1.5)
    draws = g.sample(50_000, seed=4)
    fit = D.GeneralizedPareto.fit_riegel(draws, 1000.0)
    assert 1000.0 / fit.beta == pytest.approx(3.0, rel=0.05)
    assert 1.0 / fit.xi == pytest.approx(1.5, rel=0.05)
    truth = D.PiecewisePareto([1000.0, 2500.0], [1.2, 0.8], truncation=60_000.0, truncation_type="wd")
    fit = D.PiecewisePareto.fit(truth.sample(50_000, seed=5), [1000.0, 2500.0],
                                truncation=60_000.0, truncation_type="wd")
    assert fit.truncation_type == "wd"
    assert fit.alpha == pytest.approx([1.2, 0.8], rel=0.05)


def test_local_pareto_conversion():
    exact = D.LogAffinePareto(1000.0, 1.5, 0.4)
    pp, err, end = D.local_pareto_to_piecewise(1000.0, exact.local_alpha, rel_tolerance=1e-5)
    assert err <= 1e-5 and end > 1000.0
    for x in (1500.0, 7777.0, 1e5):
        if x < end:
            assert pp.survival(x) == pytest.approx(exact.survival(x), rel=1.01e-5)

    def bad(x):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        D.local_pareto_to_piecewise(1.0, bad)
    with pytest.raises(ValueError, match="alpha"):
        D.local_pareto_to_piecewise(1.0, lambda x: -1.0)


def test_issue_213_policy_limits_and_standard_errors():
    from prospicio.pricing import CollectiveModel

    x = D.Pareto(100.0, 2.0)
    c = D.PolicyLimits(x, [1000.0, 5000.0, math.inf], [2, 1, 1])
    assert c.weights == pytest.approx([0.5, 0.25, 0.25])
    # E[min(X, L)] = t (2 - t / L) for alpha 2, and the mean beyond no limit.
    want = 0.5 * 100.0 * 1.9 + 0.25 * 100.0 * 1.98 + 0.25 * 200.0
    assert c.mean() == pytest.approx(want, rel=1e-12)
    # A layer reaches only the policies whose limit exceeds its attachment.
    assert c.layer(500.0, 1500.0) == pytest.approx(0.5 * x.layer(500.0, 1500.0), rel=1e-12)
    # The atom at 1000: half of the losses above it.
    assert c.cdf(1000.0) - c.cdf(999.999999) == pytest.approx(0.5 * 0.01, rel=1e-5)
    # Discretized and compounded like any severity: local moment matching
    # keeps the limited mean at its last point, and rounding puts the atom
    # at 1000 (point 100) whole.
    grid, _ = D.Grid.local_moment(c, 10.0, 2000)
    assert grid.mean() == pytest.approx(c.lev(1999 * 10.0), rel=1e-9)
    rounded, _ = D.Grid.rounding(c, 10.0, 2000)
    assert rounded.probs[100] == pytest.approx(c.cdf(1005.0) - c.cdf(995.0), rel=1e-12)
    assert rounded.probs[100] > 0.5 * 0.01
    model = CollectiveModel(D.Poisson(3.0), c)
    assert model.layer_mean(1000.0, 500.0) == pytest.approx(3.0 * c.layer(1000.0, 500.0), rel=1e-12)
    back = D.from_json(D.to_json(c))
    assert back.mean() == pytest.approx(c.mean(), rel=1e-15)
    with pytest.raises(ValueError):
        D.PolicyLimits(x, [1000.0], [1.0, 2.0])

    losses = [1500.0, 2500.0, 4000.0, 10_000.0]
    fit, se = D.Pareto.fit_with_standard_error(losses, 1000.0, censored=[False, False, False, True])
    assert se == pytest.approx(fit.alpha / 3 ** 0.5, rel=1e-14)
    fit, se = D.Pareto.fit_with_standard_error(losses[:3], 1000.0, truncation=20_000.0)
    assert 0.0 < se < math.inf
    pw, ses = D.PiecewisePareto.fit_with_standard_errors([1200.0, 1500.0, 2500.0, 6000.0],
                                                         [1000.0, 2000.0])
    assert ses == pytest.approx([a / 2 ** 0.5 for a in pw.alpha], rel=1e-14)
    with pytest.raises(ValueError):
        D.PiecewisePareto.fit_with_standard_errors(losses, [1000.0, 2000.0], truncation=50_000.0,
                                                   truncation_type="wd")


def test_issue_194_severity_fits_with_truncation_and_censoring():
    y = [120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 4000.0, 650.0, 90.0, 2100.0]
    logs = [math.log(v) for v in y]
    mu = sum(logs) / len(y)
    sd = math.sqrt(sum((v - mu) ** 2 for v in logs) / len(y))
    fit = D.fit_severity("lognormal", y)
    assert fit.parameters["meanlog"] == pytest.approx(mu, abs=1e-7)
    assert fit.parameters["sdlog"] == pytest.approx(sd, rel=1e-7)
    assert fit.standard_errors["meanlog"] == pytest.approx(sd / math.sqrt(len(y)), rel=1e-4)
    assert isinstance(fit.distribution, D.Lognormal)
    assert fit.aic == pytest.approx(4 - 2 * fit.log_likelihood)
    # Truncated and censored data: the reporting thresholds change the fit,
    # and every family returns its own class.
    r = [100.0 if v > 100 else 0.0 for v in y]
    c = [v >= 4000.0 for v in y]
    for family, cls in [("gamma", D.Gamma), ("weibull", D.Weibull), ("burr", D.Burr)]:
        f = D.fit_severity(family, y, reporting_thresholds=r, censored=c)
        assert isinstance(f.distribution, cls)
        assert len(f.covariance) == len(f.parameters)
        assert math.isfinite(f.log_likelihood)
    with pytest.raises(ValueError):
        D.fit_severity("pareto", y)


def test_issue_328_lomax_fit_solves_its_score_equations():
    # alpha = n / sum(log1p(y / scale)) and (alpha + 1) sum(y / (scale + y)) = n.
    y = [50.0, 80.0, 120.0, 200.0, 300.0, 500.0, 900.0, 1500.0, 4000.0, 12000.0]
    fit = D.fit_severity("lomax", y)
    a, s = fit.parameters["alpha"], fit.parameters["scale"]
    assert a * sum(math.log1p(v / s) for v in y) == pytest.approx(len(y), rel=1e-7)
    assert (a + 1) * sum(v / (s + v) for v in y) == pytest.approx(len(y), rel=1e-7)
    assert isinstance(fit.distribution, D.Burr)
    assert fit.distribution.survival(1000.0) == pytest.approx((1 + 1000.0 / s) ** -a, rel=1e-12)
    assert fit.aic == pytest.approx(4 - 2 * fit.log_likelihood)


def test_issue_328_truncation_from_above():
    # Claims kept only up to 3000: each conditioned on X <= 3000.
    y = [120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 2900.0, 650.0, 90.0, 2100.0]
    u = [3000.0] * len(y)
    fit = D.fit_severity("lognormal", y, truncated_above=u)
    m, s = fit.parameters["meanlog"], fit.parameters["sdlog"]

    def ll(m, s):
        phi = 0.5 * (1 + math.erf((math.log(3000.0) - m) / (s * math.sqrt(2))))
        return sum(-math.log(v * s * math.sqrt(2 * math.pi))
                   - 0.5 * ((math.log(v) - m) / s) ** 2 for v in y) - len(y) * math.log(phi)

    assert fit.log_likelihood == pytest.approx(ll(m, s), rel=1e-10)
    for dm, ds in [(1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)]:
        assert ll(m + dm, s + ds) < fit.log_likelihood
    # The bound spreads the fit wider than the plain one.
    assert s > D.fit_severity("lognormal", y).parameters["sdlog"]
    # A common bound conditions goodness of fit; differing bounds are refused.
    gof = D.goodness_of_fit(fit.distribution, y, truncated_above=u)
    assert max(gof["model_quantile"]) <= 3000.0
    with pytest.raises(ValueError):
        D.goodness_of_fit(fit.distribution, y, truncated_above=u[:-1] + [math.inf])
    with pytest.raises(ValueError):
        D.fit_severity("lognormal", y, truncated_above=[100.0] * len(y))


def test_issue_194_product_limit_and_goodness_of_fit():
    # Without truncation or censoring, the weighted empirical survival.
    pl = D.product_limit([100.0, 200.0, 200.0, 400.0], weights=[1.0, 2.0, 1.0, 4.0])
    assert pl["survival"] == pytest.approx([7 / 8, 0.5, 0.0])
    # Truncation shrinks the risk set: losses reported above 150 are not at
    # risk at 120.
    pl = D.product_limit([120.0, 200.0, 300.0], reporting_thresholds=[0.0, 150.0, 150.0])
    assert pl["at_risk"] == [1.0, 2.0, 1.0]
    y = [120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 4000.0]
    c = [False] * 6 + [True]
    fit = D.fit_severity("weibull", y, censored=c)
    gof = D.goodness_of_fit(fit.distribution, y, censored=c)
    assert gof["x"] == y[:6]
    assert 0.0 < gof["max_distance"] < 0.5
    assert len(gof["model_quantile"]) == 6
    # Any severity works, a Pareto fit included.
    assert D.goodness_of_fit(D.Pareto.fit(y, 100.0), y)["max_distance"] > 0.0
