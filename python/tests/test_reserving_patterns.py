"""Selected development patterns (``ldf``, ``ldf_kind``, ``ldf_weight``) on
the chain ladder, the expected-loss methods, Mack and the bootstraps
(``docs/design/reserving-v02.md``, decision 12). The Rust core is checked
against chainladder-python and R in ``validation/tests/reserving_patterns.rs``;
these tests check the bindings' arguments, conversions and errors."""

import math

import pytest

from prospicio.reserving import (
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    ExpectedLoss,
    Mack,
    MackBootstrap,
    OdpBootstrap,
    TailConstant,
    Triangle,
)

YEARS = [2019] * 5 + [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]
AGES = [12, 24, 36, 48, 60, 12, 24, 36, 48, 12, 24, 36, 12, 24, 12]
PAID = [100.0, 150.0, 165.0, 170.0, 172.0, 110.0, 170.0, 180.0, 185.0, 120.0, 175.0, 192.0,
        130.0, 200.0, 140.0]
PREMIUM = [250.0] * 5 + [260.0] * 4 + [270.0] * 3 + [280.0] * 2 + [290.0]


@pytest.fixture
def tri():
    return Triangle.from_long(YEARS, AGES, {"paid": PAID, "premium": PREMIUM})


def close(a, b, tol=1e-12):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == pytest.approx(y, rel=tol, abs=tol), (a, b)


def test_each_kind_selects_the_same_factors(tri):
    ldf = {12: 1.6, 24: 1.1, 36: 1.03, 48: 1.01}
    cdf = {60: 1.0}
    for age in [48, 36, 24, 12]:
        cdf[age] = cdf[age + 12] * ldf[age]
    percent = {age: 1 / c for age, c in cdf.items()}
    fits = {
        "ldf": ChainLadder(ldf=ldf, ldf_kind="ldf").fit(tri, "paid"),
        "cdf": ChainLadder(ldf=cdf, ldf_kind="cdf").fit(tri, "paid"),
        "percent_developed": ChainLadder(ldf=percent, ldf_kind="percent_developed").fit(
            tri, "paid"
        ),
    }
    for kind, fit in fits.items():
        close(fit.ldf, list(ldf.values()))
        close(fit.selected_ldf, list(ldf.values()))
        assert fit.ldf_weight == [1.0] * 4
        # A fully selected factor carries no parameter error.
        assert fit.std_err == [0.0] * 4
        assert all(s > 0 for s in fit.estimated_std_err[:3])
        assert fit.estimated_ldf != fit.ldf
    assert fits["ldf"].pattern_rescale is None
    # Given to the oldest age at 100% with no tail, a cumulative pattern
    # implies a tail of 1: none.
    assert fits["cdf"].pattern_rescale == pytest.approx(1.0)
    assert fits["percent_developed"].pattern_rescale == pytest.approx(1.0)
    assert fits["cdf"].pattern_tail == pytest.approx(1.0)
    assert fits["ldf"].pattern_tail is None
    assert fits["cdf"].tail == 1.0


def test_settings_read_back(tri):
    cl = ChainLadder(ldf={24: 1.1, 12: 1.5}, ldf_kind="ldf", ldf_weight={12: 0.25})
    assert cl.ldf == {24: 1.1, 12: 1.5}
    assert cl.ldf_kind == "ldf"
    assert cl.ldf_weight == {12: 0.25}
    assert "ldf={24: 1.1, 12: 1.5}" in repr(cl)
    assert 'ldf_kind="ldf"' in repr(cl)
    assert "ldf_weight={12: 0.25}" in repr(cl)
    plain = ChainLadder()
    assert plain.ldf is None and plain.ldf_kind is None and plain.ldf_weight is None
    assert "ldf" not in repr(plain)
    assert ChainLadder(ldf={12: 1.5}, ldf_kind="ldf", ldf_weight=0.5).ldf_weight == 0.5
    full = ChainLadder(ldf={12: 1.5}, ldf_kind="ldf")
    assert full.ldf_weight == 1.0 and "ldf_weight" not in repr(full)
    for method in [
        Mack(ldf={12: 1.5}, ldf_kind="ldf"),
        ExpectedLoss(ldf={12: 1.5}, ldf_kind="ldf"),
        BornhuetterFerguson(ldf={12: 1.5}, ldf_kind="ldf"),
        Benktander(ldf={12: 1.5}, ldf_kind="ldf"),
        CapeCod(ldf={12: 1.5}, ldf_kind="ldf"),
        OdpBootstrap(ldf={12: 1.5}, ldf_kind="ldf"),
        MackBootstrap(ldf={12: 1.5}, ldf_kind="ldf"),
    ]:
        assert method.ldf == {12: 1.5} and method.ldf_kind == "ldf"
        assert 'ldf={12: 1.5}, ldf_kind="ldf"' in repr(method)


def test_partial_pattern_for_the_immature_ages(tri):
    base = ChainLadder().fit(tri, "paid")
    fit = ChainLadder(ldf={12: 0.5, 24: 0.8}, ldf_kind="percent_developed").fit(tri, "paid")
    assert fit.selected_ldf == [pytest.approx(1.6), None, None, None]
    assert fit.ldf_weight == [1.0, 0.0, 0.0, 0.0]
    close(fit.ldf, [1.6] + base.ldf[1:])
    close(fit.estimated_ldf, base.ldf)
    close(fit.std_err[1:], base.std_err[1:])
    # The level 0.8 at 24 is rescaled to the estimated development from 24.
    assert fit.pattern_rescale == pytest.approx(1 / (0.8 * base.cdf[1]))
    # The youngest origin develops with the selected factor.
    assert fit.ultimate[-1] == pytest.approx(140.0 * 1.6 * base.cdf[1])


def test_blend_uniform_and_by_age(tri):
    base = ChainLadder().fit(tri, "paid")
    sel = {12: 1.6, 24: 1.1}
    fit = ChainLadder(ldf=sel, ldf_kind="ldf", ldf_weight=0.4).fit(tri, "paid")
    close(fit.blended_ldf[:2], [0.4 * s + 0.6 * e for s, e in zip(sel.values(), base.ldf)])
    close(fit.ldf, fit.blended_ldf)
    close(fit.std_err[:2], [0.6 * s for s in base.std_err[:2]])
    by_age = ChainLadder(ldf=sel, ldf_kind="ldf", ldf_weight={24: 0.0}).fit(tri, "paid")
    assert by_age.ldf_weight[:2] == [1.0, 0.0]
    close(by_age.ldf, [1.6] + base.ldf[1:])
    # Weight 0 is the estimate, everywhere.
    none = ChainLadder(ldf=sel, ldf_kind="ldf", ldf_weight=0.0).fit(tri, "paid")
    close(none.ultimate, base.ultimate)


def test_tail_rescale(tri):
    percent = {12: 0.5, 24: 0.8, 36: 0.9, 48: 0.93, 60: 0.95}
    fit = ChainLadder(ldf=percent, ldf_kind="percent_developed", tail=1.02).fit(tri, "paid")
    assert fit.tail == pytest.approx(1.02)
    assert fit.pattern_rescale == pytest.approx(1 / (0.95 * 1.02))
    # Rescaled, the percent developed at the oldest age times the tail is 100%.
    assert 0.95 * fit.pattern_rescale * fit.tail == pytest.approx(1.0)
    # The ratios are kept: the factor to ultimate at 12 is the given one,
    # divided by the rescale.
    assert fit.cdf[0] == pytest.approx((1 / 0.5) / fit.pattern_rescale)
    frame = fit.totals_frame()
    assert frame["pattern_rescale"][0] == pytest.approx(fit.pattern_rescale)
    # An age-to-age pattern stops at the last age and the tail multiplies on top.
    ldf = ChainLadder(ldf={12: 1.6}, ldf_kind="ldf", tail=1.02).fit(tri, "paid")
    assert ldf.pattern_rescale is None
    assert ldf.cdf[-1] == pytest.approx(1.02)


def test_expected_loss_methods_take_the_pattern(tri):
    percent = {12: 0.5, 24: 0.8, 36: 0.9, 48: 0.95, 60: 1.0}
    cdf = [1 / p for p in percent.values()]
    args = {"ldf": percent, "ldf_kind": "percent_developed"}
    latest = [172.0, 185.0, 192.0, 200.0, 140.0]
    exposure = [250.0, 260.0, 270.0, 280.0, 290.0]
    bf = BornhuetterFerguson(apriori=0.7, **args).fit(tri, "paid", "premium")
    close(bf.cdf, cdf, 1e-12)
    expected = [c + (1 - 1 / f) * 0.7 * e for c, f, e in zip(latest, cdf[::-1], exposure)]
    close(bf.ultimate, expected, 1e-12)
    el = ExpectedLoss(apriori=0.7, **args).fit(tri, "paid", "premium")
    close(el.cdf, cdf, 1e-12)
    bk = Benktander(apriori=0.7, n_iters=1, **args).fit(tri, "paid", "premium")
    close(bk.ultimate, bf.ultimate, 1e-12)
    cc = CapeCod(**args).fit(tri, "paid", "premium")
    close(cc.cdf, cdf, 1e-12)
    used = sum(e / f for e, f in zip(exposure, cdf[::-1]))
    assert cc.apriori[0] == pytest.approx(sum(latest) / used)
    frame = bf.development_frame()
    assert {"estimated_ldf", "selected_ldf", "ldf_weight", "blended_ldf"} <= set(frame.columns)


def test_mack_parameter_risk_scales_with_one_minus_weight(tri):
    base = Mack().fit(tri, "paid")
    own = {age: f for age, f in zip(base.development, base.ldf)}
    full = Mack(ldf=own, ldf_kind="ldf").fit(tri, "paid")
    close(full.ultimate, base.ultimate, 1e-12)
    close(full.process_risk, base.process_risk, 1e-12)
    assert full.parameter_risk == [0.0] * 5
    assert full.total_parameter_risk == 0.0
    part = Mack(ldf=own, ldf_kind="ldf", ldf_weight=0.4).fit(tri, "paid")
    close(part.parameter_risk, [0.6 * r for r in base.parameter_risk], 1e-9)
    assert part.total_parameter_risk == pytest.approx(0.6 * base.total_parameter_risk, rel=1e-9)
    close(part.process_risk, base.process_risk, 1e-12)
    close(part.std_err, [0.6 * s for s in base.std_err], 1e-12)
    close(part.estimated_std_err, base.std_err, 1e-12)
    # The one-year view takes the pattern: no parameter error at all leaves
    # each origin's one-year error its process error.
    cdr = full.claims_development_result()
    plain = base.claims_development_result()
    assert cdr.total_one_year_standard_error < plain.total_one_year_standard_error


def test_bootstraps_take_the_pattern(tri):
    base = Mack().fit(tri, "paid")
    own = {age: f for age, f in zip(base.development, base.ldf)}
    # Mack's bootstrap with every factor selected and no process error is
    # the deterministic reserve in every simulation.
    fixed = MackBootstrap(n_sims=200, seed=1, process="none", ldf=own, ldf_kind="ldf").fit(
        tri, "paid"
    )
    assert fixed.reserves.mean() == pytest.approx(base.total_reserve, rel=1e-9)
    assert fixed.reserves.variance() == pytest.approx(0.0, abs=1e-9)
    free = MackBootstrap(n_sims=200, seed=1, process="none").fit(tri, "paid")
    assert free.reserves.variance() > 0
    odp = OdpBootstrap(n_sims=500, seed=1, ldf=own, ldf_kind="ldf").fit(tri, "paid")
    odp_free = OdpBootstrap(n_sims=500, seed=1).fit(tri, "paid")
    assert odp.reserves.variance() < odp_free.reserves.variance()
    # One-year view: the bootstrap's pattern must be the method's.
    method = ChainLadder(ldf=own, ldf_kind="ldf")
    one = OdpBootstrap(n_sims=200, seed=1, ldf=own, ldf_kind="ldf").one_year(tri, "paid", method)
    assert math.isfinite(one.cdr.variance())
    MackBootstrap(n_sims=200, seed=1).one_year(tri, "paid", method)
    with pytest.raises(ValueError, match="pattern"):
        OdpBootstrap(n_sims=20, seed=1, ldf=own, ldf_kind="ldf").one_year(
            tri, "paid", ChainLadder()
        )


def test_one_year_compares_patterns_by_what_they_select(tri):
    base = Mack().fit(tri, "paid")
    own = {age: f for age, f in zip(base.development, base.ldf)}
    reversed_own = dict(reversed(list(own.items())))
    assert list(reversed_own) != list(own)
    method = ChainLadder(ldf=own, ldf_kind="ldf")
    # The same pattern with its keys in another order, or with the weight
    # given per age, is the method's.
    for kwargs in (
        {"ldf": reversed_own, "ldf_kind": "ldf"},
        {"ldf": own, "ldf_kind": "ldf", "ldf_weight": {12: 1.0}},
    ):
        for boot in (OdpBootstrap, MackBootstrap):
            fit = boot(n_sims=50, seed=1, **kwargs).one_year(tri, "paid", method)
            assert math.isfinite(fit.cdr.variance())
    # Another weight is not.
    with pytest.raises(ValueError, match="pattern"):
        OdpBootstrap(n_sims=20, seed=1, ldf=own, ldf_kind="ldf", ldf_weight=0.5).one_year(
            tri, "paid", method
        )


def test_one_year_bootstrap_takes_the_methods_pattern_by_default(tri):
    # As R's bootstrap_ldf = TRUE: a bootstrap without ldf takes the
    # method's, so its simulations are blended exactly as one given it.
    base = Mack().fit(tri, "paid")
    own = {age: f for age, f in zip(base.development, base.ldf)}
    method = ChainLadder(ldf=own, ldf_kind="ldf", ldf_weight=0.5)
    implicit = OdpBootstrap(n_sims=300, seed=3).one_year(tri, "paid", method)
    explicit = OdpBootstrap(n_sims=300, seed=3, ldf=own, ldf_kind="ldf", ldf_weight=0.5).one_year(
        tri, "paid", method
    )
    assert implicit.cdr.variance() == explicit.cdr.variance()
    free = OdpBootstrap(n_sims=300, seed=3).one_year(tri, "paid", method, bootstrap_ldf=False)
    assert free.cdr.variance() != implicit.cdr.variance()
    with pytest.raises(ValueError, match="bootstrap_ldf=False"):
        OdpBootstrap(n_sims=20, seed=1, ldf=own, ldf_kind="ldf").one_year(
            tri, "paid", method, bootstrap_ldf=False
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"ldf": {12: 1.5}}, ValueError, "ldf needs ldf_kind"),
        ({"ldf": {12: 1.5}, "ldf_kind": "LDF"}, ValueError, "ldf_kind must be"),
        ({"ldf": {12: 1.5}, "ldf_kind": "age_to_age"}, ValueError, "ldf_kind must be"),
        ({"ldf_kind": "ldf"}, ValueError, "ldf_kind is given without ldf"),
        ({"ldf_weight": 0.5}, ValueError, "ldf_weight is given without ldf"),
        ({"ldf": [1.5, 1.1], "ldf_kind": "ldf"}, TypeError, "dict from age"),
        ({"ldf": {"12": 1.5}, "ldf_kind": "ldf"}, TypeError, "whole number of months"),
        ({"ldf": {12: "a"}, "ldf_kind": "ldf"}, TypeError, "must be a number"),
        ({"ldf": {12: 1.5}, "ldf_kind": "ldf", "ldf_weight": "x"}, TypeError, "ldf_weight"),
        ({"ldf": {-12: 1.5}, "ldf_kind": "ldf"}, ValueError, "not an age"),
    ],
)
def test_argument_errors(kwargs, error, match):
    with pytest.raises(error, match=match):
        ChainLadder(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ldf": {18: 1.5}, "ldf_kind": "ldf"}, "age 18 months: not a development age"),
        ({"ldf": {60: 1.01}, "ldf_kind": "ldf"}, "past it: use a tail"),
        ({"ldf": {12: 0.0}, "ldf_kind": "ldf"}, "finite and positive"),
        ({"ldf": {12: 0.5, 24: 1.2}, "ldf_kind": "percent_developed"}, r"in \(0, 1\]"),
        ({"ldf": {12: 0.8, 24: 0.5}, "ldf_kind": "percent_developed"}, "not decrease"),
        ({"ldf": {12: 0.9, 24: 1.2}, "ldf_kind": "cdf"}, "at least 1"),
        ({"ldf": {12: 1.2, 24: 1.5}, "ldf_kind": "cdf"}, "not increase"),
        ({"ldf": {12: 1.5}, "ldf_kind": "cdf"}, "two or more ages"),
        ({"ldf": {12: 1.5}, "ldf_kind": "ldf", "ldf_weight": 1.5}, "between 0 and 1"),
        ({"ldf": {12: 1.5}, "ldf_kind": "ldf", "ldf_weight": {24: 0.5}}, "no selected factor"),
        (
            {"ldf": {36: 1.05}, "ldf_kind": "ldf", "tail": TailConstant(1.1, attachment_age=36)},
            "tail",
        ),
    ],
)
def test_fit_errors(tri, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ChainLadder(**kwargs).fit(tri, "paid")


def test_cumulative_pattern_may_end_at_the_tail_attachment(tri):
    fit = ChainLadder(
        ldf={12: 2.0, 24: 1.25, 36: 1.1},
        ldf_kind="cdf",
        tail=TailConstant(1.1, attachment_age=36),
    ).fit(tri, "paid")
    close(fit.ldf[:2], [1.6, 1.25 / 1.1])
    assert fit.pattern_rescale == pytest.approx(1.0)


def test_segments_report_the_pattern():
    tri = Triangle.from_long(
        YEARS * 2, AGES * 2, {"paid": PAID + [p / 2 for p in PAID]},
        keys={"lob": ["auto"] * 15 + ["home"] * 15},
    )
    fit = ChainLadder(ldf={12: 0.5, 24: 0.8}, ldf_kind="percent_developed").fit(tri, "paid")
    frame = fit.development_frame()
    assert list(frame.columns[-5:]) == [
        "estimated_ldf", "selected_ldf", "ldf_weight", "blended_ldf", "estimated_std_err"
    ]
    assert list(fit.totals_frame()["pattern_rescale"].round(12)) == [
        round(fit.segment(lob=lob).pattern_rescale, 12) for lob in ["auto", "home"]
    ]
    with pytest.raises(ValueError, match="single-segment"):
        fit.selected_ldf
    plain = ChainLadder().fit(tri, "paid")
    assert "selected_ldf" not in plain.development_frame().columns
    assert "pattern_rescale" not in plain.totals_frame().columns
    assert "pattern_tail" not in plain.totals_frame().columns


def test_a_pattern_to_the_oldest_age_without_a_tail_keeps_its_level_as_the_tail(tri):
    percent = {12: 0.5, 24: 0.8, 36: 0.9, 48: 0.93, 60: 0.95}
    fit = ChainLadder(ldf=percent, ldf_kind="percent_developed").fit(tri, "paid")
    assert fit.pattern_tail == pytest.approx(1 / 0.95)
    assert fit.tail == pytest.approx(1 / 0.95)
    assert fit.pattern_rescale == pytest.approx(1.0)
    assert fit.tail_std_err == 0.0
    # The youngest origin is 50% developed.
    assert fit.ultimate[-1] == pytest.approx(140.0 / 0.5)
    assert fit.totals_frame()["pattern_tail"][0] == pytest.approx(1 / 0.95)
    # With a tail, the level is rescaled to it and implies nothing.
    tailed = ChainLadder(ldf=percent, ldf_kind="percent_developed", tail=1.02).fit(tri, "paid")
    assert tailed.pattern_tail is None
    # A pattern that stops before the oldest age implies no tail.
    young = ChainLadder(ldf={12: 0.5, 24: 0.8}, ldf_kind="percent_developed").fit(tri, "paid")
    assert young.pattern_tail is None and young.tail == 1.0
    # Mack: the tail carries no parameter error unless one is given.
    mack = Mack(ldf=percent, ldf_kind="percent_developed", tail_sigma=1.0).fit(tri, "paid")
    assert mack.pattern_tail == pytest.approx(1 / 0.95)
    assert mack.chain_ladder.tail_std_err == 0.0
    given = Mack(
        ldf=percent, ldf_kind="percent_developed", tail_sigma=1.0, tail_std_err=0.01
    ).fit(tri, "paid")
    assert given.chain_ladder.tail_std_err == 0.01
    assert given.total_parameter_risk > mack.total_parameter_risk


def test_missing_ages_are_interpolated(tri):
    # 50% at 12 and 98% at 60: log cdf is linear in age between them.
    fit = ChainLadder(ldf={12: 0.5, 60: 0.98}, ldf_kind="percent_developed").fit(tri, "paid")
    step = (0.98 / 0.5) ** 0.25
    close(fit.selected_ldf, [step] * 4)
    assert fit.pattern_tail == pytest.approx(1 / 0.98)
    # The same factors given in full select the same.
    full = {age: 0.5 * step ** ((age - 12) // 12) for age in [12, 24, 36, 48, 60]}
    same = ChainLadder(ldf=full, ldf_kind="percent_developed").fit(tri, "paid")
    close(same.ultimate, fit.ultimate, 1e-10)


def test_bootstraps_keep_the_implied_tail(tri):
    percent = {12: 0.5, 24: 0.8, 36: 0.9, 48: 0.93, 60: 0.95}
    method = ChainLadder(ldf=percent, ldf_kind="percent_developed")
    odp = OdpBootstrap(n_sims=200, seed=1).one_year(tri, "paid", method)
    assert math.isfinite(odp.cdr.variance())
    # A method with a tail rescales the pattern's level to it, and the
    # bootstrap that takes the pattern reads it the same way: no implied
    # tail. A bootstrap's own ldf implies its tail, which is not the method's.
    tailed = ChainLadder(ldf=percent, ldf_kind="percent_developed", tail=1.02)
    for boot in [OdpBootstrap(n_sims=50, seed=1), MackBootstrap(n_sims=50, seed=1)]:
        taken = boot.one_year(tri, "paid", tailed)
        assert math.isfinite(taken.cdr.variance())
    with pytest.raises(ValueError, match="the tail the bootstrap's pattern implies"):
        OdpBootstrap(n_sims=50, seed=1, ldf=percent, ldf_kind="percent_developed").one_year(
            tri, "paid", tailed
        )
    # A bootstrap tail of the factor the method's pattern implies is the same tail.
    same = OdpBootstrap(n_sims=50, seed=1, tail=1 / 0.95).one_year(tri, "paid", method)
    assert math.isfinite(same.cdr.variance())
    life = OdpBootstrap(
        n_sims=200, seed=1, process="none", ldf=percent, ldf_kind="percent_developed"
    ).fit(tri, "paid")
    assert life.chain_ladder.tail == pytest.approx(1 / 0.95)
