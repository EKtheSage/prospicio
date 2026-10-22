"""Excess-layer development from a ground-up triangle
(``docs/design/reserving-excess.md``, PR 3).

The Rust core is checked against the literature in
``validation/tests/reserving_excess.rs``; these tests reproduce the same
printed targets through the Python bindings (Bloom and Patrik, CAS E-Forum
Fall 2013; Pinto and Gogol, PCAS 74; Bear, PCAS 79) and check the bindings'
arguments, conversions and errors. ``R/prospicio/tests/test-reserving-excess.R``
asserts the same numbers."""

import math
import pickle

import pytest

from prospicio.distributions import Grid, Lognormal, Pareto
from prospicio.pricing import Mbbefd
from prospicio.reserving import (
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    ExcessLayer,
    ExcessPattern,
    ExpectedLoss,
    Mack,
    ParetoShape,
    ScaleShift,
    TailConstant,
    Triangle,
    layer_at_shift,
    layer_exposure,
    layer_share,
)

# The four claims of Bloom and Patrik's example at 12 months, and the
# ultimate claims 450, 1,687.5, 1,125 and 2,250 times the scale, each with
# probability 1/4, as a grid of step 112.5 times the scale.
CLAIMS_TOTAL = 100.0 + 375.0 + 250.0 + 500.0


def ultimate(scale):
    probs = [0.0] * 21
    for j in (4, 15, 10, 20):
        probs[j] = 0.25
    return Grid(112.5 * scale, probs)


def ground_up(scale):
    total = CLAIMS_TOTAL * scale
    return Triangle.from_long(
        [2020, 2020, 2020], [12, 24, 36], {"incurred": [total, total * 3.6, total * 4.5]}
    )


def pattern(scale, layer):
    return ExcessPattern.from_ground_up(
        ground_up(scale), "incurred", layer, ScaleShift(ultimate(scale))
    )


def test_bloom_patrik_year_one():
    p = pattern(1.0, ExcessLayer(350.0))
    assert p.ages == [12, 24, 36]
    assert p.layer_ldf[0] == pytest.approx(17.2, abs=1e-11)
    assert p.layer_ldf[1] == pytest.approx(4112.5 / 3010.0, abs=1e-11)
    assert p.layer_cdf[0] == pytest.approx(23.5, abs=1e-11)
    assert p.shifts == pytest.approx([4.5, 1.25, 1.0], abs=1e-12)
    assert p.ground_up_ldf == pytest.approx([3.6, 1.25], abs=1e-12)
    assert p.tail == 1.0


def test_bloom_patrik_years_two_and_three():
    two = pattern(1.05, ExcessLayer(350.0))
    assert two.layer_ldf[0] == pytest.approx(14.768, abs=5e-4)
    assert two.layer_ldf[1] == pytest.approx(1.3583, abs=5e-5)
    assert two.layer_cdf[0] == pytest.approx(20.06, abs=5e-3)
    three = pattern(1.1025, ExcessLayer(350.0))
    assert three.layer_ldf[0] == pytest.approx(13.0797, abs=5e-5)
    assert three.layer_ldf[1] == pytest.approx(1.3511, abs=5e-5)
    assert three.layer_cdf[0] == pytest.approx(17.6719, abs=5e-5)


def test_the_layer_values_at_each_shift():
    # 175, 3,010 and 4,112.5 per the four claims at 12, 24 and 36 months.
    layer = ExcessLayer(350.0)
    u = ultimate(1.0)
    for shift, want in [(4.5, 175.0), (1.25, 3010.0), (1.0, 4112.5)]:
        assert 4.0 * layer_at_shift(u, layer, shift) == pytest.approx(want, abs=1e-9)


def test_a_sparse_layer_triangle_runs_on_the_derived_pattern():
    p = pattern(1.0, ExcessLayer(350.0))
    # The layer triangle's 12-month column is empty: its own factor does
    # not exist, and the derived pattern fully selected fills it in.
    years = [2020, 2020, 2020, 2021, 2021, 2022]
    ages = [12, 24, 36, 12, 24, 12]
    layer_tri = Triangle.from_long(
        years, ages, {"layer": [0.0, 3010.0, 4112.5, 0.0, 3231.0, 0.0]}
    )
    with pytest.raises(ValueError):
        ChainLadder().fit(layer_tri, "layer")
    sel = p.selected()
    assert sel["ldf_kind"] == "ldf" and sel["ldf_weight"] == 1.0
    assert list(sel["ldf"]) == [12, 24]
    fit = ChainLadder(tail=p.tail, **sel).fit(layer_tri, "layer")
    assert fit.ldf[0] == pytest.approx(17.2, abs=1e-11)
    assert fit.ldf[1] == pytest.approx(4112.5 / 3010.0, abs=1e-11)
    assert fit.ultimate[1] == pytest.approx(3231.0 * 4112.5 / 3010.0, abs=1e-9)
    # The same through Mack and the expected-loss methods.
    full = Triangle.from_long(
        [2020] * 3 + [2021] * 3 + [2022] * 3 + [2023] * 2 + [2024],
        [12, 24, 36] * 1 + [12, 24, 36] + [12, 24, 36] + [12, 24] + [12],
        {"layer": [175.0, 3010.0, 4112.5] * 3 + [175.0, 3010.0] + [175.0],
         "premium": [10000.0] * 12},
    )
    for method in (
        Mack(tail=p.tail, **p.selected()),
        ExpectedLoss(tail=p.tail, **p.selected()),
        BornhuetterFerguson(tail=p.tail, **p.selected()),
        CapeCod(tail=p.tail, **p.selected()),
    ):
        assert method.ldf_kind == "ldf"
        assert method.ldf == p.ldf
    assert Mack(tail=p.tail, **p.selected()).fit(full, "layer").ldf[0] == pytest.approx(
        17.2, abs=1e-11
    )
    bf = BornhuetterFerguson(apriori=0.2, tail=p.tail, **p.selected())
    assert bf.fit(full, "layer", "premium").total_reserve > 0


def test_a_weight_blends_with_the_triangles_own_factors():
    p = pattern(1.0, ExcessLayer(350.0))
    sel = p.selected(0.5)
    assert sel["ldf_weight"] == 0.5
    by_age = p.selected({12: 1.0, 24: 0.25})
    assert by_age["ldf_weight"] == {12: 1.0, 24: 0.25}
    with pytest.raises(TypeError):
        p.selected("half")
    with pytest.raises(TypeError):
        p.selected(True)


def test_pinto_gogol_shape_change_p247():
    # k = 25,000, retention 100,000, q from 1.75 to 1.25, d = 2.5 for the
    # losses excess of 25,000: 2.5 (4)^0.5 = 5.0.
    tri = Triangle.from_long([2020, 2020], [12, 24], {"base": [1.0, 2.5]})
    model = ParetoShape(
        25_000.0, {12: 1.75, 24: 1.25}, 1.25, base=ExcessLayer(25_000.0)
    )
    p = ExcessPattern.from_ground_up(tri, "base", ExcessLayer(100_000.0), model)
    assert p.layer_ldf[0] == pytest.approx(5.0, abs=1e-12)
    assert p.tail == 1.0
    assert p.shifts == []
    # The same from the counts above k: n_j / n_i = 2.5 * 0.25 / 0.75.
    tri = Triangle.from_long([2020, 2020], [12, 24], {"base": [1.0, 2.5 * 0.25 / 0.75]})
    model = ParetoShape(25_000.0, {12: 1.75, 24: 1.25}, 1.25, base="counts")
    p = ExcessPattern.from_ground_up(tri, "base", ExcessLayer(100_000.0), model)
    assert p.layer_ldf[0] == pytest.approx(5.0, abs=1e-12)


def test_bear_bounded_layer_p138():
    # SPP parameters 1.25 at 24 months and 1.10 at ultimate above 100,000;
    # the layer 100,000 to 300,000 develops by 3.5, so the layer 200,000 to
    # 600,000 by 3.5 (2.0)^0.15 = 3.8835, printed 3.88.
    tri = Triangle.from_long([2020, 2020], [12, 24], {"base": [1.0, 3.5]})
    model = ParetoShape(
        100_000.0, {12: 1.25, 24: 1.10}, 1.10, base=ExcessLayer(100_000.0, 200_000.0)
    )
    p = ExcessPattern.from_ground_up(tri, "base", ExcessLayer(200_000.0, 400_000.0), model)
    assert p.layer_ldf[0] == pytest.approx(3.5 * 2.0**0.15, abs=1e-12)
    assert p.layer_ldf[0] == pytest.approx(3.8835, abs=5e-5)
    assert p.layer_ldf[0] == pytest.approx(3.88, abs=5e-3)


def test_credibility_weights():
    # Bloom and Patrik's factors 17.2 and 4112.5 / 3010 on layer triangles
    # whose every ratio equals them (172 to 235 is 4112.5 / 3010), so the
    # cap is idle and the weights are k / (n + k).
    p = pattern(1.0, ExcessLayer(350.0))
    rows = [[10.0, 172.0, 235.0], [10.0, 172.0, 235.0], [10.0, 172.0], [10.0]]
    counts = [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [5.0, 5.0], [1.0]]
    years, ages, loss, cnt = [], [], [], []
    for k, (row, crow) in enumerate(zip(rows, counts)):
        for d, (v, c) in enumerate(zip(row, crow)):
            years.append(2020 + k)
            ages.append(12 * (d + 1))
            loss.append(v)
            cnt.append(c)
    tri = Triangle.from_long(years, ages, {"layer": loss, "count": cnt})
    # Without counts n is the origins with a positive earlier value: 3 and 2.
    w = p.credibility_weights(tri, "layer", 3.0, 0.1)
    assert list(w) == [12, 24]
    assert w[12] == pytest.approx(0.5, abs=1e-12)
    assert w[24] == pytest.approx(0.6, abs=1e-12)
    # With counts n is the claims at the earlier age: 10 and 5.
    w = p.credibility_weights(tri, "layer", 3.0, 0.1, counts="count")
    assert w[12] == pytest.approx(3.0 / 13.0, abs=1e-12)
    assert w[24] == pytest.approx(3.0 / 8.0, abs=1e-12)
    # The weights go straight into a method.
    fit = ChainLadder(tail=p.tail, **p.selected(w)).fit(tri, "layer")
    assert fit.ldf_weight == pytest.approx([w[12], w[24]], abs=1e-12)
    # k and the cap are required and positive.
    with pytest.raises(ValueError):
        p.credibility_weights(tri, "layer", 0.0, 0.1)
    with pytest.raises(ValueError):
        p.credibility_weights(tri, "layer", 1.0, 0.0)
    with pytest.raises(TypeError):
        p.credibility_weights(tri, "layer", 1.0)


def test_one_huge_link_ratio_does_not_dominate_the_blend():
    # The whole loss (a layer from 0) returns the ground-up factors, here 2
    # and 1, so the derived pattern is known exactly.
    ground = Triangle.from_long([2020] * 3, [12, 24, 36], {"values": [10.0, 20.0, 20.0]})
    p = ExcessPattern.from_ground_up(
        ground, "values", ExcessLayer(0.0), ScaleShift(Lognormal(8.0, 1.0))
    )
    assert p.layer_ldf == pytest.approx([2.0, 1.0], abs=1e-12)
    # Three origins develop by 2; the fourth from 0.1 to 50 (500).
    tri = Triangle.from_long(
        [2017] * 3 + [2018] * 3 + [2019] * 2 + [2020] * 2,
        [12, 24, 36] + [12, 24, 36] + [12, 24] + [12, 24],
        {"values": [10.0, 20.0, 20.0] * 2 + [10.0, 20.0] + [0.1, 50.0]},
    )
    own = 110.0 / 30.1
    pull = own - 2.0
    # Uncapped (a share of 10, never binding): n = 4, k = 1, so the weight
    # on the derived pattern is 1 - 0.8.
    loose = p.credibility_weights(tri, "values", 1.0, 10.0)
    assert loose[12] == pytest.approx(0.2, abs=1e-15)
    # Capped at 10%: the blend moves by exactly 10% of itself.
    capped = p.credibility_weights(tri, "values", 1.0, 0.1)
    z = 1.0 - capped[12]
    blend = 2.0 + z * pull
    assert z * pull == pytest.approx(0.1 * blend, abs=1e-12)
    assert blend < 2.25
    # The chain ladder applies it.
    fit = ChainLadder(tail=p.tail, **p.selected(capped)).fit(tri, "values")
    assert fit.ldf[0] == pytest.approx(blend, abs=1e-12)


def test_layer_share_and_exposure():
    assert layer_share(Pareto(1.0, 2.0), ExcessLayer(4.0)) == pytest.approx(0.125, abs=1e-12)
    # The layer at shift 1 over the mean is the share.
    ln = Lognormal(8.0, 1.0)
    layer = ExcessLayer(5000.0, 20000.0)
    assert layer_at_shift(ln, layer, 1.0) / ln.mean() == pytest.approx(
        layer_share(ln, layer), abs=1e-12
    )
    tri = Triangle.from_long(
        [2020, 2020, 2021],
        [12, 24, 12],
        {"paid": [10.0, 15.0, 11.0], "premium": [100.0, 100.0, 120.0]},
    )
    out = layer_exposure(tri, "premium", [0.1, 0.2], "layer_premium")
    assert "layer_premium" in out.columns
    assert tri.columns == ["paid", "premium"]
    with pytest.raises(ValueError):
        layer_exposure(tri, "premium", [0.1, 0.2, 0.3], "x")
    with pytest.raises(ValueError):
        layer_exposure(tri, "premium", 1.5, "x")
    # The exposure-curve route (decision 2): the curve's own layer share.
    share = Mbbefd.swiss_re(3.0).layer_share(5e6, 5e6, 10e6)
    assert 0.0 < share < 1.0
    one = layer_exposure(tri, "premium", share, "layer_premium")
    assert one.columns == ["paid", "premium", "layer_premium"]
    prem = one.values
    assert math.isfinite(share)
    assert prem is not None


def test_a_capped_ground_up_column():
    # The layer from 0 to the cap is the capped column itself, so its
    # factors are the ground-up ones, whatever the severity.
    ln = Lognormal(6.0, 1.0)
    tri = Triangle.from_long(
        [2020] * 3, [12, 24, 36], {"capped": [800.0, 1500.0, 1700.0]}
    )
    p = ExcessPattern.from_ground_up(
        tri, "capped", ExcessLayer(0.0, 1000.0), ScaleShift(ln, ground_up_cap=1000.0)
    )
    assert p.layer_ldf == pytest.approx(p.ground_up_ldf, abs=1e-9)
    assert ScaleShift(ln, ground_up_cap=1000.0).ground_up_cap == 1000.0
    with pytest.raises(ValueError):
        ExcessPattern.from_ground_up(
            tri, "capped", ExcessLayer(0.0, 1000.0), ScaleShift(ln, ground_up_cap=-1.0)
        )


def test_a_tail_and_counts():
    tri = Triangle.from_long(
        [2020, 2020, 2020],
        [12, 24, 36],
        {"incurred": [1225.0, 4410.0, 5512.5], "n": [10.0, 12.0, 12.0]},
    )
    model = ScaleShift(ultimate(1.0))
    plain = ExcessPattern.from_ground_up(tri, "incurred", ExcessLayer(350.0), model)
    with_counts = ExcessPattern.from_ground_up(
        tri, "incurred", ExcessLayer(350.0), model, counts="n"
    )
    assert with_counts.layer_ldf != plain.layer_ldf
    tail = ExcessPattern.from_ground_up(tri, "incurred", ExcessLayer(350.0), model, tail=1.1)
    assert tail.ground_up_tail == pytest.approx(1.1)
    assert tail.tail != 1.0
    via_object = ExcessPattern.from_ground_up(
        tri, "incurred", ExcessLayer(350.0), model, tail=TailConstant(1.1)
    )
    assert via_object.layer_cdf == pytest.approx(tail.layer_cdf, abs=1e-12)


def test_errors_and_objects():
    tri = ground_up(1.0)
    model = ScaleShift(ultimate(1.0))
    with pytest.raises(ValueError, match="paid"):
        ExcessPattern.from_ground_up(tri, "incurred", ExcessLayer(350.0), model, basis="paid")
    with pytest.raises(ValueError):
        ExcessPattern.from_ground_up(tri, "incurred", ExcessLayer(350.0), model, basis="x")
    with pytest.raises(TypeError):
        ExcessPattern.from_ground_up(tri, "incurred", 350.0, model)
    with pytest.raises(TypeError):
        ExcessPattern.from_ground_up(tri, "incurred", ExcessLayer(350.0), "scale")
    with pytest.raises(ValueError):
        ExcessPattern.from_ground_up(tri, "nope", ExcessLayer(350.0), model)
    with pytest.raises(ValueError):
        ExcessLayer(-1.0)
    with pytest.raises(ValueError):
        ExcessLayer(0.0, 0.0)
    with pytest.raises(ValueError):
        ParetoShape(1.0, {12: 1.0}, 1.0, base="losses")
    # A Pareto shape model takes no count column.
    shape = ParetoShape(100.0, {12: 1.5, 24: 1.2, 36: 1.1}, 1.1)
    with pytest.raises(ValueError):
        ExcessPattern.from_ground_up(tri, "incurred", ExcessLayer(350.0), shape, counts="incurred")
    with pytest.raises(ValueError):
        layer_at_shift(ultimate(1.0), ExcessLayer(350.0), 0.0)
    layer = ExcessLayer(350.0, 650.0)
    assert layer == ExcessLayer(350.0, 650.0)
    assert layer != ExcessLayer(350.0)
    assert pickle.loads(pickle.dumps(layer)) == layer
    assert layer.exhaustion == 1000.0
    assert ExcessLayer(350.0).limit == math.inf
    assert repr(ExcessLayer(350.0)) == "ExcessLayer(attachment=350.0, limit=inf)"
    assert "ExcessPattern" in repr(pattern(1.0, layer))
