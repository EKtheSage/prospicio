"""Casualty exposure rating (prospicio_pricing::casualty) from Python."""

import math

import pytest

from prospicio.distributions import Pareto
from prospicio.pricing import (
    CasualtyProfile,
    CasualtyRating,
    ElfTable,
    IlfTable,
    LargeLossExperience,
    ParetoAnchor,
    poisson_tail,
)

# Clark's workers compensation example: (state, premium, ELR, ELF at 250K,
# ELF at 1M, factor, treaty loss), 750,000 xs 250,000.
CLARK = [
    ("AL", 100e3, 0.70, 0.030, 0.006, 0.024, 1680.0),
    ("AL", 100e3, 0.70, 0.040, 0.008, 0.032, 2240.0),
    ("NJ", 100e3, 0.85, 0.070, 0.020, 0.050, 4250.0),
    ("NJ", 100e3, 0.85, 0.100, 0.035, 0.065, 5525.0),
]


def test_clarks_workers_compensation_example():
    book = CasualtyProfile(
        [ElfTable([250e3, 1e6], [r[3], r[4]]) for r in CLARK],
        limit=[math.inf] * 4,
        premium=[r[1] for r in CLARK],
        elr=[r[2] for r in CLARK],
        curve=[0, 1, 2, 3],
        tag=[r[0] for r in CLARK],
    )
    assert len(book) == 4
    r = book.rate(750e3, 250e3, alae="in_severity")
    assert isinstance(r, CasualtyRating)
    rows = r.rows()
    assert rows["factor"] == pytest.approx([c[5] for c in CLARK], abs=1e-15)
    assert rows["ceded"] == pytest.approx([c[6] for c in CLARK], abs=1e-9)
    assert r.ceded == pytest.approx(13_695.0, abs=1e-9)
    assert round(r.loss_cost * 100, 2) == 3.42
    assert r.without("AL").ceded == pytest.approx(9_775.0, abs=1e-9)
    g = r.by_tag()
    assert g["tag"] == ["AL", "NJ"]
    assert g["premium"] == [200e3, 200e3]


def test_alae_has_no_default_and_matches_clarks_on_top_formula():
    sev = Pareto(100e3, 1.3)
    book = CasualtyProfile([sev], limit=[1e6], premium=[1.0], elr=[1.0])
    with pytest.raises(TypeError):
        book.rate(500e3, 250e3)
    with pytest.raises(ValueError):
        book.rate(500e3, 250e3, alae="pro_rata")
    with pytest.raises(ValueError):
        book.rate(500e3, 250e3, alae="in_severity", loading=0.1)
    f = book.exposure_factors(500e3, 250e3, alae="on_top", loading=0.5)[0]
    want = (sev.lev(min(1e6, 750e3 / 1.5)) - sev.lev(min(1e6, 250e3 / 1.5))) / sev.lev(1e6)
    assert f == pytest.approx(want, rel=1e-13)
    r = book.rate(500e3, 250e3, alae="pro_rata", loading=0.1)
    assert r.ceded == pytest.approx(1.1 * book.exposure_factors(500e3, 250e3, alae="pro_rata",
                                                                  loading=0.1)[0], rel=1e-14)


def test_the_forms_sweep_anchor_and_count_test_agree():
    sev = Pareto(1e6, 1.4)
    book = CasualtyProfile(
        [sev, IlfTable([1e6, 5e6, 10e6], [1.0, 1.5, 1.7])],
        limit=[10e6, 5e6],
        premium=[2e6, 1e6],
        elr=[0.65, 0.6],
        curve=[0, 1],
    )
    counts = book.occurrences_above(1e6)
    assert counts[0] == pytest.approx(2e6 * 0.65 / sev.layer(10e6, 0.0), rel=1e-14)
    # The table's slope above 1M is 0.5 / 4M per LEV unit of 1.5.
    assert counts[1] == pytest.approx(0.6e6 / 1.5 * 0.5 / 4e6, rel=1e-14)
    one = CasualtyProfile([sev], limit=[10e6], premium=[2e6], elr=[0.65])
    ilf = one.rate(5e6, 5e6, alae="in_severity")
    freq = one.rate_by_frequency(5e6, 5e6, alae="in_severity", threshold=1e6,
                                 counts=one.occurrences_above(1e6), excess=sev)
    assert freq.ceded == pytest.approx(ilf.ceded, rel=1e-12)
    s = book.sweep(5e6, 5e6, alae="in_severity", threshold=1e6, counts=counts,
                   alphas=[1.0, 1.4, 2.0])
    assert s["alpha"] == [1.0, 1.4, 2.0]
    assert s["ceded"][0] > s["ceded"][1] > s["ceded"][2]
    assert s["loss_cost"][1] == pytest.approx(s["ceded"][1] / 3e6, rel=1e-15)

    a = ParetoAnchor.fit(1e6, [1.1e6, 1.7e6, 2.4e6, 2.0e6, 0.5e6],
                         censored=[False, False, False, True, False])
    assert (a.exceedances, a.censored) == (4, 1)
    want = 3 / sum(math.log(v / 1e6) for v in [1.1e6, 1.7e6, 2.4e6, 2.0e6])
    assert a.alpha == pytest.approx(want, rel=1e-14)
    lo, hi = a.interval(1.0)
    assert hi - lo == pytest.approx(2 * a.standard_error, rel=1e-12)

    t = book.count_test([1e6, 3e6], threshold=1e6, counts=counts, excess=sev,
                        exposure_years=5.0, observed=[1.2e6, 3.5e6, 0.9e6])
    assert t["observed"] == [2, 1]
    lam = 5.0 * book.expected_claims(1e6, 1e6, counts, sev)
    assert t["expected"][0] == pytest.approx(lam, rel=1e-14)
    assert t["p_value"][0] == pytest.approx(poisson_tail(lam, 2), rel=1e-15)
    assert poisson_tail(3.0, 0) == pytest.approx(math.exp(-3.0), rel=1e-15)


def test_burning_cost_with_and_without_a_segment():
    e = LargeLossExperience(
        ["gl-1", "wf-1", "gl-2"], [2020.5, 2021.5, 2022.5], [1.5e6, 1.8e6, 1.2e6]
    )
    p = e.prepare(factors=[1.0, 1.0, 1.0])
    assert p.trended == [1.5e6, 1.8e6, 1.2e6]
    assert p.without(["wf-1"]).trended == [1.5e6, 1.2e6]
