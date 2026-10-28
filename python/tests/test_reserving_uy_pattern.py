"""An accident-period development pattern on an underwriting basis
(``docs/design/reserving-v02.md``, decision 21, issue #633):
``UnderwritingBasis`` and ``ldf_underwriting`` on the methods that take
``ldf``, and the reading of an annual accident-year pattern at monthly
ages (decision 25, issue #646). The Rust core is checked in
``validation/tests/reserving_uy_pattern.rs``; these tests reproduce its
hand-computed parallelogram through the binding."""

from datetime import date

import pytest

from prospicio.reserving import (
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    ExpectedLoss,
    LatestDiagonal,
    Mack,
    MackBootstrap,
    OdpBootstrap,
    Triangle,
    UnderwritingBasis,
)

# An accident month reports t/60 of its ultimate at t months.
LINEAR = {t: t / 60.0 for t in range(1, 61)}


def parallelogram(a):
    """The parallelogram's share developed at age ``a`` of a linear pattern,
    by hand: the exposure (2t + 1)/288 to 11 months and (47 - 2t)/288 from
    12 to 23."""
    if a <= 12:
        return a * (a + 1) * (2 * a + 1) / (1728.0 * 60.0)
    if 23 <= a <= 60:
        return (a - 11.5) / 60.0
    if a >= 83:
        return 1.0
    raise ValueError(a)


def test_the_default_basis_is_the_parallelogram():
    b = UnderwritingBasis()
    assert b.policy_months == 12
    assert b.writing == pytest.approx([1 / 12] * 12, rel=1e-15)
    e = b.exposure()
    assert len(e) == 24
    for t, x in enumerate(e):
        want = (2 * t + 1) / 288 if t <= 11 else (47 - 2 * t) / 288
        assert x == pytest.approx(want, rel=1e-14)
    assert b.reading == "weibull"
    assert repr(UnderwritingBasis([1.0, 2.0], 6)) == (
        'UnderwritingBasis(writing=[1.0, 2.0], policy_months=6, reading="weibull")'
    )
    assert UnderwritingBasis(reading="ln-linear").reading == "ln-linear"
    assert UnderwritingBasis(reading="loglogistic").reading == "loglogistic"
    with pytest.raises(ValueError, match="reading must be"):
        UnderwritingBasis(reading="pchip")


def test_an_accident_month_pattern_converts_to_the_parallelogram():
    d = LatestDiagonal([2017, 2018, 2019], [83, 36, 12], [100.0, 100.0, 100.0])
    cl = ChainLadder(
        ldf=LINEAR,
        ldf_kind="percent_developed",
        ldf_origin="M",
        ldf_underwriting=UnderwritingBasis(),
    )
    assert cl.ldf_origin == "M"
    assert cl.ldf_underwriting.policy_months == 12
    assert "ldf_underwriting=UnderwritingBasis(" in repr(cl)
    assert cl.ldf == LINEAR
    fit = cl.fit_diagonal(d)
    for u, a in zip(fit.ultimate, [83, 36, 12]):
        assert u == pytest.approx(100.0 / parallelogram(a), rel=1e-13)


def test_an_accident_year_pattern_needs_no_ldf_origin():
    # The accident-year average of the linear pattern at every month: the
    # accident-year route gives the parallelogram too.
    def year(a):
        return sum(min(max(a - q, 0), 60) for q in range(12)) / (12 * 60.0)

    ay = {a: year(a) for a in range(1, 72)}
    d = LatestDiagonal([2017, 2018, 2019], [83, 36, 12], [100.0, 100.0, 100.0])
    fit = ChainLadder(
        ldf=ay, ldf_kind="percent_developed", ldf_underwriting=UnderwritingBasis()
    ).fit_diagonal(d)
    for u, a in zip(fit.ultimate, [83, 36, 12]):
        assert u == pytest.approx(100.0 / parallelogram(a), rel=1e-13)
    # On a triangle: the selected factor from 12 to 24 months.
    tri = Triangle.from_long([2018, 2018, 2019], [12, 24, 12], {"paid": [10.0, 30.0, 12.0]})
    fit = ChainLadder(
        ldf=ay, ldf_kind="percent_developed", ldf_underwriting=UnderwritingBasis()
    ).fit(tri, "paid")
    p24 = sum(
        e * min(max(24 - t, 0), 60) / 60.0
        for t, e in enumerate(UnderwritingBasis().exposure())
    )
    assert fit.selected_ldf[0] == pytest.approx(p24 / parallelogram(12), rel=1e-12)


def test_the_basis_and_its_errors():
    with pytest.raises(ValueError, match="at least one month"):
        UnderwritingBasis(policy_months=0)
    with pytest.raises(ValueError, match="not negative"):
        UnderwritingBasis([1.0, -1.0])
    with pytest.raises(ValueError, match="positive sum"):
        UnderwritingBasis([0.0, 0.0])
    with pytest.raises(ValueError, match="no months"):
        UnderwritingBasis([])
    # Seasonal writing and six-month policies on an accident-year pattern.
    d = LatestDiagonal([2019], [12], [100.0])
    seasonal = UnderwritingBasis([1.0, 2.0] * 6, 6)
    with pytest.raises(ValueError, match="give an accident-month pattern"):
        ChainLadder(
            ldf={12: 0.5, 24: 1.0}, ldf_kind="percent_developed", ldf_underwriting=seasonal
        ).fit_diagonal(d)
    # Even writing and six-month policies convert.
    ChainLadder(
        ldf={12: 0.5, 24: 1.0},
        ldf_kind="percent_developed",
        ldf_underwriting=UnderwritingBasis(policy_months=6),
    ).fit_diagonal(d)
    with pytest.raises(ValueError, match="ldf_underwriting is given without ldf"):
        ChainLadder(ldf_underwriting=UnderwritingBasis())
    for method in (Mack, ExpectedLoss, BornhuetterFerguson, Benktander, CapeCod,
                   OdpBootstrap, MackBootstrap):
        m = method(ldf={12: 1.5}, ldf_kind="ldf", ldf_underwriting=UnderwritingBasis())
        assert m.ldf_underwriting.policy_months == 12
        assert method(ldf={12: 1.5}, ldf_kind="ldf").ldf_underwriting is None
        with pytest.raises(ValueError, match="ldf_underwriting is given without ldf"):
            method(ldf_underwriting=UnderwritingBasis())


def test_the_writing_fits_in_the_triangles_origin_period():
    # Underwriting quarters: the default's year of writing is an error, and
    # three months of writing convert.
    tri = Triangle.from_long(
        [date(2023, 1, 1), date(2023, 1, 1), date(2023, 4, 1)],
        [3, 6, 3],
        {"paid": [10.0, 30.0, 12.0]},
        origin_grain="Q",
        development_grain="Q",
    )
    with pytest.raises(ValueError, match="longer than the triangle's origin period"):
        ChainLadder(
            ldf=LINEAR,
            ldf_kind="percent_developed",
            ldf_origin="M",
            ldf_underwriting=UnderwritingBasis(),
        ).fit(tri, "paid")
    quarter = UnderwritingBasis([1.0, 1.0, 1.0])
    fit = ChainLadder(
        ldf=LINEAR, ldf_kind="percent_developed", ldf_origin="M", ldf_underwriting=quarter
    ).fit(tri, "paid")

    # Even writing over a quarter and 12-month policies, by hand: the
    # exposure of 15 accident months, each on the linear pattern.
    def share(a):
        return sum(
            e * min(max(a - t, 0), 60) / 60.0 for t, e in enumerate(quarter.exposure())
        )

    assert len(quarter.exposure()) == 15
    assert fit.selected_ldf[0] == pytest.approx(share(6) / share(3), rel=1e-12)


def book_year(a):
    """The book's accident year at age ``a``: the average of its twelve
    months, each reporting ``1 - 0.9^t``, held from 60 months."""
    return sum(1.0 - 0.9 ** min(a - q, 60) for q in range(12)) / 12.0


def test_an_annual_accident_year_pattern_is_read_at_monthly_ages():
    # The Rust validation's book (validation/tests/reserving_uy_pattern.rs):
    # its underwriting year's own share developed at 12 months.
    annual = {a: book_year(a) for a in range(12, 121, 12)}

    def uy_truth(a):
        e = UnderwritingBasis().exposure()
        return sum(x * (1.0 - 0.9 ** min(a - t, 60)) for t, x in enumerate(e) if a - t > 0)

    d = LatestDiagonal([2019], [12], [100.0])
    shares = {}
    for reading in ("weibull", "loglogistic", "ln-linear"):
        fit = ChainLadder(
            ldf=annual,
            ldf_kind="percent_developed",
            ldf_underwriting=UnderwritingBasis(reading=reading),
        ).fit_diagonal(d)
        shares[reading] = 100.0 / fit.ultimate[0] / uy_truth(12)
    # The design study's ratios (decision 25): the Weibull 0.9989, the
    # loglogistic 0.9308, decision 18's rule 1.3116.
    assert shares["weibull"] == pytest.approx(0.998928, abs=1e-5)
    assert shares["loglogistic"] == pytest.approx(0.930813, abs=1e-5)
    assert shares["ln-linear"] == pytest.approx(1.311567, abs=1e-5)
    # An accident-year pattern at every month converts the same under every
    # reading.
    every = {a: book_year(a) if a >= 12 else sum(
        1.0 - 0.9 ** (a - q) for q in range(a)) / 12.0 for a in range(1, 121)}
    ultimates = {
        reading: ChainLadder(
            ldf=every,
            ldf_kind="percent_developed",
            ldf_underwriting=UnderwritingBasis(reading=reading),
        ).fit_diagonal(d).ultimate[0]
        for reading in ("weibull", "ln-linear")
    }
    assert ultimates["weibull"] == ultimates["ln-linear"]
    # A single value has no shape for a curve to read.
    with pytest.raises(ValueError, match="single value"):
        ChainLadder(
            ldf={12: 0.5}, ldf_kind="percent_developed", ldf_underwriting=UnderwritingBasis()
        ).fit_diagonal(d)
