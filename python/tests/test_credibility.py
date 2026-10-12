import math

import pytest

from prospicio.credibility import LimitedFluctuation, buhlmann_straub


def test_limited_fluctuation():
    lf = LimitedFluctuation(0.90, 0.05)
    n = lf.frequency_standard()
    assert n == pytest.approx(1082.217, abs=1e-3)
    assert lf.pure_premium_standard(2.0) == pytest.approx(5 * n)
    assert lf.credibility(n / 4, n) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        LimitedFluctuation(1.0, 0.05)


def test_buhlmann_straub():
    # Hand-worked Bühlmann case; the actuar parity is in
    # validation/tests/pricing.rs.
    fit = buhlmann_straub([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])
    assert fit.within_variance == pytest.approx(4.0)
    assert fit.between_variance == pytest.approx(8 - 4 / 3)
    assert fit.k == pytest.approx(4.0 / (8 - 4 / 3))
    z = 3 / (3 + fit.k)
    assert fit.credibility == pytest.approx([z, z])
    assert fit.premiums[0] == pytest.approx(fit.collective_mean + z * (2 - fit.collective_mean))
    nan = math.nan
    fit = buhlmann_straub(
        [[1.0, 2.0, nan], [4.0, 5.0, 9.0], [2.0, 2.5, 3.0]],
        [[10.0, 20.0, nan], [5.0, 5.0, 8.0], [30.0, 40.0, 50.0]],
        estimator="iterative",
    )
    assert fit.group_weights == [30.0, 18.0, 120.0]
    with pytest.raises(ValueError):
        buhlmann_straub([[1.0, 2.0]], estimator="unbiased")
    with pytest.raises(ValueError):
        buhlmann_straub([[1.0, 2.0], [3.0, 4.0]], estimator="bayes")
