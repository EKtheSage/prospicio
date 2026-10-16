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


def test_credibility_premiums_match_actuar_hierarchy():
    # actuar::cm(~cohort + cohort:state, hachemeister) with the
    # Buhlmann-Gisler estimator: its structure parameters plugged in.
    import csv
    from pathlib import Path

    from prospicio.credibility import credibility_premiums

    root = Path(__file__).resolve().parents[2]
    ratios = [[0.0] * 12 for _ in range(5)]
    weights = [[0.0] * 12 for _ in range(5)]
    with open(root / "validation/reference/hachemeister.csv") as f:
        for row in csv.DictReader(f):
            i, j = int(row["state"]) - 1, int(row["quarter"]) - 1
            ratios[i][j] = float(row["ratio"])
            weights[i][j] = float(row["weight"])
    p = credibility_premiums(
        ratios,
        weights,
        within_variance=139120025.92528549,
        between_variances=[87263.695756774905, 13414.843135533525],
        levels=[[0, 1, 0, 1, 1]],
    )
    assert p["collective_mean"] == pytest.approx(1742.22, abs=0.01)
    assert p["premiums"] == pytest.approx(
        [2049.733, 1522.032, 1864.280, 1488.504, 1587.097], abs=1e-3
    )
    assert p["level_premiums"][0] == pytest.approx([1941.675, 1542.765], abs=1e-3)


def test_hierarchical_credibility_fits_and_predicts():
    from prospicio.credibility import hierarchical_credibility

    x = [[1.0, 2.0, 3.0, 2.0], [4.0, 5.0, 6.0, 5.0], [2.0, 3.0, 4.0, 3.0], [3.0, 3.5, 2.5, 3.0]]
    w = [[1.0] * 4] * 4
    fit = hierarchical_credibility(
        x,
        w,
        levels=[[0, 0, 1, 1]],
        within_prior=("half_cauchy", 5.0),
        chains=2,
        tune=300,
        draws=300,
        seed=4,
    )
    assert len(fit.premiums) == 4 and len(fit.level_premiums[0]) == 2
    assert fit.priors["within"] == ("half_cauchy", 5.0)
    assert len(fit.priors["between"]) == 2
    names = [s["name"] for s in fit.summary()]
    assert names == ["within_variance", "between_variance[0]", "between_variance[1]"]
    pd = fit.predictive(weights=[1.0] * 4, n_sims=2_000, seed=1)
    assert pd.dims == ["group"]
    with pytest.raises(ValueError):
        hierarchical_credibility(x, w, within_prior=("lognormal", 1.0))
