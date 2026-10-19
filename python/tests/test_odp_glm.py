"""OdpGlm in Python: Taylor and McGuire, CAS Monograph 3, Table 5-4 (issue #333)."""

import math

import numpy as np
import pytest

from prospicio.reserving import ChainLadder, OdpGlm, Triangle

INCREMENTS = [  # Table 1-1, accident years 1988 to 1997
    [41821, 34729, 20147, 15965, 11285, 5924, 4775, 3742, 3435, 2958],
    [48167, 39495, 24444, 18178, 10840, 7379, 5683, 4758, 3959],
    [52058, 47459, 27359, 17916, 11448, 8846, 5869, 5391],
    [57251, 49510, 27036, 20871, 14304, 10552, 7742],
    [59213, 54129, 29566, 22484, 14114, 10000],
    [59475, 52076, 26836, 22332, 14756],
    [65607, 44648, 27062, 22655],
    [56748, 39315, 26748],
    [52212, 40030],
    [43962],
]

TABLE_5_4 = [  # forecast, RMSEP for 1989 to 1997, then the total
    (3398, 924), (8155, 1363), (14579, 1775), (22645, 2169), (31865, 2523),
    (45753, 3036), (60093, 3577), (80983, 4538), (105874, 6786), (373346, 14076),
]


def triangle():
    cumulative = [np.cumsum(row) for row in INCREMENTS]
    return Triangle.from_long(
        [1988 + k for k, row in enumerate(cumulative) for _ in row],
        [12 * (j + 1) for row in cumulative for j in range(len(row))],
        {"paid": [float(v) for row in cumulative for v in row]},
    )


def test_prediction_error_matches_table_5_4():
    fit = OdpGlm().fit(triangle(), "paid")
    table = fit.prediction_error()
    assert table["origin"][-1] == "Total"
    assert table["rmsep"][0] == 0.0
    assert math.isnan(table["cv"][0])
    for i, (forecast, rmsep) in enumerate(TABLE_5_4, start=1):
        assert table["reserve"][i] == pytest.approx(forecast, abs=0.5)
        assert table["rmsep"][i] == pytest.approx(rmsep, abs=0.5)
    total = fit.total_prediction_error()
    assert total["process_se"] == pytest.approx(6539, abs=0.5)
    assert total["parameter_se"] == pytest.approx(12465, abs=0.5)
    assert total["rmsep"] == pytest.approx(14076, abs=0.5)


def test_fit_matches_chain_ladder_and_exposes_the_glm():
    tri = triangle()
    fit = OdpGlm().fit(tri, "paid")
    cl = ChainLadder().fit(tri, "paid")
    assert fit.total_reserve == pytest.approx(cl.total_reserve, rel=1e-10)
    assert fit.reserves == pytest.approx(cl.reserve, rel=1e-9, abs=1e-6)
    assert len(fit.origins) == 10 and fit.reserves[0] == 0.0
    assert len(fit.future) == 45 and fit.future[0] == ("1989", 120)
    assert sum(fit.future_means) == pytest.approx(fit.total_reserve)
    assert fit.glm.dispersion == pytest.approx(fit.dispersion)
    assert fit.glm.predict(fit.future_design) == pytest.approx(fit.future_means)
    biased = fit.predictive_means()
    assert all(b > m for b, m in zip(biased, fit.future_means))
    assert "OdpGlmFit(origins=10" in repr(fit)
    assert repr(OdpGlm()) == "OdpGlm(tolerance=1e-10, max_iterations=100)"


def total_draws(pd):
    draws = np.array(pd.draw_matrix())
    return draws.sum(axis=0 if draws.shape[0] == pd.n_components else 1)


def test_predict_distribution():
    fit = OdpGlm().fit(triangle(), "paid")
    pd = fit.predict_distribution(4000, 7)
    assert pd.dims == ["origin", "development"]
    assert pd.n_components == 45
    totals = total_draws(pd)
    se = totals.std() / math.sqrt(4000)
    assert abs(totals.mean() - fit.total_reserve) < 4 * se
    fixed = total_draws(fit.predict_distribution(4000, 7, parameters="fixed"))
    assert fixed.std() < totals.std()
    with pytest.raises(ValueError, match="parameters"):
        fit.predict_distribution(10, 1, parameters="bayes")


def test_errors():
    tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 24], {"paid": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="ODP GLM"):
        OdpGlm().fit(tri, "paid")
