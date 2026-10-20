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


HEAT_MAP = [  # Figure 6-22, actual over fitted in percent
    [98, 100, 100, 104, 113, 87, 96, 92, 100, 100],
    [99, 99, 106, 103, 95, 95, 99, 102, 100],
    [96, 108, 107, 91, 90, 102, 92, 104],
    [97, 103, 96, 97, 103, 111, 111],
    [95, 107, 100, 100, 97, 100],
    [98, 105, 93, 101, 104],
    [109, 91, 95, 104],
    [106, 90, 105],
    [103, 97],
    [100],
]


def test_residuals_match_chapter_6():
    fit = OdpGlm().fit(triangle(), "paid")
    r = fit.residuals()
    assert len(r["actual"]) == 55
    assert r["origin"][0] == "1988" and r["development"][0] == 12
    assert r["calendar"][0] == "1988" and r["calendar_index"][54] == 9
    assert [i for i, x in enumerate(r["standardized_deviance"]) if math.isnan(x)] == [9, 54]
    assert sum(r["leverage"]) == pytest.approx(19)
    assert [round(100 * x) for x in r["actual_to_fitted"]] == [v for row in HEAT_MAP for v in row]

    # The guide's spread (divisor n) by development year 1 to 8 and mean by
    # calendar year, to two decimals.
    by_age = fit.residual_summary()
    spread = [s * math.sqrt((n - 1) / n) for s, n in zip(by_age["sd"][:8], by_age["n"][:8])]
    assert spread == pytest.approx([1.34, 1.59, 0.88, 0.62, 0.83, 0.77, 0.65, 0.40], abs=0.005)
    assert by_age["label"][0] == "12" and by_age["n"][-1] == 0
    by_calendar = fit.residual_summary("calendar")
    assert by_calendar["mean"] == pytest.approx(
        [-0.38, -0.13, -0.40, 0.56, 0.42, -0.43, 0.23, -0.27, -0.26, 0.32], abs=0.005
    )
    assert fit.residual_summary(by="origin")["label"][0] == "1988"
    with pytest.raises(ValueError, match="by must be"):
        fit.residual_summary("diagonal")


def test_custom_design_matches_chapter_7():
    # Model 7-10: the accident-year effect a quadratic in k, the development
    # effects a line in j with a kink at 7.5 and a term for year 2 (Table
    # 7-3, to the guide's four decimals).
    from prospicio.models import Design

    tri = triangle()
    cells = OdpGlm.cells(tri, "paid")
    assert len(cells["origin"]) == 100 and sum(cells["observed"]) == 55
    assert sum(cells["future"]) == 45 and cells["origin"][0] == "1988"
    k = [x + 1 for x in cells["origin_index"]]
    j = [x + 1 for x in cells["development_index"]]
    columns = [
        [1.0] * len(k), k, [x * x for x in k], [x - 1 for x in j],
        [max(0.0, x - 7.5) for x in j], [float(x == 2) for x in j],
    ]
    names = ["a0", "a1", "a2", "b1", "b2", "c"]
    fit = OdpGlm().fit(tri, "paid", design=Design(columns, names))
    assert fit.glm.names == names
    assert fit.glm.coefficients == pytest.approx(
        [10.4687, 0.2001, -0.0179, -0.3577, 0.2356, 0.1545], abs=5e-5
    )
    assert fit.dispersion == pytest.approx(107.3, abs=0.05)
    total = fit.total_prediction_error()
    assert total["reserve"] == pytest.approx(373006, abs=0.5)
    assert total["rmsep"] == pytest.approx(13215, abs=0.5)
    assert len(fit.residuals()["actual"]) == 55


def test_custom_design_from_terms_and_errors():
    from prospicio.models import Design, Terms

    tri = triangle()
    cells = OdpGlm.cells(tri, "paid")
    terms = (
        Terms().intercept().factor("origin", reference="1988")
        .factor("development", reference="12")
    )
    by_hand = OdpGlm().fit(tri, "paid", design=terms.fit(cells).design(cells))
    fit = OdpGlm().fit(tri, "paid")
    assert by_hand.reserves == pytest.approx(fit.reserves, rel=1e-12, abs=1e-9)
    with pytest.raises(ValueError, match="99 rows"):
        OdpGlm().fit(tri, "paid", design=Design([[1.0] * 99], ["a0"]))
    future = [float(f) for f in cells["future"]]
    with pytest.raises(ValueError, match="zero on every observed cell"):
        OdpGlm().fit(tri, "paid", design=Design([[1.0] * 100, future], ["a0", "next"]))
