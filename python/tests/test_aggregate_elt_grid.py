"""An event loss table as a compound Poisson severity on a grid (#781),
against aggregate 2.1.0 in validation/reference/elt_grid_aggregate.csv, the
rows the Rust parity test reads."""

import pytest

from helpers import VALIDATION, read_csv
from prospicio.aggregate import EventLossTable, fft
from prospicio.reinsurance import Layer, Tower

STEP, POINTS = 10.0, 1024
MEANS = [[0.5, 130.0], [0.25, 270.5], [0.05, 1234.5], [0.01, 2999.9], [0.002, 7777.7]]
# rate, mean, max, sdi, sdc
SECONDARY = [
    [0.01, 800.0, 2000.0, 200.0, 100.0],
    [0.05, 250.0, 500.0, 0.0, 0.0],
    [0.002, 1500.0, 5000.0, 300.0, 300.0],
    [0.2, 40.0, 300.0, 30.0, 10.0],
    [0.0005, 6000.0, 20000.0, 3000.0, 1000.0],
]


def _table(rows):
    col = lambda j: [r[j] for r in rows]  # noqa: E731
    return EventLossTable(col(0), col(1), col(3), col(4), col(2))


def _grids(elt, rule):
    sev, _ = elt.severity_grid(STEP, POINTS, rule)
    annual, report, _ = elt.aggregate_grid(STEP, POINTS, rule)
    p = list(annual.probs)
    p[-1] -= report.tail_mass  # aggregate leaves the mass above the grid out
    return list(sev.probs), p


def test_matches_aggregate():
    means = _table([[r, m, 2 * m, 0.0, 0.0] for r, m in MEANS])
    secondary = _table(SECONDARY)
    m_sev, m_agg = _grids(means, "means_linear")
    s_sev, s_agg = _grids(secondary, "nearest")
    sev, _ = means.severity_grid(STEP, POINTS, "means_linear")
    tower = Tower([Layer("1000 xs 500", 1000.0, 500.0)])
    ceded = list(tower.on_grid(means.count(), sev, POINTS).ceded[0].probs)
    grids = {
        ("table=means", "sev"): m_sev,
        ("table=means", "agg"): m_agg,
        ("table=means_ceded", "agg"): ceded,
        ("table=secondary", "sev"): s_sev,
        ("table=secondary", "agg"): s_agg,
        ("table=secondary_exact", "sev"): s_sev,
    }
    failures = []
    for r in read_csv(VALIDATION / "reference" / "elt_grid_aggregate.csv"):
        got = grids[(r["params"], r["quantity"])][int(r["arg"])]
        want, tol = float(r["expected"]), float(r["abs_tol"])
        if abs(got - want) > tol:
            failures.append((r["params"], r["quantity"], r["arg"], got, want))
    assert not failures, failures[:10]


def test_the_count_and_the_reports():
    elt = _table(SECONDARY)
    assert elt.count().mean() == pytest.approx(elt.rate, abs=1e-15)
    _, narrow = elt.severity_grid(STEP, 64)
    assert narrow["bucketing"] == "nearest"
    assert narrow["rate_above"] > 0
    assert narrow["grid_mean"] < narrow["severity_mean"]
    annual, _ = fft(elt.count(), elt.severity_grid(STEP, POINTS)[0], POINTS)
    # The 20000-exposure row reaches past the grid, so the mean falls a little.
    assert annual.mean() == pytest.approx(elt.expected_annual_loss, rel=1e-2)
    assert annual.mean() < elt.expected_annual_loss


def test_a_bad_rule_is_refused():
    with pytest.raises(ValueError, match="means_linear"):
        _table(SECONDARY).severity_grid(STEP, 64, "linear")
