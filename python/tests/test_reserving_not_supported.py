"""One exception for a setting a reserving method does not support (#580,
docs/design/reserving-v02.md decision 29): ``prospicio.NotSupportedError``,
a subclass of ``ValueError``, raised where the setting is chosen when it can
be, and a plain ``ValueError`` for a failure on the data."""

import csv
from pathlib import Path

import pytest

import prospicio
from prospicio.reserving import (
    BayesOdp,
    ChainLadder,
    Compartmental,
    CompartmentalPriors,
    Mack,
    MackBootstrap,
    NotSupportedError,
    OdpBootstrap,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


@pytest.fixture(scope="module")
def raa():
    with open(VALIDATION / "data" / "raa.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )


@pytest.fixture(scope="module")
def two_lines():
    with open(VALIDATION / "data" / "raa.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows] * 2,
        development=[int(r["development"]) for r in rows] * 2,
        values={"paid": [float(r["value"]) for r in rows] * 2},
        keys={"line": ["a"] * len(rows) + ["b"] * len(rows)},
    )


def test_the_exception_is_one_class_and_a_value_error():
    assert prospicio.NotSupportedError is NotSupportedError
    assert issubclass(NotSupportedError, ValueError)
    assert NotSupportedError.__name__ == "NotSupportedError"


def test_the_geometric_average_in_mack(raa):
    with pytest.raises(NotSupportedError, match="geometric average is not supported by Mack"):
        Mack(average="geometric").fit(raa, "values")


def test_merz_wuthrich_settings(raa):
    simple = Mack(average="simple").fit(raa, "values")
    with pytest.raises(NotSupportedError, match="needs volume-weighted factors"):
        simple.claims_development_result()
    tailed = Mack(tail=1.05).fit(raa, "values")
    with pytest.raises(NotSupportedError, match="a tail factor is not supported"):
        tailed.claims_development_result()
    trimmed = Mack(drop_high=1).fit(raa, "values")
    with pytest.raises(NotSupportedError, match=r"MackBootstrap::one_year"):
        trimmed.claims_development_result()


def test_a_bootstrap_tail_other_than_the_methods(raa):
    with pytest.raises(NotSupportedError, match="must be the re-reserving method's tail"):
        OdpBootstrap(n_sims=10, tail=1.05).one_year(raa, "values", ChainLadder())
    with pytest.raises(NotSupportedError, match="must be the re-reserving method's tail"):
        MackBootstrap(n_sims=10, tail=1.05).one_year(raa, "values", ChainLadder())


def test_synchronized_bayesian_segments_before_sampling(two_lines):
    with pytest.raises(NotSupportedError, match="Synchronized dependence is not supported"):
        BayesOdp(dependence="synchronized").fit_segments(two_lines, "paid")


def test_compartmental_paid_alone_is_refused_when_built():
    gm = CompartmentalPriors.gesmann_morris()
    for fitted in ("paid_only", "incurred_only"):
        with pytest.raises(NotSupportedError, match="needs rate_order Ordered"):
            Compartmental(gm, fitted=fitted)
        assert Compartmental(gm, fitted=fitted, rate_order="ordered").fitted == fitted


def test_a_failure_on_the_data_is_a_plain_value_error(raa):
    with pytest.raises(ValueError) as caught:
        OdpBootstrap(n_sims=0).fit(raa, "values")
    assert not isinstance(caught.value, NotSupportedError)
