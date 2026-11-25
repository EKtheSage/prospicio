"""Terms and codings round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import Coding, Terms


def _data():
    x = [1.0 + 0.7 * i for i in range(12)]
    region = ["N", "S", "W"] * 4
    band = ["lo", "mid", "hi"] * 4
    return {"x": x, "region": region, "band": band}


def _terms():
    return (
        Terms()
        .intercept()
        .numeric("x")
        .factor("region")
        .factor("band", reference="mid")
        .interaction("x", "region")
        .poly("x", 3)
        .hinges("x", [3.0, 5.0])
        .steps("x", [4.0])
        .bins("x", [2.0, 6.0])
        .log("x", 2.0)
        .ordinal("band", ["lo", "mid", "hi"])
        .orthogonal_poly("x", 2)
        .natural_spline("x", [3.0, 6.0], boundary=(0.5, 10.0))
    )


def test_terms_round_trip():
    terms = _terms()
    for back in (Terms.from_json(terms.to_json()), pickle.loads(pickle.dumps(terms))):
        assert isinstance(back, Terms)
        assert back.to_json() == terms.to_json()
        assert back.fit(_data()).names == terms.fit(_data()).names


def test_coding_round_trip_builds_the_same_design():
    coding = _terms().fit(_data())
    new = {"x": [2.5, 7.0, 9.0], "region": ["S", "N", "W"], "band": ["hi", "lo", "mid"]}
    for back in (Coding.from_json(coding.to_json()), pickle.loads(pickle.dumps(coding))):
        assert isinstance(back, Coding)
        assert back.names == coding.names
        a, b = back.design(new), coding.design(new)
        assert [a.column(j) for j in range(len(a.names))] == [b.column(j) for j in range(len(b.names))]


def test_bad_artifacts_are_refused():
    coding = _terms().fit(_data())
    with pytest.raises(ValueError):
        Terms.from_json(coding.to_json())
    with pytest.raises(ValueError):
        Coding.from_json(coding.to_json().replace('"format_version":1', '"format_version":2'))
