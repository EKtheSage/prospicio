"""JSON artifacts and pickling of reserving fits (#978)."""

import json
import math
import pickle

import pytest

from prospicio.reserving import ChainLadder, ChainLadderFit, Triangle


def two_segments():
    return Triangle.from_long(
        [2020, 2020, 2020, 2021, 2021, 2022] * 2,
        [12, 24, 36, 12, 24, 12] * 2,
        {"paid": [100.0, 150.0, 160.0, 110.0, 170.0, 120.0,
                  10.0, 20.0, 25.0, 12.0, 22.0, 14.0]},
        keys={"lob": ["Auto"] * 6 + ["Home"] * 6},
    )


def test_a_chain_ladder_fit_round_trips_through_json():
    fit = ChainLadder().fit(two_segments(), "paid")
    text = fit.to_json()
    back = ChainLadderFit.from_json(text)
    assert back.to_json() == text
    assert back.index == fit.index
    assert back.ultimate == fit.ultimate
    assert back.segment(lob="Home").ldf == fit.segment(lob="Home").ldf
    document = json.loads(text)
    assert document["format"] == "risk_rs.chain_ladder_fits"
    assert document["format_version"] == 1


def test_a_chain_ladder_fit_pickles():
    fit = ChainLadder().fit(two_segments(), "paid")
    back = pickle.loads(pickle.dumps(fit))
    assert isinstance(back, ChainLadderFit)
    assert back.to_json() == fit.to_json()
    assert all(math.isclose(a, b) for a, b in zip(back.reserve, fit.reserve))


def test_another_document_is_refused():
    with pytest.raises(ValueError):
        ChainLadderFit.from_json('{"format": "risk_rs.triangle", "format_version": 1}')
    with pytest.raises(ValueError):
        ChainLadderFit.from_json("not json")
