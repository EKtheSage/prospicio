"""PredictiveDistribution to and from JSON, and through pickle (#978)."""

import pickle

import pytest

from prospicio.distributions import PredictiveDistribution


def sample():
    return PredictiveDistribution(
        ["lob", "year"],
        [("motor", 2020), ("home", 2021)],
        [[0.1 + 0.2, -0.0], [5e-324, 1.7976931348623157e308], [3.25, 4.0]],
    )


def test_json_round_trips_bit_for_bit():
    pd = sample()
    back = PredictiveDistribution.from_json(pd.to_json())
    assert back.dims == pd.dims
    assert back.components() == pd.components()
    assert back.draw_matrix() == pd.draw_matrix()
    assert back.to_json() == pd.to_json()


def test_pickle_round_trips():
    pd = sample()
    back = pickle.loads(pickle.dumps(pd))
    assert isinstance(back, PredictiveDistribution)
    assert back.to_json() == pd.to_json()


def test_from_json_refuses_other_documents():
    with pytest.raises(ValueError):
        PredictiveDistribution.from_json("{}")
    with pytest.raises(ValueError):
        PredictiveDistribution.from_json("not json")
