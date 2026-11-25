"""Triangle, EventSet and Design pickle and round-trip through JSON (#978)."""

import pickle

import pytest

from prospicio.aggregate import EventSet
from prospicio.models import Design
from prospicio.reserving import Triangle


def triangle():
    return Triangle.from_long(
        origin=[2020, 2020, 2021, 2020],
        development=[12, 24, 12, 12],
        values={"paid": [100.0, 150.0, 110.0, 50.0], "incurred": [120.0, 1e-310, 0.1, 60.0]},
        keys={"lob": ["Auto", "Auto", "Auto", "Home"]},
    )


@pytest.mark.parametrize("make", [lambda t: t, lambda t: t.to_incremental(), lambda t: t.link_ratios()])
def test_triangle_round_trips(make):
    tri = make(triangle())
    assert Triangle.from_json(tri.to_json()) == tri
    back = pickle.loads(pickle.dumps(tri))
    assert back == tri
    assert back.index == tri.index and back.valuation == tri.valuation


def test_triangle_refuses_other_documents():
    with pytest.raises(ValueError):
        Triangle.from_json("{}")


def test_event_set_round_trips_with_labels_and_times():
    e = (EventSet.from_years([[5.0, 2.0], [], [9.0]], seed=3)
         .with_labels({"peril": ["wind", "flood", "wind"]})
         .with_uniform_times())
    assert EventSet.from_json(e.to_json()) == e
    back = pickle.loads(pickle.dumps(e))
    assert back == e
    assert back.seed == 3 and back.label_combinations == e.label_combinations
    assert [back.times(i) for i in range(3)] == [e.times(i) for i in range(3)]


def test_event_set_refuses_other_documents():
    with pytest.raises(ValueError):
        EventSet.from_json("[]")


def test_design_pickles():
    d = Design([[1.0, 1.0, 1.0], [0.5, -0.0, 2.0]], ["(Intercept)", "x"], offset=[0.0, 0.1, 0.2],
               weights=[1.0, 2.0, 3.0])
    back = pickle.loads(pickle.dumps(d))
    assert back.names == d.names
    assert back.offset == d.offset and back.weights == d.weights
    assert [back.column(j) for j in range(2)] == [d.column(j) for j in range(2)]
