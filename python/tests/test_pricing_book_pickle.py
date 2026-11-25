"""Deals and books round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.pricing import DealBook, DealRun


def _run():
    return (
        DealRun("cat", [100.0, 100.1], [40.0, 0.1 + 0.2], deductions={"brokerage": [2.5, 2.5], "fee": ([1.0, 1.25], True)}, inception=6)
        .with_patterns([1.0, 1.0], [0.5, 1.0], [0.1, 0.4])
        .with_flows([[-10.0, 3.5], [-12.0, 1.0 / 3.0]], [10.0, 12.0], [5.0, 5.0])
    )


def test_run_round_trip():
    run = _run()
    for back in (DealRun.from_json(run.to_json()), pickle.loads(pickle.dumps(run))):
        assert isinstance(back, DealRun)
        assert back.to_json() == run.to_json()
        assert (back.name, back.premium, back.loss, back.inception) == (run.name, run.premium, run.loss, run.inception)


def test_book_round_trip_keeps_its_tables():
    book = DealBook([_run(), DealRun("b", [1.0, 2.0], [0.0, 3.0])])
    back = pickle.loads(pickle.dumps(book))
    assert back.to_json() == book.to_json()
    assert back.summary() == book.summary()
    assert DealBook.from_json(book.to_json()).to_json() == book.to_json()


def test_bad_documents_are_refused():
    with pytest.raises(ValueError):
        DealBook.from_json(_run().to_json())
    with pytest.raises(ValueError):
        DealRun.from_json(_run().to_json().replace('"format_version":1', '"format_version":2'))
