"""NumPy arrays in and out of EventSet (#978)."""

import gc

import numpy as np
import pytest

from prospicio.aggregate import EventSet


def test_to_numpy_views_the_losses_and_offsets():
    e = EventSet.from_years([[5.0, 2.0], [], [9.0]])
    losses, offsets = e.to_numpy()
    np.testing.assert_array_equal(losses, [5.0, 2.0, 9.0])
    np.testing.assert_array_equal(offsets, [0, 2, 2, 3])
    assert losses.dtype == np.float64 and offsets.dtype == np.uint64
    assert not losses.flags.writeable and not offsets.flags.writeable
    assert not losses.flags.owndata
    for i in range(e.n_sims):
        assert losses[offsets[i]:offsets[i + 1]].tolist() == e.events(i)
    with pytest.raises(ValueError):
        losses[0] = 0.0


def test_the_views_keep_the_event_set_alive():
    e = EventSet.from_years([np.array([1.0, 2.0]), np.array([3.0])])
    losses, offsets = e.to_numpy()
    del e
    gc.collect()
    np.testing.assert_array_equal(losses, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(offsets, [0, 2, 3])


def test_from_years_takes_arrays_per_year():
    years = [np.array([5.0, 2.0]), np.array([]), [9.0]]
    si = [np.array([10.0, 2.0]), np.array([]), np.array([50.0])]
    times = [np.array([0.1, 0.4]), [], np.array([0.9])]
    a = EventSet.from_years(years, si, times=times)
    b = EventSet.from_years([[5.0, 2.0], [], [9.0]], [[10.0, 2.0], [], [50.0]],
                            times=[[0.1, 0.4], [], [0.9]])
    assert a.counts() == b.counts()
    assert a.sums_insured(2) == b.sums_insured(2)
    assert a.times(0) == b.times(0)


def test_from_long_and_from_ylt_take_arrays():
    years = np.array([0, 2, 2], dtype=np.int64)
    losses = np.array([1.0, 3.0, 4.0])
    a = EventSet.from_long(years, losses, 3)
    b = EventSet.from_long([0, 2, 2], [1.0, 3.0, 4.0], 3)
    assert [a.events(i) for i in range(3)] == [b.events(i) for i in range(3)]

    ylt_years = np.array([2, 1, 2])
    slots = np.array([9, 1, 3])
    times = np.array([0.5, 0.2, 0.1])
    c = EventSet.from_ylt(ylt_years, np.array([5.0, 1.0, 7.0]), slots=slots, times=times,
                          n_sims=4)
    d = EventSet.from_ylt([2, 1, 2], [5.0, 1.0, 7.0], slots=[9, 1, 3], times=[0.5, 0.2, 0.1],
                          n_sims=4)
    assert [c.events(i) for i in range(4)] == [d.events(i) for i in range(4)]


def test_a_negative_year_in_an_array_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        EventSet.from_long(np.array([0, -1]), np.array([1.0, 2.0]), 2)
