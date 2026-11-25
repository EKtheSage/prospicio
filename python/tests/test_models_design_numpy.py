"""NumPy arrays in and out of Design (#978)."""

import numpy as np
import pytest

from prospicio.models import Design

NAMES = ["(Intercept)", "x", "z"]


def columns(n=6):
    x = np.linspace(0.0, 1.0, n)
    return np.vstack([np.ones(n), x, x**2])


def test_an_array_of_columns_builds_the_same_design_as_lists():
    c = columns()
    a = Design(c, NAMES)
    b = Design(c.tolist(), NAMES)
    assert a.n_rows == b.n_rows == 6
    for j in range(3):
        assert a.column(j) == b.column(j)


def test_a_list_of_arrays_and_strided_arrays_are_read():
    c = columns()
    a = Design([c[0], c[1][::-1], c[2]], NAMES)
    assert a.column(1) == c[1][::-1].tolist()
    f = np.asfortranarray(c)
    np.testing.assert_array_equal(Design(f, NAMES).to_numpy(), c.T)


def test_to_numpy_is_rows_by_names_in_fortran_order():
    c = columns()
    m = Design(c, NAMES).to_numpy()
    assert m.shape == (6, 3)
    assert m.dtype == np.float64
    assert m.flags.f_contiguous
    np.testing.assert_array_equal(m, c.T)
    m[0, 0] = 9.0  # a copy, safe to write


def test_offset_and_weights_take_arrays():
    c = columns()
    w = np.arange(1.0, 7.0)
    o = np.log(w)
    d = Design(c, NAMES, offset=o, weights=w)
    assert d.weights == w.tolist()
    assert d.offset == o.tolist()


def test_an_empty_design_gives_an_empty_array():
    d = Design([[1.0, 2.0]], ["x"]).select([])
    assert d.to_numpy().shape == (0, 1)


def test_a_ragged_list_is_still_refused():
    with pytest.raises(ValueError):
        Design([[1.0, 2.0], [1.0]], ["a", "b"])
