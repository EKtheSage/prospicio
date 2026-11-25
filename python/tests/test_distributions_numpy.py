"""NumPy arrays in and out of PredictiveDistribution (#978)."""

import gc
import pickle

import numpy as np
import pytest

from prospicio.distributions import PredictiveDistribution

COMPONENTS = [("A",), ("B",), ("C",)]


def draws(n=5):
    return np.arange(3 * n, dtype=float).reshape(n, 3) / 7.0


def test_an_array_builds_the_same_distribution_as_its_rows():
    a = draws()
    from_array = PredictiveDistribution(["line"], COMPONENTS, a)
    from_rows = PredictiveDistribution(["line"], COMPONENTS, a.tolist())
    assert from_array.draw_matrix() == from_rows.draw_matrix()
    assert from_array.components() == from_rows.components()


def test_a_transposed_or_strided_array_is_read_by_rows():
    a = draws().T.copy().T  # Fortran order
    assert not a.flags.c_contiguous
    pd = PredictiveDistribution(["line"], COMPONENTS, a)
    np.testing.assert_array_equal(pd.to_numpy(), a)
    b = np.repeat(draws(), 2, axis=0)[::2]
    pd = PredictiveDistribution(["line"], COMPONENTS, b)
    np.testing.assert_array_equal(pd.to_numpy(), b)


def test_to_numpy_is_a_read_only_view_that_keeps_the_distribution_alive():
    a = draws()
    pd = PredictiveDistribution(["line"], COMPONENTS, a)
    view = pd.to_numpy()
    assert view.shape == (5, 3)
    assert view.dtype == np.float64
    assert not view.flags.writeable
    assert not view.flags.owndata
    assert np.shares_memory(view, pd.to_numpy())
    with pytest.raises(ValueError):
        view[0, 0] = 1.0
    del pd
    gc.collect()
    np.testing.assert_array_equal(view, a)


def test_the_view_matches_draw_matrix_and_survives_a_pickle():
    pd = PredictiveDistribution(["line"], COMPONENTS, draws())
    np.testing.assert_array_equal(pd.to_numpy(), np.array(pd.draw_matrix()))
    back = pickle.loads(pickle.dumps(pd))
    np.testing.assert_array_equal(back.to_numpy(), pd.to_numpy())


def test_a_copy_is_writeable():
    pd = PredictiveDistribution(["line"], COMPONENTS, draws())
    c = np.array(pd.to_numpy())
    c[0, 0] = -1.0
    assert pd.to_numpy()[0, 0] == 0.0


def test_the_array_needs_one_column_per_component():
    with pytest.raises(ValueError, match="one per component"):
        PredictiveDistribution(["line"], COMPONENTS, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="one per component"):
        PredictiveDistribution(["line"], COMPONENTS, [[1.0, 2.0, 3.0], [1.0]])


def test_an_integer_array_is_read_row_by_row_as_before():
    a = np.arange(12, dtype=np.int64).reshape(4, 3)
    pd = PredictiveDistribution(["line"], COMPONENTS, a)
    np.testing.assert_array_equal(pd.to_numpy(), a.astype(float))
