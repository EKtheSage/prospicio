//! NumPy arrays across the boundary without boxing each number (#978).
//!
//! An argument that takes a matrix accepts a 2-D NumPy array, read in one
//! copy of its buffer, or a list of rows as before. A container that holds
//! its numbers in one immutable row-major buffer hands NumPy a read-only
//! view of that buffer, which keeps the container alive and copies nothing.

use numpy::ndarray::ArrayView2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A matrix argument: a 2-D float64 NumPy array or a sequence of rows.
#[derive(FromPyObject)]
pub(crate) enum Matrix<'py> {
    Array(PyReadonlyArray2<'py, f64>),
    Rows(Vec<Vec<f64>>),
}

impl Matrix<'_> {
    /// The values row by row, checked to have `width` columns, one per
    /// `unit` ("component") as the error puts it.
    pub(crate) fn row_major(self, width: usize, unit: &str) -> PyResult<Vec<f64>> {
        match self {
            Matrix::Array(a) => {
                let view = a.as_array();
                let (rows, cols) = view.dim();
                if cols != width && rows > 0 {
                    return Err(PyValueError::new_err(format!(
                        "the array has {cols} columns, expected one per {unit} ({width})"
                    )));
                }
                // One pass over a C-contiguous buffer is a memcpy; any other
                // layout is read in row order.
                Ok(match view.as_slice() {
                    Some(s) => s.to_vec(),
                    None => view.iter().copied().collect(),
                })
            }
            Matrix::Rows(rows) => {
                if let Some(bad) = rows.iter().position(|row| row.len() != width) {
                    return Err(PyValueError::new_err(format!(
                        "row {bad} has {} values, expected one per {unit} ({width})",
                        rows[bad].len()
                    )));
                }
                Ok(rows.into_iter().flatten().collect())
            }
        }
    }
}

/// A read-only `rows` by `cols` NumPy view of `data`, which `owner` holds.
///
/// # Safety
///
/// `data` must live in `owner`, a frozen object that never changes or
/// moves the buffer, so the view stays valid for as long as it keeps
/// `owner` alive.
pub(crate) unsafe fn view<'py>(
    owner: Bound<'py, PyAny>,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = ArrayView2::from_shape((rows, cols), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // SAFETY: the caller's contract; NumPy holds a reference to `owner`.
    let array = unsafe { PyArray2::borrow_from_array(&shape, owner) };
    array.getattr("flags")?.setattr("writeable", false)?;
    Ok(array)
}
