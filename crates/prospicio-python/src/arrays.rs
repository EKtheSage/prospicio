//! NumPy arrays across the boundary without boxing each number (#978).
//!
//! An argument that takes a matrix accepts a 2-D NumPy array, read in one
//! copy of its buffer, or a list of rows as before. A container that holds
//! its numbers in one immutable row-major buffer hands NumPy a read-only
//! view of that buffer, which keeps the container alive and copies nothing.

use numpy::ndarray::ArrayView2;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
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

/// A vector argument: a 1-D float64 NumPy array, read in one copy, or a
/// sequence of numbers.
#[derive(FromPyObject)]
pub(crate) enum Vector<'py> {
    Array(PyReadonlyArray1<'py, f64>),
    List(Vec<f64>),
}

impl Vector<'_> {
    /// The values, in order.
    pub(crate) fn into_vec(self) -> Vec<f64> {
        match self {
            Vector::Array(a) => match a.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => a.as_array().iter().copied().collect(),
            },
            Vector::List(v) => v,
        }
    }
}

/// A list of columns: a 2-D float64 NumPy array whose rows are the columns,
/// or a sequence of vectors, each a 1-D array or a list.
#[derive(FromPyObject)]
pub(crate) enum Columns<'py> {
    Array(PyReadonlyArray2<'py, f64>),
    List(Vec<Vector<'py>>),
}

impl Columns<'_> {
    /// The columns, in order.
    pub(crate) fn into_vecs(self) -> Vec<Vec<f64>> {
        match self {
            Columns::Array(a) => a
                .as_array()
                .rows()
                .into_iter()
                .map(|r| match r.as_slice() {
                    Some(s) => s.to_vec(),
                    None => r.iter().copied().collect(),
                })
                .collect(),
            Columns::List(v) => v.into_iter().map(Vector::into_vec).collect(),
        }
    }
}

/// A new `rows` by `columns.len()` NumPy array (Fortran order, so each
/// column is one contiguous copy) of `columns`, which share one length.
pub(crate) fn from_columns<'py>(
    py: Python<'py>,
    rows: usize,
    columns: &[&[f64]],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = PyArray2::<f64>::zeros(py, [rows, columns.len()], true);
    {
        let mut rw = array.readwrite();
        let out = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        for (j, c) in columns.iter().enumerate() {
            out[j * rows..(j + 1) * rows].copy_from_slice(c);
        }
    }
    Ok(array)
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
