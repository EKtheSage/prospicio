//! Triangle helpers: a triangle on the sheet turned incremental or
//! cumulative, its link ratios and its latest diagonal, over
//! `prospicio_reserving::Triangle` (`docs/design/xll.md`).

use prospicio_reserving::Triangle;

use crate::reserving::{COLUMN_POSITION, triangle_of};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// `tri`'s values as a sheet range, origins down and development across,
/// blank where unobserved.
fn range(tri: &Triangle) -> Grid {
    let [_, _, origins, ages] = tri.shape();
    Grid::from_rows(
        (0..origins)
            .map(|o| {
                (0..ages)
                    .map(|d| {
                        tri.get(0, COLUMN_POSITION, o, d)
                            .map_or(Value::Empty, Value::number)
                    })
                    .collect()
            })
            .collect(),
    )
}

/// A cumulative triangle's incremental values, laid out as it is:
/// `PRO.TRI.INCREMENTAL(triangle)`.
pub fn incremental(args: &[Grid]) -> Result<Grid> {
    Ok(range(&triangle_of(&args[0], None, true)?.to_incremental()))
}

/// An incremental triangle's cumulative values, laid out as it is:
/// `PRO.TRI.CUMULATIVE(triangle)`.
pub fn cumulative(args: &[Grid]) -> Result<Grid> {
    Ok(range(&triangle_of(&args[0], None, false)?.to_cumulative()))
}

/// A cumulative triangle's age-to-age link ratios, one column fewer than
/// its ages (column `k` is age `k` to `k + 1`), blank where either value is
/// unobserved or the earlier is zero: `PRO.TRI.LINKRATIOS(triangle)`.
pub fn link_ratios(args: &[Grid]) -> Result<Grid> {
    Ok(range(&triangle_of(&args[0], None, true)?.link_ratios()))
}

/// Each origin's latest observed value, a column beside the triangle:
/// `PRO.TRI.LATEST(triangle)`.
pub fn latest(args: &[Grid]) -> Result<Grid> {
    let tri = triangle_of(&args[0], None, true)?;
    let diagonal = tri.latest_diagonal();
    let origins = diagonal.shape()[2];
    Ok(Grid::from_rows(
        (0..origins)
            .map(|o| {
                vec![
                    diagonal
                        .get(0, COLUMN_POSITION, o)
                        .map_or(Value::Empty, |(_, x)| Value::number(x)),
                ]
            })
            .collect(),
    ))
}
