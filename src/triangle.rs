//! Loss development triangles.

use std::fmt;

/// Why a triangle could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangleError {
    /// No origins were supplied.
    Empty,
    /// An origin has no observed values.
    EmptyOrigin { origin: usize },
    /// A value is NaN or infinite.
    NonFinite { origin: usize, age: usize },
}

impl fmt::Display for TriangleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "no origins supplied"),
            Self::EmptyOrigin { origin } => write!(f, "origin {origin} has no values"),
            Self::NonFinite { origin, age } => {
                write!(f, "value at origin {origin}, age {age} is NaN or infinite")
            }
        }
    }
}

impl std::error::Error for TriangleError {}

/// A cumulative loss development triangle.
///
/// Row `i` holds the cumulative losses of origin `i` at development ages
/// `0, 1, ..., len - 1`. Rows may have any non-zero length, so the usual
/// upper-left triangle is one case among others (e.g. a trapezoid or a
/// triangle with a missing latest diagonal).
#[derive(Debug, Clone, PartialEq)]
pub struct Triangle {
    rows: Vec<Vec<f64>>,
    n_ages: usize,
}

impl Triangle {
    /// Builds a triangle from cumulative values, one row per origin.
    ///
    /// # Example
    ///
    /// ```
    /// use risk_rs::triangle::Triangle;
    ///
    /// let t = Triangle::from_cumulative(vec![
    ///     vec![100.0, 150.0, 160.0],
    ///     vec![110.0, 170.0],
    ///     vec![120.0],
    /// ])
    /// .unwrap();
    /// assert_eq!(t.n_origins(), 3);
    /// assert_eq!(t.n_ages(), 3);
    /// assert_eq!(t.latest(), vec![160.0, 170.0, 120.0]);
    /// ```
    pub fn from_cumulative(rows: Vec<Vec<f64>>) -> Result<Self, TriangleError> {
        if rows.is_empty() {
            return Err(TriangleError::Empty);
        }
        for (origin, row) in rows.iter().enumerate() {
            if row.is_empty() {
                return Err(TriangleError::EmptyOrigin { origin });
            }
            if let Some(age) = row.iter().position(|x| !x.is_finite()) {
                return Err(TriangleError::NonFinite { origin, age });
            }
        }
        let n_ages = rows.iter().map(Vec::len).max().unwrap_or(0);
        Ok(Self { rows, n_ages })
    }

    /// Builds a triangle from incremental values by accumulating each row.
    pub fn from_incremental(rows: Vec<Vec<f64>>) -> Result<Self, TriangleError> {
        let cumulative = rows
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .scan(0.0, |total, x| {
                        *total += x;
                        Some(*total)
                    })
                    .collect()
            })
            .collect();
        Self::from_cumulative(cumulative)
    }

    /// Number of origin periods.
    pub fn n_origins(&self) -> usize {
        self.rows.len()
    }

    /// Number of development ages (the length of the longest row).
    pub fn n_ages(&self) -> usize {
        self.n_ages
    }

    /// Cumulative values of each origin.
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }

    /// Cumulative value of `origin` at `age`, if observed.
    pub fn get(&self, origin: usize, age: usize) -> Option<f64> {
        self.rows.get(origin)?.get(age).copied()
    }

    /// Index of the latest observed age of each origin.
    pub fn latest_ages(&self) -> Vec<usize> {
        self.rows.iter().map(|row| row.len() - 1).collect()
    }

    /// Latest observed cumulative value of each origin (the latest diagonal).
    pub fn latest(&self) -> Vec<f64> {
        self.rows.iter().map(|row| row[row.len() - 1]).collect()
    }

    /// Paired cumulative values at `age` and `age + 1` for every origin
    /// observed at both, as `(from, to)` columns.
    ///
    /// These are the inputs to [`crate::development::volume_weighted_factor`].
    pub fn link_columns(&self, age: usize) -> (Vec<f64>, Vec<f64>) {
        self.rows
            .iter()
            .filter(|row| row.len() > age + 1)
            .map(|row| (row[age], row[age + 1]))
            .unzip()
    }

    /// Incremental values of each origin.
    pub fn incremental(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .map(|row| {
                let mut previous = 0.0;
                row.iter()
                    .map(|&x| {
                        let step = x - previous;
                        previous = x;
                        step
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// RAA triangle (Mack 1993), cumulative, origins 1981-1990.
    pub(crate) fn raa() -> Triangle {
        Triangle::from_cumulative(vec![
            vec![
                5012.0, 8269.0, 10907.0, 11805.0, 13539.0, 16181.0, 18009.0, 18608.0, 18662.0,
                18834.0,
            ],
            vec![
                106.0, 4285.0, 5396.0, 10666.0, 13782.0, 15599.0, 15496.0, 16169.0, 16704.0,
            ],
            vec![
                3410.0, 8992.0, 13873.0, 16141.0, 18735.0, 22214.0, 22863.0, 23466.0,
            ],
            vec![5655.0, 11555.0, 15766.0, 21266.0, 23425.0, 26083.0, 27067.0],
            vec![1092.0, 9565.0, 15836.0, 22169.0, 25955.0, 26180.0],
            vec![1513.0, 6445.0, 11702.0, 12935.0, 15852.0],
            vec![557.0, 4020.0, 10946.0, 12314.0],
            vec![1351.0, 6947.0, 13112.0],
            vec![3133.0, 5395.0],
            vec![2063.0],
        ])
        .unwrap()
    }

    #[test]
    fn raa_shape_and_latest() {
        let t = raa();
        assert_eq!(t.n_origins(), 10);
        assert_eq!(t.n_ages(), 10);
        assert_eq!(t.latest().iter().sum::<f64>(), 160_987.0);
        assert_eq!(t.latest_ages(), vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn link_columns_pair_observed_origins() {
        let (from, to) = raa().link_columns(8);
        assert_eq!(from, vec![18662.0]);
        assert_eq!(to, vec![18834.0]);
        assert_eq!(raa().link_columns(0).0.len(), 9);
    }

    #[test]
    fn incremental_round_trips() {
        let t = raa();
        let back = Triangle::from_incremental(t.incremental()).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn get_outside_is_none() {
        let t = raa();
        assert_eq!(t.get(9, 0), Some(2063.0));
        assert_eq!(t.get(9, 1), None);
        assert_eq!(t.get(10, 0), None);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Triangle::from_cumulative(vec![]), Err(TriangleError::Empty));
        assert_eq!(
            Triangle::from_cumulative(vec![vec![1.0], vec![]]),
            Err(TriangleError::EmptyOrigin { origin: 1 })
        );
        assert_eq!(
            Triangle::from_cumulative(vec![vec![1.0, f64::INFINITY]]),
            Err(TriangleError::NonFinite { origin: 0, age: 1 })
        );
    }
}
