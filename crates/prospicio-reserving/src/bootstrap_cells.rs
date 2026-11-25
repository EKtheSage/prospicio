//! The future cells the lifetime bootstraps keep when asked
//! ([`OdpBootstrap::fit_with_cells`](crate::OdpBootstrap::fit_with_cells),
//! [`MackBootstrap::fit_with_cells`](crate::MackBootstrap::fit_with_cells);
//! `reserving-inflation.md`, decisions A to I).
//!
//! A simulation's row is its reserves, then its pattern shares, then each
//! segment's cells in turn, origin by origin and age by age. The cells'
//! columns carry placeholder keys in the row (one value per dimension of
//! the reserves); [`CellLayout::take`] copies them out before the patterns
//! are split off, and [`CellDraws::into_distribution`] gives them their
//! real dimensions: the segment keys, `origin` and `development`.

use prospicio_core::{Lag, Period};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::error::{Error, Result};
use crate::pattern_draws::pattern_ages;
use crate::triangle::Label;

/// The future cells of one segment, origin by origin, ages increasing:
/// each origin's ages of the grid after its latest one (`latest` indexes
/// `ages`), and with a tail one more, one development period past the
/// oldest age ([`pattern_ages`]), even when the tail adds nothing, so that
/// every simulation has the same cells (decision C).
pub(crate) fn segment_cells(
    origins: &[Period],
    latest: &[usize],
    ages: &[Lag],
    tail: bool,
) -> Vec<(Period, Lag)> {
    let tail_age = if tail {
        pattern_ages(ages, true).last().copied()
    } else {
        None
    };
    origins
        .iter()
        .zip(latest)
        .flat_map(|(&origin, &last)| {
            ages[(last + 1).min(ages.len())..]
                .iter()
                .copied()
                .chain(tail_age)
                .map(move |age| (origin, age))
        })
        .collect()
}

/// The cell columns of a lifetime simulation: from column `start` of each
/// row, each segment's cells in turn.
pub(crate) struct CellLayout {
    start: usize,
    segments: Vec<(Label, Vec<(Period, Lag)>)>,
}

impl CellLayout {
    /// The layout of `segments` (each a label and its cells from
    /// [`segment_cells`]) from column `start`. Fails with
    /// [`Error::Bootstrap`] when no segment has a future cell (decision G).
    pub(crate) fn new(start: usize, segments: Vec<(Label, Vec<(Period, Lag)>)>) -> Result<Self> {
        if segments.iter().all(|(_, cells)| cells.is_empty()) {
            return Err(Error::Bootstrap(
                "the triangle is fully developed and has no tail: there are no future cells to keep",
            ));
        }
        Ok(Self { start, segments })
    }

    /// The cells' placeholder keys in the simulation's row, for `n_dims`
    /// dimensions: the segment's label (padded to the key dimensions) and
    /// `"cell <origin> <age>"`, which no reserve's or pattern's key is.
    pub(crate) fn keys(&self, n_dims: usize) -> Vec<ComponentKey> {
        self.segments
            .iter()
            .flat_map(|(label, cells)| {
                cells.iter().map(move |(origin, age)| {
                    let mut key: ComponentKey =
                        label.parts().iter().map(|p| p.as_str().into()).collect();
                    key.resize(n_dims - 1, KeyValue::Text(String::new()));
                    key.push(KeyValue::Text(format!("cell {origin} {age}")));
                    key
                })
            })
            .collect()
    }

    /// Where segment `s`'s cells start, counted from the first cell
    /// column.
    pub(crate) fn offset(&self, s: usize) -> usize {
        self.segments[..s]
            .iter()
            .map(|(_, cells)| cells.len())
            .sum()
    }

    /// The number of cells segment `s` writes.
    pub(crate) fn width(&self, s: usize) -> usize {
        self.segments[s].1.len()
    }

    /// Each segment's cells in every simulation of `wide`, copied out.
    pub(crate) fn take(self, wide: &PredictiveDistribution) -> CellDraws {
        let n = wide.n_components();
        let rows = wide.draw_matrix();
        let mut start = self.start;
        let segments = self
            .segments
            .into_iter()
            .map(|(label, cells)| {
                let end = start + cells.len();
                let draws = rows
                    .chunks_exact(n)
                    .flat_map(|row| row[start..end].iter().copied())
                    .collect();
                start = end;
                (label, cells, draws)
            })
            .collect();
        CellDraws {
            n_sims: wide.n_sims(),
            segments,
        }
    }
}

/// A segment's label, its cells and their draws, row-major over
/// simulation by cell.
type SegmentDraws = (Label, Vec<(Period, Lag)>, Vec<f64>);

/// Each segment's simulated cells, row-major over simulation by cell.
pub(crate) struct CellDraws {
    n_sims: usize,
    segments: Vec<SegmentDraws>,
}

impl CellDraws {
    /// Segment `s`'s rows reordered: row `i` becomes the input's row
    /// `index[i]`, as a segment's patterns move with its reserves under a
    /// rank correlation (decision F).
    pub(crate) fn permute(&mut self, s: usize, index: &[usize]) {
        let (_, cells, draws) = &mut self.segments[s];
        let n = cells.len();
        *draws = index
            .iter()
            .flat_map(|&j| draws[j * n..(j + 1) * n].iter().copied())
            .collect();
    }

    /// The cells as one distribution: dimensions `key_names`, `origin` and
    /// `development` (the age in months), components running over each
    /// segment's cells in turn, and `provenance`.
    pub(crate) fn into_distribution(
        self,
        key_names: &[String],
        provenance: Provenance,
    ) -> Result<PredictiveDistribution> {
        let mut dims = key_names.to_vec();
        dims.push("origin".into());
        dims.push("development".into());
        let components: Vec<ComponentKey> = self
            .segments
            .iter()
            .flat_map(|(label, cells, _)| {
                cells.iter().map(move |&(origin, age)| {
                    let mut key: ComponentKey =
                        label.parts().iter().map(|p| p.as_str().into()).collect();
                    key.push(origin.into());
                    key.push(KeyValue::Int(age.into()));
                    key
                })
            })
            .collect();
        let mut draws = Vec::with_capacity(self.n_sims * components.len());
        for i in 0..self.n_sims {
            for (_, cells, values) in &self.segments {
                let n = cells.len();
                draws.extend_from_slice(&values[i * n..(i + 1) * n]);
            }
        }
        Ok(PredictiveDistribution::from_draws(
            dims, components, draws, provenance,
        )?)
    }
}

/// The provenance of a bootstrap's cells: its reserves', with
/// `cells = incremental` (decision H).
pub(crate) fn cells_provenance(reserves: &PredictiveDistribution) -> Provenance {
    reserves.provenance().clone().param("cells", "incremental")
}

/// The part of a bootstrap's `cells` in the segment `label` (the components
/// whose keys start with its parts), with the same dimensions; `None` when
/// the segment has no future cell.
pub(crate) fn segment_part(
    cells: &PredictiveDistribution,
    label: &Label,
) -> Result<Option<PredictiveDistribution>> {
    let prefix: Vec<KeyValue> = label.parts().iter().map(|p| p.as_str().into()).collect();
    let picked: Vec<usize> = cells
        .components()
        .iter()
        .enumerate()
        .filter(|(_, key)| key.starts_with(&prefix))
        .map(|(j, _)| j)
        .collect();
    if picked.is_empty() {
        return Ok(None);
    }
    let n = cells.n_components();
    let draws = cells
        .draw_matrix()
        .chunks_exact(n)
        .flat_map(|row| picked.iter().map(move |&j| row[j]))
        .collect();
    Ok(Some(PredictiveDistribution::from_draws(
        cells.dims().to_vec(),
        picked
            .iter()
            .map(|&j| cells.components()[j].clone())
            .collect(),
        draws,
        cells.provenance().clone(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_cells_follow_the_latest_diagonal_and_the_tail() {
        let origins: Vec<Period> = (2020..2023).map(Period::year).collect();
        let ages = [12, 24, 36];
        let latest = [2, 1, 0];
        let cells = segment_cells(&origins, &latest, &ages, false);
        assert_eq!(
            cells,
            [
                (Period::year(2021), 36),
                (Period::year(2022), 24),
                (Period::year(2022), 36),
            ]
        );
        let with_tail = segment_cells(&origins, &latest, &ages, true);
        assert_eq!(with_tail.len(), 6);
        assert_eq!(with_tail[0], (Period::year(2020), 48));
        assert_eq!(with_tail[2], (Period::year(2021), 48));
        assert_eq!(with_tail[5], (Period::year(2022), 48));
        // Fully developed and no tail: no cell, and the layout refuses.
        let none = segment_cells(&origins, &[2, 2, 2], &ages, false);
        assert!(none.is_empty());
        assert!(matches!(
            CellLayout::new(0, vec![(Label::default(), none)]),
            Err(Error::Bootstrap(_))
        ));
    }

    #[test]
    fn bootstrap_cells_are_taken_permuted_and_keyed() {
        let label = Label::new(["motor"]);
        let cells = vec![(Period::year(2021), 36), (Period::year(2022), 24)];
        let layout = CellLayout::new(1, vec![(label.clone(), cells)]).unwrap();
        let keys = layout.keys(2);
        assert_eq!(keys[0][0], KeyValue::from("motor"));
        assert_eq!(keys[1][1], KeyValue::Text("cell 2022 24".into()));
        let mut components = vec![vec![KeyValue::from("motor"), Period::year(2021).into()]];
        components.extend(keys);
        let wide = PredictiveDistribution::from_draws(
            vec!["lob".into(), "origin".into()],
            components,
            vec![9.0, 1.0, 2.0, 9.0, 3.0, 4.0],
            Provenance::new("test"),
        )
        .unwrap();
        let mut draws = layout.take(&wide);
        draws.permute(0, &[1, 0]);
        let out = draws
            .into_distribution(&["lob".into()], Provenance::new("test"))
            .unwrap();
        assert_eq!(out.dims(), ["lob", "origin", "development"]);
        assert_eq!(
            out.components()[1],
            vec![
                KeyValue::from("motor"),
                Period::year(2022).into(),
                KeyValue::Int(24)
            ]
        );
        assert_eq!(out.draw_matrix(), [3.0, 4.0, 1.0, 2.0]);
        let part = segment_part(&out, &label).unwrap().unwrap();
        assert_eq!(part.draw_matrix(), out.draw_matrix());
        assert!(segment_part(&out, &Label::new(["home"])).unwrap().is_none());
    }
}
