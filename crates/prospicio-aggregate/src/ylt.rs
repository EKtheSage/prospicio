//! Year loss tables: a vendor's simulated years, one row per non-zero event
//! loss, checked and expanded to a run's number of simulations.
//!
//! The table comes in as columns, so the crate reads no files; the Python
//! and R packages read CSV and Parquet into them. Rows are ordered by year,
//! then by event slot, the file's order breaking ties, and each row is one
//! loss of the [`EventSet`]: rows that share a slot (one event hitting
//! several classes) stay separate losses, so an occurrence layer sees each
//! of them, not their sum.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_prob::provenance::InputHasher;

use crate::EventSet;

/// How a table's years become a run's simulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expansion {
    /// Simulation `i` takes year `i mod n_years`, so the number of
    /// simulations must be a whole multiple of the table's years.
    Tile,
    /// Simulation `i` takes a year drawn uniformly, with replacement, from
    /// `StreamRng::new(seed, i)`.
    Resample {
        /// The seed of the draws.
        seed: u64,
    },
}

/// A year loss table: each row's year, from 1, its loss and, optionally,
/// its event slot and labels.
///
/// ```
/// use prospicio_aggregate::ylt::{Expansion, YearLossTable};
///
/// let peril = vec!["wind".to_string(), "quake".into(), "wind".into()];
/// let ylt = YearLossTable::new(&[2, 1, 2], &[5.0, 1.0, 7.0])
///     .unwrap()
///     .with_slots(vec![9, 1, 3])
///     .unwrap()
///     .with_labels(vec![("peril".into(), peril)])
///     .unwrap();
/// ylt.check_unique(&["peril"]).unwrap();
/// ylt.check_values("peril", &["wind", "quake"]).unwrap();
/// let events = ylt.events(Some(4), Expansion::Tile).unwrap();
/// // Year 2's rows in slot order: slot 3 (7) before slot 9 (5).
/// assert_eq!(events.events(1), &[7.0, 5.0]);
/// assert_eq!(events.events(3), &[7.0, 5.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct YearLossTable {
    /// Each row's year, from 0.
    years: Vec<usize>,
    losses: Vec<f64>,
    slots: Option<Vec<u64>>,
    labels: Vec<(String, Vec<String>)>,
    n_years: usize,
    source_hash: Option<String>,
}

impl YearLossTable {
    /// The rows' years, from 1, and losses, which must be finite and
    /// non-negative. The table's years run to the largest year given; a
    /// table whose last years have no loss says how many it has with
    /// [`with_n_years`](Self::with_n_years).
    pub fn new(years: &[usize], losses: &[f64]) -> Result<Self> {
        if years.len() != losses.len() {
            return Err(Error::Data(format!(
                "{} years for {} losses",
                years.len(),
                losses.len()
            )));
        }
        if years.is_empty() {
            return Err(Error::Data(
                "a year loss table needs at least one row".into(),
            ));
        }
        if let Some(k) = years.iter().position(|&y| y == 0) {
            return Err(Error::Data(format!("row {}: years run from 1", k + 1)));
        }
        if let Some(k) = losses.iter().position(|x| !(x.is_finite() && *x >= 0.0)) {
            return Err(Error::Data(format!(
                "row {}: the loss {} is not finite and non-negative",
                k + 1,
                losses[k]
            )));
        }
        let n_years = *years.iter().max().expect("years is not empty");
        Ok(Self {
            years: years.iter().map(|y| y - 1).collect(),
            losses: losses.to_vec(),
            slots: None,
            labels: Vec::new(),
            n_years,
            source_hash: None,
        })
    }

    /// Each row's event slot, which orders the rows within a year.
    pub fn with_slots(mut self, slots: Vec<u64>) -> Result<Self> {
        if slots.len() != self.losses.len() {
            return Err(Error::Data(format!(
                "{} slots for {} rows",
                slots.len(),
                self.losses.len()
            )));
        }
        self.slots = Some(slots);
        Ok(self)
    }

    /// Each row's labels, a peril or a class say: one `(dimension, values)`
    /// pair per dimension, a value per row, as [`EventSet::with_labels`]
    /// takes them.
    pub fn with_labels(mut self, labels: Vec<(String, Vec<String>)>) -> Result<Self> {
        for (dim, values) in &labels {
            if values.len() != self.losses.len() {
                return Err(Error::Data(format!(
                    "{} values of label {dim:?} for {} rows",
                    values.len(),
                    self.losses.len()
                )));
            }
        }
        self.labels = labels;
        Ok(self)
    }

    /// The table's number of years, at least its largest year.
    pub fn with_n_years(mut self, n_years: usize) -> Result<Self> {
        let largest = self.years.iter().max().map_or(0, |y| y + 1);
        if n_years < largest {
            return Err(Error::Data(format!(
                "n_years = {n_years} is below the largest year, {largest}"
            )));
        }
        self.n_years = n_years;
        Ok(self)
    }

    /// A hash of the table's source, recorded as the events' and results'
    /// input hash: [`hash_bytes`] of the file, say.
    pub fn with_source_hash(mut self, hash: impl Into<String>) -> Self {
        self.source_hash = Some(hash.into());
        self
    }

    /// The table's number of years.
    pub fn n_years(&self) -> usize {
        self.n_years
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.losses.len()
    }

    /// Whether the table has no rows, which [`new`](Self::new) rules out.
    pub fn is_empty(&self) -> bool {
        self.losses.is_empty()
    }

    /// The hash of the table's source, if given.
    pub fn source_hash(&self) -> Option<&str> {
        self.source_hash.as_deref()
    }

    /// Checks that no two rows share a year, a slot and their values in the
    /// label dimensions `dims`: with `["peril"]`, one event slot of a year
    /// may hit several classes but not appear twice for one peril.
    pub fn check_unique(&self, dims: &[&str]) -> Result<()> {
        let Some(slots) = &self.slots else {
            return Err(Error::Data(
                "rows are unique by year and slot: give the slots".into(),
            ));
        };
        let columns = dims
            .iter()
            .map(|d| self.label(d))
            .collect::<Result<Vec<_>>>()?;
        let mut keys: Vec<(usize, u64, Vec<&str>, usize)> = (0..self.len())
            .map(|k| {
                let values = columns.iter().map(|c| c[k].as_str()).collect();
                (self.years[k], slots[k], values, k)
            })
            .collect();
        keys.sort();
        for pair in keys.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if (a.0, a.1, &a.2) == (b.0, b.1, &b.2) {
                let by = if dims.is_empty() {
                    String::new()
                } else {
                    format!(", {} {:?}", dims.join(", "), a.2)
                };
                return Err(Error::Data(format!(
                    "rows {} and {} repeat year {}, slot {}{by}",
                    a.3 + 1,
                    b.3 + 1,
                    a.0 + 1,
                    a.1
                )));
            }
        }
        Ok(())
    }

    /// Checks that every row's value in label dimension `dim` is one of
    /// `expected`: the perils a run models, say.
    pub fn check_values(&self, dim: &str, expected: &[&str]) -> Result<()> {
        let column = self.label(dim)?;
        if let Some(k) = column.iter().position(|v| !expected.contains(&v.as_str())) {
            return Err(Error::Data(format!(
                "row {}: {dim} {:?} is not one of {expected:?}",
                k + 1,
                column[k]
            )));
        }
        Ok(())
    }

    fn label(&self, dim: &str) -> Result<&[String]> {
        self.labels
            .iter()
            .find(|(d, _)| d == dim)
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| Error::Data(format!("no label dimension {dim:?}")))
    }

    /// The year each of `n_sims` simulations takes (from 0), `n_sims`
    /// defaulting to the table's years taken once each.
    pub fn years_of_sims(&self, n_sims: Option<usize>, expansion: Expansion) -> Result<Vec<usize>> {
        let n_sims = n_sims.unwrap_or(self.n_years);
        if n_sims == 0 {
            return Err(Error::InvalidParameter {
                name: "n_sims",
                value: 0.0,
                reason: "must be positive",
            });
        }
        Ok(match expansion {
            Expansion::Tile => {
                if n_sims % self.n_years != 0 {
                    return Err(Error::Data(format!(
                        "tiling {} years into {n_sims} simulations: the simulations must be \
                         a whole multiple of the years, or resample them",
                        self.n_years
                    )));
                }
                (0..n_sims).map(|i| i % self.n_years).collect()
            }
            Expansion::Resample { seed } => (0..n_sims)
                .map(|i| {
                    let u = StreamRng::new(seed, i as u64).next_open01();
                    ((u * self.n_years as f64) as usize).min(self.n_years - 1)
                })
                .collect(),
        })
    }

    /// The table's years as `n_sims` simulated years of events, by
    /// `expansion`, each row a loss with its labels, in order of slot within
    /// its year (the table's order without slots). The events' seed is the
    /// resampling seed (0 when tiled), their samplers unrecorded and their
    /// input hash the table's source hash.
    pub fn events(&self, n_sims: Option<usize>, expansion: Expansion) -> Result<EventSet> {
        let sims = self.years_of_sims(n_sims, expansion)?;
        let mut order: Vec<usize> = (0..self.len()).collect();
        match &self.slots {
            Some(slots) => order.sort_by_key(|&k| (self.years[k], slots[k])),
            None => order.sort_by_key(|&k| self.years[k]),
        }
        // Each year's rows, in order.
        let mut rows = vec![Vec::new(); self.n_years];
        for k in order {
            rows[self.years[k]].push(k);
        }
        let years = sims
            .iter()
            .map(|&y| rows[y].iter().map(|&k| self.losses[k]).collect())
            .collect();
        let seed = match expansion {
            Expansion::Tile => 0,
            Expansion::Resample { seed } => seed,
        };
        let mut events = EventSet::from_years(years, seed)?;
        if !self.labels.is_empty() {
            let labels = self
                .labels
                .iter()
                .map(|(dim, values)| {
                    let per_loss = sims
                        .iter()
                        .flat_map(|&y| rows[y].iter().map(|&k| values[k].clone()))
                        .collect();
                    (dim.clone(), per_loss)
                })
                .collect();
            events = events.with_labels(labels)?;
        }
        Ok(events.with_input_hash(self.source_hash.clone()))
    }
}

/// A hash of a source file's bytes, `"blake3:<hex>"`, for
/// [`YearLossTable::with_source_hash`].
///
/// ```
/// let h = prospicio_aggregate::ylt::hash_bytes(b"year,value\n1,5\n");
/// assert!(h.starts_with("blake3:"));
/// assert_ne!(h, prospicio_aggregate::ylt::hash_bytes(b"year,value\n1,6\n"));
/// ```
pub fn hash_bytes(bytes: &[u8]) -> String {
    InputHasher::new()
        .str("year_loss_table")
        .bytes(bytes)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> YearLossTable {
        // Three years, the third with no loss; year 1 has an event hitting
        // two classes.
        let peril = ["wind", "wind", "quake", "wind"].map(String::from).to_vec();
        let class = ["home", "auto", "home", "home"].map(String::from).to_vec();
        YearLossTable::new(&[1, 1, 2, 1], &[10.0, 4.0, 30.0, 2.0])
            .unwrap()
            .with_slots(vec![5, 5, 1, 2])
            .unwrap()
            .with_labels(vec![("peril".into(), peril), ("class".into(), class)])
            .unwrap()
            .with_n_years(3)
            .unwrap()
            .with_source_hash("blake3:00")
    }

    #[test]
    fn rows_are_ordered_by_year_then_slot() {
        let events = table().events(None, Expansion::Tile).unwrap();
        assert_eq!(events.n_sims(), 3);
        // Slot 2 first, then slot 5's two rows in the table's order.
        assert_eq!(events.events(0), &[2.0, 10.0, 4.0]);
        assert_eq!(events.events(1), &[30.0]);
        assert!(events.events(2).is_empty());
        let labels = events.labels(0).unwrap();
        assert_eq!(labels[2], ["wind".to_string(), "auto".into()]);
        assert_eq!(events.input_hash(), Some("blake3:00"));
        let totals = events.totals().unwrap();
        assert_eq!(totals.provenance().input_hash.as_deref(), Some("blake3:00"));
        assert_eq!(totals.provenance().samplers, None);
    }

    #[test]
    fn tiling_repeats_the_years_and_needs_a_multiple() {
        let t = table();
        let events = t.events(Some(6), Expansion::Tile).unwrap();
        for i in 0..6 {
            assert_eq!(events.events(i), events.events(i % 3));
        }
        let err = t.events(Some(7), Expansion::Tile).unwrap_err();
        assert!(err.to_string().contains("whole multiple"), "{err}");
    }

    #[test]
    fn resampling_draws_each_simulation_from_its_own_stream() {
        let t = table();
        let years = t
            .years_of_sims(Some(3_000), Expansion::Resample { seed: 7 })
            .unwrap();
        let again = t
            .years_of_sims(Some(10), Expansion::Resample { seed: 7 })
            .unwrap();
        assert_eq!(&years[..10], &again[..]);
        for y in 0..3 {
            let share = years.iter().filter(|&&s| s == y).count() as f64 / 3_000.0;
            assert!((share - 1.0 / 3.0).abs() < 0.03, "{y}: {share}");
        }
        let events = t.events(Some(7), Expansion::Resample { seed: 7 }).unwrap();
        assert_eq!(events.seed(), 7);
        for (i, &y) in years[..7].iter().enumerate() {
            assert_eq!(
                events.events(i),
                table().events(None, Expansion::Tile).unwrap().events(y)
            );
        }
    }

    #[test]
    fn the_checks_name_the_offending_rows() {
        let t = table();
        // Slot 5 of year 1 hits two classes of one peril: unique by peril
        // and class, not by peril alone.
        t.check_unique(&["peril", "class"]).unwrap();
        let err = t.check_unique(&["peril"]).unwrap_err().to_string();
        assert!(err.contains("rows 1 and 2"), "{err}");
        t.check_values("peril", &["wind", "quake"]).unwrap();
        let err = t.check_values("peril", &["wind"]).unwrap_err().to_string();
        assert!(err.contains("row 3"), "{err}");
        assert!(t.check_values("region", &["eu"]).is_err());

        assert!(YearLossTable::new(&[0], &[1.0]).is_err());
        assert!(YearLossTable::new(&[1], &[-1.0]).is_err());
        assert!(YearLossTable::new(&[1], &[f64::NAN]).is_err());
        assert!(YearLossTable::new(&[1, 2], &[1.0]).is_err());
        assert!(
            YearLossTable::new(&[4], &[1.0])
                .unwrap()
                .with_n_years(3)
                .is_err()
        );
        let no_slots = YearLossTable::new(&[1], &[1.0]).unwrap();
        assert!(no_slots.check_unique(&[]).is_err());
    }

    #[test]
    fn merged_sets_combine_their_hashes() {
        let a = table().events(None, Expansion::Tile).unwrap();
        let b = EventSet::from_years(vec![vec![1.0]; 3], 0).unwrap();
        assert_eq!(a.merge(&b).unwrap().input_hash(), Some("blake3:00"));
        let c = b.clone().with_input_hash(Some("blake3:11".into()));
        let ac = a.merge(&c).unwrap();
        let h = ac.input_hash().unwrap();
        assert!(h.starts_with("blake3:") && h != "blake3:00" && h != "blake3:11");
        assert_ne!(Some(h), c.merge(&a).unwrap().input_hash());
    }
}
