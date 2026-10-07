//! Fitting every segment of a triangle at once, with long results
//! (`docs/design/triangle.md`, decision 5).
//!
//! A method's `fit_segments` fits one measure column in each segment
//! (index position) of a triangle and returns a [`SegmentFits`]: the key
//! names, each segment's label and its fit. Its long tables have one row
//! per segment × origin, segment or segment × age, with the key values as
//! named columns.

use prospicio_core::{Lag, Period};

use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::mack::MackFit;
use crate::triangle::{Label, Segment, Triangle};

/// A long table of method results: key columns, then an origin or age
/// column when the table has one row per origin or per age, then value
/// columns. Every column has one value per row.
#[derive(Debug, Clone, PartialEq)]
pub struct FitTable {
    /// Key columns as `(name, values)`, in the triangle's key order.
    pub keys: Vec<(String, Vec<String>)>,
    /// Origin period of each row, in a table by origin.
    pub origin: Option<Vec<Period>>,
    /// Development age of each row, in a table by age.
    pub age: Option<Vec<Lag>>,
    /// Value columns as `(name, values)`.
    pub values: Vec<(String, Vec<f64>)>,
}

impl FitTable {
    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.values.first().map_or(0, |(_, v)| v.len())
    }

    /// The value column named `name`.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.values
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }
}

/// One method's fits of every segment of a triangle column, in the
/// triangle's index order.
///
/// ```
/// use prospicio_reserving::{ChainLadder, DevelopmentColumn, Grain, Label, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2021, 2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &["Auto", "Auto", "Auto", "Home", "Home", "Home"])],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
///     values: &[("paid", &[100.0, 150.0, 200.0, 10.0, 20.0, 30.0])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fits = ChainLadder::default().fit_segments(&tri, "paid")?;
/// assert_eq!(fits.len(), 2);
/// assert_eq!(fits.get(&Label::new(["Home"])).unwrap().ultimate, vec![20.0, 60.0]);
/// let long = fits.to_long();
/// assert_eq!(long.keys[0].1, ["Auto", "Auto", "Home", "Home"]);
/// assert_eq!(long.column("reserve").unwrap(), [0.0, 100.0, 0.0, 30.0]);
/// assert_eq!(fits.totals().column("reserve").unwrap(), [100.0, 30.0]);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentFits<T> {
    /// Names of the triangle's key columns; empty without keys.
    pub key_names: Vec<String>,
    /// Label of each segment, one part per key.
    pub labels: Vec<Label>,
    /// Fit of each segment, following `labels`.
    pub fits: Vec<T>,
}

impl<T> SegmentFits<T> {
    /// Number of segments.
    pub fn len(&self) -> usize {
        self.fits.len()
    }

    /// Whether there are no segments (never, for a fitted triangle).
    pub fn is_empty(&self) -> bool {
        self.fits.is_empty()
    }

    /// The fit of the segment labelled `label`.
    pub fn get(&self, label: &Label) -> Option<&T> {
        let i = self.labels.iter().position(|l| l == label)?;
        Some(&self.fits[i])
    }

    /// Each segment's label and fit, in index order.
    pub fn iter(&self) -> impl Iterator<Item = (&Label, &T)> {
        self.labels.iter().zip(&self.fits)
    }

    /// Position of the one segment whose keys have the given values. Keys
    /// not named may take any value, so with one segment `&[]` finds it.
    /// An unknown or repeated key, a value no segment has, or a choice that
    /// matches several segments is an error.
    pub fn position(&self, keys: &[(&str, &str)]) -> Result<usize> {
        find_segment(&self.key_names, &self.labels, keys)
    }

    /// The one segment chosen as in [`position`](Self::position), as a
    /// single-segment result with the same key names.
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self>
    where
        T: Clone,
    {
        let i = self.position(keys)?;
        Ok(Self {
            key_names: self.key_names.clone(),
            labels: vec![self.labels[i].clone()],
            fits: vec![self.fits[i].clone()],
        })
    }

    /// The same segments with `f` applied to each fit.
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> SegmentFits<U> {
        SegmentFits {
            key_names: self.key_names.clone(),
            labels: self.labels.clone(),
            fits: self.fits.iter().map(f).collect(),
        }
    }

    /// Key columns for `rows[s]` rows of segment `s`.
    pub(crate) fn key_columns(&self, rows: impl Fn(usize) -> usize) -> Vec<(String, Vec<String>)> {
        self.key_names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let values = self
                    .labels
                    .iter()
                    .enumerate()
                    .flat_map(|(s, l)| std::iter::repeat_n(l.parts()[k].clone(), rows(s)))
                    .collect();
                (name.clone(), values)
            })
            .collect()
    }
}

/// Fits `column` in every segment of `triangle` with `fit`. A failure is
/// reported with the segment's label when the triangle has keys.
pub(crate) fn fit_each<T>(
    triangle: &Triangle,
    column: &str,
    mut fit: impl FnMut(&Segment) -> Result<T>,
) -> Result<SegmentFits<T>> {
    let segments = triangle.segments(column)?;
    collect_fits(triangle, segments.iter().map(&mut fit))
}

/// Fits `column` in every segment of `triangle` with `fit`, which also gets
/// the same index position's `exposure` column. A failure is reported with
/// the segment's label when the triangle has keys.
pub(crate) fn fit_each_with_exposure<T>(
    triangle: &Triangle,
    column: &str,
    exposure: &str,
    mut fit: impl FnMut(&Segment, &Segment) -> Result<T>,
) -> Result<SegmentFits<T>> {
    let segments = triangle.segments(column)?;
    let exposures = triangle.segments(exposure)?;
    collect_fits(
        triangle,
        segments.iter().zip(&exposures).map(|(s, e)| fit(s, e)),
    )
}

/// The fits of every index position of `triangle`, in index order, with a
/// failure labelled by its segment when the triangle has keys.
fn collect_fits<T>(
    triangle: &Triangle,
    fits: impl Iterator<Item = Result<T>>,
) -> Result<SegmentFits<T>> {
    let keyed = !triangle.key_names().is_empty();
    let fits = fits
        .zip(triangle.index())
        .map(|(fit, label)| {
            fit.map_err(|e| {
                if keyed {
                    Error::InSegment {
                        label: label.to_string(),
                        source: Box::new(e),
                    }
                } else {
                    e
                }
            })
        })
        .collect::<Result<_>>()?;
    Ok(SegmentFits {
        key_names: triangle.key_names().to_vec(),
        labels: triangle.index().to_vec(),
        fits,
    })
}

/// A per-segment fit built on a chain-ladder projection, whose results go
/// into the long tables of [`SegmentFits`].
pub trait ReserveFit {
    /// The chain-ladder projection: origins, development pattern and latest
    /// values.
    fn chain_ladder(&self) -> &ChainLadderFit;

    /// This method's ultimate per origin; the chain ladder's by default.
    /// The long tables' `ultimate` and `reserve` come from it.
    fn ultimate(&self) -> &[f64] {
        &self.chain_ladder().ultimate
    }

    /// Value columns per origin after `latest`, `ultimate` and `reserve`.
    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        Vec::new()
    }

    /// Value columns of the segment total after `latest`, `ultimate` and
    /// `reserve`.
    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
}

impl ReserveFit for ChainLadderFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        self
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        tail_columns(self)
    }
}

/// The segment's tail: its factor from the oldest age to ultimate, sigma
/// and standard error.
fn tail_columns(cl: &ChainLadderFit) -> Vec<(&'static str, f64)> {
    vec![
        ("tail", cl.tail.factor),
        ("tail_sigma", cl.tail.sigma),
        ("tail_std_err", cl.tail.std_err),
    ]
}

impl ReserveFit for MackFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![
            ("process_risk", self.process_risk.clone()),
            ("parameter_risk", self.parameter_risk.clone()),
            ("standard_error", self.standard_error.clone()),
        ]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        let mut columns = vec![
            ("process_risk", self.total_process_risk),
            ("parameter_risk", self.total_parameter_risk),
            ("standard_error", self.total_standard_error),
        ];
        columns.extend(tail_columns(&self.chain_ladder));
        columns
    }
}

/// Appends `columns` of every segment, in segment order, to `values`.
fn push_columns(
    values: &mut Vec<(String, Vec<f64>)>,
    per_segment: impl Iterator<Item = Vec<(&'static str, Vec<f64>)>>,
) {
    for (s, columns) in per_segment.enumerate() {
        if s == 0 {
            values.extend(columns.into_iter().map(|(n, v)| (n.to_string(), v)));
        } else {
            let start = values.len() - columns.len();
            for ((_, all), (_, v)) in values[start..].iter_mut().zip(columns) {
                all.extend(v);
            }
        }
    }
}

impl<T: ReserveFit> SegmentFits<T> {
    /// One row per segment × origin: `latest`, `ultimate`, `reserve` and
    /// the method's own per-origin columns (for Mack the standard errors).
    pub fn to_long(&self) -> FitTable {
        let n_origins = |s: usize| self.fits[s].chain_ladder().origins.len();
        let mut values = Vec::new();
        push_columns(
            &mut values,
            self.fits.iter().map(|f| {
                let mut columns = vec![
                    ("latest", f.chain_ladder().latest.clone()),
                    ("ultimate", f.ultimate().to_vec()),
                    ("reserve", reserves(f)),
                ];
                columns.extend(f.origin_columns());
                columns
            }),
        );
        FitTable {
            keys: self.key_columns(n_origins),
            origin: Some(
                self.fits
                    .iter()
                    .flat_map(|f| f.chain_ladder().origins.iter().copied())
                    .collect(),
            ),
            age: None,
            values,
        }
    }

    /// One row per segment: total `latest`, `ultimate`, `reserve` and the
    /// method's own totals (for Mack the standard errors of the total; for
    /// the chain ladder and Mack the segment's `tail`, `tail_sigma` and
    /// `tail_std_err`).
    pub fn totals(&self) -> FitTable {
        let mut values = Vec::new();
        push_columns(
            &mut values,
            self.fits.iter().map(|f| {
                let mut columns = vec![
                    ("latest", vec![f.chain_ladder().latest.iter().sum()]),
                    ("ultimate", vec![total_ultimate(f)]),
                    ("reserve", vec![total_reserve(f)]),
                ];
                columns.extend(f.total_columns().into_iter().map(|(n, v)| (n, vec![v])));
                columns
            }),
        );
        FitTable {
            keys: self.key_columns(|_| 1),
            origin: None,
            age: None,
            values,
        }
    }

    /// One row per segment × development age: `ldf` (the selected factor
    /// from this age to the next), `cdf` (to ultimate, with the tail),
    /// `sigma` and `std_err` (as estimated). The oldest age has no next age,
    /// so its `ldf`, `sigma` and `std_err` are NaN; its `cdf` is the tail
    /// factor.
    pub fn development_table(&self) -> FitTable {
        let n_ages = |s: usize| self.fits[s].chain_ladder().cdf.len();
        let padded = |v: &[f64]| {
            let mut v = v.to_vec();
            v.push(f64::NAN);
            v
        };
        let mut values = Vec::new();
        push_columns(
            &mut values,
            self.fits.iter().map(|f| {
                let cl = f.chain_ladder();
                let dev = &cl.development;
                vec![
                    ("ldf", padded(cl.ldf())),
                    ("cdf", cl.cdf.clone()),
                    ("sigma", padded(&dev.sigma)),
                    ("std_err", padded(&dev.std_err)),
                ]
            }),
        );
        FitTable {
            keys: self.key_columns(n_ages),
            origin: None,
            age: Some(
                self.fits
                    .iter()
                    .flat_map(|f| f.chain_ladder().development.development.iter().copied())
                    .collect(),
            ),
            values,
        }
    }

    /// Total ultimate over every segment and origin.
    pub fn total_ultimate(&self) -> f64 {
        self.fits.iter().map(total_ultimate).sum()
    }

    /// Total reserve over every segment and origin.
    pub fn total_reserve(&self) -> f64 {
        self.fits.iter().map(total_reserve).sum()
    }
}

/// A fit's reserve (its ultimate minus the latest value) per origin.
fn reserves(fit: &impl ReserveFit) -> Vec<f64> {
    let latest = &fit.chain_ladder().latest;
    fit.ultimate()
        .iter()
        .zip(latest)
        .map(|(u, l)| u - l)
        .collect()
}

/// A fit's ultimate summed over origins.
fn total_ultimate(fit: &impl ReserveFit) -> f64 {
    fit.ultimate().iter().sum()
}

/// A fit's total ultimate minus its total latest value.
fn total_reserve(fit: &impl ReserveFit) -> f64 {
    total_ultimate(fit) - fit.chain_ladder().latest.iter().sum::<f64>()
}

/// Position of the one label whose keys have the given values, as
/// [`SegmentFits::position`] and [`Triangle::view`] choose a segment. Keys
/// not named may take any value. An unknown or repeated key, a value no
/// label has, or a choice that matches several labels is an error.
pub(crate) fn find_segment(
    key_names: &[String],
    labels: &[Label],
    keys: &[(&str, &str)],
) -> Result<usize> {
    let mut positions: Vec<usize> = Vec::with_capacity(keys.len());
    for (key, value) in keys {
        let k = key_names
            .iter()
            .position(|n| n == key)
            .ok_or_else(|| Error::UnknownKey(key.to_string()))?;
        if positions.contains(&k) {
            return Err(Error::DuplicateKey(key.to_string()));
        }
        if !labels.iter().any(|l| l.parts()[k] == *value) {
            return Err(Error::UnknownKeyValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }
        positions.push(k);
    }
    let matches: Vec<usize> = (0..labels.len())
        .filter(|&i| {
            let parts = labels[i].parts();
            keys.iter()
                .zip(&positions)
                .all(|((_, value), &k)| parts[k] == *value)
        })
        .collect();
    match matches[..] {
        [i] => Ok(i),
        [] => Err(Error::NoSegments),
        _ => Err(Error::AmbiguousSegment(matches.len())),
    }
}

#[cfg(test)]
mod tests {
    use crate::{ChainLadder, DevelopmentColumn, Error, Grain, Label, Long, Month, Triangle};

    fn two_segments(home: &[f64]) -> Triangle {
        let origin = [2020, 2020, 2021, 2020, 2020, 2021].map(Month::january);
        Triangle::from_long(&Long {
            keys: &[("lob", &["Auto", "Auto", "Auto", "Home", "Home", "Home"])],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
            values: &[("paid", &[&[100.0, 150.0, 200.0][..], home].concat())],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// Auto covers 2019-2022 at four ages; Home starts in 2020, has an
    /// empty 2021 inside its range when `gap`, and three ages.
    fn ragged(gap: bool) -> Triangle {
        let mut keys = Vec::new();
        let (mut origin, mut ages, mut paid) = (Vec::new(), Vec::new(), Vec::new());
        let auto: [&[f64]; 4] = [
            &[100.0, 150.0, 165.0, 170.0],
            &[110.0, 160.0, 180.0],
            &[120.0, 175.0],
            &[130.0],
        ];
        for (k, row) in auto.iter().enumerate() {
            for (d, v) in row.iter().enumerate() {
                keys.push("Auto");
                origin.push(Month::january(2019 + k as i32));
                ages.push(12 * (d as u32 + 1));
                paid.push(*v);
            }
        }
        let home: [(i32, &[f64]); 3] = [
            (2020, &[10.0, 15.0, 16.0]),
            (2021, if gap { &[] } else { &[11.0, 17.0] }),
            (2022, &[12.0]),
        ];
        for (year, row) in home {
            for (d, v) in row.iter().enumerate() {
                keys.push("Home");
                origin.push(Month::january(year));
                ages.push(12 * (d as u32 + 1));
                paid.push(*v);
            }
        }
        Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn segments_fit_on_their_own_origins_and_ages() {
        // Home starts a year after Auto and has one age fewer: it fits on
        // 2020-2022 and three ages, exactly as on its own.
        let tri = ragged(false);
        let fits = ChainLadder::default().fit_segments(&tri, "paid").unwrap();
        let home = fits.segment(&[("lob", "Home")]).unwrap();
        let alone = ChainLadder::default()
            .fit(&tri.select(&[("lob", &["Home"])]).unwrap(), "paid")
            .unwrap();
        // Equal up to the unestimable last sigma (NaN in both).
        let (h, a) = (&home.fits[0], &alone);
        assert_eq!(h.origins, a.origins);
        assert_eq!(h.development.ldf, a.development.ldf);
        assert_eq!(h.latest, a.latest);
        assert_eq!(h.ultimate, a.ultimate);
        assert_eq!(alone.origins.len(), 3);
        assert_eq!(alone.development.development, [12, 24, 36]);
        assert_eq!(alone.development.ldf.len(), 2);
        let auto = fits.segment(&[("lob", "Auto")]).unwrap();
        assert_eq!(auto.fits[0].origins.len(), 4);
        // The bootstrap runs on the ragged segments too: one joint
        // distribution with 4 + 3 origin components.
        let boot = crate::OdpBootstrap {
            n_sims: 50,
            ..Default::default()
        }
        .fit_segments(&tri, "paid")
        .unwrap();
        assert_eq!(boot.reserves.n_components(), 7);
    }

    #[test]
    fn a_failing_segment_is_named() {
        // A gap inside Home's range (2021) is still an error.
        let tri = ragged(true);
        assert_eq!(
            ChainLadder::default().fit_segments(&tri, "paid"),
            Err(Error::InSegment {
                label: "Home".into(),
                source: Box::new(Error::EmptyOrigin("2021".into())),
            })
        );
        let msg = ChainLadder::default()
            .fit_segments(&tri, "paid")
            .unwrap_err()
            .to_string();
        assert_eq!(msg, "segment Home: origin 2021 has no observed values");
    }

    #[test]
    fn tables_follow_segments() {
        let fits = ChainLadder::default()
            .fit_segments(&two_segments(&[10.0, 20.0, 30.0]), "paid")
            .unwrap();
        let dev = fits.development_table();
        assert_eq!(
            dev.keys,
            [(
                "lob".to_string(),
                vec!["Auto", "Auto", "Home", "Home"]
                    .into_iter()
                    .map(String::from)
                    .collect()
            )]
        );
        assert_eq!(dev.age, Some(vec![12, 24, 12, 24]));
        assert_eq!(dev.column("cdf").unwrap(), [1.5, 1.0, 2.0, 1.0]);
        let one = fits.segment(&[("lob", "Home")]).unwrap();
        assert_eq!(one.labels, [Label::new(["Home"])]);
        assert_eq!(one.totals().column("reserve").unwrap(), [30.0]);
        assert_eq!(fits.totals().column("tail").unwrap(), [1.0, 1.0]);
        assert_eq!(fits.totals().column("tail_sigma").unwrap(), [0.0, 0.0]);
        assert_eq!(fits.position(&[]), Err(Error::AmbiguousSegment(2)));
        assert_eq!(
            fits.position(&[("lob", "Auto"), ("lob", "Auto")]),
            Err(Error::DuplicateKey("lob".into()))
        );
    }
}
