//! Aggregate lane: wrappers over `prospicio_aggregate` for the R `aggregate.R`
//! API (compound distributions and simulated events).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::{
    CompoundMethod, CompoundReport, EventLossTable, EventSet as EventSetInner,
};
use prospicio_prob::{Counting, Severity, SeverityDist};

use crate::distributions::{
    Grid, NegativeBinomial, Poisson, PredictiveDistribution, severity_from_robj,
};
use crate::{to_r, whole};

/// A claim count accepted by the aggregation functions.
pub(crate) enum AnyCount {
    Poisson(prospicio_prob::Poisson),
    NegativeBinomial(prospicio_prob::NegativeBinomial),
    Binomial(prospicio_prob::Binomial),
    Other(prospicio_prob::count_families::CountDist),
}

impl AnyCount {
    pub(crate) fn from_robj(obj: &Robj) -> Result<Self> {
        if let Ok(n) = <&Poisson>::try_from(obj) {
            return Ok(Self::Poisson(n.inner));
        }
        if let Ok(n) = <&NegativeBinomial>::try_from(obj) {
            return Ok(Self::NegativeBinomial(n.inner));
        }
        if let Ok(n) = <&crate::pareto::Binomial>::try_from(obj) {
            return Ok(Self::Binomial(n.inner));
        }
        if let Ok(n) = <&crate::counts::ClaimCount>::try_from(obj) {
            return Ok(Self::Other(n.inner.clone()));
        }
        Err(Error::Other(
            "frequency must be a poisson_count, negative_binomial_count, binomial_count or \
             claim_count_dist"
                .into(),
        ))
    }

    pub(crate) fn as_counting(&self) -> &(dyn Counting + Sync) {
        match self {
            Self::Poisson(n) => n,
            Self::NegativeBinomial(n) => n,
            Self::Binomial(n) => n,
            Self::Other(n) => n,
        }
    }
}

impl Counting for AnyCount {
    fn pmf(&self, k: u64) -> f64 {
        self.as_counting().pmf(k)
    }
    fn mean(&self) -> f64 {
        self.as_counting().mean()
    }
    fn variance(&self) -> f64 {
        self.as_counting().variance()
    }
    fn panjer_ab(&self) -> Option<(f64, f64)> {
        self.as_counting().panjer_ab()
    }
    fn pgf(&self, z: f64) -> f64 {
        self.as_counting().pgf(z)
    }
    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        self.as_counting().pgf_complex(z)
    }
    fn third_moment(&self) -> f64 {
        self.as_counting().third_moment()
    }
    fn third_cumulant(&self) -> f64 {
        self.as_counting().third_cumulant()
    }
}

pub(crate) fn compound_list(r: &CompoundReport) -> List {
    let method = match r.method {
        CompoundMethod::Panjer => "panjer",
        CompoundMethod::Fft => "fft",
    };
    list!(
        method = method,
        points = r.points as f64,
        tail_mass = r.tail_mass,
        aliasing_error = r.aliasing_error,
        expected_mean = r.expected_mean,
        grid_mean = r.grid_mean,
        mean_error = r.mean_error()
    )
}

/// Compound distribution of `S = X_1 + ... + X_N` on the severity grid's
/// step; `method` is "panjer" or "fft".
#[extendr]
fn compound(frequency: Robj, severity: Robj, points: f64, method: &str) -> Result<Grid> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("severity must be a grid_distribution".into()))?;
    let points = whole(points, "points")? as usize;
    let (inner, report) = match method {
        "panjer" => prospicio_aggregate::panjer(n.as_counting(), &sev.inner, points),
        "fft" => prospicio_aggregate::fft(n.as_counting(), &sev.inner, points),
        other => {
            return Err(Error::Other(format!(
                "method must be panjer or fft, got {other}"
            )));
        }
    }
    .map_err(to_r)?;
    Ok(Grid::with_report(inner, compound_list(&report)))
}

fn sizing(log2: f64, p: f64, p_star: f64) -> Result<prospicio_aggregate::Sizing> {
    Ok(prospicio_aggregate::Sizing {
        log2: whole(log2, "log2")? as u32,
        p,
        p_star,
        ..prospicio_aggregate::Sizing::default()
    })
}

fn grid_size_list(g: &prospicio_aggregate::GridSize) -> List {
    let method = match g.method {
        prospicio_aggregate::SizingMethod::Moments => "moments",
        prospicio_aggregate::SizingMethod::SingleBigJump => "single_big_jump",
    };
    list!(
        step = g.step,
        points = g.points as f64,
        extent = g.extent,
        method = method,
        moment_extent = g.moment_extent.unwrap_or(f64::NAN),
        jump_extent = g.jump_extent.unwrap_or(f64::NAN),
        tail_estimate = g.tail_estimate
    )
}

/// `aggregate`'s `round_bucket`, elementwise.
#[extendr]
fn round_bucket_rust(bs: &[f64]) -> Result<Vec<f64>> {
    bs.iter()
        .map(|&b| prospicio_aggregate::round_bucket(b).map_err(to_r))
        .collect()
}

/// The recommended FFT grid as a list.
#[extendr]
fn recommend_grid_rust(
    frequency: Robj,
    severity: Robj,
    log2: f64,
    p: f64,
    p_star: f64,
) -> Result<List> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = crate::distributions::severity_from_robj(&severity)?;
    let g = prospicio_aggregate::recommend_grid(n.as_counting(), &sev, &sizing(log2, p, p_star)?)
        .map_err(to_r)?;
    Ok(grid_size_list(&g))
}

fn units_from_lists(
    frequencies: &List,
    severities: &List,
) -> Result<Vec<(AnyCount, SeverityDist)>> {
    if frequencies.len() != severities.len() {
        return Err(Error::Other(format!(
            "{} frequencies but {} severities",
            frequencies.len(),
            severities.len()
        )));
    }
    frequencies
        .values()
        .zip(severities.values())
        .map(|(n, x)| Ok((AnyCount::from_robj(&n)?, severity_from_robj(&x)?)))
        .collect()
}

fn unit_refs(units: &[(AnyCount, SeverityDist)]) -> Vec<(&dyn Counting, &dyn Severity)> {
    units
        .iter()
        .map(|(n, x)| (n.as_counting() as &dyn Counting, x as &dyn Severity))
        .collect()
}

/// The recommended FFT grid for a sum of independent units as a list.
#[extendr]
fn recommend_portfolio_grid_rust(
    frequencies: List,
    severities: List,
    log2: f64,
    p: f64,
    p_star: f64,
) -> Result<List> {
    let units = units_from_lists(&frequencies, &severities)?;
    let g = prospicio_aggregate::recommend_portfolio_grid(
        &unit_refs(&units),
        &sizing(log2, p, p_star)?,
    )
    .map_err(to_r)?;
    Ok(grid_size_list(&g))
}

/// `settings` is `eps, power, good_enough, max_doublings, log2_cap, log2_floor`.
fn audit_settings(settings: &[f64]) -> Result<prospicio_aggregate::AuditSettings> {
    let [eps, power, good_enough, doublings, cap, floor] = settings else {
        return Err(Error::Other("the audit takes six settings".into()));
    };
    Ok(prospicio_aggregate::AuditSettings {
        eps: *eps,
        power: *power,
        good_enough: *good_enough,
        max_doublings: whole(*doublings, "max_doublings")? as u32,
        log2_cap: whole(*cap, "log2_cap")? as u32,
        log2_floor: whole(*floor, "log2_floor")? as u32,
    })
}

/// The center: `step` and `log2` when given (NaN when not), else the
/// recommended grid at `log2` (16 when not given).
fn audit_center(
    step: f64,
    log2: f64,
    recommend: impl FnOnce(
        &prospicio_aggregate::Sizing,
    ) -> prospicio_core::Result<prospicio_aggregate::GridSize>,
) -> Result<(f64, u32)> {
    let log2 = if log2.is_nan() {
        16
    } else {
        whole(log2, "log2")? as u32
    };
    if !step.is_nan() {
        return Ok((step, log2));
    }
    let s = prospicio_aggregate::Sizing {
        log2,
        ..prospicio_aggregate::Sizing::default()
    };
    let g = recommend(&s).map_err(to_r)?;
    Ok((g.step, g.points.trailing_zeros()))
}

/// The audit as a list: the cells as columns, the chosen cell's 1-based
/// index, and the verdict.
fn grid_audit_list(a: &prospicio_aggregate::GridAudit) -> List {
    let col = |f: &dyn Fn(&prospicio_aggregate::GridCell) -> f64| -> Vec<f64> {
        a.cells.iter().map(f).collect()
    };
    let term = |f: &dyn Fn(&prospicio_aggregate::ScoreTerms) -> Option<f64>| -> Vec<f64> {
        a.cells
            .iter()
            .map(|c| f(&c.terms).unwrap_or(f64::NAN))
            .collect()
    };
    let cells = list!(
        d_step = col(&|c| f64::from(c.d_step)),
        d_log2 = col(&|c| f64::from(c.d_log2)),
        step = col(&|c| c.step),
        log2 = col(&|c| f64::from(c.log2)),
        score = col(&|c| c.score),
        severity_mean = term(&|t| t.severity_mean),
        severity_cv = term(&|t| t.severity_cv),
        severity_skew = term(&|t| t.severity_skew),
        aggregate_mean = term(&|t| t.aggregate_mean),
        aggregate_cv = term(&|t| t.aggregate_cv),
        aggregate_skew = term(&|t| t.aggregate_skew),
        tail_mass = col(&|c| c.tail_mass),
        aliasing_error = col(&|c| c.aliasing_error),
        defective = a.cells.iter().map(|c| c.defective).collect::<Vec<_>>(),
        selected = a.cells.iter().map(|c| c.selected).collect::<Vec<_>>(),
        note = a.cells.iter().map(|c| c.note.as_str()).collect::<Vec<_>>()
    );
    let b = a.best();
    list!(
        cells = cells,
        selected = (a.selected + 1) as f64,
        step = b.step,
        log2 = f64::from(b.log2),
        score = b.score,
        probed = a.probed,
        grew = a.grew,
        reason = a.reason.as_str()
    )
}

/// `aggregate`'s `sharpen` audit of one compound distribution's grid.
#[extendr]
fn audit_grid_rust(
    frequency: Robj,
    severity: Robj,
    step: f64,
    log2: f64,
    settings: &[f64],
) -> Result<List> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = severity_from_robj(&severity)?;
    let s = audit_settings(settings)?;
    let (step, log2) = audit_center(step, log2, |z| {
        prospicio_aggregate::recommend_grid(n.as_counting(), &sev, z)
    })?;
    let a = prospicio_aggregate::audit_grid(n.as_counting(), &sev, step, log2, &s).map_err(to_r)?;
    Ok(grid_audit_list(&a))
}

/// The same audit for a sum of independent units.
#[extendr]
fn audit_portfolio_grid_rust(
    frequencies: List,
    severities: List,
    step: f64,
    log2: f64,
    settings: &[f64],
) -> Result<List> {
    let units = units_from_lists(&frequencies, &severities)?;
    let refs = unit_refs(&units);
    let s = audit_settings(settings)?;
    let (step, log2) = audit_center(step, log2, |z| {
        prospicio_aggregate::recommend_portfolio_grid(&refs, z)
    })?;
    let a = prospicio_aggregate::audit_portfolio_grid(&refs, step, log2, &s).map_err(to_r)?;
    Ok(grid_audit_list(&a))
}

/// The compound distribution by FFT on the recommended grid; its report
/// also holds the grid's sizing under `sizing`.
#[extendr]
fn compound_auto_rust(
    frequency: Robj,
    severity: Robj,
    log2: f64,
    p: f64,
    p_star: f64,
) -> Result<Grid> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = crate::distributions::severity_from_robj(&severity)?;
    let (inner, report, size) =
        prospicio_aggregate::fft_auto(n.as_counting(), &sev, &sizing(log2, p, p_star)?)
            .map_err(to_r)?;
    let mut rep = compound_list(&report);
    let mut names: Vec<String> = rep
        .names()
        .map_or_else(Vec::new, |n| n.map(String::from).collect());
    let mut values: Vec<Robj> = rep.values().collect();
    names.push("sizing".into());
    values.push(grid_size_list(&size).into());
    rep = List::from_names_and_values(names, values).map_err(|e| Error::Other(e.to_string()))?;
    Ok(Grid::with_report(inner, rep))
}

/// Simulated years of individual losses.
#[extendr]
pub(crate) struct EventSet {
    pub(crate) inner: EventSetInner,
}

#[extendr]
impl EventSet {
    fn simulate(frequency: Robj, severity: Robj, n_sims: f64, seed: f64) -> Result<Self> {
        let n = AnyCount::from_robj(&frequency)?;
        let sev = severity_from_robj(&severity)?;
        let n_sims = whole(n_sims, "n_sims")? as usize;
        let inner = prospicio_aggregate::simulate_events(
            n.as_counting(),
            &sev,
            n_sims,
            whole(seed, "seed")?,
        )
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// `years` is a list of numeric vectors; `sums_insured` and `times`
    /// NULL or lists of the same shape.
    fn from_years(years: List, sums_insured: Robj, seed: f64, times: Robj) -> Result<Self> {
        let to_vecs = |l: &List, what: &str| -> Result<Vec<Vec<f64>>> {
            l.values()
                .map(|v| {
                    v.as_real_vector()
                        .or_else(|| (v.len() == 0).then(Vec::new))
                        .ok_or_else(|| Error::Other(format!("{what} must be numeric vectors")))
                })
                .collect()
        };
        let years = to_vecs(&years, "years")?;
        let shape: Vec<usize> = years.iter().map(Vec::len).collect();
        let mut inner = EventSetInner::from_years(years, whole(seed, "seed")?).map_err(to_r)?;
        if !sums_insured.is_null() {
            let list = List::try_from(sums_insured)
                .map_err(|_| Error::Other("sums_insured must be a list or NULL".into()))?;
            let si = to_vecs(&list, "sums_insured")?;
            if si.iter().map(Vec::len).ne(shape.iter().copied()) {
                return Err(Error::Other(
                    "sums_insured must have the shape of years".into(),
                ));
            }
            inner = inner
                .with_sums_insured(si.into_iter().flatten().collect())
                .map_err(to_r)?;
        }
        if !times.is_null() {
            let list = List::try_from(times)
                .map_err(|_| Error::Other("times must be a list or NULL".into()))?;
            let t = to_vecs(&list, "times")?;
            if t.iter().map(Vec::len).ne(shape.iter().copied()) {
                return Err(Error::Other("times must have the shape of years".into()));
            }
            inner = inner
                .with_times(t.into_iter().flatten().collect())
                .map_err(to_r)?;
        }
        Ok(Self { inner })
    }

    /// Years in long form: `years` from 1 to `n_years`, one per loss;
    /// labels `label_values[[d]]` in dimension `label_dims[d]`, one per row
    /// (none when `label_dims` is empty).
    fn from_long(
        years: &[f64],
        losses: &[f64],
        n_years: f64,
        seed: f64,
        label_dims: Vec<String>,
        label_values: List,
    ) -> Result<Self> {
        let years = years
            .iter()
            .map(|&y| {
                let y = whole(y, "years")?;
                if y == 0 {
                    return Err(Error::Other("years run from 1".into()));
                }
                Ok(y as usize - 1)
            })
            .collect::<Result<Vec<_>>>()?;
        let (n_years, seed) = (whole(n_years, "n_years")? as usize, whole(seed, "seed")?);
        let inner = if label_dims.is_empty() {
            EventSetInner::from_long(&years, losses, n_years, seed)
        } else {
            let labels = labels_from(label_dims, label_values)?;
            EventSetInner::from_long_labeled(&years, losses, labels, n_years, seed)
        }
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// A year loss table's columns, checked and expanded: `years` from 1,
    /// `slots` NULL or one per row, `times` NULL or one per row (fractions
    /// of the year), labels as in `from_long`, `n_years` and
    /// `n_sims` NULL for their defaults, `expand` "tile" or "resample",
    /// `unique_by` NULL for every label dimension, the values allowed in
    /// `expected_dims[d]` in `expected_values[[d]]`, `source_hash` NULL or a
    /// string.
    #[allow(clippy::too_many_arguments)]
    fn from_ylt(
        years: &[f64],
        losses: &[f64],
        slots: Robj,
        times: Nullable<Vec<f64>>,
        label_dims: Vec<String>,
        label_values: List,
        n_years: Nullable<f64>,
        n_sims: Nullable<f64>,
        expand: &str,
        seed: f64,
        unique_by: Robj,
        expected_dims: Vec<String>,
        expected_values: List,
        source_hash: Robj,
    ) -> Result<Self> {
        use prospicio_aggregate::ylt::{Expansion, YearLossTable};
        let seed = whole(seed, "seed")?;
        let expansion = match expand {
            "tile" => Expansion::Tile,
            "resample" => Expansion::Resample { seed },
            other => {
                return Err(Error::Other(format!(
                    "expand must be \"tile\" or \"resample\", not {other:?}"
                )));
            }
        };
        let years = years
            .iter()
            .map(|&y| whole(y, "years").map(|y| y as usize))
            .collect::<Result<Vec<_>>>()?;
        let mut ylt = YearLossTable::new(&years, losses).map_err(to_r)?;
        let has_slots = !slots.is_null();
        if has_slots {
            let slots = slots
                .as_real_vector()
                .or_else(|| {
                    slots
                        .as_integer_vector()
                        .map(|v| v.into_iter().map(f64::from).collect())
                })
                .ok_or_else(|| Error::Other("slots must be numeric or NULL".into()))?
                .into_iter()
                .map(|s| whole(s, "slots"))
                .collect::<Result<Vec<_>>>()?;
            ylt = ylt.with_slots(slots).map_err(to_r)?;
        }
        if let Nullable::NotNull(times) = times {
            ylt = ylt.with_times(times).map_err(to_r)?;
        }
        let dims = label_dims.clone();
        if !label_dims.is_empty() {
            ylt = ylt
                .with_labels(labels_from(label_dims, label_values)?)
                .map_err(to_r)?;
        }
        if let Nullable::NotNull(n_years) = n_years {
            ylt = ylt
                .with_n_years(whole(n_years, "n_years")? as usize)
                .map_err(to_r)?;
        }
        if !source_hash.is_null() {
            let h = source_hash
                .as_str()
                .ok_or_else(|| Error::Other("source_hash must be a string or NULL".into()))?;
            ylt = ylt.with_source_hash(h);
        }
        if has_slots {
            let by: Vec<String> = if unique_by.is_null() {
                dims
            } else {
                unique_by
                    .as_string_vector()
                    .ok_or_else(|| Error::Other("unique_by must be character or NULL".into()))?
            };
            ylt.check_unique(&by.iter().map(String::as_str).collect::<Vec<_>>())
                .map_err(to_r)?;
        }
        for (dim, values) in labels_from(expected_dims, expected_values)? {
            ylt.check_values(&dim, &values.iter().map(String::as_str).collect::<Vec<_>>())
                .map_err(to_r)?;
        }
        let n_sims = match n_sims {
            Nullable::NotNull(n) => Some(whole(n, "n_sims")? as usize),
            Nullable::Null => None,
        };
        Ok(Self {
            inner: ylt.events(n_sims, expansion).map_err(to_r)?,
        })
    }

    /// Labels `values[[d]]` in dimension `dims[d]`, one per loss.
    fn with_labels(&self, dims: Vec<String>, values: List) -> Result<Self> {
        let labels = labels_from(dims, values)?;
        Ok(Self {
            inner: self.inner.clone().with_labels(labels).map_err(to_r)?,
        })
    }

    fn label_dims(&self) -> Vec<String> {
        self.inner.label_dims().to_vec()
    }

    /// The label combinations as columns, one character vector per
    /// dimension.
    fn label_combinations(&self) -> List {
        let combinations = self.inner.label_combinations();
        List::from_values((0..self.inner.label_dims().len()).map(|d| {
            Robj::from(
                combinations
                    .iter()
                    .map(|c| c[d].clone())
                    .collect::<Vec<String>>(),
            )
        }))
    }

    /// Year `sim`'s labels (1-based), one character vector per dimension;
    /// empty when there are none.
    fn labels(&self, sim: f64) -> Result<List> {
        let sim = whole(sim, "sim")? as usize;
        if sim == 0 || sim > self.inner.n_sims() {
            return Err(Error::Other(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        let Some(year) = self.inner.labels(sim - 1) else {
            return Ok(List::new(0));
        };
        Ok(List::from_values((0..self.inner.label_dims().len()).map(
            |d| Robj::from(year.iter().map(|v| v[d].clone()).collect::<Vec<String>>()),
        )))
    }

    fn with_aggregate_part(&self, part: &PredictiveDistribution) -> Result<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_aggregate_part(&part.inner)
                .map_err(to_r)?,
        })
    }

    fn has_aggregate_part(&self) -> bool {
        self.inner.has_aggregate_part()
    }

    /// Year `sim`'s aggregate part (1-based), one amount per column;
    /// empty without one.
    fn aggregate_part(&self, sim: f64) -> Result<Vec<f64>> {
        let sim = whole(sim, "sim")? as usize;
        if sim == 0 || sim > self.inner.n_sims() {
            return Err(Error::Other(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self
            .inner
            .aggregate_part(sim - 1)
            .map_or_else(Vec::new, <[f64]>::to_vec))
    }

    /// The aggregate part's columns' labels, one character vector per
    /// dimension; empty without a part or labels.
    fn aggregate_part_labels(&self) -> List {
        let Some(columns) = self.inner.aggregate_part_labels() else {
            return List::new(0);
        };
        List::from_values((0..self.inner.label_dims().len()).map(|d| {
            Robj::from(
                columns
                    .iter()
                    .map(|c| c[d].clone())
                    .collect::<Vec<String>>(),
            )
        }))
    }

    fn merge(&self, other: &EventSet) -> Result<Self> {
        Ok(Self {
            inner: self.inner.merge(&other.inner).map_err(to_r)?,
        })
    }

    /// The losses whose label `label` takes one of `values`, every year
    /// kept.
    fn filter(&self, label: &str, values: Vec<String>) -> Result<Self> {
        let values: Vec<&str> = values.iter().map(String::as_str).collect();
        Ok(Self {
            inner: self.inner.filter(label, &values).map_err(to_r)?,
        })
    }

    /// The years in another order: year `i` of the result is year
    /// `index[i]` (1-based) of this set.
    fn permute(&self, index: &[f64]) -> Result<Self> {
        let index = index
            .iter()
            .map(|&i| {
                if i.fract() == 0.0 && i >= 1.0 {
                    Ok(i as usize - 1)
                } else {
                    Err(Error::Other(format!(
                        "index must hold whole numbers from 1, not {i}"
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner: self.inner.permute(&index).map_err(to_r)?,
        })
    }

    /// A list of `n_years` event sets, the term's consecutive blocks.
    fn term_years(&self, n_years: f64) -> Result<List> {
        let years = self
            .inner
            .term_years(whole(n_years, "n_years")? as usize)
            .map_err(to_r)?;
        Ok(List::from_values(
            years.into_iter().map(|inner| Robj::from(Self { inner })),
        ))
    }

    /// A list of `n_years` event sets whose totals have the correlation
    /// `correlation` (one value or a row-major matrix).
    fn correlated_years(
        &self,
        n_years: f64,
        correlation: &[f64],
        kind: &str,
        draws: &str,
        seed: f64,
    ) -> Result<List> {
        let years = self
            .inner
            .correlated_years(
                whole(n_years, "n_years")? as usize,
                correlation,
                correlation_kind(kind)?,
                year_draws(draws)?,
                whole(seed, "seed")?,
            )
            .map_err(to_r)?;
        Ok(List::from_values(
            years.into_iter().map(|inner| Robj::from(Self { inner })),
        ))
    }

    /// The calibration: `correlation`, `ratio` and the `years`.
    fn calibrate_year_correlation(
        &self,
        n_years: f64,
        ratio: f64,
        kind: &str,
        draws: &str,
        seed: f64,
    ) -> Result<List> {
        let fit = self
            .inner
            .calibrate_year_correlation(
                whole(n_years, "n_years")? as usize,
                ratio,
                correlation_kind(kind)?,
                year_draws(draws)?,
                whole(seed, "seed")?,
            )
            .map_err(to_r)?;
        Ok(list!(
            correlation = fit.correlation,
            ratio = fit.ratio,
            years = List::from_values(
                fit.years
                    .into_iter()
                    .map(|inner| Robj::from(Self { inner }))
            )
        ))
    }

    /// The hours clause, labels `"drop"`, `"agree"` or `"largest"`, within
    /// each value of label dimension `by` when given (`"drop"` then read as
    /// `"agree"`).
    fn grouped(&self, window_hours: f64, labels: &str, by: Nullable<String>) -> Result<Self> {
        use prospicio_aggregate::GroupLabels;
        let by = match by {
            Nullable::NotNull(d) => Some(d),
            Nullable::Null => None,
        };
        let rule = match (labels, &by) {
            ("drop", None) => GroupLabels::Drop,
            ("drop" | "agree", _) => GroupLabels::Agree,
            ("largest", _) => GroupLabels::Largest,
            (other, _) => {
                return Err(Error::Other(format!(
                    "labels must be \"drop\", \"agree\" or \"largest\", not {other:?}"
                )));
            }
        };
        let inner = match by {
            Some(dim) => self.inner.grouped_by(&dim, window_hours, rule),
            None => self.inner.grouped_with(window_hours, rule),
        };
        Ok(Self {
            inner: inner.map_err(to_r)?,
        })
    }

    fn with_uniform_times(&self) -> Self {
        Self {
            inner: self.inner.clone().with_uniform_times(),
        }
    }

    fn with_seasonal_times(&self, weights: Vec<f64>) -> Result<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_seasonal_times(&weights)
                .map_err(to_r)?,
        })
    }

    fn has_times(&self) -> bool {
        self.inner.has_times()
    }

    /// Year `sim`'s times (1-based); empty when not known.
    fn times(&self, sim: f64) -> Result<Vec<f64>> {
        let sim = whole(sim, "sim")? as usize;
        if sim == 0 || sim > self.inner.n_sims() {
            return Err(Error::Other(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self
            .inner
            .times(sim - 1)
            .map(<[f64]>::to_vec)
            .unwrap_or_default())
    }

    fn has_sums_insured(&self) -> bool {
        self.inner.has_sums_insured()
    }

    /// Year `sim`'s sums insured (1-based); empty when not known.
    fn sums_insured(&self, sim: f64) -> Result<Vec<f64>> {
        let sim = whole(sim, "sim")? as usize;
        if sim == 0 || sim > self.inner.n_sims() {
            return Err(Error::Other(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self
            .inner
            .sums_insured(sim - 1)
            .map(<[f64]>::to_vec)
            .unwrap_or_default())
    }

    fn n_sims(&self) -> f64 {
        self.inner.n_sims() as f64
    }

    fn seed(&self) -> f64 {
        self.inner.seed() as f64
    }

    /// The hash of the years' source, or NULL.
    fn input_hash(&self) -> Nullable<String> {
        match self.inner.input_hash() {
            Some(h) => Nullable::NotNull(h.to_string()),
            None => Nullable::Null,
        }
    }

    /// Year `sim`'s losses; `sim` is 1-based, as in R.
    fn events(&self, sim: f64) -> Result<Vec<f64>> {
        let sim = whole(sim, "sim")? as usize;
        if sim == 0 || sim > self.inner.n_sims() {
            return Err(Error::Other(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self.inner.events(sim - 1).to_vec())
    }

    fn counts(&self) -> Vec<f64> {
        self.inner.counts().into_iter().map(|c| c as f64).collect()
    }

    fn totals(&self) -> Result<PredictiveDistribution> {
        let inner = self.inner.totals().map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    fn maxima(&self) -> Result<PredictiveDistribution> {
        let inner = self.inner.maxima().map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// A hash of the file at `path`, as `ylt::hash_bytes` gives it.
#[extendr]
fn ylt_file_hash_rust(path: &str) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| Error::Other(format!("{path}: {e}")))?;
    Ok(prospicio_aggregate::ylt::hash_bytes(&bytes))
}

/// The occurrence exceedance `1 − G_N(F(x))` at each `x`.
#[extendr]
fn occurrence_exceedance_rust(frequency: Robj, severity: Robj, x: &[f64]) -> Result<Vec<f64>> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = severity_from_robj(&severity)?;
    Ok(x.iter()
        .map(|&x| prospicio_aggregate::occurrence_exceedance(n.as_counting(), &sev, x))
        .collect())
}

/// The occurrence PML at each annual probability, as columns.
#[extendr]
fn occurrence_pml_rust(frequency: Robj, severity: Robj, p: &[f64]) -> Result<List> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = severity_from_robj(&severity)?;
    let pmls = p
        .iter()
        .map(|&p| prospicio_aggregate::occurrence_pml(n.as_counting(), &sev, p).map_err(to_r))
        .collect::<Result<Vec<_>>>()?;
    let column = |f: fn(&prospicio_aggregate::OccurrencePml) -> f64| -> Vec<f64> {
        pmls.iter().map(f).collect()
    };
    Ok(list!(
        probability = column(|x| x.probability),
        loss = column(|x| x.loss),
        severity_survival = column(|x| x.severity_survival),
        occurrence_return_period = column(|x| x.occurrence_return_period),
        annual_return_period = column(|x| x.annual_return_period)
    ))
}

/// `c(mean, variance, third_cumulant, cv, skew)` of the compound sum.
#[extendr]
fn compound_moments_rust(frequency: Robj, severity: Robj) -> Result<Vec<f64>> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = severity_from_robj(&severity)?;
    let m = prospicio_aggregate::compound_moments(n.as_counting(), &sev).map_err(to_r)?;
    Ok(vec![m.mean, m.variance, m.third_cumulant, m.cv(), m.skew()])
}

fn approximation(moments: &[f64], kind: &str) -> Result<prospicio_aggregate::Approximation> {
    use prospicio_aggregate::ApproximationKind as K;
    let kind = match kind {
        "normal" => K::Normal,
        "gamma" => K::Gamma,
        "lognormal" => K::Lognormal,
        "shifted_gamma" => K::ShiftedGamma,
        "shifted_lognormal" => K::ShiftedLognormal,
        "normal_power" => K::NormalPower,
        "wilson_hilferty" => K::WilsonHilferty,
        other => return Err(Error::Other(format!("unknown approximation {other}"))),
    };
    let [mean, variance, third_cumulant] = moments else {
        return Err(Error::Other(
            "moments must be c(mean, variance, third_cumulant)".into(),
        ));
    };
    prospicio_aggregate::CompoundMoments {
        mean: *mean,
        variance: *variance,
        third_cumulant: *third_cumulant,
    }
    .approximate(kind)
    .map_err(to_r)
}

/// The fitted parameters, named.
#[extendr]
fn approximation_parameters_rust(moments: &[f64], kind: &str) -> Result<List> {
    use prospicio_aggregate::Approximation as A;
    Ok(match approximation(moments, kind)? {
        A::Normal { mean, sd } => list!(mean = mean, sd = sd),
        A::Gamma { shape, scale } => list!(shape = shape, scale = scale),
        A::Lognormal { meanlog, sdlog } => list!(meanlog = meanlog, sdlog = sdlog),
        A::ShiftedGamma {
            shape,
            scale,
            shift,
        } => list!(shape = shape, scale = scale, shift = shift),
        A::ShiftedLognormal {
            meanlog,
            sdlog,
            shift,
        } => list!(meanlog = meanlog, sdlog = sdlog, shift = shift),
        A::NormalPower { mean, sd, skew } | A::WilsonHilferty { mean, sd, skew } => {
            list!(mean = mean, sd = sd, skew = skew)
        }
    })
}

/// The approximation's cdf at `x`, or its quantiles at `p` when `quantile`.
#[extendr]
fn approximation_eval_rust(
    moments: &[f64],
    kind: &str,
    values: &[f64],
    quantile: bool,
) -> Result<Vec<f64>> {
    let a = approximation(moments, kind)?;
    values
        .iter()
        .map(|&v| if quantile { a.quantile(v) } else { a.cdf(v) }.map_err(to_r))
        .collect()
}

/// Eventual ruin by Pollaczeck–Khinchine: `part` is "all", "cap",
/// "excess" or "stop_loss" with its `value`; `points` 0 for the
/// severity's. Capitals at each `p`, NA where the grid does not reach.
#[extendr]
fn eventual_ruin_rust(
    severity: Robj,
    rho: f64,
    points: f64,
    part: &str,
    value: f64,
    p: &[f64],
) -> Result<List> {
    use prospicio_aggregate::SeverityPart;
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("the severity must be a grid_distribution".into()))?;
    let part = match part {
        "all" => SeverityPart::All,
        "cap" => SeverityPart::AtMost(value),
        "excess" => SeverityPart::Above(value),
        "stop_loss" => SeverityPart::StopLoss(value),
        other => return Err(Error::Other(format!("unknown severity part {other:?}"))),
    };
    let points = match whole(points, "points")? as usize {
        0 => sev.inner.len(),
        n => n,
    };
    let ruin =
        prospicio_aggregate::pollaczeck_khinchine(&sev.inner, rho, points, part).map_err(to_r)?;
    let h = ruin.step();
    let na = |v: Option<f64>| v.unwrap_or(f64::NAN);
    Ok(list!(
        u = (0..ruin.psi().len())
            .map(|j| j as f64 * h)
            .collect::<Vec<f64>>(),
        psi = ruin.psi().to_vec(),
        capital = p.iter().map(|&p| na(ruin.capital(p))).collect::<Vec<f64>>(),
        capital_interpolated = p
            .iter()
            .map(|&p| na(ruin.capital_interpolated(p)))
            .collect::<Vec<f64>>(),
        lundberg_exponent = na(ruin.lundberg_exponent()),
        severity_mean = ruin.severity_mean(),
        aliasing_error = ruin.report().aliasing_error
    ))
}

/// Finite-horizon ruin for R's `finite_horizon_ruin()`: a `count`
/// (`frequency`) or a Poisson `rate` split into `sub_periods`; `points` 0
/// for the severity's. `psi` column-major, one column per period;
/// `capital` and `capital_interpolated` likewise, one row per `p`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn finite_horizon_ruin_rust(
    severity: Robj,
    premium: f64,
    periods: f64,
    frequency: Robj,
    rate: f64,
    sub_periods: f64,
    points: f64,
    p: &[f64],
) -> Result<List> {
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("the severity must be a grid_distribution".into()))?;
    let periods = whole(periods, "periods")? as usize;
    let sub_periods = whole(sub_periods, "sub_periods")? as usize;
    let points = match whole(points, "points")? as usize {
        0 => sev.inner.len(),
        n => n,
    };
    let ruin = if frequency.is_null() {
        prospicio_aggregate::finite_horizon_poisson(
            &sev.inner,
            rate,
            premium,
            periods,
            sub_periods,
            points,
        )
    } else {
        let n = AnyCount::from_robj(&frequency)?;
        prospicio_aggregate::finite_horizon(&sev.inner, n.as_counting(), premium, periods, points)
    }
    .map_err(to_r)?;
    let h = ruin.step();
    let na = |v: Option<f64>| v.unwrap_or(f64::NAN);
    let mut psi = Vec::with_capacity(points * periods);
    let mut cap = Vec::with_capacity(p.len() * periods);
    let mut cap_i = Vec::with_capacity(p.len() * periods);
    for t in 1..=periods {
        psi.extend_from_slice(ruin.psi(t).map_err(to_r)?);
        for &q in p {
            cap.push(na(ruin.capital(q, t).map_err(to_r)?));
            cap_i.push(na(ruin.capital_interpolated(q, t).map_err(to_r)?));
        }
    }
    Ok(list!(
        u = (0..points).map(|j| j as f64 * h).collect::<Vec<f64>>(),
        psi = psi,
        capital = cap,
        capital_interpolated = cap_i,
        step = h,
        aliasing_error = ruin.report().aliasing_error
    ))
}

/// Eventual ruin for renewal claims by Wiener–Hopf: `list(u, psi, maximum,
/// capital, capital_interpolated, premium_rate, severity_mean, mean_step)`.
/// `points` 0 for the severity's.
#[extendr]
fn wiener_hopf_rust(severity: Robj, wait: Robj, rho: f64, points: f64, p: &[f64]) -> Result<List> {
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("the severity must be a grid_distribution".into()))?;
    let wait = crate::distributions::dist_from_robj(&wait)?;
    let points = match whole(points, "points")? as usize {
        0 => sev.inner.len(),
        n => n,
    };
    let ruin = prospicio_aggregate::wiener_hopf(&sev.inner, &wait, rho, points).map_err(to_r)?;
    let h = ruin.step();
    let na = |v: Option<f64>| v.unwrap_or(f64::NAN);
    Ok(list!(
        u = (0..ruin.psi().len())
            .map(|j| j as f64 * h)
            .collect::<Vec<f64>>(),
        psi = ruin.psi().to_vec(),
        maximum = ruin.maximum().to_vec(),
        capital = p.iter().map(|&p| na(ruin.capital(p))).collect::<Vec<f64>>(),
        capital_interpolated = p
            .iter()
            .map(|&p| na(ruin.capital_interpolated(p)))
            .collect::<Vec<f64>>(),
        premium_rate = ruin.premium_rate(),
        severity_mean = ruin.severity_mean(),
        mean_step = ruin.mean_step()
    ))
}

/// Layer loss picks: `list(severity, straight_bottom, layers)`.
#[extendr]
fn layer_loss_picks_rust(
    severity: Robj,
    attachments: &[f64],
    picks: &[f64],
    claims: f64,
) -> Result<List> {
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("the severity must be a grid_distribution".into()))?;
    let p = prospicio_aggregate::layer_loss_picks(&sev.inner, attachments, picks, claims)
        .map_err(to_r)?;
    let col = |f: fn(&prospicio_aggregate::PickedLayer) -> f64| -> Vec<f64> {
        p.layers.iter().map(f).collect()
    };
    let layers = list!(
        bottom = col(|x| x.bottom),
        top = col(|x| x.top),
        before = col(|x| x.before),
        target = col(|x| x.target),
        weight = col(|x| x.weight),
        above = col(|x| x.above)
    );
    Ok(list!(
        severity = Grid::wrap(p.severity.clone()),
        straight_bottom = p.straight_bottom,
        layers = layers
    ))
}

fn elt(
    rate: &[f64],
    mean: &[f64],
    sd_independent: &[f64],
    sd_correlated: &[f64],
    exposure: &[f64],
) -> Result<EventLossTable> {
    EventLossTable::new(rate, mean, sd_independent, sd_correlated, exposure).map_err(to_r)
}

/// An event loss table's rate, expected annual loss, and at each of `x` its
/// exceedance frequency and OEP, and at each of `p` its occurrence PML.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn aggregate_elt_values(
    rate: &[f64],
    mean: &[f64],
    sd_independent: &[f64],
    sd_correlated: &[f64],
    exposure: &[f64],
    x: &[f64],
    p: &[f64],
) -> Result<List> {
    let t = elt(rate, mean, sd_independent, sd_correlated, exposure)?;
    let pml = p
        .iter()
        .map(|&q| t.occurrence_pml(q).map_err(to_r))
        .collect::<Result<Vec<f64>>>()?;
    Ok(list!(
        rate = t.rate(),
        expected_annual_loss = t.expected_annual_loss(),
        exceedance_frequency = x
            .iter()
            .map(|&v| t.exceedance_frequency(v))
            .collect::<Vec<_>>(),
        oep = x.iter().map(|&v| t.oep(v)).collect::<Vec<_>>(),
        pml = pml
    ))
}

/// Simulated years from an event loss table: `list(events, ids)`, with
/// `ids` a list per year of each loss's event row, from 1.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn aggregate_elt_simulate(
    rate: &[f64],
    mean: &[f64],
    sd_independent: &[f64],
    sd_correlated: &[f64],
    exposure: &[f64],
    n_years: f64,
    seed: f64,
) -> Result<List> {
    let t = elt(rate, mean, sd_independent, sd_correlated, exposure)?;
    let (inner, ids) = t
        .simulate(whole(n_years, "n_years")? as usize, whole(seed, "seed")?)
        .map_err(to_r)?;
    let ids = List::from_values(
        ids.into_iter()
            .map(|y| y.into_iter().map(|k| k as i32 + 1).collect::<Vec<i32>>()),
    );
    Ok(list!(events = EventSet { inner }, ids = ids))
}

/// A joint grid's results for R: the density row by row with its shape and
/// steps, both marginals and the total as grids, and the moments.
fn joint_list(
    joint: &prospicio_aggregate::JointGrid,
    report: Option<&prospicio_aggregate::BivariateReport>,
) -> Result<List> {
    let [rows, cols] = joint.shape();
    let [ha, hb] = joint.steps();
    let report: Robj = match report {
        Some(r) => list!(
            tail_mass = r.tail_mass.to_vec(),
            aliasing_mass = r.aliasing_mass.to_vec(),
            expected_means = r.expected_means.to_vec(),
            grid_means = r.grid_means.to_vec()
        )
        .into(),
        None => ().into(),
    };
    Ok(list!(
        probs = joint.probs().to_vec(),
        rows = rows as f64,
        cols = cols as f64,
        steps = vec![ha, hb],
        marginal_a = Grid::wrap(joint.marginal(0).map_err(to_r)?),
        marginal_b = Grid::wrap(joint.marginal(1).map_err(to_r)?),
        total = Grid::wrap(joint.total().map_err(to_r)?),
        means = vec![joint.mean(0), joint.mean(1)],
        variances = vec![joint.variance(0), joint.variance(1)],
        covariance = joint.covariance(),
        correlation = joint.correlation(),
        report = report
    ))
}

/// The joint annual loss of two lines by a 2-D FFT; `probs` is the
/// per-event severity row by row (`rows` points of line A).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fft_bivariate_rust(
    frequency: Robj,
    probs: &[f64],
    rows: f64,
    cols: f64,
    step_a: f64,
    step_b: f64,
    points_a: f64,
    points_b: f64,
) -> Result<List> {
    let n = AnyCount::from_robj(&frequency)?;
    let shape = [whole(rows, "rows")? as usize, whole(cols, "cols")? as usize];
    let sev = prospicio_aggregate::JointGrid::new([step_a, step_b], shape, probs.to_vec())
        .map_err(to_r)?;
    let points = [
        whole(points_a, "points")? as usize,
        whole(points_b, "points")? as usize,
    ];
    let (joint, report) =
        prospicio_aggregate::fft_bivariate(n.as_counting(), &sev, points).map_err(to_r)?;
    joint_list(&joint, Some(&report))
}

fn clash_from(counts: &[f64]) -> Result<prospicio_aggregate::Clash> {
    match counts {
        [a, b, c] => prospicio_aggregate::Clash::new(*a, *b, *c).map_err(to_r),
        _ => Err(Error::Other("a clash model takes three counts".into())),
    }
}

fn grid_arg(obj: &Robj, name: &str) -> Result<prospicio_prob::Grid> {
    <&Grid>::try_from(obj)
        .map(|g| g.inner.clone())
        .map_err(|_| Error::Other(format!("{name} must be a grid_distribution")))
}

/// The clash model from its counts: `c(neither, events, trigger_a, trigger_b)`.
#[extendr]
fn clash_rust(counts: &[f64]) -> Result<Vec<f64>> {
    let c = clash_from(counts)?;
    Ok(vec![c.neither(), c.events(), c.trigger_a(), c.trigger_b()])
}

/// The clash model's per-event severity, or with `frequency` its joint
/// annual loss.
#[extendr]
fn clash_aggregate_rust(
    counts: &[f64],
    frequency: Robj,
    a: Robj,
    b: Robj,
    points_a: f64,
    points_b: f64,
) -> Result<List> {
    let clash = clash_from(counts)?;
    let (a, b) = (grid_arg(&a, "a")?, grid_arg(&b, "b")?);
    if frequency.is_null() {
        return joint_list(&clash.severity(&a, &b).map_err(to_r)?, None);
    }
    let n = AnyCount::from_robj(&frequency)?;
    let points = [
        whole(points_a, "points")? as usize,
        whole(points_b, "points")? as usize,
    ];
    let (joint, report) = clash
        .aggregate(n.as_counting(), &a, &b, points)
        .map_err(to_r)?;
    joint_list(&joint, Some(&report))
}

/// The per-event severity of two losses coupled by a copula, each line hit
/// with its trigger probability.
#[extendr]
fn copula_severity_rust(
    a: Robj,
    b: Robj,
    copula: Robj,
    trigger_a: f64,
    trigger_b: f64,
) -> Result<List> {
    use prospicio_aggregate::bivariate::{JointGrid, triggered};
    let (a, b) = (grid_arg(&a, "a")?, grid_arg(&b, "b")?);
    let copula = <&crate::risk::RiskCopula>::try_from(&copula)
        .map_err(|_| Error::Other("copula must be a copula".into()))?;
    let a = triggered(&a, trigger_a).map_err(to_r)?;
    let b = triggered(&b, trigger_b).map_err(to_r)?;
    let joint = JointGrid::from_copula(&a, &b, copula.inner.as_copula()).map_err(to_r)?;
    joint_list(&joint, None)
}

extendr_module! {
    mod aggregate;
    fn copula_severity_rust;
    fn layer_loss_picks_rust;
    fn eventual_ruin_rust;
    fn finite_horizon_ruin_rust;
    fn wiener_hopf_rust;
    fn fft_bivariate_rust;
    fn clash_rust;
    fn clash_aggregate_rust;
    fn compound_moments_rust;
    fn approximation_parameters_rust;
    fn approximation_eval_rust;
    fn compound;
    fn compound_auto_rust;
    fn recommend_grid_rust;
    fn recommend_portfolio_grid_rust;
    fn audit_grid_rust;
    fn audit_portfolio_grid_rust;
    fn round_bucket_rust;
    fn occurrence_exceedance_rust;
    fn occurrence_pml_rust;
    fn ylt_file_hash_rust;
    fn year_correlation_for_ratio_rust;
    fn exposure_uniform_rust;
    impl EventSet;
    impl Timeline;
    fn aggregate_elt_values;
    fn aggregate_elt_simulate;
}

/// Labels from dimension names and a list of character vectors.
fn labels_from(dims: Vec<String>, values: List) -> Result<Vec<(String, Vec<String>)>> {
    if dims.len() != values.len() {
        return Err(Error::Other(format!(
            "{} label names for {} label vectors",
            dims.len(),
            values.len()
        )));
    }
    dims.into_iter()
        .zip(values.values())
        .map(|(dim, v)| {
            let v: Vec<String> = v
                .as_string_vector()
                .ok_or_else(|| Error::Other(format!("label {dim:?} must be character")))?;
            Ok((dim, v))
        })
        .collect()
}

fn correlation_kind(kind: &str) -> Result<prospicio_prob::portfolio::CorrelationKind> {
    use prospicio_prob::portfolio::CorrelationKind;
    match kind {
        "pearson" => Ok(CorrelationKind::Pearson),
        "spearman" => Ok(CorrelationKind::Spearman),
        _ => Err(Error::Other(format!(
            "kind must be \"pearson\" or \"spearman\", not {kind:?}"
        ))),
    }
}

fn year_draws(draws: &str) -> Result<prospicio_aggregate::term::YearDraws> {
    use prospicio_aggregate::term::YearDraws;
    match draws {
        "same" => Ok(YearDraws::Same),
        "blocks" => Ok(YearDraws::Blocks),
        _ => Err(Error::Other(format!(
            "draws must be \"same\" or \"blocks\", not {draws:?}"
        ))),
    }
}

/// `(T r² − 1) / (T − 1)`.
#[extendr]
fn year_correlation_for_ratio_rust(n_years: f64, ratio: f64) -> Result<f64> {
    prospicio_aggregate::term::year_correlation_for_ratio(
        whole(n_years, "n_years")? as usize,
        ratio,
    )
    .map_err(to_r)
}

pub(crate) fn cover_basis(basis: &str) -> Result<prospicio_aggregate::timeline::CoverBasis> {
    use prospicio_aggregate::timeline::CoverBasis;
    match basis {
        "losses_occurring" | "lod" => Ok(CoverBasis::LossesOccurring),
        "risks_attaching" | "rad" => Ok(CoverBasis::RisksAttaching),
        _ => Err(Error::Other(format!(
            "basis must be \"losses_occurring\" or \"risks_attaching\", not {basis:?}"
        ))),
    }
}

/// An exposure from its shares, row-major over `n_years` rows.
pub(crate) fn exposure(
    shares: &[f64],
    n_years: f64,
) -> Result<prospicio_aggregate::timeline::Exposure> {
    let n = whole(n_years, "n_years")? as usize;
    if n == 0 || shares.len() % n != 0 {
        return Err(Error::Other(format!(
            "{} shares do not make {n} rows",
            shares.len()
        )));
    }
    let rows = shares
        .chunks(shares.len() / n)
        .map(<[f64]>::to_vec)
        .collect();
    prospicio_aggregate::timeline::Exposure::new(rows).map_err(to_r)
}

/// The events of a term once, in accident months, with the aggregate
/// part by accident period.
#[extendr]
pub(crate) struct Timeline {
    pub(crate) inner: prospicio_aggregate::timeline::Timeline,
}

#[extendr]
impl Timeline {
    /// The timeline of `events` over `months`, the aggregate part spread
    /// by `pattern` (twelve weights, or `NULL` for uniform).
    fn new(events: Robj, months: f64, pattern: Nullable<Vec<f64>>) -> Result<Self> {
        let events = <&EventSet>::try_from(&events)
            .map_err(|_| Error::Other("events must be an event_set".into()))?;
        let inner = prospicio_aggregate::timeline::Timeline::new(
            events.inner.clone(),
            whole(months, "months")? as usize,
        )
        .map_err(to_r)?;
        Self::patterned(inner, pattern)
    }

    /// A timeline stitched from one event set per accident year.
    fn from_years(years: List, pattern: Nullable<Vec<f64>>) -> Result<Self> {
        let years = years
            .values()
            .map(|y| {
                <&EventSet>::try_from(&y)
                    .map(|e| e.inner.clone())
                    .map_err(|_| Error::Other("each year must be an event_set".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let inner = prospicio_aggregate::timeline::Timeline::from_years(&years).map_err(to_r)?;
        Self::patterned(inner, pattern)
    }

    /// The events, without the aggregate part.
    fn events(&self) -> EventSet {
        EventSet {
            inner: self.inner.events().clone(),
        }
    }

    fn months(&self) -> f64 {
        self.inner.months() as f64
    }

    fn aggregate_pattern(&self) -> Vec<f64> {
        self.inner.aggregate_pattern().to_vec()
    }

    fn has_aggregate_part(&self) -> bool {
        self.inner.has_aggregate_part()
    }

    /// The contract years under `basis`, with the exposure's shares
    /// row-major over `n_years` rows; with `sources`, a list of `years`
    /// and `sources` (each loss's index among the timeline's losses, from
    /// 1).
    fn years(&self, basis: &str, shares: &[f64], n_years: f64, sources: bool) -> Result<Robj> {
        let exposure = exposure(shares, n_years)?;
        let basis = cover_basis(basis)?;
        let (years, src) = if sources {
            let (y, s) = self
                .inner
                .years_with_sources(basis, &exposure)
                .map_err(to_r)?;
            (y, Some(s))
        } else {
            (self.inner.years(basis, &exposure).map_err(to_r)?, None)
        };
        let years = List::from_values(
            years
                .into_iter()
                .map(|inner| Robj::from(EventSet { inner })),
        );
        Ok(match src {
            None => years.into(),
            Some(src) => list!(
                years = years,
                sources = List::from_values(
                    src.into_iter()
                        .map(|s| { s.into_iter().map(|i| i as f64 + 1.0).collect::<Vec<f64>>() })
                )
            )
            .into(),
        })
    }
}

impl Timeline {
    fn patterned(
        inner: prospicio_aggregate::timeline::Timeline,
        pattern: Nullable<Vec<f64>>,
    ) -> Result<Self> {
        let inner = match pattern {
            Nullable::NotNull(p) => inner.with_aggregate_pattern(&p).map_err(to_r)?,
            Nullable::Null => inner,
        };
        Ok(Self { inner })
    }
}

/// `Exposure::uniform`'s shares, row-major, with its `months`.
#[extendr]
fn exposure_uniform_rust(n_years: f64, policy_months: f64) -> Result<List> {
    let e = prospicio_aggregate::timeline::Exposure::uniform(
        whole(n_years, "n_years")? as usize,
        whole(policy_months, "policy_months")? as usize,
    )
    .map_err(to_r)?;
    Ok(list!(
        shares = e.shares().iter().flatten().copied().collect::<Vec<f64>>(),
        months = e.months() as f64
    ))
}
