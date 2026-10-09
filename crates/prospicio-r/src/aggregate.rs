//! Aggregate lane: wrappers over `prospicio_aggregate` for the R `aggregate.R`
//! API (compound distributions and simulated events).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::{CompoundMethod, CompoundReport, EventSet as EventSetInner};
use prospicio_prob::Counting;

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
}

extendr_module! {
    mod aggregate;
    fn compound;
    fn compound_auto_rust;
    fn recommend_grid_rust;
    fn round_bucket_rust;
    impl EventSet;
}
