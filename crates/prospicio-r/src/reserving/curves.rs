//! Growth curves: `ClarkFit`, `DevelopmentCurveFit`, `BenchmarkCurveFit` and `benchmark_curve_rust`.

use super::*;

/// A Clark LDF or Cape Cod fit of every segment of a triangle column.
/// Per-origin vectors run over the origins of each segment in turn.
#[extendr]
pub(crate) struct ClarkFit {
    pub(crate) inner: SegmentFits<ClarkInner>,
}

impl ClarkFit {
    /// The fit of a single-segment result, for `field`.
    fn one(&self, field: &str) -> Result<&ClarkInner> {
        single(&self.inner, field, "totals_frame()")
    }
}

#[extendr]
impl ClarkFit {
    /// The volume-weighted chain ladder of the same column.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
        }
    }

    /// "ldf" or "cape_cod".
    fn method(&self) -> &'static str {
        if self.inner.fits[0].elr.is_some() {
            "cape_cod"
        } else {
            "ldf"
        }
    }

    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.fits[0].curve)
    }

    /// `Inf` when development runs to infinity.
    fn max_age(&self) -> f64 {
        self.inner.fits[0].max_age.unwrap_or(f64::INFINITY)
    }

    fn omega(&self) -> Result<f64> {
        Ok(self.one("omega")?.omega)
    }

    fn theta(&self) -> Result<f64> {
        Ok(self.one("theta")?.theta)
    }

    /// The expected loss ratio; NaN for the LDF method.
    fn elr(&self) -> Result<f64> {
        if self.inner.fits[0].elr.is_none() {
            return Ok(f64::NAN);
        }
        Ok(self.one("elr")?.elr.unwrap_or(f64::NAN))
    }

    /// Length of the origin period in months.
    fn origin_width(&self) -> f64 {
        self.inner.fits[0].origin_width
    }

    /// Number of observed incremental values fitted.
    fn n_observations(&self) -> Result<i32> {
        let f = single(&self.inner, "n_observations", "segment()")?;
        Ok(f.n_observations as i32)
    }

    fn scale(&self) -> Result<f64> {
        Ok(self.one("scale")?.scale)
    }

    /// The parameter covariance, row-major.
    fn covariance(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "covariance", "segment()")?;
        Ok(f.covariance.concat())
    }

    /// Exposure per origin; empty for the LDF method.
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.exposure.clone().unwrap_or_default())
    }

    fn expected_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_ultimate.clone())
    }

    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ClarkInner::reserves)
    }

    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
    }

    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn total_process_risk(&self) -> Result<f64> {
        Ok(self.one("total_process_risk")?.total_process_risk)
    }

    fn total_parameter_risk(&self) -> Result<f64> {
        Ok(self.one("total_parameter_risk")?.total_parameter_risk)
    }

    fn total_standard_error(&self) -> Result<f64> {
        Ok(self.one("total_standard_error")?.total_standard_error)
    }

    /// Share of the expected ultimate developed by each development age
    /// (months; `Inf` gives 1).
    fn growth(&self, age: &[f64]) -> Result<Vec<f64>> {
        let f = single(&self.inner, "growth", "segment()")?;
        Ok(age.iter().map(|&a| f.growth(a)).collect())
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

/// The settings of `development_curve()`.
pub(crate) fn development_curve_settings(
    shape: &str,
    fit_from: Nullable<f64>,
    fit_to: Nullable<f64>,
    keep_observed_until: Nullable<f64>,
) -> Result<DevelopmentCurve> {
    Ok(DevelopmentCurve {
        shape: curve_shape(shape, "shape")?,
        fit_period: (
            optional_age(fit_from, "fit_period")?,
            optional_age(fit_to, "fit_period")?,
        ),
        keep_observed_until: optional_age(keep_observed_until, "keep_observed_until")?,
    })
}

/// The estimated development a curve is fitted to: `column` of a
/// single-segment triangle, averaged as `average` with the exclusions
/// `excl`.
pub(crate) fn curve_development(
    triangle: &TriangleInner,
    column: &str,
    average: &str,
    excl: &Robj,
) -> Result<prospicio_reserving::DevelopmentFit> {
    development(average, "log-linear", excl, None)?
        .fit(triangle, column)
        .map_err(to_r)
}

/// A development curve fitted over every age (`development_curve()`).
#[extendr]
pub(crate) struct DevelopmentCurveFit {
    pub(crate) inner: DevelopmentCurveInner,
}

#[extendr]
impl DevelopmentCurveFit {
    fn shape(&self) -> &'static str {
        curve_name(self.inner.curve.shape)
    }

    /// The fit period's ends, NaN for an open end.
    fn fit_period(&self) -> Vec<f64> {
        let end = |a: Option<Lag>| a.map_or(f64::NAN, f64::from);
        vec![
            end(self.inner.curve.fit_period.0),
            end(self.inner.curve.fit_period.1),
        ]
    }

    /// NaN for the curve at every age.
    fn keep_observed_until(&self) -> f64 {
        self.inner
            .curve
            .keep_observed_until
            .map_or(f64::NAN, f64::from)
    }

    fn parameters(&self) -> Vec<f64> {
        self.inner.parameters.clone()
    }

    fn development(&self) -> Vec<f64> {
        self.inner
            .development
            .iter()
            .map(|&a| f64::from(a))
            .collect()
    }

    fn observed(&self) -> Vec<f64> {
        self.inner.observed.clone()
    }

    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted.clone()
    }

    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.clone()
    }

    fn in_fit(&self) -> Vec<bool> {
        self.inner.in_fit.clone()
    }

    /// The first position the curve replaces, 0-based.
    fn attachment(&self) -> f64 {
        self.inner.attachment as f64
    }

    fn ldf(&self) -> Vec<f64> {
        self.inner.ldf.clone()
    }

    fn n_points(&self) -> f64 {
        self.inner.n_points as f64
    }

    fn n_parameters(&self) -> f64 {
        self.inner.n_parameters as f64
    }

    fn sum_of_squares(&self) -> f64 {
        self.inner.sum_of_squares
    }

    fn r_squared(&self) -> f64 {
        self.inner.r_squared
    }

    fn adjusted_r_squared(&self) -> f64 {
        self.inner.adjusted_r_squared
    }

    /// The curve's factor at each 0-based development position.
    fn factor(&self, position: &[f64]) -> Result<Vec<f64>> {
        position
            .iter()
            .map(|&p| Ok(self.inner.factor(whole(p, "position")? as usize)))
            .collect()
    }

    /// The product of the curve's factors over `extrap_periods` periods
    /// past the oldest age.
    fn tail(&self, extrap_periods: f64) -> Result<f64> {
        Ok(self
            .inner
            .tail(whole(extrap_periods, "extrap_periods")? as usize))
    }

    /// The selected pattern's ages and age-to-age factors.
    fn pattern(&self) -> List {
        let p = self.inner.pattern();
        list!(
            ages = p
                .values
                .iter()
                .map(|&(a, _)| f64::from(a))
                .collect::<Vec<_>>(),
            values = p.values.iter().map(|&(_, v)| v).collect::<Vec<_>>()
        )
    }
}

/// A growth curve fitted to a benchmark pattern (`benchmark_curve()`).
#[extendr]
pub(crate) struct BenchmarkCurveFit {
    inner: BenchmarkCurveInner,
}

/// The fit of `benchmark_curve()`: `values` at `ages` as `kind`
/// ("percent_developed" or "cdf"), `weights` one per value or `NULL` for
/// all 1, `max_age` `Inf` for none and `origin_width` `NULL` for no shift.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn benchmark_curve_rust(
    ages: Vec<f64>,
    values: Vec<f64>,
    kind: &str,
    curve: &str,
    weights: Nullable<Vec<f64>>,
    max_age: f64,
    origin_width: Nullable<f64>,
) -> Result<BenchmarkCurveFit> {
    let kind = match PatternKind::from_name(kind) {
        Some(PatternKind::AgeToAge) | None => {
            return Err(crate::unknown_name("kind", kind, &PatternKind::NAMES[2..]));
        }
        Some(k) => k,
    };
    if ages.len() != values.len() {
        return Err(Error::Other("pattern needs one age for each value".into()));
    }
    let pairs: Vec<(f64, f64)> = ages.into_iter().zip(values).collect();
    let weights = match weights {
        Nullable::NotNull(w) => Some(w),
        Nullable::Null => None,
    };
    let settings = BenchmarkCurve {
        curve: growth_curve(curve)?,
        max_age: clark_max_age(max_age),
        origin_width: optional(origin_width),
    };
    let inner = settings
        .fit(kind, &pairs, weights.as_deref())
        .map_err(to_r)?;
    Ok(BenchmarkCurveFit { inner })
}

#[extendr]
impl BenchmarkCurveFit {
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    fn omega(&self) -> f64 {
        self.inner.omega
    }

    fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// `Inf` when development runs to infinity.
    fn max_age(&self) -> f64 {
        self.inner.max_age.unwrap_or(f64::INFINITY)
    }

    /// NaN without the average-date shift.
    fn origin_width(&self) -> f64 {
        self.inner.origin_width.unwrap_or(f64::NAN)
    }

    fn ages(&self) -> Vec<f64> {
        self.inner.ages.clone()
    }

    fn observed(&self) -> Vec<f64> {
        self.inner.observed.clone()
    }

    fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }

    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted.clone()
    }

    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.clone()
    }

    fn sum_of_squares(&self) -> f64 {
        self.inner.sum_of_squares
    }

    /// The fitted percent developed at each development age in months.
    fn percent_developed(&self, age: &[f64]) -> Vec<f64> {
        age.iter()
            .map(|&a| self.inner.percent_developed(a))
            .collect()
    }

    /// The fitted factor to ultimate at each development age in months.
    fn cdf(&self, age: &[f64]) -> Vec<f64> {
        age.iter().map(|&a| self.inner.cdf(a)).collect()
    }

    /// The fitted pattern's percents developed at whole development ages,
    /// checked as `BenchmarkCurveFit::pattern` checks them.
    fn pattern(&self, ages: &[f64]) -> Result<Vec<f64>> {
        let lags = ages.iter().map(|&a| age(a)).collect::<Result<Vec<Lag>>>()?;
        let pattern = self.inner.pattern(&lags).map_err(to_r)?;
        Ok(pattern.values.iter().map(|&(_, p)| p).collect())
    }
}

extendr_module! {
    mod curves;
    fn benchmark_curve_rust;
    impl ClarkFit;
    impl BenchmarkCurveFit;
    impl DevelopmentCurveFit;
}
