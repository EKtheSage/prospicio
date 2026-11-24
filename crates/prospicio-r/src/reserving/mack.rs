//! Mack's model: `MackFit`, `ClaimsDevelopmentResult`, and the calendar-year and development-correlation tests.

use super::*;

/// A Mack chain ladder fitted to every segment of a triangle column.
#[extendr]
pub(crate) struct MackFit {
    pub(crate) inner: SegmentFits<MackInner>,
}

#[extendr]
impl MackFit {
    /// The underlying chain-ladder projection.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.map(|m| m.chain_ladder.clone()),
        }
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

    fn total_process_risk(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_process_risk", "totals_frame()")?.total_process_risk)
    }

    fn total_parameter_risk(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_parameter_risk", "totals_frame()")?.total_parameter_risk)
    }

    fn total_standard_error(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_standard_error", "totals_frame()")?.total_standard_error)
    }

    fn total_cv(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_cv", "totals_frame()")?.total_cv())
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }

    /// Merz and Wüthrich's one-year view of a single-segment fit.
    fn claims_development_result(&self) -> Result<ClaimsDevelopmentResult> {
        let fit = single(&self.inner, "claims_development_result", "")?;
        Ok(ClaimsDevelopmentResult {
            inner: fit.claims_development_result().map_err(to_r)?,
        })
    }

    /// Mack's calendar-year effect test on the link ratios a
    /// single-segment fit averaged.
    fn calendar_year_effect(&self, ci: f64) -> Result<CalendarYearTest> {
        let fit = single(&self.inner, "calendar_year_effect", "")?;
        Ok(CalendarYearTest {
            inner: fit.calendar_year_effect(ci).map_err(to_r)?,
        })
    }

    /// Mack's development-factor correlation test on the link ratios a
    /// single-segment fit averaged.
    fn development_correlation(&self, ci: f64) -> Result<DevelopmentCorrelationTest> {
        let fit = single(&self.inner, "development_correlation", "")?;
        Ok(DevelopmentCorrelationTest {
            inner: fit.development_correlation(ci).map_err(to_r)?,
        })
    }
}

/// Merz and Wüthrich's (2008) one-year view of a Mack fit: standard errors
/// of the claims development result per origin and in total, in the next
/// calendar year and in each later one. `by_calendar_year` is row-major
/// over calendar year x origin.
#[extendr]
pub(crate) struct ClaimsDevelopmentResult {
    inner: ClaimsDevelopmentInner,
}

#[extendr]
impl ClaimsDevelopmentResult {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn one_year_standard_error(&self) -> Vec<f64> {
        self.inner.one_year_standard_error.clone()
    }

    fn total_one_year_standard_error(&self) -> f64 {
        self.inner.total_one_year_standard_error
    }

    fn by_calendar_year(&self) -> Vec<f64> {
        self.inner.by_calendar_year.concat()
    }

    fn total_by_calendar_year(&self) -> Vec<f64> {
        self.inner.total_by_calendar_year.clone()
    }

    fn run_off_standard_error(&self) -> Vec<f64> {
        self.inner.run_off_standard_error()
    }

    fn total_run_off_standard_error(&self) -> f64 {
        self.inner.total_run_off_standard_error()
    }
}

/// Mack's calendar-year effect test (`docs/design/reserving-v02.md`,
/// decision 27). `table` is one list entry per column of R ChainLadder's
/// `test_table`, one value per calendar diagonal.
#[extendr]
pub(crate) struct CalendarYearTest {
    pub(crate) inner: CalendarYearInner,
}

#[extendr]
impl CalendarYearTest {
    fn z(&self) -> f64 {
        self.inner.z
    }

    fn expected(&self) -> f64 {
        self.inner.expected
    }

    fn variance(&self) -> f64 {
        self.inner.variance
    }

    fn ci(&self) -> f64 {
        self.inner.ci
    }

    fn lower(&self) -> f64 {
        self.inner.lower
    }

    fn upper(&self) -> f64 {
        self.inner.upper
    }

    fn passes(&self) -> bool {
        self.inner.passes()
    }

    fn table(&self) -> List {
        let d = &self.inner.diagonals;
        let int = |f: fn(&prospicio_reserving::CalendarDiagonal) -> usize| -> Vec<i32> {
            d.iter().map(|d| f(d) as i32).collect()
        };
        list!(
            valuation = d
                .iter()
                .map(|d| d.valuation.to_string())
                .collect::<Vec<_>>(),
            small = int(|d| d.small),
            large = int(|d| d.large),
            z = int(|d| d.z),
            n = int(|d| d.n),
            m = int(|d| d.m),
            expected = d.iter().map(|d| d.expected).collect::<Vec<_>>(),
            variance = d.iter().map(|d| d.variance).collect::<Vec<_>>()
        )
    }
}

/// Mack's test for correlation between adjacent development factors.
/// `table` is one list entry per column, one value per pair of adjacent
/// link columns.
#[extendr]
pub(crate) struct DevelopmentCorrelationTest {
    pub(crate) inner: DevelopmentCorrelationInner,
}

#[extendr]
impl DevelopmentCorrelationTest {
    fn t(&self) -> f64 {
        self.inner.t
    }

    fn variance(&self) -> f64 {
        self.inner.variance
    }

    fn ci(&self) -> f64 {
        self.inner.ci
    }

    fn lower(&self) -> f64 {
        self.inner.lower
    }

    fn upper(&self) -> f64 {
        self.inner.upper
    }

    fn passes(&self) -> bool {
        self.inner.passes()
    }

    fn table(&self) -> List {
        let p = &self.inner.pairs;
        list!(
            age = p.iter().map(|p| p.age as i32).collect::<Vec<_>>(),
            n = p.iter().map(|p| p.n as i32).collect::<Vec<_>>(),
            t = p.iter().map(|p| p.t).collect::<Vec<_>>(),
            weight = p.iter().map(|p| p.weight).collect::<Vec<_>>()
        )
    }
}

extendr_module! {
    mod mack;
    impl MackFit;
    impl ClaimsDevelopmentResult;
    impl CalendarYearTest;
    impl DevelopmentCorrelationTest;
}
