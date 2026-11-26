//! The Mack bootstrap's fit, `MackBootstrapFit`.

use super::*;

/// Mack's bootstrap, the lifetime view, of every segment of a triangle
/// column. `residuals` is row-major over origin x development, NaN where
/// there is no residual.
#[extendr]
pub(crate) struct MackBootstrapFit {
    pub(crate) inner: MackBootstrapFits,
}

#[extendr]
impl MackBootstrapFit {
    /// The chain ladder of Mack's model.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.mack.chain_ladder.clone()),
        }
    }

    /// Mack's model on the observed triangle, with the bootstrap's tail.
    fn mack(&self) -> MackFit {
        MackFit {
            inner: self.inner.segments.map(|s| s.mack.clone()),
        }
    }

    fn residuals(&self) -> Result<Vec<f64>> {
        let s = single(&self.inner.segments, "residuals", "")?;
        Ok(s.residuals.clone())
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Each simulation's future incremental values, or `NULL` when the fit
    /// did not keep them (`cells = FALSE`, or a segment picked with
    /// `segment()` that has no future cell).
    fn cells(&self) -> Nullable<PredictiveDistribution> {
        match &self.inner.cells {
            Some(inner) => Nullable::NotNull(PredictiveDistribution {
                inner: inner.clone(),
            }),
            None => Nullable::Null,
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
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
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

extendr_module! {
    mod mack_bootstrap;
    impl MackBootstrapFit;
}
