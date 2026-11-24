//! The Bayesian chain ladder's fits, `BayesOdpFit` and `BayesOdpFits`.

use super::*;

/// A Bayesian ODP chain ladder of a single-segment triangle column.
#[extendr]
pub(crate) struct BayesOdpFit {
    pub(crate) inner: BayesOdpInner,
}

#[extendr]
impl BayesOdpFit {
    /// The volume-weighted chain ladder the model takes its latest values
    /// and scale from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
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

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed increments: `list(estimates, pointwise,
    /// pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
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

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// The Bayesian ODP chain ladder of every segment of a triangle column:
/// each segment's posterior and the joint reserves. The posterior of one
/// segment needs a single-segment fit.
#[extendr]
pub(crate) struct BayesOdpFits {
    pub(crate) inner: BayesOdpFitsInner,
}

impl BayesOdpFits {
    fn one(&self, field: &str, instead: &str) -> Result<&BayesOdpSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[extendr]
impl BayesOdpFits {
    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    fn scale(&self) -> Result<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
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

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self.one("draws", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("draws", "")?.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed increments, as
    /// `BayesOdpFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
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
    mod bayes_odp;
    impl BayesOdpFit;
    impl BayesOdpFits;
}
