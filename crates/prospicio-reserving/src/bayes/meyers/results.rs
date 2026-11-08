//! The fitted Meyers models: `MeyersFit`, `MeyersSegment` and `MeyersFits`, with their tables and predictives.

use super::*;

/// A fitted [`Meyers`].
#[derive(Debug, Clone)]
pub struct MeyersFit {
    /// The volume-weighted chain ladder of the fitted column: latest values
    /// and the long tables' `ultimate` and `reserve` come from it.
    pub chain_ladder: ChainLadderFit,
    /// The premium of each origin, the latest value of the premium column.
    pub premium: Vec<f64>,
    /// Posterior draws of `logelr`, `alpha[2]..alpha[n]`,
    /// `beta[1]..beta[m-1]`, `gamma` (CSR only) or `rho` (CAY only) and
    /// `sigma[1]..sigma[m]`,
    /// by chain.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by origin: dimension `origin`, one component per
    /// origin; an origin observed at the oldest age keeps its value.
    pub ultimates: PredictiveDistribution,
    /// The ultimates less the latest values, simulation by simulation:
    /// the same layout as [`OdpBootstrapFit::reserves`](crate::OdpBootstrapFit::reserves).
    pub reserves: PredictiveDistribution,
    /// The development pattern of the posterior draw each simulation took
    /// (the shares `exp(beta_d g)` of the value at the oldest age, `g` the
    /// next origin's speed-up in CSR and 1 otherwise), row `i` simulation
    /// `i` of `ultimates` and `reserves` (`reserving-v02.md` decision 28).
    pub patterns: DevelopmentPatterns,
    /// The model, for the pointwise log likelihood of [`loo`](Self::loo).
    pub(super) model: Model,
}

/// One segment of a [`MeyersFits`]: its chain ladder, premium, posterior
/// and diagnostics, as in [`MeyersFit`]; its ultimates and reserves are
/// its part of the joint distributions of [`MeyersFits`].
#[derive(Debug, Clone)]
pub struct MeyersSegment {
    /// The volume-weighted chain ladder of the segment's fitted column.
    pub chain_ladder: ChainLadderFit,
    /// The premium of each origin, the latest value of the premium column.
    pub premium: Vec<f64>,
    /// Posterior draws, named as in [`MeyersFit::posterior`].
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    pub(super) model: Model,
    /// The posterior draws on the unconstrained scale NUTS samples (an
    /// affine map of it), for the one-year view (decision 19); a
    /// [`MeyersFit`] does not keep them.
    pub(super) sampled: PosteriorDraws,
}

impl MeyersSegment {
    /// The model that produced this fit, CRC, CSR or CAY.
    pub fn model(&self) -> MeyersModel {
        self.model.kind
    }

    /// Posterior summary of each parameter, as [`MeyersFit::summary`].
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the segment's observed cells, as [`MeyersFit::loo`].
    pub fn loo(&self) -> Result<Loo> {
        loo(&self.model, &self.posterior)
    }
}

impl ReserveFit for MeyersSegment {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn exposure(&self) -> Option<&[f64]> {
        Some(&self.premium)
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}

/// A [`Meyers`] fit of every segment of a triangle column
/// ([`Meyers::fit_segments`]): each segment's own posterior, and the joint
/// distributions of the ultimates and of the reserves, with the layout of
/// [`OdpBootstrapFits`](crate::OdpBootstrapFits).
///
/// ```
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_reserving::bayes::Meyers;
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// // Five origins of cumulative paid with premium, and the same doubled.
/// let paid: [&[f64]; 5] = [
///     &[952.0, 1529.0, 2813.0, 3647.0, 3724.0],
///     &[849.0, 1564.0, 2202.0, 2432.0],
///     &[983.0, 2211.0, 2830.0],
///     &[1657.0, 2685.0],
///     &[932.0],
/// ];
/// let premium = [5812.0, 4908.0, 5454.0, 5165.0, 5214.0];
/// let (mut lob, mut origin, mut age, mut values, mut prem) = (vec![], vec![], vec![], vec![], vec![]);
/// for (line, scale) in [("Auto", 1.0), ("Home", 2.0)] {
///     for (w, row) in paid.iter().enumerate() {
///         for (d, &v) in row.iter().enumerate() {
///             lob.push(line);
///             origin.push(Month::january(1988 + w as i32));
///             age.push(12 * (d as u32 + 1));
///             values.push(v * scale);
///             prem.push(premium[w] * scale);
///         }
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &lob)],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &values), ("premium", &prem)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = Meyers {
///     sampler: Sampler { tune: 200, draws: 200, ..Meyers::default().sampler },
///     n_sims: 1_000,
///     ..Meyers::default()
/// };
/// let fits = model.fit_segments(&tri, "paid", "premium")?;
/// assert_eq!(fits.reserves.dims(), ["lob", "origin"]);
/// assert_eq!(fits.reserves.n_components(), 10);
/// assert_eq!(fits.ultimates.n_components(), 10);
/// assert_eq!(fits.segments.len(), 2);
/// assert_eq!(fits.totals().n_rows(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct MeyersFits {
    /// Each segment's chain ladder, premium, posterior and diagnostics.
    pub segments: SegmentFits<MeyersSegment>,
    /// Joint posterior predictive distribution of the ultimate by segment
    /// and origin: the key names and `origin` are its dimensions, and its
    /// components run over the origins of each segment in turn, like the
    /// rows of [`to_long`](Self::to_long).
    pub ultimates: PredictiveDistribution,
    /// The ultimates less the latest values, simulation by simulation,
    /// with the same layout.
    pub reserves: PredictiveDistribution,
    /// Each segment's development pattern in every simulation, moving with
    /// the segment's ultimates and reserves (`reserving-v02.md` decision
    /// 28), as in [`MeyersFit::patterns`].
    pub patterns: DevelopmentPatterns,
}

/// Appends `ulr_mean`, the row's posterior mean ultimate `ultimate_mean`
/// over its `premium`, to a Meyers table.
fn push_ulr_mean(table: &mut FitTable) {
    push_combined(table, "ulr_mean", "ultimate_mean", "premium", |u, p| u / p);
}

impl MeyersFits {
    /// One row per segment × origin: the chain ladder's `latest`,
    /// `ultimate` and `reserve`, the `premium`, `ulr` (the chain ladder's
    /// ultimate over the premium), the `mean` and `std_dev` of the
    /// posterior predictive reserve, the `ultimate_mean` and
    /// `ultimate_std_dev` of the predictive ultimate, and `ulr_mean`, the
    /// `ultimate_mean` over the premium.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.segments.to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        push_moments(&mut table, "ultimate_", &component_sums(&self.ultimates));
        push_ulr_mean(&mut table);
        table
    }

    /// One row per segment: the chain ladder's totals, the total
    /// `premium` and `ulr`, the `mean` and `std_dev` of the segment's total
    /// reserve, the `ultimate_mean` and `ultimate_std_dev` of its total
    /// ultimate, and `ulr_mean`, the total `ultimate_mean` over the total
    /// premium.
    pub fn totals(&self) -> FitTable {
        let mut table = self.segments.totals();
        push_moments(
            &mut table,
            "",
            &segment_sums(&self.segments, &self.reserves),
        );
        push_moments(
            &mut table,
            "ultimate_",
            &segment_sums(&self.segments, &self.ultimates),
        );
        push_ulr_mean(&mut table);
        table
    }

    /// The chain ladders' development factors, one row per segment × age.
    pub fn development_table(&self) -> FitTable {
        self.segments.development_table()
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its
    /// part of the joint ultimates and reserves (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let (segments, reserves) = pick_segment(&self.segments, &self.reserves, keys)?;
        let (_, ultimates) = pick_segment(&self.segments, &self.ultimates, keys)?;
        let patterns = self.patterns.only(&segments.labels[0]);
        Ok(Self {
            segments,
            ultimates,
            reserves,
            patterns,
        })
    }
}

impl MeyersSegment {
    /// The fit of a single segment, with its lifetime predictive.
    pub(super) fn into_fit(
        self,
        ultimates: PredictiveDistribution,
        reserves: PredictiveDistribution,
        patterns: DevelopmentPatterns,
    ) -> MeyersFit {
        let MeyersSegment {
            chain_ladder,
            premium,
            posterior,
            diagnostics,
            model,
            sampled: _,
        } = self;
        MeyersFit {
            chain_ladder,
            premium,
            posterior,
            diagnostics,
            ultimates,
            reserves,
            patterns,
            model,
        }
    }
}

impl MeyersFit {
    /// The model that produced this fit, CRC, CSR or CAY.
    pub fn model(&self) -> MeyersModel {
        self.model.kind
    }

    /// The posterior predictive of a new underwriting year with `premium`,
    /// the origin after the triangle's newest (`reserving-bayes.md`
    /// decision 28, issue #598), with the level drawn as
    /// [`NewOriginLevel::Exchangeable`]: see
    /// [`new_year_with_level`](Self::new_year_with_level).
    ///
    /// ```
    /// use prospicio_reserving::bayes::Meyers;
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_prob::Distribution;
    ///
    /// let paid: [&[f64]; 6] = [
    ///     &[952.0, 1529.0, 2813.0, 3647.0, 3724.0, 3832.0],
    ///     &[849.0, 1564.0, 2202.0, 2432.0, 2468.0],
    ///     &[983.0, 2211.0, 2830.0, 3832.0],
    ///     &[1657.0, 2685.0, 3169.0],
    ///     &[932.0, 1940.0],
    ///     &[1162.0],
    /// ];
    /// let (mut origin, mut age, mut value, mut premium) = (vec![], vec![], vec![], vec![]);
    /// for (w, row) in paid.iter().enumerate() {
    ///     for (d, &v) in row.iter().enumerate() {
    ///         origin.push(Month::january(2015 + w as i32));
    ///         age.push(12 * (d as u32 + 1));
    ///         value.push(v);
    ///         premium.push(5000.0);
    ///     }
    /// }
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&age),
    ///     values: &[("paid", &value), ("premium", &premium)],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let model = Meyers {
    ///     sampler: Sampler { tune: 300, draws: 300, ..Meyers::default().sampler },
    ///     n_sims: 500,
    ///     ..Meyers::default()
    /// };
    /// let fit = model.fit(&tri, "paid", "premium")?;
    /// let year = fit.new_year(5000.0, 2_000, 3)?;
    /// assert_eq!(year.origin.to_string(), "2021");
    /// assert!(year.loss_ratio.mean() > 0.0);
    /// // The share at the oldest age is 1 in every simulation.
    /// assert_eq!(year.patterns.segment(&[])?.draw(0)[5], 1.0);
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn new_year(&self, premium: f64, n_sims: usize, seed: u64) -> Result<NewYear> {
        self.new_year_with_level(premium, NewOriginLevel::Exchangeable, n_sims, seed)
    }

    /// The posterior predictive of a new underwriting year with `premium`,
    /// the origin `n + 1` after the triangle's newest: `n_sims` simulations
    /// from `seed`, each taking a posterior draw and from it the new
    /// origin's ultimate (its value at the oldest age `m`),
    ///
    /// ```text
    /// log C ~ N(log premium + l + rho r_nm, sigma_m^2)
    /// ```
    ///
    /// with `l` the new origin's level `logelr + alpha` drawn as `level`
    /// says, `sigma_m` the draw's, and in CAY `rho` times the newest
    /// origin's residual at age `m`, observed if it is closed and drawn
    /// with its ultimate otherwise (0 in CRC and CSR, where `beta_m = 0`
    /// also takes CSR's speed-up out of the ultimate). The loss ratio is
    /// `C / premium`, so it does not depend on `premium`, which only scales
    /// [`NewYear::ultimate`].
    ///
    /// The result's [`patterns`](NewYear::patterns) are the draws'
    /// development patterns as the fit's (the shares `exp(beta_d g)`, `g`
    /// the new origin's speed-up in CSR), so their
    /// [`monthly`](DevelopmentPatterns::monthly) is the new year's
    /// emergence by month.
    ///
    /// Errors: `premium` not finite and positive, `n_sims` 0, and fewer
    /// than three origins under [`NewOriginLevel::Exchangeable`].
    pub fn new_year_with_level(
        &self,
        premium: f64,
        level: NewOriginLevel,
        n_sims: usize,
        seed: u64,
    ) -> Result<NewYear> {
        new_year::validate(premium, n_sims)?;
        let model = &self.model;
        let n = model.n_origins;
        if level == NewOriginLevel::Exchangeable && n < 3 {
            return Err(Error::Bayes(format!(
                "an exchangeable new origin needs at least three origins to estimate their \
                 spread, and the triangle has {n}; use NewOriginLevel::Prior with an origin_sd \
                 set to the spread you believe"
            )));
        }
        let nd = model.n_dev;
        let closed: Vec<bool> = self
            .chain_ladder
            .latest_position
            .iter()
            .map(|&d| d + 1 == nd)
            .collect();
        let latest = &self.chain_ladder.latest;
        let premiums = &self.premium;
        let log_premium = premium.ln();
        let (origin_sd, df) = (model.origin_sd, n as f64 - 1.0);
        let inflation = (1.0 + 1.0 / n as f64).sqrt();
        let mut provenance = Provenance::new("meyers_new_year")
            .param("model", model.kind.name())
            .param("level", level.name())
            .param("origin_sd", origin_sd)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"));
        if let Some(hash) = &self.ultimates.provenance().input_hash {
            provenance = provenance.input_hash(hash.clone());
        }
        new_year::simulate(
            &self.posterior,
            &self.chain_ladder.origins,
            &self.chain_ladder.development.development,
            premium,
            n_sims,
            seed,
            provenance,
            |nat, rng, shares| {
                let logelr = nat[0];
                let new_level = match level {
                    NewOriginLevel::Prior => logelr + origin_sd * norm_quantile(rng.next_open01()),
                    NewOriginLevel::Exchangeable => {
                        let level_of = |w: usize| logelr + if w > 0 { nat[w] } else { 0.0 };
                        let mean = (0..n).map(level_of).sum::<f64>() / n as f64;
                        let var = (0..n).map(|w| (level_of(w) - mean).powi(2)).sum::<f64>() / df;
                        let z = norm_quantile(rng.next_open01());
                        let chi2 = new_year::gamma_draw(0.5 * df, 2.0, rng);
                        mean + var.sqrt() * inflation * z / (chi2 / df).sqrt()
                    }
                };
                let rho_term = if model.cay() {
                    // The newest origin's residual at age m, with the
                    // others' ultimates drawn along the way and discarded.
                    let mut row = vec![0.0; n];
                    let residual =
                        model.draw_ultimates(nat, premiums, &closed, latest, rng, &mut row);
                    nat[model.after_beta()] * residual
                } else {
                    0.0
                };
                let sigma_m = nat[model.dim() - 1];
                let z = norm_quantile(rng.next_open01());
                model.pattern(nat, shares);
                (log_premium + new_level + rho_term + sigma_m * z).exp()
            },
        )
    }

    /// Posterior summary of each parameter: mean, sd, 5/50/95%
    /// quantiles, R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the observed cells, with each cell's relative
    /// efficiency from the chains. The pointwise term is Meyers', the
    /// normal density of `log C_wd` (`normal_lpdf` in his Stan code), so
    /// the result is on the log scale and comparable only between CRC, CSR
    /// and CAY fits of the same column (Meyers 2019, Tables 6.4, 7.3 and
    /// 8.3). In CAY a cell's mean holds the previous origin's observed
    /// residual, so its term is conditional on that cell, as in Meyers'
    /// code.
    pub fn loo(&self) -> Result<Loo> {
        loo(&self.model, &self.posterior)
    }

    /// One row per origin: the chain ladder's `latest`, `ultimate` and
    /// `reserve`, the `premium`, `ulr` (the chain ladder's ultimate over
    /// the premium), the `mean` and `std_dev` of the posterior predictive
    /// reserve, the `ultimate_mean` and `ultimate_std_dev` of the
    /// predictive ultimate, and `ulr_mean`, the `ultimate_mean` over the
    /// premium.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.plug_in().to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        push_moments(&mut table, "ultimate_", &component_sums(&self.ultimates));
        push_ulr_mean(&mut table);
        table
    }

    /// One row: the chain ladder's totals, the total `premium` and `ulr`,
    /// the `mean` and `std_dev` of the posterior predictive total reserve,
    /// the `ultimate_mean` and `ultimate_std_dev` of the total ultimate,
    /// and `ulr_mean`, the total `ultimate_mean` over the total premium.
    pub fn totals(&self) -> FitTable {
        let fits = self.plug_in();
        let mut table = fits.totals();
        push_moments(&mut table, "", &segment_sums(&fits, &self.reserves));
        push_moments(
            &mut table,
            "ultimate_",
            &segment_sums(&fits, &self.ultimates),
        );
        push_ulr_mean(&mut table);
        table
    }

    /// The chain ladder's development factors, one row per age.
    pub fn development_table(&self) -> FitTable {
        self.plug_in().development_table()
    }

    /// The fit as the single segment of the long tables.
    fn plug_in(&self) -> SegmentFits<PlugIn> {
        SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![PlugIn {
                chain_ladder: self.chain_ladder.clone(),
                premium: self.premium.clone(),
            }],
        }
    }
}

impl ReserveFit for MeyersFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn exposure(&self) -> Option<&[f64]> {
        Some(&self.premium)
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}

/// The chain ladder and premium of a fit, for its long tables.
#[derive(Debug, Clone)]
struct PlugIn {
    chain_ladder: ChainLadderFit,
    premium: Vec<f64>,
}

impl ReserveFit for PlugIn {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn exposure(&self) -> Option<&[f64]> {
        Some(&self.premium)
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}
