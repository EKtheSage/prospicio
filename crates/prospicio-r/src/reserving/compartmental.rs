//! Gesmann and Morris' compartmental model: its settings, fits, one-year views and backtest model.

use super::*;

/// The `CompartmentalPriors` of `priors` (31 numbers) and `families` (10
/// names): `k_er`, `k_p`, `rlr` and `rrf` as (median, sd); the half priors
/// on the effect sds of `RLR` and `RRF` as (nu, scale) for "student_t" or
/// (scale, NaN) for "normal"; `lkj_eta`; the priors on `log sigma_os` and
/// `log sigma_pd` as (nu, location, scale) for "student_t" or (mean, sd,
/// NaN) for "normal"; then the optional half priors on the origin effect
/// sds of `k_er` and `k_p` and on the development effect sds of `k_er`,
/// `k_p`, `rlr` and `rrf`, each two numbers with family "none" (NaN, NaN)
/// for no such effect.
fn compartmental_priors(priors: &[f64], families: &[String]) -> Result<CompartmentalPriors> {
    if priors.len() != 31 || families.len() != 10 {
        return Err(Error::Other(
            "priors must hold 31 numbers and 10 family names".into(),
        ));
    }
    let positive = |i: usize| PositivePrior::LogNormal {
        median: priors[i],
        sd: priors[i + 1],
    };
    let half = |family: &str, i: usize, arg: &str| -> Result<HalfPrior> {
        match family {
            "student_t" => Ok(HalfPrior::StudentT {
                nu: priors[i],
                scale: priors[i + 1],
            }),
            "normal" => Ok(HalfPrior::Normal { scale: priors[i] }),
            _ => Err(Error::Other(format!(
                "{arg} family must be \"student_t\" or \"normal\", got \"{family}\""
            ))),
        }
    };
    let real = |family: &str, i: usize, arg: &str| -> Result<RealPrior> {
        match family {
            "student_t" => Ok(RealPrior::StudentT {
                nu: priors[i],
                location: priors[i + 1],
                scale: priors[i + 2],
            }),
            "normal" => Ok(RealPrior::Normal {
                mean: priors[i],
                sd: priors[i + 1],
            }),
            _ => Err(Error::Other(format!(
                "{arg} family must be \"student_t\" or \"normal\", got \"{family}\""
            ))),
        }
    };
    let optional = |j: usize, arg: &str| -> Result<Option<HalfPrior>> {
        let family = &families[4 + j];
        if family == "none" {
            Ok(None)
        } else {
            half(family, 19 + 2 * j, arg).map(Some)
        }
    };
    Ok(CompartmentalPriors {
        k_er: positive(0),
        k_p: positive(2),
        rlr: positive(4),
        rrf: positive(6),
        rlr_effect_sd: half(&families[0], 8, "rlr_effect_sd")?,
        rrf_effect_sd: half(&families[1], 10, "rrf_effect_sd")?,
        lkj_eta: priors[12],
        log_sigma: [
            real(&families[2], 13, "log_sigma[[1]]")?,
            real(&families[3], 16, "log_sigma[[2]]")?,
        ],
        ker_effect_sd: optional(0, "ker_effect_sd")?,
        kp_effect_sd: optional(1, "kp_effect_sd")?,
        development_effect_sd: [
            optional(2, "development_effect_sd$k_er")?,
            optional(3, "development_effect_sd$k_p")?,
            optional(4, "development_effect_sd$rlr")?,
            optional(5, "development_effect_sd$rrf")?,
        ],
    })
}

/// The `Compartmental` of `compartmental` and
/// `compartmental_prior_predictive`.
pub(crate) fn compartmental_settings(
    priors: &[f64],
    families: &[String],
    settings: &[String],
    run: &[f64],
    dependence: SegmentDependence,
) -> Result<Compartmental> {
    let [
        likelihood,
        scale,
        paid_response,
        fitted,
        rate_order,
        development_levels,
        start,
    ] = settings
    else {
        return Err(Error::Other("settings must hold 7 names".into()));
    };
    let &[chains, tune, draws, n_sims, seed, target_accept, max_depth] = run else {
        return Err(Error::Other("run must hold 7 numbers".into()));
    };
    let bad = |arg: &str, value: &str, expected: &str| {
        Err(Error::Other(format!(
            "{arg} must be {expected}, got \"{value}\""
        )))
    };
    let likelihood = match likelihood.as_str() {
        "normal" => CompartmentalLikelihood::Normal,
        "lognormal" => CompartmentalLikelihood::LogNormal,
        v => return bad("likelihood", v, "\"normal\" or \"lognormal\""),
    };
    let scale = match scale.as_str() {
        "amount" => CompartmentalScale::Amount,
        "loss_ratio" => CompartmentalScale::LossRatio,
        v => return bad("scale", v, "\"amount\" or \"loss_ratio\""),
    };
    let paid_response = match paid_response.as_str() {
        "cumulative" => PaidResponse::Cumulative,
        "incremental" => PaidResponse::Incremental,
        v => return bad("paid_response", v, "\"cumulative\" or \"incremental\""),
    };
    let fitted = match fitted.as_str() {
        "paid_and_outstanding" => Fitted::PaidAndOutstanding,
        "paid_and_incurred" => Fitted::PaidAndIncurred,
        "paid_only" => Fitted::PaidOnly,
        "incurred_only" => Fitted::IncurredOnly,
        v => {
            return bad(
                "fitted",
                v,
                "\"paid_and_outstanding\", \"paid_and_incurred\", \"paid_only\" or \"incurred_only\"",
            );
        }
    };
    let rate_order = match rate_order.as_str() {
        "free" => RateOrder::Free,
        "ordered" => RateOrder::Ordered,
        v => return bad("rate_order", v, "\"free\" or \"ordered\""),
    };
    let development_levels = match development_levels.as_str() {
        "per_age" => DevelopmentLevels::PerAge,
        "by_year" => DevelopmentLevels::ByYear,
        v => return bad("development_levels", v, "\"per_age\" or \"by_year\""),
    };
    let start = match start.as_str() {
        "data" => CompartmentalStart::Data,
        "prior_centre" => CompartmentalStart::PriorCentre,
        v => return bad("start", v, "\"data\" or \"prior_centre\""),
    };
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    let model = Compartmental {
        priors: compartmental_priors(priors, families)?,
        likelihood,
        scale,
        paid_response,
        fitted,
        rate_order,
        development_levels,
        start,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept,
            max_depth: whole(max_depth, "max_depth")?,
        },
        n_sims,
        seed: whole(seed, "seed")?,
        dependence,
    };
    // Refuse a combination of settings the model does not support when the
    // settings are read, before any data or sampling (#580).
    model.check_supported().map_err(to_r)?;
    Ok(model)
}

/// The flat pieces of a `Backtest` that R builds its tables from.
pub(crate) fn backtest_list(cells: &TriangleFrame, bt: &Backtest) -> Result<List> {
    let (nm, ns) = (bt.models().len(), bt.n_splits());
    // The calendar period of each split, as the frame labels it (the
    // index when the frame has no calendar labels).
    let labels = match cells.frame().column("calendar") {
        Some(Column::Categorical(c)) => Some(c),
        _ => None,
    };
    let diagonals: Vec<String> = bt
        .scored_rows()
        .iter()
        .zip(bt.diagonals())
        .map(|(rows, &index)| match (labels, rows.first()) {
            (Some(c), Some(&r)) => c[r].clone(),
            _ => index.to_string(),
        })
        .collect();
    let mut scores = Vec::with_capacity(nm * bt.metrics().len() * ns);
    for m in 0..nm {
        for k in 0..bt.metrics().len() {
            scores.extend_from_slice(bt.split_scores(m, k));
        }
    }
    // One row per model and split, model-major; NaN where the model
    // has no sampler.
    let (mut rhat, mut bulk, mut tail, mut div, mut nwarn) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for m in 0..nm {
        for d in bt.diagnostics(m) {
            rhat.push(d.as_ref().map_or(f64::NAN, |d| d.max_rhat));
            bulk.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_bulk));
            tail.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_tail));
            div.push(d.as_ref().map_or(f64::NAN, |d| d.divergences as f64));
            nwarn.push(d.as_ref().map_or(f64::NAN, |d| d.warnings.len() as f64));
        }
    }
    let warnings = bt.warnings();
    let log_densities: Vec<Robj> = bt
        .log_densities()
        .iter()
        .map(|l| Nullable::from(l.clone()).into())
        .collect();
    Ok(list!(
        models = bt.models().to_vec(),
        metrics = bt
            .metrics()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        n_splits = ns as f64,
        diagonals = diagonals,
        scores = scores,
        excluded = bt.excluded().iter().map(|&e| e as f64).collect::<Vec<_>>(),
        scored_rows = List::from_values(
            bt.scored_rows()
                .iter()
                .map(|r| r.iter().map(|&x| x as f64).collect::<Vec<_>>())
        ),
        log_densities = List::from_values(log_densities),
        max_rhat = rhat,
        min_ess_bulk = bulk,
        min_ess_tail = tail,
        divergences = div,
        n_warnings = nwarn,
        warning_model = warnings.iter().map(|w| w.0 as f64).collect::<Vec<_>>(),
        warning_split = warnings.iter().map(|w| w.1 as f64).collect::<Vec<_>>(),
        warning_message = warnings.iter().map(|w| w.2.clone()).collect::<Vec<_>>()
    ))
}

/// Where a compartmental fit's outstanding comes from: the `outstanding`
/// column itself ("column") or incurred less paid ("incurred").
pub(crate) fn compartmental_outstanding<'a>(
    outstanding: &'a str,
    outstanding_from: &str,
) -> Result<Outstanding<'a>> {
    match outstanding_from {
        "column" => Ok(Outstanding::Column(outstanding)),
        "incurred" => Ok(Outstanding::FromIncurred(outstanding)),
        _ => Err(Error::Other(format!(
            "outstanding_from must be \"column\" or \"incurred\", got \"{outstanding_from}\""
        ))),
    }
}

/// A compartmental model as a candidate of the diagonal backtest: its
/// settings with the triangle and columns it reads. Put in the `models` of
/// `diagonal_backtest` (the backtest takes a clone that shares the fits the
/// handle keeps), its `outstanding_backtest` scores the held-out
/// outstanding on those fits.
#[extendr]
pub(crate) struct CompartmentalBacktestModel {
    pub(crate) inner: CompartmentalCandidate,
}

#[extendr]
impl CompartmentalBacktestModel {
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn paid(&self) -> String {
        self.inner.paid().to_string()
    }

    fn premium(&self) -> String {
        self.inner.premium().to_string()
    }

    /// The flat pieces of the outstanding `Backtest` (as
    /// `Triangle$diagonal_backtest` returns the paid one).
    fn outstanding_backtest(
        &self,
        n_diagonals: f64,
        n_sims: f64,
        seed: f64,
        interval: f64,
    ) -> Result<List> {
        let c = &self.inner;
        let cells = TriangleFrame::new(c.triangle(), c.paid(), None).map_err(to_r)?;
        let bt = c
            .outstanding_backtest(
                &cells,
                whole(n_diagonals, "n_diagonals")? as usize,
                whole(n_sims, "n_sims")? as usize,
                whole(seed, "seed")?,
                interval,
            )
            .map_err(to_r)?;
        backtest_list(&cells, &bt)
    }
}

/// The one-year view of a compartmental model: the lifetime fit, the
/// opening reserve, the claims development result and each simulation's
/// Pareto k, by the premium column's origins.
#[extendr]
pub(crate) struct CompartmentalOneYearFit {
    pub(crate) inner: BayesOneYearInner<CompartmentalInner>,
}

#[extendr]
impl CompartmentalOneYearFit {
    fn fit(&self) -> CompartmentalFit {
        CompartmentalFit {
            inner: self.inner.fit.clone(),
        }
    }

    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner.fit.latest_paid.clone()
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        self.inner
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        self.inner
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        self.inner
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    fn refit_warnings(&self) -> f64 {
        self.inner.refit_warnings as f64
    }

    /// Why the view is approximate (one string), or none: set when `refit`
    /// was left to a model with development effects whose default
    /// light refit was turned off (`light_refit = FALSE`), so that it
    /// reweights the flagged simulations only.
    fn approximate(&self) -> Vec<String> {
        self.inner.approximate.iter().cloned().collect()
    }

    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    /// The view itself, as for any single-segment fit without keys: a
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

/// The one-year view of a compartmental model on every segment of a
/// triangle: each segment's posterior, opening reserve and Pareto k, and
/// the joint claims development result. The posterior of one segment needs
/// a single-segment fit.
#[extendr]
pub(crate) struct CompartmentalOneYearFits {
    pub(crate) inner: CompartmentalOneYearFitsInner,
}

impl CompartmentalOneYearFits {
    fn one(
        &self,
        field: &str,
        instead: &str,
    ) -> Result<&BayesOneYearSegment<CompartmentalSegment>> {
        single(&self.inner.segments, field, instead)
    }
}

#[extendr]
impl CompartmentalOneYearFits {
    fn n_segments(&self) -> i32 {
        self.inner.segments.len() as i32
    }

    /// Origins of each segment's premium column, in turn.
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    fn premium(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.premium.clone())
    }

    fn latest_paid(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.latest_paid.clone())
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_ultimate.clone())
    }

    fn opening_reserve(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_reserve.clone())
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Why the views are approximate (one string), or none: set when
    /// `refit` was left to a model with development effects whose default
    /// light refit was turned off (`light_refit = FALSE`), so that it
    /// reweights the flagged simulations only.
    fn approximate(&self) -> Vec<String> {
        self.inner.approximate.iter().cloned().collect()
    }

    fn pareto_k(&self) -> Result<Vec<f64>> {
        Ok(self.one("pareto_k", "")?.pareto_k.clone())
    }

    /// The simulations of a single-segment fit refitted with NUTS, 1-based.
    fn refitted(&self) -> Result<Vec<f64>> {
        Ok(self
            .one("refitted", "")?
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued, 1-based.
    fn moment_matched(&self) -> Result<Vec<f64>> {
        Ok(self
            .one("moment_matched", "")?
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    fn unrefitted(&self) -> Result<Vec<f64>> {
        Ok(self
            .one("unrefitted", "")?
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    fn refit_warnings(&self) -> Result<f64> {
        Ok(self.one("refit_warnings", "")?.refit_warnings as f64)
    }

    /// Each segment's number of refits and of refits with a diagnostic
    /// warning, for the warning of the one-year views over segments.
    fn refit_counts(&self) -> List {
        let fits = &self.inner.segments.fits;
        list!(
            refits = fits
                .iter()
                .map(|s| s.refitted.len() as f64)
                .collect::<Vec<_>>(),
            warned = fits
                .iter()
                .map(|s| s.refit_warnings as f64)
                .collect::<Vec<_>>(),
            unrefitted = fits
                .iter()
                .map(|s| s.unrefitted.len() as f64)
                .collect::<Vec<_>>()
        )
    }

    fn k_threshold(&self) -> f64 {
        self.inner.segments.fits[0].k_threshold
    }

    fn flagged_share(&self) -> Result<f64> {
        Ok(self.one("flagged_share", "totals_frame()")?.flagged_share())
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.fit.fit.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.fit.fit.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("chains", "")?.fit.fit.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.fit.fit.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// `CompartmentalFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.fit.fit.loo().map_err(to_r)?;
        let e = &l.loo.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.loo.pareto_k,
            k_threshold = l.loo.k_threshold,
            n_outstanding = l.n_outstanding as f64,
            elpd_outstanding = l.elpd_outstanding,
            elpd_paid = l.elpd_paid
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// A fitted Gesmann and Morris compartmental model of every segment of a
/// triangle: each segment's posterior and the joint reserves and future
/// paid and outstanding cells. The posterior of one segment needs a
/// single-segment fit.
#[extendr]
pub(crate) struct CompartmentalFits {
    pub(crate) inner: CompartmentalFitsInner,
}

impl CompartmentalFits {
    fn one(&self, field: &str, instead: &str) -> Result<&CompartmentalInner> {
        single(&self.inner.segments, field, instead).map(|s| &s.fit)
    }
}

#[extendr]
impl CompartmentalFits {
    fn n_segments(&self) -> i32 {
        self.inner.segments.len() as i32
    }

    /// Origins of each segment's premium column, in turn.
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.premium.iter().copied())
            .collect()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.latest_paid.iter().copied())
            .collect()
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

    fn paid(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// The future outstanding (or incurred) cells, or `NULL` for a fit of
    /// paid alone.
    fn outstanding(&self) -> Robj {
        match &self.inner.outstanding {
            Some(inner) => PredictiveDistribution {
                inner: inner.clone(),
            }
            .into(),
            None => ().into(),
        }
    }

    /// Posterior draws with `k_er <= k_p` in each segment, in index order.
    fn rate_swaps(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .map(|s| s.fit.rate_swaps as f64)
            .collect()
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.fit.diagnostics
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

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// `CompartmentalFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.loo().map_err(to_r)?;
        let e = &l.loo.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.loo.pareto_k,
            k_threshold = l.loo.k_threshold,
            n_outstanding = l.n_outstanding as f64,
            elpd_outstanding = l.elpd_outstanding,
            elpd_paid = l.elpd_paid
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn paid_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.paid_table().map_err(to_r)?))
    }

    fn outstanding_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.outstanding_table().map_err(to_r)?))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// A fitted Gesmann and Morris compartmental model of a single-segment triangle.
#[extendr]
pub(crate) struct CompartmentalFit {
    pub(crate) inner: CompartmentalInner,
}

#[extendr]
impl CompartmentalFit {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    fn population_origins(&self) -> Vec<String> {
        self.inner
            .population_origins
            .iter()
            .map(ToString::to_string)
            .collect()
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

    fn paid(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// The future outstanding (or incurred) cells, or `NULL` for a fit of
    /// paid alone.
    fn outstanding(&self) -> Robj {
        match &self.inner.outstanding {
            Some(inner) => PredictiveDistribution {
                inner: inner.clone(),
            }
            .into(),
            None => ().into(),
        }
    }

    fn ultimate_loss_ratio(&self) -> Vec<f64> {
        self.inner.ultimate_loss_ratio.clone()
    }

    fn rate_swaps(&self) -> f64 {
        self.inner.rate_swaps as f64
    }

    /// The share of draws and observed cells with `k_er <= k_p` when the
    /// rates vary by cell; NA when they are shared.
    fn cell_rate_swaps(&self) -> Rfloat {
        self.inner
            .cell_rate_swaps
            .map_or(Rfloat::na(), Rfloat::from)
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

    /// PSIS-LOO of the observed cells, outstanding first: `list(estimates,
    /// pointwise, pareto_k, k_threshold, n_outstanding, elpd_outstanding,
    /// elpd_paid)`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.loo.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.loo.pareto_k,
            k_threshold = l.loo.k_threshold,
            n_outstanding = l.n_outstanding as f64,
            elpd_outstanding = l.elpd_outstanding,
            elpd_paid = l.elpd_paid
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn paid_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.paid_table().map_err(to_r)?))
    }

    fn outstanding_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.outstanding_table().map_err(to_r)?))
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

extendr_module! {
    mod compartmental;
    impl CompartmentalFit;
    impl CompartmentalBacktestModel;
    impl CompartmentalFits;
    impl CompartmentalOneYearFit;
    impl CompartmentalOneYearFits;
}
