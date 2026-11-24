//! The Bayesian models' one-year views: `BayesOneYearFit` and `BayesOneYearFits`.

use super::*;

/// The one-year view of a Bayesian model on a single-segment triangle
/// column: the lifetime fit, the opening reserve, the claims development
/// result and each simulation's Pareto k.
#[extendr]
pub(crate) struct BayesOneYearFit {
    pub(crate) inner: BayesOneYearOf,
}

/// The one-year view of each Bayesian model with one, by its lifetime fit.
#[derive(Clone)]
pub(crate) enum BayesOneYearOf {
    Odp(BayesOneYearInner<BayesOdpInner>),
    Bf(BayesOneYearInner<BayesBfInner>),
    BfNormalised(BayesOneYearInner<BayesBfNormalisedInner>),
    Meyers(Box<BayesOneYearInner<MeyersInner>>),
}

/// Evaluates `$body` with `$f` bound to the view, whatever its model.
macro_rules! each_bayes_one_year {
    ($view:expr, $f:ident => $body:expr) => {
        match $view {
            BayesOneYearOf::Odp($f) => $body,
            BayesOneYearOf::Bf($f) => $body,
            BayesOneYearOf::BfNormalised($f) => $body,
            BayesOneYearOf::Meyers($f) => $body,
        }
    };
}

#[extendr]
impl BayesOneYearFit {
    /// The model behind the view: "bayes_odp", "bayes_bf",
    /// "bayes_bf_normalised" or "meyers".
    fn model(&self) -> &'static str {
        match &self.inner {
            BayesOneYearOf::Odp(_) => "bayes_odp",
            BayesOneYearOf::Bf(_) => "bayes_bf",
            BayesOneYearOf::BfNormalised(_) => "bayes_bf_normalised",
            BayesOneYearOf::Meyers(_) => "meyers",
        }
    }

    /// The lifetime fit, as `bayes_odp` gives it; an error for another
    /// model.
    fn fit(&self) -> Result<BayesOdpFit> {
        match &self.inner {
            BayesOneYearOf::Odp(f) => Ok(BayesOdpFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_odp's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `bayes_bf` gives it; an error for another
    /// model.
    fn bf_fit(&self) -> Result<BayesBfFit> {
        match &self.inner {
            BayesOneYearOf::Bf(f) => Ok(BayesBfFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_bf's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `bayes_bf_normalised` gives it; an error for
    /// another model.
    fn bf_normalised_fit(&self) -> Result<BayesBfNormalisedFit> {
        match &self.inner {
            BayesOneYearOf::BfNormalised(f) => Ok(BayesBfNormalisedFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_bf_normalised's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `meyers` gives it; an error for another model.
    fn meyers_fit(&self) -> Result<MeyersFit> {
        match &self.inner {
            BayesOneYearOf::Meyers(f) => Ok(MeyersFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not meyers'",
                self.model()
            ))),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_ultimate.clone())
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_reserve.clone())
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_bayes_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.pareto_k.clone())
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    fn refit_warnings(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.refit_warnings as f64)
    }

    fn k_threshold(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.k_threshold)
    }

    fn flagged_share(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.flagged_share())
    }

    fn long_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.fit.development_table()))
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

/// The one-year view of a Bayesian model on every segment of a triangle
/// column: each segment's posterior, opening reserve and Pareto k, and the
/// joint claims development result.
#[extendr]
pub(crate) struct BayesOneYearFits {
    pub(crate) inner: BayesOneYearFitsOf,
}

/// The one-year view over segments of each Bayesian model with one, by
/// its segment fit.
pub(crate) enum BayesOneYearFitsOf {
    Odp(BayesOneYearFitsInner<BayesOdpSegment>),
    Meyers(BayesOneYearFitsInner<MeyersSegment>),
}

/// Evaluates `$body` with `$f` bound to the views over segments, whatever
/// the model.
macro_rules! each_bayes_one_years {
    ($views:expr, $f:ident => $body:expr) => {
        match $views {
            BayesOneYearFitsOf::Odp($f) => $body,
            BayesOneYearFitsOf::Meyers($f) => $body,
        }
    };
}

#[extendr]
impl BayesOneYearFits {
    /// The model behind the view: "bayes_odp" or "meyers".
    fn model(&self) -> &'static str {
        match &self.inner {
            BayesOneYearFitsOf::Odp(_) => "bayes_odp",
            BayesOneYearFitsOf::Meyers(_) => "meyers",
        }
    }

    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: each_bayes_one_years!(&self.inner, f => f
                .segments
                .map(|s| s.fit.chain_ladder.clone())),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_ultimate
            .clone()))
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_reserve
            .clone()))
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_bayes_one_years!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// The plugged-in scale of a single-segment `bayes_odp` view.
    fn scale(&self) -> Result<f64> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                Ok(single(&f.segments, "scale", "totals_frame()")?.fit.scale)
            }
            BayesOneYearFitsOf::Meyers(_) => Err(Error::Other(
                "scale is the ODP's plugged-in dispersion; Meyers' models have none".into(),
            )),
        }
    }

    /// The premium of each origin of a single-segment `meyers` view.
    fn premium(&self) -> Result<Vec<f64>> {
        match &self.inner {
            BayesOneYearFitsOf::Meyers(f) => Ok(single(&f.segments, "premium", "totals_frame()")?
                .fit
                .premium
                .clone()),
            BayesOneYearFitsOf::Odp(_) => Err(Error::Other(
                "premium is Meyers' input; bayes_odp has none".into(),
            )),
        }
    }

    fn pareto_k(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "pareto_k", "")?
            .pareto_k
            .clone()))
    }

    /// The simulations of a single-segment fit refitted with NUTS, 1-based.
    fn refitted(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refitted", "")?
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued, 1-based.
    fn moment_matched(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "moment_matched", "")?
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    fn unrefitted(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "unrefitted", "")?
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    fn refit_warnings(&self) -> Result<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refit_warnings", "")?
            .refit_warnings as f64))
    }

    /// Each segment's number of refits and of refits with a diagnostic
    /// warning, for the warning of the one-year views over segments.
    fn refit_counts(&self) -> List {
        each_bayes_one_years!(&self.inner, f => {
            let fits = &f.segments.fits;
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
        })
    }

    fn k_threshold(&self) -> f64 {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0].k_threshold)
    }

    fn flagged_share(&self) -> Result<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(
            &f.segments,
            "flagged_share",
            "totals_frame()"
        )?
        .flagged_share()))
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                segment_diagnostics(&f.segments, f.totals(), |s| &s.fit.diagnostics)
            }
            BayesOneYearFitsOf::Meyers(f) => {
                segment_diagnostics(&f.segments, f.totals(), |s| &s.fit.diagnostics)
            }
        }
    }

    fn long_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.segments.development_table()))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let keys = choice(&keys, &values)?;
        let inner = match &self.inner {
            BayesOneYearFitsOf::Odp(f) => BayesOneYearFitsOf::Odp(f.segment(&keys).map_err(to_r)?),
            BayesOneYearFitsOf::Meyers(f) => {
                BayesOneYearFitsOf::Meyers(f.segment(&keys).map_err(to_r)?)
            }
        };
        Ok(Self { inner })
    }
}

extendr_module! {
    mod bayes_one_year;
    impl BayesOneYearFit;
    impl BayesOneYearFits;
}
