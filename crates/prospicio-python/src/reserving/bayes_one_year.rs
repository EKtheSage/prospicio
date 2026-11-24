//! The Bayesian models' one-year views: `BayesOneYearFit` and `BayesOneYearFits`.

use super::*;

/// The settings of a Bayesian one-year view: `k_threshold`, the `refit`
/// policy by name (`None` for the model's one-year default), the refit
/// budget and the number of simulations (`None` for the model's
/// one-year default).
pub(crate) fn one_year_settings(
    k_threshold: f64,
    refit: Option<&str>,
    max_refits: Option<usize>,
    n_sims: Option<usize>,
) -> PyResult<BayesOneYear> {
    let refit = match refit {
        None => None,
        Some("refit_flagged") => Some(OneYearRefit::RefitFlagged),
        Some("reweight_only") => Some(OneYearRefit::ReweightOnly),
        Some("refit_all") => Some(OneYearRefit::RefitAll),
        Some("moment_match_then_refit") => Some(OneYearRefit::MomentMatchThenRefit),
        Some(refit) => {
            return Err(PyValueError::new_err(format!(
                "refit must be \"moment_match_then_refit\", \"refit_flagged\", \
                 \"reweight_only\" or \"refit_all\", got {refit:?}"
            )));
        }
    };
    Ok(BayesOneYear {
        k_threshold,
        refit,
        max_refits,
        n_sims,
        light_refit: None,
        no_light_refit: false,
    })
}

/// A compartmental or hierarchical one-year view's ``light_refit``:
/// ``None`` for the model's default (model 2's light refit), ``False`` to
/// turn it off, ``True`` for [`LightRefit::default`] (4 chains of 150
/// warm-up and 250 kept draws), or ``(chains, tune, draws)``. Returns the
/// sampler and whether the default is turned off.
pub(crate) fn light_refit_arg(
    light_refit: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<LightRefit>, bool)> {
    let Some(arg) = light_refit else {
        return Ok((None, false));
    };
    if arg.is_none() {
        return Ok((None, false));
    }
    if let Ok(flag) = arg.cast::<pyo3::types::PyBool>() {
        return Ok(if flag.is_true() {
            (Some(LightRefit::default()), false)
        } else {
            (None, true)
        });
    }
    let message = "light_refit must be None, a bool or a tuple (chains, tune, draws) of                    positive integers";
    let (chains, tune, draws): (usize, usize, usize) =
        arg.extract().map_err(|_| PyValueError::new_err(message))?;
    Ok((
        Some(LightRefit {
            chains,
            tune,
            draws,
        }),
        false,
    ))
}

/// One ``UserWarning`` from `model` if any NUTS refit of a one-year view
/// failed its diagnostics, and another if any simulation due for a refit
/// was left unrefitted by ``max_refits``: `refits` holds each segment's
/// label prefix, its number of refits, how many of them warned and how
/// many simulations were left unrefitted.
pub(crate) fn warn_refits(
    py: Python<'_>,
    model: &str,
    refits: &[(String, usize, usize, usize)],
) -> PyResult<()> {
    let warn = |message: String| -> PyResult<()> {
        let message =
            std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, category.as_any(), &message, 1)
    };
    let failed: Vec<String> = refits
        .iter()
        .filter(|(_, _, warned, _)| *warned > 0)
        .map(|(prefix, n, warned, _)| format!("{prefix}{warned} of {n} refits"))
        .collect();
    if !failed.is_empty() {
        warn(format!(
            "{model}: the MCMC diagnostics of NUTS refits in the one-year view failed (R-hat, \
             ESS or a divergence): {}",
            failed.join("; ")
        ))?;
    }
    let left: Vec<String> = refits
        .iter()
        .filter(|(_, _, _, left)| *left > 0)
        .map(|(prefix, _, _, left)| format!("{prefix}{left}"))
        .collect();
    if !left.is_empty() {
        warn(format!(
            "{model}: simulations due for a refit beyond max_refits were reweighted instead, \
             their reweighting as unreliable as their Pareto k (listed in unrefitted): {}",
            left.join("; ")
        ))?;
    }
    Ok(())
}

/// The one-year view of a Bayesian model on a single-segment triangle,
/// from ``BayesOdp.one_year``, ``BayesBf.one_year``,
/// ``BayesBfNormalised.one_year`` or ``Meyers.one_year``: the lifetime fit,
/// the opening reserve, the claims development result over the coming
/// year and how reliable each simulation's importance reweighting was.
///
/// ``cdr`` is the joint distribution of the claims development result by
/// origin (dimension ``"origin"``), the opening reserve less the year's
/// payment and the posterior mean reserve after the year, so a negative
/// value is an adverse development and ``cdr.quantile(0.005)`` is minus
/// the one-year value at risk at 99.5%. ``pareto_k`` has one Pareto ``k``
/// per simulation; ``flagged_share`` is the share above ``k_threshold``,
/// whether or not they were refitted, ``refitted`` lists the simulations
/// refitted with NUTS and ``moment_matched`` those moment matching
/// rescued (``refit="moment_match_then_refit"``, the default). ``fit`` is
/// the model's own lifetime fit, a ``BayesOdpFit``, a ``BayesBfFit``, a
/// ``BayesBfNormalisedFit`` or a ``MeyersFit``.
#[pyclass(name = "BayesOneYearFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOneYearFit {
    pub(crate) inner: BayesOneYearOf,
}

/// The one-year view of each Bayesian model with one, by its lifetime fit.
pub(crate) enum BayesOneYearOf {
    Odp(BayesOneYearFit<BayesOdpFit>),
    Bf(BayesOneYearFit<BayesBfFit>),
    BfNormalised(BayesOneYearFit<BayesBfNormalisedFit>),
    Meyers(Box<BayesOneYearFit<MeyersFit>>),
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

impl PyBayesOneYearFit {
    /// The lifetime fit's chain ladder.
    fn chain_ladder_fit(&self) -> &ChainLadderFit {
        each_bayes_one_year!(&self.inner, f => f.fit.chain_ladder())
    }
}

#[pymethods]
impl PyBayesOneYearFit {
    /// The model fitted to the observed triangle, with its posterior and
    /// lifetime reserves, as the model's ``fit`` gives it: a
    /// ``BayesOdpFit`` from ``BayesOdp.one_year``, a ``BayesBfFit`` from
    /// ``BayesBf.one_year``, a ``BayesBfNormalisedFit`` from
    /// ``BayesBfNormalised.one_year``, a ``MeyersFit`` from
    /// ``Meyers.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOdpFit, BayesBfFit, BayesBfNormalisedFit or MeyersFit
    #[getter]
    fn fit(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.inner {
            BayesOneYearOf::Odp(f) => Py::new(
                py,
                PyBayesOdpFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::Bf(f) => Py::new(
                py,
                PyBayesBfFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::BfNormalised(f) => Py::new(
                py,
                PyBayesBfNormalisedFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::Meyers(f) => Py::new(
                py,
                PyMeyersFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
        })
    }

    /// The model's volume-weighted chain ladder.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.chain_ladder_fit().clone()],
            },
        }
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.chain_ladder_fit()
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.chain_ladder_fit().development.development.clone()
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.chain_ladder_fit().latest.clone()
    }

    /// The latest value plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_ultimate.clone())
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// mean over the posterior draws of the expected future increments.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_reserve.clone())
    }

    /// Joint distribution of the claims development result by origin:
    /// dimension ``"origin"``, one component per origin, one row per
    /// simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_bayes_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// Each simulation's Pareto ``k``, in the order of the simulations
    /// (``inf`` when too few weights differ to fit the tail).
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.pareto_k.clone())
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.refitted.clone())
    }

    /// The simulations moment matching rescued (their ``k`` brought to at
    /// most ``k_threshold``), in order (0-based); empty unless ``refit``
    /// was ``"moment_match_then_refit"``, when each flagged simulation is
    /// in this, in ``refitted`` or in ``unrefitted``.
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.moment_matched.clone())
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead (as unreliable as their ``k`` says), in order (0-based);
    /// empty without ``max_refits``.
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.unrefitted.clone())
    }

    /// How many of the refits had a diagnostic warning (R-hat, ESS or a
    /// divergence).
    #[getter]
    fn refit_warnings(&self) -> usize {
        each_bayes_one_year!(&self.inner, f => f.refit_warnings)
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.k_threshold)
    }

    /// The share of simulations whose ``k`` is above ``k_threshold`` (or
    /// NaN), whether or not they were refitted.
    #[getter]
    fn flagged_share(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.flagged_share())
    }

    /// One row per origin: ``origin``, ``latest``, ``opening_ultimate``,
    /// ``opening_reserve``, the model's own columns (``BayesBf``'s
    /// ``exposure`` and ``prior_ultimate`` and its ``opening_ulr``, the
    /// opening ultimate over the exposure), and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_year!(&self.inner, f => f.to_long()))
    }

    /// One row: the totals of ``to_frame()``'s columns, the ``scale``, the
    /// ``flagged_share``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// total claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_year!(&self.inner, f => f.totals()))
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_bayes_one_year!(&self.inner, f => f.fit.development_table()),
        )
    }

    fn __repr__(&self) -> String {
        each_bayes_one_year!(&self.inner, f => format!(
            "BayesOneYearFit(origins={}, n_sims={}, flagged_share={:?}, refitted={})",
            f.opening_reserve.len(),
            f.cdr.n_sims(),
            f.flagged_share(),
            f.refitted.len(),
        ))
    }
}

/// The one-year view of a Bayesian model in every segment of a triangle,
/// from ``BayesOdp.one_year_segments`` or ``Meyers.one_year_segments``,
/// laid out as ``OneYearFit``.
///
/// ``cdr`` is one joint distribution of the claims development result with
/// the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``cdr``. ``scale`` (the
/// ODP's only), ``premium`` (Meyers' only), ``diagnostics``, ``pareto_k``,
/// ``refitted``, ``refit_warnings`` and ``flagged_share`` are one
/// segment's and need a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (the scales or premiums and the
/// flagged shares) or ``diagnostics_frame()``.
#[pyclass(name = "BayesOneYearFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOneYearFits {
    pub(crate) inner: BayesOneYearFitsOf,
}

/// The one-year view over segments of each Bayesian model with one, by
/// its segment fit.
pub(crate) enum BayesOneYearFitsOf {
    Odp(BayesOneYearFits<BayesOdpSegment>),
    Meyers(BayesOneYearFits<MeyersSegment>),
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

/// A segment's label prefix, its number of refits and how many of them
/// warned, for ``warn_refits``.
pub(crate) fn segment_refits<S>(
    segments: &SegmentFits<BayesOneYearSegment<S>>,
) -> Vec<(String, usize, usize, usize)> {
    segments
        .labels
        .iter()
        .zip(&segments.fits)
        .map(|(label, s)| {
            let prefix = if segments.key_names.is_empty() {
                String::new()
            } else {
                format!("{label}: ")
            };
            (
                prefix,
                s.refitted.len(),
                s.refit_warnings,
                s.unrefitted.len(),
            )
        })
        .collect()
}

#[pymethods]
impl PyBayesOneYearFits {
    /// The volume-weighted chain ladder of every segment.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: each_bayes_one_years!(&self.inner, f => f
                .segments
                .map(|s| s.fit.chain_ladder.clone())),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        each_bayes_one_years!(&self.inner, f => f.segments.key_names.clone())
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        each_bayes_one_years!(&self.inner, f => segment_labels(py, &f.segments))
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        each_bayes_one_years!(&self.inner, f => origin_labels(&f.segments))
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0]
            .fit
            .chain_ladder
            .development
            .development
            .clone())
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .fit
            .chain_ladder
            .latest
            .clone()))
    }

    /// The latest value plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_ultimate
            .clone()))
    }

    /// Each origin's posterior mean reserve on the observed triangle.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_reserve
            .clone()))
    }

    /// Joint distribution of the claims development result by segment and
    /// origin: the triangle's keys and ``"origin"`` are its dimensions, one
    /// component per segment and origin, one row per simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_bayes_one_years!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// The plugged-in scale ``phi`` of a single-segment ``BayesOdp`` view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For several segments, or for a model without a scale
    ///     (``Meyers``).
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                Ok(single(&f.segments, "scale", "totals_frame()")?.fit.scale)
            }
            BayesOneYearFitsOf::Meyers(_) => Err(PyValueError::new_err(
                "scale is the ODP's plugged-in dispersion; Meyers' models have none",
            )),
        }
    }

    /// The premium of each origin of a single-segment ``Meyers`` view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For several segments, or for a model without premium
    ///     (``BayesOdp``).
    #[getter]
    fn premium(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            BayesOneYearFitsOf::Meyers(f) => Ok(single(&f.segments, "premium", "totals_frame()")?
                .fit
                .premium
                .clone()),
            BayesOneYearFitsOf::Odp(_) => Err(PyValueError::new_err(
                "premium is Meyers' input; BayesOdp has none",
            )),
        }
    }

    /// MCMC diagnostics of a single-segment fit's posterior, as
    /// ``BayesOdpFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        each_bayes_one_years!(&self.inner, f => diagnostics_dict(
            py,
            &single(&f.segments, "diagnostics", "diagnostics_frame()")?
                .fit
                .diagnostics,
        ))
    }

    /// Each simulation's Pareto ``k`` in a single-segment fit.
    #[getter]
    fn pareto_k(&self) -> PyResult<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "pareto_k", "")?
            .pareto_k
            .clone()))
    }

    /// The simulations of a single-segment fit refitted with NUTS.
    #[getter]
    fn refitted(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refitted", "")?
            .refitted
            .clone()))
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued.
    #[getter]
    fn moment_matched(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "moment_matched", "")?
            .moment_matched
            .clone()))
    }

    /// The simulations of a single-segment fit due for a refit beyond
    /// ``max_refits``, reweighted instead.
    #[getter]
    fn unrefitted(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "unrefitted", "")?
            .unrefitted
            .clone()))
    }

    /// How many of a single-segment fit's refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> PyResult<usize> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refit_warnings", "")?
            .refit_warnings))
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0].k_threshold)
    }

    /// The share of a single-segment fit's simulations whose ``k`` is
    /// above ``k_threshold`` (or NaN).
    #[getter]
    fn flagged_share(&self) -> PyResult<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(
            &f.segments,
            "flagged_share",
            "totals_frame()"
        )?
        .flagged_share()))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` of the segment's posterior, as
    /// ``BayesOdpFits.diagnostics_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                segment_diagnostics_frame(py, &f.segments, |s| &s.fit.diagnostics)
            }
            BayesOneYearFitsOf::Meyers(f) => {
                segment_diagnostics_frame(py, &f.segments, |s| &s.fit.diagnostics)
            }
        }
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``opening_ultimate``, ``opening_reserve``, the model's
    /// own columns (Meyers' ``premium`` and ``opening_ulr``, the opening
    /// ultimate over the premium), and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_years!(&self.inner, f => f.to_long()))
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the model's own totals (the ODP's ``scale``, Meyers'
    /// ``premium``), the ``flagged_share``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the segment's total claims development result.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_years!(&self.inner, f => f.totals()))
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_bayes_one_years!(&self.inner, f => f.segments.development_table()),
        )
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let inner = match &self.inner {
            BayesOneYearFitsOf::Odp(f) => BayesOneYearFitsOf::Odp(f.segment(&keys).map_err(err)?),
            BayesOneYearFitsOf::Meyers(f) => {
                BayesOneYearFitsOf::Meyers(f.segment(&keys).map_err(err)?)
            }
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        each_bayes_one_years!(&self.inner, f => {
            let segments = &f.segments;
            let detail = match segments.fits.as_slice() {
                [one] => format!(
                    ", flagged_share={:?}, refitted={}",
                    one.flagged_share(),
                    one.refitted.len()
                ),
                _ => String::new(),
            };
            format!(
                "BayesOneYearFits({}origins={}, n_sims={}{detail})",
                segments_prefix(segments),
                origin_labels(segments).len(),
                f.cdr.n_sims(),
            )
        })
    }
}
