//! The priors: `LayerSharePrior`, `PatternPrior` and `IndustryPrior`.

use super::*;

/// The prior of a layer's expected ultimate for ``BayesBfNormalised``, from
/// the ground-up expected loss ratio and the layer share, each with an
/// uncertainty (``docs/design/reserving-excess.md``, PR 4).
///
/// The layer's a-priori ultimate for origin ``i`` is the ground-up exposure
/// times ``apriori`` times the share. The normalised BF puts a gamma prior
/// on the layer's expected ultimate, centred there, whose coefficient of
/// variation is that of a product of two independent variables,
/// ``cv = sqrt((1 + apriori_cv**2) * (1 + share_cv**2) - 1)``: the moments
/// are exact. One share error is common to all origins (the default,
/// ``share_dependence="common"``): the model carries one more parameter, a
/// gamma multiplier ``s`` with mean 1 and cv ``share_cv``, sampled with the
/// others, and each origin's prior ultimate is a gamma with mean
/// ``exposure * apriori * share * s`` and cv ``apriori_cv``, so every origin
/// keeps the marginal cv above, the origins' priors are correlated and the
/// data update the share for every origin at once (``log_share`` in the
/// fit's parameters). ``share_dependence="per_origin"`` is the first
/// version: an independent share error at each origin, which understates
/// the total's interval. The layer's development pattern is estimated from
/// the layer's own triangle under the model's vague prior on the pattern;
/// an excess pattern from ``ExcessPattern`` does not enter (open decision
/// 22 of the design note).
///
/// Parameters
/// ----------
/// apriori : float
///     The ground-up expected ultimate loss ratio.
/// shares : float or list of float
///     The layer share (``layer_share`` of a severity, or an exposure
///     curve's ``layer_share``), one for all origins or one per origin,
///     each in ``(0, 1]``.
/// apriori_cv : float, default 0.1
///     The loss ratio's coefficient of variation (``inf`` is flat).
/// share_cv : float, default 0.0
///     The share's coefficient of variation, the same at every origin.
/// share_dependence : {"common", "per_origin"}, default "common"
///     Whether one share error is common to all origins or each origin has
///     its own. A known share (``share_cv`` 0) or a flat prior has nothing
///     to share and is the per-origin model; a common share error needs a
///     positive ``apriori_cv``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import LayerSharePrior
/// >>> prior = LayerSharePrior(0.65, 0.2, apriori_cv=0.1, share_cv=0.3)
/// >>> round(prior.layer_apriori(0), 12), round(prior.cv ** 2, 12)
/// (0.13, 0.1009)
/// >>> prior.share_dependence
/// 'common'
#[pyclass(name = "LayerSharePrior", module = "prospicio.reserving", frozen)]
pub(crate) struct PyLayerSharePrior {
    inner: LayerSharePrior,
}

fn share_dependence_arg(name: &str) -> PyResult<ShareDependence> {
    match name {
        "common" => Ok(ShareDependence::Common),
        "per_origin" => Ok(ShareDependence::PerOrigin),
        _ => Err(PyValueError::new_err(format!(
            "unknown share_dependence {name:?}; expected \"common\" or \"per_origin\""
        ))),
    }
}

fn shares_arg(shares: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(one) = shares.extract::<f64>() {
        return Ok(vec![one]);
    }
    shares.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "shares must be a number or a list of numbers, got {}",
            py_repr(shares)
        ))
    })
}

#[pymethods]
impl PyLayerSharePrior {
    #[new]
    #[pyo3(signature = (apriori, shares, apriori_cv = 0.1, share_cv = 0.0, share_dependence = "common"))]
    fn new(
        apriori: f64,
        shares: &Bound<'_, PyAny>,
        apriori_cv: f64,
        share_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LayerSharePrior::new(apriori, apriori_cv, &shares_arg(shares)?, share_cv)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// A prior from one share and its standard error: the share's
    /// coefficient of variation is ``share_se / share``.
    ///
    /// Parameters
    /// ----------
    /// apriori : float
    /// share : float
    /// share_se : float
    /// apriori_cv : float, default 0.1
    /// share_dependence : {"common", "per_origin"}, default "common"
    ///
    /// Returns
    /// -------
    /// LayerSharePrior
    #[staticmethod]
    #[pyo3(signature = (apriori, share, share_se, apriori_cv = 0.1, share_dependence = "common"))]
    fn from_share_se(
        apriori: f64,
        share: f64,
        share_se: f64,
        apriori_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LayerSharePrior::from_share_se(apriori, apriori_cv, share, share_se)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// A prior with the share of a severity in a layer (``layer_share``,
    /// ultimate basis) and its coefficient of variation.
    ///
    /// Parameters
    /// ----------
    /// apriori : float
    /// severity : severity distribution
    /// layer : ExcessLayer
    /// apriori_cv : float, default 0.1
    /// share_cv : float, default 0.0
    /// share_dependence : {"common", "per_origin"}, default "common"
    ///
    /// Returns
    /// -------
    /// LayerSharePrior
    #[staticmethod]
    #[pyo3(signature = (apriori, severity, layer, apriori_cv = 0.1, share_cv = 0.0, share_dependence = "common"))]
    fn from_severity(
        apriori: f64,
        severity: &Bound<'_, PyAny>,
        layer: &Bound<'_, PyAny>,
        apriori_cv: f64,
        share_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        let sev = crate::distributions::extract_severity(severity)?;
        let layer = crate::excess::layer_of(layer, "layer")?;
        Ok(Self {
            inner: LayerSharePrior::from_severity(apriori, apriori_cv, &sev, layer, share_cv)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// The ground-up expected ultimate loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// The loss ratio's coefficient of variation.
    #[getter]
    fn apriori_cv(&self) -> f64 {
        self.inner.apriori_cv
    }

    /// The layer shares, one or one per origin.
    #[getter]
    fn shares(&self) -> Vec<f64> {
        self.inner.shares.clone()
    }

    /// The share's coefficient of variation.
    #[getter]
    fn share_cv(&self) -> f64 {
        self.inner.share_cv
    }

    /// ``"common"`` or ``"per_origin"``: whether one share error is common
    /// to all origins.
    #[getter]
    fn share_dependence(&self) -> &'static str {
        match self.inner.dependence {
            ShareDependence::Common => "common",
            ShareDependence::PerOrigin => "per_origin",
        }
    }

    /// The marginal coefficient of variation of the prior on each origin's
    /// expected ultimate: ``sqrt((1 + apriori_cv**2) * (1 + share_cv**2) -
    /// 1)``, ``inf`` if either is. It is the gamma's cv under
    /// ``"per_origin"``; under ``"common"`` the model's own cv is
    /// ``apriori_cv`` and the multiplier's ``share_cv``.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv()
    }

    /// Whether the model carries a multiplier common to all origins: the
    /// share error is ``"common"`` and both cvs are positive and finite.
    #[getter]
    fn has_common_factor(&self) -> bool {
        self.inner.has_common_factor()
    }

    /// The mean and standard deviation of the prior of the layer's total
    /// expected ultimate, in closed form: ``c_e**2 * (1 + c_s**2) * sum(m**2)
    /// + c_s**2 * sum(m)**2`` for a common share error and ``((1 + c_e**2) *
    /// (1 + c_s**2) - 1) * sum(m**2)`` for independent ones, with ``m`` each
    /// origin's prior ultimate ``exposure * apriori * share``.
    ///
    /// Parameters
    /// ----------
    /// exposure : list of float
    ///     The ground-up exposure of each origin.
    ///
    /// Returns
    /// -------
    /// tuple of float
    ///     ``(mean, sd)``; ``sd`` is ``inf`` for a flat prior.
    fn prior_total(&self, exposure: Vec<f64>) -> PyResult<(f64, f64)> {
        self.inner.prior_total(&exposure).map_err(err)
    }

    /// The layer's a-priori loss ratio at origin ``i`` (0-based): the
    /// ground-up loss ratio times that origin's share.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///
    /// Returns
    /// -------
    /// float
    fn layer_apriori(&self, i: usize) -> PyResult<f64> {
        if self.inner.shares.len() > 1 && i >= self.inner.shares.len() {
            return Err(PyValueError::new_err(format!(
                "origin {i} is out of range for {} shares",
                self.inner.shares.len()
            )));
        }
        Ok(self.inner.layer_apriori(i))
    }

    /// ``model`` (default settings if omitted) with this prior's ground-up
    /// ``apriori`` and its ``cv``, for the triangle of ``triangle``.
    ///
    /// Parameters
    /// ----------
    /// model : BayesBfNormalised, optional
    ///     Supplies the sampler, process error, simulations, seed and
    ///     ``pattern_prior``; its own ``apriori``, ``cv`` and ``common_cv``
    ///     are replaced.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalised
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A common share error with ``apriori_cv`` 0.
    #[pyo3(signature = (model = None))]
    fn model(
        &self,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalised> {
        let pattern_prior = model.as_ref().and_then(|m| m.pattern_prior.clone());
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        Ok(PyBayesBfNormalised {
            inner: self.inner.model(&base).map_err(err)?,
            pattern_prior,
        })
    }

    /// ``triangle`` with a ``"layer_exposure"`` column: the ground-up
    /// exposure column times each origin's share.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// exposure : str
    ///     The ground-up exposure column.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn triangle(&self, triangle: PyRef<'_, PyTriangle>, exposure: &str) -> PyResult<PyTriangle> {
        Ok(PyTriangle {
            inner: self
                .inner
                .triangle(&triangle.inner, exposure)
                .map_err(err)?,
        })
    }

    /// Fits the layer losses ``column`` of a single-segment cumulative
    /// triangle that also holds the ground-up ``exposure`` column, with
    /// this prior: ``BayesBfNormalised.fit`` on the layer exposure. Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The layer losses.
    /// exposure : str
    ///     The ground-up exposure column.
    /// model : BayesBfNormalised, optional
    ///     The sampler and other settings; its ``apriori`` and ``cv`` are
    ///     replaced by the prior's.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``BayesBfNormalised.fit``; if the prior's ``cv`` is 0.
    #[pyo3(signature = (triangle, column, exposure, model = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let pattern = model.as_ref().and_then(|m| m.pattern_prior.clone());
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| match &pattern {
                Some(p) => prior.model(&base)?.fit_with_pattern_prior(
                    &prior.triangle(tri, exposure)?,
                    column,
                    prospicio_reserving::bayes::LAYER_EXPOSURE_COLUMN,
                    p,
                ),
                None => prior.fit(&base, tri, column, exposure),
            })
            .map_err(err)?;
        warn_diagnostics(py, "LayerSharePrior", &inner.diagnostics.warnings)?;
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of the same fit (``BayesBfNormalised.one_year``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// model : BayesBfNormalised, optional
    /// k_threshold : float, default 0.7
    /// refit : str, default "moment_match_then_refit"
    /// max_refits : int, optional
    /// n_sims : int, optional
    ///     As in ``BayesBfNormalised.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    #[pyo3(signature = (triangle, column, exposure, model = None, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let pattern = model.as_ref().and_then(|m| m.pattern_prior.clone());
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| match &pattern {
                Some(p) => prior.model(&base)?.one_year_with_pattern_prior(
                    &prior.triangle(tri, exposure)?,
                    column,
                    prospicio_reserving::bayes::LAYER_EXPOSURE_COLUMN,
                    &settings,
                    p,
                ),
                None => prior.one_year(&base, tri, column, exposure, &settings),
            })
            .map_err(err)?;
        warn_diagnostics(py, "LayerSharePrior", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "LayerSharePrior",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerSharePrior(apriori={:?}, shares={:?}, apriori_cv={:?}, share_cv={:?}, share_dependence={:?})",
            self.inner.apriori,
            self.inner.shares,
            self.inner.apriori_cv,
            self.inner.share_cv,
            self.share_dependence(),
        )
    }
}

/// A normal prior on the development pattern of ``BayesBfNormalised``: at
/// each age after the first, ``beta[d] = log(p[d] / p[1])`` (the log ratio
/// of the share of the ultimate at age ``d`` to the first age's) is normal
/// with mean ``mean[d]`` and sd ``sd[d]``, in place of the model's
/// ``N(0, development_sd**2)`` (``docs/design/reserving-bayes.md``,
/// decision 29). ``IndustryPrior.pattern`` builds one from other companies'
/// triangles.
///
/// Parameters
/// ----------
/// ages : list of int
///     The development ages in months, increasing; at least two.
/// mean : list of float
///     The prior mean of each ``beta[d]``, one fewer than the ages.
/// sd : list of float
///     The prior sd of each ``beta[d]``, finite and positive.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import PatternPrior
/// >>> prior = PatternPrior.from_shares([12, 24, 36], [0.5, 0.3, 0.2], [0.1, 0.1])
/// >>> [round(p, 12) for p in prior.shares()]
/// [0.5, 0.3, 0.2]
/// >>> prior.truncated(24).ages
/// [12, 24]
#[pyclass(name = "PatternPrior", module = "prospicio.reserving", frozen)]
pub(crate) struct PyPatternPrior {
    pub(crate) inner: PatternPrior,
}

#[pymethods]
impl PyPatternPrior {
    #[new]
    fn new(ages: Vec<Lag>, mean: Vec<f64>, sd: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PatternPrior::new(&ages, &mean, &sd).map_err(err)?,
        })
    }

    /// A prior centred on incremental ``shares`` of the ultimate (positive;
    /// they need not sum to 1), with ``sd`` on each log ratio after the
    /// first age.
    ///
    /// Parameters
    /// ----------
    /// ages : list of int
    /// shares : list of float
    ///     One per age.
    /// sd : list of float
    ///     One per age after the first.
    ///
    /// Returns
    /// -------
    /// PatternPrior
    #[staticmethod]
    fn from_shares(ages: Vec<Lag>, shares: Vec<f64>, sd: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PatternPrior::from_shares(&ages, &shares, &sd).map_err(err)?,
        })
    }

    /// The development ages in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// The prior mean of each ``beta[d]``, ages after the first.
    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean.clone()
    }

    /// The prior sd of each ``beta[d]``, ages after the first.
    #[getter]
    fn sd(&self) -> Vec<f64> {
        self.inner.sd.clone()
    }

    /// The incremental shares at the prior mean, summing to 1.
    ///
    /// Returns
    /// -------
    /// list of float
    fn shares(&self) -> Vec<f64> {
        self.inner.shares()
    }

    /// The percent developed at each age at the prior mean, ending at 1.
    ///
    /// Returns
    /// -------
    /// list of float
    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed()
    }

    /// The prior on the ages up to ``age`` (one of its ages, not the
    /// first): the log ratios to the first age keep their means and sds.
    ///
    /// Parameters
    /// ----------
    /// age : int
    ///
    /// Returns
    /// -------
    /// PatternPrior
    fn truncated(&self, age: Lag) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.truncated(age).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PatternPrior(ages={:?}, mean={:?}, sd={:?})",
            self.inner.ages, self.inner.mean, self.inner.sd
        )
    }
}

/// An industry prior for ``BayesBfNormalised`` from other companies'
/// triangles (``docs/design/reserving-bayes.md``, decision 29): the mean
/// ultimate loss ratio over the companies (``apriori``), the cv of a
/// company's level about it (``between_cv``, the model's ``common_cv``),
/// the cv of an origin about its company's level (``cv``, the model's
/// ``cv``) and a ``PatternPrior`` with the mean and sd over the companies
/// of each age's log share ratio. Each company's loss ratios and pattern
/// come from its own volume-weighted chain ladder (no tail, so the
/// triangles' last age is the ultimate); only origins developed at least
/// ``min_developed`` by the industry pattern count. The estimators are
/// moments on the natural scale (a one-way random-effects decomposition),
/// written out in the Rust docs of ``IndustryPrior``. No data ship with
/// the library: the triangles are the user's, for example Schedule P
/// triangles read with the ``cas-schedule-p`` package.
///
/// ``within_cv`` is the realised origin-to-origin spread and ``cv`` equals
/// it by default; ``subtract_process=True`` takes out the ODP process error
/// that each company's Pearson scale implies (``process_cv``), which the
/// model adds back through its own scale. On Schedule P that scale often
/// explains more than the whole spread (incurred above all), so the default
/// keeps the realised spread and errs wide.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import IndustryPrior, Triangle
/// >>> def company(scale):
/// ...     return Triangle.from_long(
/// ...         [2020, 2020, 2020, 2021, 2021, 2022],
/// ...         [12, 24, 36, 12, 24, 12],
/// ...         {"paid": [v * scale for v in [50.0, 80.0, 100.0, 55.0, 88.0, 60.0]]},
/// ...     )
/// >>> prior = IndustryPrior.from_triangles(
/// ...     [company(1.0), company(1.2), company(0.9)], "paid", [[160.0] * 3] * 3
/// ... )
/// >>> prior.report["n_companies_used"]
/// 3
/// >>> prior.pattern.ages
/// [12, 24, 36]
#[pyclass(name = "IndustryPrior", module = "prospicio.reserving", frozen)]
pub(crate) struct PyIndustryPrior {
    inner: IndustryPrior,
}

fn industry_settings(
    min_developed: f64,
    min_companies: usize,
    subtract_process: bool,
) -> IndustryPriorSettings {
    IndustryPriorSettings {
        min_developed,
        min_companies,
        subtract_process,
    }
}

#[pymethods]
impl PyIndustryPrior {
    /// The prior from single-segment triangles, one per company, of loss
    /// ``column`` (cumulative or incremental), with each company's premium
    /// per origin. Every triangle must have the same development ages.
    ///
    /// Parameters
    /// ----------
    /// triangles : list of Triangle
    ///     One per company; at least two.
    /// column : str
    ///     The losses.
    /// premiums : list of list of float
    ///     Each company's premium per origin, in origin order. An origin
    ///     whose premium is not positive is left out.
    /// min_developed : float, default 0.5
    ///     The smallest share of the ultimate developed (by the industry
    ///     pattern) at an origin's latest age for its loss ratio to count.
    /// min_companies : int, default 10
    ///     Fewer companies than this add a warning to ``report``.
    /// subtract_process : bool, default False
    ///     Whether ``cv`` is ``within_cv`` less the ODP process error.
    ///
    /// Returns
    /// -------
    /// IndustryPrior
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If fewer than two companies have a loss ratio, the triangles'
    ///     ages differ, a company's premiums do not match its origins, no
    ///     company has a pattern at some age, or ``min_developed`` is
    ///     outside 0 to 1.
    #[staticmethod]
    #[pyo3(signature = (triangles, column, premiums, min_developed = 0.5, min_companies = 10, subtract_process = false))]
    fn from_triangles(
        py: Python<'_>,
        triangles: Vec<PyRef<'_, PyTriangle>>,
        column: &str,
        premiums: Vec<Vec<f64>>,
        min_developed: f64,
        min_companies: usize,
        subtract_process: bool,
    ) -> PyResult<Self> {
        let triangles: Vec<Triangle> = triangles.iter().map(|t| t.inner.clone()).collect();
        let settings = industry_settings(min_developed, min_companies, subtract_process);
        let inner = py
            .detach(|| settings.from_triangles(&triangles, column, &premiums))
            .map_err(err)?;
        Ok(Self { inner })
    }

    /// The prior from one triangle with a segment per company (a key such
    /// as the group code), loss ``column``, and each origin's premium from
    /// the ``exposure`` column (its latest observed value).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// min_developed, min_companies, subtract_process
    ///     As in ``from_triangles``.
    ///
    /// Returns
    /// -------
    /// IndustryPrior
    #[staticmethod]
    #[pyo3(signature = (triangle, column, exposure, min_developed = 0.5, min_companies = 10, subtract_process = false))]
    fn from_segments(
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        min_developed: f64,
        min_companies: usize,
        subtract_process: bool,
    ) -> PyResult<Self> {
        let settings = industry_settings(min_developed, min_companies, subtract_process);
        let tri = &triangle.inner;
        let inner = py
            .detach(|| settings.from_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(Self { inner })
    }

    /// The mean over the companies of their mean ultimate loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// The standard error of ``apriori``.
    #[getter]
    fn apriori_se(&self) -> f64 {
        self.inner.apriori_se
    }

    /// The cv of a company's level about ``apriori``.
    #[getter]
    fn between_cv(&self) -> f64 {
        self.inner.between_cv
    }

    /// The cv of the realised loss ratios of a company's origins about its
    /// level.
    #[getter]
    fn within_cv(&self) -> f64 {
        self.inner.within_cv
    }

    /// The part of ``within_cv`` that the companies' ODP process error
    /// explains.
    #[getter]
    fn process_cv(&self) -> f64 {
        self.inner.process_cv
    }

    /// The model's ``cv``: ``within_cv``, or less the process error with
    /// ``subtract_process``; at least 0.01.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// The model's ``common_cv``: ``between_cv``, or ``None`` when it is 0.
    #[getter]
    fn common_cv(&self) -> Option<f64> {
        self.inner.common_cv()
    }

    /// The prior on the pattern.
    #[getter]
    fn pattern(&self) -> PyPatternPrior {
        PyPatternPrior {
            inner: self.inner.pattern.clone(),
        }
    }

    /// The percent developed at each age at the pattern prior's mean.
    #[getter]
    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed.clone()
    }

    /// The sd over the companies of their percent developed at each age.
    #[getter]
    fn percent_developed_sd(&self) -> Vec<f64> {
        self.inner.percent_developed_sd.clone()
    }

    /// Each company's mean ultimate loss ratio, companies left out omitted.
    #[getter]
    fn company_loss_ratios(&self) -> Vec<f64> {
        self.inner.company_loss_ratios.clone()
    }

    /// The factor from the pattern's last age to the ultimate: 1, or more
    /// for a prior cut by ``at_age``.
    #[getter]
    fn tail_factor(&self) -> f64 {
        self.inner.tail_factor
    }

    /// The report: ``n_companies``, ``n_companies_used``,
    /// ``n_origins_used``, ``origins_used`` (per company),
    /// ``companies_by_age`` (with a pattern at each age after the first),
    /// ``excluded`` and ``warnings`` (lists of str).
    #[getter]
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = &self.inner.report;
        let d = PyDict::new(py);
        d.set_item("n_companies", r.n_companies)?;
        d.set_item("n_companies_used", r.n_companies_used)?;
        d.set_item("n_origins_used", r.n_origins_used)?;
        d.set_item("origins_used", r.origins_used.clone())?;
        d.set_item("companies_by_age", r.companies_by_age.clone())?;
        d.set_item("excluded", r.excluded.clone())?;
        d.set_item("warnings", r.warnings.clone())?;
        Ok(d)
    }

    /// The prior for a triangle whose last age is ``age``: ``apriori``
    /// becomes the loss ratio developed to it, the pattern keeps the ages up
    /// to it and ``tail_factor`` is the factor from it to ultimate, which
    /// the user applies outside the model.
    ///
    /// Parameters
    /// ----------
    /// age : int
    ///
    /// Returns
    /// -------
    /// IndustryPrior
    fn at_age(&self, age: Lag) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.at_age(age).map_err(err)?,
        })
    }

    /// ``model`` (default settings if omitted) with this prior's
    /// ``apriori``, ``cv``, ``common_cv`` and ``pattern`` as its
    /// ``pattern_prior``; its other settings are kept.
    ///
    /// Parameters
    /// ----------
    /// model : BayesBfNormalised, optional
    ///
    /// Returns
    /// -------
    /// BayesBfNormalised
    #[pyo3(signature = (model = None))]
    fn model(&self, model: Option<PyRef<'_, PyBayesBfNormalised>>) -> PyBayesBfNormalised {
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        PyBayesBfNormalised {
            inner: self.inner.model(&base),
            pattern_prior: Some(self.inner.pattern.clone()),
        }
    }

    /// ``model()`` on the default settings.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalised
    fn to_bf_normalised_settings(&self) -> PyBayesBfNormalised {
        self.model(None)
    }

    /// ``self.model(model).fit(triangle, column, exposure)``: the pattern
    /// prior's ages must be the programs' ages (``at_age`` cuts it). Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// model : BayesBfNormalised, optional
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    #[pyo3(signature = (triangle, column, exposure, model = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| prior.fit(&base, tri, column, exposure))
            .map_err(err)?;
        warn_diagnostics(py, "IndustryPrior", &inner.diagnostics.warnings)?;
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of the same fit (``BayesBfNormalised.one_year``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// model : BayesBfNormalised, optional
    /// k_threshold : float, default 0.7
    /// refit : str, default "moment_match_then_refit"
    /// max_refits : int, optional
    /// n_sims : int, optional
    ///     As in ``BayesBfNormalised.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    #[pyo3(signature = (triangle, column, exposure, model = None, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| prior.one_year(&base, tri, column, exposure, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "IndustryPrior", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "IndustryPrior",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "IndustryPrior(apriori={:.4}, between_cv={:.4}, cv={:.4}, ages={:?}, companies={})",
            self.inner.apriori,
            self.inner.between_cv,
            self.inner.cv,
            self.inner.pattern.ages,
            self.inner.report.n_companies_used,
        )
    }
}
