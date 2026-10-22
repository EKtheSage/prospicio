//! `prospicio.reserving` excess-layer development (issue #192,
//! `docs/design/reserving-excess.md`, PR 3): a layer's development pattern
//! derived from the ground-up triangle, over `prospicio_reserving::excess`.
//!
//! The derived pattern goes into the chain ladder, Mack and the
//! expected-loss methods through their `ldf`, `ldf_kind` and `ldf_weight`
//! arguments (`ExcessPattern.selected`). Exposure curves reach the reserving
//! crate only here (decision 2): a curve's `layer_share` is passed to
//! `layer_exposure` as a share.

use prospicio_core::Lag;
use prospicio_prob::SeverityDist;
use prospicio_reserving::PatternWeight;
use prospicio_reserving::excess::{
    Basis, ExcessPattern, Layer, SeverityByMaturity, ShapeBase, layer_at_shift as at_shift,
    layer_exposure as exposure_core, layer_share as share,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString};

use crate::distributions::extract_severity;
use crate::reserving::{PyTriangle, age_values, py_repr, tail_arg};

fn err(e: prospicio_reserving::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// An excess layer ``limit xs attachment``, the same for every origin.
///
/// Parameters
/// ----------
/// attachment : float
///     Where the layer starts: finite and not negative (0 for a layer from
///     the ground up).
/// limit : float, default inf
///     The layer's width: positive, or infinite for an unlimited layer
///     (``attachment`` and above).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ExcessLayer
/// >>> ExcessLayer(350.0)
/// ExcessLayer(attachment=350.0, limit=inf)
/// >>> ExcessLayer(500.0, 500.0).exhaustion
/// 1000.0
#[pyclass(name = "ExcessLayer", module = "prospicio.reserving", frozen)]
pub(crate) struct PyExcessLayer {
    inner: Layer,
}

#[pymethods]
impl PyExcessLayer {
    #[new]
    #[pyo3(signature = (attachment, limit = f64::INFINITY))]
    fn new(attachment: f64, limit: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Layer::new(attachment, limit).map_err(err)?,
        })
    }

    /// Where the layer starts.
    #[getter]
    fn attachment(&self) -> f64 {
        self.inner.attachment
    }

    /// The layer's width; infinite for an unlimited layer.
    #[getter]
    fn limit(&self) -> f64 {
        self.inner.limit
    }

    /// The top of the layer, ``attachment + limit``.
    #[getter]
    fn exhaustion(&self) -> f64 {
        self.inner.exhaustion()
    }

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.attachment, self.inner.limit)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyExcessLayer>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "ExcessLayer(attachment={:?}, limit={:?})",
            self.inner.attachment, self.inner.limit
        )
    }
}

fn layer_of(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Layer> {
    obj.extract::<PyRef<'_, PyExcessLayer>>()
        .map(|l| l.inner)
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{what} must be an ExcessLayer, got {}",
                py_repr(obj)
            ))
        })
}

/// The scale shift of the severity by maturity (decision 1, the default):
/// the severity of a claim at each development age is the ultimate
/// severity divided by a shift ``c_j`` solved from the ground-up triangle,
/// ``X_ultimate = c_j X_j``.
///
/// Parameters
/// ----------
/// ultimate : severity distribution
///     The ultimate severity at the price level of the layer's origins,
///     incurred basis (trended by the caller). Any severity of
///     ``prospicio.distributions`` with exact limited moments.
/// ground_up_cap : float, optional
///     The per-claim limit of a capped ground-up column (as in a triangle
///     limited at 1 million); ``None`` for an unlimited column.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Lognormal
/// >>> from prospicio.reserving import ScaleShift
/// >>> ScaleShift(Lognormal(8.0, 1.0)).ground_up_cap is None
/// True
#[pyclass(name = "ScaleShift", module = "prospicio.reserving", frozen)]
pub(crate) struct PyScaleShift {
    ultimate: Py<PyAny>,
    severity: SeverityDist,
    ground_up_cap: Option<f64>,
}

#[pymethods]
impl PyScaleShift {
    #[new]
    #[pyo3(signature = (ultimate, ground_up_cap = None))]
    fn new(ultimate: &Bound<'_, PyAny>, ground_up_cap: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            severity: extract_severity(ultimate)?,
            ultimate: ultimate.clone().unbind(),
            ground_up_cap,
        })
    }

    /// The ultimate severity.
    #[getter]
    fn ultimate(&self, py: Python<'_>) -> Py<PyAny> {
        self.ultimate.clone_ref(py)
    }

    /// The per-claim limit of a capped ground-up column, or ``None``.
    #[getter]
    fn ground_up_cap(&self) -> Option<f64> {
        self.ground_up_cap
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "ScaleShift(ultimate={}, ground_up_cap={:?})",
            py_repr(self.ultimate.bind(py)),
            self.ground_up_cap
        )
    }
}

/// Pinto and Gogol's Pareto shape drift above a threshold: the claims above
/// ``threshold`` are Pareto at every development age, with a shape that moves
/// with maturity, and the layer's development follows from a base
/// development the triangle column holds (Pinto and Gogol, PCAS 74 (1987);
/// Bear, PCAS 79 (1992)).
///
/// Parameters
/// ----------
/// threshold : float
///     The threshold above which claims are Pareto.
/// alphas : dict of int to float
///     The Pareto shape at every development age of the column, by age in
///     months (none is interpolated).
/// ultimate_alpha : float
///     The shape of the fully developed claims.
/// base : {"counts"} or ExcessLayer, default "counts"
///     What the triangle column holds: ``"counts"`` for the number of
///     claims above the threshold, or an ``ExcessLayer`` (attaching at or
///     above the threshold) for the losses in a base layer, such as
///     ``ExcessLayer(threshold)`` for the losses excess of the threshold.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ParetoShape
/// >>> ParetoShape(100.0, {12: 1.0, 24: 1.5}, 2.0).base
/// 'counts'
#[pyclass(name = "ParetoShape", module = "prospicio.reserving", frozen)]
pub(crate) struct PyParetoShape {
    threshold: f64,
    alphas: Vec<(Lag, f64)>,
    ultimate_alpha: f64,
    base: ShapeBase,
}

#[pymethods]
impl PyParetoShape {
    #[new]
    #[pyo3(signature = (threshold, alphas, ultimate_alpha, base = None))]
    fn new(
        threshold: f64,
        alphas: &Bound<'_, PyAny>,
        ultimate_alpha: f64,
        base: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let base = match base.filter(|b| !b.is_none()) {
            None => ShapeBase::Counts,
            Some(b) if b.is_instance_of::<PyString>() => {
                if b.extract::<String>()? == "counts" {
                    ShapeBase::Counts
                } else {
                    return Err(PyValueError::new_err(format!(
                        "base must be \"counts\" or an ExcessLayer, got {}",
                        py_repr(b)
                    )));
                }
            }
            Some(b) => ShapeBase::Losses(layer_of(b, "base")?),
        };
        Ok(Self {
            threshold,
            alphas: age_values(alphas, "alphas")?,
            ultimate_alpha,
            base,
        })
    }

    /// The threshold above which claims are Pareto.
    #[getter]
    fn threshold(&self) -> f64 {
        self.threshold
    }

    /// The shapes by age in months.
    #[getter]
    fn alphas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for &(age, a) in &self.alphas {
            d.set_item(age, a)?;
        }
        Ok(d)
    }

    /// The shape of the fully developed claims.
    #[getter]
    fn ultimate_alpha(&self) -> f64 {
        self.ultimate_alpha
    }

    /// ``"counts"``, or the ``ExcessLayer`` of the base losses.
    #[getter]
    fn base(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.base {
            ShapeBase::Counts => Ok("counts".into_pyobject(py)?.into_any().unbind()),
            ShapeBase::Losses(inner) => {
                Ok(Bound::new(py, PyExcessLayer { inner })?.into_any().unbind())
            }
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let base = self.base(py)?;
        Ok(format!(
            "ParetoShape(threshold={:?}, alphas={:?}, ultimate_alpha={:?}, base={})",
            self.threshold,
            self.alphas,
            self.ultimate_alpha,
            py_repr(base.bind(py))
        ))
    }
}

fn model_of(obj: &Bound<'_, PyAny>) -> PyResult<SeverityByMaturity> {
    if let Ok(m) = obj.extract::<PyRef<'_, PyScaleShift>>() {
        return Ok(match m.ground_up_cap {
            None => SeverityByMaturity::scale_shift(Box::new(m.severity.clone())),
            Some(cap) => SeverityByMaturity::scale_shift_capped(Box::new(m.severity.clone()), cap),
        });
    }
    if let Ok(m) = obj.extract::<PyRef<'_, PyParetoShape>>() {
        return Ok(SeverityByMaturity::pareto_shape(
            m.threshold,
            m.alphas.clone(),
            m.ultimate_alpha,
            m.base,
        ));
    }
    Err(PyTypeError::new_err(format!(
        "model must be a ScaleShift or a ParetoShape, got {}",
        py_repr(obj)
    )))
}

/// A weight on the derived pattern as the methods' ``ldf_weight`` takes
/// it: a number, or a dict from age in months to a number.
fn weight_of(obj: &Bound<'_, PyAny>) -> PyResult<PatternWeight> {
    if obj.hasattr("items")? {
        return Ok(PatternWeight::ByAge(age_values(obj, "weight")?));
    }
    if !obj.is_instance_of::<PyBool>() {
        if let Ok(w) = obj.extract::<f64>() {
            return Ok(PatternWeight::All(w));
        }
    }
    Err(PyTypeError::new_err(format!(
        "weight must be a number or a dict from age in months to a number, got {}",
        py_repr(obj)
    )))
}

fn weight_object(py: Python<'_>, w: &PatternWeight) -> PyResult<Py<PyAny>> {
    match w {
        PatternWeight::All(x) => Ok(x.into_pyobject(py)?.into_any().unbind()),
        PatternWeight::ByAge(by_age) => {
            let d = PyDict::new(py);
            for &(age, x) in by_age {
                d.set_item(age, x)?;
            }
            Ok(d.into_any().unbind())
        }
    }
}

/// A layer's development pattern derived from the ground-up triangle:
/// the layer's age-to-age factors, its factors to ultimate and its tail,
/// from the ground-up incurred factors and a severity by maturity
/// (Bloom and Patrik, CAS E-Forum Fall 2013; Sahasrabuddhe, E-Forum Fall
/// 2010; Pinto and Gogol, PCAS 74 (1987)). Build one with
/// ``ExcessPattern.from_ground_up``.
///
/// The pattern goes into ``ChainLadder``, ``Mack`` and the expected-loss
/// methods of a *layer* triangle through ``selected``, which gives their
/// ``ldf``, ``ldf_kind`` and ``ldf_weight`` arguments, and ``tail``:
/// ``ChainLadder(tail=p.tail, **p.selected(weight))``.
///
/// Examples
/// --------
/// Bloom and Patrik's four claims (100, 375, 250 and 500 at 12 months)
/// developing 3.60 and 1.25 to ultimate, in the layer excess of 350.
///
/// >>> from prospicio.distributions import Grid
/// >>> from prospicio.reserving import ExcessLayer, ExcessPattern, ScaleShift, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020], [12, 24, 36],
/// ...     {"incurred": [1225.0, 4410.0, 5512.5]},
/// ... )
/// >>> probs = [0.0] * 21
/// >>> for j in (4, 15, 10, 20):
/// ...     probs[j] = 0.25
/// >>> ultimate = Grid(112.5, probs)
/// >>> p = ExcessPattern.from_ground_up(
/// ...     tri, "incurred", ExcessLayer(350.0), ScaleShift(ultimate)
/// ... )
/// >>> round(p.layer_ldf[0], 6), round(p.layer_cdf[0], 6)
/// (17.2, 23.5)
#[pyclass(name = "ExcessPattern", module = "prospicio.reserving", frozen)]
pub(crate) struct PyExcessPattern {
    inner: ExcessPattern,
}

#[pymethods]
impl PyExcessPattern {
    /// The layer pattern from the ground-up incurred column, an optional
    /// claim count column and the ground-up tail.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     A single-segment triangle holding the ground-up column.
    /// column : str
    ///     The ground-up incurred (reported) loss column. Under a
    ///     ``ParetoShape`` it is the base development the model names (its
    ///     counts above the threshold, or its base losses).
    /// layer : ExcessLayer
    ///     The layer, with positive expected loss at every age in the model.
    /// model : ScaleShift or ParetoShape
    ///     How the severity at each maturity relates to the ultimate one.
    /// counts : str, optional
    ///     A reported claim count column on the same ages. Without it the
    ///     counts are taken as complete at every age, which mixes frequency
    ///     development into the severity shift. Not allowed with a
    ///     ``ParetoShape``.
    /// tail : float or tail estimator, optional
    ///     The ground-up tail of the chain ladder: it fixes one more shift
    ///     for the oldest age and the layer's tail. No tail by default.
    /// basis : {"incurred"}, default "incurred"
    ///     The loss basis of the column. Only ``"incurred"`` is derived;
    ///     ``"paid"`` is an error (design note, decision 9).
    ///
    /// Returns
    /// -------
    /// ExcessPattern
    #[staticmethod]
    #[pyo3(signature = (triangle, column, layer, model, counts = None, tail = None, basis = "incurred"))]
    #[allow(clippy::too_many_arguments)]
    fn from_ground_up(
        triangle: &PyTriangle,
        column: &str,
        layer: &Bound<'_, PyAny>,
        model: &Bound<'_, PyAny>,
        counts: Option<&str>,
        tail: Option<&Bound<'_, PyAny>>,
        basis: &str,
    ) -> PyResult<Self> {
        let basis = match basis {
            "incurred" => Basis::Incurred,
            "paid" => Basis::Paid,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "basis must be \"incurred\" (\"paid\" is not derived), got {basis:?}"
                )));
            }
        };
        let layer = layer_of(layer, "layer")?;
        let model = model_of(model)?;
        let tail = tail_arg(tail)?;
        let inner = ExcessPattern::from_ground_up(
            &triangle.inner,
            column,
            basis,
            counts,
            &tail,
            layer,
            &model,
        )
        .map_err(err)?;
        Ok(Self { inner })
    }

    /// Development ages (months) of the ground-up triangle.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// The layer's age-to-age factors, one fewer than the ages.
    #[getter]
    fn layer_ldf(&self) -> Vec<f64> {
        self.inner.layer_ldf.clone()
    }

    /// The layer's factors from each age to ultimate, with the tail.
    #[getter]
    fn layer_cdf(&self) -> Vec<f64> {
        self.inner.layer_cdf.clone()
    }

    /// The ground-up age-to-age factors the pattern was derived from.
    #[getter]
    fn ground_up_ldf(&self) -> Vec<f64> {
        self.inner.ground_up_ldf.clone()
    }

    /// The ground-up factor from the oldest age to ultimate.
    #[getter]
    fn ground_up_tail(&self) -> f64 {
        self.inner.ground_up_tail
    }

    /// The scale shift of each age, the last for the tail; empty under a
    /// ``ParetoShape``.
    #[getter]
    fn shifts(&self) -> Vec<f64> {
        self.inner.shifts.clone()
    }

    /// The layer's factor from the oldest age to ultimate (1 without a
    /// ground-up tail under a scale shift). Pass it as the ``tail`` of the
    /// layer's chain ladder.
    #[getter]
    fn tail(&self) -> f64 {
        self.inner.layer_tail
    }

    /// The layer's age-to-age factors as a dict from age in months, the
    /// ``ldf`` argument of the methods with ``ldf_kind="ldf"``.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (&age, &f) in self.inner.ages.iter().zip(&self.inner.layer_ldf) {
            d.set_item(age, f)?;
        }
        Ok(d)
    }

    /// The derived pattern as the arguments of the chain ladder, Mack and
    /// the expected-loss methods.
    ///
    /// Parameters
    /// ----------
    /// weight : float or dict, default 1.0
    ///     The weight on the derived factors, a number or a dict from age
    ///     in months to a number (``credibility_weights`` gives one); the
    ///     rest goes to the factors estimated from the layer triangle.
    ///     Weight 1 selects the derived factors alone, which a sparse layer
    ///     triangle needs.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``ldf``, ``ldf_kind`` and ``ldf_weight``, to be passed as keyword
    ///     arguments. The layer's tail is separate: ``tail``.
    #[pyo3(signature = (weight = None))]
    fn selected<'py>(
        &self,
        py: Python<'py>,
        weight: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let weight = match weight.filter(|w| !w.is_none()) {
            Some(w) => weight_of(w)?,
            None => PatternWeight::All(1.0),
        };
        let d = PyDict::new(py);
        d.set_item("ldf", self.ldf(py)?)?;
        d.set_item("ldf_kind", "ldf")?;
        d.set_item("ldf_weight", weight_object(py, &weight)?)?;
        Ok(d)
    }

    /// The weight on the derived pattern at each age, ``1 - z`` with
    /// ``z = n / (n + k)``, ``n`` the layer triangle's claim count (or its
    /// origins with a positive earlier value without counts) at the earlier
    /// age and ``k`` the credibility constant, capped so that removing any
    /// one origin moves the blended factor by at most ``max_influence`` of
    /// it. Where the layer's own factor cannot be estimated the weight is 1.
    ///
    /// Parameters
    /// ----------
    /// layer_triangle : Triangle
    ///     The layer's own triangle, a single segment on the pattern's ages.
    /// column : str
    ///     The layer loss column.
    /// k : float
    ///     The credibility constant, positive. No default (design note,
    ///     decision 19): the simulation study favours about 20.
    /// max_influence : float
    ///     The share of the blended factor one origin may move, positive.
    /// counts : str, optional
    ///     The layer's claim count column on the same origins and ages.
    ///
    /// Returns
    /// -------
    /// dict of int to float
    ///     The weight at each age except the oldest, for ``ldf_weight``.
    #[pyo3(signature = (layer_triangle, column, k, max_influence, counts = None))]
    fn credibility_weights<'py>(
        &self,
        py: Python<'py>,
        layer_triangle: &PyTriangle,
        column: &str,
        k: f64,
        max_influence: f64,
        counts: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let w = self
            .inner
            .credibility_weights(&layer_triangle.inner, column, counts, k, max_influence)
            .map_err(err)?;
        let PatternWeight::ByAge(by_age) = w else {
            return Err(PyValueError::new_err("expected a weight by age"));
        };
        let d = PyDict::new(py);
        for (age, x) in by_age {
            d.set_item(age, x)?;
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "ExcessPattern(ages={:?}, layer_ldf={:?}, tail={:?})",
            self.inner.ages, self.inner.layer_ldf, self.inner.layer_tail
        )
    }
}

/// The layer's share of a severity's mean, ``(LEV(b) - LEV(a)) / mean``,
/// on the ultimate basis.
///
/// An exposure curve's share comes from the curve, ``curve.layer_share(limit,
/// attachment, mpl)``, and goes to ``layer_exposure`` the same way.
///
/// Parameters
/// ----------
/// severity : severity distribution
///     Any severity with exact limited moments.
/// layer : ExcessLayer
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Pareto
/// >>> from prospicio.reserving import ExcessLayer, layer_share
/// >>> round(layer_share(Pareto(1.0, 2.0), ExcessLayer(4.0)), 12)
/// 0.125
#[pyfunction]
pub(crate) fn layer_share(severity: &Bound<'_, PyAny>, layer: &Bound<'_, PyAny>) -> PyResult<f64> {
    let sev = extract_severity(severity)?;
    share(&sev, layer_of(layer, "layer")?).map_err(err)
}

/// The expected loss to the layer of one claim at the maturity whose
/// claims are the ultimate ones divided by ``shift``:
/// ``(LEV(c b) - LEV(c a)) / c``. With ``shift = 1`` it is the layer at
/// ultimate.
///
/// Parameters
/// ----------
/// ultimate : severity distribution
///     The ultimate severity.
/// layer : ExcessLayer
/// shift : float
///     The scale shift ``c`` (``X_ultimate = c X_j``), positive.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Pareto
/// >>> from prospicio.reserving import ExcessLayer, layer_at_shift
/// >>> round(layer_at_shift(Pareto(1.0, 2.0), ExcessLayer(4.0), 1.0), 12)
/// 0.25
#[pyfunction]
pub(crate) fn layer_at_shift(
    ultimate: &Bound<'_, PyAny>,
    layer: &Bound<'_, PyAny>,
    shift: f64,
) -> PyResult<f64> {
    if !(shift.is_finite() && shift > 0.0) {
        return Err(PyValueError::new_err("shift must be finite and positive"));
    }
    let sev = extract_severity(ultimate)?;
    Ok(at_shift(&sev, layer_of(layer, "layer")?, shift))
}

/// A copy of the triangle with a column ``name``: the ``exposure`` column
/// times each origin's layer share, so that a Bornhuetter-Ferguson or Cape
/// Cod prior reads the layer's exposure.
///
/// The share comes from a severity (``layer_share``) or from an exposure
/// curve of ``prospicio.pricing`` (``curve.layer_share(limit, attachment,
/// mpl)``): the reserving core takes shares and not curves (design note,
/// decision 2).
///
/// Parameters
/// ----------
/// triangle : Triangle
/// exposure : str
///     The exposure (premium) column.
/// shares : float or list of float
///     One share for every origin, or one per origin, each between 0 and 1.
/// name : str
///     The new column's name, not already a column.
///
/// Returns
/// -------
/// Triangle
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Mbbefd
/// >>> from prospicio.reserving import Triangle, layer_exposure
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [10.0, 15.0, 11.0], "premium": [100.0, 100.0, 120.0]},
/// ... )
/// >>> share = Mbbefd.swiss_re(3.0).layer_share(5e6, 5e6, 10e6)
/// >>> layered = layer_exposure(tri, "premium", share, "layer_premium")
/// >>> "layer_premium" in layered.columns
/// True
#[pyfunction]
pub(crate) fn layer_exposure(
    triangle: &PyTriangle,
    exposure: &str,
    shares: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<PyTriangle> {
    let shares: Vec<f64> = if let Ok(one) = shares.extract::<f64>() {
        vec![one]
    } else {
        shares.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "shares must be a number or a list of numbers, got {}",
                py_repr(shares)
            ))
        })?
    };
    let inner = exposure_core(&triangle.inner, exposure, &shares, name).map_err(err)?;
    Ok(PyTriangle { inner })
}
