//! The basis of the age-to-age factors and `UnderwritingBasis`.

use super::*;

// Latest diagonals (`reserving-v02.md`, decisions 15 to 19).

/// The name of a grain, as ``origin_grain`` gives it.
pub(crate) fn grain_name(g: Grain) -> &'static str {
    match g {
        Grain::Month => "M",
        Grain::Quarter => "Q",
        Grain::Semester => "S",
        Grain::Year => "Y",
    }
}

/// How a method converts its selected pattern at fit time: ``ldf_origin``,
/// the origin grain it was measured on, and ``ldf_underwriting``, the
/// underwriting basis it is converted to.
#[derive(Debug, Clone, Default)]
pub(crate) struct LdfBasis {
    pub(crate) origin: Option<Grain>,
    pub(crate) underwriting: Option<UnderwritingBasis>,
}

/// The ``ldf_origin`` and ``ldf_underwriting`` arguments: each ``None``,
/// or, with ``ldf``, the origin grain the selected pattern was measured on
/// and the underwriting basis it is converted to.
pub(crate) fn ldf_basis_arg(
    ldf_origin: Option<&str>,
    ldf_underwriting: Option<&PyUnderwritingBasis>,
    ldf: Option<&Bound<'_, PyAny>>,
) -> PyResult<LdfBasis> {
    let no_ldf = ldf.is_none_or(|l| l.is_none());
    if no_ldf && ldf_underwriting.is_some() {
        return Err(PyValueError::new_err(
            "ldf_underwriting is given without ldf",
        ));
    }
    let underwriting = ldf_underwriting.map(|b| b.inner.clone());
    let Some(name) = ldf_origin else {
        return Ok(LdfBasis {
            origin: None,
            underwriting,
        });
    };
    if no_ldf {
        return Err(PyValueError::new_err("ldf_origin is given without ldf"));
    }
    let origin = match name {
        "M" => Grain::Month,
        "Q" => Grain::Quarter,
        "S" => Grain::Semester,
        "Y" => Grain::Year,
        _ => {
            return Err(PyValueError::new_err(format!(
                "ldf_origin must be \"M\", \"Q\", \"S\" or \"Y\", got {name:?}"
            )));
        }
    };
    Ok(LdfBasis {
        origin: Some(origin),
        underwriting,
    })
}

/// A selected pattern converted as `basis` says, for origins of `origin`:
/// to an underwriting basis from accident periods of ``ldf_origin`` (the
/// target's grain without it, `SelectedPattern::to_underwriting_basis`),
/// its writing no longer than a period of `origin`
/// (`UnderwritingBasis::check_within`),
/// or averaged from origin periods of ``ldf_origin`` over the periods of
/// `origin` (`SelectedPattern::averaged_over_origins`); unchanged with
/// neither.
pub(crate) fn averaged(
    pattern: &Option<SelectedPattern>,
    basis: &LdfBasis,
    origin: Grain,
) -> PyResult<Option<SelectedPattern>> {
    match (pattern, basis) {
        (
            Some(p),
            LdfBasis {
                underwriting: Some(u),
                origin: from,
            },
        ) => {
            u.check_within(origin).map_err(err)?;
            Ok(Some(
                p.to_underwriting_basis(from.unwrap_or(origin), u)
                    .map_err(err)?,
            ))
        }
        (
            Some(p),
            LdfBasis {
                origin: Some(g), ..
            },
        ) => Ok(Some(p.averaged_over_origins(*g, origin).map_err(err)?)),
        _ => Ok(pattern.clone()),
    }
}

/// A chain ladder with its pattern converted as `basis` says.
pub(crate) fn chain_ladder_on(
    cl: &ChainLadder,
    basis: &LdfBasis,
    origin: Grain,
) -> PyResult<ChainLadder> {
    let mut cl = cl.clone();
    cl.development.pattern = averaged(&cl.development.pattern, basis, origin)?;
    Ok(cl)
}

/// `impl_resolved!(PyX => X, ...)`: `PyX::resolved(origin)`, the method
/// with its pattern converted as ``ldf_origin`` and ``ldf_underwriting``
/// say, for origins of `origin`: through the method's chain ladder
/// (`chain_ladder`), or its own pattern at the path after `pattern`.
macro_rules! impl_resolved {
    ($ty:ty => $method:ty, chain_ladder) => {
        impl $ty {
            pub(crate) fn resolved(&self, origin: Grain) -> PyResult<$method> {
                let mut m = self.inner.clone();
                m.chain_ladder = chain_ladder_on(&m.chain_ladder, &self.ldf_basis, origin)?;
                Ok(m)
            }
        }
    };
    ($ty:ty => $method:ty, pattern $($field:ident).+) => {
        impl $ty {
            pub(crate) fn resolved(&self, origin: Grain) -> PyResult<$method> {
                let mut m = self.inner.clone();
                m.$($field).+ = averaged(&m.$($field).+, &self.ldf_basis, origin)?;
                Ok(m)
            }
        }
    };
}
pub(crate) use impl_resolved;

/// A method's ``repr`` with ``ldf_origin`` and ``ldf_underwriting`` before
/// its closing parenthesis, when they are set.
pub(crate) fn with_ldf_basis(repr: String, basis: &LdfBasis) -> String {
    let mut extra = String::new();
    if let Some(g) = basis.origin {
        extra.push_str(&format!(", ldf_origin={:?}", grain_name(g)));
    }
    if let Some(u) = &basis.underwriting {
        extra.push_str(&format!(", ldf_underwriting={}", underwriting_repr(u)));
    }
    match repr.strip_suffix(')') {
        Some(head) if !extra.is_empty() => format!("{head}{extra})"),
        _ => repr,
    }
}

fn underwriting_repr(u: &UnderwritingBasis) -> String {
    format!(
        "UnderwritingBasis(writing={:?}, policy_months={}, reading={:?})",
        u.writing,
        u.policy_months,
        reading_name(u.reading())
    )
}

/// The reading named `"weibull"`, `"loglogistic"` or `"ln-linear"`.
fn annual_reading(name: &str) -> PyResult<AnnualReading> {
    match name {
        "weibull" => Ok(AnnualReading::Curve(GrowthCurve::Weibull)),
        "loglogistic" => Ok(AnnualReading::Curve(GrowthCurve::LogLogistic)),
        "ln-linear" => Ok(AnnualReading::LnLinear),
        _ => Err(PyValueError::new_err(format!(
            "reading must be \"weibull\", \"loglogistic\" or \"ln-linear\", got {name:?}"
        ))),
    }
}

fn reading_name(reading: AnnualReading) -> &'static str {
    match reading {
        AnnualReading::Curve(GrowthCurve::Weibull) => "weibull",
        AnnualReading::Curve(GrowthCurve::LogLogistic) => "loglogistic",
        AnnualReading::LnLinear => "ln-linear",
    }
}

/// How an underwriting period's policies are written and earned, to convert
/// an accident-period development pattern to the underwriting period's.
///
/// Parameters
/// ----------
/// writing : sequence of float, optional
///     The share of the period's policies written in each month from its
///     start, each written at an even rate through its month: finite, not
///     negative, with a positive sum, and scaled to sum to 1. Not given,
///     even writing over 12 months, an underwriting year's. The writing,
///     its trailing zeros dropped, must be no longer than the triangle's
///     origin period, within which its policies are written: on
///     underwriting quarters give three months (``[1, 1, 1]`` for even
///     writing); longer writing is an error at fit time.
/// policy_months : int, default 12
///     Each policy's term in months; it earns evenly over the term.
/// reading : {"weibull", "loglogistic", "ln-linear"}, default "weibull"
///     How a pattern of accident periods longer than a month (an
///     accident-year pattern) is read between its ages when they are not
///     every month from 1: ``"weibull"`` or ``"loglogistic"`` fit that
///     growth curve through the period averages (an accident year at age
///     ``a`` reports the average of its twelve months) and correct it to
///     the nearest monotone accident-month pattern whose averages are the
///     given values exactly, so the given values are reproduced; increments
///     may fall only within a span where the given pattern itself develops
///     downward. ``"ln-linear"`` is the rule before this reading (linear
///     from 0 below the first age, ``ln CDF`` linear between ages), kept to
///     reproduce earlier results: on a simulated parallelogram book it puts
///     an underwriting year 31% too far developed at 12 months, where the
///     Weibull is within 0.2% at every age. An accident-month pattern, and
///     one given at every month, convert the same under every reading. A
///     curve needs two or more given values.
///
/// The underwriting period's exposure in its ``t``-th accident month is the
/// writing convolved with a policy's earning by month (``exposure``), and
/// the period's percent developed at age ``a`` from its start is
/// ``P(a) = sum_t e(t) p(a - t)``, ``p`` an accident month's (Rust's
/// ``SelectedPattern::to_underwriting_basis``). Pass it as
/// ``ldf_underwriting`` to a method that takes ``ldf``. The default, even
/// writing over a year and 12-month policies, is the parallelogram: the
/// exposure is triangular over 24 months.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import UnderwritingBasis
/// >>> e = UnderwritingBasis().exposure()
/// >>> len(e), round(e[0] * 288, 12), round(e[12] * 288, 12)
/// (24, 1.0, 23.0)
#[pyclass(name = "UnderwritingBasis", module = "prospicio.reserving", frozen)]
pub(crate) struct PyUnderwritingBasis {
    pub(crate) inner: UnderwritingBasis,
}

#[pymethods]
impl PyUnderwritingBasis {
    #[new]
    #[pyo3(signature = (writing = None, policy_months = 12, reading = "weibull"))]
    fn new(writing: Option<Vec<f64>>, policy_months: u32, reading: &str) -> PyResult<Self> {
        let inner = match writing {
            Some(w) => UnderwritingBasis::new(w, policy_months),
            None => UnderwritingBasis::even(12, policy_months),
        }
        .with_reading(annual_reading(reading)?);
        inner.exposure().map_err(err)?;
        Ok(Self { inner })
    }

    /// How a pattern of longer accident periods is read between its ages:
    /// ``"weibull"``, ``"loglogistic"`` or ``"ln-linear"``.
    #[getter]
    fn reading(&self) -> &'static str {
        reading_name(self.inner.reading())
    }

    /// The writing shares by month, as given.
    #[getter]
    fn writing(&self) -> Vec<f64> {
        self.inner.writing.clone()
    }

    /// The policy term in months.
    #[getter]
    fn policy_months(&self) -> u32 {
        self.inner.policy_months
    }

    /// The share of the period's exposure in each accident month from its
    /// start, summing to 1.
    fn exposure(&self) -> PyResult<Vec<f64>> {
        self.inner.exposure().map_err(err)
    }

    fn __repr__(&self) -> String {
        underwriting_repr(&self.inner)
    }
}

/// The label parts of a segment: ``None`` (a fit of one segment), one
/// string, or a tuple or list of strings.
pub(crate) fn segment_parts(segment: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let Some(s) = segment.filter(|s| !s.is_none()) else {
        return Ok(Vec::new());
    };
    if let Ok(one) = s.extract::<String>() {
        return Ok(vec![one]);
    }
    s.extract::<Vec<String>>().map_err(|_| {
        PyTypeError::new_err("segment must be None, a string, or a tuple or list of strings")
    })
}

/// The monthly basis a ``basis`` argument names.
pub(crate) fn monthly_basis(basis: Option<&Bound<'_, PyAny>>) -> PyResult<MonthlyBasis> {
    let Some(b) = basis.filter(|b| !b.is_none()) else {
        return Ok(MonthlyBasis::default());
    };
    if let Ok(u) = b.extract::<PyRef<'_, PyUnderwritingBasis>>() {
        return Ok(MonthlyBasis::Underwriting(u.inner.clone()));
    }
    match b.extract::<String>().as_deref() {
        Ok("origin") => Ok(MonthlyBasis::Origin(AnnualReading::default())),
        Ok("origin_ln_linear") => Ok(MonthlyBasis::Origin(AnnualReading::LnLinear)),
        _ => Err(PyValueError::new_err(
            "basis must be None, \"origin\", \"origin_ln_linear\" or an UnderwritingBasis",
        )),
    }
}
