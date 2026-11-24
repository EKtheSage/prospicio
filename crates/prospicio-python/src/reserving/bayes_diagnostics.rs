//! The Bayesian models' diagnostics: warnings and tables.

use super::*;

/// One ``UserWarning`` from `model` listing failed diagnostics, if any.
pub(crate) fn warn_diagnostics(py: Python<'_>, model: &str, warnings: &[String]) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let message = format!(
        "{model}: the MCMC diagnostics failed; check them before relying on the fit: {}",
        warnings.join("; ")
    );
    let message =
        std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &message, 1)
}

/// `Compartmental`'s warnings: the warning of cumulative paid on cells finer
/// than a year fires on healthy fits too, so it comes as a ``UserWarning`` of
/// its own and not as a failed diagnostic.
pub(crate) fn warn_compartmental(py: Python<'_>, warnings: &[String]) -> PyResult<()> {
    let (notes, failed): (Vec<String>, Vec<String>) = warnings
        .iter()
        .cloned()
        .partition(|w| w.contains(CUMULATIVE_SUB_ANNUAL_WARNING));
    warn_diagnostics(py, "Compartmental", &failed)?;
    if notes.is_empty() {
        return Ok(());
    }
    let message = format!("Compartmental: {}", notes.join("; "));
    let message =
        std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &message, 1)
}

/// MCMC diagnostics as a dict, as ``BayesOdpFit.diagnostics``.
pub(crate) fn diagnostics_dict<'py>(
    py: Python<'py>,
    d: &Diagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("max_rhat", d.max_rhat)?;
    out.set_item("min_ess_bulk", d.min_ess_bulk)?;
    out.set_item("min_ess_tail", d.min_ess_tail)?;
    out.set_item("divergences", d.divergences)?;
    out.set_item("warnings", d.warnings.clone())?;
    Ok(out)
}

/// Every segment's failed diagnostics, each message prefixed with its
/// segment's label when the triangle has keys.
pub(crate) fn segment_warnings<T>(
    segments: &SegmentFits<T>,
    diag: fn(&T) -> &Diagnostics,
) -> Vec<String> {
    segments
        .labels
        .iter()
        .zip(&segments.fits)
        .flat_map(|(label, fit)| {
            let prefix = if segments.key_names.is_empty() {
                String::new()
            } else {
                format!("{label}: ")
            };
            diag(fit)
                .warnings
                .iter()
                .map(move |w| format!("{prefix}{w}"))
        })
        .collect()
}

/// One row per segment: the key columns, then ``max_rhat``,
/// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and ``warnings``
/// (the segment's messages joined by ``"; "``).
pub(crate) fn segment_diagnostics_frame<'py, T>(
    py: Python<'py>,
    segments: &SegmentFits<T>,
    diag: fn(&T) -> &Diagnostics,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    for (k, name) in segments.key_names.iter().enumerate() {
        let values: Vec<String> = segments
            .labels
            .iter()
            .map(|l| l.parts().get(k).cloned().unwrap_or_default())
            .collect();
        out.set_item(name, values)?;
    }
    let col = |g: fn(&Diagnostics) -> f64| {
        segments
            .fits
            .iter()
            .map(|f| g(diag(f)))
            .collect::<Vec<f64>>()
    };
    out.set_item("max_rhat", col(|d| d.max_rhat))?;
    out.set_item("min_ess_bulk", col(|d| d.min_ess_bulk))?;
    out.set_item("min_ess_tail", col(|d| d.min_ess_tail))?;
    out.set_item(
        "divergences",
        segments
            .fits
            .iter()
            .map(|f| diag(f).divergences)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "warnings",
        segments
            .fits
            .iter()
            .map(|f| diag(f).warnings.join("; "))
            .collect::<Vec<_>>(),
    )?;
    py.import("pandas")?.call_method1("DataFrame", (out,))
}

/// A posterior summary as a pandas DataFrame, as ``BayesOdpFit.summary``.
pub(crate) fn summary_frame<'py>(
    py: Python<'py>,
    s: &[ParamSummary],
) -> PyResult<Bound<'py, PyAny>> {
    let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
    let out = PyDict::new(py);
    out.set_item(
        "parameter",
        s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
    )?;
    out.set_item("mean", col(|p| p.mean))?;
    out.set_item("sd", col(|p| p.sd))?;
    out.set_item("q05", col(|p| p.q05))?;
    out.set_item("q50", col(|p| p.q50))?;
    out.set_item("q95", col(|p| p.q95))?;
    out.set_item("rhat", col(|p| p.rhat))?;
    out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
    out.set_item("ess_tail", col(|p| p.ess_tail))?;
    py.import("pandas")?.call_method1("DataFrame", (out,))
}
