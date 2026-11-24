//! `DevelopmentPatterns`, each simulation's development pattern of a stochastic fit.

use super::*;

/// Each simulation's development pattern of a stochastic reserving fit
/// (``fit.patterns``), to pair with the simulation's loss ratio in the
/// cash-flow engine.
///
/// A fit that draws its development per simulation (the bootstraps'
/// pseudo factors, the Bayesian models' posterior draws) has a pattern
/// for each simulation: row ``i`` is simulation ``i`` of the fit's
/// ``reserves`` and of ``ultimate_draws``. A segment is named by its label
/// parts, a tuple of strings; a fit of one segment needs no label.
///
/// Examples
/// --------
/// >>> # patterns = fit.patterns
/// >>> # paid = patterns.monthly(120, monotone="running_max")
/// >>> # terms.run(u, paid_patterns=paid)
#[pyclass(name = "DevelopmentPatterns", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDevelopmentPatterns {
    pub(crate) inner: DevelopmentPatterns,
}

#[pymethods]
impl PyDevelopmentPatterns {
    /// The number of simulations, the fit's.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The label of each segment, a tuple of strings (empty for a fit of
    /// one segment).
    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        self.inner
            .segments()
            .iter()
            .map(|s| PyTuple::new(py, s.label()))
            .collect()
    }

    /// The development ages in months: the triangle's (for ``Compartmental``,
    /// its square's, to the last age the reserves reach), and with a tail
    /// one development period past the oldest, where the share is 1.
    ///
    /// Parameters
    /// ----------
    /// segment : None, str or tuple of str
    ///     The segment's label parts; ``None`` for a fit of one segment.
    #[pyo3(signature = (segment = None))]
    fn ages(&self, segment: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<u32>> {
        let parts = segment_parts(segment)?;
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        Ok(self.inner.segment(&refs).map_err(err)?.ages().to_vec())
    }

    /// The cumulative share of the ultimate at each age, one list per
    /// simulation.
    ///
    /// Parameters
    /// ----------
    /// segment : None, str or tuple of str
    ///     As in ``ages``.
    ///
    /// Returns
    /// -------
    /// list of list of float
    #[pyo3(signature = (segment = None))]
    fn pattern_draws(&self, segment: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<f64>>> {
        let parts = segment_parts(segment)?;
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        self.inner.pattern_draws(&refs).map_err(err)
    }

    /// The mean share at each age over the simulations.
    ///
    /// Parameters
    /// ----------
    /// segment : None, str or tuple of str
    ///     As in ``ages``.
    #[pyo3(signature = (segment = None))]
    fn mean(&self, segment: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        let parts = segment_parts(segment)?;
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        Ok(self.inner.segment(&refs).map_err(err)?.mean())
    }

    /// Each simulation's pattern by month, for the cash-flow engine's
    /// ``paid_patterns``.
    ///
    /// Parameters
    /// ----------
    /// months : int
    ///     The number of months in each row; it must reach the month at
    ///     which the converted pattern reaches 1.
    /// segment : None, str or tuple of str
    ///     As in ``ages``.
    /// basis : None, "origin", "origin_ln_linear" or UnderwritingBasis
    ///     The basis the pattern is put on. ``None`` and ``"origin"``: the
    ///     origin's own basis, a month from inception per month of age, a
    ///     pattern of longer origin periods read between its ages by the
    ///     Weibull growth curve; ``"origin_ln_linear"`` reads it as
    ///     ``ln CDF`` linear between ages. An ``UnderwritingBasis``: the
    ///     pattern of a year of policies written and earned as it says.
    /// monotone : {"refuse", "running_max"}, default "refuse"
    ///     What to do with a simulation whose share falls from one age to
    ///     the next (a pseudo factor below 1) or is not positive (a
    ///     negative factor, which Mack's bootstrap can draw):
    ///     ``"refuse"`` raises an error naming how many simulations and the
    ///     first; ``"running_max"`` holds each share at the largest before
    ///     it, capped at 1, and gives a share that is not positive the next
    ///     positive share, which the simulation then pays from its first age.
    ///     Bootstrap draws usually need ``"running_max"``: at the oldest
    ///     ages, where one or two cells set the factor, the ODP bootstrap
    ///     draws a pseudo factor below 1 in some simulations, which puts
    ///     the share above 1 before it falls back to 1 (77 of 2,000 draws
    ///     on the sample submission's paid triangle to 108 months).
    ///
    /// Returns
    /// -------
    /// list of list of float
    ///     One row of ``months`` values per simulation.
    ///
    /// Notes
    /// -----
    /// Under a growth-curve reading each simulation's curve is searched
    /// from the mean pattern's, and the simulations are read on all cores,
    /// so 2,000 draws at 144 months take well under a second on 16 cores
    /// (a few seconds on one). The result does not depend on the number of
    /// threads; it is within about ``1e-9`` of a share of the search from
    /// every start that converting a single pattern makes.
    #[pyo3(signature = (months, segment = None, basis = None, monotone = "refuse"))]
    fn monthly(
        &self,
        py: Python<'_>,
        months: usize,
        segment: Option<&Bound<'_, PyAny>>,
        basis: Option<&Bound<'_, PyAny>>,
        monotone: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let parts = segment_parts(segment)?;
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        let basis = monthly_basis(basis)?;
        let monotone = match monotone {
            "refuse" => Monotone::Refuse,
            "running_max" => Monotone::RunningMax,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "monotone must be \"refuse\" or \"running_max\", got {monotone:?}"
                )));
            }
        };
        py.detach(|| self.inner.monthly(&refs, &basis, months, monotone))
            .map_err(err)
    }

    fn __repr__(&self) -> String {
        format!(
            "DevelopmentPatterns(n_sims={}, segments={})",
            self.inner.n_sims(),
            self.inner.segments().len()
        )
    }
}
