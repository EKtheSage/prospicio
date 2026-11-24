//! Merz and Wuthrich's paid-incurred chain, `PaidIncurredFit`.

use super::*;

/// A paid-incurred chain fit of a single-segment triangle (Merz and
/// Wuthrich 2010). Per-origin vectors run over the triangle's origins.
#[extendr]
pub(crate) struct PaidIncurredFit {
    pub(crate) inner: PicFit,
}

#[extendr]
impl PaidIncurredFit {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn development(&self) -> Vec<i32> {
        self.inner.ages.iter().map(|&a| a as i32).collect()
    }

    fn latest(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    fn latest_incurred(&self) -> Vec<f64> {
        self.inner.latest_incurred.clone()
    }

    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }

    fn reserve(&self) -> Vec<f64> {
        self.inner.reserve.clone()
    }

    fn incurred_weight(&self) -> Vec<f64> {
        self.inner.incurred_weight.clone()
    }

    fn std_err(&self) -> Vec<f64> {
        self.inner.std_err.clone()
    }

    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn total_std_err(&self) -> f64 {
        self.inner.total_std_err
    }

    fn sigma(&self) -> Vec<f64> {
        self.inner.sigma.clone()
    }

    fn tau(&self) -> Vec<f64> {
        self.inner.tau.clone()
    }

    fn phi(&self) -> Vec<f64> {
        self.inner.phi().to_vec()
    }

    fn psi(&self) -> Vec<f64> {
        self.inner.psi().to_vec()
    }

    fn posterior_sd(&self) -> Vec<f64> {
        self.inner.posterior_sd()
    }

    /// Row-major, `(Phi_0..Phi_J, Psi_0..Psi_{J-1})` on both axes.
    fn posterior_covariance(&self) -> Vec<f64> {
        self.inner.posterior_covariance.clone()
    }

    fn n_segments(&self) -> i32 {
        1
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
}

extendr_module! {
    mod paid_incurred;
    impl PaidIncurredFit;
}
