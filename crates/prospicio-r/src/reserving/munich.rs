//! The Munich chain ladder, `MunichFit`.

use super::*;

/// The Munich chain ladder fitted to every segment of a triangle's paid
/// and incurred columns (`docs/design/reserving-v02.md`, decision 33).
/// Per-origin vectors run over the origins of each segment in turn.
#[extendr]
pub(crate) struct MunichFit {
    pub(crate) inner: SegmentFits<MunichInner>,
}

impl MunichFit {
    fn per_origin(&self, f: impl Fn(&MunichInner) -> Vec<f64>) -> Vec<f64> {
        by_origin(&self.inner, f)
    }
}

#[extendr]
impl MunichFit {
    fn n_segments(&self) -> i32 {
        self.inner.len() as i32
    }

    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// The segment label (parts joined by " / ") of each per-origin value.
    fn row_segments(&self) -> Vec<String> {
        self.inner
            .iter()
            .flat_map(|(l, f)| std::iter::repeat_n(l.to_string(), f.origins.len()))
            .collect()
    }

    fn origins(&self) -> Vec<String> {
        self.inner
            .fits
            .iter()
            .flat_map(|f| f.origins.iter().map(ToString::to_string))
            .collect()
    }

    fn development(&self) -> Result<Vec<i32>> {
        let fit = single(&self.inner, "development", "development_frame()")?;
        Ok(fit.development.iter().map(|&a| a as i32).collect())
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.per_origin(|f| f.latest_paid.clone())
    }

    fn latest_incurred(&self) -> Vec<f64> {
        self.per_origin(|f| f.latest_incurred.clone())
    }

    fn ultimate_paid(&self) -> Vec<f64> {
        self.per_origin(|f| f.ultimate_paid.clone())
    }

    fn ultimate_incurred(&self) -> Vec<f64> {
        self.per_origin(|f| f.ultimate_incurred.clone())
    }

    fn reserve_paid(&self) -> Vec<f64> {
        self.per_origin(MunichInner::reserve_paid)
    }

    fn ibnr_incurred(&self) -> Vec<f64> {
        self.per_origin(MunichInner::ibnr_incurred)
    }

    fn latest_ratio(&self) -> Vec<f64> {
        self.per_origin(MunichInner::latest_ratio)
    }

    fn ultimate_ratio(&self) -> Vec<f64> {
        self.per_origin(MunichInner::ultimate_ratio)
    }

    fn chain_ladder_paid(&self) -> Vec<f64> {
        self.per_origin(|f| f.paid.chain_ladder.ultimate.clone())
    }

    fn chain_ladder_incurred(&self) -> Vec<f64> {
        self.per_origin(|f| f.incurred.chain_ladder.ultimate.clone())
    }

    fn total_ultimate_paid(&self) -> f64 {
        self.inner
            .fits
            .iter()
            .map(MunichInner::total_ultimate_paid)
            .sum()
    }

    fn total_ultimate_incurred(&self) -> f64 {
        self.inner
            .fits
            .iter()
            .map(MunichInner::total_ultimate_incurred)
            .sum()
    }

    fn lambda_paid(&self) -> Result<f64> {
        Ok(single(&self.inner, "lambda_paid", "totals_frame()")?.lambda_paid)
    }

    fn lambda_incurred(&self) -> Result<f64> {
        Ok(single(&self.inner, "lambda_incurred", "totals_frame()")?.lambda_incurred)
    }

    fn n_residuals_paid(&self) -> Result<i32> {
        Ok(single(&self.inner, "n_residuals_paid", "totals_frame()")?.n_residuals_paid as i32)
    }

    fn n_residuals_incurred(&self) -> Result<i32> {
        Ok(
            single(&self.inner, "n_residuals_incurred", "totals_frame()")?.n_residuals_incurred
                as i32,
        )
    }

    fn q(&self) -> Result<Vec<f64>> {
        Ok(single(&self.inner, "q", "development_frame()")?.q.clone())
    }

    fn q_inverse(&self) -> Result<Vec<f64>> {
        Ok(single(&self.inner, "q_inverse", "development_frame()")?
            .q_inverse
            .clone())
    }

    fn rho_paid(&self) -> Result<Vec<f64>> {
        Ok(single(&self.inner, "rho_paid", "development_frame()")?
            .rho_paid
            .clone())
    }

    fn rho_incurred(&self) -> Result<Vec<f64>> {
        Ok(single(&self.inner, "rho_incurred", "development_frame()")?
            .rho_incurred
            .clone())
    }

    /// Mack's model of the paid column alone.
    fn mack_paid(&self) -> MackFit {
        MackFit {
            inner: self.inner.map(|f| f.paid.clone()),
        }
    }

    /// Mack's model of the incurred column alone.
    fn mack_incurred(&self) -> MackFit {
        MackFit {
            inner: self.inner.map(|f| f.incurred.clone()),
        }
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.munich_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.munich_totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.munich_development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

extendr_module! {
    mod munich;
    impl MunichFit;
}
