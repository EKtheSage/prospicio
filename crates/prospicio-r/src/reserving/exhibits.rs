//! The exhibits: `DevelopmentExhibit`, `UltimateExhibit` and `SelectedUltimates`.

use super::*;

/// Friedland's development exhibit of one column
/// (`docs/design/reserving-v02.md`, decision 34). `values` and
/// `link_ratios` are row-major over origins, `averages` row-major over the
/// averages (one row each, a value per link).
#[extendr]
pub(crate) struct DevelopmentExhibit {
    pub(crate) inner: DevelopmentExhibitInner,
}

#[extendr]
impl DevelopmentExhibit {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn ages(&self) -> Vec<i32> {
        self.inner.ages.iter().map(|&a| a as i32).collect()
    }

    fn link_labels(&self) -> Vec<String> {
        self.inner.link_labels()
    }

    fn values(&self) -> Vec<f64> {
        self.inner.values.concat()
    }

    fn link_ratios(&self) -> Vec<f64> {
        self.inner.link_ratios.concat()
    }

    fn average_names(&self) -> Vec<String> {
        self.inner.averages.iter().map(|(n, _)| n.clone()).collect()
    }

    fn averages(&self) -> Vec<f64> {
        self.inner
            .averages
            .iter()
            .flat_map(|(_, f)| f.iter().copied())
            .collect()
    }

    fn selected(&self) -> Vec<f64> {
        self.inner.selected.clone()
    }

    fn tail(&self) -> f64 {
        self.inner.tail
    }

    fn cdf(&self) -> Vec<f64> {
        self.inner.cdf.clone()
    }

    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed.clone()
    }
}

/// A summary of each method's ultimate per origin beside the claims to
/// date (`docs/design/reserving-v02.md`, decision 34). `reported` and
/// `paid` are empty when not given.
#[extendr]
pub(crate) struct UltimateExhibit {
    pub(crate) inner: UltimateExhibitInner,
}

#[extendr]
impl UltimateExhibit {
    /// From the claims to date; `reported` and `paid` may be `NULL`.
    fn from_claims(
        origin_year: &[i32],
        origin_month: &[i32],
        origin_grain: &str,
        ages: &[f64],
        reported: Nullable<Vec<f64>>,
        paid: Nullable<Vec<f64>>,
    ) -> Result<Self> {
        let g = grain(origin_grain, "origin_grain")?;
        let origins = months(origin_year, origin_month, "origin")?
            .into_iter()
            .map(|m| Period::containing(m, g))
            .collect();
        let ages = ages
            .iter()
            .map(|&a| {
                let a = whole(a, "age")?;
                Lag::try_from(a)
                    .map_err(|_| Error::Other("age must be a whole number from 0".into()))
            })
            .collect::<Result<Vec<Lag>>>()?;
        let opt = |v: Nullable<Vec<f64>>| match v {
            Nullable::NotNull(v) => Some(v),
            Nullable::Null => None,
        };
        let inner =
            UltimateExhibitInner::new(origins, ages, opt(reported), opt(paid)).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn with_method(&self, name: &str, ultimate: Vec<f64>) -> Result<Self> {
        let inner = self
            .inner
            .clone()
            .with_method(name, ultimate)
            .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn ages(&self) -> Vec<i32> {
        self.inner.ages.iter().map(|&a| a as i32).collect()
    }

    fn reported(&self) -> Vec<f64> {
        self.inner.reported.clone().unwrap_or_default()
    }

    fn paid(&self) -> Vec<f64> {
        self.inner.paid.clone().unwrap_or_default()
    }

    fn methods(&self) -> Vec<String> {
        self.inner.methods.iter().map(|(n, _)| n.clone()).collect()
    }

    fn ultimate(&self, name: &str) -> Result<Vec<f64>> {
        self.inner
            .ultimate(name)
            .map(<[f64]>::to_vec)
            .ok_or_else(|| Error::Other(format!("the exhibit has no method {name:?}")))
    }

    fn ibnr(&self, name: &str) -> Result<Vec<f64>> {
        self.inner.ibnr(name).map_err(to_r)
    }

    fn unpaid(&self, name: &str) -> Result<Vec<f64>> {
        self.inner.unpaid(name).map_err(to_r)
    }

    /// Each named method's weight per origin, `weights` column-major
    /// (origins x `names`).
    fn select_by_origin(&self, names: Vec<String>, weights: &[f64]) -> Result<SelectedUltimates> {
        let n = self.inner.n_origins();
        if weights.len() != n * names.len() {
            return Err(Error::Other(format!(
                "the weights have {} values for {n} origins and {} methods",
                weights.len(),
                names.len()
            )));
        }
        let by = names
            .into_iter()
            .zip(weights.chunks(n.max(1)))
            .map(|(name, w)| (name, w.to_vec()))
            .collect();
        self.select(MethodWeights::ByOrigin(by))
    }

    /// Weights by age: `weights` row-major (`ages` x `names`).
    fn select_by_age(
        &self,
        ages: &[f64],
        names: Vec<String>,
        weights: &[f64],
    ) -> Result<SelectedUltimates> {
        if weights.len() != ages.len() * names.len() {
            return Err(Error::Other(format!(
                "the weights have {} values for {} ages and {} methods",
                weights.len(),
                ages.len(),
                names.len()
            )));
        }
        let mut rows = Vec::with_capacity(ages.len());
        for (k, &age) in ages.iter().enumerate() {
            let age = Lag::try_from(whole(age, "age")?)
                .map_err(|_| Error::Other("an age must be a whole number from 0".into()))?;
            let row = names
                .iter()
                .enumerate()
                .map(|(m, name)| (name.clone(), weights[k * names.len() + m]))
                .collect();
            rows.push((age, row));
        }
        rows.sort_by_key(|(age, _)| *age);
        self.select(MethodWeights::ByAge(rows))
    }

    fn select_single(&self, name: &str) -> Result<SelectedUltimates> {
        self.select(MethodWeights::Single(name.to_string()))
    }
}

impl UltimateExhibit {
    fn select(&self, weights: MethodWeights) -> Result<SelectedUltimates> {
        let inner = self
            .inner
            .select(&UltimateSelection::new(weights))
            .map_err(to_r)?;
        Ok(SelectedUltimates { inner })
    }
}

/// The selected ultimate per origin and the weights that gave it.
/// `weights` is column-major (origins x `weight_names`); `ibnr`, `unpaid`,
/// `reported` and `paid` are empty when not defined.
#[extendr]
pub(crate) struct SelectedUltimates {
    inner: SelectedUltimatesInner,
}

#[extendr]
impl SelectedUltimates {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn ages(&self) -> Vec<i32> {
        self.inner.ages.iter().map(|&a| a as i32).collect()
    }

    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }

    fn ibnr(&self) -> Vec<f64> {
        self.inner.ibnr().unwrap_or_default()
    }

    fn unpaid(&self) -> Vec<f64> {
        self.inner.unpaid().unwrap_or_default()
    }

    fn reported(&self) -> Vec<f64> {
        self.inner.reported.clone().unwrap_or_default()
    }

    fn paid(&self) -> Vec<f64> {
        self.inner.paid.clone().unwrap_or_default()
    }

    fn weight_names(&self) -> Vec<String> {
        self.inner.weights.iter().map(|(n, _)| n.clone()).collect()
    }

    fn weights(&self) -> Vec<f64> {
        self.inner
            .weights
            .iter()
            .flat_map(|(_, w)| w.iter().copied())
            .collect()
    }
}

extendr_module! {
    mod exhibits;
    impl DevelopmentExhibit;
    impl UltimateExhibit;
    impl SelectedUltimates;
}
