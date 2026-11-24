//! The priors: `pattern_prior_rust`, `industry_prior_rust` and `IndustryPriorFit`.

use super::*;

/// A pattern prior from R's `pattern_prior()` arguments: `ages` (`NULL`
/// for none), then the means and sds of the log share ratios after the
/// first age.
pub(crate) fn pattern_arg(
    ages: Nullable<Vec<f64>>,
    mean: Vec<f64>,
    sd: Vec<f64>,
) -> Result<Option<PatternPrior>> {
    match ages {
        Nullable::Null => Ok(None),
        Nullable::NotNull(ages) => {
            let ages = lags(&ages)?;
            Ok(Some(PatternPrior::new(&ages, &mean, &sd).map_err(to_r)?))
        }
    }
}

/// Whole months from R numbers.
pub(crate) fn lags(ages: &[f64]) -> Result<Vec<Lag>> {
    ages.iter()
        .map(|&a| {
            let a = whole(a, "ages")?;
            Lag::try_from(a).map_err(|_| Error::Other(format!("age {a} is too large")))
        })
        .collect()
}

/// The prior's means and sds from `ages` and either `mean` or, with
/// `from_shares`, incremental shares, checked as `PatternPrior` checks
/// them: a list of `ages`, `mean` and `sd`.
#[extendr]
fn pattern_prior_rust(
    ages: Vec<f64>,
    values: Vec<f64>,
    sd: Vec<f64>,
    from_shares: bool,
) -> Result<List> {
    let lags = lags(&ages)?;
    let prior = if from_shares {
        PatternPrior::from_shares(&lags, &values, &sd)
    } else {
        PatternPrior::new(&lags, &values, &sd)
    }
    .map_err(to_r)?;
    Ok(list!(
        ages = prior
            .ages
            .iter()
            .map(|&a| f64::from(a))
            .collect::<Vec<f64>>(),
        mean = prior.mean.clone(),
        sd = prior.sd.clone(),
        shares = prior.shares(),
        percent_developed = prior.percent_developed()
    ))
}

/// `settings` is `min_developed`, `min_companies` and `subtract_process`
/// (0 or 1).
fn industry_settings(settings: &[f64]) -> Result<IndustryPriorSettings> {
    let &[min_developed, min_companies, subtract] = settings else {
        return Err(Error::Other(
            "industry prior settings are min_developed, min_companies and subtract_process".into(),
        ));
    };
    Ok(IndustryPriorSettings {
        min_developed,
        min_companies: whole(min_companies, "min_companies")? as usize,
        subtract_process: subtract != 0.0,
    })
}

/// The industry prior of `triangles` (a list of triangle pointers, one per
/// company) of loss `column`, with `premiums` (a list of numeric vectors,
/// one per company).
#[extendr]
fn industry_prior_rust(
    triangles: List,
    column: &str,
    premiums: List,
    settings: Vec<f64>,
) -> Result<IndustryPriorFit> {
    let mut tris = Vec::with_capacity(triangles.len());
    for robj in triangles.values() {
        let t: &Triangle = (&robj)
            .try_into()
            .map_err(|_| Error::Other("each of triangles must be a triangle".into()))?;
        tris.push(t.inner.clone());
    }
    let mut prem = Vec::with_capacity(premiums.len());
    for robj in premiums.values() {
        let p: Vec<f64> = robj
            .as_real_vector()
            .ok_or_else(|| Error::Other("each of premiums must be a numeric vector".into()))?;
        prem.push(p);
    }
    let inner = industry_settings(&settings)?
        .from_triangles(&tris, column, &prem)
        .map_err(to_r)?;
    Ok(IndustryPriorFit { inner })
}

/// The industry prior of a triangle with one segment per company, premium
/// from its `exposure` column.
#[extendr]
fn industry_prior_segments_rust(
    triangle: &Triangle,
    column: &str,
    exposure: &str,
    settings: Vec<f64>,
) -> Result<IndustryPriorFit> {
    let inner = industry_settings(&settings)?
        .from_segments(&triangle.inner, column, exposure)
        .map_err(to_r)?;
    Ok(IndustryPriorFit { inner })
}

/// An industry prior for the Bayesian BF (decision 29).
#[extendr]
pub(crate) struct IndustryPriorFit {
    inner: IndustryPrior,
}

#[extendr]
impl IndustryPriorFit {
    /// `apriori`, `apriori_se`, `between_cv`, `within_cv`, `process_cv`,
    /// `cv` and `tail_factor`.
    fn numbers(&self) -> List {
        let p = &self.inner;
        list!(
            apriori = p.apriori,
            apriori_se = p.apriori_se,
            between_cv = p.between_cv,
            within_cv = p.within_cv,
            process_cv = p.process_cv,
            cv = p.cv,
            tail_factor = p.tail_factor
        )
    }

    /// The pattern prior: `ages`, `mean`, `sd`, `shares` and
    /// `percent_developed`.
    fn pattern(&self) -> List {
        let prior = &self.inner.pattern;
        list!(
            ages = prior
                .ages
                .iter()
                .map(|&a| f64::from(a))
                .collect::<Vec<f64>>(),
            mean = prior.mean.clone(),
            sd = prior.sd.clone(),
            shares = prior.shares(),
            percent_developed = prior.percent_developed()
        )
    }

    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed.clone()
    }

    fn percent_developed_sd(&self) -> Vec<f64> {
        self.inner.percent_developed_sd.clone()
    }

    fn company_loss_ratios(&self) -> Vec<f64> {
        self.inner.company_loss_ratios.clone()
    }

    fn report(&self) -> List {
        let r = &self.inner.report;
        let ints = |v: &[usize]| v.iter().map(|&n| n as i32).collect::<Vec<i32>>();
        list!(
            n_companies = r.n_companies as i32,
            n_companies_used = r.n_companies_used as i32,
            n_origins_used = r.n_origins_used as i32,
            origins_used = ints(&r.origins_used),
            companies_by_age = ints(&r.companies_by_age),
            excluded = r.excluded.clone(),
            warnings = r.warnings.clone()
        )
    }

    /// The prior for a triangle whose last age is `age`.
    fn at_age(&self, age: f64) -> Result<Self> {
        let age = Lag::try_from(whole(age, "age")?)
            .map_err(|_| Error::Other(format!("age {age} is too large")))?;
        Ok(Self {
            inner: self.inner.at_age(age).map_err(to_r)?,
        })
    }
}

extendr_module! {
    mod priors;
    fn pattern_prior_rust;
    fn industry_prior_rust;
    fn industry_prior_segments_rust;
    impl IndustryPriorFit;
}
