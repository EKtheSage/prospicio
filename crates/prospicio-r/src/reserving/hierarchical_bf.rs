//! The hierarchical Bayesian Bornhuetter-Ferguson's fits, `HierarchicalBfFit` and `HierarchicalBfOneYearFit`.

use super::*;

/// A hierarchical Bayesian Bornhuetter-Ferguson over the programs of a
/// triangle column.
#[extendr]
pub(crate) struct HierarchicalBfFit {
    pub(crate) inner: HierarchicalBfInner,
}

#[extendr]
impl HierarchicalBfFit {
    /// One row per program, the core's table (#838).
    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    /// Each program's segment label, joined by " / ", in the triangle's
    /// order.
    fn program_labels(&self) -> Vec<String> {
        self.inner
            .programs
            .iter()
            .map(|p| p.label.to_string())
            .collect()
    }

    /// One program as a list: `label`, `key` (its key values), `origins`,
    /// `exposure`, `latest`, `prior_ultimate`, `scale`, `expected_ultimate`,
    /// `pattern`, `loss_ratio`, `own_loss_ratio` and `shrinkage`. `index` is
    /// 1-based.
    fn program(&self, index: f64) -> Result<List> {
        let p = self.program_at(index)?;
        Ok(list!(
            label = p.label.to_string(),
            key = p.label.parts().to_vec(),
            origins = p
                .origins
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            exposure = p.exposure.clone(),
            latest = p.latest.clone(),
            prior_ultimate = p.prior_ultimate.clone(),
            scale = p.scale,
            expected_ultimate = p.expected_ultimate.clone(),
            pattern = p.pattern.clone(),
            loss_ratio = p.loss_ratio,
            own_loss_ratio = p.own_loss_ratio,
            shrinkage = p.shrinkage
        ))
    }

    /// The program's part of the joint reserves (`index` is 1-based).
    fn program_reserves(&self, index: f64) -> Result<PredictiveDistribution> {
        Ok(PredictiveDistribution {
            inner: self.program_at(index)?.reserves.clone(),
        })
    }

    /// The development ages of every program's pattern, in months.
    fn ages(&self) -> Vec<f64> {
        self.inner.ages.iter().map(|&a| f64::from(a)).collect()
    }

    /// The pooled scale per unit of premium.
    fn scale_per_premium(&self) -> f64 {
        self.inner.scale_per_premium
    }

    /// The joint reserves by program and origin.
    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary of every sampled parameter.
    fn summary(&self) -> Result<List> {
        Ok(summary_list(&self.inner.summary().map_err(to_r)?))
    }

    /// The posterior summary of the hyperparameters `level`, `tau` and
    /// `omega` on their natural scale.
    fn hyperparameters(&self) -> Result<List> {
        Ok(summary_list(&self.inner.hyperparameters().map_err(to_r)?))
    }
}

impl HierarchicalBfFit {
    fn program_at(&self, index: f64) -> Result<&HierarchicalBfProgram> {
        let n = self.inner.programs.len();
        let i = whole(index, "index")? as usize;
        if i == 0 || i > n {
            return Err(Error::Other(format!(
                "program index {i} is outside 1 to {n}"
            )));
        }
        Ok(&self.inner.programs[i - 1])
    }
}

/// The one-year view of a hierarchical Bayesian Bornhuetter-Ferguson over
/// every program: the lifetime fit, the opening reserve, the claims
/// development result by program and origin and each simulation's Pareto k.
#[extendr]
pub(crate) struct HierarchicalBfOneYearFit {
    pub(crate) inner: BayesOneYearInner<HierarchicalBfInner>,
}

#[extendr]
impl HierarchicalBfOneYearFit {
    /// One row per program and origin, the core's table (#838).
    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    /// One row per program, the core's table (#838).
    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    /// The lifetime fit, as `bayes_hierarchical_bf` gives it.
    fn fit(&self) -> HierarchicalBfFit {
        HierarchicalBfFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// The latest value plus the opening reserve, per program and origin.
    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    /// Each origin's posterior mean reserve on the observed triangle, per
    /// program and origin.
    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    /// The joint claims development result by program and origin.
    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        self.inner
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        self.inner
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        self.inner
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    fn refit_warnings(&self) -> f64 {
        self.inner.refit_warnings as f64
    }

    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    /// What else the view warns of, one message each: the programs whose
    /// next diagonal falls at development ages they have never observed.
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }
}

extendr_module! {
    mod hierarchical_bf;
    impl HierarchicalBfFit;
    impl HierarchicalBfOneYearFit;
}
