//! A book of deals on one shared simulation (Aggregate lane) over
//! `prospicio_pricing::book` (`docs/design/time-value.md`, decisions 21 to
//! 27). `R/book.R` builds the arguments.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::book::{DealBook, DealRun, Tail};
use prospicio_pricing::cashflow::MonthlyPatterns;
use prospicio_pricing::returns::{Irr, IrrOptions};
use prospicio_prob::statistics::StatisticsGrid;

use crate::cashflow::Cashflow;
use crate::distributions::PredictiveDistribution;
use crate::reinsurance::ReinsuranceTower;
use crate::risk::{quantile_method, tvar_method};
use crate::{to_r, whole};

fn month(x: f64, name: &str) -> Result<usize> {
    whole(x, name).map(|m| m as usize)
}

/// One deal's amounts in each simulation; `R/book.R`'s `deal_run()`
/// documents the arguments.
#[extendr]
pub(crate) struct BookDeal {
    inner: DealRun,
}

#[extendr]
impl BookDeal {
    /// `deductions` column-major, one column of `n` amounts a name.
    fn new(
        name: &str,
        premium: Vec<f64>,
        loss: Vec<f64>,
        names: Vec<String>,
        deductions: &[f64],
        outside: &[i32],
        inception: f64,
    ) -> Result<Self> {
        let n = premium.len();
        if deductions.len() != names.len() * n || outside.len() != names.len() {
            return Err(Error::Other(
                "each deduction needs one amount a simulation".into(),
            ));
        }
        let mut run = DealRun::new(name, premium, loss).map_err(to_r)?;
        for (k, name) in names.iter().enumerate() {
            let values = deductions[k * n..(k + 1) * n].to_vec();
            run = run.deduction(name, values, outside[k] != 0).map_err(to_r)?;
        }
        Ok(BookDeal {
            inner: run.inception(month(inception, "inception")?),
        })
    }

    fn from_tower(
        name: &str,
        tower: Robj,
        result: Robj,
        layer: &str,
        inception: f64,
        reinstatement_premium: bool,
    ) -> Result<Self> {
        let tower = <&ReinsuranceTower>::try_from(&tower)
            .map_err(|_| Error::Other("tower must be a reinsurance_tower".into()))?;
        let pd = <&PredictiveDistribution>::try_from(&result)
            .map_err(|_| Error::Other("result must be a predictive_distribution".into()))?;
        let run =
            DealRun::from_tower_with(name, &tower.inner, &pd.inner, layer, reinstatement_premium)
                .map_err(to_r)?;
        Ok(BookDeal {
            inner: run.inception(month(inception, "inception")?),
        })
    }

    fn with_inception(&self, inception: f64) -> Result<Self> {
        Ok(BookDeal {
            inner: self.inner.clone().inception(month(inception, "inception")?),
        })
    }

    fn with_patterns(&self, written: Vec<f64>, earned: Vec<f64>, paid: Vec<f64>) -> Result<Self> {
        let p = MonthlyPatterns::new(written, earned, paid).map_err(to_r)?;
        Ok(BookDeal {
            inner: self.inner.clone().patterns(p),
        })
    }

    /// `flows` row-major, one row of `months` a simulation; `terms`, when
    /// not `NULL`, a `Cashflow` whose patterns the deal takes.
    fn with_flows(
        &self,
        months: f64,
        flows: Vec<f64>,
        funded: Vec<f64>,
        opening: Vec<f64>,
        terms: Robj,
    ) -> Result<Self> {
        let mut run = self
            .inner
            .clone()
            .flows(month(months, "months")?, flows, funded, opening)
            .map_err(to_r)?;
        if !terms.is_null() {
            let terms = <&Cashflow>::try_from(&terms)
                .map_err(|_| Error::Other("terms must be a cashflow_terms".into()))?;
            run = run.patterns(terms.inner.patterns.clone());
        }
        Ok(BookDeal { inner: run })
    }

    fn name(&self) -> String {
        self.inner.name().into()
    }

    fn n_sims(&self) -> i32 {
        self.inner.n_sims() as i32
    }

    fn inception(&self) -> i32 {
        self.inner.inception_month() as i32
    }

    fn premium(&self) -> Vec<f64> {
        self.inner.premium().to_vec()
    }

    fn loss(&self) -> Vec<f64> {
        self.inner.loss().to_vec()
    }
}

fn irr_list(r: &[Irr]) -> List {
    list!(
        rate = r.iter().map(|x| x.rate).collect::<Vec<_>>(),
        found = r.iter().map(|x| x.found).collect::<Vec<_>>(),
        npv = r.iter().map(|x| x.npv).collect::<Vec<_>>(),
        sign_changes = r.iter().map(|x| x.sign_changes as i32).collect::<Vec<_>>()
    )
}

/// Several deals on one shared simulation; `R/book.R`'s `deal_book()`.
#[extendr]
pub(crate) struct Book {
    inner: DealBook,
}

#[extendr]
impl Book {
    fn new(deals: List) -> Result<Self> {
        let runs = deals
            .values()
            .map(|d| {
                <&BookDeal>::try_from(&d)
                    .map(|d| d.inner.clone())
                    .map_err(|_| Error::Other("every deal must be a deal_run".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let inner = DealBook::new(runs).map_err(to_r)?;
        Ok(Book { inner })
    }

    fn select(&self, names: Vec<String>) -> Result<Self> {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let inner = self.inner.select(&names).map_err(to_r)?;
        Ok(Book { inner })
    }

    fn shifted(&self, name: &str, inception: f64) -> Result<Self> {
        let inner = self
            .inner
            .shifted(name, month(inception, "inception")?)
            .map_err(to_r)?;
        Ok(Book { inner })
    }

    fn names(&self) -> Vec<String> {
        self.inner.names().into_iter().map(String::from).collect()
    }

    fn n_sims(&self) -> i32 {
        self.inner.n_sims() as i32
    }

    fn items(&self) -> Vec<String> {
        self.inner.items()
    }

    fn amounts(&self) -> Result<PredictiveDistribution> {
        let inner = self.inner.amounts().map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    fn summary(&self) -> Result<List> {
        let rows = self.inner.summary().map_err(to_r)?;
        Ok(list!(
            deal = rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
            item = rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
            mean = rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            ratio = rows.iter().map(|r| r.3).collect::<Vec<_>>()
        ))
    }

    /// The book's statistics (`names`, `total`) and each deal's
    /// contribution and marginal effect on the cells `co_names`, column-major
    /// with one column a deal.
    #[allow(clippy::too_many_arguments)]
    fn distribution(
        &self,
        item: &str,
        percentiles: &[f64],
        tvar_levels: &[f64],
        sd_multiples: &[f64],
        quantile: &str,
        tvar: &str,
        tail: &str,
    ) -> Result<List> {
        let grid = StatisticsGrid::new(percentiles.to_vec())
            .and_then(|g| g.tvar_levels(tvar_levels.to_vec()))
            .and_then(|g| g.sd_multiples(sd_multiples.to_vec()))
            .map_err(to_r)?
            .quantile_method(quantile_method(quantile)?)
            .tvar_method(tvar_method(tvar)?);
        let tail = match tail {
            "upper" => Tail::Upper,
            "lower" => Tail::Lower,
            other => {
                return Err(Error::Other(format!(
                    "unknown tail {other:?}; use \"upper\" or \"lower\""
                )));
            }
        };
        let out = self.inner.distribution(item, &grid, tail).map_err(to_r)?;
        let co_names = out
            .deals
            .first()
            .map(|c| c.contribution.names.clone())
            .unwrap_or_default();
        Ok(list!(
            names = out.total.names,
            total = out.total.values,
            co_names = co_names,
            deals = out.deals.iter().map(|c| c.deal.clone()).collect::<Vec<_>>(),
            contribution = out
                .deals
                .iter()
                .flat_map(|c| c.contribution.values.clone())
                .collect::<Vec<_>>(),
            marginal = out
                .deals
                .iter()
                .flat_map(|c| c.marginal.values.clone())
                .collect::<Vec<_>>()
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn cashflows(
        &self,
        rate: f64,
        lower: f64,
        upper: f64,
        points: f64,
        bisections: f64,
        newton_steps: f64,
    ) -> Result<List> {
        let options = IrrOptions {
            lower,
            upper,
            points: whole(points, "points")? as usize,
            bisections: whole(bisections, "bisections")? as usize,
            newton_steps: whole(newton_steps, "newton_steps")? as usize,
        };
        let out = self.inner.cashflows(rate, &options).map_err(to_r)?;
        let by_deal = List::from_values(out.irr_by_deal.iter().map(|r| irr_list(r)));
        Ok(list!(
            months = out.months as i32,
            deals = out.deals,
            flows = out.flows,
            mean_by_deal = out.mean_by_deal.concat(),
            irr = irr_list(&out.irr),
            npv = out.npv,
            moic = out.moic,
            irr_by_deal = by_deal
        ))
    }

    fn calendar(&self, start_year: i32) -> Result<PredictiveDistribution> {
        let inner = self.inner.calendar(start_year as i64).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

extendr_module! {
    mod book;
    impl BookDeal;
    impl Book;
}
