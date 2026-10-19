//! Casualty exposure rating (Aggregate lane) over
//! `prospicio_pricing::casualty`; the R API is in `R/casualty.R`.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::casualty::{
    Alae, CasualtyCurve, CasualtyPolicy, CasualtyProfile, CasualtyRating, ElfTable, IlfTable,
    ParetoAnchor,
};
use prospicio_pricing::layer::XsLayer;

use crate::distributions::severity_from_robj;
use crate::to_r;

/// `alae` and its `loading` (NaN for none): there is no default.
fn alae(name: &str, loading: f64) -> Result<Alae> {
    let need = || {
        if loading.is_nan() {
            Err(Error::Other(format!(
                "alae = \"{name}\" needs a loading (0 for none)"
            )))
        } else {
            Ok(loading)
        }
    };
    match name {
        "pro_rata" => Ok(Alae::ProRata(need()?)),
        "on_top" => Ok(Alae::OnTop(need()?)),
        "in_severity" if loading.is_nan() => Ok(Alae::InSeverity),
        "in_severity" => Err(Error::Other(
            "alae = \"in_severity\" takes no loading: the severity holds the ALAE".into(),
        )),
        _ => Err(Error::Other(format!(
            "alae must be \"pro_rata\", \"on_top\" or \"in_severity\", not \"{name}\""
        ))),
    }
}

fn treaty(limit: f64, attachment: f64) -> Result<XsLayer> {
    XsLayer::new(limit, attachment).map_err(to_r)
}

fn rating(r: &CasualtyRating) -> List {
    let rows = r.rows();
    list!(
        factor = rows.iter().map(|x| x.factor).collect::<Vec<_>>(),
        expected_loss = rows.iter().map(|x| x.expected_loss).collect::<Vec<_>>(),
        ceded = rows.iter().map(|x| x.ceded).collect::<Vec<_>>(),
        premium = rows.iter().map(|x| x.premium).collect::<Vec<_>>()
    )
}

/// A book of casualty policies.
#[extendr]
pub(crate) struct CasualtyBook {
    inner: CasualtyProfile,
}

/// `kinds[k]` is "severity", "ilf" or "elf"; a severity is `severities[k]`,
/// a table `limits[k]` and `values[k]`. Policies are columns; `curve` is
/// 0-based, `tag` "" for none.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn casualty_book_rust(
    kinds: Vec<String>,
    severities: List,
    limits: List,
    values: List,
    limit: &[f64],
    premium: &[f64],
    elr: &[f64],
    attachment: &[f64],
    share: &[f64],
    curve: &[f64],
    tag: Vec<String>,
) -> Result<CasualtyBook> {
    let mut curves = Vec::with_capacity(kinds.len());
    for (k, kind) in kinds.iter().enumerate() {
        let floats = |l: &List| -> Result<Vec<f64>> {
            l.elt(k)?
                .as_real_vector()
                .ok_or_else(|| Error::Other(format!("curve {}: a table needs numbers", k + 1)))
        };
        curves.push(match kind.as_str() {
            "severity" => CasualtyCurve::severity(severity_from_robj(&severities.elt(k)?)?),
            "ilf" => CasualtyCurve::Ilf(
                IlfTable::new(&floats(&limits)?, &floats(&values)?).map_err(to_r)?,
            ),
            "elf" => CasualtyCurve::Elf(
                ElfTable::new(&floats(&limits)?, &floats(&values)?).map_err(to_r)?,
            ),
            _ => return Err(Error::Other(format!("unknown curve kind {kind}"))),
        });
    }
    let n = limit.len();
    if [
        premium.len(),
        elr.len(),
        attachment.len(),
        share.len(),
        curve.len(),
        tag.len(),
    ]
    .iter()
    .any(|&m| m != n)
    {
        return Err(Error::Other(
            "every policy column needs one value per policy".into(),
        ));
    }
    let policies = (0..n)
        .map(|i| {
            let p = CasualtyPolicy::new(limit[i], premium[i], elr[i])
                .attachment(attachment[i])
                .share(share[i])
                .curve(crate::whole(curve[i], "curve")? as usize);
            Ok(if tag[i].is_empty() {
                p
            } else {
                p.tag(tag[i].clone())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CasualtyBook {
        inner: CasualtyProfile::new(curves, policies).map_err(to_r)?,
    })
}

#[extendr]
impl CasualtyBook {
    fn rate(&self, limit: f64, attachment: f64, alae_name: &str, loading: f64) -> Result<List> {
        let r = self
            .inner
            .rate(&treaty(limit, attachment)?, alae(alae_name, loading)?)
            .map_err(to_r)?;
        Ok(rating(&r))
    }

    #[allow(clippy::too_many_arguments)]
    fn rate_by_frequency(
        &self,
        limit: f64,
        attachment: f64,
        alae_name: &str,
        loading: f64,
        threshold: f64,
        counts: &[f64],
        excess: Robj,
    ) -> Result<List> {
        let excess = severity_from_robj(&excess)?;
        let r = self
            .inner
            .rate_by_frequency(
                &treaty(limit, attachment)?,
                alae(alae_name, loading)?,
                threshold,
                counts,
                &excess,
            )
            .map_err(to_r)?;
        Ok(rating(&r))
    }

    fn occurrences_above(&self, t: f64) -> Result<Vec<f64>> {
        self.inner.occurrences_above(t).map_err(to_r)
    }

    #[allow(clippy::too_many_arguments)]
    fn sweep(
        &self,
        limit: f64,
        attachment: f64,
        alae_name: &str,
        loading: f64,
        threshold: f64,
        counts: &[f64],
        alphas: &[f64],
    ) -> Result<List> {
        let s = self
            .inner
            .sweep(
                &treaty(limit, attachment)?,
                alae(alae_name, loading)?,
                threshold,
                counts,
                alphas,
            )
            .map_err(to_r)?;
        Ok(list!(
            alpha = s.iter().map(|p| p.alpha).collect::<Vec<_>>(),
            ceded = s.iter().map(|p| p.ceded).collect::<Vec<_>>(),
            loss_cost = s.iter().map(|p| p.loss_cost).collect::<Vec<_>>()
        ))
    }

    fn expected_claims(&self, t: f64, threshold: f64, counts: &[f64], excess: Robj) -> Result<f64> {
        let excess = severity_from_robj(&excess)?;
        self.inner
            .expected_claims(t, threshold, counts, &excess)
            .map_err(to_r)
    }

    #[allow(clippy::too_many_arguments)]
    fn count_test(
        &self,
        sizes: &[f64],
        threshold: f64,
        counts: &[f64],
        excess: Robj,
        exposure_years: f64,
        observed: &[f64],
    ) -> Result<List> {
        let excess = severity_from_robj(&excess)?;
        let c = self
            .inner
            .count_test(sizes, threshold, counts, &excess, exposure_years, observed)
            .map_err(to_r)?;
        Ok(list!(
            size = c.iter().map(|x| x.size).collect::<Vec<_>>(),
            expected = c.iter().map(|x| x.expected).collect::<Vec<_>>(),
            observed = c.iter().map(|x| x.observed as f64).collect::<Vec<_>>(),
            p_value = c.iter().map(|x| x.p_value).collect::<Vec<_>>()
        ))
    }
}

/// `list(threshold, alpha, standard_error, exceedances, censored)`.
#[extendr]
fn pareto_anchor_rust(threshold: f64, losses: &[f64], censored: &[f64]) -> Result<List> {
    let censored: Vec<bool> = censored.iter().map(|&c| c != 0.0).collect();
    let a = ParetoAnchor::fit(threshold, losses, &censored).map_err(to_r)?;
    Ok(list!(
        threshold = a.threshold,
        alpha = a.alpha,
        standard_error = a.standard_error,
        exceedances = a.exceedances as f64,
        censored = a.censored as f64
    ))
}

#[extendr]
fn poisson_tail_rust(expected: f64, observed: f64) -> Result<f64> {
    let k = crate::whole(observed, "observed")? as usize;
    prospicio_pricing::casualty::poisson_tail(expected, k).map_err(to_r)
}

extendr_module! {
    mod casualty;
    fn casualty_book_rust;
    impl CasualtyBook;
    fn pareto_anchor_rust;
    fn poisson_tail_rust;
}
