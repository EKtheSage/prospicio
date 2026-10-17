//! Casualty exposure rating over a table of policies
//! (`docs/design/casualty-exposure.md`).
//!
//! Each policy `i` pays `Y = s · min((X − a)+, l)` for a ground-up loss
//! `X` per occurrence: an underlying attachment `a` (0 for a primary
//! policy), a limit `l` and a share `s` of the layer it writes. A
//! per-policy treaty `R` xs `D` cedes `min((Y − D)+, R)`. The share of the
//! policy's expected loss that the treaty takes is its *exposure factor*,
//!
//! ```text
//!            LEV(a + min(l, (D + R)/s)) − LEV(a + min(l, D/s))
//! factor  =  -------------------------------------------------
//!                         LEV(a + l) − LEV(a)
//! ```
//!
//! Clark's three liability formulas (*Basics of Reinsurance Pricing*,
//! sections 3A(b) and 3B(a)) are its special cases: a primary policy
//! (`a = 0`), an umbrella over an underlying limit `UL` (`a = UL`), and
//! ALAE added on top of the capped loss ([`Alae::OnTop`], which divides `D`
//! and `R` by `1 + e`). With no policy limit (workers compensation) and
//! `a = 0` the factor is `ELF(D) − ELF(D + R)`.
//!
//! Two forms give the treaty's expected loss:
//!
//! - [`CasualtyProfile::rate`], the ILF form: each policy's expected loss
//!   `premium × ELR`, times its factor, on its own curve (a severity, an
//!   ILF table or an ELF table; [`CasualtyCurve`]).
//! - [`CasualtyProfile::rate_by_frequency`]: each policy's expected number
//!   of occurrences above a threshold `t`, times the expected ceded amount
//!   given `X > t` under a severity of the excess losses (a Pareto from
//!   `t`, say). The body of the curve below `t` never enters.

use std::sync::Arc;

use prospicio_core::{Error, Result};
use prospicio_prob::Severity;

use crate::layer::XsLayer;

/// How ALAE (allocated loss adjustment expense) enters the treaty. There
/// is no default: the treatment moves an excess layer materially.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alae {
    /// The treaty shares ALAE in proportion to loss: `e` is ALAE as a
    /// share of loss, and every ceded amount carries `1 + e`. The ELR is
    /// loss only; with an ELR that already holds ALAE, give `e = 0`
    /// (Clark's first case).
    ProRata(f64),
    /// The policy limit applies to loss only, ALAE is `e` times the capped
    /// loss, and the treaty's attachment and limit apply to loss plus ALAE
    /// (Clark's second formula: `D` and `R` divided by `1 + e`). The ELR
    /// is loss only, and the ceded amount carries `1 + e`. A treaty
    /// attaching at or above `(1 + e)` times the policy's top is not
    /// exposed.
    OnTop(f64),
    /// The curve is of loss plus ALAE, the limits apply to the sum, and
    /// the ELR is of loss plus ALAE.
    InSeverity,
}

impl Alae {
    fn loading(self) -> Result<f64> {
        match self {
            Alae::ProRata(e) | Alae::OnTop(e) => {
                if !e.is_finite() || e < 0.0 {
                    return Err(Error::InvalidParameter {
                        name: "alae",
                        value: e,
                        reason: "must be finite and non-negative",
                    });
                }
                Ok(1.0 + e)
            }
            Alae::InSeverity => Ok(1.0),
        }
    }

    /// The treaty's attachment and limit in the policy's loss units.
    fn treaty(self, treaty: &XsLayer) -> Result<(f64, f64)> {
        let (d, r) = (treaty.attachment(), treaty.limit());
        match self {
            Alae::OnTop(_) => {
                let k = self.loading()?;
                Ok((d / k, r / k))
            }
            _ => Ok((d, r)),
        }
    }
}

/// A table of increased limits factors, read as a limited expected value
/// curve up to a constant: `LEV(x) ∝ ILF(x)`.
///
/// It is linear between its limits and from the origin to its first limit
/// (`LEV(0) = 0`), and must be concave: the slopes may not increase. Beyond
/// its last limit it is an error rather than an extrapolation; for a
/// Pareto above the table, use the frequency form or a spliced severity.
#[derive(Debug, Clone, PartialEq)]
pub struct IlfTable {
    limits: Vec<f64>,
    factors: Vec<f64>,
}

impl IlfTable {
    /// Limits (finite, positive, increasing) and their factors (positive,
    /// increasing), at any basic limit.
    pub fn new(limits: &[f64], factors: &[f64]) -> Result<Self> {
        let (x, y) = check_table(limits, factors, "factors")?;
        check_concave(&x, &y)?;
        Ok(Self {
            limits: x,
            factors: y,
        })
    }

    /// The factor at `limit`, interpolated.
    pub fn ilf(&self, limit: f64) -> Result<f64> {
        interpolate(&self.limits, &self.factors, 0.0, limit, "ILF table")
    }
}

/// A table of excess loss factors, `ELF(x) = E[(X − x)+] / E[X]`, as
/// published for workers compensation, read as `LEV(x) / E[X] = 1 −
/// ELF(x)`.
///
/// `ELF(0) = 1`, and the table is linear between its limits and from 0 to
/// its first limit. `LEV` must stay concave (the ELF's slopes may not
/// increase in size). An unlimited policy has `LEV(∞) / E[X] = 1`; a finite
/// amount beyond the last limit is an error.
#[derive(Debug, Clone, PartialEq)]
pub struct ElfTable {
    limits: Vec<f64>,
    elfs: Vec<f64>,
}

impl ElfTable {
    /// Limits (finite, positive, increasing) and their excess loss factors
    /// (in `[0, 1]`, decreasing).
    pub fn new(limits: &[f64], elfs: &[f64]) -> Result<Self> {
        if let Some(&bad) = elfs.iter().find(|e| !(0.0..=1.0).contains(*e)) {
            return Err(Error::InvalidParameter {
                name: "elfs",
                value: bad,
                reason: "excess loss factors must lie in [0, 1]",
            });
        }
        let lev: Vec<f64> = elfs.iter().map(|e| 1.0 - e).collect();
        let (x, y) = check_table(limits, &lev, "elfs (decreasing)")?;
        check_concave(&x, &y)?;
        Ok(Self {
            limits: x,
            elfs: elfs.to_vec(),
        })
    }

    /// The excess loss factor at `limit`, interpolated (0 at `+inf`).
    pub fn elf(&self, limit: f64) -> Result<f64> {
        if limit == f64::INFINITY {
            return Ok(0.0);
        }
        interpolate(&self.limits, &self.elfs, 1.0, limit, "ELF table")
    }
}

fn check_table(x: &[f64], y: &[f64], name: &'static str) -> Result<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() || x.len() != y.len() {
        return Err(Error::Data(
            "a table needs as many values as limits, and at least one".into(),
        ));
    }
    let mut last = (0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y) {
        if !xi.is_finite() || xi <= last.0 {
            return Err(Error::InvalidParameter {
                name: "limits",
                value: xi,
                reason: "must be finite, positive and increasing",
            });
        }
        if !yi.is_finite() || yi <= last.1 {
            return Err(Error::InvalidParameter {
                name,
                value: yi,
                reason: "must be finite and increase with the limit",
            });
        }
        last = (xi, yi);
    }
    Ok((x.to_vec(), y.to_vec()))
}

/// The slopes from the origin through the points may not increase.
fn check_concave(x: &[f64], y: &[f64]) -> Result<()> {
    let mut prev = (0.0, 0.0, f64::INFINITY);
    for (&xi, &yi) in x.iter().zip(y) {
        let slope = (yi - prev.1) / (xi - prev.0);
        if slope > prev.2 * (1.0 + 1e-12) {
            return Err(Error::InvalidParameter {
                name: "limits",
                value: xi,
                reason: "the limited expected value must be concave: its slopes may not increase",
            });
        }
        prev = (xi, yi, slope);
    }
    Ok(())
}

/// Linear between the points, and from `(0, origin)` to the first.
fn interpolate(x: &[f64], y: &[f64], origin: f64, at: f64, what: &str) -> Result<f64> {
    if at.is_nan() || at < 0.0 {
        return Err(Error::InvalidParameter {
            name: "limit",
            value: at,
            reason: "must be non-negative",
        });
    }
    let last = *x.last().expect("a table is never empty");
    if at > last * (1.0 + 1e-12) {
        return Err(Error::Data(format!(
            "{at} is beyond the {what}'s last limit {last}; extend the table or use a severity"
        )));
    }
    let k = x.partition_point(|&xi| xi < at);
    if k < x.len() && x[k] <= at {
        return Ok(y[k]);
    }
    let (x0, y0) = if k == 0 {
        (0.0, origin)
    } else {
        (x[k - 1], y[k - 1])
    };
    let (x1, y1) = (x[k.min(x.len() - 1)], y[k.min(x.len() - 1)]);
    Ok(y0 + (y1 - y0) * (at - x0) / (x1 - x0))
}

/// The curve a policy is rated on: anything that gives `LEV` up to a
/// constant.
#[derive(Clone)]
pub enum CasualtyCurve {
    /// A ground-up severity (of loss, or of loss and ALAE for
    /// [`Alae::InSeverity`]).
    Severity(Arc<dyn Severity + Send + Sync>),
    Ilf(IlfTable),
    Elf(ElfTable),
}

impl std::fmt::Debug for CasualtyCurve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CasualtyCurve::Severity(_) => f.write_str("Severity(..)"),
            CasualtyCurve::Ilf(t) => f.debug_tuple("Ilf").field(t).finish(),
            CasualtyCurve::Elf(t) => f.debug_tuple("Elf").field(t).finish(),
        }
    }
}

impl CasualtyCurve {
    /// A severity curve.
    pub fn severity<S: Severity + Send + Sync + 'static>(severity: S) -> Self {
        CasualtyCurve::Severity(Arc::new(severity))
    }

    /// `LEV(x)` up to the curve's constant; `x = +inf` is the mean.
    fn lev(&self, x: f64) -> Result<f64> {
        match self {
            CasualtyCurve::Severity(s) => Ok(s.lev(x)),
            CasualtyCurve::Ilf(t) => {
                if x == f64::INFINITY {
                    return Err(Error::Data(
                        "an ILF table has no unlimited value; give the policy a limit".into(),
                    ));
                }
                t.ilf(x)
            }
            CasualtyCurve::Elf(t) => Ok(1.0 - t.elf(x)?),
        }
    }
}

/// One policy of a [`CasualtyProfile`].
#[derive(Debug, Clone, PartialEq)]
pub struct CasualtyPolicy {
    /// Underlying attachment, ground up (0 for a primary policy).
    pub attachment: f64,
    /// Policy limit above the attachment; `+inf` for none (workers
    /// compensation).
    pub limit: f64,
    /// Share of the layer the policy writes, in `(0, 1]`.
    pub share: f64,
    /// On-level subject premium.
    pub premium: f64,
    /// Expected loss ratio (loss only, unless [`Alae::InSeverity`]).
    pub elr: f64,
    /// Index of the policy's curve in the profile.
    pub curve: usize,
    /// Optional segment or line, for reporting and for leaving a segment
    /// out.
    pub tag: Option<String>,
}

impl CasualtyPolicy {
    /// A primary policy with the whole of its limit, on curve 0, untagged.
    pub fn new(limit: f64, premium: f64, elr: f64) -> Self {
        Self {
            attachment: 0.0,
            limit,
            share: 1.0,
            premium,
            elr,
            curve: 0,
            tag: None,
        }
    }

    pub fn attachment(mut self, attachment: f64) -> Self {
        self.attachment = attachment;
        self
    }

    pub fn share(mut self, share: f64) -> Self {
        self.share = share;
        self
    }

    pub fn curve(mut self, curve: usize) -> Self {
        self.curve = curve;
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// The ground-up band the treaty takes, `(bottom, top)`, for a treaty
    /// `d` xs `r` in the policy's loss units; empty when `bottom ≥ top`.
    fn band(&self, d: f64, r: f64) -> (f64, f64) {
        let s = self.share;
        let bottom = self.attachment + self.limit.min(d / s);
        let top = self.attachment + self.limit.min((d + r) / s);
        (bottom, top)
    }

    fn check(&self, i: usize, curves: usize) -> Result<()> {
        let bad =
            |name, value, reason| Err(Error::Data(format!("policy {i}: {name} {value} {reason}")));
        if !self.attachment.is_finite() || self.attachment < 0.0 {
            return bad(
                "attachment",
                self.attachment,
                "must be finite and non-negative",
            );
        }
        if self.limit.is_nan() || self.limit <= 0.0 {
            return bad("limit", self.limit, "must be positive (+inf for none)");
        }
        if !(self.share > 0.0 && self.share <= 1.0) {
            return bad("share", self.share, "must be in (0, 1]");
        }
        if !self.premium.is_finite() || self.premium < 0.0 {
            return bad("premium", self.premium, "must be finite and non-negative");
        }
        if !self.elr.is_finite() || self.elr < 0.0 {
            return bad("elr", self.elr, "must be finite and non-negative");
        }
        if self.curve >= curves {
            return bad("curve", self.curve as f64, "is not a curve of the profile");
        }
        Ok(())
    }
}

/// A book of casualty policies, each rated on its own curve.
///
/// ```
/// use prospicio_pricing::casualty::{Alae, CasualtyCurve, CasualtyPolicy, CasualtyProfile, ElfTable};
/// use prospicio_pricing::layer::XsLayer;
///
/// // Clark's workers compensation example: 750,000 xs 250,000.
/// let curve = ElfTable::new(&[250e3, 1e6], &[0.030, 0.006]).unwrap();
/// let book = CasualtyProfile::new(
///     vec![CasualtyCurve::Elf(curve)],
///     vec![CasualtyPolicy::new(f64::INFINITY, 100e3, 0.70)],
/// ).unwrap();
/// let r = book.rate(&XsLayer::new(750e3, 250e3).unwrap(), Alae::InSeverity).unwrap();
/// assert!((r.ceded() - 1_680.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct CasualtyProfile {
    curves: Vec<CasualtyCurve>,
    policies: Vec<CasualtyPolicy>,
}

impl CasualtyProfile {
    /// Fails on a policy out of range or naming a curve not given.
    pub fn new(curves: Vec<CasualtyCurve>, policies: Vec<CasualtyPolicy>) -> Result<Self> {
        if curves.is_empty() {
            return Err(Error::Data("a profile needs at least one curve".into()));
        }
        for (i, p) in policies.iter().enumerate() {
            p.check(i, curves.len())?;
        }
        Ok(Self { curves, policies })
    }

    pub fn policies(&self) -> &[CasualtyPolicy] {
        &self.policies
    }

    /// Each policy's exposure factor for `treaty`, as in the module
    /// documentation.
    pub fn exposure_factors(&self, treaty: &XsLayer, alae: Alae) -> Result<Vec<f64>> {
        let (d, r) = alae.treaty(treaty)?;
        self.policies
            .iter()
            .map(|p| {
                let (bottom, top) = p.band(d, r);
                if bottom >= top {
                    return Ok(0.0);
                }
                let c = &self.curves[p.curve];
                let below = c.lev(p.attachment)?;
                let policy = c.lev(p.attachment + p.limit)? - below;
                if policy <= 0.0 {
                    return Err(Error::Data(format!(
                        "a policy at {} with limit {} has no expected loss on its curve",
                        p.attachment, p.limit
                    )));
                }
                Ok((c.lev(top)? - c.lev(bottom)?) / policy)
            })
            .collect()
    }

    /// The ILF form: each policy's expected loss `premium × ELR` times its
    /// exposure factor, and the ALAE loading.
    pub fn rate(&self, treaty: &XsLayer, alae: Alae) -> Result<CasualtyRating> {
        let k = alae.loading()?;
        let factors = self.exposure_factors(treaty, alae)?;
        let rows = self
            .policies
            .iter()
            .zip(factors)
            .map(|(p, factor)| {
                let expected_loss = p.premium * p.elr;
                PolicyCession {
                    factor,
                    expected_loss,
                    ceded: expected_loss * factor * k,
                    premium: p.premium,
                    tag: p.tag.clone(),
                }
            })
            .collect();
        Ok(CasualtyRating { rows })
    }

    /// The frequency form: each policy's expected number of occurrences
    /// above `threshold` (`counts`, one per policy) times the expected
    /// ceded amount given `X > threshold`, under `excess`, a severity of
    /// the ground-up losses above the threshold (a Pareto from it, say).
    ///
    /// Fails when a policy exposed to the treaty would take losses below
    /// the threshold, which the counts do not see. The policies' ELRs and
    /// curves are not used; their premiums give the loss cost.
    pub fn rate_by_frequency<S: Severity + ?Sized>(
        &self,
        treaty: &XsLayer,
        alae: Alae,
        threshold: f64,
        counts: &[f64],
        excess: &S,
    ) -> Result<CasualtyRating> {
        if counts.len() != self.policies.len() {
            return Err(Error::Data(format!(
                "{} counts for {} policies",
                counts.len(),
                self.policies.len()
            )));
        }
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(Error::InvalidParameter {
                name: "threshold",
                value: threshold,
                reason: "must be finite and non-negative",
            });
        }
        let k = alae.loading()?;
        let (d, r) = alae.treaty(treaty)?;
        let mut rows = Vec::with_capacity(counts.len());
        for (i, (p, &n)) in self.policies.iter().zip(counts).enumerate() {
            if !n.is_finite() || n < 0.0 {
                return Err(Error::Data(format!(
                    "policy {i}: count {n} must be finite and non-negative"
                )));
            }
            let (bottom, top) = p.band(d, r);
            let per = if bottom >= top {
                0.0
            } else {
                if bottom < threshold * (1.0 - 1e-12) {
                    return Err(Error::Data(format!(
                        "policy {i}: the treaty takes its losses from {bottom}, below the \
                         threshold {threshold} the counts start at"
                    )));
                }
                p.share * (excess.lev(top) - excess.lev(bottom))
            };
            let ceded = n * per * k;
            rows.push(PolicyCession {
                factor: f64::NAN,
                expected_loss: f64::NAN,
                ceded,
                premium: p.premium,
                tag: p.tag.clone(),
            });
        }
        Ok(CasualtyRating { rows })
    }
}

/// One policy's part of a [`CasualtyRating`].
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyCession {
    /// Exposure factor (NaN in the frequency form).
    pub factor: f64,
    /// `premium × ELR` (NaN in the frequency form).
    pub expected_loss: f64,
    /// The treaty's expected loss from this policy, with ALAE.
    pub ceded: f64,
    pub premium: f64,
    pub tag: Option<String>,
}

/// The treaty's expected loss, kept policy by policy so that a segment can
/// be shown or left out without rating again.
#[derive(Debug, Clone, PartialEq)]
pub struct CasualtyRating {
    rows: Vec<PolicyCession>,
}

impl CasualtyRating {
    pub fn rows(&self) -> &[PolicyCession] {
        &self.rows
    }

    /// The treaty's expected loss.
    pub fn ceded(&self) -> f64 {
        self.rows.iter().map(|r| r.ceded).sum()
    }

    /// Subject premium.
    pub fn premium(&self) -> f64 {
        self.rows.iter().map(|r| r.premium).sum()
    }

    /// Expected treaty loss over subject premium.
    pub fn loss_cost(&self) -> f64 {
        self.ceded() / self.premium()
    }

    /// The rating with the policies tagged `tag` left out.
    pub fn without(&self, tag: &str) -> Self {
        Self {
            rows: self
                .rows
                .iter()
                .filter(|r| r.tag.as_deref() != Some(tag))
                .cloned()
                .collect(),
        }
    }

    /// Ceded loss and premium by tag, in first-seen order (untagged as
    /// `None`).
    pub fn by_tag(&self) -> Vec<(Option<String>, f64, f64)> {
        let mut out: Vec<(Option<String>, f64, f64)> = Vec::new();
        for r in &self.rows {
            match out.iter_mut().find(|(t, _, _)| *t == r.tag) {
                Some(e) => {
                    e.1 += r.ceded;
                    e.2 += r.premium;
                }
                None => out.push((r.tag.clone(), r.ceded, r.premium)),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Lognormal, Pareto};

    fn layer(r: f64, d: f64) -> XsLayer {
        XsLayer::new(r, d).unwrap()
    }

    #[test]
    fn tables_interpolate_from_the_origin_and_refuse_beyond() {
        let t = IlfTable::new(&[100e3, 1e6, 2e6], &[1.0, 1.6, 1.8]).unwrap();
        assert_eq!(t.ilf(1e6).unwrap(), 1.6);
        assert!((t.ilf(50e3).unwrap() - 0.5).abs() < 1e-15);
        assert!((t.ilf(1.5e6).unwrap() - 1.7).abs() < 1e-15);
        assert!(t.ilf(2.5e6).is_err());
        assert!(IlfTable::new(&[1e5, 1e6], &[1.0, 11.0]).is_err(), "convex");
        assert!(IlfTable::new(&[1e6, 1e5], &[1.0, 1.5]).is_err(), "unsorted");
        let e = ElfTable::new(&[250e3, 1e6], &[0.03, 0.006]).unwrap();
        assert_eq!(e.elf(250e3).unwrap(), 0.03);
        assert_eq!(e.elf(f64::INFINITY).unwrap(), 0.0);
        assert!(e.elf(2e6).is_err());
        assert!(ElfTable::new(&[1e5], &[1.2]).is_err());
    }

    /// Clark's three liability formulas are the general factor's special
    /// cases, on any severity.
    #[test]
    fn the_factor_reduces_to_clarks_formulas() {
        let sev = Lognormal::from_mean_cv(80e3, 4.0).unwrap();
        let curves = || vec![CasualtyCurve::severity(sev)];
        let lev = |x: f64| sev.lev(x);
        // Primary: [LEV(min(PL, AP + Lim)) − LEV(min(PL, AP))] / LEV(PL).
        let (pl, ap, lim) = (1e6, 250e3, 500e3);
        let p = CasualtyProfile::new(curves(), vec![CasualtyPolicy::new(pl, 1.0, 1.0)]).unwrap();
        let f = p
            .exposure_factors(&layer(lim, ap), Alae::InSeverity)
            .unwrap()[0];
        let want = (lev(pl.min(ap + lim)) - lev(pl.min(ap))) / lev(pl);
        assert!((f - want).abs() < 1e-14);
        // ALAE on top, e = 0.5: AP and AP + Lim divided by 1.5.
        let f = p
            .exposure_factors(&layer(lim, ap), Alae::OnTop(0.5))
            .unwrap()[0];
        let want = (lev(pl.min((ap + lim) / 1.5)) - lev(pl.min(ap / 1.5))) / lev(pl);
        assert!((f - want).abs() < 1e-14);
        // Clark: a 1M limit with a 1.5 loading leaves a treaty at 1.5M unexposed.
        let f = p
            .exposure_factors(&layer(1e6, 1.5e6), Alae::OnTop(0.5))
            .unwrap()[0];
        assert_eq!(f, 0.0);
        // Umbrella 1M xs 1M under 500K xs 500K:
        // [LEV(2M) − LEV(1.5M)] / [LEV(2M) − LEV(1M)].
        let u = CasualtyProfile::new(
            curves(),
            vec![CasualtyPolicy::new(1e6, 1.0, 1.0).attachment(1e6)],
        )
        .unwrap();
        let f = u
            .exposure_factors(&layer(500e3, 500e3), Alae::InSeverity)
            .unwrap()[0];
        let want = (lev(2e6) - lev(1.5e6)) / (lev(2e6) - lev(1e6));
        assert!((f - want).abs() < 1e-14);
    }

    /// One primary policy: the factor is the layer's expected loss over the
    /// policy's, as `layer` and `ilf` give it.
    #[test]
    fn one_policy_matches_layer_and_ilf() {
        let sev = Lognormal::from_mean_cv(80e3, 4.0).unwrap();
        let p = CasualtyProfile::new(
            vec![CasualtyCurve::severity(sev)],
            vec![CasualtyPolicy::new(2e6, 50e3, 0.65)],
        )
        .unwrap();
        let r = p.rate(&layer(1e6, 500e3), Alae::InSeverity).unwrap();
        let want = 50e3 * 0.65 * sev.layer(1e6, 500e3) / sev.lev(2e6);
        assert!((r.ceded() / want - 1.0).abs() < 1e-13);
        // The same through ILFs at a 100K basic limit.
        let ilf = |x| crate::layer::ilf(&sev, x, 1e5).unwrap();
        let by_ilf = 50e3 * 0.65 * (ilf(1.5e6) - ilf(5e5)) / ilf(2e6);
        assert!((r.ceded() / by_ilf - 1.0).abs() < 1e-13);
    }

    #[test]
    fn a_share_scales_the_treaty_into_the_policys_units() {
        // Half of 2M: the policy pays s · min(X, 2M), so 500K xs 250K of
        // its payments is 1M xs 500K ground up.
        let sev = Lognormal::from_mean_cv(80e3, 4.0).unwrap();
        let p = CasualtyProfile::new(
            vec![CasualtyCurve::severity(sev)],
            vec![CasualtyPolicy::new(2e6, 1.0, 1.0).share(0.5)],
        )
        .unwrap();
        let f = p
            .exposure_factors(&layer(500e3, 250e3), Alae::InSeverity)
            .unwrap()[0];
        let want = (sev.lev(1.5e6) - sev.lev(500e3)) / sev.lev(2e6);
        assert!((f - want).abs() < 1e-14);
    }

    /// For an excess policy above the threshold, the two forms agree when
    /// the counts carry the same expected loss.
    #[test]
    fn the_ilf_and_frequency_forms_agree_above_the_threshold() {
        let t = 1e6;
        let pareto = Pareto::new(t, 1.4).unwrap();
        let policy = CasualtyPolicy::new(5e6, 300e3, 0.6).attachment(2e6);
        let p = CasualtyProfile::new(vec![CasualtyCurve::severity(pareto)], vec![policy]).unwrap();
        let treaty = layer(2e6, 1e6);
        let ilf = p.rate(&treaty, Alae::ProRata(0.1)).unwrap();
        // E[policy loss | X > t] = LEV(7M) − LEV(2M), all of X > t here.
        let per = pareto.lev(7e6) - pareto.lev(2e6);
        let n = 300e3 * 0.6 / per;
        let freq = p
            .rate_by_frequency(&treaty, Alae::ProRata(0.1), t, &[n], &pareto)
            .unwrap();
        assert!((ilf.ceded() / freq.ceded() - 1.0).abs() < 1e-12);
        // A primary policy whose cession starts below the threshold is refused.
        let low = CasualtyProfile::new(
            vec![CasualtyCurve::severity(pareto)],
            vec![CasualtyPolicy::new(1e6, 1.0, 1.0)],
        )
        .unwrap();
        assert!(
            low.rate_by_frequency(&layer(500e3, 250e3), Alae::InSeverity, t, &[1.0], &pareto)
                .is_err()
        );
    }

    #[test]
    fn segments_can_be_left_out_and_grouped() {
        let sev = Lognormal::from_mean_cv(80e3, 4.0).unwrap();
        let p = CasualtyProfile::new(
            vec![CasualtyCurve::severity(sev)],
            vec![
                CasualtyPolicy::new(1e6, 100.0, 0.6).tag("wildfire"),
                CasualtyPolicy::new(2e6, 300.0, 0.6),
                CasualtyPolicy::new(1e6, 200.0, 0.6).tag("wildfire"),
            ],
        )
        .unwrap();
        let r = p.rate(&layer(500e3, 500e3), Alae::InSeverity).unwrap();
        let w = r.without("wildfire");
        assert_eq!(w.rows().len(), 1);
        assert!((w.ceded() - r.rows()[1].ceded).abs() < 1e-12);
        let g = r.by_tag();
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].0.as_deref(), Some("wildfire"));
        assert!((g[0].2 - 300.0).abs() < 1e-12);
        assert!((r.loss_cost() - r.ceded() / 600.0).abs() < 1e-15);
    }

    #[test]
    fn bad_input_is_refused() {
        let sev = Lognormal::from_mean_cv(80e3, 4.0).unwrap();
        let c = || vec![CasualtyCurve::severity(sev)];
        assert!(
            CasualtyProfile::new(c(), vec![CasualtyPolicy::new(1e6, 1.0, 1.0).share(0.0)]).is_err()
        );
        assert!(
            CasualtyProfile::new(c(), vec![CasualtyPolicy::new(1e6, 1.0, 1.0).curve(1)]).is_err()
        );
        assert!(CasualtyProfile::new(c(), vec![CasualtyPolicy::new(-1.0, 1.0, 1.0)]).is_err());
        assert!(CasualtyProfile::new(vec![], vec![]).is_err());
        let p = CasualtyProfile::new(c(), vec![CasualtyPolicy::new(1e6, 1.0, 1.0)]).unwrap();
        assert!(p.rate(&layer(1e6, 1e6), Alae::OnTop(-0.1)).is_err());
        // An ILF table has no unlimited value.
        let t = IlfTable::new(&[1e5, 1e6], &[1.0, 1.5]).unwrap();
        let p = CasualtyProfile::new(
            vec![CasualtyCurve::Ilf(t)],
            vec![CasualtyPolicy::new(f64::INFINITY, 1.0, 1.0)],
        )
        .unwrap();
        assert!(p.rate(&layer(1e5, 1e5), Alae::InSeverity).is_err());
    }
}
