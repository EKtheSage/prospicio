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
//!
//! [`CasualtyProfile::occurrences_above`] reads each policy's counts above
//! `t` off its own curve, [`ParetoAnchor`] fits the Pareto above `t` to
//! the trended claims with those closed at a limit censored, and
//! [`CasualtyProfile::sweep`] runs the frequency form over a range of
//! alphas. [`CasualtyProfile::count_test`] sets the claims expected at a
//! few sizes beside those observed. A per-policy rating cannot see one
//! occurrence hitting several policies (clash), nor an umbrella's
//! underlying aggregate exhausting; both are the caller's to load.

use std::sync::Arc;

use prospicio_core::{Error, Result};
use prospicio_prob::{Counting, LargeLosses, Pareto, Poisson, Severity};

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

/// The slope of the interpolation of [`interpolate`] just above `at`.
fn slope(x: &[f64], y: &[f64], origin: f64, at: f64, what: &str) -> Result<f64> {
    let last = *x.last().expect("a table is never empty");
    if at.is_nan() || at < 0.0 || at >= last {
        return Err(Error::Data(format!(
            "{at} is not below the {what}'s last limit {last}, where its slope ends"
        )));
    }
    let k = x.partition_point(|&xi| xi <= at);
    let (x0, y0) = if k == 0 {
        (0.0, origin)
    } else {
        (x[k - 1], y[k - 1])
    };
    Ok((y[k] - y0) / (x[k] - x0))
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

    /// `P(X > x)` times the same constant: the right derivative of the
    /// curve's `LEV`. A table's is its slope from `x` to the next limit, so
    /// it fails at or beyond the last.
    fn survival(&self, x: f64) -> Result<f64> {
        match self {
            CasualtyCurve::Severity(s) => Ok(s.survival(x)),
            CasualtyCurve::Ilf(t) => slope(&t.limits, &t.factors, 0.0, x, "ILF table"),
            CasualtyCurve::Elf(t) => Ok(-slope(&t.limits, &t.elfs, 1.0, x, "ELF table")?),
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

impl CasualtyProfile {
    /// Each policy's expected number of ground-up occurrences above `t` a
    /// year, from its own curve: the policy's expected loss over its
    /// expected payment per occurrence, `premium × ELR / (s · [LEV(a + l) −
    /// LEV(a)])`, times `P(X > t)`. These are the `counts` the frequency
    /// form takes, so that the body of each curve sets the frequency and
    /// a severity of the excess losses (the [`sweep`](Self::sweep)'s
    /// Pareto) the tail. A table's `P(X > t)` is its slope just above `t`,
    /// so `t` must be below its last limit; the curve's constant cancels.
    pub fn occurrences_above(&self, t: f64) -> Result<Vec<f64>> {
        if !t.is_finite() || t < 0.0 {
            return Err(Error::InvalidParameter {
                name: "t",
                value: t,
                reason: "must be finite and non-negative",
            });
        }
        self.policies
            .iter()
            .map(|p| {
                let c = &self.curves[p.curve];
                let per = p.share * (c.lev(p.attachment + p.limit)? - c.lev(p.attachment)?);
                if per <= 0.0 {
                    return Err(Error::Data(format!(
                        "a policy at {} with limit {} has no expected loss on its curve",
                        p.attachment, p.limit
                    )));
                }
                Ok(p.premium * p.elr / per * c.survival(t)?)
            })
            .collect()
    }

    /// The frequency form at each alpha of a Pareto from `threshold`:
    /// the treaty's expected loss when the losses above the threshold are
    /// `Pareto(threshold, α)`, the counts held fixed (decision 5 of the
    /// design note). The central alpha and its source are the caller's to
    /// state beside the result; the library ships no benchmark.
    ///
    /// ```
    /// use prospicio_pricing::casualty::{Alae, CasualtyCurve, CasualtyPolicy, CasualtyProfile};
    /// use prospicio_pricing::layer::XsLayer;
    /// use prospicio_prob::Pareto;
    ///
    /// // One 10M policy with one occurrence above 1M a year: 5M xs 5M.
    /// let book = CasualtyProfile::new(
    ///     vec![CasualtyCurve::severity(Pareto::new(1e6, 1.5).unwrap())],
    ///     vec![CasualtyPolicy::new(10e6, 1e6, 0.6)],
    /// ).unwrap();
    /// let treaty = XsLayer::new(5e6, 5e6).unwrap();
    /// let s = book.sweep(&treaty, Alae::InSeverity, 1e6, &[1.0], &[1.0, 2.0]).unwrap();
    /// // E[min((X − 5M)+, 5M)] for X ~ Pareto(1M, α): 1M ln 2 at α = 1.
    /// assert!((s[0].ceded - 1e6 * 2f64.ln()).abs() < 1e-6);
    /// assert!(s[0].ceded / s[1].ceded > 6.0); // about 7 apart
    /// ```
    pub fn sweep(
        &self,
        treaty: &XsLayer,
        alae: Alae,
        threshold: f64,
        counts: &[f64],
        alphas: &[f64],
    ) -> Result<Vec<SweepPoint>> {
        alphas
            .iter()
            .map(|&alpha| {
                let excess = Pareto::new(threshold, alpha)?;
                let r = self.rate_by_frequency(treaty, alae, threshold, counts, &excess)?;
                Ok(SweepPoint {
                    alpha,
                    ceded: r.ceded(),
                    loss_cost: r.loss_cost(),
                })
            })
            .collect()
    }

    /// The expected number of claims a year recorded at or above `t`, a
    /// ground-up size, under the frequency form (`threshold`, `counts`,
    /// `excess` as in [`rate_by_frequency`](Self::rate_by_frequency)).
    ///
    /// A policy records an occurrence only once it exceeds the attachment,
    /// and caps it at `a + l`, so it contributes `counts_i · S(max(t, a))`
    /// when `t ≤ a + l` and nothing above, `S` the survival of `excess`
    /// (1 at the threshold for a Pareto from it). The experience counted
    /// against it must be ground-up and capped the same way: the trended
    /// losses of the anchor, a claim closed at its limit counted at the
    /// limit.
    pub fn expected_claims<S: Severity + ?Sized>(
        &self,
        t: f64,
        threshold: f64,
        counts: &[f64],
        excess: &S,
    ) -> Result<f64> {
        if counts.len() != self.policies.len() {
            return Err(Error::Data(format!(
                "{} counts for {} policies",
                counts.len(),
                self.policies.len()
            )));
        }
        if !(t.is_finite() && t >= threshold * (1.0 - 1e-12)) {
            return Err(Error::Data(format!(
                "a count at {t} needs a size at or above the threshold {threshold}"
            )));
        }
        Ok(self
            .policies
            .iter()
            .zip(counts)
            .filter(|(p, _)| t <= (p.attachment + p.limit) * (1.0 + 1e-12))
            .map(|(p, &n)| n * excess.survival(t.max(p.attachment)))
            .sum())
    }

    /// Check 1 of the design note: at each threshold in `sizes`, the
    /// expected number of claims over the experience period set beside the
    /// number observed, with the Poisson probability of a count at least as
    /// extreme in the direction observed. `exposure_years` is the
    /// experience period in units of the profile's year (each year's
    /// on-level premium over the profile's, summed), and `observed` the
    /// trended ground-up losses (from
    /// [`Prepared::trended`](crate::experience::Prepared::trended)).
    ///
    /// Where the layer has never been reached, this tests the alpha more
    /// sharply than a burning cost of zero.
    pub fn count_test<S: Severity + ?Sized>(
        &self,
        sizes: &[f64],
        threshold: f64,
        counts: &[f64],
        excess: &S,
        exposure_years: f64,
        observed: &[f64],
    ) -> Result<Vec<CountCheck>> {
        if !(exposure_years.is_finite() && exposure_years > 0.0) {
            return Err(Error::InvalidParameter {
                name: "exposure_years",
                value: exposure_years,
                reason: "must be finite and positive",
            });
        }
        sizes
            .iter()
            .map(|&t| {
                let expected =
                    exposure_years * self.expected_claims(t, threshold, counts, excess)?;
                let n = observed.iter().filter(|&&y| y >= t).count();
                Ok(CountCheck {
                    size: t,
                    expected,
                    observed: n,
                    p_value: poisson_tail(expected, n)?,
                })
            })
            .collect()
    }
}

/// The probability under a Poisson with mean `expected` of a count at
/// least as far from it as `observed`, on the side observed: `P(N ≥ n)`
/// when `n ≥ expected`, `P(N ≤ n)` below it.
///
/// ```
/// use prospicio_pricing::casualty::poisson_tail;
///
/// // No claim where 3 were expected: e^-3.
/// assert!((poisson_tail(3.0, 0).unwrap() - (-3f64).exp()).abs() < 1e-15);
/// ```
pub fn poisson_tail(expected: f64, observed: usize) -> Result<f64> {
    let n = Poisson::new(expected)?;
    let k = observed as u64;
    Ok(if observed as f64 >= expected {
        if k == 0 {
            1.0
        } else {
            (1.0 - n.cdf(k - 1)).max(0.0)
        }
    } else {
        n.cdf(k)
    })
}

/// One alpha of a [`CasualtyProfile::sweep`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepPoint {
    pub alpha: f64,
    /// The treaty's expected loss.
    pub ceded: f64,
    /// Over the subject premium.
    pub loss_cost: f64,
}

/// One size of a [`CasualtyProfile::count_test`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CountCheck {
    /// The ground-up size counted at or above.
    pub size: f64,
    /// Expected claims over the experience period.
    pub expected: f64,
    pub observed: usize,
    /// [`poisson_tail`] of the observed count.
    pub p_value: f64,
}

/// The Pareto above a threshold fitted to the trended losses, with claims
/// closed at their policy limit censored (decision 4 of the design note):
/// `α = uncensored count / Σ ln(y / t)`, its standard error `α / √n`, and
/// the number of exceedances it rests on.
///
/// ```
/// use prospicio_pricing::casualty::ParetoAnchor;
///
/// let a = ParetoAnchor::fit(1e6, &[1.5e6, 2.5e6, 4e6, 2e6, 0.8e6], &[false, false, false, true, false])
///     .unwrap();
/// assert_eq!((a.exceedances, a.censored), (4, 1));
/// assert!((a.alpha - 3.0 / (1.5f64 * 2.5 * 4.0 * 2.0).ln()).abs() < 1e-14);
/// assert!((a.standard_error - a.alpha / 3f64.sqrt()).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParetoAnchor {
    pub threshold: f64,
    pub alpha: f64,
    pub standard_error: f64,
    /// Losses at or above the threshold.
    pub exceedances: usize,
    /// Of which censored at a policy limit.
    pub censored: usize,
}

impl ParetoAnchor {
    /// Fits [`Pareto::fit_with_standard_error`] to the `losses` at or above
    /// `threshold`, `censored[i]` where loss `i` was closed at its policy
    /// limit. Losses below the threshold are ignored.
    pub fn fit(threshold: f64, losses: &[f64], censored: &[bool]) -> Result<Self> {
        if losses.len() != censored.len() {
            return Err(Error::Data(format!(
                "{} censoring flags for {} losses",
                censored.len(),
                losses.len()
            )));
        }
        let (values, flags): (Vec<f64>, Vec<bool>) = losses
            .iter()
            .zip(censored)
            .filter(|(y, _)| **y >= threshold)
            .map(|(&y, &c)| (y, c))
            .unzip();
        if !flags.iter().any(|c| !c) {
            return Err(Error::Data(format!(
                "no uncensored loss at or above {threshold} to fit an alpha to"
            )));
        }
        let data = LargeLosses::new(values)?.censored(flags.clone())?;
        let (fit, se) = Pareto::fit_with_standard_error(threshold, &data, None)?;
        Ok(Self {
            threshold,
            alpha: fit.alpha(),
            standard_error: se,
            exceedances: flags.len(),
            censored: flags.iter().filter(|&&c| c).count(),
        })
    }

    /// The fitted alpha `± z` standard errors, the lower end kept above 0:
    /// a range for the [`sweep`](CasualtyProfile::sweep).
    pub fn interval(&self, z: f64) -> (f64, f64) {
        let lo = (self.alpha - z * self.standard_error).max(self.alpha * 1e-3);
        (lo, self.alpha + z * self.standard_error)
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
    use prospicio_prob::{Distribution, Lognormal, Pareto};

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

    /// The body of each curve sets the counts and a Pareto the tail: for a
    /// ground-up Pareto from the threshold, the frequency form on the
    /// counts from `occurrences_above` is the ILF form.
    #[test]
    fn occurrences_above_join_the_two_forms() {
        let sev = Pareto::new(1e6, 1.4).unwrap();
        let p = CasualtyProfile::new(
            vec![CasualtyCurve::severity(sev)],
            vec![
                CasualtyPolicy::new(5e6, 2e6, 0.65)
                    .attachment(5e6)
                    .share(0.5),
                CasualtyPolicy::new(10e6, 1e6, 0.7).attachment(10e6),
            ],
        )
        .unwrap();
        let counts = p.occurrences_above(1e6).unwrap();
        // premium × ELR / (s · layer) × S(1M) = premium × ELR / (s · layer).
        let want = 2e6 * 0.65 / (0.5 * sev.layer(5e6, 5e6));
        assert!((counts[0] / want - 1.0).abs() < 1e-14);
        let treaty = layer(2e6, 1e6);
        let ilf = p.rate(&treaty, Alae::InSeverity).unwrap();
        let freq = p
            .rate_by_frequency(&treaty, Alae::InSeverity, 1e6, &counts, &sev)
            .unwrap();
        assert!((ilf.ceded() / freq.ceded() - 1.0).abs() < 1e-12);
        // At α the sweep is the same frequency form.
        let s = p
            .sweep(&treaty, Alae::InSeverity, 1e6, &counts, &[1.4])
            .unwrap();
        assert!((s[0].ceded / ilf.ceded() - 1.0).abs() < 1e-12);
        assert!((s[0].loss_cost - s[0].ceded / 3e6).abs() < 1e-15);
    }

    /// A table's survival is its slope, and its constant cancels.
    #[test]
    fn a_tables_occurrences_are_its_slope() {
        let one = |factors: &[f64]| {
            let t = IlfTable::new(&[100e3, 1e6, 2e6], factors).unwrap();
            CasualtyProfile::new(
                vec![CasualtyCurve::Ilf(t)],
                vec![CasualtyPolicy::new(1e6, 1000.0, 0.6)],
            )
            .unwrap()
        };
        // LEV(1M) = 1.6 in table units; the slope above 500K is 0.6/900K.
        let n = one(&[1.0, 1.6, 1.8]).occurrences_above(500e3).unwrap()[0];
        assert!((n - 600.0 / 1.6 * 0.6 / 900e3).abs() < 1e-15);
        let m = one(&[2.0, 3.2, 3.6]).occurrences_above(500e3).unwrap()[0];
        assert!((m / n - 1.0).abs() < 1e-14);
        // At a limit, the slope above it; at the last, none.
        let k = one(&[1.0, 1.6, 1.8]).occurrences_above(1e6).unwrap()[0];
        assert!((k - 600.0 / 1.6 * 0.2 / 1e6).abs() < 1e-15);
        assert!(one(&[1.0, 1.6, 1.8]).occurrences_above(2e6).is_err());
        let e = ElfTable::new(&[250e3, 1e6], &[0.03, 0.006]).unwrap();
        let w = CasualtyProfile::new(
            vec![CasualtyCurve::Elf(e)],
            vec![CasualtyPolicy::new(f64::INFINITY, 100e3, 0.7)],
        )
        .unwrap();
        // ELF slope −0.024 / 750K; the policy's LEV range is 1.
        let n = w.occurrences_above(500e3).unwrap()[0];
        assert!((n - 70e3 * 0.024 / 750e3).abs() < 1e-15);
    }

    #[test]
    fn expected_claims_respect_attachments_and_limits() {
        let sev = Lognormal::from_mean_cv(80e3, 4.0).unwrap();
        let p = CasualtyProfile::new(
            vec![CasualtyCurve::severity(sev)],
            vec![
                CasualtyPolicy::new(1e6, 1.0, 1.0),
                CasualtyPolicy::new(5e6, 1.0, 1.0).attachment(2e6),
            ],
        )
        .unwrap();
        let x = Pareto::new(500e3, 1.5).unwrap();
        let c = [3.0, 2.0];
        let at = |t| p.expected_claims(t, 500e3, &c, &x).unwrap();
        // At 800K both record, the excess one only above its 2M attachment.
        let want = 3.0 * x.survival(800e3) + 2.0 * x.survival(2e6);
        assert!((at(800e3) - want).abs() < 1e-14);
        // At its 1M limit the primary policy still records (capped there).
        assert!((at(1e6) - (3.0 * x.survival(1e6) + 2.0 * x.survival(2e6))).abs() < 1e-14);
        assert!((at(3e6) - 2.0 * x.survival(3e6)).abs() < 1e-14);
        assert_eq!(at(8e6), 0.0);
        assert!(p.expected_claims(400e3, 500e3, &c, &x).is_err());
        // Over five years, against five losses from 1M up.
        let checks = p
            .count_test(&[1e6, 3e6], 500e3, &c, &x, 5.0, &[1.2e6, 3.5e6, 0.9e6])
            .unwrap();
        assert_eq!((checks[0].observed, checks[1].observed), (2, 1));
        let lambda = 5.0 * at(1e6);
        assert!((checks[0].expected - lambda).abs() < 1e-12);
        let below: f64 = (0..=2)
            .map(|k| (-lambda).exp() * lambda.powi(k) / [1.0, 1.0, 2.0][k as usize])
            .sum();
        assert!(lambda > 2.0);
        assert!((checks[0].p_value - below).abs() < 1e-14);
    }

    #[test]
    fn the_poisson_tail_is_one_sided_toward_the_observation() {
        let at_least = 1.0 - (-2f64).exp() * (1.0 + 2.0 + 2.0 + 4.0 / 3.0);
        assert!((poisson_tail(2.0, 4).unwrap() - at_least).abs() < 1e-15);
        assert!((poisson_tail(2.0, 0).unwrap() - (-2f64).exp()).abs() < 1e-15);
        assert_eq!(poisson_tail(0.0, 0).unwrap(), 1.0);
        assert!(poisson_tail(-1.0, 0).is_err());
    }

    #[test]
    fn the_anchor_is_the_censored_fit() {
        let y = [1.1e6, 1.7e6, 2.4e6, 2.0e6, 3.1e6, 0.6e6];
        let c = [false, false, false, true, false, true];
        let a = ParetoAnchor::fit(1e6, &y, &c).unwrap();
        let data = LargeLosses::new(y[..5].to_vec())
            .unwrap()
            .censored(c[..5].to_vec())
            .unwrap();
        let (fit, se) = Pareto::fit_with_standard_error(1e6, &data, None).unwrap();
        assert_eq!(a.alpha, fit.alpha());
        assert_eq!(a.standard_error, se);
        assert_eq!((a.exceedances, a.censored), (5, 1));
        let (lo, hi) = a.interval(1.0);
        assert!((hi - lo - 2.0 * se).abs() < 1e-12);
        assert!(ParetoAnchor::fit(1e6, &[2e6], &[true]).is_err());
        assert!(ParetoAnchor::fit(1e6, &[2e6], &[]).is_err());
    }
}
