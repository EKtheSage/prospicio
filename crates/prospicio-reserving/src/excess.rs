//! Excess-layer loss development from a ground-up triangle
//! (`docs/design/reserving-excess.md`, PRs 1 and 2).
//!
//! A layer's own triangle is sparse and unstable. The ground-up triangle is
//! dense, and a severity model says how much of its development lands in a
//! layer: [`ExcessPattern::from_ground_up`] turns the ground-up factors, an
//! optional claim count column and the ultimate severity into the layer's
//! age-to-age factors, which then go into any method as a
//! [`SelectedPattern`].
//!
//! The model is the scale shift (decision 1): the claims reported at an age
//! are the ultimate ones divided by a shift `c_j`, `X_j = X_inf / c_j`, with
//! one `c_j` per age identified by the ground-up triangle. With the counts
//! `n_j`, the layer `l xs a` (`b = a + l`) has the expected loss
//! `n_j (LEV_inf(c_j b) - LEV_inf(c_j a)) / c_j` at age `j`, so its factor to
//! ultimate is
//!
//! ```text
//! CDF_layer(j) = [n_inf / n_j] (LEV_inf(b) - LEV_inf(a)) c_j
//!                / (LEV_inf(c_j b) - LEV_inf(c_j a)).
//! ```
//!
//! Pinto and Gogol's shape drift ([`SeverityByMaturity::ParetoShape`]) lets
//! the Pareto shape above a threshold fall with maturity instead, from a
//! base development (the counts above the threshold or a lower layer's
//! losses). [`ExcessPattern::credibility_weights`] gives the weight on the
//! derived pattern in a blend with the layer's own factors, `1 - n / (n +
//! k)`, capped so that one origin's link ratio cannot dominate.
//!
//! Basis, trend and layers follow the note's decisions: incurred losses only,
//! a severity the caller has trended to the origins' price level, and one
//! layer for every origin.

use prospicio_core::Lag;
use prospicio_math::roots::bisect_log;
use prospicio_prob::{Pareto, Severity};

use crate::development::{Development, cumulative_factors};
use crate::error::{Error, Result};
use crate::pattern::{PatternKind, PatternWeight, SelectedPattern};
use crate::tail::Tail;
use crate::triangle::Triangle;

/// An excess layer `limit xs attachment`, the same for every origin.
///
/// The limit may be `f64::INFINITY` for an unlimited layer
/// (`attachment` and above); the attachment is 0 for a limited layer from
/// the ground up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layer {
    pub attachment: f64,
    pub limit: f64,
}

impl Layer {
    /// The layer `limit xs attachment`: the attachment finite and not
    /// negative, the limit positive (infinite for an unlimited layer).
    pub fn new(attachment: f64, limit: f64) -> Result<Self> {
        let layer = Self { attachment, limit };
        layer.check()?;
        Ok(layer)
    }

    /// The unlimited layer above `attachment`.
    pub fn unlimited(attachment: f64) -> Result<Self> {
        Self::new(attachment, f64::INFINITY)
    }

    /// The top of the layer, `attachment + limit` (infinite when unlimited).
    pub fn exhaustion(&self) -> f64 {
        self.attachment + self.limit
    }

    fn check(&self) -> Result<()> {
        if !(self.attachment.is_finite() && self.attachment >= 0.0) {
            return Err(Error::InvalidSetting {
                name: "attachment",
                value: self.attachment,
                expected: "finite and not negative",
            });
        }
        if self.limit.is_nan() || self.limit <= 0.0 {
            return Err(Error::InvalidSetting {
                name: "limit",
                value: self.limit,
                expected: "positive, or infinite for an unlimited layer",
            });
        }
        Ok(())
    }
}

/// The loss basis of the ground-up triangle.
///
/// Only [`Incurred`](Self::Incurred) is derived (decision 9): the factor
/// scales the sizes of reported claims, case reserves included. A paid
/// layer pattern needs a payment-timing assumption by claim size and is a
/// later option, not an impossible one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Reported (paid plus case reserves) losses.
    Incurred,
    /// Paid losses; rejected with an error.
    Paid,
}

/// The development a [`SeverityByMaturity::ParetoShape`] model starts from
/// (decision 16): what the triangle column of
/// [`ExcessPattern::from_ground_up`] holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeBase {
    /// The number of claims above the threshold: Pinto and Gogol's `n_j /
    /// n_i`.
    Counts,
    /// The losses in this layer, which attaches at or above the threshold:
    /// Pinto and Gogol's development `d` of the losses excess of the
    /// threshold (`Layer::unlimited(threshold)`), or Bear's development of a
    /// lower bounded layer.
    Losses(Layer),
}

/// How the severity at each maturity relates to the ultimate severity.
///
/// The scale shift (decision 1, the default) is identified by the ground-up
/// triangle alone; the Pareto shape drift of Pinto and Gogol needs the
/// shapes by maturity and a base development.
#[non_exhaustive]
pub enum SeverityByMaturity {
    /// `X_inf = c_j X_j`, with `c_j` solved from the ground-up triangle.
    ScaleShift {
        /// The ultimate severity `S_inf`, at the price level of the layer's
        /// origins (incurred basis, trended by the caller).
        ultimate: Box<dyn Severity>,
        /// The ground-up column is limited per claim at this amount (a
        /// capped triangle, as in Sahasrabuddhe's limit of 1m); `None` for
        /// an unlimited column, where `c_j` is the ground-up factor over the
        /// count factor in closed form. With a cap `c_j` solves
        /// `c / LEV(c u) = r_j / LEV(u)` numerically.
        ground_up_cap: Option<f64>,
    },
    /// Pinto and Gogol's shape drift: the claims above `threshold` follow a
    /// single-parameter Pareto whose shape moves with maturity, `alpha_j` at
    /// age `j` and `ultimate_alpha` at ultimate. With the base development
    /// `D_j` to ultimate (of the counts above the threshold, or of the
    /// losses in a base layer `B`) and `g_q(L)` the expected loss to layer
    /// `L` of one claim from `Pareto(threshold, q)` (1 for counts), the
    /// layer `A` develops to ultimate by
    ///
    /// ```text
    /// CDF_A(j) = D_j [g_alpha_j(B) / g_ultimate(B)] [g_ultimate(A) / g_alpha_j(A)],
    /// ```
    ///
    /// which is `(n_j / n_i) ((q_i - 1) / (q_j - 1)) c^(q_i - q_j)` for the
    /// layer excess of `c` times the threshold (Pinto and Gogol, PCAS 74,
    /// p. 246) and Bear's `d c^(q_i - q_j)` for layers with equal
    /// limit-to-attachment ratios (PCAS 79, p. 137).
    ParetoShape {
        /// The lower bound `k` of the Pareto tail; the layer and a base
        /// layer attach at or above it.
        threshold: f64,
        /// `(age, alpha)` at every development age of the column.
        alphas: Vec<(Lag, f64)>,
        /// The shape of the fully developed claims.
        ultimate_alpha: f64,
        /// What the triangle column holds.
        base: ShapeBase,
    },
}

impl SeverityByMaturity {
    /// Pinto and Gogol's shape drift above `threshold`, with the shapes by
    /// age, the ultimate shape and the base development the column holds.
    pub fn pareto_shape(
        threshold: f64,
        alphas: Vec<(Lag, f64)>,
        ultimate_alpha: f64,
        base: ShapeBase,
    ) -> Self {
        Self::ParetoShape {
            threshold,
            alphas,
            ultimate_alpha,
            base,
        }
    }

    /// The scale shift on an unlimited ground-up column.
    pub fn scale_shift(ultimate: Box<dyn Severity>) -> Self {
        Self::ScaleShift {
            ultimate,
            ground_up_cap: None,
        }
    }

    /// The scale shift on a ground-up column limited per claim at `cap`.
    pub fn scale_shift_capped(ultimate: Box<dyn Severity>, cap: f64) -> Self {
        Self::ScaleShift {
            ultimate,
            ground_up_cap: Some(cap),
        }
    }
}

impl std::fmt::Debug for SeverityByMaturity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScaleShift {
                ultimate,
                ground_up_cap,
            } => f
                .debug_struct("ScaleShift")
                .field("mean", &ultimate.mean())
                .field("ground_up_cap", ground_up_cap)
                .finish(),
            Self::ParetoShape {
                threshold,
                alphas,
                ultimate_alpha,
                base,
            } => f
                .debug_struct("ParetoShape")
                .field("threshold", threshold)
                .field("alphas", alphas)
                .field("ultimate_alpha", ultimate_alpha)
                .field("base", base)
                .finish(),
        }
    }
}

/// A layer's development pattern derived from the ground-up triangle.
///
/// ```
/// use prospicio_prob::{Grid, Severity};
/// use prospicio_reserving::excess::{Basis, ExcessPattern, Layer, SeverityByMaturity};
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Tail, Triangle};
///
/// // Bloom and Patrik (2013), section 3: four claims, ground-up factors
/// // 3.60 and 1.25, a deductible of 350.
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &[Month::january(2020); 3],
///     development: DevelopmentColumn::Age(&[12, 24, 36]),
///     values: &[("incurred", &[1225.0, 4410.0, 5512.5])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })
/// .unwrap();
/// // The ultimate claims 450, 1687.5, 1125 and 2250 sit at points of a grid of step 112.5.
/// let mut probs = vec![0.0; 21];
/// for j in [4, 15, 10, 20] {
///     probs[j] = 0.25;
/// }
/// let ultimate = Grid::new(112.5, probs).unwrap();
/// let pattern = ExcessPattern::from_ground_up(
///     &tri,
///     "incurred",
///     Basis::Incurred,
///     None,
///     &Tail::default(),
///     Layer::unlimited(350.0).unwrap(),
///     &SeverityByMaturity::scale_shift(Box::new(ultimate)),
/// )
/// .unwrap();
/// assert!((pattern.layer_cdf[0] - 23.5).abs() < 1e-12);
/// assert!((pattern.layer_ldf[0] - 17.2).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExcessPattern {
    /// Development ages (months) of the ground-up triangle's segment.
    pub ages: Vec<Lag>,
    /// The layer's age-to-age factors, one fewer than the ages.
    pub layer_ldf: Vec<f64>,
    /// The layer's factors from each age to ultimate, with the tail:
    /// `layer_cdf[k] = layer_tail * prod(layer_ldf[k..])`.
    pub layer_cdf: Vec<f64>,
    /// The ground-up age-to-age factors the pattern was derived from (the
    /// ones a chain ladder with the same tail would use), the check. Under
    /// [`SeverityByMaturity::ParetoShape`] they are the base column's.
    pub ground_up_ldf: Vec<f64>,
    /// The ground-up factor from the oldest age to ultimate (the base
    /// column's under [`SeverityByMaturity::ParetoShape`]).
    pub ground_up_tail: f64,
    /// The scale shift `c_j` of each age (the last for the tail), the
    /// factor by which the severity at the age grows to ultimate; empty
    /// under [`SeverityByMaturity::ParetoShape`], whose shapes are inputs.
    pub shifts: Vec<f64>,
    /// The layer's factor from the oldest age to ultimate, derived from the
    /// ground-up tail; 1 without a tail.
    pub layer_tail: f64,
}

impl ExcessPattern {
    /// The layer pattern from the ground-up column, an optional claim count
    /// column and the ground-up tail.
    ///
    /// * `column` is a cumulative or incremental reported (incurred) loss
    ///   column of a single-segment triangle; `basis` must be
    ///   [`Basis::Incurred`].
    /// * `counts` is a reported claim count column on the same ages. Without
    ///   it the counts are taken as complete at every age, which mixes
    ///   frequency development into the severity shift (decision 3).
    ///   Counts are taken as complete at the oldest age: the count tail is 1.
    /// * `tail` is the ground-up tail of the chain ladder; it fixes one more
    ///   shift, for the oldest age, and the layer's tail factor. The default
    ///   `Tail` is no tail, and `layer_tail` is 1.
    /// * The ground-up factors are the volume-weighted ones
    ///   (`Development::default()`).
    ///
    /// Under [`SeverityByMaturity::ParetoShape`] the column is the base
    /// development the model names ([`ShapeBase`]: the counts above the
    /// threshold, or the losses in a base layer), `counts` must be `None`,
    /// and `tail` is the base column's tail; the layer's tail factor then
    /// also carries the shape's move from the oldest age's alpha to the
    /// ultimate one, so it is not 1 when they differ, even without a tail.
    ///
    /// The layer must have positive expected loss in the model at every age.
    pub fn from_ground_up(
        triangle: &Triangle,
        column: &str,
        basis: Basis,
        counts: Option<&str>,
        tail: &Tail,
        layer: Layer,
        model: &SeverityByMaturity,
    ) -> Result<Self> {
        if basis == Basis::Paid {
            return Err(Error::NotSupported {
                setting: "a paid basis",
                method: "ExcessPattern::from_ground_up",
                reason: "the derived factor scales the sizes of reported claims; a paid layer \
                         pattern needs a payment-timing assumption by claim size",
            });
        }
        layer.check()?;
        match model {
            SeverityByMaturity::ScaleShift {
                ultimate,
                ground_up_cap,
            } => Self::scale_shift(
                triangle,
                column,
                counts,
                tail,
                layer,
                ultimate.as_ref(),
                *ground_up_cap,
            ),
            SeverityByMaturity::ParetoShape {
                threshold,
                alphas,
                ultimate_alpha,
                base,
            } => {
                if counts.is_some() {
                    return Err(Error::Excess(
                        "a Pareto shape model reads its base development from `column`; pass no \
                         count column"
                            .into(),
                    ));
                }
                Self::pareto_shape(
                    triangle,
                    column,
                    tail,
                    layer,
                    *threshold,
                    alphas,
                    *ultimate_alpha,
                    *base,
                )
            }
        }
    }

    fn scale_shift(
        triangle: &Triangle,
        column: &str,
        counts: Option<&str>,
        tail: &Tail,
        layer: Layer,
        ultimate: &dyn Severity,
        ground_up_cap: Option<f64>,
    ) -> Result<Self> {
        if let Some(cap) = ground_up_cap
            && !(cap.is_finite() && cap > 0.0)
        {
            return Err(Error::InvalidSetting {
                name: "ground_up_cap",
                value: cap,
                expected: "finite and positive",
            });
        }
        let mean = ultimate.mean();
        if !(mean.is_finite() && mean > 0.0) {
            return Err(Error::Excess(
                "the ultimate severity needs a finite, positive mean".into(),
            ));
        }

        // A single segment: layers that differ by origin or line are
        // fitted as segments of their own (decision 6).
        let segment = triangle.segment(column)?;
        let dev = Development::default().fit(triangle, column)?;
        let n_links = dev.ldf.len();
        let tail_fit = tail.fit(&dev)?;
        let ground_up_ldf = tail_fit.ldf[..n_links].to_vec();
        let gu_cdf = cumulative_factors(&ground_up_ldf, tail_fit.factor);

        let count_cdf = match counts {
            Some(name) => {
                let count_segment = triangle.segment(name)?;
                if count_segment.ages != segment.ages {
                    return Err(Error::Excess(
                        "the count column must be observed at the same development ages".into(),
                    ));
                }
                let cdev = Development::default().fit(triangle, name)?;
                cumulative_factors(&cdev.ldf, 1.0)
            }
            None => vec![1.0; n_links + 1],
        };

        let lev = |x: f64| ultimate.lev(x);
        let layer_ultimate = lev(layer.exhaustion()) - lev(layer.attachment);
        if !(layer_ultimate.is_finite() && layer_ultimate > 0.0) {
            return Err(Error::Excess(
                "the layer has no expected loss at ultimate under the severity".into(),
            ));
        }
        let mut shifts = Vec::with_capacity(n_links + 1);
        let mut layer_cdf = Vec::with_capacity(n_links + 1);
        for age in 0..=n_links {
            // The ground-up severity factor: the ground-up factor over the
            // factor of the claim count.
            let r = gu_cdf[age] / count_cdf[age];
            if !(r.is_finite() && r > 0.0) {
                return Err(Error::Excess(format!(
                    "the ground-up severity factor at age {} is not finite and positive",
                    segment.ages[age]
                )));
            }
            let c = match ground_up_cap {
                None => r,
                Some(cap) => solve_shift(&lev, cap, r, segment.ages[age])?,
            };
            let at_age = layer_at_shift(ultimate, layer, c);
            if !(at_age.is_finite() && at_age > 0.0) {
                return Err(Error::Excess(format!(
                    "the layer has no expected loss at age {} under the severity",
                    segment.ages[age]
                )));
            }
            shifts.push(c);
            layer_cdf.push(count_cdf[age] * layer_ultimate / at_age);
        }
        let layer_ldf = (0..n_links)
            .map(|k| layer_cdf[k] / layer_cdf[k + 1])
            .collect();
        Ok(Self {
            ages: segment.ages.clone(),
            layer_ldf,
            layer_tail: layer_cdf[n_links],
            layer_cdf,
            ground_up_ldf,
            ground_up_tail: tail_fit.factor,
            shifts,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn pareto_shape(
        triangle: &Triangle,
        column: &str,
        tail: &Tail,
        layer: Layer,
        threshold: f64,
        alphas: &[(Lag, f64)],
        ultimate_alpha: f64,
        base: ShapeBase,
    ) -> Result<Self> {
        if !(threshold.is_finite() && threshold > 0.0) {
            return Err(Error::InvalidSetting {
                name: "threshold",
                value: threshold,
                expected: "finite and positive",
            });
        }
        let check_alpha = |alpha: f64| {
            if alpha.is_finite() && alpha > 0.0 {
                Ok(alpha)
            } else {
                Err(Error::InvalidSetting {
                    name: "alpha",
                    value: alpha,
                    expected: "finite and positive",
                })
            }
        };
        check_alpha(ultimate_alpha)?;
        let below = |l: Layer, what: &str| {
            if l.attachment < threshold {
                Err(Error::Excess(format!(
                    "the {what} attaches at {}, below the Pareto threshold {threshold}",
                    l.attachment
                )))
            } else {
                Ok(())
            }
        };
        below(layer, "layer")?;
        let base_layer = match base {
            ShapeBase::Counts => None,
            ShapeBase::Losses(b) => {
                b.check()?;
                below(b, "base layer")?;
                Some(b)
            }
        };

        let segment = triangle.segment(column)?;
        let ages = &segment.ages;
        let mut by_age: Vec<Option<f64>> = vec![None; ages.len()];
        for &(age, alpha) in alphas {
            let Some(p) = ages.iter().position(|&a| a == age) else {
                return Err(Error::Excess(format!(
                    "an alpha is given at age {age}, which is not a development age of the column"
                )));
            };
            if by_age[p].is_some() {
                return Err(Error::Excess(format!(
                    "the alpha at age {age} is given twice"
                )));
            }
            by_age[p] = Some(check_alpha(alpha)?);
        }
        let shapes = by_age
            .iter()
            .zip(ages)
            .map(|(alpha, age)| {
                alpha.ok_or_else(|| {
                    Error::Excess(format!(
                        "no alpha at age {age}: the shape drift needs one at every development age"
                    ))
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        let dev = Development::default().fit(triangle, column)?;
        let n_links = dev.ldf.len();
        let tail_fit = tail.fit(&dev)?;
        let base_ldf = tail_fit.ldf[..n_links].to_vec();
        let base_cdf = cumulative_factors(&base_ldf, tail_fit.factor);

        // The expected loss to `l` of one claim above the threshold.
        let per_claim = |alpha: f64, l: Layer| -> Result<f64> {
            let pareto = Pareto::new(threshold, alpha).map_err(|e| Error::Excess(e.to_string()))?;
            let v = pareto.layer(l.limit, l.attachment);
            if v.is_finite() && v > 0.0 {
                Ok(v)
            } else {
                Err(Error::Excess(format!(
                    "the layer {} xs {} has no finite, positive expected loss at alpha {alpha}: \
                     an unlimited layer needs every alpha above 1",
                    l.limit, l.attachment
                )))
            }
        };
        let base_at = |alpha: f64| base_layer.map_or(Ok(1.0), |b| per_claim(alpha, b));
        let layer_ultimate = per_claim(ultimate_alpha, layer)?;
        let base_ultimate = base_at(ultimate_alpha)?;
        let mut layer_cdf = Vec::with_capacity(n_links + 1);
        for (age, &alpha) in shapes.iter().enumerate().take(n_links + 1) {
            let cdf = base_cdf[age]
                * (base_at(alpha)? / base_ultimate)
                * (layer_ultimate / per_claim(alpha, layer)?);
            if !(cdf.is_finite() && cdf > 0.0) {
                return Err(Error::Excess(format!(
                    "the base development at age {} is not finite and positive",
                    ages[age]
                )));
            }
            layer_cdf.push(cdf);
        }
        let layer_ldf = (0..n_links)
            .map(|k| layer_cdf[k] / layer_cdf[k + 1])
            .collect();
        Ok(Self {
            ages: ages.clone(),
            layer_ldf,
            layer_tail: layer_cdf[n_links],
            layer_cdf,
            ground_up_ldf: base_ldf,
            ground_up_tail: tail_fit.factor,
            shifts: Vec::new(),
        })
    }

    /// The weight on this derived pattern at each age for a blend with the
    /// layer triangle's own factors (decision 4 (a)): `w_j = 1 - z_j`, with
    /// the credibility of the layer's own factor
    ///
    /// ```text
    /// z_j = n_j / (n_j + k),   capped as below,
    /// ```
    ///
    /// for [`selected`](Self::selected), which blends `w_j * derived + (1 -
    /// w_j) * own`. User weights stay available through
    /// [`PatternWeight`] directly.
    ///
    /// * `n_j` measures the information in the link from age `j`: with a
    ///   `counts` column (the number of claims in the layer, cumulative or
    ///   incremental like the losses), the claims in the layer at age `j`
    ///   summed over the origins observed at both ages; without one, the
    ///   number of those origins whose layer value at `j` is positive.
    /// * `k` is the credibility constant, required with no default
    ///   (decision 11 (a)); finite and positive.
    /// * **The cap.** `n / (n + k)` ignores how variable the link ratios
    ///   are, so `z_j` is also capped so that removing any one origin moves
    ///   the blended factor by at most `max_influence` times the blended
    ///   factor: `z_j |f_own - f_own(-i)| <= max_influence * f_blend(z_j)`
    ///   for every origin `i`, where `f_own` is the layer's volume-weighted
    ///   factor and `f_own(-i)` the same without origin `i` (the derived
    ///   factor when no other origin has a positive value at `j`). One huge
    ///   ratio from a small earlier value therefore gets little weight.
    /// * Where the layer's own factor cannot be estimated (every earlier
    ///   value zero) `z_j = 0` and the weight is exactly 1, whatever the
    ///   counts say, so `Development` uses its sparse-age path there
    ///   (decision 8 (b)): the own factor is reported missing and the
    ///   derived one is used.
    ///
    /// The layer triangle must be a single segment observed at this
    /// pattern's ages. The own factor here is the volume-weighted one
    /// (`Development::default()`), as the chain ladder of the blend uses by
    /// default.
    pub fn credibility_weights(
        &self,
        layer_triangle: &Triangle,
        column: &str,
        counts: Option<&str>,
        k: f64,
        max_influence: f64,
    ) -> Result<PatternWeight> {
        if !(k.is_finite() && k > 0.0) {
            return Err(Error::InvalidSetting {
                name: "k",
                value: k,
                expected: "finite and positive",
            });
        }
        if !(max_influence.is_finite() && max_influence > 0.0) {
            return Err(Error::InvalidSetting {
                name: "max_influence",
                value: max_influence,
                expected: "finite and positive",
            });
        }
        let segment = layer_triangle.segment(column)?;
        if segment.ages != self.ages {
            return Err(Error::Excess(
                "the layer triangle must be observed at the derived pattern's development ages"
                    .into(),
            ));
        }
        let count_segment = match counts {
            Some(name) => {
                let s = layer_triangle.segment(name)?;
                if s.ages != segment.ages || s.origins != segment.origins {
                    return Err(Error::Excess(
                        "the layer count column must be observed at the same origins and ages"
                            .into(),
                    ));
                }
                Some(s)
            }
            None => None,
        };

        let mut weights = Vec::with_capacity(self.layer_ldf.len());
        for (j, &derived) in self.layer_ldf.iter().enumerate() {
            // The origins observed at both ages of the link.
            let pairs: Vec<(usize, f64, f64)> = (0..segment.n_origins)
                .filter_map(|o| Some((o, segment.get(o, j)?, segment.get(o, j + 1)?)))
                .collect();
            let earlier: f64 = pairs.iter().map(|p| p.1).sum();
            let later: f64 = pairs.iter().map(|p| p.2).sum();
            let n = match &count_segment {
                Some(c) => pairs
                    .iter()
                    .filter_map(|&(o, _, _)| c.get(o, j))
                    .sum::<f64>(),
                None => pairs.iter().filter(|p| p.1 > 0.0).count() as f64,
            };
            let z = if earlier <= 0.0 || n <= 0.0 {
                0.0
            } else {
                let own = later / earlier;
                let mut z = n / (n + k);
                let gap = own - derived;
                for &(_, before, after) in &pairs {
                    let rest = earlier - before;
                    let without = if rest > 0.0 {
                        (later - after) / rest
                    } else {
                        derived
                    };
                    let excess = (own - without).abs() - max_influence * gap;
                    if excess > 0.0 {
                        z = z.min(max_influence * derived / excess);
                    }
                }
                z
            };
            weights.push((self.ages[j], 1.0 - z));
        }
        Ok(PatternWeight::ByAge(weights))
    }

    /// The layer's age-to-age factors as a [`SelectedPattern`] with
    /// `weight` on them (decision 12 of `reserving-v02.md`): `All(1.0)`
    /// selects them fully, which is what a sparse layer triangle needs. The
    /// tail is separate: [`tail`](Self::tail).
    pub fn selected(&self, weight: PatternWeight) -> SelectedPattern {
        let values = self
            .ages
            .iter()
            .copied()
            .zip(self.layer_ldf.iter().copied())
            .collect();
        SelectedPattern::new(PatternKind::AgeToAge, values).with_weight(weight)
    }

    /// The layer's tail as a constant [`Tail`], for the chain ladder of the
    /// layer triangle.
    pub fn tail(&self) -> Tail {
        Tail::from(self.layer_tail)
    }
}

/// The expected loss to `layer` of one claim at the maturity whose claims
/// are the ultimate ones divided by `shift` (`X_j = X_inf / c`):
/// `(LEV(c b) - LEV(c a)) / c` with `LEV` the ultimate severity's.
///
/// With `shift = 1` it is the layer at ultimate. A layer share at a
/// maturity is this over the severity's mean over `shift`.
pub fn layer_at_shift(ultimate: &dyn Severity, layer: Layer, shift: f64) -> f64 {
    (ultimate.lev(shift * layer.exhaustion()) - ultimate.lev(shift * layer.attachment)) / shift
}

/// The shift `c` with `c / LEV(c u) = r / LEV(u)`: a ground-up column
/// capped at `u` per claim developing by the severity factor `r`. The
/// left side rises with `c` (the limited mean is concave), so the root is
/// unique and found by bisection on a log scale.
fn solve_shift(lev: &dyn Fn(f64) -> f64, cap: f64, r: f64, age: Lag) -> Result<f64> {
    if r == 1.0 {
        return Ok(1.0);
    }
    let target = r / lev(cap);
    let h = |c: f64| c / lev(c * cap);
    let (mut lo, mut hi) = (1.0_f64, 1.0_f64);
    if r > 1.0 {
        while h(hi) < target {
            hi *= 2.0;
            if hi > 1e60 {
                return Err(no_shift(age));
            }
        }
    } else {
        while h(lo) > target {
            lo *= 0.5;
            if lo < 1e-60 {
                return Err(no_shift(age));
            }
        }
    }
    Ok(bisect_log(lo, hi, |c| h(c) < target))
}

fn no_shift(age: Lag) -> Error {
    Error::Excess(format!(
        "no scale shift reproduces the capped ground-up factor at age {age}: the factor is \
         out of the range the severity allows below the cap"
    ))
}

/// The layer's share of a severity's mean, `(LEV(b) - LEV(a)) / mean`, on
/// the ultimate basis (open decision 2: an exposure curve's share is passed
/// in by the caller from the bindings).
///
/// ```
/// use prospicio_prob::Pareto;
/// use prospicio_reserving::excess::{Layer, layer_share};
///
/// // A Pareto(1, alpha 2): the share above 4 is 4^(1 - 2) / 2 = 1/8.
/// let s = layer_share(&Pareto::new(1.0, 2.0).unwrap(), Layer::unlimited(4.0).unwrap()).unwrap();
/// assert!((s - 0.125).abs() < 1e-12);
/// ```
pub fn layer_share(severity: &dyn Severity, layer: Layer) -> Result<f64> {
    layer.check()?;
    let mean = severity.mean();
    if !(mean.is_finite() && mean > 0.0) {
        return Err(Error::Excess(
            "the severity needs a finite, positive mean for a layer share".into(),
        ));
    }
    Ok((severity.lev(layer.exhaustion()) - severity.lev(layer.attachment)) / mean)
}

/// A copy of `triangle` with the column `name`: the `exposure` column times
/// the layer share of each origin (`shares` has one share for every origin,
/// or one per origin), so that a BF or Cape Cod prior reads the layer's
/// exposure `E_i s_i` at the latest value of each origin. Each share is
/// between 0 and 1.
///
/// The cells keep the exposure column's basis (cumulative or incremental),
/// which a share per origin scales alike.
pub fn layer_exposure(
    triangle: &Triangle,
    exposure: &str,
    shares: &[f64],
    name: &str,
) -> Result<Triangle> {
    let n_origins = triangle.shape()[2];
    let per_origin: Vec<f64> = match shares.len() {
        1 => vec![shares[0]; n_origins],
        n if n == n_origins => shares.to_vec(),
        n => {
            return Err(Error::LengthMismatch {
                column: "shares".into(),
                expected: n_origins,
                found: n,
            });
        }
    };
    if let Some(&s) = per_origin
        .iter()
        .find(|s| !(s.is_finite() && (0.0..=1.0).contains(*s)))
    {
        return Err(Error::InvalidSetting {
            name: "share",
            value: s,
            expected: "between 0 and 1",
        });
    }
    triangle.with_scaled_column(exposure, name, &per_origin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::annual;
    use prospicio_prob::{Grid, Lognormal, Pareto};

    /// The four ultimate claims of Bloom and Patrik's section 3 times
    /// `scale`, equally likely, on a grid of step `112.5 scale`.
    fn four_claims(scale: f64) -> Grid {
        let mut probs = vec![0.0; 21];
        for j in [4, 15, 10, 20] {
            probs[j] = 0.25;
        }
        Grid::new(112.5 * scale, probs).unwrap()
    }

    fn one_origin(values: &[f64]) -> Triangle {
        annual(2020, &[values])
    }

    fn pattern_for(layer: Layer, scale: f64) -> ExcessPattern {
        let tri = one_origin(&[1225.0 * scale, 4410.0 * scale, 5512.5 * scale]);
        ExcessPattern::from_ground_up(
            &tri,
            "values",
            Basis::Incurred,
            None,
            &Tail::default(),
            layer,
            &SeverityByMaturity::scale_shift(Box::new(four_claims(scale))),
        )
        .unwrap()
    }

    #[test]
    fn bloom_patrik_year_one() {
        let p = pattern_for(Layer::unlimited(350.0).unwrap(), 1.0);
        assert_eq!(p.ages, [12, 24, 36]);
        for (got, want) in p.shifts.iter().zip([4.5, 1.25, 1.0]) {
            assert!((got - want).abs() < 1e-12);
        }
        assert!((p.layer_cdf[0] - 23.5).abs() < 1e-12);
        assert!((p.layer_ldf[0] - 17.2).abs() < 1e-12);
        assert!((p.layer_ldf[1] - 4112.5 / 3010.0).abs() < 1e-12);
        assert_eq!(p.layer_tail, 1.0);
        assert!((p.ground_up_ldf[0] - 3.6).abs() < 1e-12);
    }

    #[test]
    fn a_limited_layer_from_zero_is_the_limited_triangle() {
        // Limited to 350: 12:24 = 1400 / 1050 and ultimate 1400 / 1050.
        let p = pattern_for(Layer::new(0.0, 350.0).unwrap(), 1.0);
        assert!((p.layer_ldf[0] - 1400.0 / 1050.0).abs() < 1e-12);
        assert!((p.layer_ldf[1] - 1.0).abs() < 1e-12);
        assert!((p.layer_cdf[0] - 1400.0 / 1050.0).abs() < 1e-12);
    }

    #[test]
    fn the_whole_loss_returns_the_ground_up_factors() {
        let p = pattern_for(Layer::unlimited(0.0).unwrap(), 1.0);
        for (l, g) in p.layer_ldf.iter().zip(&p.ground_up_ldf) {
            assert!((l - g).abs() < 1e-12);
        }
    }

    #[test]
    fn counts_separate_frequency_from_severity() {
        // Counts 2, 3, 4 on the first claims' triangle with ground-up
        // factors unchanged: the severity shift is the ground-up factor
        // over the count factor, and a lognormal's whole-loss layer
        // returns the ground-up factor again.
        let tri = Triangle::from_long(&crate::Long {
            keys: &[],
            origin: &[crate::Month::january(2020); 3],
            development: crate::DevelopmentColumn::Age(&[12, 24, 36]),
            values: &[
                ("incurred", &[100.0, 360.0, 450.0]),
                ("count", &[2.0, 3.0, 4.0]),
            ],
            origin_grain: crate::Grain::Year,
            development_grain: crate::Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let sev = Lognormal::new(7.0, 1.0).unwrap();
        let model = SeverityByMaturity::scale_shift(Box::new(sev));
        let p = ExcessPattern::from_ground_up(
            &tri,
            "incurred",
            Basis::Incurred,
            Some("count"),
            &Tail::default(),
            Layer::unlimited(0.0).unwrap(),
            &model,
        )
        .unwrap();
        assert!((p.shifts[0] - 4.5 / 2.0).abs() < 1e-12);
        assert!((p.shifts[1] - 1.25 / (4.0 / 3.0)).abs() < 1e-12);
        assert!((p.layer_ldf[0] - 3.6).abs() < 1e-12);
    }

    #[test]
    fn the_ground_up_tail_gives_the_layer_tail() {
        // Ground-up factors 1.3 and a tail of 1.2.
        let sev = Lognormal::new(50_000f64.ln() - 1.5 * 1.5 / 2.0, 1.5).unwrap();
        let tri = one_origin(&[100.0, 130.0]);
        let model = SeverityByMaturity::scale_shift(Box::new(sev));
        let layer = Layer::new(250_000.0, 750_000.0).unwrap();
        let p = ExcessPattern::from_ground_up(
            &tri,
            "values",
            Basis::Incurred,
            None,
            &Tail::from(1.2),
            layer,
            &model,
        )
        .unwrap();
        // The oldest age's ground-up factor is the tail, 1.2.
        assert_eq!(p.ground_up_tail, 1.2);
        assert!((p.shifts[1] - 1.2).abs() < 1e-12);
        assert!((p.shifts[0] - 1.3 * 1.2).abs() < 1e-12);
        assert!(p.layer_tail > 1.2);
        // Cumulative factors telescope.
        assert!((p.layer_cdf[0] - p.layer_ldf[0] * p.layer_tail).abs() < 1e-12);
        assert_eq!(p.tail(), Tail::from(p.layer_tail));
    }

    #[test]
    fn paid_and_mixed_inputs_are_errors() {
        let tri = one_origin(&[100.0, 130.0]);
        let model = SeverityByMaturity::scale_shift(Box::new(four_claims(1.0)));
        let paid = ExcessPattern::from_ground_up(
            &tri,
            "values",
            Basis::Paid,
            None,
            &Tail::default(),
            Layer::unlimited(350.0).unwrap(),
            &model,
        );
        assert!(matches!(paid, Err(Error::NotSupported { .. })));
        let lines = Triangle::from_long(&crate::Long {
            keys: &[("line", &["a", "b", "a", "b"])],
            origin: &[crate::Month::january(2020); 4],
            development: crate::DevelopmentColumn::Age(&[12, 12, 24, 24]),
            values: &[("values", &[1.0, 2.0, 3.0, 4.0])],
            origin_grain: crate::Grain::Year,
            development_grain: crate::Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let mixed = ExcessPattern::from_ground_up(
            &lines,
            "values",
            Basis::Incurred,
            None,
            &Tail::default(),
            Layer::unlimited(350.0).unwrap(),
            &model,
        );
        assert!(matches!(mixed, Err(Error::MultipleSegments(2))));
        assert!(Layer::new(-1.0, 5.0).is_err());
        assert!(Layer::new(1.0, 0.0).is_err());
        assert!(Layer::new(f64::NAN, 5.0).is_err());
        assert!(Layer::new(0.0, f64::INFINITY).is_ok());
    }

    /// A capped triangle built from a known scale path is recovered.
    #[test]
    fn the_capped_solve_recovers_the_scales() {
        let sev = Lognormal::new(9.0, 1.4).unwrap();
        let cap = 200_000.0;
        let c_true = [3.0, 1.5, 1.1, 1.0];
        let lev = |x: f64| sev.lev(x);
        // Capped ground-up value per claim at each age, one claim: the
        // ultimate severity divided by c, capped at `cap`.
        let capped: Vec<f64> = c_true.iter().map(|&c| lev(c * cap) / c).collect();
        let tri = one_origin(&capped);
        let model = SeverityByMaturity::scale_shift_capped(Box::new(sev), cap);
        let p = ExcessPattern::from_ground_up(
            &tri,
            "values",
            Basis::Incurred,
            None,
            &Tail::default(),
            Layer::new(100_000.0, 400_000.0).unwrap(),
            &model,
        )
        .unwrap();
        for (got, want) in p.shifts.iter().zip(c_true) {
            assert!((got / want - 1.0).abs() < 1e-9, "got {got}, want {want}");
        }
        // The factors the pattern gives agree with the layer expectations.
        let layer = Layer::new(100_000.0, 400_000.0).unwrap();
        let at_ult = layer_at_shift(&sev, layer, 1.0);
        for (k, &c) in c_true.iter().enumerate() {
            let want = at_ult / layer_at_shift(&sev, layer, c);
            assert!((p.layer_cdf[k] / want - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn the_capped_solve_fails_out_of_range() {
        // A column that falls (factor below 1) further than the cap allows.
        let sev = Lognormal::new(9.0, 1.4).unwrap();
        let tri = one_origin(&[1000.0, 1.0]);
        let model = SeverityByMaturity::scale_shift_capped(Box::new(sev), 50_000.0);
        let r = ExcessPattern::from_ground_up(
            &tri,
            "values",
            Basis::Incurred,
            None,
            &Tail::default(),
            Layer::unlimited(10_000.0).unwrap(),
            &model,
        );
        assert!(matches!(r, Err(Error::Excess(_))), "{r:?}");
        let r = ExcessPattern::from_ground_up(
            &tri,
            "values",
            Basis::Incurred,
            None,
            &Tail::default(),
            Layer::unlimited(10_000.0).unwrap(),
            &SeverityByMaturity::scale_shift_capped(Box::new(sev), -5.0),
        );
        assert!(matches!(r, Err(Error::InvalidSetting { .. })));
    }

    #[test]
    fn lognormal_scale_shift_illustration_of_the_note() {
        // Mean 50,000 at the earlier age and 65,000 at ultimate (every
        // claim 1.3 times larger), sigma 1.5, counts fixed: layer factors
        // 1.275, 1.545, 1.761, 1.982.
        let sigma = 1.5;
        let sev = Lognormal::new(65_000f64.ln() - sigma * sigma / 2.0, sigma).unwrap();
        let tri = one_origin(&[100.0, 130.0]);
        let model = SeverityByMaturity::scale_shift(Box::new(sev));
        let cases = [
            (0.0, 1_000_000.0, 1.275),
            (250_000.0, 750_000.0, 1.545),
            (1_000_000.0, 2_000_000.0, 1.761),
            (3_000_000.0, 7_000_000.0, 1.982),
        ];
        for (a, l, want) in cases {
            let p = ExcessPattern::from_ground_up(
                &tri,
                "values",
                Basis::Incurred,
                None,
                &Tail::default(),
                Layer::new(a, l).unwrap(),
                &model,
            )
            .unwrap();
            assert!(
                (p.layer_ldf[0] - want).abs() < 6e-4,
                "{l} xs {a}: {}",
                p.layer_ldf[0]
            );
        }
    }

    #[test]
    fn layer_share_closed_form_and_errors() {
        let pareto = Pareto::new(1.0, 2.5).unwrap();
        let s = layer_share(&pareto, Layer::new(4.0, 6.0).unwrap()).unwrap();
        let want = ((4.0f64).powf(-1.5) - (10.0f64).powf(-1.5)) / 2.5;
        assert!((s - want).abs() < 1e-12);
        assert!(layer_share(&pareto, Layer::new(0.0, 1e9).unwrap()).unwrap() < 1.0);
        let heavy = Pareto::new(1.0, 0.9).unwrap();
        assert!(layer_share(&heavy, Layer::unlimited(2.0).unwrap()).is_err());
    }

    #[test]
    fn layer_exposure_scales_the_column() {
        let tri = Triangle::from_long(&crate::Long {
            keys: &[],
            origin: &[
                crate::Month::january(2020),
                crate::Month::january(2020),
                crate::Month::january(2021),
            ],
            development: crate::DevelopmentColumn::Age(&[12, 24, 12]),
            values: &[("paid", &[1.0, 2.0, 3.0]), ("premium", &[10.0, 10.0, 20.0])],
            origin_grain: crate::Grain::Year,
            development_grain: crate::Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let one = layer_exposure(&tri, "premium", &[0.1], "layer premium").unwrap();
        assert_eq!(one.columns(), ["paid", "premium", "layer premium"]);
        let c = one.column_position("layer premium").unwrap();
        assert_eq!(one.get(0, c, 0, 1), Some(1.0));
        assert_eq!(one.get(0, c, 1, 0), Some(2.0));
        assert_eq!(one.get(0, c, 1, 1), None);
        let each = layer_exposure(&tri, "premium", &[0.1, 0.5], "lp").unwrap();
        let c = each.column_position("lp").unwrap();
        assert_eq!(each.get(0, c, 1, 0), Some(10.0));
        assert!(layer_exposure(&tri, "premium", &[0.1, 0.2, 0.3], "x").is_err());
        assert!(layer_exposure(&tri, "premium", &[1.5], "x").is_err());
        assert!(layer_exposure(&tri, "premium", &[0.1], "paid").is_err());
        assert!(layer_exposure(&tri, "nope", &[0.1], "x").is_err());
    }

    /// A pattern on the ages 12, 24 and 36 with the given layer factors.
    fn derived(ldf: [f64; 2]) -> ExcessPattern {
        ExcessPattern {
            ages: vec![12, 24, 36],
            layer_ldf: ldf.to_vec(),
            layer_cdf: vec![ldf[0] * ldf[1], ldf[1], 1.0],
            ground_up_ldf: vec![1.0, 1.0],
            ground_up_tail: 1.0,
            shifts: Vec::new(),
            layer_tail: 1.0,
        }
    }

    /// An annual triangle with a loss and a count column.
    fn losses_and_counts(rows: &[&[f64]], counts: &[&[f64]]) -> Triangle {
        let (mut origin, mut ages, mut values, mut n) = (vec![], vec![], vec![], vec![]);
        for (k, (row, count)) in rows.iter().zip(counts).enumerate() {
            for (d, (&v, &c)) in row.iter().zip(count.iter()).enumerate() {
                origin.push(crate::Month::january(2020 + k as i32));
                ages.push(12 * (d as Lag + 1));
                values.push(v);
                n.push(c);
            }
        }
        Triangle::from_long(&crate::Long {
            keys: &[],
            origin: &origin,
            development: crate::DevelopmentColumn::Age(&ages),
            values: &[("layer", &values), ("count", &n)],
            origin_grain: crate::Grain::Year,
            development_grain: crate::Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn by_age(w: &PatternWeight) -> Vec<(Lag, f64)> {
        match w {
            PatternWeight::ByAge(v) => v.clone(),
            PatternWeight::All(_) => panic!("a weight per age"),
        }
    }

    #[test]
    fn credibility_weights_are_k_over_n_plus_k() {
        let p = derived([2.0, 1.2]);
        let rows: [&[f64]; 4] = [
            &[10.0, 20.0, 24.0],
            &[10.0, 20.0, 24.0],
            &[10.0, 20.0],
            &[10.0],
        ];
        let counts: [&[f64]; 4] = [&[2.0, 2.0, 2.0], &[3.0, 3.0, 3.0], &[5.0, 5.0], &[1.0]];
        let tri = losses_and_counts(&rows, &counts);
        // Without counts n is the origins with a positive earlier value: 3
        // and 2. Every ratio equals the derived one, so the cap is idle.
        let w = by_age(
            &p.credibility_weights(&tri, "layer", None, 3.0, 0.1)
                .unwrap(),
        );
        assert_eq!(w[0].0, 12);
        assert!((w[0].1 - 0.5).abs() < 1e-15 && (w[1].1 - 0.6).abs() < 1e-15);
        // With counts n is the claims at the earlier age: 10 and 5.
        let w = by_age(
            &p.credibility_weights(&tri, "layer", Some("count"), 3.0, 0.1)
                .unwrap(),
        );
        assert!((w[0].1 - 3.0 / 13.0).abs() < 1e-15 && (w[1].1 - 3.0 / 8.0).abs() < 1e-15);
        // Errors: k, the cap, a triangle on other ages.
        assert!(
            p.credibility_weights(&tri, "layer", None, 0.0, 0.1)
                .is_err()
        );
        assert!(
            p.credibility_weights(&tri, "layer", None, f64::NAN, 0.1)
                .is_err()
        );
        assert!(
            p.credibility_weights(&tri, "layer", None, 1.0, 0.0)
                .is_err()
        );
        let short = annual(2020, &[&[1.0, 2.0], &[1.0]]);
        assert!(matches!(
            p.credibility_weights(&short, "values", None, 1.0, 0.1),
            Err(Error::Excess(_))
        ));
    }

    #[test]
    fn one_huge_link_ratio_does_not_dominate_the_blend() {
        let p = derived([2.0, 1.0]);
        // Three origins develop by 2; the fourth from 0.1 to 50 (500).
        let tri = annual(
            2017,
            &[
                &[10.0, 20.0, 20.0],
                &[10.0, 20.0, 20.0],
                &[10.0, 20.0],
                &[0.1, 50.0],
            ],
        );
        let own = 110.0 / 30.1;
        let shift = own - 2.0; // the fourth origin's pull on the own factor
        // Uncapped (a share of 10, never binding): n = 4, k = 1, z = 0.8,
        // and the fourth origin moves the blend by about 40%.
        let w = by_age(
            &p.credibility_weights(&tri, "values", None, 1.0, 10.0)
                .unwrap(),
        );
        assert!((w[0].1 - 0.2).abs() < 1e-15);
        // Capped at 10%: the blend moves by exactly 10% of itself.
        let w = by_age(
            &p.credibility_weights(&tri, "values", None, 1.0, 0.1)
                .unwrap(),
        );
        let z = 1.0 - w[0].1;
        let blend = 2.0 + z * shift;
        assert!((z * shift - 0.1 * blend).abs() < 1e-12, "z {z}");
        assert!(blend < 2.25);
        // The chain ladder applies it.
        let cl = crate::ChainLadder {
            development: Development {
                pattern: Some(p.selected(PatternWeight::ByAge(w))),
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(&tri, "values")
        .unwrap();
        assert!((cl.ldf()[0] - blend).abs() < 1e-12);
    }

    #[test]
    fn an_empty_age_gets_the_whole_weight() {
        // Every origin is zero at 12 months, though the counts say claims
        // are in the layer: the own factor is missing, so the weight is 1.
        let p = derived([3.0, 1.1]);
        let rows: [&[f64]; 3] = [&[0.0, 5.0, 6.0], &[0.0, 4.0], &[0.0]];
        let counts: [&[f64]; 3] = [&[1.0, 2.0, 2.0], &[1.0, 1.0], &[0.0]];
        let tri = losses_and_counts(&rows, &counts);
        let w = by_age(
            &p.credibility_weights(&tri, "layer", Some("count"), 1.0, 0.1)
                .unwrap(),
        );
        assert_eq!(w[0].1, 1.0);
        assert!(w[1].1 < 1.0);
        let cl = crate::ChainLadder {
            development: Development {
                pattern: Some(p.selected(PatternWeight::ByAge(w))),
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(&tri, "layer")
        .unwrap();
        assert_eq!(cl.ldf()[0], 3.0);
        assert!(cl.development.has_missing_factor());
    }

    fn shape(
        column: &[f64],
        alphas: &[f64],
        ultimate: f64,
        layer: Layer,
        base: ShapeBase,
    ) -> Result<ExcessPattern> {
        let ages: Vec<(Lag, f64)> = alphas
            .iter()
            .enumerate()
            .map(|(d, &a)| (12 * (d as Lag + 1), a))
            .collect();
        ExcessPattern::from_ground_up(
            &one_origin(column),
            "values",
            Basis::Incurred,
            None,
            &Tail::default(),
            layer,
            &SeverityByMaturity::pareto_shape(1000.0, ages, ultimate, base),
        )
    }

    #[test]
    fn pareto_shape_from_counts_above_the_threshold() {
        // Counts 100 and 150 above 1,000, alphas 2 and 1.5: the layer above
        // 2,000 develops by (150 / 100) (1 / 0.5) 2^0.5.
        let up = Layer::unlimited(2000.0).unwrap();
        let p = shape(&[100.0, 150.0], &[2.0, 1.5], 1.5, up, ShapeBase::Counts).unwrap();
        assert!((p.layer_ldf[0] - 1.5 * 2.0 * 2f64.sqrt()).abs() < 1e-12);
        assert_eq!(p.layer_tail, 1.0);
        assert!(p.shifts.is_empty());
        assert!((p.ground_up_ldf[0] - 1.5).abs() < 1e-15);
        // A bounded layer is the ratio of the Pareto layer integrals.
        let bounded = Layer::new(2000.0, 3000.0).unwrap();
        let p = shape(
            &[100.0, 150.0],
            &[2.0, 1.5],
            1.5,
            bounded,
            ShapeBase::Counts,
        )
        .unwrap();
        let at = |q: f64| Pareto::new(1000.0, q).unwrap().layer(3000.0, 2000.0);
        assert!((p.layer_ldf[0] - 1.5 * at(1.5) / at(2.0)).abs() < 1e-12);
    }

    #[test]
    fn pareto_shape_tail_is_the_last_shape_move() {
        // The base is complete at 24 months, the shape still moves from 1.5
        // to 1.2: the layer above 4,000 (c = 4) has the tail 4^0.3.
        let base = ShapeBase::Losses(Layer::unlimited(1000.0).unwrap());
        let high = Layer::unlimited(4000.0).unwrap();
        let p = shape(&[10.0, 20.0], &[2.0, 1.5], 1.2, high, base).unwrap();
        assert!((p.layer_tail - 4f64.powf(0.3)).abs() < 1e-12);
        assert!((p.layer_ldf[0] - 2.0 * 4f64.powf(0.5)).abs() < 1e-12);
        assert!((p.layer_cdf[0] - p.layer_ldf[0] * p.layer_tail).abs() < 1e-12);
        // The base layer itself develops as its column.
        let own = Layer::unlimited(1000.0).unwrap();
        let p = shape(&[10.0, 20.0], &[2.0, 1.5], 1.5, own, base).unwrap();
        assert!((p.layer_ldf[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn pareto_shape_input_errors() {
        let up = Layer::unlimited(2000.0).unwrap();
        let counts = ShapeBase::Counts;
        let col = [1.0, 2.0];
        // Below the threshold, a missing or extra alpha, an unlimited layer
        // at alpha 1 or below, a bad ultimate alpha.
        let low = Layer::unlimited(500.0).unwrap();
        assert!(shape(&col, &[2.0, 1.5], 1.5, low, counts).is_err());
        let low_base = ShapeBase::Losses(Layer::unlimited(10.0).unwrap());
        assert!(shape(&col, &[2.0, 1.5], 1.5, up, low_base).is_err());
        assert!(shape(&col, &[2.0], 1.5, up, counts).is_err());
        assert!(shape(&col, &[2.0, 1.5, 1.4], 1.5, up, counts).is_err());
        assert!(shape(&col, &[2.0, 1.0], 1.0, up, counts).is_err());
        assert!(shape(&col, &[2.0, 1.5], -1.0, up, counts).is_err());
        // A bounded layer is fine at alpha below 1.
        let bounded = Layer::new(2000.0, 1000.0).unwrap();
        assert!(shape(&col, &[2.0, 0.9], 0.9, bounded, counts).is_ok());
        let tri = one_origin(&col);
        let fit = |model: &SeverityByMaturity, counts: Option<&str>| {
            ExcessPattern::from_ground_up(
                &tri,
                "values",
                Basis::Incurred,
                counts,
                &Tail::default(),
                up,
                model,
            )
        };
        let twice = vec![(12, 2.0), (12, 2.0), (24, 1.5)];
        let r = fit(
            &SeverityByMaturity::pareto_shape(1000.0, twice, 1.5, counts),
            None,
        );
        assert!(matches!(r, Err(Error::Excess(_))));
        let alphas = vec![(12, 2.0), (24, 1.5)];
        let r = fit(
            &SeverityByMaturity::pareto_shape(0.0, alphas.clone(), 1.5, counts),
            None,
        );
        assert!(matches!(r, Err(Error::InvalidSetting { .. })));
        // The base comes from the column: a count column is an error.
        let ok = SeverityByMaturity::pareto_shape(1000.0, alphas, 1.5, counts);
        assert!(matches!(fit(&ok, Some("values")), Err(Error::Excess(_))));
        assert!(format!("{ok:?}").contains("ParetoShape"));
    }
}
