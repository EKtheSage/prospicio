//! Gesmann and Morris' hierarchical compartmental reserving model with one
//! outstanding compartment (`docs/design/reserving-compartmental.md`):
//! paid and outstanding fitted jointly from one set of interpretable
//! parameters, sampled with NUTS on a density written once over
//! [`Scalar`] and differentiated with the chunked dual type (decision 4),
//! or with per-cell duals and a hand-written chain rule when the priors put
//! effects on each cell (model 2).

mod year;

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::{Grain, Lag, Period, StreamRng};
use prospicio_math::special::{norm_cdf, norm_quantile};
use prospicio_prob::{
    Distribution, Gamma, InputHasher, KeyValue, PredictiveDistribution, Provenance,
};

use super::dual::{CHUNK, Dual, gradient};
use super::laplace::{Posterior, Whitened, mode};
use super::one_year::{self, BayesOneYear, BayesOneYearFit, OneYearRefit};
use super::scalar::{Scalar, exprel_neg};
use super::{Diagnostics, Stream, derive_seed, one_year_seed, pilot_seed};
use crate::bootstrap::{component_sums, push_moments};
use crate::dependence::SegmentDependence;
use crate::error::{Error, Result};
use crate::segments::{FitTable, SegmentFits, fit_each};
use crate::triangle::{Segment, Triangle};

/// The warning of a one-year view that reweights only by its model's
/// default ([`Compartmental::one_year_refit`]).
const APPROXIMATE: &str = "the one-year view reweights its flagged simulations only, the \
default with development effects (model 2): moment matching rescues none of them and each \
refit takes about as long as a fit. The view is approximate, about 0.4 sd of the total \
claims development result off a refit per simulation on company 337; set refit (with \
max_refits as a budget) to refit them";

/// The one-compartment closed form (decision 2) per unit of premium:
/// `OS(t) / (Π RLR) = k_er g(t)` and
/// `PD(t) / (Π RLR RRF) = (1 - e^(-k_p t)) - k_p g(t)`, with
/// `g(t) = t e^(-k_p t) (1 - e^(-Δ t)) / (Δ t)` and `Δ = k_er - k_p`,
/// finite and smooth through `k_er = k_p` by [`exprel_neg`].
pub(crate) fn one_os<S: Scalar>(k_er: S, k_p: S, t: f64) -> (S, S) {
    let decay = k_p * (-t);
    let g = decay.exp() * t * exprel_neg((k_er - k_p) * t);
    (k_er * g, -decay.exp_m1() - k_p * g)
}

/// Where the outstanding amounts come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outstanding<'a> {
    /// A column of outstanding amounts (a case reserve, or one with bulk and
    /// IBNR: the user's choice of column).
    Column(&'a str),
    /// A column of incurred amounts: outstanding is incurred less paid,
    /// cell by cell, where both are observed. Gesmann and Morris' outstanding
    /// on company 337 is incurred with bulk and IBNR less paid.
    FromIncurred(&'a str),
}

/// The likelihood of the observed amounts (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompartmentalLikelihood {
    /// Normal, each response type with its own sigma: Gesmann and Morris'
    /// model 1.
    #[default]
    Normal,
    /// Log-normal, each response type with its own sigma on the log scale,
    /// `log y ~ N(log μ, σ²)`: Gesmann and Morris' model 2. Every observed
    /// response must be positive.
    LogNormal,
}

/// The scale of the response (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scale {
    /// Amounts, as model 1.
    #[default]
    Amount,
    /// Loss ratios to each origin's premium, as model 2: the response and
    /// a normal likelihood's sigma are per unit of premium (a log-normal
    /// sigma is the same on either scale). Predictions and reserves stay
    /// amounts.
    LossRatio,
}

/// The paid response (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaidResponse {
    /// Cumulative paid, as model 1.
    #[default]
    Cumulative,
    /// Incremental paid, as model 2: the paid of each period, whose mean is
    /// the closed form at the cell's age less the same at the age before,
    /// both with the cell's own parameters (the paper's `claimsprocess`).
    /// Each observed paid cell after the first age needs the paid of the
    /// age before.
    Incremental,
}

/// Which responses are fitted (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fitted {
    /// Paid and outstanding, as both of the paper's models.
    #[default]
    PaidAndOutstanding,
    /// Paid and incurred: not yet built ([`Error::Bayes`]).
    PaidAndIncurred,
    /// Paid alone: not yet built ([`Error::Bayes`]); from paid alone only
    /// `RLR × RRF` is identified (decision 2).
    PaidOnly,
    /// Incurred alone: not yet built ([`Error::Bayes`]).
    IncurredOnly,
}

/// How the two rates are kept apart (decision 2, "Identification"; open
/// decision 4, settled as (c)). The means are exactly invariant under the
/// swap `(k_er, k_p, RLR_i, RRF_i) -> (k_p, k_er, RLR_i k_er / k_p,
/// RRF_i k_p / k_er)`, so only the priors tell the two modes apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RateOrder {
    /// The paper's independent rate priors; the fit warns (in its
    /// diagnostics) when any posterior draw has `k_er <= k_p`.
    #[default]
    Free,
    /// `k_er = k_p + e^delta`: the same rate priors restricted to
    /// `k_er > k_p` (the prior on `k_er`'s `o` is kept, with the Jacobian of
    /// the map), for a user who widens the rate priors.
    Ordered,
}

/// Where the sampler's pilot run starts (decision 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompartmentalStart {
    /// A least-squares fit of the population curve (shared rates and
    /// ratios, no origin effects), mapped to the intercepts, with each
    /// `log sigma` at the log of its type's residual sd; it falls back to
    /// [`PriorCentre`](Self::PriorCentre) when that fit fails.
    #[default]
    Data,
    /// Every `o` at 0 (the prior medians), each `log sigma` at the log of
    /// its type's residual sd about the prior-median curve.
    PriorCentre,
}

/// A prior on a positive parameter `θ` (decision 2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositivePrior {
    /// `θ = median e^(sd o)` with `o ~ N(0, 1)`: log-normal with that
    /// median and log-scale sd, written as the paper writes it.
    LogNormal { median: f64, sd: f64 },
}

/// A prior on a standard deviation, half of a distribution centred on 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HalfPrior {
    /// Half Student-t with `nu` degrees of freedom and scale `scale`.
    StudentT { nu: f64, scale: f64 },
    /// Half normal with scale `scale`.
    Normal { scale: f64 },
}

/// A prior on a real parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RealPrior {
    /// Student-t with `nu` degrees of freedom, location and scale (Cauchy
    /// at `nu = 1`).
    StudentT { nu: f64, location: f64, scale: f64 },
    /// Normal with mean and sd.
    Normal { mean: f64, sd: f64 },
}

/// The priors of [`Compartmental`], a required argument (open decision 2,
/// settled as (a)): the rates and ratios are specific to a line and a book,
/// so the library does not choose them silently. Check a set with
/// [`Compartmental::prior_predictive`] before fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompartmentalPriors {
    /// The reporting rate `k_er`.
    pub k_er: PositivePrior,
    /// The payment rate `k_p`.
    pub k_p: PositivePrior,
    /// The reported loss ratio `RLR` (its population value; each origin
    /// adds its effect on the `o` scale).
    pub rlr: PositivePrior,
    /// The reserve robustness factor `RRF`, paid over reported.
    pub rrf: PositivePrior,
    /// The sd `τ₁` of the origin effects on `RLR`'s `o` scale.
    pub rlr_effect_sd: HalfPrior,
    /// The sd `τ₂` of the origin effects on `RRF`'s `o` scale.
    pub rrf_effect_sd: HalfPrior,
    /// The LKJ shape of the correlation of the two effects (1: uniform on
    /// `(-1, 1)`).
    pub lkj_eta: f64,
    /// Priors on `log σ_OS` and `log σ_PD`, in that order.
    pub log_sigma: [RealPrior; 2],
    /// The sd of origin effects on `k_er`'s `o` scale, uncorrelated with the
    /// others; `None` (model 1): `k_er` has no origin effect.
    pub ker_effect_sd: Option<HalfPrior>,
    /// The same for `k_p`.
    pub kp_effect_sd: Option<HalfPrior>,
    /// The sds of development-year effects on the `o` scales of `k_er`,
    /// `k_p`, `RLR` and `RRF`, in that order, each uncorrelated; `None`: no
    /// development effect on that parameter (model 1 has none). Each
    /// development age with an observed cell is a level; a later age takes
    /// a new effect from the population in each simulation.
    pub development_effect_sd: [Option<HalfPrior>; 4],
}

impl CompartmentalPriors {
    /// Gesmann and Morris' (2020) case study, model 1 (sec. 5.2.1, code
    /// 7.2.2): `k_er = 3 e^(0.1 o)`, `k_p = e^(0.1 o)`, `RLR = 0.7 e^(0.2 o)`,
    /// `RRF = 0.8 e^(0.1 o)`, origin effect sds half `student_t(10, 0, 0.2)`
    /// and `student_t(10, 0, 0.1)` on the `o` scale, LKJ(1), and
    /// `student_t(1, 0, 1000)` on each `log σ` (brms' class `b` prior on
    /// `sigma ~ 0 + deltaf`, effectively flat). On company 337 the data pull
    /// the rates 5 to 9 prior sds from these medians (decision 2): the
    /// posterior is a compromise with the preset, not a description of it.
    pub fn gesmann_morris() -> Self {
        let cauchy = RealPrior::StudentT {
            nu: 1.0,
            location: 0.0,
            scale: 1000.0,
        };
        Self {
            k_er: PositivePrior::LogNormal {
                median: 3.0,
                sd: 0.1,
            },
            k_p: PositivePrior::LogNormal {
                median: 1.0,
                sd: 0.1,
            },
            rlr: PositivePrior::LogNormal {
                median: 0.7,
                sd: 0.2,
            },
            rrf: PositivePrior::LogNormal {
                median: 0.8,
                sd: 0.1,
            },
            rlr_effect_sd: HalfPrior::StudentT {
                nu: 10.0,
                scale: 0.2,
            },
            rrf_effect_sd: HalfPrior::StudentT {
                nu: 10.0,
                scale: 0.1,
            },
            lkj_eta: 1.0,
            log_sigma: [cauchy, cauchy],
            ker_effect_sd: None,
            kp_effect_sd: None,
            development_effect_sd: [None; 4],
        }
    }

    /// Gesmann and Morris' (2020) case study, model 2 (code 7.2.3, where it
    /// and the text disagree the code wins): the rate and ratio priors of
    /// [`gesmann_morris`](Self::gesmann_morris); origin effects on all four
    /// parameters (`RLR` and `RRF` correlated, LKJ(1)) and development-year
    /// effects on all four, every effect sd half `student_t(10, 0, scale)`
    /// on the `o` scale with one scale per parameter, shared by its two
    /// groups as brms shares it: 0.3 for `k_er` and `k_p`, 0.7 for `RLR`
    /// and 0.5 for `RRF`; and `normal(log 0.2, 0.2)` on each `log σ` (the
    /// text gives a median of 0.1). Use it with the log-normal likelihood
    /// on loss ratios with incremental paid, the paper's model 2.
    pub fn gesmann_morris_2() -> Self {
        let half = |scale: f64| HalfPrior::StudentT { nu: 10.0, scale };
        let log_sigma = RealPrior::Normal {
            mean: 0.2f64.ln(),
            sd: 0.2,
        };
        Self {
            rlr_effect_sd: half(0.7),
            rrf_effect_sd: half(0.5),
            log_sigma: [log_sigma, log_sigma],
            ker_effect_sd: Some(half(0.3)),
            kp_effect_sd: Some(half(0.3)),
            development_effect_sd: [
                Some(half(0.3)),
                Some(half(0.3)),
                Some(half(0.7)),
                Some(half(0.5)),
            ],
            ..Self::gesmann_morris()
        }
    }

    /// Whether any effect moves the rates by cell.
    fn rates_vary(&self) -> bool {
        self.ker_effect_sd.is_some()
            || self.kp_effect_sd.is_some()
            || self.development_effect_sd[..2].iter().any(Option::is_some)
    }

    fn validate(&self) -> Result<()> {
        let positive = |name: &'static str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidSetting {
                    name,
                    value,
                    expected: "a finite, positive prior setting",
                })
            }
        };
        for (median_name, sd_name, prior) in [
            ("k_er median", "k_er sd", self.k_er),
            ("k_p median", "k_p sd", self.k_p),
            ("rlr median", "rlr sd", self.rlr),
            ("rrf median", "rrf sd", self.rrf),
        ] {
            let PositivePrior::LogNormal { median, sd } = prior;
            positive(median_name, median)?;
            positive(sd_name, sd)?;
        }
        let dev = self.development_effect_sd;
        for (nu_name, scale_name, prior) in [
            (
                "rlr_effect_sd nu",
                "rlr_effect_sd scale",
                Some(self.rlr_effect_sd),
            ),
            (
                "rrf_effect_sd nu",
                "rrf_effect_sd scale",
                Some(self.rrf_effect_sd),
            ),
            (
                "ker_effect_sd nu",
                "ker_effect_sd scale",
                self.ker_effect_sd,
            ),
            ("kp_effect_sd nu", "kp_effect_sd scale", self.kp_effect_sd),
            (
                "development_effect_sd k_er nu",
                "development_effect_sd k_er scale",
                dev[0],
            ),
            (
                "development_effect_sd k_p nu",
                "development_effect_sd k_p scale",
                dev[1],
            ),
            (
                "development_effect_sd rlr nu",
                "development_effect_sd rlr scale",
                dev[2],
            ),
            (
                "development_effect_sd rrf nu",
                "development_effect_sd rrf scale",
                dev[3],
            ),
        ] {
            let Some(prior) = prior else { continue };
            match prior {
                HalfPrior::StudentT { nu, scale } => {
                    positive(nu_name, nu)?;
                    positive(scale_name, scale)?;
                }
                HalfPrior::Normal { scale } => positive(scale_name, scale)?,
            }
        }
        positive("lkj_eta", self.lkj_eta)?;
        for prior in self.log_sigma {
            match prior {
                RealPrior::StudentT {
                    nu,
                    location,
                    scale,
                } => {
                    positive("log_sigma nu", nu)?;
                    positive("log_sigma scale", scale)?;
                    if !location.is_finite() {
                        return Err(Error::InvalidSetting {
                            name: "log_sigma location",
                            value: location,
                            expected: "a finite prior location",
                        });
                    }
                }
                RealPrior::Normal { mean, sd } => {
                    positive("log_sigma sd", sd)?;
                    if !mean.is_finite() {
                        return Err(Error::InvalidSetting {
                            name: "log_sigma mean",
                            value: mean,
                            expected: "a finite prior mean",
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Below this prior mass on `k_er > k_p`, [`Compartmental::prior_predictive`]
/// under [`RateOrder::Ordered`] is an error: the restricted prior is then
/// a sliver at the boundary rather than a prior worth checking.
const MIN_ORDERED_MASS: f64 = 1e-6;

/// The restriction `k_er > k_p` on the rate intercepts' standard normals
/// `(b_ker, b_kp)` as `z > a`, where `z = (s_er b_ker - s_p b_kp) / s` is
/// standard normal and `s = sqrt(s_er^2 + s_p^2)`: `(a, s)`. The priors'
/// mass on the restriction is `Φ(-a)`.
fn rate_order_bound(pr: CompartmentalPriors) -> (f64, f64) {
    let (m_er, s_er) = lognormal(pr.k_er);
    let (m_p, s_p) = lognormal(pr.k_p);
    let s = s_er.hypot(s_p);
    ((m_p / m_er).ln() / s, s)
}

/// The rate intercepts' standard normals `(b_ker, b_kp)` drawn exactly
/// from their priors restricted to `k_er > k_p`, a half-plane: the
/// projection `z` of [`rate_order_bound`] from its normal truncated to
/// `z > a` by the inverse of the upper tail (`Φ(-z) = u Φ(-a)`, accurate
/// however far out `a` is), and the orthogonal projection from its normal.
fn ordered_rates(pr: CompartmentalPriors, rng: &mut StreamRng) -> (f64, f64) {
    let (_, s_er) = lognormal(pr.k_er);
    let (_, s_p) = lognormal(pr.k_p);
    let (a, s) = rate_order_bound(pr);
    let z = -norm_quantile(norm_cdf(-a) * rng.next_open01());
    let y = norm_quantile(rng.next_open01());
    ((s_er * z + s_p * y) / s, (s_er * y - s_p * z) / s)
}

/// `(median, sd)` of a [`PositivePrior`].
fn lognormal(prior: PositivePrior) -> (f64, f64) {
    let PositivePrior::LogNormal { median, sd } = prior;
    (median, sd)
}

/// The log density of a half prior on `τ = e^l`, in `l`, with the Jacobian
/// `l` and without constants.
fn half_on_log<S: Scalar>(l: S, prior: HalfPrior) -> S {
    match prior {
        HalfPrior::StudentT { nu, scale } => {
            let x = l.exp() / scale;
            l - (x * x / nu).ln_1p() * ((nu + 1.0) / 2.0)
        }
        HalfPrior::Normal { scale } => {
            let x = l.exp() / scale;
            l - x * x * 0.5
        }
    }
}

/// The log density of a real prior at `x`, without constants.
fn real_prior<S: Scalar>(x: S, prior: RealPrior) -> S {
    match prior {
        RealPrior::StudentT {
            nu,
            location,
            scale,
        } => {
            let r = (x - location) / scale;
            -(r * r / nu).ln_1p() * ((nu + 1.0) / 2.0)
        }
        RealPrior::Normal { mean, sd } => {
            let r = (x - mean) / sd;
            -(r * r * 0.5)
        }
    }
}

/// About the median of a half prior: `0.6745` times its scale (the half
/// normal's; the half Student-t(10)'s is `0.6998`), for the start only.
fn half_median(prior: HalfPrior) -> f64 {
    match prior {
        HalfPrior::StudentT { scale, .. } | HalfPrior::Normal { scale } => 0.6745 * scale,
    }
}

/// One draw of a half prior.
fn draw_half(prior: HalfPrior, rng: &mut StreamRng) -> f64 {
    let z = norm_quantile(rng.next_open01()).abs();
    match prior {
        HalfPrior::StudentT { nu, scale } => {
            let chi2 = Gamma::new(nu / 2.0, 2.0)
                .expect("nu was validated")
                .sample(rng, 1)[0];
            scale * z / (chi2 / nu).sqrt()
        }
        HalfPrior::Normal { scale } => scale * z,
    }
}

/// Gesmann and Morris' (2020) hierarchical compartmental reserving model
/// with one outstanding compartment (`docs/design/reserving-compartmental.md`).
/// Claims flow from exposure into outstanding as they are reported, at rate
/// `k_er`, and from outstanding into paid as they settle, at rate `k_p`:
///
/// ```text
/// dEX/dt = -k_er EX                      EX(0) = Π_i
/// dOS/dt =  k_er RLR_i EX - k_p OS       OS(0) = 0
/// dPD/dt =  k_p RRF_i OS                 PD(0) = 0
/// ```
///
/// for origin `i` with premium `Π_i` and development time `t` in years. In
/// closed form (the paper's Eq. 2), with `g(t) = (e^(-k_p t) - e^(-k_er t))
/// / (k_er - k_p)` computed stably through `k_er = k_p`:
///
/// ```text
/// OS_i(t) = Π_i RLR_i k_er g(t)
/// PD_i(t) = Π_i RLR_i RRF_i [(1 - e^(-k_p t)) - k_p g(t)]
/// ```
///
/// so `RLR_i` is the reported loss ratio, `RRF_i` the reserve robustness
/// factor (paid over reported) and `RLR_i RRF_i` the ultimate loss ratio.
/// The rates are shared; `RLR` and `RRF` vary by origin with correlated
/// effects on the paper's standardised `o` scale (model 1, as its brms
/// code):
///
/// ```text
/// RLR_i = m_RLR e^(s_RLR (b_RLR + u_i1)),   RRF_i = m_RRF e^(s_RRF (b_RRF + u_i2))
/// k_er = m_ker e^(s_ker b_ker),             k_p = m_kp e^(s_kp b_kp)
/// u_i = diag(τ) L z_i,  z_i ~ N(0, I),  L L' = [[1, ρ], [ρ, 1]],  ρ ~ LKJ(η)
/// b ~ N(0, 1),  τ_1, τ_2 ~ half priors,  log σ_OS, log σ_PD ~ real priors
/// y_OS ~ N(OS_i(t), σ_OS²),  y_PD ~ N(PD_i(t), σ_PD²)   (cumulative paid)
/// ```
///
/// with each `m e^(s o)` a [`PositivePrior::LogNormal`] of
/// [`CompartmentalPriors`]. The priors are a required argument;
/// [`CompartmentalPriors::gesmann_morris`] is the paper's case study.
///
/// NUTS samples the intercepts `b`, `log τ`, `atanh ρ`, the `z_i` of each
/// origin with an observed cell and the two `log σ`, on a dense scale from a
/// pilot run as [`Meyers`](super::Meyers) does: a scale whitened at a start
/// point ([`CompartmentalStart`]), a pilot run of `tune` warm-up and
/// `max(tune, 100)` kept draws per chain, then the pilot draws' mean and
/// covariance. The gradient comes from the density written once over
/// dual numbers (decision 4) for priors without effects by cell (model 1).
/// With effects by cell (model 2, or any origin effect on a rate or
/// development effect), it is per-cell dual numbers over the cell's four
/// parameters and `log σ`, then a hand-written chain rule to the sampled
/// parameters, checked against the single dual pass over the generic
/// density by a unit test. A non-finite value counts as a divergence.
///
/// An origin with a positive premium and no observed cell (the papers'
/// training triangle has none for its youngest accident year) has no `z`:
/// each predictive simulation draws its effect from the population with the
/// draw's `τ` and `ρ` (decision 2); the fit lists such origins in
/// [`CompartmentalFit::population_origins`]. The reserve is the paid to the
/// square's last age less the latest observed paid (decision 5): the square
/// has as many ages as the premium column has origins, and an origin
/// observed at that age has a reserve of 0. The origin and development
/// grains must be equal.
///
/// The likelihood, the response scale and the paid response are settings
/// (decision 1), each combination allowed: [`likelihood`](Self::likelihood)
/// normal or log-normal, [`scale`](Self::scale) amounts or loss ratios to
/// premium, [`paid_response`](Self::paid_response) cumulative or
/// incremental. The hierarchy is in the priors: origin effects on `k_er`
/// and `k_p` and development-year effects on any of the four parameters
/// (`ker_effect_sd`, `kp_effect_sd` and `development_effect_sd` of
/// [`CompartmentalPriors`]), so each cell has its own parameters, and the
/// mean of an incremental paid cell is the closed form at its age less the
/// same at the age before, both with that cell's parameters. Gesmann and
/// Morris' model 2 is
/// `Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())`
/// ([`model_2`](Self::model_2) sets the log-normal likelihood on loss
/// ratios with incremental paid and a longer, finer sampler).
///
/// [`fitted`](Self::fitted) takes paid and outstanding only, for now; any
/// other value is an [`Error::Bayes`] saying it is not yet built.
///
/// ```
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_reserving::bayes::{Compartmental, CompartmentalPriors, Outstanding};
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_prob::{Distribution, KeyValue};
///
/// // Five origins of cumulative paid and incurred with premium.
/// let paid: [&[f64]; 5] = [
///     &[9.6, 22.8, 33.3, 40.3, 45.1],
///     &[7.9, 19.5, 29.6, 36.8],
///     &[8.7, 24.3, 35.4],
///     &[13.3, 33.0],
///     &[11.4],
/// ];
/// let incurred: [&[f64]; 5] = [
///     &[62.7, 64.0, 65.6, 65.3, 62.3],
///     &[62.1, 56.7, 56.6, 56.8],
///     &[64.0, 61.5, 63.2],
///     &[73.9, 75.1],
///     &[77.1],
/// ];
/// let premium = [104.4, 88.9, 86.0, 99.3, 104.9];
/// let (mut origin, mut age, mut pd, mut inc, mut prem) = (vec![], vec![], vec![], vec![], vec![]);
/// for w in 0..5 {
///     for d in 0..paid[w].len() {
///         origin.push(Month::january(1988 + w as i32));
///         age.push(12 * (d as u32 + 1));
///         pd.push(paid[w][d]);
///         inc.push(incurred[w][d]);
///         prem.push(premium[w]);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &pd), ("incurred", &inc), ("premium", &prem)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// // The paper's case study priors; a short run for the example.
/// let preset = Compartmental::new(CompartmentalPriors::gesmann_morris());
/// let model = Compartmental {
///     sampler: Sampler { chains: 2, tune: 150, draws: 150, target_accept: 0.9, ..preset.sampler },
///     n_sims: 1_000,
///     ..preset
/// };
/// // Check the priors first: paths of the mean paid and outstanding they
/// // imply, here the paid at age 5 of the oldest origin over its premium.
/// let paths = model.prior_predictive(&tri, "premium", 1_000)?;
/// let key = vec![tri.origins()[0].into(), KeyValue::Int(60)];
/// let paid_5 = paths.paid.marginal(&key).unwrap();
/// let (lo, hi) = (paid_5.quantile(0.05)?, paid_5.quantile(0.95)?);
/// assert!(lo / 104.4 > 0.2 && hi / 104.4 < 1.2);
/// let fit = model.fit(&tri, "paid", Outstanding::FromIncurred("incurred"), "premium")?;
/// assert_eq!(fit.reserves.n_components(), 5);
/// assert_eq!(fit.posterior.names()[0], "k_er");
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Compartmental {
    /// The priors (decision 2; a required argument of [`new`](Self::new)).
    pub priors: CompartmentalPriors,
    /// The likelihood.
    pub likelihood: CompartmentalLikelihood,
    /// The response scale.
    pub scale: Scale,
    /// Cumulative or incremental paid.
    pub paid_response: PaidResponse,
    /// The responses fitted: [`PaidAndOutstanding`](Fitted::PaidAndOutstanding)
    /// only, for now.
    pub fitted: Fitted,
    /// How the rates are kept apart.
    pub rate_order: RateOrder,
    /// Where the pilot run starts.
    pub start: CompartmentalStart,
    /// NUTS settings. Its `seed` is not used: the chains take a seed derived
    /// from [`seed`](Self::seed) (module docs, "Seeds").
    pub sampler: Sampler,
    /// Number of predictive simulations. The one-year view runs at most
    /// [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS) of them unless its settings'
    /// [`n_sims`](BayesOneYear::n_sims) says otherwise.
    pub n_sims: usize,
    /// Seed of the chains (and the pilot run) and of the simulations, each
    /// through its own derived seed (module docs, "Seeds").
    pub seed: u64,
    /// How segments depend on each other in a fit of several segments
    /// ([`fit_segments`](Self::fit_segments), decision 8; `Synchronized` is
    /// an error there); [`fit`](Self::fit) has one segment and does not use
    /// it.
    pub dependence: SegmentDependence,
}

impl Compartmental {
    /// The most one-year simulations [`one_year`](Self::one_year) runs by
    /// default (`BayesOneYear::n_sims` `None`): 2,000, against the 10,000
    /// of the lifetime view, since nearly every simulation of these models
    /// is flagged and costs a moment match, a NUTS refit or (model 2) the
    /// reweighting of 95 parameters (the user's decision of 2026-10-16,
    /// `docs/design/reserving-bayes.md` decision 20). A model whose
    /// [`n_sims`](Self::n_sims) is smaller runs that many.
    pub const ONE_YEAR_SIMS: usize = 2_000;

    /// The model with `priors` and the defaults: model 1 (normal on amounts,
    /// cumulative paid, paid and outstanding), free rate order, the
    /// data-based start, four chains of 1,000 warm-up and 2,000 kept draws
    /// with target acceptance 0.99 (the paper's `adapt_delta`) and tree depth
    /// at most 10, 10,000 simulations, seed 0, independent segments.
    pub fn new(priors: CompartmentalPriors) -> Self {
        Self {
            priors,
            likelihood: CompartmentalLikelihood::Normal,
            scale: Scale::Amount,
            paid_response: PaidResponse::Cumulative,
            fitted: Fitted::PaidAndOutstanding,
            rate_order: RateOrder::Free,
            start: CompartmentalStart::Data,
            sampler: Sampler {
                target_accept: 0.99,
                draws: 2000,
                ..Sampler::default()
            },
            n_sims: 10_000,
            seed: 0,
            dependence: SegmentDependence::Independent,
        }
    }

    /// Gesmann and Morris' model 2 settings with `priors` (the paper's are
    /// [`CompartmentalPriors::gesmann_morris_2`]): the log-normal
    /// likelihood on loss ratios with incremental paid, and a longer, finer
    /// sampler than [`new`](Self::new)'s, four chains of 2,000 warm-up and
    /// 8,000 kept draws with target acceptance 0.999 (decision 5: with 87
    /// parameters, effect sds whose posteriors pile up near 0 and a slowly
    /// mixing `k_p`, model 1's sampler left divergent transitions or an
    /// R-hat above 1.01 on most seeds); the rest as [`new`](Self::new).
    pub fn model_2(priors: CompartmentalPriors) -> Self {
        let model_1 = Self::new(priors);
        Self {
            likelihood: CompartmentalLikelihood::LogNormal,
            scale: Scale::LossRatio,
            paid_response: PaidResponse::Incremental,
            sampler: Sampler {
                tune: 2000,
                draws: 8000,
                target_accept: 0.999,
                ..model_1.sampler
            },
            ..model_1
        }
    }

    /// Fits paid and outstanding of a single-segment cumulative triangle
    /// with premium from its `premium` column (the latest value per origin).
    /// The fit's origins are the premium column's; an origin with no paid
    /// or outstanding cell is drawn from the population. Failed diagnostics
    /// do not fail the fit: they are warnings in
    /// [`CompartmentalFit::diagnostics`], as is any posterior draw with
    /// `k_er <= k_p` under [`RateOrder::Free`].
    pub fn fit(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
    ) -> Result<CompartmentalFit> {
        self.fit_streams(
            triangle,
            paid,
            outstanding,
            premium,
            Stream::Chains,
            Stream::Predictive,
        )
    }

    /// [`fit`](Self::fit) of a single-segment triangle, its chains (and
    /// pilot run) and its simulations taking the seeds that `chains` and
    /// `predictive` derive (module docs, "Seeds").
    fn fit_streams(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        chains: Stream,
        predictive: Stream,
    ) -> Result<CompartmentalFit> {
        self.validate()?;
        let data = Data::new(triangle, paid, outstanding, premium)?;
        let model = Model::new(
            &data,
            self.priors,
            self.rate_order == RateOrder::Ordered,
            self.response(),
        )?;
        if model.cells.len() <= model.dim() {
            return Err(Error::Bayes(format!(
                "{} observed cells for {} parameters; the model needs more cells than parameters",
                model.cells.len(),
                model.dim()
            )));
        }
        let (posterior, sampled, mut diagnostics) =
            self.sample(&model, derive_seed(self.seed, chains))?;
        let rate_swaps = {
            let (j_er, j_p) = (0, 1);
            posterior
                .values()
                .chunks_exact(posterior.dim())
                .filter(|d| d[j_er] <= d[j_p])
                .count()
        };
        if rate_swaps > 0 {
            diagnostics.warnings.push(format!(
                "{rate_swaps} of {} posterior draws have k_er <= k_p, the mirror image of the \
                 rate swap; check the rate priors, or set rate_order to Ordered",
                posterior.n_draws()
            ));
        }
        let cell_rate_swaps = model.cell_rate_swaps(&posterior);

        let mut hasher = InputHasher::new();
        hasher.str(paid);
        hasher.str(outstanding_name(outstanding));
        hasher.str(premium);
        hasher.f64s(&data.premium);
        for c in &model.cells {
            hasher.f64s(&[c.origin as f64, c.t, c.y, f64::from(u8::from(c.paid))]);
        }
        let provenance = self.provenance(paid, outstanding, premium, hasher);

        let future = data.future_cells();
        let n_paid = future.paid.len();
        let n_future = n_paid + future.outstanding.len();
        let joint = posterior.predictive(
            vec!["cell".into()],
            (0..n_future)
                .map(|j| vec![KeyValue::Int(j as i64)])
                .collect(),
            self.n_sims,
            derive_seed(self.seed, predictive),
            provenance.clone(),
            |nat, rng, row| model.simulate(nat, &future, rng, row),
        )?;
        let provenance = joint.provenance().clone();
        let n_origins = data.origins.len();
        let mut paid_draws = Vec::with_capacity(self.n_sims * n_paid);
        let mut os_draws = Vec::with_capacity(self.n_sims * (n_future - n_paid));
        let mut reserve_draws = Vec::with_capacity(self.n_sims * n_origins);
        for row in joint.draw_matrix().chunks_exact(n_future) {
            paid_draws.extend_from_slice(&row[..n_paid]);
            os_draws.extend_from_slice(&row[n_paid..]);
            for (i, last) in future.last_paid.iter().enumerate() {
                reserve_draws.push(match last {
                    Some(j) => row[*j] - data.latest_paid[i].map_or(0.0, |(_, v)| v),
                    None => 0.0,
                });
            }
        }
        let cell_keys = |cells: &[(usize, usize)]| -> Vec<Vec<KeyValue>> {
            cells
                .iter()
                .map(|&(i, k)| {
                    vec![
                        data.origins[i].into(),
                        KeyValue::Int(i64::from(data.age(k))),
                    ]
                })
                .collect()
        };
        let with_output = |name: &str| {
            let mut p = provenance.clone();
            p.parameters.retain(|(k, _)| k != "output");
            p.param("output", name)
        };
        let reserves = PredictiveDistribution::from_draws(
            vec!["origin".into()],
            data.origins.iter().map(|&o| vec![o.into()]).collect(),
            reserve_draws,
            with_output("reserves"),
        )?;
        let paid_pd = PredictiveDistribution::from_draws(
            vec!["origin".into(), "age".into()],
            cell_keys(&future.paid),
            paid_draws,
            with_output("paid"),
        )?;
        let os_pd = PredictiveDistribution::from_draws(
            vec!["origin".into(), "age".into()],
            cell_keys(&future.outstanding),
            os_draws,
            with_output("outstanding"),
        )?;
        let ultimate_loss_ratio = model.mean_ultimate_loss_ratios(&posterior);
        Ok(CompartmentalFit {
            origins: data.origins.clone(),
            premium: data.premium.clone(),
            latest_paid: data
                .latest_paid
                .iter()
                .map(|l| l.map_or(0.0, |(_, v)| v))
                .collect(),
            population_origins: (0..n_origins)
                .filter(|&i| model.level[i].is_none())
                .map(|i| data.origins[i])
                .collect(),
            posterior,
            diagnostics,
            rate_swaps,
            cell_rate_swaps,
            reserves,
            paid: paid_pd,
            outstanding: os_pd,
            ultimate_loss_ratio,
            model,
            sampled,
            data,
        })
    }

    /// Fits paid and outstanding in every segment of a cumulative triangle,
    /// each segment with its own premium column and posterior (the same
    /// model and settings, sampled on their own), into joint distributions
    /// of the reserves and the future paid and outstanding cells, the
    /// segments depending on each other as [`dependence`](Self::dependence)
    /// says (decision 8), as in
    /// [`Meyers::fit_segments`](super::Meyers::fit_segments):
    ///
    /// * [`Independent`](SegmentDependence::Independent): each segment's
    ///   simulations as drawn. Segment `k` (in index order) takes its own
    ///   derived seeds for its chains (and its pilot run) and its
    ///   simulations (module docs, "Seeds"), so the segments share no
    ///   random numbers; a segment's draws therefore differ from fitting it
    ///   alone with [`fit`](Self::fit), which a one-segment triangle's
    ///   segment 0 streams reproduce.
    /// * [`RankCorrelation`](SegmentDependence::RankCorrelation): the
    ///   segments' simulations reordered as whole rows by Iman–Conover on the
    ///   segment totals of the reserves; each segment's reserves, future
    ///   paid cells and future outstanding cells move together, so every
    ///   simulation's reserve is still its paid at the last age less the
    ///   latest paid.
    /// * [`Synchronized`](SegmentDependence::Synchronized) is an
    ///   [`Error::Bayes`]: it draws the residuals of the same positions in
    ///   every segment, and a posterior has no residuals to share.
    ///
    /// A bad rank-correlation matrix fails before any sampling, and a
    /// failure names its segment. Each segment is read as
    /// [`fit`](Self::fit) reads a single-segment triangle.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
    ) -> Result<CompartmentalFits> {
        self.validate()?;
        if self.dependence == SegmentDependence::Synchronized {
            return Err(Error::Bayes(
                "Synchronized dependence draws the residuals of the same positions in every \
                 segment, and a posterior has no residuals; use Independent or RankCorrelation"
                    .into(),
            ));
        }
        self.dependence.validate(triangle.index().len())?;
        let mut k = 0;
        let mut fitted = fit_each(triangle, paid, |_| {
            let label = &triangle.index()[k];
            let values: Vec<[&str; 1]> = label.parts().iter().map(|p| [p.as_str()]).collect();
            let conditions: Vec<(&str, &[&str])> = triangle
                .key_names()
                .iter()
                .zip(&values)
                .map(|(name, value)| (name.as_str(), value.as_slice()))
                .collect();
            let fit = triangle.select(&conditions).and_then(|one| {
                self.fit_streams(
                    &one,
                    paid,
                    outstanding,
                    premium,
                    Stream::SegmentChains(k),
                    Stream::SegmentPredictive(k),
                )
            });
            k += 1;
            fit.map(|fit| CompartmentalSegment { fit })
        })?;

        // One row per simulation: every segment's reserves, then its future
        // paid cells, then its future outstanding cells. Iman–Conover ranks
        // a segment by its reserves alone and moves the whole of its row.
        let (mut ranges, mut ranked, mut width) = (Vec::new(), Vec::new(), 0);
        for s in &fitted.fits {
            let f = &s.fit;
            let m = f.reserves.n_components();
            let len = m + f.paid.n_components() + f.outstanding.n_components();
            ranges.push(width..width + len);
            ranked.push(width..width + m);
            width += len;
        }
        let mut rows = Vec::with_capacity(self.n_sims * width);
        for i in 0..self.n_sims {
            for s in &fitted.fits {
                for part in [&s.fit.reserves, &s.fit.paid, &s.fit.outstanding] {
                    let n = part.n_components();
                    rows.extend_from_slice(&part.draw_matrix()[i * n..(i + 1) * n]);
                }
            }
        }
        let rows = self
            .dependence
            .reorder_rows(rows, width, &ranges, &ranked, self.seed)?;
        for (s, range) in fitted.fits.iter_mut().zip(&ranges) {
            let f = &mut s.fit;
            let mut start = range.start;
            let mut rebuilt = Vec::with_capacity(3);
            for part in [&f.reserves, &f.paid, &f.outstanding] {
                let n = part.n_components();
                let draws = rows
                    .chunks_exact(width)
                    .flat_map(|row| row[start..start + n].iter().copied())
                    .collect();
                rebuilt.push(PredictiveDistribution::from_draws(
                    part.dims().to_vec(),
                    part.components().to_vec(),
                    draws,
                    part.provenance().clone(),
                )?);
                start += n;
            }
            let mut rebuilt = rebuilt.into_iter();
            f.reserves = rebuilt.next().expect("three parts");
            f.paid = rebuilt.next().expect("three parts");
            f.outstanding = rebuilt.next().expect("three parts");
        }

        let mut hasher = InputHasher::new();
        hasher.str(paid);
        hasher.str(outstanding_name(outstanding));
        hasher.str(premium);
        for (label, s) in fitted.iter() {
            hasher.str(&label.to_string());
            hasher.f64s(&s.fit.premium);
            for c in s.fit.data.paid.iter().chain(&s.fit.data.outstanding) {
                hasher.f64s(&[c.0 as f64, c.1 as f64, c.2]);
            }
        }
        let premiums: Vec<f64> = fitted
            .fits
            .iter()
            .map(|s| s.fit.premium.iter().sum())
            .collect();
        let provenance = self
            .provenance(paid, outstanding, premium, hasher)
            .param("premium_total", format!("{premiums:?}"))
            .param("segments", fitted.len())
            .param("dependence", format!("{:?}", self.dependence))
            .param(
                "posterior_draws",
                fitted.fits.first().map_or(0, |s| s.fit.posterior.n_draws()),
            );
        CompartmentalFits::assemble(fitted, provenance)
    }

    /// The one-year view (`docs/design/reserving-bayes.md`, decisions 16
    /// to 18 and 20): fits as [`fit`](Self::fit), then draws
    /// [`n_sims`](Self::n_sims) years of the paid and outstanding cells
    /// valued in the twelve months after the valuation (each from a
    /// posterior draw, with the likelihood's noise), reweights the
    /// posterior draws by each year's likelihood with Pareto smoothed
    /// importance sampling, and takes the claims development result of
    /// each origin as the opening reserve less the year's payment and the
    /// reserve after the year, each reserve the (weighted) posterior mean
    /// of the expected paid to the square's last age less the paid so
    /// far. A simulation whose Pareto `k̂` is above the settings' threshold
    /// is moment-matched, and refitted with NUTS (this model's sampler, on
    /// the triangle extended by its year) only when matching cannot bring
    /// its `k̂` to the threshold, or as [`BayesOneYear::refit`] says.
    ///
    /// The defaults (the user's decision of 2026-10-16, decision 20): at
    /// most [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS) simulations, and the
    /// refit mode of [`one_year_refit`](Self::one_year_refit) when the
    /// settings' `refit` is `None`. With development effects (model 2) that
    /// is reweighting only: moment matching rescues none of its flagged
    /// simulations on company 337 and a refit takes about as long as a fit,
    /// so the default view is approximate (about 0.4 sd of the total claims
    /// development result off a refit per simulation) and says so in
    /// [`approximate`](BayesOneYearFit::approximate). Set `refit` to refit
    /// (with [`max_refits`](BayesOneYear::max_refits) as a budget).
    ///
    /// An origin with no observed cell (the papers' 1997) and, with
    /// development effects (model 2), an age with none take their effects
    /// from the population in [`fit`](Self::fit)'s simulations; here they
    /// are part of each posterior draw (standard normal `z` added to the
    /// fit's draws, which is their posterior), so the year's cells update
    /// them. A refit takes about as long as the fit: on the papers'
    /// triangle about 5.5 s for model 1 and about a minute for model 2 in a
    /// release build. The view's seed is the simulations' derived seed of
    /// [`seed`](Self::seed); the fit is the same as `fit`'s.
    ///
    /// ```
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_reserving::bayes::{
    ///     BayesOneYear, Compartmental, CompartmentalPriors, OneYearRefit, Outstanding,
    /// };
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    /// use prospicio_prob::Distribution;
    ///
    /// let paid: [&[f64]; 5] = [
    ///     &[9.6, 22.8, 33.3, 40.3, 45.1],
    ///     &[7.9, 19.5, 29.6, 36.8],
    ///     &[8.7, 24.3, 35.4],
    ///     &[13.3, 33.0],
    ///     &[11.4],
    /// ];
    /// let incurred: [&[f64]; 5] = [
    ///     &[62.7, 64.0, 65.6, 65.3, 62.3],
    ///     &[62.1, 56.7, 56.6, 56.8],
    ///     &[64.0, 61.5, 63.2],
    ///     &[73.9, 75.1],
    ///     &[77.1],
    /// ];
    /// let premium = [104.4, 88.9, 86.0, 99.3, 104.9];
    /// let (mut origin, mut age, mut pd, mut inc, mut prem) = (vec![], vec![], vec![], vec![], vec![]);
    /// for w in 0..5 {
    ///     for d in 0..paid[w].len() {
    ///         origin.push(Month::january(1988 + w as i32));
    ///         age.push(12 * (d as u32 + 1));
    ///         pd.push(paid[w][d]);
    ///         inc.push(incurred[w][d]);
    ///         prem.push(premium[w]);
    ///     }
    /// }
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&age),
    ///     values: &[("paid", &pd), ("incurred", &inc), ("premium", &prem)],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let preset = Compartmental::new(CompartmentalPriors::gesmann_morris());
    /// let model = Compartmental {
    ///     sampler: Sampler { chains: 2, tune: 150, draws: 150, target_accept: 0.9, ..preset.sampler },
    ///     n_sims: 500,
    ///     ..preset
    /// };
    /// // Reweighting only, for speed: model 1's default moment-matches the
    /// // flagged simulations and refits those it cannot rescue.
    /// assert_eq!(model.one_year_refit(), OneYearRefit::MomentMatchThenRefit);
    /// let only = BayesOneYear { refit: Some(OneYearRefit::ReweightOnly), ..BayesOneYear::default() };
    /// let view = model.one_year(&tri, "paid", Outstanding::FromIncurred("incurred"), "premium", &only)?;
    /// assert_eq!(view.cdr.n_components(), 5);
    /// // 1988 is at the square's last age: no cell in the year, no change.
    /// assert!(view.cdr.draw_matrix().chunks(5).all(|row| row[0] == 0.0));
    /// // A refit mode the settings choose is used as it is: not approximate.
    /// assert_eq!(view.approximate, None);
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn one_year(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit<CompartmentalFit>> {
        settings.validate()?;
        let (settings, n_sims, approximate) = self.one_year_defaults(settings);
        let settings = &settings;
        let fit = self.fit(triangle, paid, outstanding, premium)?;
        let view_seed = one_year_seed(derive_seed(self.seed, Stream::Predictive));
        let year = year::CompartmentalYear::new(self, &fit.data)?;
        // The added `z` of each draw: stream `d` of a seed that the
        // simulations' streams (`0..n_sims` of `view_seed`) do not reach.
        let opening = year.opening_draws(
            &fit.model,
            &fit.sampled,
            StreamRng::new(view_seed, u64::MAX).next_u64(),
        )?;
        let draws = one_year::simulate(&year, &opening, settings, n_sims, view_seed)?;
        let mut provenance = fit.reserves.provenance().clone();
        provenance.parameters.retain(|(k, _)| k != "output");
        provenance.model = format!("{}_one_year", provenance.model);
        let provenance = one_year::provenance(provenance, settings, n_sims);
        let cdr = PredictiveDistribution::from_draws(
            vec!["origin".into()],
            fit.origins.iter().map(|&o| vec![o.into()]).collect(),
            draws.cdr,
            provenance,
        )?;
        Ok(BayesOneYearFit {
            opening_ultimate: fit
                .latest_paid
                .iter()
                .zip(&draws.opening)
                .map(|(l, r)| l + r)
                .collect(),
            fit,
            opening_reserve: draws.opening,
            cdr,
            pareto_k: draws.pareto_k,
            refitted: draws.refitted,
            moment_matched: draws.moment_matched,
            unrefitted: draws.unrefitted,
            refit_warnings: draws.refit_warnings,
            k_threshold: settings.k_threshold,
            approximate,
        })
    }

    /// The one-year view's refit mode when its settings' `refit` is `None`
    /// (the user's decision of 2026-10-16, `docs/design/reserving-bayes.md`
    /// decision 20): [`OneYearRefit::ReweightOnly`] with development effects
    /// in the priors (model 2), whose flagged simulations moment matching
    /// does not rescue and whose refits each take about as long as a fit;
    /// [`OneYearRefit::MomentMatchThenRefit`] without (model 1).
    pub fn one_year_refit(&self) -> OneYearRefit {
        if self
            .priors
            .development_effect_sd
            .iter()
            .any(Option::is_some)
        {
            OneYearRefit::ReweightOnly
        } else {
            OneYearRefit::MomentMatchThenRefit
        }
    }

    /// `settings` with this model's one-year defaults in place of a `None`
    /// refit mode ([`one_year_refit`](Self::one_year_refit)), the number of
    /// one-year simulations (the settings' `n_sims`, or this model's capped
    /// at [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS)), and the view's
    /// [`approximate`](BayesOneYearFit::approximate) warning: set when the
    /// default leaves the flagged simulations reweighted only.
    fn one_year_defaults(&self, settings: &BayesOneYear) -> (BayesOneYear, usize, Option<String>) {
        let default_refit = self.one_year_refit();
        let approximate = (settings.refit.is_none() && default_refit == OneYearRefit::ReweightOnly)
            .then(|| APPROXIMATE.to_string());
        let n_sims = settings.sims(self.n_sims.min(Self::ONE_YEAR_SIMS));
        (
            settings.with_refit_default(default_refit),
            n_sims,
            approximate,
        )
    }

    /// Paths of the mean outstanding and cumulative paid of each origin of
    /// the premium column at each age of the square, drawn from the priors
    /// alone (Gesmann and Morris, sec. 3.4): a check of a prior set before
    /// fitting. Each simulation draws the intercepts, the effect sds, the
    /// correlation and each origin's effects, and with development effects
    /// in the priors each age's effects (shared by the origins); with
    /// effects by cell and incremental paid, the cumulative paid path is
    /// the sum of the cells' mean increments. The paths are means, without
    /// the likelihood's noise: the preset's prior on `log σ` is effectively
    /// flat, so its draws of `σ` would swamp them. Under
    /// [`RateOrder::Ordered`] the rate intercepts are drawn from their priors
    /// restricted to `k_er > k_p`, exactly; if those priors put less than
    /// `1e-6` of their mass on `k_er > k_p` the check is an
    /// [`Error::Bayes`]. The seed is the simulations' derived
    /// seed of [`seed`](Self::seed).
    pub fn prior_predictive(
        &self,
        triangle: &Triangle,
        premium: &str,
        n_sims: usize,
    ) -> Result<CompartmentalPaths> {
        self.validate_priors()?;
        if n_sims == 0 {
            return Err(Error::InvalidSetting {
                name: "n_sims",
                value: 0.0,
                expected: "a positive number of simulations",
            });
        }
        if self.rate_order == RateOrder::Ordered {
            let (a, _) = rate_order_bound(self.priors);
            let mass = norm_cdf(-a);
            if mass < MIN_ORDERED_MASS {
                return Err(Error::Bayes(format!(
                    "the rate priors give almost no mass to k_er > k_p ({mass:.1e}, \
                     below {MIN_ORDERED_MASS:.0e}), so the prior check under \
                     RateOrder::Ordered cannot draw from them; centre the rate \
                     priors with k_er above k_p, or use RateOrder::Free"
                )));
            }
        }
        let (origins, premiums, grain) = premium_of(triangle, premium)?;
        let n = origins.len();
        let ages: Vec<Lag> = (1..=n as u32).map(|k| k * grain.months()).collect();
        let times: Vec<f64> = ages.iter().map(|&a| f64::from(a) / 12.0).collect();
        let pr = self.priors;
        let ordered = self.rate_order == RateOrder::Ordered;
        let cellwise = pr.rates_vary() || pr.development_effect_sd.iter().any(Option::is_some);
        let incremental = self.paid_response == PaidResponse::Incremental;
        let mut keys = Vec::with_capacity(n * n);
        for &o in &origins {
            for &a in &ages {
                keys.push(vec![o.into(), KeyValue::Int(i64::from(a))]);
            }
        }
        let provenance = Provenance::new("compartmental_prior_predictive")
            .param("n_sims", n_sims)
            .param("seed", self.seed)
            .param("rate_order", format!("{:?}", self.rate_order))
            .param("priors", format!("{pr:?}"))
            .param("paid_response", format!("{:?}", self.paid_response))
            .param("premium", premium)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"));
        let both = PredictiveDistribution::simulate(
            vec!["cell".into()],
            (0..2 * n * n)
                .map(|j| vec![KeyValue::Int(j as i64)])
                .collect(),
            n_sims,
            derive_seed(self.seed, Stream::Predictive),
            provenance.clone(),
            |rng, row| {
                let normal = |rng: &mut StreamRng| norm_quantile(rng.next_open01());
                let (m_er, s_er) = lognormal(pr.k_er);
                let (m_p, s_p) = lognormal(pr.k_p);
                let (m_rlr, s_rlr) = lognormal(pr.rlr);
                let (m_rrf, s_rrf) = lognormal(pr.rrf);
                let (b_ker, b_kp) = if ordered {
                    ordered_rates(pr, rng)
                } else {
                    (normal(rng), normal(rng))
                };
                let k_er = m_er * (s_er * b_ker).exp();
                let k_p = m_p * (s_p * b_kp).exp();
                let b_rlr = normal(rng);
                let b_rrf = normal(rng);
                let tau1 = draw_half(pr.rlr_effect_sd, rng);
                let tau2 = draw_half(pr.rrf_effect_sd, rng);
                let gamma = Gamma::new(pr.lkj_eta, 1.0).expect("lkj_eta was validated");
                let (g1, g2) = (gamma.sample(rng, 1)[0], gamma.sample(rng, 1)[0]);
                let rho = 2.0 * g1 / (g1 + g2) - 1.0;
                let rho_c = (1.0 - rho * rho).max(0.0).sqrt();
                if !cellwise {
                    // Model 1's hierarchy: shared rates, one curve per age.
                    let curves: Vec<(f64, f64)> =
                        times.iter().map(|&t| one_os(k_er, k_p, t)).collect();
                    for i in 0..n {
                        let (z1, z2) = (normal(rng), normal(rng));
                        let u1 = tau1 * z1;
                        let u2 = tau2 * (rho * z1 + rho_c * z2);
                        let rlr = m_rlr * (s_rlr * (b_rlr + u1)).exp() * premiums[i];
                        let rrf = m_rrf * (s_rrf * (b_rrf + u2)).exp();
                        for (k, (os, pd)) in curves.iter().enumerate() {
                            row[i * n + k] = rlr * os;
                            row[n * n + i * n + k] = rlr * rrf * pd;
                        }
                    }
                    return;
                }
                // Effects by cell: the origin effects on the rates and the
                // development effects, each drawn from its population.
                let tau_rate =
                    [pr.ker_effect_sd, pr.kp_effect_sd].map(|p| p.map(|h| draw_half(h, rng)));
                let tau_dev = pr
                    .development_effect_sd
                    .map(|p| p.map(|h| draw_half(h, rng)));
                let dev: Vec<[f64; 4]> = (0..n)
                    .map(|_| tau_dev.map(|t| t.map_or(0.0, |t| t * normal(rng))))
                    .collect();
                let medians = [(m_er, s_er), (m_p, s_p), (m_rlr, s_rlr), (m_rrf, s_rrf)];
                for i in 0..n {
                    let (z1, z2) = (normal(rng), normal(rng));
                    let u = [
                        tau_rate[0].map_or(0.0, |t| t * normal(rng)),
                        tau_rate[1].map_or(0.0, |t| t * normal(rng)),
                        tau1 * z1,
                        tau2 * (rho * z1 + rho_c * z2),
                    ];
                    let b = [b_ker, b_kp, b_rlr, b_rrf];
                    let mut cum = 0.0;
                    for (k, &t) in times.iter().enumerate() {
                        let [ker, kp, rlr, rrf] = [0, 1, 2, 3].map(|p| {
                            medians[p].0 * (medians[p].1 * (b[p] + u[p] + dev[k][p])).exp()
                        });
                        let (os, pd) = one_os(ker, kp, t);
                        let base = rlr * premiums[i];
                        row[i * n + k] = base * os;
                        cum = if incremental {
                            let before = if k == 0 {
                                0.0
                            } else {
                                one_os(ker, kp, times[k - 1]).1
                            };
                            cum + base * rrf * (pd - before)
                        } else {
                            base * rrf * pd
                        };
                        row[n * n + i * n + k] = cum;
                    }
                }
            },
        )?;
        let split = |half: usize, name: &str| -> Result<PredictiveDistribution> {
            let draws: Vec<f64> = both
                .draw_matrix()
                .chunks_exact(2 * n * n)
                .flat_map(|r| r[half * n * n..(half + 1) * n * n].to_vec())
                .collect();
            Ok(PredictiveDistribution::from_draws(
                vec!["origin".into(), "age".into()],
                keys.clone(),
                draws,
                provenance.clone().param("output", name),
            )?)
        };
        Ok(CompartmentalPaths {
            outstanding: split(0, "outstanding")?,
            paid: split(1, "paid")?,
        })
    }

    /// NUTS on `model` with the chains' seed `chains`: a pilot run on a
    /// scale whitened at the start, then the main run on the pilot draws'
    /// scale, as [`Meyers`](super::Meyers) samples. The draws on the
    /// natural scale, the same draws on the unconstrained scale of `theta`
    /// (an affine map of the whitened scale NUTS samples on, for the
    /// one-year view) and their diagnostics.
    fn sample(
        &self,
        model: &Model,
        chains: u64,
    ) -> Result<(PosteriorDraws, PosteriorDraws, Diagnostics)> {
        let p = model.dim();
        let start = model.start(self.start);
        let rough = Whitened::regularised(model, &start)?;
        let pilot = sample(
            &rough,
            &vec![0.0; p],
            Sampler {
                seed: pilot_seed(chains),
                draws: self.sampler.tune.max(MIN_PILOT_DRAWS),
                ..self.sampler
            },
        )?;
        let pilot = pilot.transform(model.sampled_names(), |z| rough.theta(z))?;
        let whitened = Whitened::from_draws(model, &pilot)?;
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed: chains,
                ..self.sampler
            },
        )?;
        let sampled = draws.transform(model.sampled_names(), |z| whitened.theta(z))?;
        let posterior = sampled.transform(model.names(), |theta| model.natural(theta))?;
        let summary = posterior.summary()?;
        let diagnostics = Diagnostics::new(
            &summary,
            posterior.chains(),
            posterior.divergences().iter().sum(),
        );
        Ok((posterior, sampled, diagnostics))
    }

    fn validate_priors(&self) -> Result<()> {
        self.priors.validate()
    }

    fn response(&self) -> Response {
        Response {
            likelihood: self.likelihood,
            loss_ratio: self.scale == Scale::LossRatio,
            incremental: self.paid_response == PaidResponse::Incremental,
        }
    }

    fn validate(&self) -> Result<()> {
        let not_yet = |what: &str| {
            Err(Error::Bayes(format!(
                "{what} is not yet built; this release fits paid and outstanding together"
            )))
        };
        match self.fitted {
            Fitted::PaidAndOutstanding => {}
            Fitted::PaidAndIncurred => return not_yet("fitting paid and incurred"),
            Fitted::PaidOnly => return not_yet("fitting paid alone"),
            Fitted::IncurredOnly => return not_yet("fitting incurred alone"),
        }
        self.validate_priors()?;
        if self.n_sims == 0 {
            return Err(Error::InvalidSetting {
                name: "n_sims",
                value: 0.0,
                expected: "a positive number of simulations",
            });
        }
        Ok(())
    }

    fn provenance(
        &self,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        hasher: InputHasher,
    ) -> Provenance {
        let s = &self.sampler;
        let pr = &self.priors;
        let response = self.response();
        // Model 1 and model 2 by name when the settings and the hierarchy
        // are the paper's.
        let model = if response == Response::MODEL_1 && pr.hierarchy() == Hierarchy::Model1 {
            "compartmental_gm1"
        } else if response == Response::MODEL_2 && pr.hierarchy() == Hierarchy::Model2 {
            "compartmental_gm2"
        } else {
            "compartmental"
        };
        Provenance::new(model)
            .param("priors", format!("{pr:?}"))
            .param("likelihood", format!("{:?}", self.likelihood))
            .param("scale", format!("{:?}", self.scale))
            .param("paid_response", format!("{:?}", self.paid_response))
            .param("fitted", format!("{:?}", self.fitted))
            .param("rate_order", format!("{:?}", self.rate_order))
            .param("start", format!("{:?}", self.start))
            .param("chains", s.chains)
            .param("tune", s.tune)
            .param("draws", s.draws)
            .param("target_accept", s.target_accept)
            .param("max_depth", s.max_depth)
            .param("n_sims", self.n_sims)
            .param("seed", self.seed)
            .param("paid", paid)
            .param("outstanding", format!("{outstanding:?}"))
            .param("premium", premium)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

fn outstanding_name<'a>(outstanding: Outstanding<'a>) -> &'a str {
    match outstanding {
        Outstanding::Column(c) | Outstanding::FromIncurred(c) => c,
    }
}

/// Fewest kept draws per chain of the pilot run.
const MIN_PILOT_DRAWS: usize = 100;

/// Paths drawn from the priors alone ([`Compartmental::prior_predictive`]):
/// the mean outstanding and cumulative paid of each origin at each age of
/// the square, dimensions `origin` and `age` (months).
#[derive(Debug, Clone)]
pub struct CompartmentalPaths {
    /// The mean outstanding.
    pub outstanding: PredictiveDistribution,
    /// The mean cumulative paid.
    pub paid: PredictiveDistribution,
}

/// A fitted [`Compartmental`].
#[derive(Debug, Clone)]
pub struct CompartmentalFit {
    /// The origins, those of the premium column.
    pub origins: Vec<Period>,
    /// Each origin's premium, the latest value of the premium column.
    pub premium: Vec<f64>,
    /// Each origin's latest observed paid (0 with none).
    pub latest_paid: Vec<f64>,
    /// The origins with no observed cell, whose effects each simulation
    /// draws from the population (decision 2).
    pub population_origins: Vec<Period>,
    /// Posterior draws, by chain: on the natural scale `k_er`, `k_p`, `RLR`
    /// and `RRF` (the population values, at a zero effect), `sigma_os`,
    /// `sigma_pd`, then `RLR[o]` and `RRF[o]` for each origin `o` with an
    /// observed cell (at a zero development effect), with origin effects on
    /// the rates `k_er[o]` and `k_p[o]` too, and with development effects
    /// `dev_ker[a]`, `dev_kp[a]`, `dev_RLR[a]` and `dev_RRF[a]`, each
    /// development age `a` (months) with an observed cell's effect on the
    /// `o` scale; and on the scale the paper prints, the intercepts `b_ker`,
    /// `b_kp`, `b_RLR`, `b_RRF`, the effect sds `tau_RLR` and `tau_RRF` (and
    /// `tau_ker`, `tau_kp`) on the `o` scale, `rho` (and `tau_dev_ker`,
    /// `tau_dev_kp`, `tau_dev_RLR`, `tau_dev_RRF`), `log_sigma_os` and
    /// `log_sigma_pd`. Each name in parentheses is there only when the
    /// priors have that effect.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds and
    /// (under [`RateOrder::Free`]) for draws with `k_er <= k_p`.
    pub diagnostics: Diagnostics,
    /// The number of posterior draws whose population rates have
    /// `k_er <= k_p` (0 under [`RateOrder::Ordered`]).
    pub rate_swaps: usize,
    /// With effects on the rates (each cell its own `k_er` and `k_p`), the
    /// share of posterior draws and observed cells with `k_er <= k_p`: a
    /// diagnostic, not a warning, since each cell's means are invariant
    /// under the swap on their own (decision 2). `None` when the rates are
    /// shared.
    pub cell_rate_swaps: Option<f64>,
    /// Joint posterior predictive distribution of each origin's reserve, the
    /// paid at the square's last age less the latest paid: dimension
    /// `origin`, one component per origin, with the layout of
    /// [`OdpBootstrapFit::reserves`](crate::OdpBootstrapFit::reserves).
    pub reserves: PredictiveDistribution,
    /// Posterior predictive cumulative paid of each future cell of the square
    /// (after each origin's latest paid), dimensions `origin` and `age`
    /// (months), with the same simulations as the reserves.
    pub paid: PredictiveDistribution,
    /// Posterior predictive outstanding of each future cell of the square
    /// (after each origin's latest outstanding), the same layout.
    pub outstanding: PredictiveDistribution,
    /// Each origin's posterior mean ultimate loss ratio `RLR_i RRF_i`, at a
    /// zero development effect (for an origin drawn from the population, the
    /// mean over its effect too). The model's ultimate, without process
    /// error past the square.
    pub ultimate_loss_ratio: Vec<f64>,
    model: Model,
    /// The same draws on the unconstrained scale of `theta`, for the
    /// one-year view (`docs/design/reserving-bayes.md`, decision 20).
    sampled: PosteriorDraws,
    /// The observed cells, for the one-year view.
    data: Data,
}

/// One segment of [`Compartmental::fit_segments`]: its [`CompartmentalFit`],
/// whose `reserves`, `paid` and `outstanding` are the segment's part of the
/// joint distributions of [`CompartmentalFits`], after any reordering for
/// the rank correlation between segments.
#[derive(Debug, Clone)]
pub struct CompartmentalSegment {
    /// The segment's fit: posterior, diagnostics, rate swaps, `summary()`,
    /// `loo()`, `to_long()` and the rest.
    pub fit: CompartmentalFit,
}

/// The fits of every segment of a triangle ([`Compartmental::fit_segments`]),
/// with joint distributions across segments.
#[derive(Debug, Clone)]
pub struct CompartmentalFits {
    /// Each segment's fit.
    pub segments: SegmentFits<CompartmentalSegment>,
    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the key names and `origin` are its dimensions, and its
    /// components run over the origins of each segment in turn, like the
    /// rows of [`to_long`](Self::to_long).
    pub reserves: PredictiveDistribution,
    /// The future cumulative paid of every segment, dimensions the key names,
    /// `origin` and `age` (months), from the same simulations as
    /// [`reserves`](Self::reserves).
    pub paid: PredictiveDistribution,
    /// The future outstanding of every segment, the same layout.
    pub outstanding: PredictiveDistribution,
}

impl CompartmentalFits {
    /// The joint distributions of `segments`' own parts, simulation by
    /// simulation, with `provenance` (and their `output`).
    fn assemble(
        segments: SegmentFits<CompartmentalSegment>,
        provenance: Provenance,
    ) -> Result<Self> {
        let joint = |part: fn(&CompartmentalFit) -> &PredictiveDistribution,
                     output: &str|
         -> Result<PredictiveDistribution> {
            let first = part(&segments.fits[0].fit);
            let mut dims = segments.key_names.clone();
            dims.extend(first.dims().iter().cloned());
            let mut components = Vec::new();
            for (label, s) in segments.iter() {
                for key in part(&s.fit).components() {
                    let mut k: Vec<KeyValue> =
                        label.parts().iter().map(|p| p.as_str().into()).collect();
                    k.extend(key.iter().cloned());
                    components.push(k);
                }
            }
            let n_sims = first.n_sims();
            let mut draws = Vec::with_capacity(n_sims * components.len());
            for i in 0..n_sims {
                for s in &segments.fits {
                    let p = part(&s.fit);
                    let n = p.n_components();
                    draws.extend_from_slice(&p.draw_matrix()[i * n..(i + 1) * n]);
                }
            }
            let mut p = provenance.clone();
            p.parameters.retain(|(k, _)| k != "output");
            Ok(PredictiveDistribution::from_draws(
                dims,
                components,
                draws,
                p.param("output", output),
            )?)
        };
        Ok(Self {
            reserves: joint(|f| &f.reserves, "reserves")?,
            paid: joint(|f| &f.paid, "paid")?,
            outstanding: joint(|f| &f.outstanding, "outstanding")?,
            segments,
        })
    }

    /// The segments' tables, stacked in index order under the key columns.
    fn stack(&self, tables: Vec<FitTable>) -> FitTable {
        let rows: Vec<usize> = tables
            .iter()
            .map(|t| t.origin.as_ref().map_or(1, Vec::len))
            .collect();
        let mut stacked = FitTable {
            keys: self.segments.key_columns(|s| rows[s]),
            origin: tables[0].origin.as_ref().map(|_| Vec::new()),
            age: tables[0].age.as_ref().map(|_| Vec::new()),
            values: tables[0]
                .values
                .iter()
                .map(|(name, _)| (name.clone(), Vec::new()))
                .collect(),
        };
        for t in tables {
            if let (Some(all), Some(o)) = (&mut stacked.origin, t.origin) {
                all.extend(o);
            }
            if let (Some(all), Some(a)) = (&mut stacked.age, t.age) {
                all.extend(a);
            }
            for ((_, all), (_, v)) in stacked.values.iter_mut().zip(t.values) {
                all.extend(v);
            }
        }
        stacked
    }

    /// One row per segment × origin: the columns of
    /// [`CompartmentalFit::to_long`] (`premium`, `latest_paid`, the `mean`
    /// and `std_dev` of the reserve, `ultimate_loss_ratio`).
    pub fn to_long(&self) -> FitTable {
        self.stack(self.segments.fits.iter().map(|s| s.fit.to_long()).collect())
    }

    /// One row per segment: the columns of [`CompartmentalFit::totals`].
    pub fn totals(&self) -> FitTable {
        self.stack(self.segments.fits.iter().map(|s| s.fit.totals()).collect())
    }

    /// One row per segment × future cell of [`paid`](Self::paid), as
    /// [`CompartmentalFit::paid_table`].
    pub fn paid_table(&self) -> Result<FitTable> {
        let tables = self.segments.fits.iter().map(|s| s.fit.paid_table());
        Ok(self.stack(tables.collect::<Result<_>>()?))
    }

    /// The same for [`outstanding`](Self::outstanding).
    pub fn outstanding_table(&self) -> Result<FitTable> {
        let tables = self.segments.fits.iter().map(|s| s.fit.outstanding_table());
        Ok(self.stack(tables.collect::<Result<_>>()?))
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its part
    /// of the joint distributions (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let segments = self.segments.segment(keys)?;
        Self::assemble(segments, self.reserves.provenance().clone())
    }
}

/// The one-year view of a compartmental model
/// ([`Compartmental::one_year`]): its tables, by the premium column's
/// origins, as [`CompartmentalFit`]'s.
impl BayesOneYearFit<CompartmentalFit> {
    /// The share of simulations whose `k̂` is above the threshold (or
    /// NaN), whether or not they were refitted.
    pub fn flagged_share(&self) -> f64 {
        one_year::flagged_share(&self.pareto_k, self.k_threshold)
    }

    /// One row per origin: `premium`, `latest_paid`, `opening_ultimate`,
    /// `opening_reserve`, and the `cdr_mean` and `cdr_std_dev` of the
    /// claims development result.
    pub fn to_long(&self) -> FitTable {
        let mut table = FitTable {
            keys: Vec::new(),
            origin: Some(self.fit.origins.clone()),
            age: None,
            values: vec![
                ("premium".into(), self.fit.premium.clone()),
                ("latest_paid".into(), self.fit.latest_paid.clone()),
                ("opening_ultimate".into(), self.opening_ultimate.clone()),
                ("opening_reserve".into(), self.opening_reserve.clone()),
            ],
        };
        push_moments(&mut table, "cdr_", &component_sums(&self.cdr));
        table
    }

    /// One row: the totals of [`to_long`](Self::to_long)'s columns, the
    /// `flagged_share`, and the `cdr_mean` and `cdr_std_dev` of the total
    /// claims development result.
    pub fn totals(&self) -> FitTable {
        let total = self.cdr.total();
        let sum = |v: &[f64]| vec![v.iter().sum::<f64>()];
        FitTable {
            keys: Vec::new(),
            origin: None,
            age: None,
            values: vec![
                ("premium".into(), sum(&self.fit.premium)),
                ("latest_paid".into(), sum(&self.fit.latest_paid)),
                ("opening_ultimate".into(), sum(&self.opening_ultimate)),
                ("opening_reserve".into(), sum(&self.opening_reserve)),
                ("flagged_share".into(), vec![self.flagged_share()]),
                ("cdr_mean".into(), vec![total.mean()]),
                ("cdr_std_dev".into(), vec![total.std_dev()]),
            ],
        }
    }
}

/// [`CompartmentalFit::loo`]: PSIS-LOO over every observed cell, with the
/// elpd split by type.
#[derive(Debug, Clone)]
pub struct CompartmentalLoo {
    /// PSIS-LOO over the outstanding cells and then the paid cells, each by
    /// origin and age.
    pub loo: Loo,
    /// The number of outstanding cells, which come first.
    pub n_outstanding: usize,
    /// The elpd of the outstanding cells.
    pub elpd_outstanding: f64,
    /// The elpd of the paid cells.
    pub elpd_paid: f64,
}

impl CompartmentalFit {
    /// Posterior summary of each parameter: mean, sd, 5/50/95% quantiles,
    /// R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the observed cells, with each cell's relative efficiency
    /// from the chains. The pointwise term is the full log density of each
    /// cell's response under the fit's likelihood (brms' `log_lik`): normal,
    /// or log-normal including its `-ln y` term, on the fit's scale (amounts
    /// or loss ratios to premium) and paid response (cumulative or
    /// incremental). So elpd_loo compares only fits with the same scale and
    /// paid response: model 2's elpd on loss ratios does not compare with
    /// model 1's on amounts. Check each cell's Pareto k: model 2 on the
    /// papers' training triangle has a few above 0.7 (up to about 1.3), so
    /// there the estimate is rough.
    pub fn loo(&self) -> Result<CompartmentalLoo> {
        let model = &self.model;
        let n = model.cells.len();
        let post = &self.posterior;
        let s = post.n_draws();
        let mut ll = Vec::with_capacity(s * n);
        for nat in post.values().chunks_exact(post.dim()) {
            model.log_likelihoods(nat, &mut ll);
        }
        let chains = post.chains();
        let per_chain = s / chains;
        let r_eff: Vec<f64> = (0..n)
            .map(|i| {
                let by_chain: Vec<Vec<f64>> = (0..chains)
                    .map(|c| {
                        (0..per_chain)
                            .map(|k| ll[(c * per_chain + k) * n + i].exp())
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[f64]> = by_chain.iter().map(Vec::as_slice).collect();
                let ess = prospicio_bayes::ess_mean(&refs).unwrap_or(s as f64);
                if ess.is_finite() && ess > 0.0 {
                    ess / s as f64
                } else {
                    1.0
                }
            })
            .collect();
        let loo = prospicio_bayes::elpd::loo(&ll, n, Some(&r_eff))?;
        let n_os = model.n_outstanding;
        let elpd_outstanding = loo.estimate.pointwise[..n_os].iter().sum();
        let elpd_paid = loo.estimate.pointwise[n_os..].iter().sum();
        Ok(CompartmentalLoo {
            loo,
            n_outstanding: n_os,
            elpd_outstanding,
            elpd_paid,
        })
    }

    /// One row per origin: `premium`, `latest_paid`, the `mean` and
    /// `std_dev` of the posterior predictive reserve, and the posterior mean
    /// `ultimate_loss_ratio`.
    pub fn to_long(&self) -> FitTable {
        let sums = component_sums(&self.reserves);
        FitTable {
            keys: Vec::new(),
            origin: Some(self.origins.clone()),
            age: None,
            values: vec![
                ("premium".into(), self.premium.clone()),
                ("latest_paid".into(), self.latest_paid.clone()),
                ("mean".into(), sums.iter().map(|d| d.mean()).collect()),
                ("std_dev".into(), sums.iter().map(|d| d.std_dev()).collect()),
                (
                    "ultimate_loss_ratio".into(),
                    self.ultimate_loss_ratio.clone(),
                ),
            ],
        }
    }

    /// One row: the total `premium` and `latest_paid`, and the `mean` and
    /// `std_dev` of the posterior predictive total reserve.
    pub fn totals(&self) -> FitTable {
        let total = self.reserves.total();
        FitTable {
            keys: Vec::new(),
            origin: None,
            age: None,
            values: vec![
                ("premium".into(), vec![self.premium.iter().sum()]),
                ("latest_paid".into(), vec![self.latest_paid.iter().sum()]),
                ("mean".into(), vec![total.mean()]),
                ("std_dev".into(), vec![total.std_dev()]),
            ],
        }
    }

    /// One row per future cell of [`paid`](Self::paid): its origin and age,
    /// and the `mean`, `std_dev` and 5%, 50% and 95% quantiles (`q05`,
    /// `q50`, `q95`) of the posterior predictive cumulative paid.
    pub fn paid_table(&self) -> Result<FitTable> {
        cell_table(&self.paid)
    }

    /// The same for [`outstanding`](Self::outstanding).
    pub fn outstanding_table(&self) -> Result<FitTable> {
        cell_table(&self.outstanding)
    }
}

/// A long table of a cell distribution with dimensions `origin` and `age`.
fn cell_table(cells: &PredictiveDistribution) -> Result<FitTable> {
    let sums = component_sums(cells);
    let mut origin = Vec::with_capacity(sums.len());
    let mut age = Vec::with_capacity(sums.len());
    for key in cells.components() {
        match (&key[0], &key[1]) {
            (KeyValue::Period(o), KeyValue::Int(a)) => {
                origin.push(*o);
                age.push(*a as Lag);
            }
            _ => unreachable!("cell keys are (origin, age)"),
        }
    }
    let q = |p: f64| -> Result<Vec<f64>> {
        sums.iter()
            .map(|d| Ok(d.quantile(p)?))
            .collect::<Result<Vec<f64>>>()
    };
    Ok(FitTable {
        keys: Vec::new(),
        origin: Some(origin),
        age: Some(age),
        values: vec![
            ("mean".into(), sums.iter().map(|d| d.mean()).collect()),
            ("std_dev".into(), sums.iter().map(|d| d.std_dev()).collect()),
            ("q05".into(), q(0.05)?),
            ("q50".into(), q(0.5)?),
            ("q95".into(), q(0.95)?),
        ],
    })
}

/// Each origin of the premium column, its premium (the latest value, which
/// must be finite and positive) and the grain of the square.
fn premium_of(triangle: &Triangle, premium: &str) -> Result<(Vec<Period>, Vec<f64>, Grain)> {
    let grain = triangle.development_grain();
    if triangle.origin_grain() != grain {
        return Err(Error::Bayes(format!(
            "origin grain {} and development grain {} differ; the compartmental model's \
             square needs them equal (not yet built otherwise)",
            triangle.origin_grain(),
            grain
        )));
    }
    let segment = triangle.segment(premium)?;
    let premiums = (0..segment.n_origins)
        .map(|o| {
            segment
                .latest(o)
                .ok()
                .map(|(_, v)| v)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    Error::Bayes(format!(
                        "origin {} has no positive premium in column {premium}",
                        segment.origins[o]
                    ))
                })
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok((segment.origins.clone(), premiums, grain))
}

/// The observed cells of a fit, by the premium column's origins.
#[derive(Debug, Clone)]
struct Data {
    origins: Vec<Period>,
    premium: Vec<f64>,
    grain: Grain,
    /// The square's number of ages.
    n_ages: usize,
    /// `(origin, age index, value)` of each observed outstanding cell.
    outstanding: Vec<(usize, usize, f64)>,
    /// The same for paid.
    paid: Vec<(usize, usize, f64)>,
    /// Each origin's latest observed paid: age index and value.
    latest_paid: Vec<Option<(usize, f64)>>,
    /// Each origin's latest observed outstanding age index.
    latest_outstanding: Vec<Option<usize>>,
}

/// The future cells of the square: `(origin, age index)` of each paid and
/// outstanding cell, and the position in `paid` of each origin's cell at
/// the last age, if it is in the future.
#[derive(Debug, Clone)]
struct Future {
    paid: Vec<(usize, usize)>,
    outstanding: Vec<(usize, usize)>,
    last_paid: Vec<Option<usize>>,
}

impl Data {
    fn new(
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
    ) -> Result<Self> {
        let (origins, premiums, grain) = premium_of(triangle, premium)?;
        let premium_segment = triangle.segment(premium)?;
        let paid_segment = triangle.segment(paid)?;
        let months = grain.months();
        // The fit's origin of a segment's origin `o`.
        let origin_of = |segment: &Segment, o: usize, column: &str| -> Result<usize> {
            (segment.origin_offset + o)
                .checked_sub(premium_segment.origin_offset)
                .filter(|&i| i < origins.len())
                .ok_or_else(|| {
                    Error::Bayes(format!(
                        "origin {} has {column} but no premium in column {premium}",
                        segment.origins[o]
                    ))
                })
        };
        let age_index = |age: Lag| -> Result<usize> {
            if age == 0 || age % months != 0 {
                return Err(Error::Bayes(format!(
                    "development age {age} months is not a whole number of {grain} periods"
                )));
            }
            Ok((age / months - 1) as usize)
        };
        let cells_of = |segment: &Segment, column: &str| -> Result<Vec<(usize, usize, f64)>> {
            let mut out = Vec::new();
            for o in 0..segment.n_origins {
                for d in 0..segment.n_dev {
                    if let Some(v) = segment.get(o, d) {
                        if !v.is_finite() {
                            return Err(Error::Bayes(format!(
                                "origin {} at development age {} months has {column} {v}",
                                segment.origins[o], segment.ages[d]
                            )));
                        }
                        out.push((
                            origin_of(segment, o, column)?,
                            age_index(segment.ages[d])?,
                            v,
                        ));
                    }
                }
            }
            Ok(out)
        };
        let paid_cells = cells_of(&paid_segment, paid)?;
        let os_cells = match outstanding {
            Outstanding::Column(column) => cells_of(&triangle.segment(column)?, column)?,
            Outstanding::FromIncurred(column) => {
                let incurred = cells_of(&triangle.segment(column)?, column)?;
                incurred
                    .into_iter()
                    .filter_map(|(i, k, v)| {
                        paid_cells
                            .iter()
                            .find(|c| c.0 == i && c.1 == k)
                            .map(|c| (i, k, v - c.2))
                    })
                    .collect()
            }
        };
        if os_cells.is_empty() {
            return Err(Error::Bayes(format!(
                "column {} gives no outstanding cell; the model needs the outstanding column \
                 (from paid alone only RLR x RRF is identified)",
                outstanding_name(outstanding)
            )));
        }
        let n = origins.len();
        let observed_ages = paid_cells
            .iter()
            .chain(&os_cells)
            .map(|c| c.1 + 1)
            .max()
            .unwrap_or(0);
        let n_ages = n.max(observed_ages);
        let mut data = Self {
            origins,
            premium: premiums,
            grain,
            n_ages,
            outstanding: os_cells,
            paid: paid_cells,
            latest_paid: Vec::new(),
            latest_outstanding: Vec::new(),
        };
        data.set_latest();
        Ok(data)
    }

    /// Each origin's latest paid and outstanding, from the cells.
    fn set_latest(&mut self) {
        let n = self.origins.len();
        self.latest_paid = vec![None; n];
        for &(i, k, v) in &self.paid {
            if self.latest_paid[i].is_none_or(|(l, _)| k > l) {
                self.latest_paid[i] = Some((k, v));
            }
        }
        self.latest_outstanding = vec![None; n];
        for &(i, k, _) in &self.outstanding {
            if self.latest_outstanding[i].is_none_or(|l| k > l) {
                self.latest_outstanding[i] = Some(k);
            }
        }
    }

    /// The same data with the cells `paid` (cumulative) and `outstanding`
    /// added, each `(origin, age index, value)` within the square.
    fn extended(
        &self,
        paid: impl IntoIterator<Item = (usize, usize, f64)>,
        outstanding: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> Self {
        let mut data = self.clone();
        data.paid.extend(paid);
        data.outstanding.extend(outstanding);
        data.set_latest();
        data
    }

    /// The age in months of age index `k`.
    fn age(&self, k: usize) -> Lag {
        (k as Lag + 1) * self.grain.months()
    }

    fn future_cells(&self) -> Future {
        let n = self.origins.len();
        let mut paid = Vec::new();
        let mut last_paid = vec![None; n];
        for (i, last) in last_paid.iter_mut().enumerate() {
            let from = self.latest_paid[i].map_or(0, |(k, _)| k + 1);
            for k in from..self.n_ages {
                if k + 1 == self.n_ages {
                    *last = Some(paid.len());
                }
                paid.push((i, k));
            }
        }
        let mut outstanding = Vec::new();
        for i in 0..n {
            let from = self.latest_outstanding[i].map_or(0, |k| k + 1);
            for k in from..self.n_ages {
                outstanding.push((i, k));
            }
        }
        Future {
            paid,
            outstanding,
            last_paid,
        }
    }
}

/// The response settings of a fit (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Response {
    likelihood: CompartmentalLikelihood,
    /// Loss ratios to premium (else amounts).
    loss_ratio: bool,
    /// Incremental paid (else cumulative).
    incremental: bool,
}

impl Response {
    /// Gesmann and Morris' model 1: normal on amounts, cumulative paid.
    const MODEL_1: Self = Self {
        likelihood: CompartmentalLikelihood::Normal,
        loss_ratio: false,
        incremental: false,
    };
    /// Their model 2: log-normal on loss ratios, incremental paid.
    const MODEL_2: Self = Self {
        likelihood: CompartmentalLikelihood::LogNormal,
        loss_ratio: true,
        incremental: true,
    };

    fn log_normal(self) -> bool {
        self.likelihood == CompartmentalLikelihood::LogNormal
    }
}

/// Which of the paper's hierarchies a prior set has, for the provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hierarchy {
    /// Origin effects on `RLR` and `RRF` only.
    Model1,
    /// Origin effects on all four parameters and development effects on
    /// all four.
    Model2,
    Other,
}

impl CompartmentalPriors {
    fn hierarchy(&self) -> Hierarchy {
        let extra = [
            self.ker_effect_sd.is_some(),
            self.kp_effect_sd.is_some(),
            self.development_effect_sd[0].is_some(),
            self.development_effect_sd[1].is_some(),
            self.development_effect_sd[2].is_some(),
            self.development_effect_sd[3].is_some(),
        ];
        if extra.iter().all(|&e| !e) {
            Hierarchy::Model1
        } else if extra.iter().all(|&e| e) {
            Hierarchy::Model2
        } else {
            Hierarchy::Other
        }
    }

    /// The four positive priors, in the order `k_er`, `k_p`, `RLR`, `RRF`.
    fn positive(&self) -> [PositivePrior; 4] {
        [self.k_er, self.k_p, self.rlr, self.rrf]
    }

    /// The origin effect sd priors on the rates, `k_er` then `k_p`.
    fn rate_effect_sd(&self) -> [Option<HalfPrior>; 2] {
        [self.ker_effect_sd, self.kp_effect_sd]
    }
}

/// The four parameters, in the order of [`CompartmentalPriors::positive`].
const KER: usize = 0;
const KP: usize = 1;
const RLR: usize = 2;
const RRF: usize = 3;
/// Their names in the posterior's origin values (`RLR[1990]`) and, short,
/// in the names of their effects (`tau_dev_ker`, `dev_ker[24]`).
const NATURAL: [&str; 4] = ["k_er", "k_p", "RLR", "RRF"];
const SHORT: [&str; 4] = ["ker", "kp", "RLR", "RRF"];

/// An observed cell.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Obs {
    origin: usize,
    /// The age index, into [`Model::times`].
    k: usize,
    t: f64,
    /// For an incremental paid cell after the first age, the time of the
    /// age before.
    t_before: Option<f64>,
    /// The response: outstanding, cumulative or incremental paid, as an
    /// amount or a loss ratio.
    y: f64,
    /// `ln y` under the log-normal likelihood, 0 otherwise.
    ln_y: f64,
    /// The premium under the amount scale, 1 under the loss-ratio scale.
    w: f64,
    /// Paid (true) or outstanding.
    paid: bool,
}

/// Positions in `theta` past the fixed ones, and in a natural-scale draw.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    /// `log τ` of the origin effects on `k_er` and `k_p`.
    tau_rate: [Option<usize>; 2],
    /// `log τ` of the development effects on each parameter.
    tau_dev: [Option<usize>; 4],
    /// The first `z`: the origin pairs of `RLR` and `RRF`.
    z0: usize,
    /// The first `z` of the origin effects on `k_er` and `k_p`.
    z_rate: [Option<usize>; 2],
    /// The first `z` of the development effects on each parameter.
    z_dev: [Option<usize>; 4],
    /// `log σ_OS`; `log σ_PD` follows.
    log_sigma0: usize,
    /// In a natural-scale draw: the first origin value of each parameter.
    nat_origin: [Option<usize>; 4],
    /// The first development effect (on the `o` scale) of each parameter.
    nat_dev: [Option<usize>; 4],
    /// `b_ker`, then `b_kp`, `b_RLR`, `b_RRF`, `tau_RLR` and `tau_RRF`.
    nat_b0: usize,
    nat_tau_rate: [Option<usize>; 2],
    nat_rho: usize,
    nat_tau_dev: [Option<usize>; 4],
    /// `log_sigma_os`; `log_sigma_pd` follows.
    nat_log_sigma: usize,
}

impl Layout {
    fn new(priors: &CompartmentalPriors, n_levels: usize, n_dev_levels: usize) -> Self {
        let rate = priors.rate_effect_sd();
        let dev = priors.development_effect_sd;
        let mut at = ATANH_RHO + 1;
        let next = |on: bool, size: usize, at: &mut usize| {
            on.then(|| {
                *at += size;
                *at - size
            })
        };
        let tau_rate = rate.map(|p| next(p.is_some(), 1, &mut at));
        let tau_dev = dev.map(|p| next(p.is_some(), 1, &mut at));
        let z0 = at;
        at += 2 * n_levels;
        let z_rate = rate.map(|p| next(p.is_some(), n_levels, &mut at));
        let z_dev = dev.map(|p| next(p.is_some(), n_dev_levels, &mut at));
        let log_sigma0 = at;
        let mut n = 6 + 2 * n_levels;
        let nat_origin = [
            next(rate[0].is_some(), n_levels, &mut n),
            next(rate[1].is_some(), n_levels, &mut n),
            Some(6),
            Some(6 + n_levels),
        ];
        let nat_dev = dev.map(|p| next(p.is_some(), n_dev_levels, &mut n));
        let nat_b0 = n;
        n += 6;
        let nat_tau_rate = rate.map(|p| next(p.is_some(), 1, &mut n));
        let nat_rho = n;
        n += 1;
        let nat_tau_dev = dev.map(|p| next(p.is_some(), 1, &mut n));
        Self {
            tau_rate,
            tau_dev,
            z0,
            z_rate,
            z_dev,
            log_sigma0,
            nat_origin,
            nat_dev,
            nat_b0,
            nat_tau_rate,
            nat_rho,
            nat_tau_dev,
            nat_log_sigma: n,
        }
    }
}

/// The log posterior on the unconstrained `theta = (b_ker | delta, b_kp,
/// b_RLR, b_RRF, log τ_RLR, log τ_RRF, atanh ρ, [log τ_ker, log τ_kp,
/// log τ_dev_ker, .., log τ_dev_RRF], z_11, z_12, .., z_m1, z_m2, [z_ker,
/// z_kp by origin, z_dev_ker, .., z_dev_RRF by development level],
/// log σ_OS, log σ_PD)` over the `m` origins with an observed cell, where
/// the bracketed blocks are there only for the effects the priors have and
/// under [`RateOrder::Ordered`] `k_er = k_p + e^delta` (the population
/// rates). A cell's parameters are `θ = m e^(s (b + u_origin + u_dev))`.
#[derive(Debug, Clone, PartialEq)]
struct Model {
    /// Outstanding cells first, then paid, each by origin and age.
    cells: Vec<Obs>,
    n_outstanding: usize,
    premium: Vec<f64>,
    /// Each origin's latest observed cumulative paid (0 with none).
    latest_paid: Vec<f64>,
    /// Each origin's level (position among the origins with data), if any.
    level: Vec<Option<usize>>,
    /// The origin of each level.
    level_origin: Vec<usize>,
    /// The origin label of each level, for the names.
    labels: Vec<String>,
    /// Each age index's development level (position among the ages with an
    /// observed cell), if any.
    dev_level: Vec<Option<usize>>,
    /// The age in months of each development level, for the names.
    dev_labels: Vec<String>,
    /// Development time of each age index of the square, in years.
    times: Vec<f64>,
    /// The age indices that hold an observed cell, `0..n_observed`.
    n_observed: usize,
    priors: CompartmentalPriors,
    ordered: bool,
    response: Response,
    /// Whether any effect varies by cell beyond `RLR` and `RRF` by origin
    /// (model 1 has none: its rates are shared, one curve per age).
    cellwise: bool,
    layout: Layout,
}

/// The most entries of `theta` without a `z` (model 2: 13 and the two
/// `log σ`), the dual width of [`Model::prior_globals`].
const GLOBALS: usize = 16;

/// Positions in `theta`.
const B_KER: usize = 0;
const B_KP: usize = 1;
const B_RLR: usize = 2;
const B_RRF: usize = 3;
const LOG_TAU: usize = 4;
const ATANH_RHO: usize = 6;

impl Model {
    /// The model of `data`'s observed cells: a level (with its `z`) for
    /// each origin and each age with an observed cell.
    fn new(
        data: &Data,
        priors: CompartmentalPriors,
        ordered: bool,
        response: Response,
    ) -> Result<Self> {
        Self::build(data, priors, ordered, response, false)
    }

    /// The model of `data`'s observed cells with a level for every origin
    /// of the premium column and every age of the square, observed or not
    /// (`docs/design/reserving-bayes.md`, decision 20): an effect no cell
    /// informs has only its prior, so this posterior is [`new`](Self::new)'s
    /// with independent standard normal `z` for the unobserved levels.
    fn full(
        data: &Data,
        priors: CompartmentalPriors,
        ordered: bool,
        response: Response,
    ) -> Result<Self> {
        Self::build(data, priors, ordered, response, true)
    }

    fn build(
        data: &Data,
        priors: CompartmentalPriors,
        ordered: bool,
        response: Response,
        all_levels: bool,
    ) -> Result<Self> {
        let n = data.origins.len();
        let mut has = vec![all_levels; n];
        let mut has_age = vec![all_levels; data.n_ages];
        for c in data.outstanding.iter().chain(&data.paid) {
            has[c.0] = true;
            has_age[c.1] = true;
        }
        let mut level = vec![None; n];
        let mut level_origin = Vec::new();
        for i in 0..n {
            if has[i] {
                level[i] = Some(level_origin.len());
                level_origin.push(i);
            }
        }
        let labels = level_origin
            .iter()
            .map(|&i| data.origins[i].to_string())
            .collect();
        let mut dev_level = vec![None; data.n_ages];
        let mut dev_labels = Vec::new();
        for k in 0..data.n_ages {
            if has_age[k] {
                dev_level[k] = Some(dev_labels.len());
                dev_labels.push(data.age(k).to_string());
            }
        }
        let times: Vec<f64> = (0..data.n_ages)
            .map(|k| f64::from(data.age(k)) / 12.0)
            .collect();
        let log_normal = response.log_normal();
        let obs =
            |origin: usize, k: usize, value: f64, before: Option<f64>, paid: bool| -> Result<Obs> {
                let amount = before.map_or(value, |b| value - b);
                let w = if response.loss_ratio {
                    1.0
                } else {
                    data.premium[origin]
                };
                let y = if response.loss_ratio {
                    amount / data.premium[origin]
                } else {
                    amount
                };
                if log_normal && !(y > 0.0 && y.is_finite()) {
                    let what = match (paid, before.is_some()) {
                        (false, _) => "outstanding",
                        (true, true) => "incremental paid",
                        (true, false) => "paid",
                    };
                    return Err(Error::Bayes(format!(
                        "origin {} at development age {} months has {what} {amount}; the \
                     log-normal likelihood needs every observed response positive",
                        data.origins[origin],
                        data.age(k)
                    )));
                }
                Ok(Obs {
                    origin,
                    k,
                    t: times[k],
                    t_before: before.map(|_| times[k - 1]),
                    y,
                    ln_y: if log_normal { y.ln() } else { 0.0 },
                    w,
                    paid,
                })
            };
        let mut os = data
            .outstanding
            .iter()
            .map(|&(i, k, v)| obs(i, k, v, None, false))
            .collect::<Result<Vec<Obs>>>()?;
        let mut pd = data
            .paid
            .iter()
            .map(|&(i, k, v)| {
                let before = if response.incremental && k > 0 {
                    let b = data
                        .paid
                        .iter()
                        .find(|c| c.0 == i && c.1 == k - 1)
                        .ok_or_else(|| {
                            Error::Bayes(format!(
                                "origin {} has paid at development age {} months but none at \
                                 {} months; incremental paid needs the paid of the age before \
                                 each paid cell",
                                data.origins[i],
                                data.age(k),
                                data.age(k - 1)
                            ))
                        })?;
                    Some(b.2)
                } else {
                    None
                };
                obs(i, k, v, before, true)
            })
            .collect::<Result<Vec<Obs>>>()?;
        os.sort_by_key(|c| (c.origin, c.k));
        pd.sort_by_key(|c| (c.origin, c.k));
        let n_outstanding = os.len();
        let mut cells = os;
        cells.extend(pd);
        let n_observed = cells.iter().map(|c| c.k + 1).max().unwrap_or(0);
        let layout = Layout::new(&priors, level_origin.len(), dev_labels.len());
        let cellwise = priors.hierarchy() != Hierarchy::Model1;
        debug_assert!(layout.z0 + 2 <= GLOBALS);
        Ok(Self {
            cells,
            n_outstanding,
            premium: data.premium.clone(),
            latest_paid: data
                .latest_paid
                .iter()
                .map(|l| l.map_or(0.0, |(_, v)| v))
                .collect(),
            level,
            level_origin,
            labels,
            dev_level,
            dev_labels,
            times,
            n_observed,
            priors,
            ordered,
            response,
            cellwise,
            layout,
        })
    }

    /// The prior terms of [`log_density`](Self::log_density) that do not
    /// involve a `z`, at `g`: `theta[..z0]` followed by `log σ_OS` and
    /// `log σ_PD`.
    fn prior_globals<S: Scalar>(&self, g: &[S]) -> S {
        let pr = &self.priors;
        let lay = &self.layout;
        let mut f = S::from_f64(0.0);
        for &b in &g[B_KP..=B_RRF] {
            f -= b * b * 0.5;
        }
        let (k_er, _, b_ker) = self.rates(g);
        f -= b_ker * b_ker * 0.5;
        if self.ordered {
            f += g[B_KER] - k_er.ln();
        }
        f += half_on_log(g[LOG_TAU], pr.rlr_effect_sd)
            + half_on_log(g[LOG_TAU + 1], pr.rrf_effect_sd);
        let rho = g[ATANH_RHO].tanh();
        f += (-(rho * rho) + 1.0).ln() * pr.lkj_eta;
        for (at, prior) in lay.tau_rate.iter().zip(pr.rate_effect_sd()) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(g[*at], prior);
            }
        }
        for (at, prior) in lay.tau_dev.iter().zip(pr.development_effect_sd) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(g[*at], prior);
            }
        }
        let n = lay.z0;
        f + real_prior(g[n], pr.log_sigma[0]) + real_prior(g[n + 1], pr.log_sigma[1])
    }

    /// One cell's term of the log likelihood, `-r² / 2 - log σ`, from the
    /// `o` values of its four parameters and its type's `log σ`, with the
    /// same operations as [`log_density`](Self::log_density)'s cell by cell
    /// path.
    fn cell_term<S: Scalar>(&self, c: &Obs, o: [S; 4], ls: S) -> S {
        let ms = self.priors.positive().map(lognormal);
        let th: [S; 4] = std::array::from_fn(|p| (o[p] * ms[p].1).exp() * ms[p].0);
        let r = self.residual(c, self.cell_mean(c, th)) * (-ls).exp();
        -(r * r * 0.5) - ls
    }

    /// The log posterior and its gradient with effects by cell: the prior
    /// terms without a `z` in one dual pass over their at most `GLOBALS`
    /// inputs, the `z` terms by hand, each cell's term in a dual pass over
    /// its four `o` values and its `log σ`, and the chain rule from the `o`
    /// values (`o = b + u_origin + u_dev`, each effect `τ z`, and `RRF`'s
    /// origin effect `τ₂ (ρ z₁ + sqrt(1 - ρ²) z₂)`) to `theta` by hand,
    /// summed by level. The same value and gradient as the chunked dual pass
    /// over [`log_density`](Self::log_density), to rounding (a unit test
    /// checks it).
    fn cellwise_gradient(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let lay = &self.layout;
        let (z0, ls) = (lay.z0, self.log_sigma0());
        grad.fill(0.0);
        let mut globals = theta[..z0].to_vec();
        globals.extend([theta[ls], theta[ls + 1]]);
        let mut g = vec![0.0; globals.len()];
        let mut f = gradient::<GLOBALS>(&globals, &mut g, |x| self.prior_globals(x));
        grad[..z0].copy_from_slice(&g[..z0]);
        grad[ls] = g[z0];
        grad[ls + 1] = g[z0 + 1];
        for j in z0..ls {
            f -= theta[j] * theta[j] * 0.5;
            grad[j] -= theta[j];
        }
        // The cells, with the derivatives by `o` summed by level.
        let (u, v) = self.effects(theta);
        let (_, _, b_ker) = self.rates(theta);
        let b = [b_ker, theta[B_KP], theta[B_RLR], theta[B_RRF]];
        let mut by_origin = vec![[0.0; 4]; self.n_levels()];
        let mut by_dev = vec![[0.0; 4]; self.n_dev_levels()];
        let mut by_b = [0.0; 4];
        for c in &self.cells {
            let j = self.level[c.origin].expect("an observed origin has a level");
            let d = self.dev_level[c.k].expect("an observed age has a level");
            let o: [Dual<5>; 4] =
                std::array::from_fn(|p| Dual::variable(b[p] + u[p][j] + v[p][d], p));
            let at = ls + usize::from(c.paid);
            let t = self.cell_term(c, o, Dual::variable(theta[at], 4));
            f += t.v;
            grad[at] += t.d[4];
            for p in 0..4 {
                by_origin[j][p] += t.d[p];
                by_dev[d][p] += t.d[p];
                by_b[p] += t.d[p];
            }
        }
        // The intercepts: b_ker through the rate map (under the ordered
        // rates it depends on delta and b_kp).
        let rates = [
            Dual::<2>::variable(theta[B_KER], 0),
            Dual::variable(theta[B_KP], 1),
        ];
        let (_, _, b_ker) = self.rates(&rates);
        grad[B_KER] += by_b[KER] * b_ker.d[0];
        grad[B_KP] += by_b[KER] * b_ker.d[1] + by_b[KP];
        grad[B_RLR] += by_b[RLR];
        grad[B_RRF] += by_b[RRF];
        // The origin effects.
        let rho = theta[ATANH_RHO].tanh();
        let rho_c = (1.0 - rho * rho).sqrt();
        let (tau1, tau2) = (theta[LOG_TAU].exp(), theta[LOG_TAU + 1].exp());
        for (j, gj) in by_origin.iter().enumerate() {
            let (a, bz) = (lay.z0 + 2 * j, lay.z0 + 2 * j + 1);
            let (z1, z2) = (theta[a], theta[bz]);
            grad[LOG_TAU] += gj[RLR] * u[RLR][j];
            grad[LOG_TAU + 1] += gj[RRF] * u[RRF][j];
            grad[a] += gj[RLR] * tau1 + gj[RRF] * tau2 * rho;
            grad[bz] += gj[RRF] * tau2 * rho_c;
            // dρ/dv = 1 - ρ², d sqrt(1 - ρ²)/dv = -ρ sqrt(1 - ρ²).
            grad[ATANH_RHO] += gj[RRF] * tau2 * (z1 * rho_c * rho_c - z2 * rho * rho_c);
            for r in [KER, KP] {
                if let (Some(l), Some(z)) = (lay.tau_rate[r], lay.z_rate[r]) {
                    grad[l] += gj[r] * u[r][j];
                    grad[z + j] += gj[r] * theta[l].exp();
                }
            }
        }
        // The development effects.
        for (d, gd) in by_dev.iter().enumerate() {
            for p in 0..4 {
                if let (Some(l), Some(z)) = (lay.tau_dev[p], lay.z_dev[p]) {
                    grad[l] += gd[p] * v[p][d];
                    grad[z + d] += gd[p] * theta[l].exp();
                }
            }
        }
        f
    }

    fn n_levels(&self) -> usize {
        self.level_origin.len()
    }

    fn n_dev_levels(&self) -> usize {
        self.dev_labels.len()
    }

    /// Position of `log σ_OS`; `log σ_PD` follows.
    fn log_sigma0(&self) -> usize {
        self.layout.log_sigma0
    }

    /// Whether any development effect is in the priors.
    fn has_dev(&self) -> bool {
        self.layout.z_dev.iter().any(Option::is_some)
    }

    /// The names of `theta`, for the pilot draws.
    fn sampled_names(&self) -> Vec<String> {
        let lay = &self.layout;
        let mut names: Vec<String> = [
            if self.ordered { "delta" } else { "b_ker" },
            "b_kp",
            "b_RLR",
            "b_RRF",
            "log_tau_RLR",
            "log_tau_RRF",
            "atanh_rho",
        ]
        .map(String::from)
        .to_vec();
        for r in [KER, KP] {
            if lay.tau_rate[r].is_some() {
                names.push(format!("log_tau_{}", SHORT[r]));
            }
        }
        for (on, short) in lay.tau_dev.iter().zip(SHORT) {
            if on.is_some() {
                names.push(format!("log_tau_dev_{}", short));
            }
        }
        for j in 0..self.n_levels() {
            names.push(format!("z[{j},1]"));
            names.push(format!("z[{j},2]"));
        }
        for r in [KER, KP] {
            if lay.z_rate[r].is_some() {
                names.extend((0..self.n_levels()).map(|j| format!("z_{}[{j}]", SHORT[r])));
            }
        }
        for (on, short) in lay.z_dev.iter().zip(SHORT) {
            if on.is_some() {
                names.extend((0..self.n_dev_levels()).map(|d| format!("z_dev_{}[{d}]", short)));
            }
        }
        names.push("log_sigma_os".into());
        names.push("log_sigma_pd".into());
        names
    }

    /// The names of the natural-scale draws ([`CompartmentalFit::posterior`]).
    fn names(&self) -> Vec<String> {
        let lay = &self.layout;
        let mut names: Vec<String> = ["k_er", "k_p", "RLR", "RRF", "sigma_os", "sigma_pd"]
            .map(String::from)
            .to_vec();
        for p in [RLR, RRF, KER, KP] {
            if lay.nat_origin[p].is_some() {
                for j in 0..self.n_levels() {
                    names.push(format!("{}[{}]", NATURAL[p], self.labels[j]));
                }
            }
        }
        for (on, short) in lay.nat_dev.iter().zip(SHORT) {
            if on.is_some() {
                for label in &self.dev_labels {
                    names.push(format!("dev_{}[{}]", short, label));
                }
            }
        }
        names.extend(["b_ker", "b_kp", "b_RLR", "b_RRF", "tau_RLR", "tau_RRF"].map(String::from));
        for r in [KER, KP] {
            if lay.nat_tau_rate[r].is_some() {
                names.push(format!("tau_{}", SHORT[r]));
            }
        }
        names.push("rho".into());
        for (on, short) in lay.nat_tau_dev.iter().zip(SHORT) {
            if on.is_some() {
                names.push(format!("tau_dev_{}", short));
            }
        }
        names.push("log_sigma_os".into());
        names.push("log_sigma_pd".into());
        names
    }

    /// The rates `(k_er, k_p)` and `b_ker`, from `theta`.
    fn rates<S: Scalar>(&self, theta: &[S]) -> (S, S, S) {
        let (m_er, s_er) = lognormal(self.priors.k_er);
        let (m_p, s_p) = lognormal(self.priors.k_p);
        let k_p = (theta[B_KP] * s_p).exp() * m_p;
        if self.ordered {
            let k_er = k_p + theta[B_KER].exp();
            let b_ker = (k_er / m_er).ln() / s_er;
            (k_er, k_p, b_ker)
        } else {
            let b_ker = theta[B_KER];
            ((b_ker * s_er).exp() * m_er, k_p, b_ker)
        }
    }

    /// The effects on the `o` scale from `theta`: by origin level and by
    /// development level, for each of the four parameters (0 where the
    /// priors have none).
    #[allow(clippy::type_complexity)]
    fn effects<S: Scalar>(&self, theta: &[S]) -> ([Vec<S>; 4], [Vec<S>; 4]) {
        let lay = &self.layout;
        let (m, nd) = (self.n_levels(), self.n_dev_levels());
        let zero = S::from_f64(0.0);
        let rho = theta[ATANH_RHO].tanh();
        let rho_c = (-(rho * rho) + 1.0).sqrt();
        let (tau1, tau2) = (theta[LOG_TAU].exp(), theta[LOG_TAU + 1].exp());
        let mut u: [Vec<S>; 4] = std::array::from_fn(|_| vec![zero; m]);
        for j in 0..m {
            let (z1, z2) = (theta[lay.z0 + 2 * j], theta[lay.z0 + 2 * j + 1]);
            u[RLR][j] = tau1 * z1;
            u[RRF][j] = tau2 * (rho * z1 + rho_c * z2);
        }
        for r in [KER, KP] {
            if let (Some(l), Some(z)) = (lay.tau_rate[r], lay.z_rate[r]) {
                let tau = theta[l].exp();
                for (j, u) in u[r].iter_mut().enumerate() {
                    *u = tau * theta[z + j];
                }
            }
        }
        let mut v: [Vec<S>; 4] = std::array::from_fn(|_| vec![zero; nd]);
        for p in 0..4 {
            if let (Some(l), Some(z)) = (lay.tau_dev[p], lay.z_dev[p]) {
                let tau = theta[l].exp();
                for (d, v) in v[p].iter_mut().enumerate() {
                    *v = tau * theta[z + d];
                }
            }
        }
        (u, v)
    }

    /// The mean of a cell's response given its parameters `(k_er, k_p, RLR,
    /// RRF)`: outstanding, cumulative paid or incremental paid (the closed
    /// form at the cell's age less the same at the age before, both with
    /// these parameters), times the premium on the amount scale.
    fn cell_mean<S: Scalar>(&self, c: &Obs, [k_er, k_p, rlr, rrf]: [S; 4]) -> S {
        let base = rlr * c.w;
        if c.paid {
            let (_, pd) = one_os(k_er, k_p, c.t);
            let pd = match c.t_before {
                Some(t) => pd - one_os(k_er, k_p, t).1,
                None => pd,
            };
            base * rrf * pd
        } else {
            base * one_os(k_er, k_p, c.t).0
        }
    }

    /// The residual of a cell on the likelihood's scale: `y - μ`, or
    /// `ln y - ln μ`.
    fn residual<S: Scalar>(&self, c: &Obs, mean: S) -> S {
        if self.response.log_normal() {
            -(mean.ln()) + c.ln_y
        } else {
            -mean + c.y
        }
    }

    /// The log posterior, written once over [`Scalar`].
    fn log_density<S: Scalar>(&self, theta: &[S]) -> S {
        let pr = &self.priors;
        let lay = &self.layout;
        let mut f = S::from_f64(0.0);
        for &b in &theta[B_KP..=B_RRF] {
            f -= b * b * 0.5;
        }
        let (k_er, k_p, b_ker) = self.rates(theta);
        f -= b_ker * b_ker * 0.5;
        if self.ordered {
            // The Jacobian of delta -> b_ker: e^delta / (s_ker k_er), less
            // its constant.
            f += theta[B_KER] - k_er.ln();
        }
        let (l1, l2) = (theta[LOG_TAU], theta[LOG_TAU + 1]);
        f += half_on_log(l1, pr.rlr_effect_sd) + half_on_log(l2, pr.rrf_effect_sd);
        // LKJ(eta) on a 2x2 correlation is (1 - ρ²)^(eta - 1); with the
        // Jacobian 1 - ρ² of ρ = tanh(v), eta log(1 - ρ²).
        let rho = theta[ATANH_RHO].tanh();
        let rho_c2 = -(rho * rho) + 1.0;
        f += rho_c2.ln() * pr.lkj_eta;
        for (at, prior) in lay.tau_rate.iter().zip(pr.rate_effect_sd()) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(theta[*at], prior);
            }
        }
        for (at, prior) in lay.tau_dev.iter().zip(pr.development_effect_sd) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(theta[*at], prior);
            }
        }
        let ls = self.log_sigma0();
        for &z in &theta[lay.z0..ls] {
            f -= z * z * 0.5;
        }
        let (ls_os, ls_pd) = (theta[ls], theta[ls + 1]);
        f += real_prior(ls_os, pr.log_sigma[0]) + real_prior(ls_pd, pr.log_sigma[1]);
        let inv = [(-ls_os).exp(), (-ls_pd).exp()];
        let (u, v) = self.effects(theta);
        let mut n_os = 0.0;
        if self.cellwise {
            // Each cell its own parameters.
            let b = [b_ker, theta[B_KP], theta[B_RLR], theta[B_RRF]];
            let ms = pr.positive().map(lognormal);
            for c in &self.cells {
                let j = self.level[c.origin].expect("an observed origin has a level");
                let d = self.dev_level[c.k].expect("an observed age has a level");
                let th: [S; 4] =
                    std::array::from_fn(|p| ((b[p] + u[p][j] + v[p][d]) * ms[p].1).exp() * ms[p].0);
                let r = self.residual(c, self.cell_mean(c, th)) * inv[usize::from(c.paid)];
                if !c.paid {
                    n_os += 1.0;
                }
                f -= r * r * 0.5;
            }
        } else {
            let (m_rlr, s_rlr) = lognormal(pr.rlr);
            let (m_rrf, s_rrf) = lognormal(pr.rrf);
            // The rates are shared, so the closed form is one per age.
            let curves: Vec<(S, S)> = self.times[..self.n_observed]
                .iter()
                .map(|&t| one_os(k_er, k_p, t))
                .collect();
            // Π RLR and Π RLR RRF of each level (RLR and RLR RRF on the
            // loss-ratio scale).
            let scale: Vec<(S, S)> = self
                .level_origin
                .iter()
                .enumerate()
                .map(|(j, &i)| {
                    let w = if self.response.loss_ratio {
                        m_rlr
                    } else {
                        m_rlr * self.premium[i]
                    };
                    let rlr = ((theta[B_RLR] + u[RLR][j]) * s_rlr).exp() * w;
                    let rrf = ((theta[B_RRF] + u[RRF][j]) * s_rrf).exp() * m_rrf;
                    (rlr, rlr * rrf)
                })
                .collect();
            for c in &self.cells {
                let (os, pd) = curves[c.k];
                let (a, b) = scale[self.level[c.origin].expect("an observed origin has a level")];
                let mean = if !c.paid {
                    n_os += 1.0;
                    a * os
                } else if c.t_before.is_some() {
                    b * (pd - curves[c.k - 1].1)
                } else {
                    b * pd
                };
                let r = self.residual(c, mean) * inv[usize::from(c.paid)];
                f -= r * r * 0.5;
            }
        }
        let n_pd = (self.cells.len() - self.n_outstanding) as f64;
        f - ls_os * n_os - ls_pd * n_pd
    }

    /// `theta` on the natural scale ([`names`](Self::names)).
    fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let lay = &self.layout;
        let ms = self.priors.positive().map(lognormal);
        let (m_rlr, s_rlr) = ms[RLR];
        let (m_rrf, s_rrf) = ms[RRF];
        let (k_er, k_p, b_ker) = self.rates(theta);
        let ls = self.log_sigma0();
        let (u, v) = self.effects(theta);
        let mut out = vec![
            k_er,
            k_p,
            m_rlr * (s_rlr * theta[B_RLR]).exp(),
            m_rrf * (s_rrf * theta[B_RRF]).exp(),
            theta[ls].exp(),
            theta[ls + 1].exp(),
        ];
        let b = [b_ker, theta[B_KP], theta[B_RLR], theta[B_RRF]];
        for p in [RLR, RRF, KER, KP] {
            if lay.nat_origin[p].is_some() {
                out.extend(u[p].iter().map(|u| ms[p].0 * (ms[p].1 * (b[p] + u)).exp()));
            }
        }
        for (on, v) in lay.nat_dev.iter().zip(&v) {
            if on.is_some() {
                out.extend_from_slice(v);
            }
        }
        out.extend([
            b_ker,
            theta[B_KP],
            theta[B_RLR],
            theta[B_RRF],
            theta[LOG_TAU].exp(),
            theta[LOG_TAU + 1].exp(),
        ]);
        for l in lay.tau_rate.into_iter().flatten() {
            out.push(theta[l].exp());
        }
        out.push(theta[ATANH_RHO].tanh());
        for l in lay.tau_dev.into_iter().flatten() {
            out.push(theta[l].exp());
        }
        out.extend([theta[ls], theta[ls + 1]]);
        out
    }

    /// The origin values `(k_er, k_p, RLR, RRF)` of origin `i` (at a zero
    /// development effect) from a natural-scale draw: its own for an origin
    /// with data, else new effects from the population drawn from `rng`
    /// (two normals for `RLR` and `RRF`, then one for each rate with an
    /// origin effect).
    fn origin_values(&self, nat: &[f64], i: usize, rng: &mut StreamRng) -> [f64; 4] {
        let lay = &self.layout;
        if let Some(j) = self.level[i] {
            return std::array::from_fn(|p| lay.nat_origin[p].map_or(nat[p], |at| nat[at + j]));
        }
        let b = lay.nat_b0;
        let (b_rlr, b_rrf, tau1, tau2, rho) = (
            nat[b + 2],
            nat[b + 3],
            nat[b + 4],
            nat[b + 5],
            nat[lay.nat_rho],
        );
        let ms = self.priors.positive().map(lognormal);
        let (m_rlr, s_rlr) = ms[RLR];
        let (m_rrf, s_rrf) = ms[RRF];
        let z1 = norm_quantile(rng.next_open01());
        let z2 = norm_quantile(rng.next_open01());
        let rho_c = (1.0 - rho * rho).max(0.0).sqrt();
        let mut out = [
            nat[0],
            nat[1],
            m_rlr * (s_rlr * (b_rlr + tau1 * z1)).exp(),
            m_rrf * (s_rrf * (b_rrf + tau2 * (rho * z1 + rho_c * z2))).exp(),
        ];
        for r in [KER, KP] {
            if let Some(at) = lay.nat_tau_rate[r] {
                let z = norm_quantile(rng.next_open01());
                out[r] = ms[r].0 * (ms[r].1 * (nat[b + r] + nat[at] * z)).exp();
            }
        }
        out
    }

    /// The development multipliers `e^(s v)` of each age of the square and
    /// each parameter from a natural-scale draw: its own effect for an age
    /// with data, else a new one from the population drawn from `rng` (one
    /// normal per parameter with a development effect, age by age), shared
    /// by every origin; 1 for a parameter with none.
    fn dev_multipliers(&self, nat: &[f64], rng: &mut StreamRng) -> Vec<[f64; 4]> {
        let lay = &self.layout;
        let ms = self.priors.positive().map(lognormal);
        self.dev_level
            .iter()
            .map(|level| {
                let mut out = [1.0; 4];
                for p in 0..4 {
                    if let (Some(first), Some(tau)) = (lay.nat_dev[p], lay.nat_tau_dev[p]) {
                        let v = match level {
                            Some(d) => nat[first + d],
                            None => nat[tau] * norm_quantile(rng.next_open01()),
                        };
                        out[p] = (ms[p].1 * v).exp();
                    }
                }
                out
            })
            .collect()
    }

    /// The parameters of every observed cell from a natural-scale draw (its
    /// origin's values times its age's development multipliers).
    fn observed_parameters(&self, nat: &[f64]) -> Vec<[f64; 4]> {
        let lay = &self.layout;
        let ms = self.priors.positive().map(lognormal);
        // No new level is drawn for an observed cell, so the stream is
        // never read.
        let mut unused = StreamRng::new(0, 0);
        self.cells
            .iter()
            .map(|c| {
                let mut th = self.origin_values(nat, c.origin, &mut unused);
                let d = self.dev_level[c.k].expect("an observed age has a level");
                for (p, th) in th.iter_mut().enumerate() {
                    if let Some(first) = lay.nat_dev[p] {
                        *th *= (ms[p].1 * nat[first + d]).exp();
                    }
                }
                th
            })
            .collect()
    }

    /// A draw of a cell given its mean amount, the type's sigma and the
    /// origin `i`: normal (with sigma times the premium on the loss-ratio
    /// scale) or log-normal.
    fn draw_cell(&self, mean: f64, sigma: f64, i: usize, rng: &mut StreamRng) -> f64 {
        let e = norm_quantile(rng.next_open01());
        if self.response.log_normal() {
            mean * (sigma * e).exp()
        } else if self.response.loss_ratio {
            mean + sigma * self.premium[i] * e
        } else {
            mean + sigma * e
        }
    }

    /// One simulation of the future cells, paid then outstanding, into
    /// `row`, given a natural-scale draw: each cell (each increment under
    /// incremental paid, summed onto the latest paid) normal or log-normal
    /// about its mean with its type's σ, independent given the draw. An
    /// origin with no observed cell takes one new set of effects for all its
    /// cells, and an age with none one new set for all its cells.
    fn simulate(&self, nat: &[f64], future: &Future, rng: &mut StreamRng, row: &mut [f64]) {
        let (k_er, k_p) = (nat[0], nat[1]);
        let (sigma_os, sigma_pd) = (nat[4], nat[5]);
        let dev = if self.has_dev() {
            self.dev_multipliers(nat, rng)
        } else {
            vec![[1.0; 4]; self.times.len()]
        };
        let curves: Vec<(f64, f64)> = self.times.iter().map(|&t| one_os(k_er, k_p, t)).collect();
        let n = self.premium.len();
        let mut values: Vec<Option<[f64; 4]>> = vec![None; n];
        let mut get = |i: usize, rng: &mut StreamRng| -> [f64; 4] {
            *values[i].get_or_insert_with(|| self.origin_values(nat, i, rng))
        };
        // The cell's parameters and its closed form `(OS, PD)` at age `k`
        // and its PD at the age before (0 at the first age).
        let cell = |th: [f64; 4], k: usize| -> ([f64; 4], (f64, f64), f64) {
            if !self.cellwise {
                let before = if k == 0 { 0.0 } else { curves[k - 1].1 };
                return (th, curves[k], before);
            }
            let th: [f64; 4] = std::array::from_fn(|p| th[p] * dev[k][p]);
            let now = one_os(th[KER], th[KP], self.times[k]);
            let before = if k == 0 {
                0.0
            } else {
                one_os(th[KER], th[KP], self.times[k - 1]).1
            };
            (th, now, before)
        };
        let n_paid = future.paid.len();
        let mut current = (usize::MAX, 0.0);
        for (x, &(i, k)) in row[..n_paid].iter_mut().zip(&future.paid) {
            let (th, (_, pd), before) = cell(get(i, rng), k);
            let scale = self.premium[i] * th[RLR] * th[RRF];
            *x = if self.response.incremental {
                if current.0 != i {
                    current = (i, self.latest_paid[i]);
                }
                current.1 += self.draw_cell(scale * (pd - before), sigma_pd, i, rng);
                current.1
            } else {
                self.draw_cell(scale * pd, sigma_pd, i, rng)
            };
        }
        for (x, &(i, k)) in row[n_paid..].iter_mut().zip(&future.outstanding) {
            let (th, (os, _), _) = cell(get(i, rng), k);
            let mean = self.premium[i] * th[RLR] * os;
            *x = self.draw_cell(mean, sigma_os, i, rng);
        }
    }

    /// The full log density of each cell's response from a natural-scale
    /// draw, appended to `out`: normal, or log-normal (with its `-ln y`).
    fn log_likelihoods(&self, nat: &[f64], out: &mut Vec<f64>) {
        let (sigma_os, sigma_pd) = (nat[4], nat[5]);
        let half_log_tau = 0.5 * std::f64::consts::TAU.ln();
        for (c, th) in self.cells.iter().zip(self.observed_parameters(nat)) {
            let mean = self.cell_mean(c, th);
            let sigma = if c.paid { sigma_pd } else { sigma_os };
            out.push(if self.response.log_normal() {
                let z = (c.ln_y - mean.ln()) / sigma;
                -half_log_tau - sigma.ln() - c.ln_y - 0.5 * z * z
            } else {
                let z = (c.y - mean) / sigma;
                -half_log_tau - sigma.ln() - 0.5 * z * z
            });
        }
    }

    /// With effects on the rates, the share of draws and observed cells
    /// with `k_er <= k_p`.
    fn cell_rate_swaps(&self, posterior: &PosteriorDraws) -> Option<f64> {
        if !self.priors.rates_vary() {
            return None;
        }
        let mut swaps = 0usize;
        for nat in posterior.values().chunks_exact(posterior.dim()) {
            swaps += self
                .observed_parameters(nat)
                .iter()
                .filter(|th| th[KER] <= th[KP])
                .count();
        }
        Some(swaps as f64 / (posterior.n_draws() * self.cells.len()) as f64)
    }

    /// Each origin's posterior mean `RLR_i RRF_i` (at a zero development
    /// effect); for an origin with no data, `E[RLR RRF | draw] = m e^(s_RLR
    /// b_RLR + s_RRF b_RRF + v / 2)` with `v` the variance of `s_RLR u_1 +
    /// s_RRF u_2`, averaged over the draws.
    fn mean_ultimate_loss_ratios(&self, posterior: &PosteriorDraws) -> Vec<f64> {
        let lay = &self.layout;
        let (m_rlr, s_rlr) = lognormal(self.priors.rlr);
        let (m_rrf, s_rrf) = lognormal(self.priors.rrf);
        let (at_rlr, at_rrf) = (
            lay.nat_origin[RLR].expect("RLR by origin"),
            lay.nat_origin[RRF].expect("RRF by origin"),
        );
        let n = self.premium.len();
        let mut sums = vec![0.0; n];
        for nat in posterior.values().chunks_exact(posterior.dim()) {
            let b = lay.nat_b0;
            let (b_rlr, b_rrf, tau1, tau2, rho) = (
                nat[b + 2],
                nat[b + 3],
                nat[b + 4],
                nat[b + 5],
                nat[lay.nat_rho],
            );
            for (i, sum) in sums.iter_mut().enumerate() {
                *sum += match self.level[i] {
                    Some(j) => nat[at_rlr + j] * nat[at_rrf + j],
                    None => {
                        let (a, c) = (s_rlr * tau1, s_rrf * tau2);
                        let v = a * a + c * c + 2.0 * rho * a * c;
                        m_rlr * m_rrf * (s_rlr * b_rlr + s_rrf * b_rrf + 0.5 * v).exp()
                    }
                };
            }
        }
        let draws = posterior.n_draws() as f64;
        sums.iter().map(|s| s / draws).collect()
    }

    /// The start of the pilot run (decision 5).
    fn start(&self, start: CompartmentalStart) -> Vec<f64> {
        if start == CompartmentalStart::Data
            && let Some(theta) = self.data_start()
        {
            return theta;
        }
        self.prior_centre_start()
    }

    /// `theta` with the population values `(k_er, k_p, RLR, RRF)`, every `z`
    /// and `atanh ρ` at 0, each `log τ` at about its prior median and the
    /// `log σ` at the log of each type's residual sd about the population
    /// curve.
    fn theta_at(&self, population: [f64; 4]) -> Option<Vec<f64>> {
        let [k_er, k_p, rlr, rrf] = population;
        let pr = &self.priors;
        let lay = &self.layout;
        let o = |prior: PositivePrior, v: f64| {
            let (m, s) = lognormal(prior);
            (v / m).ln() / s
        };
        let mut theta = vec![0.0; self.dim()];
        theta[B_KER] = if self.ordered {
            if k_er <= k_p {
                return None;
            }
            (k_er - k_p).ln()
        } else {
            o(pr.k_er, k_er)
        };
        theta[B_KP] = o(pr.k_p, k_p);
        theta[B_RLR] = o(pr.rlr, rlr);
        theta[B_RRF] = o(pr.rrf, rrf);
        theta[LOG_TAU] = half_median(pr.rlr_effect_sd).ln();
        theta[LOG_TAU + 1] = half_median(pr.rrf_effect_sd).ln();
        for (at, prior) in lay.tau_rate.iter().zip(pr.rate_effect_sd()) {
            if let (Some(at), Some(prior)) = (at, prior) {
                theta[*at] = half_median(prior).ln();
            }
        }
        for (at, prior) in lay.tau_dev.iter().zip(pr.development_effect_sd) {
            if let (Some(at), Some(prior)) = (at, prior) {
                theta[*at] = half_median(prior).ln();
            }
        }
        let rms = self.residual_rms(population);
        let ls = self.log_sigma0();
        theta[ls] = rms[0].ln();
        theta[ls + 1] = rms[1].ln();
        let mut grad = vec![0.0; theta.len()];
        let f = self.log_posterior(&theta, &mut grad);
        (f.is_finite() && theta.iter().all(|v| v.is_finite())).then_some(theta)
    }

    /// The root mean square residual (on the likelihood's scale) of the
    /// outstanding and the paid cells about the population curve, each at
    /// least `1e-6` of its cells' mean absolute value (of 1 on the log
    /// scale).
    fn residual_rms(&self, population: [f64; 4]) -> [f64; 2] {
        let mut sum = [0.0; 2];
        let mut size = [0.0; 2];
        let mut count = [0.0; 2];
        for c in &self.cells {
            let k = usize::from(c.paid);
            let r = self.residual(c, self.cell_mean(c, population));
            sum[k] += r * r;
            size[k] += if self.response.log_normal() {
                1.0
            } else {
                c.y.abs()
            };
            count[k] += 1.0;
        }
        [0, 1].map(|k| {
            let floor = 1e-6 * (size[k] / count[k]).max(1.0);
            (sum[k] / count[k]).sqrt().max(floor)
        })
    }

    /// The data-based start: least squares of the population curve (on the
    /// likelihood's scale) of `log (k_er, k_p, RLR, RRF)`, each type
    /// weighted by the inverse of its residual variance (twice, from the
    /// prior medians), then the representative of the rate swap the priors
    /// prefer.
    fn data_start(&self) -> Option<Vec<f64>> {
        let pr = &self.priors;
        let medians = pr.positive().map(|p| lognormal(p).0);
        let mut phi = medians.map(f64::ln).to_vec();
        for _ in 0..2 {
            let population = [0, 1, 2, 3].map(|j| phi[j].exp());
            let rms = self.residual_rms(population);
            let ls = LeastSquares {
                model: self,
                weight: rms.map(|r| 1.0 / (r * r)),
            };
            phi = mode(&ls, phi).ok()?;
        }
        let [k_er, k_p, rlr, rrf] = [0, 1, 2, 3].map(|j| phi[j].exp());
        let swapped = [k_p, k_er, rlr * k_er / k_p, rrf * k_p / k_er];
        let candidates = [self.theta_at([k_er, k_p, rlr, rrf]), self.theta_at(swapped)];
        let mut grad = vec![0.0; self.dim()];
        candidates
            .into_iter()
            .flatten()
            .map(|theta| (self.log_posterior(&theta, &mut grad), theta))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, theta)| theta)
    }

    /// The fallback start at the prior medians. Under
    /// [`RateOrder::Ordered`], with medians that break the order, `k_er` is
    /// raised to `1.1 k_p`.
    fn prior_centre_start(&self) -> Vec<f64> {
        let pr = &self.priors;
        let [k_er, k_p, rlr, rrf] = pr.positive().map(|p| lognormal(p).0);
        let k_er = if self.ordered {
            k_er.max(1.1 * k_p)
        } else {
            k_er
        };
        self.theta_at([k_er, k_p, rlr, rrf]).unwrap_or_else(|| {
            // A curve too far from the data to give a finite density: the
            // log σ at the scale of the data.
            let mut theta = vec![0.0; self.dim()];
            let ls = self.log_sigma0();
            let rms = self.residual_rms([k_er, k_p, rlr, rrf]);
            theta[ls] = rms[0].ln();
            theta[ls + 1] = rms[1].ln();
            theta
        })
    }
}

impl Posterior for Model {
    fn dim(&self) -> usize {
        self.log_sigma0() + 2
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let f = if self.cellwise {
            self.cellwise_gradient(theta, grad)
        } else {
            gradient::<CHUNK>(theta, grad, |x| self.log_density(x))
        };
        if f.is_finite() { f } else { f64::NAN }
    }
}

/// The population curve's weighted least squares on
/// `phi = log (k_er, k_p, RLR, RRF)`, as a [`Posterior`] for [`mode`].
struct LeastSquares<'a> {
    model: &'a Model,
    weight: [f64; 2],
}

impl LeastSquares<'_> {
    fn value<S: Scalar>(&self, phi: &[S]) -> S {
        let population = [phi[0].exp(), phi[1].exp(), phi[2].exp(), phi[3].exp()];
        let mut f = S::from_f64(0.0);
        for c in &self.model.cells {
            let r = self.model.residual(c, self.model.cell_mean(c, population));
            f -= r * r * (0.5 * self.weight[usize::from(c.paid)]);
        }
        f
    }
}

impl Posterior for LeastSquares<'_> {
    fn dim(&self) -> usize {
        4
    }

    fn log_posterior(&self, phi: &[f64], grad: &mut [f64]) -> f64 {
        let f = gradient::<4>(phi, grad, |x| self.value(x));
        if f.is_finite() { f } else { f64::NAN }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayes::laplace::tests::assert_gradient;
    use crate::bayes::one_year::OneYearRefit;
    use crate::triangle::Long;
    use crate::{DevelopmentColumn, Month};
    use prospicio_math::special::norm_pdf;

    /// Company 337 workers' compensation, the papers' training triangle
    /// (validation/data/clrd_wkcomp_337.csv, calendar years up to 1996):
    /// cumulative paid and incurred (with bulk and IBNR), USD thousands.
    const PAID: [&[f64]; 9] = [
        &[
            9558., 22778., 33298., 40348., 45146., 48048., 49782., 50623., 51812.,
        ],
        &[
            7913., 19472., 29622., 36816., 40975., 43302., 44707., 45871.,
        ],
        &[8744., 24302., 35406., 43412., 48057., 50897., 52879.],
        &[13301., 32950., 47201., 56394., 61650., 65039.],
        &[11424., 29086., 42034., 50910., 56406.],
        &[11792., 27161., 38229., 46722.],
        &[11194., 26893., 38488.],
        &[12550., 31604.],
        &[13194.],
    ];
    const INCURRED: [&[f64]; 9] = [
        &[
            62679., 64000., 65607., 65292., 62250., 61185., 59387., 57138., 53473.,
        ],
        &[
            62058., 56660., 56598., 56831., 55294., 53481., 51379., 48446.,
        ],
        &[63955., 61523., 63166., 61402., 59474., 57613., 57161.],
        &[73918., 75094., 73188., 71199., 71056., 70831.],
        &[77143., 75133., 73284., 73155., 68284.],
        &[79925., 78263., 78163., 68546.],
        &[73628., 73554., 70736.],
        &[69521., 80281.],
        &[69720.],
    ];
    /// Direct earned premium, 1988 to 1997.
    const PREMIUM: [f64; 10] = [
        104437., 88883., 85956., 99339., 104897., 119427., 110784., 77731., 63646., 48052.,
    ];

    /// The training triangle with columns `paid`, `incurred`, `os`
    /// (incurred - paid) and `premium`, and 1997 with its premium only when
    /// `with_1997`.
    fn triangle(with_1997: bool) -> Triangle {
        let (mut origin, mut ages) = (vec![], vec![]);
        let (mut paid, mut inc, mut os, mut prem) = (vec![], vec![], vec![], vec![]);
        for (w, row) in PAID.iter().enumerate() {
            for (d, &p) in row.iter().enumerate() {
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as Lag + 1));
                paid.push(p);
                inc.push(INCURRED[w][d]);
                os.push(INCURRED[w][d] - p);
                prem.push(PREMIUM[w]);
            }
        }
        if with_1997 {
            origin.push(Month::january(1997));
            ages.push(12);
            paid.push(f64::NAN);
            inc.push(f64::NAN);
            os.push(f64::NAN);
            prem.push(PREMIUM[9]);
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[
                ("paid", &paid),
                ("incurred", &inc),
                ("os", &os),
                ("premium", &prem),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn preset() -> Compartmental {
        Compartmental::new(CompartmentalPriors::gesmann_morris())
    }

    fn quick(seed: u64) -> Compartmental {
        Compartmental {
            sampler: Sampler {
                chains: 2,
                tune: 100,
                draws: 100,
                target_accept: 0.9,
                ..preset().sampler
            },
            n_sims: 500,
            seed,
            ..preset()
        }
    }

    fn incurred() -> Outstanding<'static> {
        Outstanding::FromIncurred("incurred")
    }

    fn model(ordered: bool) -> Model {
        model_with(
            CompartmentalPriors::gesmann_morris(),
            ordered,
            Response::MODEL_1,
        )
    }

    fn model_with(priors: CompartmentalPriors, ordered: bool, response: Response) -> Model {
        let tri = triangle(true);
        let data = Data::new(&tri, "paid", incurred(), "premium").unwrap();
        Model::new(&data, priors, ordered, response).unwrap()
    }

    /// Gesmann and Morris' model 2 on the training triangle.
    fn model_2(ordered: bool) -> Model {
        model_with(
            CompartmentalPriors::gesmann_morris_2(),
            ordered,
            Response::MODEL_2,
        )
    }

    fn preset_2() -> Compartmental {
        Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
    }

    /// The one-year defaults of the user's decision of 2026-10-16: model 1
    /// moment-matches, model 2 (development effects) reweights only with a
    /// warning, a refit mode the settings choose is kept without one, and
    /// both run at most 2,000 one-year simulations unless told otherwise.
    #[test]
    fn the_one_year_defaults_follow_the_model() {
        let default = BayesOneYear::default();
        let (one, sims, warning) = preset().one_year_defaults(&default);
        assert_eq!(one.refit, Some(OneYearRefit::MomentMatchThenRefit));
        assert_eq!((sims, warning), (Compartmental::ONE_YEAR_SIMS, None));
        let (two, sims, warning) = preset_2().one_year_defaults(&default);
        assert_eq!(two.refit, Some(OneYearRefit::ReweightOnly));
        assert_eq!(sims, 2_000);
        let warning = warning.expect("model 2's default view warns");
        assert!(warning.contains("approximate") && warning.contains("0.4 sd"));
        // Refits stay a choice: an explicit mode, even the other models'
        // default, is used as it is and is not approximate.
        for refit in [
            OneYearRefit::MomentMatchThenRefit,
            OneYearRefit::RefitFlagged,
            OneYearRefit::ReweightOnly,
        ] {
            let chosen = BayesOneYear {
                refit: Some(refit),
                n_sims: Some(300),
                ..default
            };
            let (two, sims, warning) = preset_2().one_year_defaults(&chosen);
            assert_eq!((two.refit, sims, warning), (Some(refit), 300, None));
        }
        // A model with fewer simulations runs that many; more are capped.
        let small = Compartmental {
            n_sims: 500,
            ..preset_2()
        };
        assert_eq!(small.one_year_defaults(&default).1, 500);
        let large = Compartmental {
            n_sims: 50_000,
            ..preset()
        };
        assert_eq!(large.one_year_defaults(&default).1, 2_000);
        let more = BayesOneYear {
            n_sims: Some(20_000),
            ..default
        };
        assert_eq!(large.one_year_defaults(&more).1, 20_000);
    }

    /// Points around the paper's posterior (model 1's, or model 2's when
    /// the model has effects by cell): `theta` at the printed means with a
    /// deterministic spread.
    fn points(model: &Model) -> Vec<Vec<f64>> {
        let two = model.cellwise;
        (0..20)
            .map(|k| {
                let mut x = vec![0.0; model.dim()];
                let wobble = |j: usize| 0.3 * ((7 * k + 3 * j) as f64).sin();
                x[B_KER] = match (model.ordered, two) {
                    (true, _) => 0.2 + wobble(0),
                    (false, false) => -5.4 + wobble(0),
                    (false, true) => -1.31 + wobble(0),
                };
                let (b, tau, rho, ls) = if two {
                    ([-5.07, 1.54, -1.45], [0.69, 0.73], 0.37, [-1.78, -1.89])
                } else {
                    ([-8.49, 1.44, -1.36], [0.64, 0.74], 0.6, [8.25, 6.66])
                };
                x[B_KP] = b[0] + wobble(1);
                x[B_RLR] = b[1] + wobble(2);
                x[B_RRF] = b[2] + wobble(3);
                x[LOG_TAU] = tau[0].ln() + wobble(4);
                x[LOG_TAU + 1] = tau[1].ln() + wobble(5);
                x[ATANH_RHO] = rho + wobble(6);
                let z0 = model.layout.z0;
                // Model 2's other effect sds: 0.26, 2.23 by origin, then
                // 0.27, 0.45, 0.35, 1.11 by development year (the printed
                // m2fit).
                let sds = [0.26f64, 2.23, 0.27, 0.45, 0.35, 1.11];
                for (j, l) in x[ATANH_RHO + 1..z0].iter_mut().enumerate() {
                    *l = sds[j].ln() + wobble(7 + j);
                }
                let ls0 = model.log_sigma0();
                let spread = if two { 1.0 } else { 3.0 };
                for (j, z) in x[z0..ls0].iter_mut().enumerate() {
                    *z = spread * wobble(13 + j);
                }
                x[ls0] = ls[0] + wobble(ls0);
                x[ls0 + 1] = ls[1] + wobble(ls0 + 1);
                x
            })
            .collect()
    }

    #[test]
    fn the_layout_on_the_papers_training_triangle() {
        let m = model(false);
        assert_eq!(m.dim(), 27);
        assert_eq!((m.cells.len(), m.n_outstanding), (90, 45));
        assert_eq!(m.level[9], None);
        assert_eq!(m.n_levels(), 9);
        assert_eq!(m.times.len(), 10);
        assert_eq!(m.n_observed, 9);
        let names = m.names();
        assert_eq!(names.len(), 6 + 18 + 9);
        assert_eq!(names[6], "RLR[1988]");
        assert_eq!(names[15], "RRF[1988]");
        // Outstanding from incurred less paid.
        assert_eq!(m.cells[0].y, 62679. - 9558.);
        assert!(m.cells[45].paid && m.cells[45].y == 9558.);
    }

    #[test]
    fn the_dual_gradient_matches_central_differences() {
        for ordered in [false, true] {
            let m = model(ordered);
            for x in points(&m) {
                assert_gradient(|t, g| m.log_posterior(t, g), &x);
            }
            // Model 2 (87 parameters): fewer points, for the debug build's
            // time.
            let m = model_2(ordered);
            for x in points(&m).iter().step_by(4) {
                assert_gradient(|t, g| m.log_posterior(t, g), x);
            }
        }
        // Model 1's hierarchy with model 2's response (the fast path with
        // increments on the log scale).
        let m = model_with(
            CompartmentalPriors::gesmann_morris(),
            false,
            Response::MODEL_2,
        );
        for x in points(&m).iter().step_by(4) {
            let mut x = x.clone();
            let ls = m.log_sigma0();
            (x[ls], x[ls + 1]) = (-1.8, -1.9);
            assert_gradient(|t, g| m.log_posterior(t, g), &x);
        }
    }

    #[test]
    fn the_density_is_the_model_written_out() {
        // At one point, against the model written out in f64 with the
        // paper's form of the closed form (away from k_er = k_p).
        let m = model(false);
        let x = &points(&m)[3];
        let half_t = |l: f64, s: f64| l - 5.5 * ((l.exp() / s).powi(2) / 10.0).ln_1p();
        let cauchy = |v: f64| -((v / 1000.0).powi(2)).ln_1p();
        let (a, b) = (3.0 * (0.1 * x[0]).exp(), (0.1 * x[1]).exp());
        let rho = x[6].tanh();
        let ls = m.log_sigma0();
        let mut f = -0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
            + half_t(x[4], 0.2)
            + half_t(x[5], 0.1)
            + (1.0 - rho * rho).ln()
            + cauchy(x[ls])
            + cauchy(x[ls + 1]);
        for z in &x[m.layout.z0..ls] {
            f -= 0.5 * z * z;
        }
        for c in &m.cells {
            let j = m.level[c.origin].unwrap();
            let (z1, z2) = (x[m.layout.z0 + 2 * j], x[m.layout.z0 + 2 * j + 1]);
            let u1 = x[4].exp() * z1;
            let u2 = x[5].exp() * (rho * z1 + (1.0 - rho * rho).sqrt() * z2);
            let rlr = 0.7 * (0.2 * (x[2] + u1)).exp();
            let rrf = 0.8 * (0.1 * (x[3] + u2)).exp();
            let t = c.t;
            let os = rlr * a / (a - b) * ((-b * t).exp() - (-a * t).exp());
            let pd =
                rlr * rrf / (a - b) * (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp()));
            let (mean, l) = if c.paid { (pd, x[ls + 1]) } else { (os, x[ls]) };
            let r = (c.y - PREMIUM[c.origin] * mean) / l.exp();
            f -= l + 0.5 * r * r;
        }
        let mut g = vec![0.0; m.dim()];
        let got = m.log_posterior(x, &mut g);
        assert!((got - f).abs() <= 1e-12 * f.abs(), "{got} vs {f}");
    }

    #[test]
    fn the_means_are_invariant_under_the_rate_swap() {
        let mut rng = StreamRng::new(7, 0);
        let mut worst: f64 = 0.0;
        for _ in 0..200 {
            let mut u = |lo: f64, hi: f64| lo + (hi - lo) * rng.next_open01();
            let (k_er, k_p, rlr, rrf, t) = (
                u(0.05, 6.0),
                u(0.05, 6.0),
                u(0.3, 1.5),
                u(0.5, 1.5),
                u(0.1, 15.0),
            );
            let (os, pd) = one_os(k_er, k_p, t);
            let (os2, pd2) = one_os(k_p, k_er, t);
            let (rlr2, rrf2) = (rlr * k_er / k_p, rrf * k_p / k_er);
            for (x, y) in [(rlr * os, rlr2 * os2), (rlr * rrf * pd, rlr2 * rrf2 * pd2)] {
                let err = (x - y).abs() / x.abs();
                worst = worst.max(err);
                assert!(err <= 1e-12, "{k_er} {k_p} {t}: {x} vs {y}");
            }
        }
        println!("swap invariance: worst relative difference {worst:.1e}");
    }

    #[test]
    fn the_data_start_is_the_population_least_squares() {
        let m = model(false);
        let start = m.start(CompartmentalStart::Data);
        // On the paper's side of the swap and near its posterior (k_er about
        // 1.75 and k_p about 0.43, so b_ker -5.4 and b_kp -8.5; log sigma
        // 8.25 and 6.66). Without origin effects the population curve fits
        // paid worse, so its log sigma_PD starts higher (about 8.7).
        let ls = m.log_sigma0();
        assert!(start[B_KER] > -9.0 && start[B_KER] < 0.0, "{start:?}");
        assert!(start[B_KP] > -10.0 && start[B_KP] < -7.0, "{start:?}");
        assert!((start[ls] - 8.25).abs() < 1.5, "{start:?}");
        assert!((start[ls + 1] - 6.66).abs() < 2.5, "{start:?}");
        assert!(start[m.layout.z0..ls].iter().all(|&z| z == 0.0));
        let centre = m.start(CompartmentalStart::PriorCentre);
        assert_eq!(&centre[..4], &[0.0; 4]);
        let ordered = model(true);
        let s = ordered.start(CompartmentalStart::Data);
        let (k_er, k_p, _) = ordered.rates(&s);
        assert!(k_er > k_p);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_seed_replays_whatever_the_threads() {
        let tri = triangle(true);
        let fit = |m: &Compartmental| m.fit(&tri, "paid", incurred(), "premium").unwrap();
        let a = fit(&quick(3));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| fit(&quick(3)));
        assert_eq!(a.posterior, b.posterior);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        assert_eq!(a.paid.draw_matrix(), b.paid.draw_matrix());
        let c = fit(&quick(4));
        assert_ne!(a.posterior, c.posterior);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_fit_and_its_tables() {
        let tri = triangle(true);
        let fit = quick(1)
            .fit(&tri, "paid", Outstanding::Column("os"), "premium")
            .unwrap();
        assert_eq!(fit.origins.len(), 10);
        assert_eq!(fit.population_origins, [Period::year(1997)]);
        assert_eq!((fit.latest_paid[0], fit.latest_paid[9]), (51812., 0.0));
        assert_eq!(fit.reserves.n_components(), 10);
        // Future paid: 1988 at age 10, 1989 at 9 and 10, ..., 1997 at all 10.
        assert_eq!(fit.paid.n_components(), (1..=9).sum::<usize>() + 10);
        assert_eq!(fit.outstanding.n_components(), fit.paid.n_components());
        // The reserve is the paid at age 10 less the latest paid.
        let key = vec![Period::year(1990).into(), KeyValue::Int(120)];
        let at_10 = fit.paid.marginal(&key).unwrap();
        let reserve = fit
            .reserves
            .marginal(&vec![Period::year(1990).into()])
            .unwrap();
        assert!((at_10.mean() - 52879. - reserve.mean()).abs() < 1e-6 * at_10.mean());
        let long = fit.to_long();
        assert_eq!(long.n_rows(), 10);
        assert_eq!(long.column("premium").unwrap(), PREMIUM);
        let ulr = long.column("ultimate_loss_ratio").unwrap();
        assert!(ulr.iter().all(|&v| v > 0.3 && v < 1.5), "{ulr:?}");
        let totals = fit.totals();
        assert_eq!(totals.n_rows(), 1);
        let sum: f64 = long.column("mean").unwrap().iter().sum();
        assert!((totals.column("mean").unwrap()[0] - sum).abs() < 1e-6 * sum);
        let table = fit.paid_table().unwrap();
        assert_eq!(table.n_rows(), fit.paid.n_components());
        assert_eq!(table.age.as_ref().unwrap()[0], 120);
        let q = |n| table.column(n).unwrap()[0];
        assert!(q("q05") < q("q50") && q("q50") < q("q95"));
        assert_eq!(
            fit.outstanding_table().unwrap().n_rows(),
            fit.outstanding.n_components()
        );
        assert_eq!(fit.summary().unwrap().len(), 33);
        let loo = fit.loo().unwrap();
        assert_eq!(loo.n_outstanding, 45);
        assert!(loo.elpd_outstanding.is_finite() && loo.elpd_paid.is_finite());
        let output = |pd: &PredictiveDistribution| {
            pd.provenance()
                .parameters
                .iter()
                .filter(|(k, _)| k == "output")
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(output(&fit.reserves), ["reserves"]);
        assert_eq!(output(&fit.paid), ["paid"]);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn an_origin_observed_at_the_last_age_has_no_reserve() {
        // Without 1997 the square has 9 ages, and 1988 is observed at age 9.
        let tri = triangle(false);
        let fit = quick(2).fit(&tri, "paid", incurred(), "premium").unwrap();
        assert!(fit.population_origins.is_empty());
        let r = fit
            .reserves
            .marginal(&vec![Period::year(1988).into()])
            .unwrap();
        assert_eq!((r.mean(), r.std_dev()), (0.0, 0.0));
        assert_eq!(fit.paid.n_components(), (1..=8).sum::<usize>());
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_rate_swap_warns() {
        // Rate priors centred on the mirror image of the paper's posterior:
        // the posterior sits at k_er < k_p and the fit says so.
        let priors = CompartmentalPriors {
            k_er: PositivePrior::LogNormal {
                median: 0.43,
                sd: 0.05,
            },
            k_p: PositivePrior::LogNormal {
                median: 1.75,
                sd: 0.05,
            },
            ..CompartmentalPriors::gesmann_morris()
        };
        let tri = triangle(true);
        let fit = Compartmental { priors, ..quick(5) }
            .fit(&tri, "paid", incurred(), "premium")
            .unwrap();
        assert!(fit.rate_swaps > 0);
        assert!(
            fit.diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("k_er <= k_p")),
            "{:?}",
            fit.diagnostics.warnings
        );
        let ordered = Compartmental {
            priors,
            rate_order: RateOrder::Ordered,
            ..quick(5)
        }
        .fit(&tri, "paid", incurred(), "premium")
        .unwrap();
        assert_eq!(ordered.rate_swaps, 0);
        assert!(
            !ordered
                .diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("k_er <= k_p"))
        );
    }

    #[test]
    fn the_ordered_prior_check_keeps_the_order() {
        let rates = |er: f64, p: f64, sd: f64| CompartmentalPriors {
            k_er: PositivePrior::LogNormal { median: er, sd },
            k_p: PositivePrior::LogNormal { median: p, sd },
            ..CompartmentalPriors::gesmann_morris()
        };
        // The swapped priors of the_rate_swap_warns put about 1e-88 of
        // their mass on k_er > k_p: the ordered prior check says so
        // instead of drawing rates that break the order.
        let swapped = rates(0.43, 1.75, 0.05);
        let (a, _) = rate_order_bound(swapped);
        assert!(norm_cdf(-a) < 1e-80, "{a}");
        let e = Compartmental {
            priors: swapped,
            rate_order: RateOrder::Ordered,
            ..preset()
        }
        .prior_predictive(&triangle(true), "premium", 100)
        .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(why) if why.contains("almost no mass to k_er > k_p")),
            "{e}"
        );
        // About 2.2e-5 of the mass on the order (a = 4.09): every draw
        // keeps it, and the projection's mean is the truncated normal's,
        // φ(a) / Φ(-a), with the orthogonal one's mean 0.
        let pr = rates(1.0, 1.75, 0.0968);
        let (m_er, s_er) = lognormal(pr.k_er);
        let (m_p, s_p) = lognormal(pr.k_p);
        let (a, s) = rate_order_bound(pr);
        let mass = norm_cdf(-a);
        assert!((1e-6..1e-4).contains(&mass), "{mass}");
        let mut rng = StreamRng::new(7, 0);
        let n = 20_000;
        let (mut z_sum, mut y_sum) = (0.0, 0.0);
        for _ in 0..n {
            let (b_ker, b_kp) = ordered_rates(pr, &mut rng);
            assert!(m_er * (s_er * b_ker).exp() > m_p * (s_p * b_kp).exp());
            z_sum += (s_er * b_ker - s_p * b_kp) / s;
            y_sum += (s_p * b_ker + s_er * b_kp) / s;
        }
        let mills = norm_pdf(a) / mass;
        // Standard errors: under 1 / (a sqrt(n)) = 0.002 for z (the
        // truncated normal's sd is below 1 / a) and 1 / sqrt(n) = 0.007
        // for y.
        assert!(
            (z_sum / n as f64 - mills).abs() < 0.01,
            "{}",
            z_sum / n as f64
        );
        assert!((y_sum / n as f64).abs() < 0.03, "{}", y_sum / n as f64);
    }

    #[test]
    fn settings_not_yet_built_are_errors() {
        let tri = triangle(true);
        let fit = |m: Compartmental| {
            m.fit(&tri, "paid", incurred(), "premium")
                .unwrap_err()
                .to_string()
        };
        let cases = [
            (
                Compartmental {
                    fitted: Fitted::PaidAndIncurred,
                    ..preset()
                },
                "paid and incurred",
            ),
            (
                Compartmental {
                    fitted: Fitted::PaidOnly,
                    ..preset()
                },
                "paid alone",
            ),
            (
                Compartmental {
                    fitted: Fitted::IncurredOnly,
                    ..preset()
                },
                "incurred alone",
            ),
        ];
        for (m, what) in cases {
            let e = fit(m);
            assert!(e.contains(what) && e.contains("not yet"), "{e}");
        }
    }

    #[test]
    fn bad_settings_are_invalid() {
        let tri = triangle(true);
        let fit = |m: &Compartmental| m.fit(&tri, "paid", incurred(), "premium").unwrap_err();
        let name = |e: Error| match e {
            Error::InvalidSetting { name, .. } => name,
            other => panic!("{other}"),
        };
        let with = |f: fn(&mut CompartmentalPriors)| {
            let mut priors = CompartmentalPriors::gesmann_morris();
            f(&mut priors);
            Compartmental { priors, ..preset() }
        };
        type Change = fn(&mut CompartmentalPriors);
        let cases: [(Change, &str); 5] = [
            (
                |p| {
                    p.k_p = PositivePrior::LogNormal {
                        median: -1.0,
                        sd: 0.1,
                    }
                },
                "k_p median",
            ),
            (
                |p| {
                    p.rlr = PositivePrior::LogNormal {
                        median: 0.7,
                        sd: 0.0,
                    }
                },
                "rlr sd",
            ),
            (
                |p| p.rrf_effect_sd = HalfPrior::Normal { scale: 0.0 },
                "rrf_effect_sd scale",
            ),
            (|p| p.lkj_eta = f64::NAN, "lkj_eta"),
            (
                |p| {
                    p.log_sigma[1] = RealPrior::Normal {
                        mean: f64::INFINITY,
                        sd: 1.0,
                    }
                },
                "log_sigma mean",
            ),
        ];
        for (f, expected) in cases {
            assert_eq!(name(fit(&with(f))), expected);
        }
        let m = Compartmental {
            n_sims: 0,
            ..preset()
        };
        assert_eq!(name(fit(&m)), "n_sims");
    }

    #[test]
    fn bad_data_are_errors() {
        let tri = triangle(true);
        let err = |paid: &str, os: Outstanding<'_>, premium: &str| {
            preset()
                .fit(&tri, paid, os, premium)
                .unwrap_err()
                .to_string()
        };
        // Missing columns.
        assert!(err("nope", incurred(), "premium").contains("nope"));
        assert!(err("paid", Outstanding::Column("nope"), "premium").contains("nope"));
        assert!(err("paid", incurred(), "nope").contains("nope"));
        // An outstanding column with no cell, and unequal grains.
        let origin: Vec<Month> = (0..3).map(|w| Month::january(2000 + w)).collect();
        let small = |development_grain: Grain, ages: &[Lag], os: &[f64]| {
            Triangle::from_long(&Long {
                keys: &[],
                origin: &origin,
                development: DevelopmentColumn::Age(ages),
                values: &[
                    ("paid", &[1.0, 2.0, 3.0]),
                    ("os", os),
                    ("premium", &[10.0, 10.0, 10.0]),
                ],
                origin_grain: Grain::Year,
                development_grain,
                cumulative: true,
            })
            .unwrap()
        };
        let none = small(Grain::Year, &[12, 12, 12], &[f64::NAN; 3]);
        let e = preset()
            .fit(&none, "paid", Outstanding::Column("os"), "premium")
            .unwrap_err()
            .to_string();
        assert!(e.contains("os"), "{e}");
        let quarterly = small(Grain::Quarter, &[3, 3, 3], &[1.0, 2.0, 3.0]);
        let e = preset()
            .fit(&quarterly, "paid", Outstanding::Column("os"), "premium")
            .unwrap_err()
            .to_string();
        assert!(e.contains("grain"), "{e}");
        let few = small(Grain::Year, &[12, 12, 12], &[1.0, 2.0, 3.0]);
        let e = preset()
            .fit(&few, "paid", Outstanding::Column("os"), "premium")
            .unwrap_err()
            .to_string();
        assert!(e.contains("more cells than parameters"), "{e}");
        // An origin with paid but no premium.
        let gap = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 12, 12]),
            values: &[
                ("paid", &[1.0, 2.0, 3.0]),
                ("os", &[1.0, 2.0, 3.0]),
                ("premium", &[10.0, f64::NAN, 10.0]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let e = preset()
            .fit(&gap, "paid", Outstanding::Column("os"), "premium")
            .unwrap_err()
            .to_string();
        assert!(e.contains("2001") && e.contains("premium"), "{e}");
    }

    #[test]
    fn the_prior_predictive_paths() {
        let tri = triangle(true);
        let paths = preset().prior_predictive(&tri, "premium", 2_000).unwrap();
        assert_eq!(paths.paid.n_components(), 100);
        assert_eq!(paths.outstanding.n_components(), 100);
        assert_eq!(paths.paid.dims(), ["origin", "age"]);
        // Paid rises with age; at the preset's medians (k_er 3, k_p 1, RLR
        // 0.7, RRF 0.8) the paid at age 10 is about 0.56 of premium.
        let key = |a: i64| vec![Period::year(1988).into(), KeyValue::Int(a)];
        let at = |a| paths.paid.marginal(&key(a)).unwrap().mean();
        assert!(at(12) < at(60) && at(60) < at(120));
        let median = paths
            .paid
            .marginal(&key(120))
            .unwrap()
            .quantile(0.5)
            .unwrap();
        assert!((median / PREMIUM[0] - 0.56).abs() < 0.05, "{median}");
        let again = preset().prior_predictive(&tri, "premium", 2_000).unwrap();
        assert_eq!(paths.paid.draw_matrix(), again.paid.draw_matrix());
        let ordered = Compartmental {
            rate_order: RateOrder::Ordered,
            ..preset()
        }
        .prior_predictive(&tri, "premium", 500)
        .unwrap();
        assert!(
            ordered
                .paid
                .draw_matrix()
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        );
        assert!(matches!(
            preset().prior_predictive(&tri, "premium", 0),
            Err(Error::InvalidSetting { name: "n_sims", .. })
        ));
    }

    #[test]
    fn the_layout_of_model_2() {
        let m = model_2(false);
        // 4 intercepts, 8 sds, 1 correlation, 72 z and 2 log sigma.
        assert_eq!(m.dim(), 87);
        assert_eq!(m.sampled_names().len(), 87);
        assert_eq!((m.n_levels(), m.n_dev_levels()), (9, 9));
        assert_eq!(m.dev_level[9], None);
        assert_eq!(m.layout.z0, 13);
        assert_eq!(m.layout.z_rate, [Some(31), Some(40)]);
        assert_eq!(m.layout.z_dev, [Some(49), Some(58), Some(67), Some(76)]);
        assert_eq!(m.log_sigma0(), 85);
        let names = m.names();
        // 6 population values, 4 x 9 origin values, 4 x 9 development
        // effects, 6 intercepts and sds, 2 rate sds, rho, 4 development
        // sds and 2 log sigma.
        assert_eq!(names.len(), 6 + 36 + 36 + 6 + 2 + 1 + 4 + 2);
        assert_eq!(names[6], "RLR[1988]");
        assert_eq!(names[24], "k_er[1988]");
        assert_eq!(names[42], "dev_ker[12]");
        assert_eq!(names[77], "dev_RRF[108]");
        assert_eq!(
            &names[82..],
            [
                "tau_RLR",
                "tau_RRF",
                "tau_ker",
                "tau_kp",
                "rho",
                "tau_dev_ker",
                "tau_dev_kp",
                "tau_dev_RLR",
                "tau_dev_RRF",
                "log_sigma_os",
                "log_sigma_pd"
            ]
        );
        let nat = m.natural(&points(&m)[0]);
        assert_eq!(nat.len(), names.len());
        // Loss ratios, incremental paid.
        let c = &m.cells;
        assert!((c[0].y - (62679. - 9558.) / PREMIUM[0]).abs() < 1e-15);
        assert!(c[45].paid && c[45].y == 9558. / PREMIUM[0] && c[45].t_before.is_none());
        assert_eq!(c[46].y, (22778. - 9558.) / PREMIUM[0]);
        assert_eq!((c[46].t_before, c[46].w), (Some(1.0), 1.0));
        assert_eq!(c[46].ln_y, c[46].y.ln());
        assert_eq!(
            CompartmentalPriors::gesmann_morris_2().hierarchy(),
            Hierarchy::Model2
        );
        let some = CompartmentalPriors {
            kp_effect_sd: None,
            ..CompartmentalPriors::gesmann_morris_2()
        };
        assert_eq!(some.hierarchy(), Hierarchy::Other);
        let m = model_with(some, false, Response::MODEL_2);
        assert_eq!(m.dim(), 87 - 1 - 9);
    }

    /// The log posterior written out in `f64` with the paper's form of the
    /// closed form, for either preset's hierarchy and any response, at `x`
    /// in the documented layout.
    fn written_out(two: bool, response: Response, x: &[f64]) -> f64 {
        let half_t = |l: f64, s: f64| l - 5.5 * ((l.exp() / s).powi(2) / 10.0).ln_1p();
        let (n_tau, z0, ls) = if two { (6, 13, 85) } else { (0, 7, 25) };
        let rho = x[6].tanh();
        let mut f =
            -0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) + (1.0 - rho * rho).ln();
        let scales = if two {
            f += half_t(x[4], 0.7) + half_t(x[5], 0.5);
            [0.3, 0.3, 0.3, 0.3, 0.7, 0.5]
        } else {
            f += half_t(x[4], 0.2) + half_t(x[5], 0.1);
            [0.0; 6]
        };
        for j in 0..n_tau {
            f += half_t(x[7 + j], scales[j]);
        }
        for z in &x[z0..ls] {
            f -= 0.5 * z * z;
        }
        for l in [x[ls], x[ls + 1]] {
            f += if two {
                -0.5 * ((l - 0.2f64.ln()) / 0.2).powi(2)
            } else {
                -((l / 1000.0).powi(2)).ln_1p()
            };
        }
        let tau = |j: usize| x[7 + j].exp();
        let pd = |a: f64, b: f64, rlr: f64, rrf: f64, t: f64| {
            rlr * rrf / (a - b) * (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp()))
        };
        for (w, row) in PAID.iter().enumerate() {
            for (d, &paid) in row.iter().enumerate() {
                let (z1, z2) = (x[z0 + 2 * w], x[z0 + 2 * w + 1]);
                let mut o = [
                    x[0],
                    x[1],
                    x[2] + x[4].exp() * z1,
                    x[3] + x[5].exp() * (rho * z1 + (1.0 - rho * rho).sqrt() * z2),
                ];
                if two {
                    o[0] += tau(0) * x[31 + w];
                    o[1] += tau(1) * x[40 + w];
                    for (p, o) in o.iter_mut().enumerate() {
                        *o += tau(2 + p) * x[49 + 9 * p + d];
                    }
                }
                let (a, b) = (3.0 * (0.1 * o[0]).exp(), (0.1 * o[1]).exp());
                let (rlr, rrf) = (0.7 * (0.2 * o[2]).exp(), 0.8 * (0.1 * o[3]).exp());
                let t = (d + 1) as f64;
                let os = rlr * a / (a - b) * ((-b * t).exp() - (-a * t).exp());
                let (paid_y, paid_mean) = if response.incremental && d > 0 {
                    (
                        paid - row[d - 1],
                        pd(a, b, rlr, rrf, t) - pd(a, b, rlr, rrf, t - 1.0),
                    )
                } else {
                    (paid, pd(a, b, rlr, rrf, t))
                };
                let unit = if response.loss_ratio { 1.0 } else { PREMIUM[w] };
                for (y, mean, l) in [
                    (INCURRED[w][d] - paid, os, x[ls]),
                    (paid_y, paid_mean, x[ls + 1]),
                ] {
                    let (y, mean) = (y / PREMIUM[w] * unit, mean * unit);
                    let r = if response.log_normal() {
                        (y.ln() - mean.ln()) / l.exp()
                    } else {
                        (y - mean) / l.exp()
                    };
                    f -= l + 0.5 * r * r;
                }
            }
        }
        f
    }

    #[test]
    fn the_density_is_each_model_written_out() {
        // Both presets' hierarchies with every response, at one point.
        for two in [false, true] {
            for likelihood in [
                CompartmentalLikelihood::Normal,
                CompartmentalLikelihood::LogNormal,
            ] {
                for loss_ratio in [false, true] {
                    for incremental in [false, true] {
                        let response = Response {
                            likelihood,
                            loss_ratio,
                            incremental,
                        };
                        let priors = if two {
                            CompartmentalPriors::gesmann_morris_2()
                        } else {
                            CompartmentalPriors::gesmann_morris()
                        };
                        let m = model_with(priors, false, response);
                        let mut x = points(&m)[3].clone();
                        let ls = m.log_sigma0();
                        (x[ls], x[ls + 1]) = match (response.log_normal(), loss_ratio) {
                            (true, _) => (-1.8, -1.9),
                            (false, true) => (-3.0, -3.5),
                            (false, false) => (8.25, 6.66),
                        };
                        let want = written_out(two, response, &x);
                        let mut g = vec![0.0; m.dim()];
                        let got = m.log_posterior(&x, &mut g);
                        assert!(
                            (got - want).abs() <= 1e-12 * want.abs(),
                            "{two} {response:?}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn responses_the_likelihood_cannot_take_are_errors() {
        let origin: Vec<Month> = [0, 0, 1].map(|w| Month::january(2000 + w)).to_vec();
        let tri = |ages: &[Lag], paid: &[f64], os: &[f64]| {
            Triangle::from_long(&Long {
                keys: &[],
                origin: &origin,
                development: DevelopmentColumn::Age(ages),
                values: &[("paid", paid), ("os", os), ("premium", &[10.0; 3])],
                origin_grain: Grain::Year,
                development_grain: Grain::Year,
                cumulative: true,
            })
            .unwrap()
        };
        let err = |m: &Compartmental, t: &Triangle| {
            m.fit(t, "paid", Outstanding::Column("os"), "premium")
                .unwrap_err()
                .to_string()
        };
        // A zero outstanding, and a paid that does not grow: no log.
        let zero_os = tri(&[12, 24, 12], &[1.0, 2.0, 1.0], &[3.0, 0.0, 2.0]);
        let e = err(&preset_2(), &zero_os);
        assert!(
            e.contains("2000") && e.contains("24 months") && e.contains("outstanding 0"),
            "{e}"
        );
        let flat = tri(&[12, 24, 12], &[1.0, 1.0, 1.0], &[3.0, 2.0, 2.0]);
        let e = err(&preset_2(), &flat);
        assert!(
            e.contains("incremental paid 0") && e.contains("positive"),
            "{e}"
        );
        // Incremental paid with no paid at the age before.
        let gap = tri(&[12, 36, 12], &[1.0, 2.0, 1.0], &[3.0, 2.0, 2.0]);
        let normal = Compartmental {
            paid_response: PaidResponse::Incremental,
            ..preset()
        };
        let e = err(&normal, &gap);
        assert!(
            e.contains("36 months") && e.contains("none at 24 months"),
            "{e}"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn model_2_fits_and_predicts_increments() {
        let tri = triangle(true);
        let fit = Compartmental {
            sampler: Sampler {
                chains: 2,
                tune: 60,
                draws: 60,
                target_accept: 0.9,
                ..preset_2().sampler
            },
            n_sims: 200,
            seed: 6,
            ..preset_2()
        }
        .fit(&tri, "paid", incurred(), "premium")
        .unwrap();
        assert_eq!(fit.population_origins, [Period::year(1997)]);
        assert_eq!(fit.posterior.dim(), 93);
        assert!(fit.posterior.names().iter().any(|n| n == "tau_dev_RRF"));
        let share = fit.cell_rate_swaps.unwrap();
        assert!((0.0..=1.0).contains(&share), "{share}");
        assert_eq!(fit.loo().unwrap().loo.pareto_k.len(), 90);
        assert_eq!(fit.reserves.provenance().model, "compartmental_gm2");
        // Log-normal increments are positive, so each simulation's paid
        // grows with age from the latest paid, and the reserve is the paid
        // at age 10 less the latest.
        let keys = fit.paid.components();
        let n = keys.len();
        for (row, reserves) in fit
            .paid
            .draw_matrix()
            .chunks_exact(n)
            .zip(fit.reserves.draw_matrix().chunks_exact(10))
        {
            for j in 0..n {
                let (KeyValue::Period(o), KeyValue::Int(age)) = (&keys[j][0], &keys[j][1]) else {
                    unreachable!("cell keys are (origin, age)")
                };
                let i = fit.origins.iter().position(|p| p == o).unwrap();
                let before = if j > 0 && keys[j - 1][0] == keys[j][0] {
                    row[j - 1]
                } else {
                    fit.latest_paid[i]
                };
                assert!(row[j] > before, "{o} {age}: {} after {before}", row[j]);
                if *age == 120 {
                    assert!((row[j] - fit.latest_paid[i] - reserves[i]).abs() < 1e-9 * row[j]);
                }
            }
        }
        assert!(
            fit.ultimate_loss_ratio
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        );
    }

    #[test]
    fn the_prior_predictive_paths_of_model_2() {
        let tri = triangle(true);
        let paths = preset_2().prior_predictive(&tri, "premium", 500).unwrap();
        assert_eq!(paths.paid.n_components(), 100);
        // The increments are positive, so cumulative paid grows with age in
        // every draw.
        let draws = paths.paid.draw_matrix();
        for row in draws.chunks_exact(100) {
            for i in 0..10 {
                for k in 1..10 {
                    assert!(row[10 * i + k] > row[10 * i + k - 1]);
                }
            }
        }
        let os = paths.outstanding.draw_matrix();
        assert!(os.iter().all(|v| v.is_finite() && *v > 0.0));
        let again = preset_2().prior_predictive(&tri, "premium", 500).unwrap();
        assert_eq!(draws, again.paid.draw_matrix());
        // The provenance records the paid response and the priors.
        let param = |pd: &PredictiveDistribution, key: &str| {
            pd.provenance()
                .parameters
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        let cumulative = Compartmental {
            paid_response: PaidResponse::Cumulative,
            ..preset_2()
        }
        .prior_predictive(&tri, "premium", 10)
        .unwrap();
        assert_ne!(
            param(&paths.paid, "paid_response"),
            param(&cumulative.paid, "paid_response")
        );
        let model_1 = preset().prior_predictive(&tri, "premium", 10).unwrap();
        assert_ne!(
            param(&paths.outstanding, "priors"),
            param(&model_1.outstanding, "priors")
        );
    }

    #[test]
    fn the_cellwise_gradient_is_the_generic_densitys() {
        // The per-cell gradient against the chunked dual pass over the
        // generic density (the oracle), at 20 points around model 2's
        // posterior: both rate orders, a partial hierarchy, and amounts with
        // the normal likelihood.
        let partial = CompartmentalPriors {
            ker_effect_sd: None,
            development_effect_sd: [None, Some(HalfPrior::Normal { scale: 0.3 }), None, None],
            ..CompartmentalPriors::gesmann_morris_2()
        };
        let models = [
            model_2(false),
            model_2(true),
            model_with(partial, false, Response::MODEL_2),
            model_with(
                CompartmentalPriors::gesmann_morris_2(),
                false,
                Response::MODEL_1,
            ),
        ];
        let (mut worst_value, mut worst_grad): (f64, f64) = (0.0, 0.0);
        for (n, m) in models.iter().enumerate() {
            assert!(m.cellwise);
            for mut x in points(m) {
                if n == 3 {
                    let ls = m.log_sigma0();
                    (x[ls], x[ls + 1]) = (8.25, 6.66);
                }
                let mut want = vec![0.0; m.dim()];
                let f = gradient::<CHUNK>(&x, &mut want, |t| m.log_density(t));
                let mut got = vec![0.0; m.dim()];
                let g = m.cellwise_gradient(&x, &mut got);
                let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                worst_value = worst_value.max((g - f).abs() / f.abs());
                for (a, b) in got.iter().zip(&want) {
                    worst_grad = worst_grad.max((a - b).abs() / scale);
                }
            }
        }
        println!("cellwise gradient: value {worst_value:.1e}, gradient {worst_grad:.1e}");
        assert!(worst_value <= 1e-12 && worst_grad <= 1e-12);
    }

    /// The cost of model 2's log posterior on the paper's training triangle
    /// (87 parameters), release build: median time of the value and of the
    /// value with its gradient.
    #[test]
    #[ignore = "a release benchmark; run with --release --ignored --nocapture"]
    fn model_2_gradient_benchmark() {
        let median = |f: &mut dyn FnMut()| {
            let mut times: Vec<f64> = (0..41)
                .map(|_| {
                    let start = std::time::Instant::now();
                    for _ in 0..200 {
                        f();
                    }
                    start.elapsed().as_secs_f64() / 200.0 * 1e6
                })
                .collect();
            times.sort_by(f64::total_cmp);
            times[20]
        };
        for (label, m) in [("model 1", model(false)), ("model 2", model_2(false))] {
            let x = points(&m)[0].clone();
            let mut g = vec![0.0; m.dim()];
            let value = median(&mut || {
                std::hint::black_box(m.log_density::<f64>(std::hint::black_box(&x)));
            });
            let chunked = median(&mut || {
                std::hint::black_box(gradient::<CHUNK>(std::hint::black_box(&x), &mut g, |t| {
                    m.log_density(t)
                }));
            });
            let grad = median(&mut || {
                std::hint::black_box(m.log_posterior(std::hint::black_box(&x), &mut g));
            });
            println!(
                "{label}, p = {}: value {value:.2} us, chunked dual gradient {chunked:.2} us \
                 ({:.1}x), the sampler's gradient {grad:.2} us ({:.1}x)",
                m.dim(),
                chunked / value,
                grad / value
            );
        }
    }

    /// The training triangle's square of 1988 to 1992 at ages up to 60
    /// months: every origin at the square's last age, so no year.
    fn closed_triangle() -> Triangle {
        let (mut origin, mut ages) = (vec![], vec![]);
        let (mut paid, mut inc, mut prem) = (vec![], vec![], vec![]);
        for w in 0..5 {
            for d in 0..5 {
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as Lag + 1));
                paid.push(PAID[w][d]);
                inc.push(INCURRED[w][d]);
                prem.push(PREMIUM[w]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("incurred", &inc), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn the_coming_year_of_the_training_triangle() {
        use crate::bayes::one_year::YearModel;
        for (preset, cells) in [(preset(), 20), (preset_2(), 20)] {
            let data = Data::new(&triangle(true), "paid", incurred(), "premium").unwrap();
            let year = year::CompartmentalYear::new(&preset, &data).unwrap();
            // Every origin's next diagonal, paid and outstanding: 1988 at
            // 120 months, ..., 1997 at 12 months.
            assert_eq!(year.n_year(), cells);
            let data = Data::new(&triangle(false), "paid", incurred(), "premium").unwrap();
            // Without 1997 the square has 9 ages and 1988 is at the last.
            assert_eq!(
                year::CompartmentalYear::new(&preset, &data)
                    .unwrap()
                    .n_year(),
                16
            );
            let data = Data::new(&closed_triangle(), "paid", incurred(), "premium").unwrap();
            assert_eq!(
                year::CompartmentalYear::new(&preset, &data)
                    .unwrap()
                    .n_year(),
                0
            );
        }
    }

    /// The one-year conformance suite (`one_year::conformance`) on
    /// `preset` with a small sampler: the training triangle with 1997, and
    /// the closed square of [`closed_triangle`], each sampled directly (the
    /// closed square has fewer cells than model 2 has parameters, which
    /// `fit` refuses but the posterior does not need).
    fn one_year_conformance(name: &str, preset: Compartmental) {
        use crate::bayes::one_year::YearModel;
        let m = Compartmental {
            sampler: Sampler {
                chains: 2,
                tune: 100,
                draws: 100,
                target_accept: 0.9,
                ..preset.sampler
            },
            ..preset
        };
        let fitted = |tri: &Triangle| {
            let data = Data::new(tri, "paid", incurred(), "premium").unwrap();
            let model = Model::new(&data, m.priors, false, m.response()).unwrap();
            let (_, sampled, _) = m.sample(&model, 5).unwrap();
            (data, model, sampled)
        };
        let (open, closed) = (fitted(&triangle(true)), fitted(&closed_triangle()));
        let year = year::CompartmentalYear::new(&m, &open.0).unwrap();
        let opening = year.opening_draws(&open.1, &open.2, 9).unwrap();
        let closed_year = year::CompartmentalYear::new(&m, &closed.0).unwrap();
        let closed_opening = closed_year.opening_draws(&closed.1, &closed.2, 9).unwrap();
        // 1997's two `z` (and, with development effects, age 10's) are
        // added to each draw.
        let added = if m.priors.hierarchy() == Hierarchy::Model1 {
            2
        } else {
            4 + 4
        };
        assert_eq!(opening.dim(), open.2.dim() + added);
        assert_eq!(year.n_year(), 20);
        crate::bayes::one_year::conformance::check(
            name,
            &year,
            &opening,
            &closed_year,
            &closed_opening,
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_of_model_1_conforms() {
        one_year_conformance("compartmental model 1", preset());
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_of_model_2_conforms() {
        one_year_conformance("compartmental model 2", preset_2());
    }

    /// The opening reserve, the posterior mean over the draws with 1997's
    /// effects added, is the fit's mean reserve, whose simulations draw
    /// them from the population, within Monte Carlo error; the tables have
    /// one row per premium origin.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_opening_reserve_is_the_fits() {
        use crate::bayes::one_year::YearModel;
        let only = BayesOneYear {
            refit: Some(OneYearRefit::ReweightOnly),
            // Above the default cap of `ONE_YEAR_SIMS`.
            n_sims: Some(4000),
            ..BayesOneYear::default()
        };
        for preset in [preset(), preset_2()] {
            let m = Compartmental {
                sampler: Sampler {
                    chains: 2,
                    tune: 100,
                    draws: 100,
                    target_accept: 0.9,
                    ..preset.sampler
                },
                n_sims: 4000,
                seed: 2,
                ..preset
            };
            let view = m
                .one_year(&triangle(true), "paid", incurred(), "premium", &only)
                .unwrap();
            let total = view.fit.reserves.total();
            let opening: f64 = view.opening_reserve.iter().sum();
            // The error of both means: the simulations', and the draws'
            // (each with its own 1997 effects) of the opening.
            let year = year::CompartmentalYear::new(&m, &view.fit.data).unwrap();
            let draws = year
                .opening_draws(&view.fit.model, &view.fit.sampled, 1)
                .unwrap();
            let mut row = vec![0.0; 10];
            let by_draw: Vec<f64> = (0..draws.n_draws())
                .map(|d| {
                    year.opening(draws.draw(d), &mut row);
                    row.iter().sum()
                })
                .collect();
            let s = by_draw.len() as f64;
            let mean_d = by_draw.iter().sum::<f64>() / s;
            let var_d = by_draw.iter().map(|v| (v - mean_d).powi(2)).sum::<f64>() / s;
            let se = (total.variance() / m.n_sims as f64 + var_d / s).sqrt();

            assert!(
                (opening - total.mean()).abs() < 4.0 * se,
                "opening {opening}, fit {} (se {se})",
                total.mean()
            );
            assert_eq!(view.cdr.n_components(), 10);
            assert_eq!(view.pareto_k.len(), 4000);
            assert!(view.refitted.is_empty() && view.moment_matched.is_empty());
            assert!(view.cdr.std_dev() < view.fit.reserves.std_dev());
            let long = view.to_long();
            assert_eq!(long.origin.as_ref().unwrap().len(), 10);
            let totals = view.totals();
            assert!(totals.values.iter().any(|(n, _)| n == "flagged_share"));
            assert!(view.cdr.provenance().model.ends_with("_one_year"));
        }
    }

    /// A refit budget refits the first simulations due for one, in
    /// order, and reweights and lists the others; the years, the `k̂` and
    /// the simulations within the budget are those without one, and those
    /// beyond it are `ReweightOnly`'s, bit for bit.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_refit_budget_refits_the_first_and_lists_the_rest() {
        let m = Compartmental {
            n_sims: 12,
            ..quick(4)
        };
        let with = |refit: OneYearRefit, max_refits: Option<usize>| {
            m.one_year(
                &triangle(true),
                "paid",
                incurred(),
                "premium",
                &BayesOneYear {
                    refit: Some(refit),
                    max_refits,
                    ..BayesOneYear::default()
                },
            )
            .unwrap()
        };
        let view = |max_refits: Option<usize>| with(OneYearRefit::RefitFlagged, max_refits);
        let all = view(None);
        assert!(all.refitted.len() >= 3, "{:?}", all.pareto_k);
        assert!(all.unrefitted.is_empty());
        let two = view(Some(2));
        assert_eq!(two.pareto_k, all.pareto_k);
        assert_eq!(two.refitted, all.refitted[..2]);
        assert_eq!(two.unrefitted, all.refitted[2..]);
        let n = 10;
        for (a, b) in two
            .cdr
            .draw_matrix()
            .chunks(n)
            .zip(all.cdr.draw_matrix().chunks(n))
            .enumerate()
            .filter(|(i, _)| !two.unrefitted.contains(i))
            .map(|(_, r)| r)
        {
            assert_eq!(a, b);
        }
        let none = view(Some(0));
        assert!(none.refitted.is_empty());
        assert_eq!(none.unrefitted, all.refitted);
        let only = with(OneYearRefit::ReweightOnly, None);
        assert_eq!(none.cdr.draw_matrix(), only.cdr.draw_matrix());
        for (a, b) in two
            .cdr
            .draw_matrix()
            .chunks(n)
            .zip(only.cdr.draw_matrix().chunks(n))
            .enumerate()
            .filter(|(i, _)| two.unrefitted.contains(i))
            .map(|(_, r)| r)
        {
            assert_eq!(a, b);
        }
        assert!(
            none.cdr
                .provenance()
                .parameters
                .contains(&("max_refits".into(), "0".into()))
        );
    }

    /// The first five origins of the training triangle, ages 1 to 5 up to
    /// the diagonal, once for each of `names` (a key `lob` when `keyed`).
    fn lines(names: &[&str], keyed: bool) -> Triangle {
        let (mut lob, mut origin, mut ages) = (vec![], vec![], vec![]);
        let (mut paid, mut inc, mut prem) = (vec![], vec![], vec![]);
        for name in names {
            for w in 0..5 {
                for d in 0..5 - w {
                    lob.push(*name);
                    origin.push(Month::january(1988 + w as i32));
                    ages.push(12 * (d as Lag + 1));
                    paid.push(PAID[w][d]);
                    inc.push(INCURRED[w][d]);
                    prem.push(PREMIUM[w]);
                }
            }
        }
        let key = [("lob", &lob[..])];
        Triangle::from_long(&Long {
            keys: if keyed { &key } else { &[] },
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("incurred", &inc), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn segments_of(m: &Compartmental, tri: &Triangle) -> CompartmentalFits {
        m.fit_segments(tri, "paid", incurred(), "premium").unwrap()
    }

    /// Each line's total of `pd` (dimensions `lob`, `origin`), simulation by
    /// simulation.
    fn line_totals(pd: &PredictiveDistribution) -> Vec<Vec<f64>> {
        let by = pd.aggregate(&["lob"]).unwrap();
        let n = by.n_components();
        (0..n)
            .map(|j| by.draw_matrix().chunks(n).map(|r| r[j]).collect())
            .collect()
    }

    fn spearman(x: &[f64], y: &[f64]) -> f64 {
        let ranks = |v: &[f64]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
            let mut r = vec![0.0; v.len()];
            for (k, &i) in idx.iter().enumerate() {
                r[i] = k as f64;
            }
            r
        };
        let (rx, ry) = (ranks(x), ranks(y));
        let m = (x.len() as f64 - 1.0) / 2.0;
        let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
        let sxx: f64 = rx.iter().map(|a| (a - m).powi(2)).sum();
        sxy / sxx
    }

    /// `Synchronized` and a bad matrix fail before any sampling (the test
    /// would take minutes if either sampled), `Synchronized` as
    /// `Error::Bayes` and a bad matrix as `InvalidSetting`.
    #[test]
    fn synchronized_and_a_bad_matrix_fail_before_sampling() {
        let tri = lines(&["A", "B"], true);
        let run = |dependence| {
            Compartmental {
                dependence,
                ..quick(1)
            }
            .fit_segments(&tri, "paid", incurred(), "premium")
            .unwrap_err()
        };
        assert!(matches!(
            run(SegmentDependence::Synchronized),
            Error::Bayes(m) if m.contains("Synchronized")
        ));
        for spearman in [
            vec![1.0, 0.5],
            vec![1.0, 0.5, 0.4, 1.0],
            vec![2.0, 0.5, 0.5, 1.0],
            vec![1.0, 1.1, 1.1, 1.0],
        ] {
            assert!(
                matches!(
                    run(SegmentDependence::RankCorrelation { spearman }),
                    Error::InvalidSetting {
                        name: "spearman",
                        ..
                    }
                ),
                "bad matrix"
            );
        }
    }

    /// Each segment samples and simulates from its own streams, `fit` does
    /// not change, and a segment is `fit` on its own triangle with that
    /// segment's streams.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn each_segment_has_its_own_seeds_and_fit_is_unchanged() {
        let tri = lines(&["A", "B"], true);
        let m = quick(1);
        let fits = segments_of(&m, &tri);
        assert_eq!(fits.segments.len(), 2);
        let [a, b] = [0, 1].map(|k| &fits.segments.fits[k].fit);
        // "B" is "A" under another name: with the same streams the posteriors
        // would agree draw by draw; with their own they do not.
        assert_ne!(a.posterior, b.posterior);
        let only = |lob: &str| tri.select(&[("lob", &[lob][..])]).unwrap();
        for (k, lob) in [(0, "A"), (1, "B")] {
            let alone = m
                .fit_streams(
                    &only(lob),
                    "paid",
                    incurred(),
                    "premium",
                    Stream::SegmentChains(k),
                    Stream::SegmentPredictive(k),
                )
                .unwrap();
            let seg = &fits.segments.fits[k].fit;
            assert_eq!(alone.posterior, seg.posterior);
            assert_eq!(alone.reserves.draw_matrix(), seg.reserves.draw_matrix());
            assert_eq!(alone.paid.draw_matrix(), seg.paid.draw_matrix());
            assert_eq!(
                alone.outstanding.draw_matrix(),
                seg.outstanding.draw_matrix()
            );
        }
        // `fit` is the base streams' fit.
        let base = m.fit(&only("A"), "paid", incurred(), "premium").unwrap();
        let again = m
            .fit_streams(
                &only("A"),
                "paid",
                incurred(),
                "premium",
                Stream::Chains,
                Stream::Predictive,
            )
            .unwrap();
        assert_eq!(base.posterior, again.posterior);
        assert_eq!(base.reserves.draw_matrix(), again.reserves.draw_matrix());
        assert_ne!(base.posterior, a.posterior);
        // The joint results.
        assert_eq!(fits.reserves.dims(), ["lob", "origin"]);
        assert_eq!(fits.reserves.n_components(), 10);
        assert_eq!(fits.paid.dims(), ["lob", "origin", "age"]);
        assert_eq!(fits.paid.n_components(), 2 * a.paid.n_components());
        assert_eq!(
            fits.outstanding.n_components(),
            2 * a.outstanding.n_components()
        );
        let long = fits.to_long();
        assert_eq!(long.keys[0].1.len(), 10);
        assert_eq!(long.n_rows(), 10);
        assert_eq!(fits.totals().n_rows(), 2);
        assert_eq!(
            fits.paid_table().unwrap().n_rows(),
            fits.paid.n_components()
        );
        assert_eq!(fits.outstanding_table().unwrap().keys[0].1[0], "A");
        let one = fits.segment(&[("lob", "B")]).unwrap();
        assert_eq!(one.reserves.n_components(), 5);
        assert_eq!(one.reserves.dims(), ["lob", "origin"]);
        let param = |pd: &PredictiveDistribution, key: &str| {
            pd.provenance()
                .parameters
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(param(&fits.reserves, "segments").as_deref(), Some("2"));
        assert_eq!(param(&fits.paid, "output").as_deref(), Some("paid"));
        assert_eq!(
            param(&fits.reserves, "dependence").as_deref(),
            Some("Independent")
        );
    }

    /// A triangle without keys is one segment, fitted with segment 0's
    /// streams.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_one_segment_fit_is_fit_with_segment_zero_streams() {
        let tri = lines(&["A"], false);
        let m = quick(2);
        let fits = segments_of(&m, &tri);
        assert_eq!(fits.segments.len(), 1);
        let alone = m
            .fit_streams(
                &tri,
                "paid",
                incurred(),
                "premium",
                Stream::SegmentChains(0),
                Stream::SegmentPredictive(0),
            )
            .unwrap();
        let seg = &fits.segments.fits[0].fit;
        assert_eq!(alone.posterior, seg.posterior);
        assert_eq!(alone.reserves.draw_matrix(), seg.reserves.draw_matrix());
        assert_eq!(fits.reserves.draw_matrix(), seg.reserves.draw_matrix());
        assert_eq!(fits.paid.draw_matrix(), seg.paid.draw_matrix());
        assert_ne!(
            m.fit(&tri, "paid", incurred(), "premium")
                .unwrap()
                .posterior,
            seg.posterior
        );
    }

    /// The rank correlation reorders whole rows: each segment keeps its
    /// simulations (as a set), its reserves stay its paid at the last age
    /// less the latest paid in every simulation, and the totals reach the
    /// target.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_joint_reorder_keeps_each_segments_rows_consistent() {
        let tri = lines(&["A", "B"], true);
        let m = quick(3);
        let independent = segments_of(&m, &tri);
        let ranked = segments_of(
            &Compartmental {
                dependence: SegmentDependence::RankCorrelation {
                    spearman: vec![1.0, 0.8, 0.8, 1.0],
                },
                ..m.clone()
            },
            &tri,
        );
        let sorted = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v
        };
        let (ti, tr) = (
            line_totals(&independent.reserves),
            line_totals(&ranked.reserves),
        );
        for s in 0..2 {
            assert_eq!(sorted(ti[s].clone()), sorted(tr[s].clone()));
        }
        // The first segment's rows are shuffled, not left sorted.
        assert!(tr[0].windows(2).any(|w| w[0] > w[1]));
        let r = spearman(&ti[0], &ti[1]);
        assert!(r.abs() < 0.2, "independent lines: {r}");
        let r = spearman(&tr[0], &tr[1]);
        assert!((r - 0.8).abs() < 0.05, "ranked lines: {r}");
        // Every segment's own parts agree, simulation by simulation.
        for (s, seg) in ranked.segments.fits.iter().enumerate() {
            let f = &seg.fit;
            let n_sims = f.reserves.n_sims();
            let n = f.reserves.n_components();
            let paid_n = f.paid.n_components();
            for (i, origin) in f.origins.iter().enumerate() {
                let Some(last) = f
                    .paid
                    .components()
                    .iter()
                    .position(|k| k[0] == KeyValue::from(*origin) && k[1] == KeyValue::Int(60))
                else {
                    continue;
                };
                for sim in 0..n_sims {
                    let reserve = f.reserves.draw_matrix()[sim * n + i];
                    let at_last = f.paid.draw_matrix()[sim * paid_n + last];
                    assert!(
                        (at_last - f.latest_paid[i] - reserve).abs()
                            < 1e-6 * at_last.abs().max(1.0),
                        "segment {s} origin {origin} simulation {sim}"
                    );
                }
            }
            // The joint distributions hold the segment's parts.
            let joint_n = ranked.reserves.n_components();
            for sim in 0..n_sims {
                assert_eq!(
                    &ranked.reserves.draw_matrix()
                        [sim * joint_n + s * n..sim * joint_n + (s + 1) * n],
                    &f.reserves.draw_matrix()[sim * n..(sim + 1) * n]
                );
            }
        }
    }
}
