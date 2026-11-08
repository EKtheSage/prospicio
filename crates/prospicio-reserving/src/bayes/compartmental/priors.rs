//! The priors of a compartmental fit and the helpers that draw from them or put them on the log scale.

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_core::StreamRng;
use prospicio_math::special::{norm_cdf, norm_quantile};
use prospicio_prob::{Distribution, Gamma};

use super::super::scalar::Scalar;
#[cfg(doc)]
use super::fit::Compartmental;
#[cfg(doc)]
use super::settings::Fitted;
use crate::error::{Error, Result};

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
///
/// [`diffuse`](Self::diffuse) is the starting point for a fit to paid alone
/// ([`Fitted::PaidOnly`]), where the paid cells pin the rates and the data
/// should set them; a narrow prior, such as
/// [`gesmann_morris`](Self::gesmann_morris)'s, is the option of a user who
/// holds information about the rates that is not in the data. A fit to
/// incurred alone ([`Fitted::IncurredOnly`]) refuses a `k_p` prior wider
/// than [`MAX_INCURRED_ONLY_KP_SD`], so it needs a prior the user believes.
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

    /// A diffuse set for model 1 (decision 12, the user's decision of
    /// 2026-10-22): the paper's medians (`k_er` 3, `k_p` 1, `RLR` 0.7, `RRF`
    /// 0.8) with a log-scale sd of 1 on each of the four, 5 to 10 times the
    /// paper's, so each prior spans about a factor of 7 either side of its
    /// median at two sds and the data, not the prior, set the rates. The
    /// origin effect sds and the priors on `log σ` are
    /// [`gesmann_morris`](Self::gesmann_morris)'s.
    ///
    /// Use it as the starting point for a fit to paid alone
    /// ([`Fitted::PaidOnly`]), where the paid cells pin the rates. It is
    /// refused for a fit to incurred alone ([`Fitted::IncurredOnly`]): no
    /// paid cell is fitted there, so the payment rate `k_p`, and with it the
    /// paid run-off and the reserve, follow the prior on `k_p` at every
    /// width, and a `k_p` sd above [`MAX_INCURRED_ONLY_KP_SD`] (0.5) is an
    /// error (the user's decision of 2026-10-23). For incurred alone use a
    /// struct update on this preset with a `k_p` prior you believe (the
    /// paper's `k_p` prior, sd 0.1, put the reserve on company 337 26% above
    /// the fit to paid and outstanding), for example one from the book's
    /// paid history or from a paid-and-incurred fit of a similar book. Check
    /// any set with [`Compartmental::prior_predictive`].
    pub fn diffuse() -> Self {
        let wide = |median: f64| PositivePrior::LogNormal { median, sd: 1.0 };
        Self {
            k_er: wide(3.0),
            k_p: wide(1.0),
            rlr: wide(0.7),
            rrf: wide(0.8),
            ..Self::gesmann_morris()
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
    pub(super) fn rates_vary(&self) -> bool {
        self.ker_effect_sd.is_some()
            || self.kp_effect_sd.is_some()
            || self.development_effect_sd[..2].iter().any(Option::is_some)
    }

    pub(super) fn validate(&self) -> Result<()> {
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
pub(super) const MIN_ORDERED_MASS: f64 = 1e-6;

/// The restriction `k_er > k_p` on the rate intercepts' standard normals
/// `(b_ker, b_kp)` as `z > a`, where `z = (s_er b_ker - s_p b_kp) / s` is
/// standard normal and `s = sqrt(s_er^2 + s_p^2)`: `(a, s)`. The priors'
/// mass on the restriction is `Φ(-a)`.
pub(super) fn rate_order_bound(pr: CompartmentalPriors) -> (f64, f64) {
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
pub(super) fn ordered_rates(pr: CompartmentalPriors, rng: &mut StreamRng) -> (f64, f64) {
    let (_, s_er) = lognormal(pr.k_er);
    let (_, s_p) = lognormal(pr.k_p);
    let (a, s) = rate_order_bound(pr);
    let z = -norm_quantile(norm_cdf(-a) * rng.next_open01());
    let y = norm_quantile(rng.next_open01());
    ((s_er * z + s_p * y) / s, (s_er * y - s_p * z) / s)
}

/// The widest log-scale sd of the `k_p` prior that [`Fitted::IncurredOnly`]
/// accepts (decision 12, the user's decision of 2026-10-23). The incurred
/// alone does not identify the payment rate, so the reserve follows this
/// prior at every width; a wider prior is refused and the user must give one
/// they believe. [`CompartmentalPriors::diffuse`] (sd 1) is refused.
pub const MAX_INCURRED_ONLY_KP_SD: f64 = 0.5;

/// Under [`Fitted::IncurredOnly`] the fit warns when the posterior sd of
/// `log k_p` is more than this share of the prior's (decision 12): the data
/// have then told the payment rate little and the paid run-off is the
/// prior's. On company 337 the share is 0.99 under the paper's preset, 0.6
/// under the diffuse one and 0.9 to 0.7 in between (log-scale sds 0.3 to
/// 0.7 on the four priors), against 0.05 to 0.4 when paid is fitted, alone
/// or with outstanding
/// (`knowledge/findings/compartmental-one-response.md`).
pub(super) const PRIOR_SHARE_WARNING: f64 = 0.5;

/// The posterior sd of `log k_p` over its prior's log-scale sd, `None` for
/// fewer than two finite draws.
pub(super) fn payment_rate_prior_share(
    posterior: &PosteriorDraws,
    priors: CompartmentalPriors,
) -> Option<f64> {
    let (_, prior_sd) = lognormal(priors.k_p);
    let logs: Vec<f64> = posterior
        .values()
        .chunks_exact(posterior.dim())
        .map(|d| d[1].ln())
        .collect();
    if logs.len() < 2 || !logs.iter().all(|v| v.is_finite()) {
        return None;
    }
    let n = logs.len() as f64;
    let mean = logs.iter().sum::<f64>() / n;
    let var = logs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some(var.sqrt() / prior_sd)
}

/// `(median, sd)` of a [`PositivePrior`].
pub(super) fn lognormal(prior: PositivePrior) -> (f64, f64) {
    let PositivePrior::LogNormal { median, sd } = prior;
    (median, sd)
}

/// The log density of a half prior on `τ = e^l`, in `l`, with the Jacobian
/// `l` and without constants.
pub(super) fn half_on_log<S: Scalar>(l: S, prior: HalfPrior) -> S {
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
pub(super) fn real_prior<S: Scalar>(x: S, prior: RealPrior) -> S {
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
pub(super) fn half_median(prior: HalfPrior) -> f64 {
    match prior {
        HalfPrior::StudentT { scale, .. } | HalfPrior::Normal { scale } => 0.6745 * scale,
    }
}

/// One draw of a half prior.
pub(super) fn draw_half(prior: HalfPrior, rng: &mut StreamRng) -> f64 {
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
