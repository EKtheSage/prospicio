//! An industry prior for a thin program from other companies' triangles
//! (`docs/design/reserving-bayes.md`, decision 29, issue #604).
//!
//! For a new or thin program the credible prior of the loss ratio, its
//! dispersion and the development pattern is an industry benchmark by
//! line, typically from Schedule P. [`IndustryPrior`] turns triangles that
//! the user supplies (one per company, with each origin's premium) into
//! the inputs of [`BayesBfNormalised`](crate::bayes::BayesBfNormalised):
//!
//! * `apriori`, the mean of the companies' expected ultimate loss ratios;
//! * `between_cv`, the cv of the companies' levels about that mean, the
//!   normalised BF's `common_cv` (one multiplier shared by every origin of
//!   the program, which the program's data update; decision 25);
//! * `cv`, the cv of an origin's ultimate about its company's level, the
//!   normalised BF's `cv`: the origin-to-origin spread of the realised
//!   loss ratios (`within_cv`), or, with
//!   [`IndustryPriorSettings::subtract_process`], that spread less the ODP
//!   process error in it (`process_cv`), which the model adds back through
//!   its scale;
//! * `pattern`, a [`PatternPrior`]: the mean and standard deviation over
//!   the companies of `beta_d = log(p_d / p_1)`, the log ratio of the share
//!   of the ultimate at age `d` to the first age's, which is the scale on
//!   which the normalised BF puts its normal prior on the pattern.
//!
//! The estimators are moments on the natural scale, so that they are the
//! moments of the model the prior feeds (a gamma multiplier of mean 1 and
//! cv `between_cv` times a gamma of mean `apriori` and cv `cv`). With
//! `x_cw` the chain ladder's ultimate loss ratio of origin `w` of company
//! `c`, `n_c` the company's origins kept and `xbar_c` their mean:
//!
//! ```text
//! apriori        = mean_c xbar_c
//! r              = sum_c sum_w (x_cw - xbar_c)^2 / sum_c (n_c - 1) xbar_c^2
//! within_cv^2    = r / (1 - r / nbar)                 nbar = mean n_c (n_c >= 2)
//! 1 + q          = 1 + var_c(xbar_c) / apriori^2
//! between_cv^2   = (1 + q) / mean_c(1 + within_cv^2 / n_c) - 1
//! process_cv^2   = sum_c (1 - 1/n_c) sum_w v_cw / sum_c (n_c - 1) xbar_c^2 * within_cv^2 / r
//! v_cw           = phi_c x_cw / (P_cw premium_cw)
//! cv^2           = within_cv^2   (or within_cv^2 - process_cv^2)
//! ```
//!
//! `phi_c` is the company's ODP Pearson scale and `P_cw` the share of the
//! ultimate developed at the origin's latest age by the company's own
//! chain ladder, so `v_cw` is the ODP process variance of the projected
//! loss ratio. `within_cv` corrects `r` for the company level estimated by
//! `xbar_c`, and `between_cv` the variance of the company means for the
//! within-company noise in them; a negative `between_cv^2` is set to 0
//! (with a warning). Only origins developed at least
//! [`IndustryPriorSettings::min_developed`] by the industry pattern enter
//! the loss ratios, since an immature origin's projection is mostly the
//! pattern's.
//!
//! No data ship with the library; the triangles are the user's (for
//! example Schedule P triangles read with the `cas-schedule-p` package in
//! Python).

use prospicio_core::Lag;

use crate::bootstrap::{Exclusions, prepare_selected};
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::tail::Tail;
use crate::triangle::{Segment, Triangle};

/// The standard deviation given to `beta_d` at an age where fewer than two
/// companies have a pattern: the normalised BF's default, vague.
const VAGUE_SD: f64 = 10.0;

/// The smallest standard deviation of a `beta_d` or cv the prior keeps: a
/// spread of 0 (identical companies) would fix the parameter.
const FLOOR: f64 = 0.01;

/// A normal prior on the development pattern of
/// [`BayesBfNormalised`](crate::bayes::BayesBfNormalised): at each age
/// after the first, `beta_d = log(p_d / p_1) ~ N(mean_d, sd_d^2)`, where
/// `p_d` is the share of the ultimate that falls at age `d`. It replaces
/// the model's `beta_d ~ N(0, development_sd^2)`. The ages must be the
/// fitted triangle's.
///
/// ```
/// use prospicio_reserving::PatternPrior;
///
/// // Shares 0.5, 0.3, 0.2 held with a standard deviation of 0.1 on the
/// // log ratios.
/// let prior = PatternPrior::from_shares(&[12, 24, 36], &[0.5, 0.3, 0.2], &[0.1, 0.1])?;
/// assert!((prior.mean[0] - (0.3_f64 / 0.5).ln()).abs() < 1e-12);
/// let p = prior.shares();
/// assert!((p[2] - 0.2).abs() < 1e-12);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PatternPrior {
    /// The development ages, first to last.
    pub ages: Vec<Lag>,
    /// The prior mean of `beta_d` at each age after the first.
    pub mean: Vec<f64>,
    /// The prior standard deviation of `beta_d` at each age after the
    /// first; finite and positive.
    pub sd: Vec<f64>,
}

impl PatternPrior {
    /// A prior from the means and standard deviations of `beta_2..beta_m`
    /// at `ages` (one fewer than the ages).
    pub fn new(ages: &[Lag], mean: &[f64], sd: &[f64]) -> Result<Self> {
        if ages.len() < 2 {
            return Err(Error::IndustryPrior(format!(
                "a pattern prior needs at least two ages, found {}",
                ages.len()
            )));
        }
        if ages.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::IndustryPrior(
                "the pattern prior's ages must increase".into(),
            ));
        }
        for (name, values) in [("mean", mean), ("sd", sd)] {
            if values.len() != ages.len() - 1 {
                return Err(Error::LengthMismatch {
                    column: format!("pattern prior {name}"),
                    expected: ages.len() - 1,
                    found: values.len(),
                });
            }
        }
        if let Some(&m) = mean.iter().find(|m| !m.is_finite()) {
            return Err(Error::InvalidSetting {
                name: "mean",
                value: m,
                expected: "a finite prior mean of each log share ratio",
            });
        }
        if let Some(&s) = sd.iter().find(|s| !(s.is_finite() && **s > 0.0)) {
            return Err(Error::InvalidSetting {
                name: "sd",
                value: s,
                expected: "a finite, positive prior standard deviation",
            });
        }
        Ok(Self {
            ages: ages.to_vec(),
            mean: mean.to_vec(),
            sd: sd.to_vec(),
        })
    }

    /// A prior centred on the incremental `shares` of the ultimate at
    /// `ages` (positive; they need not sum to 1), with `sd` on each log
    /// ratio after the first age.
    pub fn from_shares(ages: &[Lag], shares: &[f64], sd: &[f64]) -> Result<Self> {
        if shares.len() != ages.len() {
            return Err(Error::LengthMismatch {
                column: "pattern prior shares".into(),
                expected: ages.len(),
                found: shares.len(),
            });
        }
        if let Some(&s) = shares.iter().find(|s| !(s.is_finite() && **s > 0.0)) {
            return Err(Error::InvalidSetting {
                name: "shares",
                value: s,
                expected: "finite, positive incremental shares",
            });
        }
        let mean: Vec<f64> = shares[1..].iter().map(|s| (s / shares[0]).ln()).collect();
        Self::new(ages, &mean, sd)
    }

    /// The incremental shares at the prior mean, `exp(mean_d)` normalised
    /// to sum to 1 (with `beta_1 = 0`).
    pub fn shares(&self) -> Vec<f64> {
        let raw: Vec<f64> = std::iter::once(0.0)
            .chain(self.mean.iter().copied())
            .map(f64::exp)
            .collect();
        let total: f64 = raw.iter().sum();
        raw.iter().map(|r| r / total).collect()
    }

    /// The cumulative shares at the prior mean: the percent developed at
    /// each age, ending at 1.
    pub fn percent_developed(&self) -> Vec<f64> {
        cumulative(&self.shares())
    }

    /// The prior on the ages up to `age` (one of the prior's): the log
    /// ratios to the first age do not depend on the ages after it, so the
    /// means and standard deviations are the first ones.
    pub fn truncated(&self, age: Lag) -> Result<Self> {
        let k = self.position(age)?;
        Self::new(&self.ages[..=k], &self.mean[..k], &self.sd[..k])
    }

    fn position(&self, age: Lag) -> Result<usize> {
        let k = self.ages.iter().position(|&a| a == age).ok_or_else(|| {
            Error::IndustryPrior(format!(
                "age {age} months is not one of the pattern's ages {:?}",
                self.ages
            ))
        })?;
        if k == 0 {
            return Err(Error::IndustryPrior(format!(
                "a pattern prior needs at least two ages: {age} months is the first"
            )));
        }
        Ok(k)
    }
}

/// How [`IndustryPrior`] reads the companies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndustryPriorSettings {
    /// The smallest share of the ultimate developed, by the industry
    /// pattern, at an origin's latest age for its loss ratio to count: an
    /// immature origin's ultimate is mostly its projection.
    pub min_developed: f64,
    /// The number of companies below which the report warns that the
    /// between-company cv rests on few degrees of freedom.
    pub min_companies: usize,
    /// Whether the prior's `cv` is the within-company cv less the ODP
    /// process error estimated from each company's Pearson scale
    /// (`true`), or the within-company cv itself (`false`, the default).
    /// The subtraction recovers the expected ultimates' cv on data that
    /// follow the ODP model, but on the calibration panel's Schedule P
    /// triangles the Pearson scale (inflated by misfit and, on incurred,
    /// by case reserve movements that cancel in the ultimate) often
    /// explains more than the whole spread, and the cv falls to its
    /// floor; the default keeps the realised spread, which counts the
    /// process error twice with the model's own and so errs wide
    /// (`reserving-bayes.md`, decision 29).
    pub subtract_process: bool,
}

impl Default for IndustryPriorSettings {
    /// `min_developed` 0.5, `min_companies` 10, `subtract_process` false.
    fn default() -> Self {
        Self {
            min_developed: 0.5,
            min_companies: 10,
            subtract_process: false,
        }
    }
}

/// What [`IndustryPrior`] used and what it could not: the sample sizes,
/// the companies or origins left out and why, and warnings.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PriorReport {
    /// The number of companies supplied.
    pub n_companies: usize,
    /// The number whose loss ratios enter the prior.
    pub n_companies_used: usize,
    /// The number of origins whose loss ratios enter the prior, over every
    /// company.
    pub n_origins_used: usize,
    /// The number of origins of each company used (0 for a company left
    /// out), in the order supplied.
    pub origins_used: Vec<usize>,
    /// The number of companies with a pattern at each age after the first.
    pub companies_by_age: Vec<usize>,
    /// Each company or origin left out, with the reason.
    pub excluded: Vec<String>,
    /// Warnings: few companies, a cv set to its floor or to 0, a vague
    /// pattern at an age, a prior for development to an age before the
    /// last.
    pub warnings: Vec<String>,
}

/// An industry prior for the Bayesian Bornhuetter-Ferguson from other
/// companies' triangles (module docs).
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, IndustryPrior, Long, Month, Triangle};
///
/// // Three companies with the same pattern and different loss ratios.
/// let company = |scale: f64| -> Triangle {
///     let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
///     let paid = [50.0, 80.0, 100.0, 55.0, 88.0, 60.0].map(|v| v * scale);
///     Triangle::from_long(&Long {
///         keys: &[],
///         origin: &origin,
///         development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
///         values: &[("paid", &paid)],
///         origin_grain: Grain::Year,
///         development_grain: Grain::Year,
///         cumulative: true,
///     })
///     .unwrap()
/// };
/// let triangles = [company(1.0), company(1.2), company(0.9)];
/// let premiums = vec![vec![160.0, 160.0, 160.0]; 3];
/// let prior = IndustryPrior::from_triangles(&triangles, "paid", &premiums)?;
/// assert_eq!(prior.report.n_companies_used, 3);
/// assert!(prior.between_cv > 0.0);
/// assert!(!prior.report.warnings.is_empty()); // only three companies
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IndustryPrior {
    /// The mean over the companies of their mean ultimate loss ratio: the
    /// expected ultimate loss ratio of an origin of a company drawn from
    /// the industry.
    pub apriori: f64,
    /// The standard error of `apriori`, the sd of the company means over
    /// the square root of their number.
    pub apriori_se: f64,
    /// The cv of a company's level about `apriori`: the normalised BF's
    /// `common_cv`.
    pub between_cv: f64,
    /// The cv of the realised ultimate loss ratios of a company's origins
    /// about its level, process error included.
    pub within_cv: f64,
    /// The part of `within_cv` that ODP process error explains (each
    /// company's Pearson scale), which the normalised BF adds through its
    /// own scale.
    pub process_cv: f64,
    /// The normalised BF's `cv`, of an origin's expected ultimate about its
    /// company's level: `within_cv`, or with
    /// [`IndustryPriorSettings::subtract_process`]
    /// `sqrt(within_cv^2 - process_cv^2)`; floored at 0.01.
    pub cv: f64,
    /// The prior on the pattern: mean and sd of `beta_d` over the
    /// companies.
    pub pattern: PatternPrior,
    /// The percent developed at each age at the pattern prior's mean.
    pub percent_developed: Vec<f64>,
    /// The sd over the companies of their percent developed at each age.
    pub percent_developed_sd: Vec<f64>,
    /// The mean ultimate loss ratio of each company used, in the order
    /// supplied (companies left out omitted).
    pub company_loss_ratios: Vec<f64>,
    /// The factor from the last age of [`pattern`](Self::pattern) to the
    /// ultimate of the triangles: 1, or `1 / percent developed` for a prior
    /// cut at an earlier age by [`at_age`](Self::at_age).
    pub tail_factor: f64,
    /// Sample sizes, exclusions and warnings.
    pub report: PriorReport,
}

/// One company read for the prior.
struct Company {
    /// The company's chain ladder's percent developed at each age, if
    /// every factor was estimable.
    developed: Vec<f64>,
    /// `beta_d` of the company's own pattern at each age after the first,
    /// NaN where its share there is not positive.
    beta: Vec<f64>,
    /// `(origin index, latest position, ultimate loss ratio, its ODP
    /// process variance)` of each origin with a positive premium.
    origins: Vec<(usize, usize, f64, f64)>,
    /// The index of each origin left out for a premium that is not a
    /// positive number.
    no_premium: Vec<usize>,
}

impl IndustryPriorSettings {
    /// The prior from single-segment `triangles`, one per company, of
    /// loss `column` (cumulative or incremental), with each company's
    /// `premiums` in origin order. Every triangle must have the same
    /// development ages.
    pub fn from_triangles(
        &self,
        triangles: &[Triangle],
        column: &str,
        premiums: &[Vec<f64>],
    ) -> Result<IndustryPrior> {
        if triangles.len() != premiums.len() {
            return Err(Error::LengthMismatch {
                column: "premiums".into(),
                expected: triangles.len(),
                found: premiums.len(),
            });
        }
        let mut companies = Vec::with_capacity(triangles.len());
        for (c, (triangle, premium)) in triangles.iter().zip(premiums).enumerate() {
            let segment = triangle
                .segment(column)
                .map_err(|e| Error::IndustryPrior(format!("company {}: {e}", c + 1)))?;
            if premium.len() != segment.n_origins {
                return Err(Error::LengthMismatch {
                    column: format!("premiums of company {}", c + 1),
                    expected: segment.n_origins,
                    found: premium.len(),
                });
            }
            companies.push((format!("company {}", c + 1), segment, premium.clone()));
        }
        self.build(companies)
    }

    /// The prior from a triangle with one segment per company (a key such
    /// as the group code), loss `column`, and each origin's premium from
    /// the `exposure` column (its latest observed value, as for
    /// [`BornhuetterFerguson`](crate::BornhuetterFerguson)).
    pub fn from_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<IndustryPrior> {
        let losses = triangle.segments(column)?;
        let exposures = triangle.segments(exposure)?;
        let mut companies = Vec::with_capacity(losses.len());
        for ((segment, exposure_segment), label) in
            losses.into_iter().zip(&exposures).zip(triangle.index())
        {
            let premium = origin_exposure(&segment, exposure_segment, exposure)
                .map_err(|e| Error::IndustryPrior(format!("segment {label}: {e}")))?;
            companies.push((format!("segment {label}"), segment, premium));
        }
        self.build(companies)
    }

    fn validate(&self) -> Result<()> {
        if !(self.min_developed.is_finite() && (0.0..=1.0).contains(&self.min_developed)) {
            return Err(Error::InvalidSetting {
                name: "min_developed",
                value: self.min_developed,
                expected: "a share of the ultimate from 0 to 1",
            });
        }
        Ok(())
    }

    fn build(&self, companies: Vec<(String, Segment, Vec<f64>)>) -> Result<IndustryPrior> {
        self.validate()?;
        let n_companies = companies.len();
        if n_companies < 2 {
            return Err(Error::IndustryPrior(format!(
                "an industry prior needs at least two companies, found {n_companies}"
            )));
        }
        let ages = companies[0].1.ages.clone();
        if ages.len() < 2 {
            return Err(Error::IndustryPrior(
                "the triangles need at least two development ages".into(),
            ));
        }
        if let Some((name, segment, _)) = companies.iter().find(|c| c.1.ages != ages) {
            return Err(Error::IndustryPrior(format!(
                "{name} has the development ages {:?}, the first company {ages:?}: every \
                 triangle needs the same ages",
                segment.ages
            )));
        }
        let mut report = PriorReport {
            n_companies,
            origins_used: vec![0; n_companies],
            ..PriorReport::default()
        };
        let read: Vec<Option<Company>> = companies
            .iter()
            .map(
                |(name, segment, premium)| match read_company(segment, premium) {
                    Ok(company) => Some(company),
                    Err(e) => {
                        report.excluded.push(format!("{name}: {e}"));
                        None
                    }
                },
            )
            .collect();

        // The pattern: mean and sd of beta_d over the companies.
        let nd = ages.len();
        let (mut mean, mut sd) = (vec![0.0; nd - 1], vec![0.0; nd - 1]);
        report.companies_by_age = vec![0; nd - 1];
        for d in 0..nd - 1 {
            let values: Vec<f64> = read
                .iter()
                .flatten()
                .map(|c| c.beta[d])
                .filter(|b| b.is_finite())
                .collect();
            report.companies_by_age[d] = values.len();
            let age = ages[d + 1];
            match values.len() {
                0 => {
                    return Err(Error::IndustryPrior(format!(
                        "no company has a positive share of its ultimate at {age} months, so \
                         the pattern there cannot be estimated"
                    )));
                }
                1 => {
                    mean[d] = values[0];
                    sd[d] = VAGUE_SD;
                    report.warnings.push(format!(
                        "only one company has a pattern at {age} months: its log share ratio \
                         has the vague sd {VAGUE_SD}"
                    ));
                }
                _ => {
                    let (m, s) = mean_sd(&values);
                    mean[d] = m;
                    sd[d] = if s < FLOOR {
                        report.warnings.push(format!(
                            "the companies' log share ratios at {age} months differ by an sd \
                             of {s:.3e}: set to {FLOOR}"
                        ));
                        FLOOR
                    } else {
                        s
                    };
                }
            }
        }
        let pattern = PatternPrior::new(&ages, &mean, &sd)?;
        let percent_developed = pattern.percent_developed();
        let percent_developed_sd = (0..nd)
            .map(|d| {
                let values: Vec<f64> = read
                    .iter()
                    .flatten()
                    .map(|c| c.developed[d])
                    .filter(|p| p.is_finite())
                    .collect();
                if values.len() < 2 {
                    f64::NAN
                } else {
                    mean_sd(&values).1
                }
            })
            .collect();

        // The loss ratios: each company's origins developed enough.
        let mut used = Vec::new();
        for (c, (company, (name, segment, _))) in read.iter().zip(&companies).enumerate() {
            let Some(company) = company else { continue };
            for &w in &company.no_premium {
                report.excluded.push(format!(
                    "{name}, origin {}: its premium is not a positive number",
                    segment.origins[w]
                ));
            }
            let mut kept = Vec::new();
            for &(w, position, ratio, variance) in &company.origins {
                let origin = &segment.origins[w];
                if percent_developed[position] < self.min_developed {
                    continue;
                }
                if !(ratio.is_finite() && ratio >= 0.0 && variance.is_finite()) {
                    report.excluded.push(format!(
                        "{name}, origin {origin}: its ultimate loss ratio {ratio} is not a \
                         finite, non-negative number"
                    ));
                    continue;
                }
                kept.push((ratio, variance));
            }
            if kept.is_empty() {
                report.excluded.push(format!(
                    "{name}: no origin developed at least {} by the industry pattern",
                    self.min_developed
                ));
                continue;
            }
            report.origins_used[c] = kept.len();
            used.push(kept);
        }
        let k = used.len();
        if k < 2 {
            return Err(Error::IndustryPrior(format!(
                "{k} of {n_companies} companies have a loss ratio: at least two are needed \
                 (excluded: {})",
                report.excluded.join("; ")
            )));
        }
        report.n_companies_used = k;
        report.n_origins_used = used.iter().map(Vec::len).sum();
        if k < self.min_companies {
            report.warnings.push(format!(
                "only {k} companies: the between-company cv rests on {} degrees of freedom \
                 (its relative standard error is about {:.0}%); a prior from at least {} is \
                 more reliable",
                k - 1,
                100.0 / (2.0 * (k - 1) as f64).sqrt(),
                self.min_companies
            ));
        }

        let means: Vec<f64> = used
            .iter()
            .map(|o| o.iter().map(|x| x.0).sum::<f64>() / o.len() as f64)
            .collect();
        let (apriori, sd_means) = mean_sd(&means);
        if apriori <= 0.0 {
            return Err(Error::IndustryPrior(format!(
                "the companies' mean loss ratio is {apriori}, not positive"
            )));
        }
        let apriori_se = sd_means / (k as f64).sqrt();

        // Within: the pooled relative spread about each company's mean.
        let (mut squares, mut scale, mut process, mut n_sum, mut n_count) =
            (0.0, 0.0, 0.0, 0.0, 0usize);
        for (origins, &xbar) in used.iter().zip(&means) {
            let n = origins.len();
            if n < 2 {
                continue;
            }
            squares += origins.iter().map(|o| (o.0 - xbar).powi(2)).sum::<f64>();
            scale += (n - 1) as f64 * xbar * xbar;
            process += (1.0 - 1.0 / n as f64) * origins.iter().map(|o| o.1).sum::<f64>();
            n_sum += n as f64;
            n_count += 1;
        }
        if n_count == 0 || scale <= 0.0 {
            return Err(Error::IndustryPrior(
                "no company has two origins with a positive mean loss ratio, so the spread \
                 within a company cannot be estimated"
                    .into(),
            ));
        }
        let r = squares / scale;
        let nbar = n_sum / n_count as f64;
        let within2 = if r / nbar < 1.0 {
            r / (1.0 - r / nbar)
        } else {
            r
        };
        let process2 = if r > 0.0 {
            process / scale * within2 / r
        } else {
            0.0
        };
        let within_cv = within2.sqrt();
        let process_cv = process2.sqrt();
        let mut expected2 = if self.subtract_process {
            within2 - process2
        } else {
            within2
        };
        if expected2 < FLOOR * FLOOR {
            report.warnings.push(if self.subtract_process {
                format!(
                    "the origins' loss ratios spread no more than ODP process error explains \
                     (within cv {within_cv:.4}, process cv {process_cv:.4}): cv set to {FLOOR}"
                )
            } else {
                format!(
                    "the origins' loss ratios hardly spread within a company (within cv \
                     {within_cv:.4}): cv set to {FLOOR}"
                )
            });
            expected2 = FLOOR * FLOOR;
        }

        // Between: the company means' spread less the within noise in them.
        let noise = used
            .iter()
            .map(|o| 1.0 + within2 / o.len() as f64)
            .sum::<f64>()
            / k as f64;
        let q = (sd_means / apriori).powi(2);
        let between2 = (1.0 + q) / noise - 1.0;
        let between_cv = if between2 > 0.0 {
            between2.sqrt()
        } else {
            report.warnings.push(
                "the company means spread no more than the noise within them: between_cv set to \
                 0, and the normalised BF has no common multiplier"
                    .into(),
            );
            0.0
        };

        Ok(IndustryPrior {
            apriori,
            apriori_se,
            between_cv,
            within_cv,
            process_cv,
            cv: expected2.sqrt(),
            pattern,
            percent_developed,
            percent_developed_sd,
            company_loss_ratios: means,
            tail_factor: 1.0,
            report,
        })
    }
}

/// One company's pattern and loss ratios from its own chain ladder and
/// ODP scale.
fn read_company(segment: &Segment, premium: &[f64]) -> Result<Company> {
    let (odp, _) = prepare_selected(
        segment,
        &segment.ages,
        Tail::default(),
        None,
        Exclusions::default(),
    )?;
    let cl = &odp.chain_ladder;
    let phi = odp.scale;
    let developed: Vec<f64> = cl.cdf.iter().map(|f| 1.0 / f).collect();
    let mut beta = vec![f64::NAN; segment.n_dev - 1];
    let first = developed[0];
    if first.is_finite() && first > 0.0 {
        for (d, b) in beta.iter_mut().enumerate() {
            let share = developed[d + 1] - developed[d];
            if share.is_finite() && share > 0.0 {
                *b = (share / first).ln();
            }
        }
    }
    let origins = (0..segment.n_origins)
        .filter(|&w| premium[w].is_finite() && premium[w] > 0.0)
        .map(|w| {
            let position = cl.latest_position[w];
            let ratio = cl.ultimate[w] / premium[w];
            let variance = phi * ratio / (developed[position] * premium[w]);
            (w, position, ratio, variance)
        })
        .collect();
    let no_premium = (0..segment.n_origins)
        .filter(|&w| !(premium[w].is_finite() && premium[w] > 0.0))
        .collect();
    Ok(Company {
        developed,
        beta,
        origins,
        no_premium,
    })
}

/// The mean and the sample standard deviation (`n - 1`) of `values`.
fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, f64::NAN);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn cumulative(shares: &[f64]) -> Vec<f64> {
    shares
        .iter()
        .scan(0.0, |acc, s| {
            *acc += s;
            Some(*acc)
        })
        .collect()
}

impl IndustryPrior {
    /// The prior from single-segment `triangles`, one per company, of
    /// loss `column` with each company's `premiums` in origin order, under
    /// the default [`IndustryPriorSettings`].
    pub fn from_triangles(
        triangles: &[Triangle],
        column: &str,
        premiums: &[Vec<f64>],
    ) -> Result<Self> {
        IndustryPriorSettings::default().from_triangles(triangles, column, premiums)
    }

    /// The prior from a triangle with one segment per company, under the
    /// default [`IndustryPriorSettings`]
    /// ([`IndustryPriorSettings::from_segments`]).
    pub fn from_segments(triangle: &Triangle, column: &str, exposure: &str) -> Result<Self> {
        IndustryPriorSettings::default().from_segments(triangle, column, exposure)
    }

    /// The prior for a triangle whose last age is `age`, one of the
    /// pattern's: the normalised BF's ultimate is then the development to
    /// `age`, so `apriori` becomes the loss ratio developed to it (times
    /// the industry percent developed there), the pattern prior keeps the
    /// ages up to it, and [`tail_factor`](Self::tail_factor) is the factor
    /// from it to ultimate, which the user applies outside the model. The
    /// cvs are kept (the spread of the percent developed is not added).
    pub fn at_age(&self, age: Lag) -> Result<Self> {
        let pattern = self.pattern.truncated(age)?;
        let k = pattern.ages.len() - 1;
        let developed = self.percent_developed[k];
        let mut report = self.report.clone();
        report.warnings.push(format!(
            "a prior for development to {age} months: {:.1}% of the ultimate develops after it, \
             outside the model; multiply the fit's ultimates by the tail factor {:.4}",
            100.0 * (1.0 - developed),
            self.tail_factor / developed
        ));
        Ok(Self {
            apriori: self.apriori * developed,
            apriori_se: self.apriori_se * developed,
            percent_developed: self.percent_developed[..=k]
                .iter()
                .map(|p| p / developed)
                .collect(),
            percent_developed_sd: self.percent_developed_sd[..=k]
                .iter()
                .map(|s| s / developed)
                .collect(),
            pattern,
            tail_factor: self.tail_factor / developed,
            report,
            ..self.clone()
        })
    }

    /// The common multiplier's cv for the normalised BF: `between_cv`, or
    /// none when it is 0.
    pub fn common_cv(&self) -> Option<f64> {
        (self.between_cv > 0.0).then_some(self.between_cv)
    }
}

#[cfg(feature = "bayes")]
mod bf {
    use super::IndustryPrior;
    use crate::bayes::{BayesBfNormalised, BayesBfNormalisedFit, BayesOneYear, BayesOneYearFit};
    use crate::error::Result;
    use crate::triangle::Triangle;

    impl IndustryPrior {
        /// `base` with this prior's `apriori`, `cv` and `common_cv`
        /// ([`common_cv`](Self::common_cv)); its other settings are kept.
        /// The pattern prior is not a setting of the model: fit with
        /// [`fit`](Self::fit) or
        /// [`BayesBfNormalised::fit_with_pattern_prior`].
        pub fn model(&self, base: &BayesBfNormalised) -> BayesBfNormalised {
            BayesBfNormalised {
                apriori: self.apriori,
                cv: self.cv,
                common_cv: self.common_cv(),
                ..base.clone()
            }
        }

        /// [`model`](Self::model) on the default [`BayesBfNormalised`].
        pub fn to_bf_normalised_settings(&self) -> BayesBfNormalised {
            self.model(&BayesBfNormalised::default())
        }

        /// [`model`](Self::model) of `base` fitted to loss `column` of
        /// `triangle` with `exposure`, under this prior's pattern prior,
        /// whose ages must be the triangle's (use [`at_age`](Self::at_age)
        /// for a younger triangle).
        pub fn fit(
            &self,
            base: &BayesBfNormalised,
            triangle: &Triangle,
            column: &str,
            exposure: &str,
        ) -> Result<BayesBfNormalisedFit> {
            self.model(base)
                .fit_with_pattern_prior(triangle, column, exposure, &self.pattern)
        }

        /// The one-year view of [`fit`](Self::fit).
        pub fn one_year(
            &self,
            base: &BayesBfNormalised,
            triangle: &Triangle,
            column: &str,
            exposure: &str,
            settings: &BayesOneYear,
        ) -> Result<BayesOneYearFit<BayesBfNormalisedFit>> {
            self.model(base).one_year_with_pattern_prior(
                triangle,
                column,
                exposure,
                settings,
                &self.pattern,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::Long;
    use crate::{DevelopmentColumn, Grain};
    use prospicio_core::Month;

    /// A company's cumulative triangle (`n` origins) with the given
    /// incremental `shares` and each origin's `ultimate`.
    fn company(shares: &[f64], ultimate: &[f64]) -> Triangle {
        let n = ultimate.len();
        let (mut origin, mut ages, mut values) = (vec![], vec![], vec![]);
        for (w, u) in ultimate.iter().enumerate() {
            let mut cum = 0.0;
            for (d, s) in shares.iter().enumerate().take(n - w) {
                cum += s * u;
                origin.push(Month::january(2010 + w as i32));
                ages.push(12 * (d as Lag + 1));
                values.push(cum);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn the_pattern_prior_round_trips_its_shares() {
        let p =
            PatternPrior::from_shares(&[12, 24, 36, 48], &[4.0, 3.0, 2.0, 1.0], &[0.1; 3]).unwrap();
        let shares = p.shares();
        for (s, e) in shares.iter().zip([0.4, 0.3, 0.2, 0.1]) {
            assert!((s - e).abs() < 1e-12);
        }
        let t = p.truncated(36).unwrap();
        assert_eq!(t.ages, vec![12, 24, 36]);
        let ts = t.shares();
        assert!((ts[0] - 4.0 / 9.0).abs() < 1e-12);
        assert!(p.truncated(12).is_err());
        assert!(p.truncated(30).is_err());
        assert!(PatternPrior::new(&[12, 24], &[0.0], &[0.0]).is_err());
        assert!(PatternPrior::new(&[24, 12], &[0.0], &[1.0]).is_err());
        assert!(PatternPrior::new(&[12, 24], &[0.0, 1.0], &[1.0]).is_err());
    }

    /// Exact cells: each company's chain ladder recovers its pattern and
    /// loss ratios, so the estimators are the sample moments.
    #[test]
    fn exact_companies_give_their_sample_moments() {
        let shares = [0.5, 0.3, 0.2];
        let levels = [0.6, 0.8, 0.7, 0.9];
        let wobble = [1.1, 0.9, 1.0];
        let triangles: Vec<Triangle> = levels
            .iter()
            .map(|l| company(&shares, &wobble.map(|w| 100.0 * l * w)))
            .collect();
        let premiums = vec![vec![100.0; 3]; 4];
        let settings = IndustryPriorSettings {
            min_developed: 0.0,
            ..IndustryPriorSettings::default()
        };
        let prior = settings
            .from_triangles(&triangles, "paid", &premiums)
            .unwrap();
        // Same pattern everywhere: the log ratios have sd 0, floored.
        for (m, e) in prior.pattern.mean.iter().zip([0.6_f64.ln(), 0.4_f64.ln()]) {
            assert!((m - e).abs() < 1e-9, "{m} {e}");
        }
        assert!(prior.pattern.sd.iter().all(|&s| s == FLOOR));
        assert!((prior.apriori - 0.75).abs() < 1e-9);
        for (x, l) in prior.company_loss_ratios.iter().zip(levels) {
            assert!((x - l).abs() < 1e-9);
        }
        // The cells are exact, so the Pearson scale is 0: no process.
        assert!(prior.process_cv.abs() < 1e-6);
        // r = sum of (0.1 l)^2 * 2 / (2 l^2) per company = 0.01.
        let r: f64 = 0.01;
        let within = r / (1.0 - r / 3.0);
        assert!((prior.within_cv - within.sqrt()).abs() < 1e-9);
        assert!((prior.cv - prior.within_cv).abs() < 1e-9);
        let q = (0.0166666666666667_f64).sqrt() / 0.75; // sd of levels / mean
        let q = q * q;
        let between = (1.0 + q) / (1.0 + within / 3.0) - 1.0;
        assert!((prior.between_cv - between.sqrt()).abs() < 1e-9);
        assert_eq!(prior.report.n_origins_used, 12);
        assert!(prior.report.warnings.iter().any(|w| w.contains("only 4")));
        // Development to 24 months: the share developed is 0.8.
        let cut = prior.at_age(24).unwrap();
        assert!((cut.apriori - 0.75 * 0.8).abs() < 1e-9);
        assert!((cut.tail_factor - 1.25).abs() < 1e-9);
        assert_eq!(cut.pattern.ages, vec![12, 24]);
        assert!((cut.percent_developed[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn immature_origins_are_left_out_and_errors_name_the_cause() {
        let shares = [0.5, 0.3, 0.2];
        let triangles: Vec<Triangle> = [0.6, 0.8]
            .iter()
            .map(|l| company(&shares, &[100.0 * l, 110.0 * l, 90.0 * l]))
            .collect();
        let premiums = vec![vec![100.0, 110.0, 90.0]; 2];
        // Default min_developed 0.5: the newest origin (0.5 developed) is
        // kept, as the comparison is "at least".
        let prior = IndustryPrior::from_triangles(&triangles, "paid", &premiums).unwrap();
        assert_eq!(prior.report.n_origins_used, 6);
        let strict = IndustryPriorSettings {
            min_developed: 0.6,
            ..IndustryPriorSettings::default()
        };
        let prior = strict
            .from_triangles(&triangles, "paid", &premiums)
            .unwrap();
        assert_eq!(prior.report.origins_used, vec![2, 2]);
        // Errors: one company, mismatched premiums, different ages.
        assert!(IndustryPrior::from_triangles(&triangles[..1], "paid", &premiums[..1]).is_err());
        assert!(IndustryPrior::from_triangles(&triangles, "paid", &premiums[..1]).is_err());
        let short = company(&[0.6, 0.4], &[60.0, 70.0]);
        let err = IndustryPrior::from_triangles(
            &[triangles[0].clone(), short],
            "paid",
            &[vec![100.0; 3], vec![100.0; 2]],
        )
        .unwrap_err();
        assert!(err.to_string().contains("same ages"), "{err}");
        let bad = IndustryPriorSettings {
            min_developed: 1.5,
            ..IndustryPriorSettings::default()
        };
        assert!(bad.from_triangles(&triangles, "paid", &premiums).is_err());
    }

    /// An origin without a positive premium is left out and reported under
    /// its own label, not the next origin's.
    #[test]
    fn an_origin_without_premium_is_reported_by_its_own_label() {
        let shares = [0.5, 0.3, 0.2];
        let triangles: Vec<Triangle> = [0.6, 0.8]
            .iter()
            .map(|l| company(&shares, &[100.0 * l, 110.0 * l, 90.0 * l]))
            .collect();
        let premiums = vec![vec![0.0, 110.0, 90.0], vec![100.0, 110.0, 90.0]];
        let prior = IndustryPrior::from_triangles(&triangles, "paid", &premiums).unwrap();
        assert_eq!(prior.report.origins_used, vec![2, 3]);

        let excluded = &prior.report.excluded;
        assert_eq!(excluded.len(), 1, "{excluded:?}");
        assert!(
            excluded[0].starts_with("company 1, origin 2010:") && excluded[0].contains("premium"),
            "{excluded:?}"
        );
        // The kept origins' loss ratios are 0.6 each: the label shift did
        // not move a value.
        assert!((prior.company_loss_ratios[0] - 0.6).abs() < 1e-9);
    }
}
