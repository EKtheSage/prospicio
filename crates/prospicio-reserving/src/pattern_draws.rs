//! A development pattern per simulation from a stochastic reserving fit,
//! for the cash-flow engine's paid patterns (`reserving-v02.md` decision
//! 28, issue #600).
//!
//! A deal's investor return depends on when its losses are paid, and a
//! slow year and a fast year fund the collateral differently. A fit that
//! draws its development per simulation (the bootstraps' pseudo factors,
//! the Bayesian models' posterior draws) knows a pattern for each
//! simulation; [`DevelopmentPatterns`] keeps them, row `i` the same
//! simulation as row `i` of the fit's `reserves` and of every vector
//! [`UltimateDraws`](crate::UltimateDraws) gives, so a simulation's loss
//! ratio and its pattern pair up in `CashflowTerms::run_with_patterns`
//! (`prospicio-pricing`).
//!
//! # What a simulation's pattern is
//!
//! The cumulative share of the ultimate that an origin with nothing yet
//! observed would have reached at each development age, under the
//! development the simulation drew: the parameter uncertainty in the
//! pattern, the same for every origin of the segment, without process
//! error. A fit with a tail reaches 1 one development period after the
//! oldest age; the share there is `1 / tail`. Drawing it takes no random
//! number, so a fit's reserves are what they were without it.
//!
//! [`monthly`](DevelopmentPatterns::monthly) turns the patterns into the
//! engine's monthly `paid`: the share at the end of each month from
//! inception, read between the ages on the origin's own basis or converted
//! to an underwriting basis.
//!
//! ```
//! use prospicio_reserving::{
//!     DevelopmentColumn, Grain, Long, Monotone, MonthlyBasis, Month, OdpBootstrap, Triangle,
//! };
//!
//! let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
//! let tri = Triangle::from_long(&Long {
//!     keys: &[],
//!     origin: &origin,
//!     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
//!     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
//!     origin_grain: Grain::Year,
//!     development_grain: Grain::Year,
//!     cumulative: true,
//! })?;
//! let fit = OdpBootstrap { n_sims: 200, seed: 1, ..Default::default() }.fit(&tri, "paid")?;
//! let segment = fit.patterns.segment(&[])?;
//! assert_eq!(segment.ages(), [12, 24, 36]);
//! // Simulation 7's share at 36 months is 1: no tail.
//! assert_eq!(segment.draw(7)[2], 1.0);
//! // Its share by the end of each of 48 months, on the accident-year basis.
//! let monthly = fit.patterns.monthly(
//!     &[],
//!     &MonthlyBasis::default(),
//!     48,
//!     Monotone::RunningMax,
//! )?;
//! assert_eq!(monthly.len(), 200);
//! assert_eq!(monthly[7].len(), 48);
//! assert_eq!(monthly[7][47], 1.0);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use prospicio_core::{Grain, Lag, Period};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution};

use crate::error::{Error, Result};
use crate::monthly_reading::AnnualReading;
use crate::pattern::{PatternKind, SelectedPattern, UnderwritingBasis};
use crate::triangle::Label;

/// Each segment's development pattern in every simulation of a fit (see
/// the [module docs](self)): row `i` is simulation `i` of the fit's
/// `reserves`.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentPatterns {
    n_sims: usize,
    segments: Vec<SegmentPatterns>,
}

/// One segment's patterns: the cumulative share of the ultimate at each of
/// [`ages`](Self::ages) in every simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentPatterns {
    label: Vec<String>,
    origin_grain: Grain,
    ages: Vec<Lag>,
    /// Row-major over simulation × age.
    draws: Vec<f64>,
}

/// The basis [`DevelopmentPatterns::monthly`] puts a pattern on.
#[derive(Debug, Clone, PartialEq)]
pub enum MonthlyBasis {
    /// The origin's own basis, a month from inception per month of age: an
    /// accident-year pattern for a deal on losses occurring in that year.
    /// A pattern of origin periods longer than a month is read at every
    /// month as `reading` says (`reserving-v02.md` decision 25).
    Origin(AnnualReading),
    /// An underwriting basis
    /// ([`SelectedPattern::to_underwriting_basis`], decision 21): the
    /// pattern of a year of policies written and earned as the basis says,
    /// read between the ages by the basis's own reading.
    Underwriting(UnderwritingBasis),
}

impl Default for MonthlyBasis {
    /// The origin's own basis with the default reading.
    fn default() -> Self {
        Self::Origin(AnnualReading::default())
    }
}

/// What [`DevelopmentPatterns::monthly`] does with a simulation whose share
/// falls from one age to the next (a pseudo factor below 1), or is not
/// positive (a negative factor, which Mack's bootstrap can draw from its
/// normal factors), neither of which the cash-flow engine takes
/// (`reserving-v02.md` decision 28 item 5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Monotone {
    /// An error naming how many simulations fall, or are not positive, and
    /// the first.
    #[default]
    Refuse,
    /// Each share held at the largest before it, capped at 1: the money is
    /// paid when first reached and none is returned. A share that is not
    /// positive (or not finite) takes the next positive share after it, so
    /// the simulation pays that share from its first age.
    RunningMax,
}

impl DevelopmentPatterns {
    /// Patterns from each segment's draws: `segments` holds each segment's
    /// label, origin grain, ages and its patterns row-major over simulation
    /// × age.
    pub(crate) fn new(n_sims: usize, segments: Vec<SegmentPatterns>) -> Self {
        Self { n_sims, segments }
    }

    /// The number of simulations, the fit's.
    pub fn n_sims(&self) -> usize {
        self.n_sims
    }

    /// Every segment's patterns, in the order of the fit's segments.
    pub fn segments(&self) -> &[SegmentPatterns] {
        &self.segments
    }

    /// The patterns of the segment labelled `label`, one part per key of
    /// the triangle. A fit of one segment also answers to `&[]` (a segment
    /// picked from several keeps its label), and one with no label to any
    /// label.
    pub fn segment(&self, label: &[&str]) -> Result<&SegmentPatterns> {
        if let [only] = self.segments.as_slice()
            && (only.label.is_empty() || label.is_empty())
        {
            return Ok(only);
        }
        self.segments
            .iter()
            .find(|s| s.label.iter().map(String::as_str).eq(label.iter().copied()))
            .ok_or_else(|| {
                Error::PatternDraws(format!(
                    "no segment is labelled [{}]; the segments are {}",
                    label.join(", "),
                    self.segments
                        .iter()
                        .map(|s| format!("[{}]", s.label.join(", ")))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }

    /// The segment's pattern in each simulation, in simulation order: the
    /// share at each of its [`ages`](SegmentPatterns::ages).
    pub fn pattern_draws(&self, label: &[&str]) -> Result<Vec<Vec<f64>>> {
        let segment = self.segment(label)?;
        Ok((0..self.n_sims).map(|i| segment.draw(i).to_vec()).collect())
    }

    /// The segment's pattern in each simulation as the cash-flow engine's
    /// monthly `paid`, in simulation order: `months` values, value `t` the
    /// share paid by the end of month `t + 1` from inception on `basis`,
    /// held at 1 once reached. The result goes to `PatternDraws::new` in
    /// `prospicio-pricing`, whose terms' `paid` must have `months` values
    /// ending at 1.
    ///
    /// Errors ([`Error::PatternDraws`]): an unknown segment; under
    /// [`Monotone::Refuse`], a simulation whose share falls; `months` short
    /// of the month at which a converted pattern reaches 1; and the
    /// conversion's own errors ([`Error::Pattern`]).
    pub fn monthly(
        &self,
        label: &[&str],
        basis: &MonthlyBasis,
        months: usize,
        monotone: Monotone,
    ) -> Result<Vec<Vec<f64>>> {
        let segment = self.segment(label)?;
        if monotone == Monotone::Refuse {
            segment.check_monotone()?;
        }
        (0..self.n_sims)
            .map(|i| {
                let mut shares = segment.draw(i).to_vec();
                if monotone == Monotone::RunningMax {
                    running_max(&mut shares);
                }
                segment.to_monthly(&shares, basis, months)
            })
            .collect()
    }

    /// One fit's patterns from each segment's own, fitted alone (a
    /// Bayesian `fit_segments`): segment `s` of the result is the only
    /// segment of `parts[s]`, labelled `labels[s]`.
    #[cfg(feature = "bayes")]
    pub(crate) fn joined(labels: &[Label], parts: Vec<DevelopmentPatterns>) -> Self {
        let n_sims = parts.first().map_or(0, |p| p.n_sims);
        let segments = labels
            .iter()
            .zip(parts)
            .flat_map(|(label, part)| {
                part.segments.into_iter().map(move |mut s| {
                    s.label = label.parts().to_vec();
                    s
                })
            })
            .collect();
        Self { n_sims, segments }
    }

    /// The patterns of the segment labelled `label` alone (a fit's
    /// `segment`); none if no segment is.
    pub(crate) fn only(&self, label: &Label) -> Self {
        Self {
            n_sims: self.n_sims,
            segments: self
                .segments
                .iter()
                .filter(|s| s.label == label.parts())
                .cloned()
                .collect(),
        }
    }

    /// Simulation `i` of segment `s` becomes the one at `index[i]`: the
    /// patterns follow a reorder of the fit's reserves.
    pub(crate) fn permute(&mut self, s: usize, index: &[usize]) {
        let segment = &mut self.segments[s];
        let n = segment.ages.len();
        segment.draws = index
            .iter()
            .flat_map(|&j| segment.draws[j * n..(j + 1) * n].iter().copied())
            .collect();
    }
}

impl SegmentPatterns {
    /// A segment's patterns: `draws` row-major over simulation × `ages`.
    pub(crate) fn new(label: &Label, origin_grain: Grain, ages: Vec<Lag>, draws: Vec<f64>) -> Self {
        Self {
            label: label.parts().to_vec(),
            origin_grain,
            ages,
            draws,
        }
    }

    /// The segment's label, one part per key of the triangle; empty for a
    /// fit of one segment.
    pub fn label(&self) -> &[String] {
        &self.label
    }

    /// The grain of the segment's origin periods.
    pub fn origin_grain(&self) -> Grain {
        self.origin_grain
    }

    /// The development ages in months: the triangle's, and with a tail one
    /// development period past the oldest, where the share is 1.
    pub fn ages(&self) -> &[Lag] {
        &self.ages
    }

    /// The share at each age in simulation `i`.
    ///
    /// # Panics
    ///
    /// If `i` is not below the number of simulations.
    pub fn draw(&self, i: usize) -> &[f64] {
        let n = self.ages.len();
        &self.draws[i * n..(i + 1) * n]
    }

    /// The mean share at each age over the simulations.
    pub fn mean(&self) -> Vec<f64> {
        let n = self.ages.len();
        let n_sims = self.draws.len() / n;
        let mut mean = vec![0.0; n];
        for row in self.draws.chunks_exact(n) {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += v;
            }
        }
        mean.iter_mut().for_each(|m| *m /= n_sims as f64);
        mean
    }

    fn check_monotone(&self) -> Result<()> {
        let n = self.ages.len();
        let invalid = |x: &f64| !(x.is_finite() && *x > 0.0);
        let bad: Vec<usize> = self
            .draws
            .chunks_exact(n)
            .enumerate()
            .filter(|(_, row)| row.iter().any(invalid))
            .map(|(i, _)| i)
            .collect();
        if let Some(&first) = bad.first() {
            let row = self.draw(first);
            let k = row.iter().position(invalid).unwrap_or(0);
            return Err(Error::PatternDraws(format!(
                "the share is not positive in {} of {} simulations (the first, simulation \
                 {first}, {} at {} months, from a negative factor), which no paid pattern can \
                 be: pass Monotone::RunningMax to give it the next positive share",
                bad.len(),
                self.draws.len() / n,
                row[k],
                self.ages[k],
            )));
        }
        let falling: Vec<usize> = self
            .draws
            .chunks_exact(n)
            .enumerate()
            .filter(|(_, row)| row.windows(2).any(|w| w[1] < w[0]))
            .map(|(i, _)| i)
            .collect();
        match falling.first() {
            None => Ok(()),
            Some(&first) => {
                let row = self.draw(first);
                let k = row.windows(2).position(|w| w[1] < w[0]).unwrap_or(0) + 1;
                Err(Error::PatternDraws(format!(
                    "the share falls in {} of {} simulations (the first, simulation {first}, \
                     from {} at {} months to {} at {}), which the cash-flow engine refuses: \
                     pass Monotone::RunningMax to pay each share when first reached",
                    falling.len(),
                    self.draws.len() / n,
                    row[k - 1],
                    self.ages[k - 1],
                    row[k],
                    self.ages[k],
                )))
            }
        }
    }

    /// `shares` at this segment's ages as `months` monthly values on
    /// `basis`.
    fn to_monthly(&self, shares: &[f64], basis: &MonthlyBasis, months: usize) -> Result<Vec<f64>> {
        let pattern = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            self.ages
                .iter()
                .copied()
                .zip(shares.iter().copied())
                .collect(),
        );
        let converted = match basis {
            MonthlyBasis::Origin(reading) => {
                pattern.monthly_on_own_basis(self.origin_grain, *reading)?
            }
            MonthlyBasis::Underwriting(basis) => {
                pattern.monthly_on_underwriting_basis(self.origin_grain, basis)?
            }
        };
        let end = converted.values.last().map_or(0, |v| v.0) as usize;
        if end > months {
            return Err(Error::PatternDraws(format!(
                "the pattern reaches 1 only at {end} months, after the {months} months asked \
                 for: give the terms at least {end} months"
            )));
        }
        let level = converted.values.last().map_or(1.0, |v| v.1);
        let mut out = vec![0.0; months];
        let mut held = 0.0f64;
        let mut values = converted.values.iter().peekable();
        for (t, value) in out.iter_mut().enumerate() {
            let age = t as Lag + 1;
            while let Some(&&(a, v)) = values.peek() {
                if a > age {
                    break;
                }
                // Rounding in a reading can fall by a hair or pass the
                // level by one; the engine refuses any fall and an end
                // other than the terms', so the share is held and capped.
                held = held.max(v).min(level);
                values.next();
            }
            *value = held;
        }
        Ok(out)
    }
}

/// The pattern columns a lifetime simulation writes after its reserves:
/// each segment's label, origin grain and pattern ages, in segment order.
/// The simulation's row is the reserves (`n_reserves` columns) then each
/// segment's shares at its ages in turn; [`split`](Self::split) takes them
/// apart, so the reserves are the simulation's as they were without the
/// patterns.
pub(crate) struct PatternLayout {
    n_reserves: usize,
    segments: Vec<(Label, Grain, Vec<Lag>)>,
}

impl PatternLayout {
    pub(crate) fn new(n_reserves: usize, segments: Vec<(Label, Grain, Vec<Lag>)>) -> Self {
        Self {
            n_reserves,
            segments,
        }
    }

    /// The pattern columns' keys, for `n_dims` dimensions: the segment's
    /// label (padded to the key dimensions) and `"pattern <age>"`, which no
    /// reserve's key is.
    pub(crate) fn keys(&self, n_dims: usize) -> Vec<ComponentKey> {
        self.segments
            .iter()
            .flat_map(|(label, _, ages)| {
                ages.iter().map(move |age| {
                    let mut key: ComponentKey =
                        label.parts().iter().map(|p| p.as_str().into()).collect();
                    key.resize(n_dims - 1, KeyValue::Text(String::new()));
                    key.push(KeyValue::Text(format!("pattern {age}")));
                    key
                })
            })
            .collect()
    }

    /// Where segment `s`'s shares start in a row.
    pub(crate) fn offset(&self, s: usize) -> usize {
        self.n_reserves
            + self.segments[..s]
                .iter()
                .map(|(_, _, ages)| ages.len())
                .sum::<usize>()
    }

    /// The number of shares segment `s` writes.
    pub(crate) fn width(&self, s: usize) -> usize {
        self.segments[s].2.len()
    }

    /// The reserves (the first `n_reserves` columns of `wide`, with its
    /// dimensions, keys and provenance) and the patterns.
    pub(crate) fn split(
        self,
        wide: PredictiveDistribution,
    ) -> Result<(PredictiveDistribution, DevelopmentPatterns)> {
        let n = wide.n_components();
        let n_sims = wide.n_sims();
        let rows = wide.draw_matrix();
        let reserves = rows
            .chunks_exact(n)
            .flat_map(|row| row[..self.n_reserves].iter().copied())
            .collect();
        let mut segments = Vec::with_capacity(self.segments.len());
        let mut start = self.n_reserves;
        for (label, grain, ages) in self.segments {
            let end = start + ages.len();
            let draws = rows
                .chunks_exact(n)
                .flat_map(|row| row[start..end].iter().copied())
                .collect();
            segments.push(SegmentPatterns::new(&label, grain, ages, draws));
            start = end;
        }
        let reserves = PredictiveDistribution::from_draws(
            wide.dims().to_vec(),
            wide.components()[..self.n_reserves].to_vec(),
            reserves,
            wide.provenance().clone(),
        )?;
        Ok((reserves, DevelopmentPatterns::new(n_sims, segments)))
    }
}

/// A Bayesian model's posterior predictive reserves of one segment with
/// its development pattern in each simulation: `simulate(theta, rng,
/// reserves, increments)` fills each origin's reserve as before and
/// `increments[d]`, without drawing, with a number proportional to the
/// expected increment at age `d` of the posterior draw `theta`; the
/// pattern is their running sum over the total (decision 28 item 1). The
/// reserves are what `posterior.predictive` gives without the pattern.
#[cfg(feature = "bayes")]
pub(crate) fn predictive_with_patterns<F>(
    posterior: &prospicio_bayes::nuts::PosteriorDraws,
    segment: &crate::triangle::Segment,
    n_sims: usize,
    seed: u64,
    provenance: prospicio_prob::Provenance,
    simulate: F,
) -> Result<(PredictiveDistribution, DevelopmentPatterns)>
where
    F: Fn(&[f64], &mut prospicio_core::StreamRng, &mut [f64], &mut [f64]) + Sync,
{
    predictive_with(
        posterior,
        segment,
        n_sims,
        seed,
        provenance,
        |theta, rng, reserves, shares| {
            simulate(theta, rng, reserves, shares);
            let mut cumulative = Vec::with_capacity(shares.len());
            push_cumulative(shares.iter().copied(), &mut cumulative);
            shares.copy_from_slice(&cumulative);
        },
    )
}

/// As [`predictive_with_patterns`], with `simulate(theta, rng, values,
/// shares)` writing the cumulative `shares` of the posterior draw `theta`
/// at the segment's ages itself, without drawing (Meyers' models, whose
/// development effects are on the cumulative values).
#[cfg(feature = "bayes")]
pub(crate) fn predictive_with_shares<F>(
    posterior: &prospicio_bayes::nuts::PosteriorDraws,
    segment: &crate::triangle::Segment,
    n_sims: usize,
    seed: u64,
    provenance: prospicio_prob::Provenance,
    simulate: F,
) -> Result<(PredictiveDistribution, DevelopmentPatterns)>
where
    F: Fn(&[f64], &mut prospicio_core::StreamRng, &mut [f64], &mut [f64]) + Sync,
{
    predictive_with(posterior, segment, n_sims, seed, provenance, simulate)
}

/// The simulations of one segment's values by origin with its pattern
/// columns after them, taken apart: `simulate(theta, rng, values, shares)`
/// fills both.
#[cfg(feature = "bayes")]
fn predictive_with<F>(
    posterior: &prospicio_bayes::nuts::PosteriorDraws,
    segment: &crate::triangle::Segment,
    n_sims: usize,
    seed: u64,
    provenance: prospicio_prob::Provenance,
    simulate: F,
) -> Result<(PredictiveDistribution, DevelopmentPatterns)>
where
    F: Fn(&[f64], &mut prospicio_core::StreamRng, &mut [f64], &mut [f64]) + Sync,
{
    predictive_keyed_with_shares(
        posterior,
        "origin",
        segment.origins.iter().map(|&o| vec![o.into()]).collect(),
        origin_grain(&segment.origins),
        segment.ages.clone(),
        n_sims,
        seed,
        provenance,
        simulate,
    )
}

/// As [`predictive_with_shares`], with the simulated values keyed by
/// `components` in the one dimension `dim` instead of by a segment's
/// origins, and the pattern at `ages` of origins of grain `grain`
/// (`Compartmental`, whose values are the future cells of its square and
/// whose pattern runs over the square's ages).
#[cfg(feature = "bayes")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn predictive_keyed_with_shares<F>(
    posterior: &prospicio_bayes::nuts::PosteriorDraws,
    dim: &str,
    mut components: Vec<ComponentKey>,
    grain: Grain,
    ages: Vec<Lag>,
    n_sims: usize,
    seed: u64,
    provenance: prospicio_prob::Provenance,
    simulate: F,
) -> Result<(PredictiveDistribution, DevelopmentPatterns)>
where
    F: Fn(&[f64], &mut prospicio_core::StreamRng, &mut [f64], &mut [f64]) + Sync,
{
    let layout = PatternLayout::new(components.len(), vec![(Label::default(), grain, ages)]);
    components.extend(layout.keys(1));
    let (offset, width) = (layout.offset(0), layout.width(0));
    let wide = posterior.predictive(
        vec![dim.into()],
        components,
        n_sims,
        seed,
        provenance,
        |theta, rng, row| {
            let (values, shares) = row.split_at_mut(offset);
            simulate(theta, rng, values, &mut shares[..width]);
        },
    )?;
    layout.split(wide)
}

/// The cumulative shares at the triangle's ages from one simulation's
/// age-to-age `factors` (one per link) and its `tail` factor to ultimate,
/// pushed onto `out`: the share at an age is 1 over the product of the
/// factors from it, and with a tail a last share of 1 one development
/// period past the oldest age.
pub(crate) fn push_shares(factors: &[f64], tail: Option<f64>, out: &mut Vec<f64>) {
    let start = out.len();
    let mut cdf = tail.unwrap_or(1.0);
    out.resize(start + factors.len() + 1, 0.0);
    out[start + factors.len()] = 1.0 / cdf;
    for (k, f) in factors.iter().enumerate().rev() {
        cdf *= f;
        out[start + k] = 1.0 / cdf;
    }
    if tail.is_some() {
        out.push(1.0);
    }
}

/// [`push_shares`] into `shares`, which has room for exactly them.
pub(crate) fn write_shares(factors: &[f64], tail: Option<f64>, shares: &mut [f64]) {
    let mut out = Vec::with_capacity(shares.len());
    push_shares(factors, tail, &mut out);
    debug_assert_eq!(out.len(), shares.len());
    shares.copy_from_slice(&out[..shares.len()]);
}

/// The grain of a segment's origin periods (every origin of a segment has
/// the triangle's).
pub(crate) fn origin_grain(origins: &[Period]) -> Grain {
    origins.first().map_or(Grain::Year, |p| p.grain())
}

/// The ages of a pattern from `push_shares`: the triangle's `ages`, and
/// with a tail one development period past the oldest.
pub(crate) fn pattern_ages(ages: &[Lag], tail: bool) -> Vec<Lag> {
    let mut out = ages.to_vec();
    if tail {
        let period = match ages {
            [.., a, b] => b - a,
            [only] => *only,
            [] => 12,
        };
        out.push(ages.last().copied().unwrap_or(0) + period);
    }
    out
}

/// The cumulative shares at each age from shares of the ultimate paid at
/// each age that sum to the ultimate, pushed onto `out`: the running sum
/// over the total, ending at exactly 1.
#[cfg(any(test, feature = "bayes"))]
pub(crate) fn push_cumulative(increments: impl IntoIterator<Item = f64>, out: &mut Vec<f64>) {
    let start = out.len();
    let mut sum = 0.0;
    for x in increments {
        sum += x;
        out.push(sum);
    }
    let last = out.len() - 1;
    for v in &mut out[start..last] {
        *v /= sum;
    }
    out[last] = 1.0;
}

fn running_max(shares: &mut [f64]) {
    let mut next = 1.0;
    for s in shares.iter_mut().rev() {
        if s.is_finite() && *s > 0.0 {
            next = *s;
        } else {
            *s = next;
        }
    }
    let mut held = 0.0f64;
    for s in shares {
        held = held.max(*s).min(1.0);
        *s = held;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(ages: Vec<Lag>, draws: Vec<f64>) -> DevelopmentPatterns {
        let n = draws.len() / ages.len();
        DevelopmentPatterns::new(
            n,
            vec![SegmentPatterns::new(
                &Label::default(),
                Grain::Year,
                ages,
                draws,
            )],
        )
    }

    #[test]
    fn shares_are_one_over_the_factors_to_ultimate() {
        let mut out = Vec::new();
        push_shares(&[2.0, 1.25], None, &mut out);
        assert_eq!(out, [0.4, 0.8, 1.0]);
        let mut out = Vec::new();
        push_shares(&[2.0, 1.25], Some(1.25), &mut out);
        assert_eq!(out, [0.32, 0.64, 0.8, 1.0]);
        assert_eq!(pattern_ages(&[12, 24, 36], true), [12, 24, 36, 48]);
        assert_eq!(pattern_ages(&[3, 6], false), [3, 6]);
    }

    #[test]
    fn cumulative_shares_end_at_exactly_one() {
        let mut out = vec![9.0];
        push_cumulative([0.1, 0.2, 0.3], &mut out);
        assert_eq!(out[0], 9.0);
        assert!((out[1] - 1.0 / 6.0).abs() < 1e-15);
        assert!((out[2] - 0.5).abs() < 1e-15);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn a_falling_share_is_refused_or_held() {
        let patterns = one(vec![12, 24, 36], vec![0.5, 0.9, 1.0, 0.6, 1.05, 1.0]);
        let err = patterns
            .monthly(&[], &MonthlyBasis::default(), 36, Monotone::Refuse)
            .unwrap_err();
        assert!(
            err.to_string().contains("falls in 1 of 2 simulations"),
            "{err}"
        );
        assert!(err.to_string().contains("simulation 1"), "{err}");
        let held = patterns
            .monthly(
                &[],
                &MonthlyBasis::Origin(AnnualReading::LnLinear),
                40,
                Monotone::RunningMax,
            )
            .unwrap();
        let row = &held[1];
        assert!(row.windows(2).all(|w| w[1] >= w[0]));
        // Capped at 1 from 24 months on, padded past 36.
        assert_eq!(row[23], 1.0);
        assert_eq!(row[39], 1.0);
        assert!((row[11] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn a_share_that_is_not_positive_is_refused_or_lifted() {
        // Mack's bootstrap can draw a negative first factor.
        let patterns = one(vec![12, 24, 36], vec![0.5, 0.9, 1.0, -2.0, 0.4, 1.0]);
        let err = patterns
            .monthly(&[], &MonthlyBasis::default(), 36, Monotone::Refuse)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("not positive in 1 of 2 simulations (the first, simulation 1, -2 at 12"),
            "{err}"
        );
        let held = patterns
            .monthly(
                &[],
                &MonthlyBasis::Origin(AnnualReading::LnLinear),
                36,
                Monotone::RunningMax,
            )
            .unwrap();
        let row = &held[1];
        assert!(row.windows(2).all(|w| w[1] >= w[0]));
        assert!((row[11] - 0.4).abs() < 1e-12);
        assert!((row[23] - 0.4).abs() < 1e-12);
        assert_eq!(row[35], 1.0);
    }

    #[test]
    fn the_origin_basis_reads_the_given_ages_exactly() {
        let patterns = one(vec![12, 24, 36], vec![0.5, 0.9, 1.0]);
        for reading in [AnnualReading::default(), AnnualReading::LnLinear] {
            let m = patterns
                .monthly(&[], &MonthlyBasis::Origin(reading), 36, Monotone::Refuse)
                .unwrap();
            assert!((m[0][11] - 0.5).abs() < 1e-9, "{reading:?}");
            assert!((m[0][23] - 0.9).abs() < 1e-9, "{reading:?}");
            assert_eq!(m[0][35], 1.0);
            assert!(m[0].windows(2).all(|w| w[1] >= w[0]));
            assert!(m[0][0] > 0.0);
        }
    }

    #[test]
    fn an_underwriting_basis_ends_a_year_later() {
        let patterns = one(vec![12, 24, 36], vec![0.5, 0.9, 1.0]);
        let basis = MonthlyBasis::Underwriting(UnderwritingBasis::default());
        let err = patterns
            .monthly(&[], &basis, 36, Monotone::Refuse)
            .unwrap_err();
        assert!(err.to_string().contains("at 48 months"), "{err}");
        let m = patterns.monthly(&[], &basis, 60, Monotone::Refuse).unwrap();
        assert!((m[0][47] - 1.0).abs() < 1e-12, "{}", m[0][47]);
        assert!(m[0][46] < 1.0 - 1e-9);
        // Less developed at 12 months than the accident year.
        assert!(m[0][11] < 0.5);
    }

    #[test]
    fn segments_are_found_by_label() {
        let patterns = DevelopmentPatterns::new(
            1,
            vec![
                SegmentPatterns::new(&Label::new(["A"]), Grain::Year, vec![12], vec![1.0]),
                SegmentPatterns::new(
                    &Label::new(["B"]),
                    Grain::Year,
                    vec![12, 24],
                    vec![0.5, 1.0],
                ),
            ],
        );
        assert_eq!(patterns.pattern_draws(&["B"]).unwrap(), [[0.5, 1.0]]);
        let err = patterns.segment(&["C"]).unwrap_err().to_string();
        assert!(err.contains("[A], [B]"), "{err}");
        assert!(patterns.segment(&[]).is_err());
        // A segment picked alone keeps its label and answers to it or to
        // no label, not to another.
        let b = patterns.only(&Label::new(["B"]));
        assert_eq!(b.pattern_draws(&[]).unwrap(), [[0.5, 1.0]]);
        assert_eq!(b.pattern_draws(&["B"]).unwrap(), [[0.5, 1.0]]);
        assert!(b.segment(&["A"]).is_err());
    }

    #[test]
    fn an_underwriting_basis_pays_before_the_first_age_under_any_reading() {
        let patterns = one(vec![12, 24, 36], vec![0.5, 0.9, 1.0]);
        for reading in [AnnualReading::default(), AnnualReading::LnLinear] {
            let basis =
                MonthlyBasis::Underwriting(UnderwritingBasis::default().with_reading(reading));
            let m = patterns.monthly(&[], &basis, 60, Monotone::Refuse).unwrap();
            assert!(m[0][0] > 0.0, "{reading:?}");
            assert!(m[0].windows(2).all(|w| w[1] >= w[0]), "{reading:?}");
            assert!((m[0][47] - 1.0).abs() < 1e-12, "{reading:?}");
        }
    }

    #[test]
    fn a_permutation_moves_whole_rows() {
        let mut patterns = one(vec![12, 24], vec![0.1, 1.0, 0.2, 1.0, 0.3, 1.0]);
        patterns.permute(0, &[2, 0, 1]);
        assert_eq!(
            patterns.pattern_draws(&[]).unwrap(),
            [[0.3, 1.0], [0.1, 1.0], [0.2, 1.0]]
        );
    }
}
