//! The arithmetic around a gradient-boosted model that the Python and R
//! boosters share: the starting score, the weighted quantile it uses for a
//! quantile objective, the Pearson dispersion of the fitted means and the
//! monotonicity check of a partial dependence curve.
//!
//! The trees themselves come from LightGBM or XGBoost in the front ends;
//! these helpers are the parts that must agree between Python and R, so
//! they live here once (#977).

use prospicio_core::{Error, Result};

use crate::family::Family;
use crate::link::Link;

/// The smallest `y` whose cumulative weight, in increasing `y`, reaches
/// `alpha` of the total weight.
///
/// ```
/// use prospicio_models::boosting::weighted_quantile;
///
/// let q = weighted_quantile(&[3.0, 1.0, 2.0], &[1.0, 1.0, 2.0], 0.5).unwrap();
/// assert_eq!(q, 2.0);
/// ```
pub fn weighted_quantile(y: &[f64], w: &[f64], alpha: f64) -> Result<f64> {
    same_length(y.len(), w.len(), "weights")?;
    if y.is_empty() {
        return Err(Error::Data(
            "weighted_quantile needs at least one value".into(),
        ));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(Error::InvalidProbability(alpha));
    }
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
    let total: f64 = w.iter().sum();
    let target = alpha * total;
    let mut cum = 0.0;
    for &i in &order {
        cum += w[i];
        if cum >= target {
            return Ok(y[i]);
        }
    }
    Ok(y[order[order.len() - 1]])
}

/// The constant link-scale score a booster starts from: the weighted mean
/// response on the link scale, given the offset.
///
/// For a log link, `ln(Σ w y / Σ w exp(offset))`, which needs a positive
/// weighted total; for an identity link, `Σ w (y - offset) / Σ w`. Other
/// links are refused.
///
/// ```
/// use prospicio_models::{Link, boosting::base_score};
///
/// let b = base_score(Link::Log, &[1.0, 3.0], &[1.0, 1.0], &[0.0, 0.0]).unwrap();
/// assert!((b - 2f64.ln()).abs() < 1e-15);
/// ```
pub fn base_score(link: Link, y: &[f64], w: &[f64], offset: &[f64]) -> Result<f64> {
    same_length(y.len(), w.len(), "weights")?;
    same_length(y.len(), offset.len(), "offset")?;
    match link {
        Link::Log => {
            let total: f64 = y.iter().zip(w).map(|(y, w)| w * y).sum();
            if total.is_nan() || total <= 0.0 {
                return Err(Error::Data(
                    "the weighted response total must be positive for a log link".into(),
                ));
            }
            let exposure: f64 = offset.iter().zip(w).map(|(o, w)| w * o.exp()).sum();
            Ok((total / exposure).ln())
        }
        Link::Identity => {
            let num: f64 = y
                .iter()
                .zip(offset)
                .zip(w)
                .map(|((y, o), w)| w * (y - o))
                .sum();
            Ok(num / w.iter().sum::<f64>())
        }
        _ => Err(Error::Data(
            "base_score takes a log or an identity link".into(),
        )),
    }
}

/// Each row's Pearson term `w (y - μ)² / V(μ)`, `V` the family's variance
/// function.
pub fn pearson_terms(family: Family, y: &[f64], mu: &[f64], w: &[f64]) -> Result<Vec<f64>> {
    same_length(y.len(), mu.len(), "mu")?;
    same_length(y.len(), w.len(), "weights")?;
    Ok(y.iter()
        .zip(mu)
        .zip(w)
        .map(|((&y, &m), &w)| w * (y - m) * (y - m) / family.variance(m))
        .collect())
}

/// The Pearson dispersion `Σ w (y - μ)² / V(μ) / n`, `n` the number of
/// rows.
///
/// ```
/// use prospicio_models::{Family, boosting::pearson_dispersion};
///
/// let phi = pearson_dispersion(Family::Poisson, &[1.0, 4.0], &[2.0, 2.0], &[1.0, 1.0]).unwrap();
/// assert_eq!(phi, (0.5 + 2.0) / 2.0);
/// ```
pub fn pearson_dispersion(family: Family, y: &[f64], mu: &[f64], w: &[f64]) -> Result<f64> {
    let terms = pearson_terms(family, y, mu, w)?;
    if terms.is_empty() {
        return Err(Error::Data(
            "pearson_dispersion needs at least one row".into(),
        ));
    }
    Ok(terms.iter().sum::<f64>() / terms.len() as f64)
}

/// The indices `i` (from 1) whose step from `means[i - 1]` goes against
/// `direction` (`1` increasing, `-1` decreasing) by more than `tol` times
/// the curve's largest absolute value (or `tol` when the curve is zero).
///
/// ```
/// use prospicio_models::boosting::monotone_violations;
///
/// assert_eq!(monotone_violations(&[1.0, 2.0, 1.5, 3.0], 1, 1e-9).unwrap(), vec![2]);
/// ```
pub fn monotone_violations(means: &[f64], direction: i32, tol: f64) -> Result<Vec<usize>> {
    if direction != 1 && direction != -1 {
        return Err(Error::InvalidParameter {
            name: "direction",
            value: f64::from(direction),
            reason: "must be 1 or -1",
        });
    }
    let largest = means.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let scale = if largest == 0.0 { 1.0 } else { largest };
    let d = f64::from(direction);
    Ok((1..means.len())
        .filter(|&i| d * (means[i] - means[i - 1]) < -tol * scale)
        .collect())
}

fn same_length(n: usize, m: usize, name: &'static str) -> Result<()> {
    if n == m {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value: m as f64,
            reason: "must have one value per observation",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weighted_quantile_matches_the_cumulative_weight() {
        let y = [5.0, 1.0, 3.0, 2.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(weighted_quantile(&y, &w, 0.0).unwrap(), 1.0);
        assert_eq!(weighted_quantile(&y, &w, 0.25).unwrap(), 1.0);
        assert_eq!(weighted_quantile(&y, &w, 0.26).unwrap(), 2.0);
        assert_eq!(weighted_quantile(&y, &w, 1.0).unwrap(), 5.0);
        assert!(weighted_quantile(&y, &w, 1.5).is_err());
        assert!(weighted_quantile(&[], &[], 0.5).is_err());
    }

    #[test]
    fn the_base_score_is_the_weighted_mean_on_the_link_scale() {
        let y = [1.0, 2.0, 6.0];
        let w = [1.0, 2.0, 1.0];
        let off = [0.0, 1.0_f64.ln(), 2.0_f64.ln()];
        let b = base_score(Link::Log, &y, &w, &off).unwrap();
        assert!((b - (11.0_f64 / 5.0).ln()).abs() < 1e-15);
        let b = base_score(Link::Identity, &y, &w, &[0.0, 0.0, 1.0]).unwrap();
        assert!((b - 10.0 / 4.0).abs() < 1e-15);
        assert!(base_score(Link::Log, &[0.0, 0.0], &[1.0, 1.0], &[0.0, 0.0]).is_err());
        assert!(base_score(Link::Logit, &y, &w, &off).is_err());
    }

    #[test]
    fn the_pearson_dispersion_uses_the_variance_function() {
        let y = [1.0, 3.0];
        let mu = [2.0, 2.0];
        let w = [1.0, 2.0];
        let gamma = pearson_dispersion(Family::Gamma, &y, &mu, &w).unwrap();
        assert!((gamma - (0.25 + 0.5) / 2.0).abs() < 1e-15);
        let tw = pearson_terms(Family::Tweedie { power: 1.5 }, &y, &mu, &w).unwrap();
        assert!((tw[1] - 2.0 / 2f64.powf(1.5)).abs() < 1e-15);
        assert!(pearson_dispersion(Family::Poisson, &[], &[], &[]).is_err());
    }

    #[test]
    fn violations_are_measured_against_the_largest_value() {
        assert!(
            monotone_violations(&[3.0, 2.0, 2.0, 1.0], -1, 1e-9)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            monotone_violations(&[3.0, 2.0, 2.5], -1, 0.1).unwrap(),
            vec![2]
        );
        assert!(
            monotone_violations(&[3.0, 2.0, 2.2], -1, 0.1)
                .unwrap()
                .is_empty()
        );
        assert!(
            monotone_violations(&[0.0, 0.0], 1, 1e-9)
                .unwrap()
                .is_empty()
        );
        assert!(monotone_violations(&[1.0], 0, 1e-9).is_err());
    }
}
