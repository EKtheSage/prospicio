//! Predictive distributions from fitted means, for engines that give a
//! mean per row and nothing else (gradient boosting through its Python or
//! R package, an imported network): the family adds the process noise, and
//! several mean vectors (bootstrap refits) add the parameter uncertainty.

use prospicio_core::{Error, Result};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::family::Family;

/// Joint draws of the responses of `n` rows, components keyed
/// `row = 0, 1, …`, as `prospicio_glm` keys its fits.
///
/// `means` holds one or more mean vectors of length `n` (one per bootstrap
/// refit, say). Simulation `i` takes stream `i` of `seed`, picks one vector
/// uniformly from it, then draws each row's response from `family` with
/// that mean, the row's dispersion and the row's weight. With one vector
/// the draws carry process noise only. `dispersion` holds one value for
/// every row, or one per row (from a dispersion model); `weights` empty
/// means every weight is 1.
///
/// ```
/// use prospicio_models::Family;
/// use prospicio_models::simulate::from_means;
/// use prospicio_prob::{Distribution, Provenance};
///
/// // Two policies with expected claim counts 0.1 and 0.4.
/// let pd = from_means(Family::Poisson, &[vec![0.1, 0.4]], &[1.0], &[], 20_000, 7,
///                     Provenance::new("example")).unwrap();
/// assert!((pd.total().mean() - 0.5).abs() < 0.02);
/// ```
pub fn from_means(
    family: Family,
    means: &[Vec<f64>],
    dispersion: &[f64],
    weights: &[f64],
    n_sims: usize,
    seed: u64,
    provenance: Provenance,
) -> Result<PredictiveDistribution> {
    family.validate()?;
    let Some(n) = means.first().map(Vec::len) else {
        return Err(Error::Data("needs at least one mean vector".into()));
    };
    if n == 0 || means.iter().any(|m| m.len() != n) {
        return Err(Error::Data(
            "every mean vector needs the same, non-zero number of rows".into(),
        ));
    }
    if let Some(&bad) = means.iter().flatten().find(|&&m| !family.valid_mu(m)) {
        return Err(Error::InvalidParameter {
            name: "means",
            value: bad,
            reason: "outside the family's range",
        });
    }
    if dispersion.len() != 1 && dispersion.len() != n {
        return Err(Error::Data(format!(
            "dispersion needs 1 or {n} values, got {}",
            dispersion.len()
        )));
    }
    if let Some(&bad) = dispersion.iter().find(|&&d| !(d.is_finite() && d > 0.0)) {
        return Err(Error::InvalidParameter {
            name: "dispersion",
            value: bad,
            reason: "must be positive and finite",
        });
    }
    let phi = |i: usize| dispersion[if dispersion.len() == 1 { 0 } else { i }];
    let ones = vec![1.0; n];
    let weights = if weights.is_empty() {
        &ones[..]
    } else {
        weights
    };
    if weights.len() != n || weights.iter().any(|w| !(w.is_finite() && *w > 0.0)) {
        return Err(Error::Data(format!(
            "weights need {n} positive finite values"
        )));
    }
    let k = means.len();
    let components: Vec<ComponentKey> = (0..n).map(|i| vec![KeyValue::from(i as i64)]).collect();
    let provenance = provenance
        .param("family", family.name())
        .param(
            "dispersion",
            if dispersion.len() == 1 {
                dispersion[0].to_string()
            } else {
                "per row".to_string()
            },
        )
        .param("mean_vectors", k);
    // Every mean is in the family's range, so a draw only fails for an
    // unusual (weight, dispersion) pair; it becomes NaN there, as prospicio_glm's
    // draws do, and from_draws reports it.
    PredictiveDistribution::simulate(
        vec!["row".into()],
        components,
        n_sims,
        seed,
        provenance,
        |rng, row| {
            let mu = &means[((rng.next_open01() * k as f64) as usize).min(k - 1)];
            for (i, (out, (&m, &w))) in row.iter_mut().zip(mu.iter().zip(weights)).enumerate() {
                *out = family
                    .draw(m, phi(i), w, rng.next_open01())
                    .unwrap_or(f64::NAN);
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::Distribution;

    #[test]
    fn draws_centre_on_the_means_with_the_family_variance() {
        let pd = from_means(
            Family::Gamma,
            &[vec![100.0, 300.0]],
            &[0.5],
            &[1.0, 2.0],
            40_000,
            3,
            Provenance::new("test"),
        )
        .unwrap();
        // Gamma: variance φ μ² / w.
        for (j, (mu, w)) in [(100.0, 1.0), (300.0, 2.0)].into_iter().enumerate() {
            let m = pd.marginal(&vec![KeyValue::from(j as i64)]).unwrap();
            assert!((m.mean() / mu - 1.0).abs() < 0.02, "{j}");
            assert!(
                (m.variance() / (0.5 * mu * mu / w) - 1.0).abs() < 0.05,
                "{j}"
            );
        }
    }

    #[test]
    fn several_mean_vectors_add_their_spread() {
        // Two refits that disagree: the mixture's variance is the process
        // variance plus the spread of the means.
        let means = [vec![2.0], vec![6.0]];
        let pd = from_means(
            Family::Gaussian,
            &means,
            &[1.0],
            &[],
            40_000,
            5,
            Provenance::new("test"),
        )
        .unwrap();
        let total = pd.total();
        assert!((total.mean() - 4.0).abs() < 0.05);
        assert!((total.variance() - (1.0 + 4.0)).abs() < 0.15);
        assert_eq!(pd.provenance().seed, Some(5));
    }

    #[test]
    fn per_row_dispersion() {
        // Gamma variance φ μ²: the second row's φ is four times the first.
        let pd = from_means(
            Family::Gamma,
            &[vec![100.0, 100.0]],
            &[0.1, 0.4],
            &[],
            40_000,
            9,
            Provenance::new("test"),
        )
        .unwrap();
        let v = |j: i64| pd.marginal(&vec![KeyValue::from(j)]).unwrap().variance();
        assert!((v(0) / 1000.0 - 1.0).abs() < 0.05);
        assert!((v(1) / 4000.0 - 1.0).abs() < 0.05);
        let p = || Provenance::new("test");
        assert!(
            from_means(
                Family::Gamma,
                &[vec![1.0, 2.0]],
                &[1.0, 1.0, 1.0],
                &[],
                10,
                1,
                p()
            )
            .is_err()
        );
        assert!(
            from_means(
                Family::Gamma,
                &[vec![1.0, 2.0]],
                &[1.0, -1.0],
                &[],
                10,
                1,
                p()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_bad_input() {
        let p = || Provenance::new("test");
        assert!(from_means(Family::Poisson, &[], &[1.0], &[], 10, 1, p()).is_err());
        assert!(
            from_means(
                Family::Poisson,
                &[vec![1.0], vec![]],
                &[1.0],
                &[],
                10,
                1,
                p()
            )
            .is_err()
        );
        assert!(from_means(Family::Poisson, &[vec![-1.0]], &[1.0], &[], 10, 1, p()).is_err());
        assert!(from_means(Family::Poisson, &[vec![1.0]], &[0.0], &[], 10, 1, p()).is_err());
        assert!(
            from_means(
                Family::Poisson,
                &[vec![1.0]],
                &[1.0],
                &[1.0, 2.0],
                10,
                1,
                p()
            )
            .is_err()
        );
    }
}
