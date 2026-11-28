//! The losses' rank dependence on the economy in the projection
//! (`docs/design/esg-dfa.md` decision 7(a)): each year's losses keep their
//! distribution, their Spearman correlation with the year's index growth is
//! the copula's, the dependence moves the incurred losses' mean the way its
//! sign says, and the cases that cannot hold it are refused.

use prospicio_aggregate::dfa::{Line, LossDependence, Projection};
use prospicio_esg::{Esg, Index, Inflation, Scenarios, Step, Vasicek};
use prospicio_prob::copula::Archimedean;

const YEARS: usize = 3;
const N: usize = 10_000;

fn scenarios() -> Scenarios {
    Esg::new(Vasicek::new(0.3, 0.04, 0.01, 0.02).unwrap())
        .with_inflation(Inflation::new(0.5, 0.025, 0.02, 0.03).unwrap())
        .with_step(Step::Quarterly)
        .simulate(YEARS, N, 17)
        .unwrap()
}

/// Distinct losses, each year a permutation of 1..=N independent of the
/// scenarios.
fn losses() -> Vec<f64> {
    (0..N)
        .flat_map(|s| (1..=YEARS).map(move |t| ((s * 7919 + t * 104_729) % N + 1) as f64))
        .collect()
}

fn ranks(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0.0; x.len()];
    for (k, &i) in order.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (rx, ry) = (ranks(x), ranks(y));
    let m = (x.len() - 1) as f64 / 2.0;
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
    let var: f64 = rx.iter().map(|a| (a - m).powi(2)).sum();
    cov / var
}

/// The year's index growth on each path, and the year's new losses in base
/// money read back from the incurred losses.
fn growth_and_losses(
    sc: &Scenarios,
    paths: &prospicio_aggregate::dfa::ProjectionPaths,
    t: usize,
) -> (Vec<f64>, Vec<f64>) {
    let level = |s, t: usize| sc.index_level(Index::Claims, s, 4 * t).unwrap();
    (0..N)
        .map(|s| {
            let g = level(s, t) / level(s, t - 1);
            (g, paths.value(Line::IncurredLosses, s, t).unwrap() / level(s, t))
        })
        .unzip()
}

#[test]
fn the_losses_keep_their_distribution_and_take_the_copulas_ranks() {
    let sc = scenarios();
    let rho = 0.6;
    let paths = Projection::new(YEARS, 1_000.0, losses(), N)
        .unwrap()
        .with_loss_dependence(LossDependence::Gaussian { rho }, 3)
        .run(&sc)
        .unwrap();
    let want = 6.0 / std::f64::consts::PI * (rho / 2.0).asin();
    for t in 1..=YEARS {
        let (growth, base) = growth_and_losses(&sc, &paths, t);
        let mut sorted = base.clone();
        sorted.sort_by(f64::total_cmp);
        for (k, v) in sorted.iter().enumerate() {
            assert!((v - (k + 1) as f64).abs() < 1e-6, "year {t}: the losses changed");
        }
        let r = spearman(&growth, &base);
        assert!((r - want).abs() < 0.03, "year {t}: Spearman {r} against {want}");
    }

    // Gumbel's upper tail: positive rank correlation.
    let gumbel = Projection::new(YEARS, 1_000.0, losses(), N)
        .unwrap()
        .with_loss_dependence(
            LossDependence::Archimedean {
                family: Archimedean::Gumbel,
                theta: 2.0,
            },
            3,
        )
        .run(&sc)
        .unwrap();
    let (growth, base) = growth_and_losses(&sc, &gumbel, 2);
    assert!(spearman(&growth, &base) > 0.5);
}

#[test]
fn the_dependence_moves_the_incurred_losses_with_its_sign() {
    let sc = scenarios();
    let mean_incurred = |dep: Option<LossDependence>| {
        let mut p = Projection::new(YEARS, 1_000.0, losses(), N).unwrap();
        if let Some(d) = dep {
            p = p.with_loss_dependence(d, 9);
        }
        let paths = p.run(&sc).unwrap();
        (0..N)
            .map(|s| paths.value(Line::IncurredLosses, s, YEARS).unwrap())
            .sum::<f64>()
            / N as f64
    };
    let up = mean_incurred(Some(LossDependence::Gaussian { rho: 0.9 }));
    let none = mean_incurred(None);
    let down = mean_incurred(Some(LossDependence::Gaussian { rho: -0.9 }));
    assert!(up > none && none > down, "{up}, {none}, {down}");
}

#[test]
fn dependence_without_a_varying_index_is_refused() {
    let dep = LossDependence::Gaussian { rho: 0.5 };
    let projection = Projection::new(2, 100.0, vec![1.0, 2.0, 3.0, 4.0], 2)
        .unwrap()
        .with_loss_dependence(dep, 1);
    assert!(projection.run_without_economy().is_err());
    let no_inflation = Esg::new(Vasicek::new(0.3, 0.04, 0.01, 0.02).unwrap())
        .simulate(2, 2, 1)
        .unwrap();
    assert!(projection.run(&no_inflation).is_err());
    let bad = Projection::new(2, 100.0, vec![1.0, 2.0, 3.0, 4.0], 2)
        .unwrap()
        .with_loss_dependence(LossDependence::Gaussian { rho: 1.5 }, 1);
    let with_inflation = Esg::new(Vasicek::new(0.3, 0.04, 0.01, 0.02).unwrap())
        .with_inflation(Inflation::new(0.5, 0.025, 0.02, 0.03).unwrap())
        .simulate(2, 2, 1)
        .unwrap();
    assert!(bad.run(&with_inflation).is_err());
}
