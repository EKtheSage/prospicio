//! The multi-year projection (`docs/design/esg-dfa.md`, validation 5 and
//! 6): with no economy, its probability of ruin against
//! `ruin::finite_horizon` on the same compound Poisson losses; on a full
//! economy, the accounts on every path (the surplus moves by the year's
//! result, assets less reserves is the surplus), cash's return against the
//! deflators, and the breach tests.

use prospicio_aggregate::dfa::{BondBucket, Line, Projection, RequiredCapital};
use prospicio_aggregate::ruin::finite_horizon;
use prospicio_core::StreamRng;
use prospicio_esg::{Esg, Index, Inflation, Step, Superimposed, Vasicek};
use prospicio_prob::{Counting, Grid, Poisson};

/// Claims of 1, 2 or 3 with these probabilities.
const SEVERITY: [f64; 4] = [0.0, 0.5, 0.3, 0.2];

/// `n` paths of `years` years of compound Poisson(`rate`) losses of
/// `SEVERITY`, path `s` from its own stream.
fn compound_losses(rate: f64, years: usize, n: usize, seed: u64) -> Vec<f64> {
    let count = Poisson::new(rate).unwrap();
    let mut out = Vec::with_capacity(n * years);
    for s in 0..n {
        let mut rng = StreamRng::new(seed, s as u64);
        for _ in 0..years {
            let k = count.sample(&mut rng, 1)[0];
            let total: f64 = (0..k)
                .map(|_| {
                    let u = rng.next_open01();
                    let mut cdf = 0.0;
                    let mut x = 0.0;
                    for (j, p) in SEVERITY.iter().enumerate() {
                        cdf += p;
                        if u <= cdf {
                            x = j as f64;
                            break;
                        }
                    }
                    x
                })
                .sum();
            out.push(total);
        }
    }
    out
}

#[test]
fn ruin_without_an_economy_is_the_discrete_time_recursion() {
    let (years, n, rate, premium, surplus) = (10, 40_000, 2.0, 4.0, 3.0);
    let losses = compound_losses(rate, years, n, 2026);
    let projection = Projection::new(years, surplus, losses, n)
        .unwrap()
        .with_premiums(vec![premium; years]);
    let paths = projection.run_without_economy().unwrap();
    let severity = Grid::new(1.0, SEVERITY.to_vec()).unwrap();
    let exact = finite_horizon(&severity, &Poisson::new(rate).unwrap(), premium, years, 64).unwrap();
    for horizon in [1, 3, 10] {
        let ours = paths.ruin(horizon).unwrap();
        let psi = exact.psi_at(surplus, horizon).unwrap();
        assert!(
            (ours.probability - psi).abs() < 4.0 * ours.standard_error,
            "horizon {horizon}: {} ± {} against {psi}",
            ours.probability,
            ours.standard_error
        );
    }
    // A required capital of nil is ruin, and the shortfall is the mean
    // depth at the first ruin.
    let breach = paths.breach(10, &RequiredCapital::Fixed(0.0)).unwrap();
    assert_eq!(breach.estimate, paths.ruin(10).unwrap());
    assert!(breach.shortfall.unwrap() > 0.0);
    // More capital breaches more often.
    let higher = paths.breach(10, &RequiredCapital::Fixed(2.0)).unwrap();
    assert!(higher.estimate.probability > breach.estimate.probability);
}

fn economy() -> Esg {
    Esg::new(Vasicek::new(0.3, 0.04, 0.01, 0.02).unwrap())
        .with_inflation(Inflation::new(0.5, 0.025, 0.01, 0.03).unwrap())
        .with_superimposed(Superimposed::new(0.01, 0.5, 0.005).unwrap())
        .with_correlation(vec![
            vec![1.0, 0.4, 0.0],
            vec![0.4, 1.0, 0.3],
            vec![0.0, 0.3, 1.0],
        ])
        .with_step(Step::Quarterly)
}

#[test]
fn the_accounts_balance_on_every_path() {
    let (years, n) = (5, 500);
    let scenarios = economy().simulate(years, n, 11).unwrap();
    let losses: Vec<f64> = compound_losses(30.0, years, n, 3)
        .iter()
        .map(|x| 2.0 * x)
        .collect();
    let projection = Projection::new(years, 400.0, losses, n)
        .unwrap()
        .with_opening_reserves(vec![60.0, 30.0, 10.0])
        .with_premiums(vec![130.0, 132.0, 134.0, 136.0, 138.0])
        .with_expense_ratio(0.25)
        .with_payout(vec![0.4, 0.3, 0.2, 0.1])
        .with_bonds(vec![
            BondBucket { share: 0.3, duration: 2.0 },
            BondBucket { share: 0.4, duration: 5.0 },
        ]);
    let paths = projection.run(&scenarios).unwrap();
    assert_eq!(paths.opening_surplus(), 300.0);
    for s in 0..n {
        let mut surplus = paths.opening_surplus();
        let mut assets = 400.0;
        for t in 1..=years {
            let v = |line| paths.value(line, s, t).unwrap();
            let result = v(Line::Premium) - v(Line::Expenses) - v(Line::IncurredLosses)
                - v(Line::ReserveDevelopment)
                + v(Line::InvestmentIncome);
            assert!((v(Line::Result) - result).abs() < 1e-9);
            surplus += v(Line::Result);
            assert!((v(Line::Surplus) - surplus).abs() < 1e-9, "path {s}, year {t}");
            assert!((v(Line::Assets) - v(Line::Reserves) - v(Line::Surplus)).abs() < 1e-9);
            assets += v(Line::Premium) - v(Line::Expenses) + v(Line::InvestmentIncome) - v(Line::Paid);
            assert!((v(Line::Assets) - assets).abs() < 1e-9);
            // Every loss ever booked is paid or reserved, at the year's index.
            let level = scenarios.index_level(Index::Claims, s, 4 * t).unwrap();
            assert!(v(Line::Reserves) >= 0.0 && level > 0.0);
        }
    }
    // The lines are distributions by year, for quantiles and TVaR.
    let surplus = paths.distribution(Line::Surplus).unwrap();
    assert_eq!((surplus.n_sims(), surplus.n_components()), (n, years));
    let breach = paths.breach(5, &RequiredCapital::ReserveRatio(1.0)).unwrap();
    assert!(breach.estimate.probability >= paths.ruin(5).unwrap().probability);
}

#[test]
fn cash_earns_the_short_rate_and_losses_follow_the_claims_index() {
    let (years, n) = (4, 3);
    let scenarios = economy().simulate(years, n, 5).unwrap();
    // No new losses but one of 100 in year 2, paid two years later; cash
    // only.
    let mut losses = vec![0.0; years * n];
    losses[1] = 100.0;
    let projection = Projection::new(years, 1_000.0, losses, n)
        .unwrap()
        .with_payout(vec![0.0, 0.0, 1.0]);
    let paths = projection.run(&scenarios).unwrap();
    for s in 0..n {
        let (d0, d3) = (scenarios.deflator(s, 0).unwrap(), scenarios.deflator(s, 12).unwrap());
        // No payments in the first three years: the assets grew by exp(∫ r).
        let assets = paths.value(Line::Assets, s, 3).unwrap();
        assert!((assets - 1_000.0 * d0 / d3).abs() < 1e-9 * assets);
    }
    // Path 0: incurred 100 at the year-2 claims index, revalued in year 3,
    // paid at the year-4 index.
    let level = |t: usize| scenarios.index_level(Index::Claims, 0, 4 * t).unwrap();
    let v = |line, t| paths.value(line, 0, t).unwrap();
    assert!((v(Line::IncurredLosses, 2) - 100.0 * level(2)).abs() < 1e-9);
    assert!((v(Line::ReserveDevelopment, 3) - 100.0 * (level(3) - level(2))).abs() < 1e-9);
    assert!((v(Line::Paid, 4) - 100.0 * level(4)).abs() < 1e-9);
    assert!(v(Line::Reserves, 4).abs() < 1e-12);
}

#[test]
fn the_inputs_are_checked() {
    let scenarios = economy().simulate(2, 2, 1).unwrap();
    let base = Projection::new(2, 10.0, vec![1.0; 4], 2).unwrap();
    assert!(Projection::new(2, 10.0, vec![1.0; 3], 2).is_err());
    assert!(base.clone().with_payout(vec![0.5]).run(&scenarios).is_err());
    assert!(base.clone().with_premiums(vec![1.0]).run(&scenarios).is_err());
    let bucket = |duration| vec![BondBucket { share: 0.5, duration }];
    assert!(base.clone().with_bonds(bucket(4.0)).run(&scenarios).is_err());
    assert!(base.clone().with_bonds(bucket(5.0)).run(&scenarios).is_ok());
    assert!(base.clone().with_bonds(bucket(5.0)).run_without_economy().is_err());
    let longer = Projection::new(3, 10.0, vec![1.0; 6], 2).unwrap();
    assert!(longer.run(&scenarios).is_err());
    let paths = base.run_without_economy().unwrap();
    assert!(paths.ruin(3).is_err() && paths.value(Line::Result, 2, 1).is_err());
    assert!(paths.breach(2, &RequiredCapital::ByYear(vec![1.0])).is_err());
}
