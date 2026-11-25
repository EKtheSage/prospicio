//! Reinsurer default over a term against closed forms (#980,
//! `docs/design/reinsurer-default.md`, "Over a term"): `Panel::apply_term`
//! and `Panel::by_reinsurer_term` on a tower's paths of independent years.
//!
//! A default is final, so with a constant one-year probability of default
//! `p` and independent years a reinsurer is in default by year `t` with
//! probability `q_t = 1 - (1 - p)^t`. With `rho = 0` the defaults are
//! independent of the losses, and without collateral the credit loss in
//! year `t` has mean `q_t (1 - R) E[E_t]`; given the simulated exposures
//! `E_jt` of the paths `j`, its sample mean has mean `q_t (1 - R) mean(E_t)`
//! and standard error `(1 - R) sqrt(q_t (1 - q_t) sum_j E_jt^2) / n`
//! exactly, since the paths default independently. With `rho > 0` each
//! path-year's factor comes from its rank among all path-years, and the
//! factor keeps the one-year probability, so with independent years the
//! rate of default by year `t` is still `q_t`. Every check passes within
//! four standard errors, and a power check shows that a default that is
//! not final (`q_t = p`) fails by more than that.

use prospicio_aggregate::{EventSet, Layer, Panel, Reinsurer, Tower, simulate_events};
use prospicio_prob::{Empirical, KeyValue, Lognormal, Poisson, PredictiveDistribution};

/// Paths.
const N_PATHS: usize = 100_000;
/// Years in a path.
const N_YEARS: usize = 4;
/// A simulated value passes within this many standard errors.
const MAX_Z: f64 = 4.0;

/// A term of independent years: one simulation of `N_PATHS * N_YEARS`
/// years split into consecutive blocks, block `t` being year `t + 1`.
fn term() -> PredictiveDistribution {
    let events = simulate_events(
        &Poisson::new(1.5).unwrap(),
        &Lognormal::new(13.0, 1.2).unwrap(),
        N_PATHS * N_YEARS,
        980,
    )
    .unwrap();
    let years: Vec<EventSet> = (0..N_YEARS)
        .map(|t| {
            let block = (t * N_PATHS..(t + 1) * N_PATHS)
                .map(|i| events.events(i).to_vec())
                .collect();
            EventSet::from_years(block, 980).unwrap()
        })
        .collect();
    Tower::new(vec![Layer::xol("A", 2e6, 5e5).unwrap()])
        .unwrap()
        .apply_term(&years)
        .unwrap()
}

fn column(pd: &PredictiveDistribution, kind: &str, name: &str, year: usize) -> Vec<f64> {
    let key = vec![
        KeyValue::from(kind),
        KeyValue::from(name),
        KeyValue::Int(year as i64),
    ];
    Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// The z-score of a default frequency against `q`.
fn z_frequency(defaults: &[f64], q: f64) -> f64 {
    (mean(defaults) - q) / (q * (1.0 - q) / defaults.len() as f64).sqrt()
}

/// Two reinsurers with independent default and no collateral.
fn independent_panel() -> Panel {
    Panel::new(vec![
        Reinsurer::new("X", 0.15, 0.4).unwrap(),
        Reinsurer::new("Y", 0.05, 0.2).unwrap(),
    ])
    .unwrap()
    .line("X", "A", 0.6)
    .unwrap()
    .line("Y", "A", 0.4)
    .unwrap()
}

#[test]
fn independent_default_is_final_and_its_credit_loss_has_its_closed_form() {
    let term = term();
    let panel = independent_panel();
    let by = panel.by_reinsurer_term(&term, N_YEARS, 2026).unwrap();
    let after = panel.apply_term(&term, N_YEARS, 2026).unwrap();
    let n = N_PATHS as f64;
    for re in panel.reinsurers() {
        let (p, recovery) = (re.probability, re.recovery);
        for t in 1..=N_YEARS {
            let q = 1.0 - (1.0 - p).powi(t as i32);
            let defaults = column(&by, "default", &re.name, t);
            let z = z_frequency(&defaults, q);
            assert!(z.abs() <= MAX_Z, "{} year {t}: default rate z = {z}", re.name);
            if t >= 2 {
                // A default that is not final would default at p a year.
                let z_not_final = z_frequency(&defaults, p);
                assert!(z_not_final > MAX_Z, "{} year {t}: {z_not_final}", re.name);
            }
            // Without collateral the exposure after collateral is E.
            let exposure = column(&by, "exposure", &re.name, t);
            let credit = column(&by, "credit_loss", &re.name, t);
            let sum_sq: f64 = exposure.iter().map(|e| e * e).sum();
            let want = q * (1.0 - recovery) * mean(&exposure);
            let se = (1.0 - recovery) * (q * (1.0 - q) * sum_sq).sqrt() / n;
            let z = (mean(&credit) - want) / se;
            assert!(z.abs() <= MAX_Z, "{} year {t}: credit loss z = {z}", re.name);
            if t >= 2 {
                let not_final = p * (1.0 - recovery) * mean(&exposure);
                let se_alt = (1.0 - recovery) * (p * (1.0 - p) * sum_sq).sqrt() / n;
                let z_alt = (mean(&credit) - not_final) / se_alt;
                assert!(z_alt > MAX_Z, "{} year {t}: {z_alt}", re.name);
            }
        }
    }
    // The layer's credit loss is the reinsurers' together.
    for t in 1..=N_YEARS {
        let layer = column(&after, "credit_loss", "A", t);
        let (x, y) = (
            column(&by, "credit_loss", "X", t),
            column(&by, "credit_loss", "Y", t),
        );
        for j in 0..N_PATHS {
            assert!((layer[j] - (x[j] + y[j])).abs() <= 1e-9 * (1.0 + x[j] + y[j]));
        }
    }
}

#[test]
fn the_factor_keeps_the_rate_of_default_by_year() {
    let term = term();
    let p = 0.12;
    let panel = Panel::new(vec![
        Reinsurer::new("X", p, 0.4)
            .unwrap()
            .correlation(0.5)
            .unwrap(),
    ])
    .unwrap()
    .line("X", "A", 1.0)
    .unwrap();
    let by = panel.by_reinsurer_term(&term, N_YEARS, 7).unwrap();
    for t in 1..=N_YEARS {
        let q = 1.0 - (1.0 - p).powi(t as i32);
        let defaults = column(&by, "default", "X", t);
        let z = z_frequency(&defaults, q);
        assert!(z.abs() <= MAX_Z, "year {t}: default rate z = {z}");
    }
    // Wrong-way risk: in year 1 the defaults gather in the paths with the
    // largest gross loss.
    let gross = column(&term, "gross", "ground_up", 1);
    let mut sorted = gross.clone();
    sorted.sort_by(f64::total_cmp);
    let top = sorted[N_PATHS * 9 / 10];
    let defaults = column(&by, "default", "X", 1);
    let worst: Vec<f64> = (0..N_PATHS)
        .filter(|&j| gross[j] > top)
        .map(|j| defaults[j])
        .collect();
    assert!(mean(&worst) > 2.0 * p, "{}", mean(&worst));
}
