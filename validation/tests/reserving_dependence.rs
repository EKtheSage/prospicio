//! Joint reserves across lines of business (`SegmentDependence`, decision
//! 9 of `docs/design/reserving-v02.md`) on the six lines of the CAS loss
//! reserve database (`data/clrd_lines.csv`, paid losses summed over
//! companies, chainladder-python 0.10.1), and the capital path from the
//! joint distribution to an allocation.
//!
//! * Synchronized, two identical lines are perfectly dependent: without
//!   process error their reserves (and one-year CDRs) are equal in every
//!   simulation, for the ODP and Mack's bootstrap.
//! * Synchronized, the lines' parameter error takes the correlation of
//!   their residuals (Kirschner, Kerley and Isaacs 2008, section 4.5;
//!   Taylor and McGuire 2007). To first order a line's reserve is
//!   `sum_c a_c r(p_c)`, with `p_c` the position drawn for cell `c`, so two
//!   lines' reserves have correlation `rho a1.a2 / (|a1| |a2|)`, `rho` the
//!   correlation of their paired residuals over the positions: at most
//!   `|rho|`, and close to it when the lines develop alike. Measured on the
//!   15 pairs with 10,000 simulations, the ODP's parameter-error
//!   correlation is within 0.055 of the residuals' (`rho` from -0.15 to
//!   0.60; the test allows 0.08), and Mack's within 0.145 (0.16 allowed;
//!   the largest gap is ppauto with wkcomp, 0.18 against 0.32: ppauto
//!   develops much faster than the others, so their reserves weigh
//!   different factors' residuals). Process error, drawn independently,
//!   dilutes it (Mack's more than the ODP's); independent segments are
//!   uncorrelated within four standard errors.
//! * Rank correlation reproduces the target Spearman matrix between the
//!   lines' totals within Monte Carlo error, and only reorders whole
//!   simulations of each line.
//! * Capital: the joint distribution's total VaR and TVaR, the TVaR's Euler
//!   allocation to the lines (`prospicio_prob::capital`), and the
//!   diversification benefit, which shrinks as the dependence grows.

use prospicio_prob::capital::AllocationMethod;
use prospicio_prob::{Distortion, Empirical, KeyValue, PredictiveDistribution};
use prospicio_reserving::{
    ChainLadder, DevelopmentColumn, Grain, Long, MackBootstrap, MackProcess, Month, OdpBootstrap,
    OneYearMethod, ProcessDistribution, SegmentDependence, Triangle,
};

const LINES: [&str; 6] = [
    "comauto", "medmal", "othliab", "ppauto", "prodliab", "wkcomp",
];
const SIMS: usize = 10_000;
const SEED: u64 = 20_261_008;

/// The lines of `data/clrd_lines.csv` named in `lines`, as segments of a
/// `lob` key, renamed by `rename` (to repeat a line under two names).
fn clrd(lines: &[(&str, &str)]) -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_lines.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_lines.csv");
    let rows: Vec<Vec<&str>> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    let (mut keys, mut origin, mut ages, mut paid) = (vec![], vec![], vec![], vec![]);
    for &(line, name) in lines {
        for f in rows.iter().filter(|f| f[0] == line) {
            keys.push(name);
            origin.push(Month::january(f[1].parse().unwrap()));
            ages.push(f[2].parse().unwrap());
            paid.push(f[3].parse::<f64>().unwrap());
        }
    }
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// Every line of the database under its own name.
fn all_lines() -> Triangle {
    clrd(&LINES.map(|l| (l, l)))
}

/// Column `j` of a distribution's draws.
fn column(pd: &PredictiveDistribution, j: usize) -> Vec<f64> {
    let m = pd.n_components();
    pd.draw_matrix().chunks(m).map(|r| r[j]).collect()
}

/// Each line's total, simulation by simulation.
fn by_line(pd: &PredictiveDistribution) -> Vec<Vec<f64>> {
    let by = pd.aggregate(&["lob"]).unwrap();
    (0..by.n_components()).map(|j| column(&by, j)).collect()
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}

fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0.0; x.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

/// Correlation of two lines' paired residuals over the positions where
/// both have one.
fn residual_correlation(a: &[f64], b: &[f64]) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = a
        .iter()
        .zip(b)
        .filter(|(x, y)| !x.is_nan() && !y.is_nan())
        .map(|(x, y)| (*x, *y))
        .unzip();
    pearson(&x, &y)
}

fn odp(process: ProcessDistribution, dependence: SegmentDependence) -> OdpBootstrap {
    OdpBootstrap {
        n_sims: SIMS,
        seed: SEED,
        process,
        dependence,
    }
}

fn mack(process: MackProcess, dependence: SegmentDependence) -> MackBootstrap {
    MackBootstrap {
        n_sims: SIMS,
        seed: SEED,
        process,
        dependence,
        ..Default::default()
    }
}

/// The standard error of a sample correlation near `r` from `SIMS` pairs.
fn correlation_se(r: f64) -> f64 {
    (1.0 - r * r) / (SIMS as f64).sqrt()
}

#[test]
fn identical_lines_are_perfectly_dependent() {
    let tri = clrd(&[("wkcomp", "a"), ("wkcomp", "b")]);
    let sync = SegmentDependence::Synchronized;
    let cl = OneYearMethod::ChainLadder(ChainLadder::default());
    let equal = |pd: &PredictiveDistribution, what: &str| {
        let lines = by_line(pd);
        assert!(
            lines[0].iter().zip(&lines[1]).all(|(a, b)| a == b),
            "{what}"
        );
        assert!(lines[0].iter().any(|&v| v != lines[0][0]), "{what} varies");
    };
    let o = odp(ProcessDistribution::None, sync.clone());
    equal(
        &o.fit_segments(&tri, "paid").unwrap().reserves,
        "ODP lifetime",
    );
    equal(
        &o.one_year_segments(&tri, "paid", &cl).unwrap().cdr,
        "ODP one-year",
    );
    let m = mack(MackProcess::None, sync.clone());
    equal(
        &m.fit_segments(&tri, "paid").unwrap().reserves,
        "Mack lifetime",
    );
    equal(
        &m.one_year_segments(&tri, "paid", &cl).unwrap().cdr,
        "Mack one-year",
    );

    // With process error, drawn independently, the two copies are
    // correlated by their parameter error only; independent, not at all.
    let full = by_line(
        &odp(ProcessDistribution::Gamma, sync)
            .fit_segments(&tri, "paid")
            .unwrap()
            .reserves,
    );
    let r = pearson(&full[0], &full[1]);
    assert!(r > 0.3 && r < 0.95, "{r}");
    let ind = by_line(
        &odp(ProcessDistribution::Gamma, SegmentDependence::Independent)
            .fit_segments(&tri, "paid")
            .unwrap()
            .reserves,
    );
    let r = pearson(&ind[0], &ind[1]);
    assert!(r.abs() < 4.0 * correlation_se(0.0), "{r}");
}

/// For each pair of lines: the residual correlation and the correlation of
/// the lines' totals under synchronized parameter error only, synchronized
/// with process error and independent with process error.
struct Pairs {
    residual: Vec<f64>,
    parameter: Vec<f64>,
    full: Vec<f64>,
    independent: Vec<f64>,
}

fn pairs(
    residuals: &[Vec<f64>],
    parameter: &[Vec<f64>],
    full: &[Vec<f64>],
    independent: &[Vec<f64>],
) -> Pairs {
    let mut p = Pairs {
        residual: vec![],
        parameter: vec![],
        full: vec![],
        independent: vec![],
    };
    for i in 0..LINES.len() {
        for j in i + 1..LINES.len() {
            p.residual
                .push(residual_correlation(&residuals[i], &residuals[j]));
            p.parameter.push(pearson(&parameter[i], &parameter[j]));
            p.full.push(pearson(&full[i], &full[j]));
            p.independent
                .push(pearson(&independent[i], &independent[j]));
        }
    }
    p
}

/// The checks of the module documentation: the parameter error's
/// correlation is within `gap` of the residuals' and not above them beyond
/// Monte Carlo error, process error dilutes it, independence removes it.
fn check(name: &str, p: &Pairs, gap: f64) {
    for k in 0..p.residual.len() {
        let (rho, par, full, ind) = (p.residual[k], p.parameter[k], p.full[k], p.independent[k]);
        let se = correlation_se(par);
        eprintln!(
            "{name} pair {k}: residuals {rho:.3} parameter {par:.3} full {full:.3} independent {ind:.3}"
        );
        assert!((par - rho).abs() < gap, "{name} pair {k}: {par} vs {rho}");
        assert!(
            par.abs() < rho.abs() + 4.0 * se,
            "{name} pair {k}: {par} vs {rho}"
        );
        if rho.abs() > 0.25 {
            assert!(
                full * rho > 0.0 && full.abs() < par.abs() + 4.0 * se,
                "{name} pair {k}: {full} vs {par}"
            );
        }
        assert!(
            ind.abs() < 4.0 * correlation_se(0.0),
            "{name} pair {k}: {ind}"
        );
    }
}

#[test]
fn synchronized_odp_takes_the_residual_correlation() {
    let tri = all_lines();
    let sync = SegmentDependence::Synchronized;
    let parameter = odp(ProcessDistribution::None, sync.clone())
        .fit_segments(&tri, "paid")
        .unwrap();
    let residuals: Vec<Vec<f64>> = parameter
        .segments
        .fits
        .iter()
        .map(|f| f.residuals.clone())
        .collect();
    let full = odp(ProcessDistribution::Gamma, sync)
        .fit_segments(&tri, "paid")
        .unwrap();
    let independent = odp(ProcessDistribution::Gamma, SegmentDependence::Independent)
        .fit_segments(&tri, "paid")
        .unwrap();
    let p = pairs(
        &residuals,
        &by_line(&parameter.reserves),
        &by_line(&full.reserves),
        &by_line(&independent.reserves),
    );
    check("ODP", &p, 0.08);
    // Each line's own distribution is its independent bootstrap's: the
    // pools are the same, only the pairing of draws across lines differs.
    let sd = |pd: &PredictiveDistribution| {
        let by = pd.aggregate(&["lob"]).unwrap();
        (0..LINES.len())
            .map(|j| {
                let x = column(&by, j);
                let m = x.iter().sum::<f64>() / x.len() as f64;
                (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
            })
            .collect::<Vec<_>>()
    };
    for (a, b) in sd(&full.reserves).iter().zip(sd(&independent.reserves)) {
        assert!((a / b - 1.0).abs() < 0.05, "{a} vs {b}");
    }
}

#[test]
fn synchronized_mack_takes_the_residual_correlation() {
    let tri = all_lines();
    let sync = SegmentDependence::Synchronized;
    let parameter = mack(MackProcess::None, sync.clone())
        .fit_segments(&tri, "paid")
        .unwrap();
    let residuals: Vec<Vec<f64>> = parameter
        .segments
        .fits
        .iter()
        .map(|f| f.residuals.clone())
        .collect();
    let full = mack(MackProcess::Gamma, sync)
        .fit_segments(&tri, "paid")
        .unwrap();
    let independent = mack(MackProcess::Gamma, SegmentDependence::Independent)
        .fit_segments(&tri, "paid")
        .unwrap();
    let p = pairs(
        &residuals,
        &by_line(&parameter.reserves),
        &by_line(&full.reserves),
        &by_line(&independent.reserves),
    );
    check("Mack", &p, 0.16);
}

#[test]
fn rank_correlation_reproduces_the_target_spearman() {
    // comauto, ppauto and wkcomp with a target Spearman matrix.
    let tri = clrd(&[
        ("comauto", "comauto"),
        ("ppauto", "ppauto"),
        ("wkcomp", "wkcomp"),
    ]);
    let target = [1.0, 0.5, 0.25, 0.5, 1.0, -0.3, 0.25, -0.3, 1.0];
    let rank = SegmentDependence::RankCorrelation {
        spearman: target.to_vec(),
    };
    let independent = odp(ProcessDistribution::Gamma, SegmentDependence::Independent)
        .fit_segments(&tri, "paid")
        .unwrap();
    let ranked = odp(ProcessDistribution::Gamma, rank.clone())
        .fit_segments(&tri, "paid")
        .unwrap();
    let lines = by_line(&ranked.reserves);
    // Spearman's rho from n pairs has a standard error near
    // (1 - rho^2) / sqrt(n) or below; Iman–Conover's scores have exactly
    // the converted correlation, so the error is smaller still.
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let want = target[i * 3 + j];
        let got = spearman(&lines[i], &lines[j]);
        eprintln!("rank ({i}, {j}): {got:.4} vs {want}");
        assert!(
            (got - want).abs() < 4.0 * correlation_se(want),
            "({i}, {j}): {got} vs {want}"
        );
    }
    // Only whole simulations of each line move: every component keeps its
    // draws, and a line's origins stay together.
    let n = independent.reserves.n_components();
    for j in 0..n {
        let mut a = column(&independent.reserves, j);
        let mut b = column(&ranked.reserves, j);
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        assert_eq!(a, b, "component {j}");
    }
    let rows: std::collections::HashSet<Vec<u64>> = independent
        .reserves
        .draw_matrix()
        .chunks(n)
        .map(|r| r[..10].iter().map(|v| v.to_bits()).collect())
        .collect();
    assert!(
        ranked
            .reserves
            .draw_matrix()
            .chunks(n)
            .all(|r| rows.contains(&r[..10].iter().map(|v| v.to_bits()).collect::<Vec<_>>()))
    );
    assert!(
        ranked
            .reserves
            .provenance()
            .parameters
            .iter()
            .any(|(k, _)| k == "rank_correlation")
    );

    // Mack's one-year view takes the same rank correlation of its CDRs.
    let cl = OneYearMethod::ChainLadder(ChainLadder::default());
    let cdr = mack(MackProcess::Gamma, rank)
        .one_year_segments(&tri, "paid", &cl)
        .unwrap()
        .cdr;
    let lines = by_line(&cdr);
    let got = spearman(&lines[0], &lines[1]);
    assert!((got - 0.5).abs() < 4.0 * correlation_se(0.5), "{got}");
}

/// The total's VaR and TVaR and the Euler allocation of the TVaR to the
/// lines, from the joint reserves of comauto and wkcomp.
struct Capital {
    var: f64,
    standalone_var: Vec<f64>,
    tvar: f64,
    allocated: Vec<f64>,
    benefit: f64,
}

fn capital(dependence: SegmentDependence) -> Capital {
    let tri = clrd(&[("comauto", "comauto"), ("wkcomp", "wkcomp")]);
    let fit = odp(ProcessDistribution::Gamma, dependence)
        .fit_segments(&tri, "paid")
        .unwrap();
    let by = fit.reserves.aggregate(&["lob"]).unwrap();
    let var = by.total().var(0.995).unwrap();
    let standalone_var = ["comauto", "wkcomp"]
        .map(|l| {
            by.marginal(&vec![KeyValue::from(l)])
                .unwrap()
                .var(0.995)
                .unwrap()
        })
        .to_vec();
    let tvar = Distortion::tvar(0.99).unwrap();
    let a = by.capital(&tvar, AllocationMethod::Euler).unwrap();
    assert!((a.total - by.total().tvar(0.99).unwrap()).abs() < 1e-6 * a.total);
    assert!((a.allocated.iter().sum::<f64>() / a.total - 1.0).abs() < 1e-9);
    Capital {
        var,
        standalone_var,
        tvar: a.total,
        allocated: a.allocated.clone(),
        benefit: a.diversification_benefit(),
    }
}

#[test]
fn capital_path_from_joint_reserves() {
    let independent = capital(SegmentDependence::Independent);
    let synchronized = capital(SegmentDependence::Synchronized);
    let strong = capital(SegmentDependence::RankCorrelation {
        spearman: vec![1.0, 0.9, 0.9, 1.0],
    });
    for (name, c) in [
        ("independent", &independent),
        ("synchronized", &synchronized),
        ("rank 0.9", &strong),
    ] {
        eprintln!(
            "{name}: VaR {:.0} (standalone {:.0} + {:.0}), TVaR {:.0}, Euler {:.0} + {:.0}, benefit {:.0}",
            c.var,
            c.standalone_var[0],
            c.standalone_var[1],
            c.tvar,
            c.allocated[0],
            c.allocated[1],
            c.benefit
        );
        // Positive dependence (or none): the standalone VaRs sum to more
        // than the portfolio's; TVaR is subadditive whatever the
        // dependence.
        assert!(c.standalone_var.iter().sum::<f64>() >= c.var, "{name}");
        assert!(c.benefit > 0.0, "{name}");
        assert!(c.allocated.iter().all(|&x| x > 0.0), "{name}");
    }
    // The more dependent the lines, the more capital and the less
    // diversification.
    assert!(independent.tvar < synchronized.tvar && synchronized.tvar < strong.tvar);
    assert!(independent.benefit > synchronized.benefit && synchronized.benefit > strong.benefit);
}
