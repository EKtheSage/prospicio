//! The Munich chain ladder (Quarg and Mack 2004) against R ChainLadder's
//! `MunichChainLadder` and chainladder-python's `MunichAdjustment` on
//! Quarg and Mack's example (`data/mcl.csv`, R's `MCLpaid` and
//! `MCLincurred`; `scripts/reserving_munich_r.R`,
//! `scripts/reserving_munich_python.py`), and the limit where the ratios
//! carry no information, which is the two separate chain ladders.

use prospicio_reserving::{
    ChainLadder, Development, DevelopmentColumn, Error, Grain, Long, LongTable, MunichChainLadder,
    MunichFit, SigmaInterpolation, Tail, Triangle,
};
use prospicio_validation::{Case, check, reference, triangle_columns};

fn mcl() -> Triangle {
    triangle_columns("mcl", &["paid", "incurred"])
}

/// The MCL triangle's long table.
fn mcl_long() -> LongTable {
    mcl().to_long()
}

fn column(long: &LongTable, name: &str) -> Vec<f64> {
    long.values
        .iter()
        .find(|(n, _)| n == name)
        .unwrap()
        .1
        .clone()
}

/// A triangle on MCL's cells with the given paid and incurred values.
fn with_values(long: &LongTable, paid: &[f64], incurred: &[f64]) -> Triangle {
    Triangle::from_long(&Long {
        keys: &[],
        origin: &long.origin,
        development: DevelopmentColumn::Age(&long.development),
        values: &[("paid", paid), ("incurred", incurred)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn development(sigma: SigmaInterpolation) -> Development {
    Development {
        sigma_interpolation: sigma,
        ..Default::default()
    }
}

fn fit(sigma: SigmaInterpolation, tail_paid: Tail, tail_incurred: Tail) -> MunichFit {
    MunichChainLadder {
        paid: development(sigma),
        incurred: development(sigma),
        tail_paid,
        tail_incurred,
    }
    .fit(&mcl(), "paid", "incurred")
    .unwrap()
}

/// The fit's value for one reference row.
fn value(fit: &MunichFit, case: &Case) -> Option<f64> {
    let by_age = |v: &[f64]| {
        let age = case.number("arg")? as u32;
        let k = fit.development.iter().position(|&a| a == age)?;
        v.get(k).copied()
    };
    let by_origin = |v: &[f64]| {
        let year = case.number("arg")? as i32;
        let o = fit.origins.iter().position(|p| p.start().year() == year)?;
        Some(v[o])
    };
    let (cp, ci) = (&fit.paid.chain_ladder, &fit.incurred.chain_ladder);
    match case.get("quantity") {
        "lambda_paid" => Some(fit.lambda_paid),
        "lambda_incurred" => Some(fit.lambda_incurred),
        "ultimate_paid" => by_origin(&fit.ultimate_paid),
        "ultimate_incurred" => by_origin(&fit.ultimate_incurred),
        "q" => by_age(&fit.q),
        "q_inverse" => by_age(&fit.q_inverse),
        "rho_paid" => by_age(&fit.rho_paid),
        "rho_incurred" => by_age(&fit.rho_incurred),
        "sigma_paid" => by_age(&cp.development.sigma),
        "sigma_incurred" => by_age(&ci.development.sigma),
        "tail_paid" => Some(cp.tail.factor),
        "tail_sigma_paid" => Some(cp.tail.sigma),
        "tail_incurred" => Some(ci.tail.factor),
        "tail_sigma_incurred" => Some(ci.tail.sigma),
        _ => None,
    }
}

/// The settings of a reference arm: `mack` or `log-linear` sigma, then
/// the tails.
fn arm(test: &str) -> MunichFit {
    let mut parts = test.split(";tail=");
    let sigma = match parts.next().unwrap() {
        "mack" => SigmaInterpolation::Mack,
        "log-linear" => SigmaInterpolation::LogLinear,
        other => panic!("unknown arm {other}"),
    };
    let (tp, ti) = match parts.next() {
        None => (Tail::default(), Tail::default()),
        Some("log-linear") => (Tail::LogLinear, Tail::LogLinear),
        Some(constants) => {
            let (p, i) = constants.split_once('/').unwrap();
            (
                Tail::from(p.parse::<f64>().unwrap()),
                Tail::from(i.parse::<f64>().unwrap()),
            )
        }
    };
    fit(sigma, tp, ti)
}

fn check_file(name: &str) {
    let cases = reference(name);
    check(&cases, |case| value(&arm(case.get("test")), case));
}

#[test]
fn r_munich_chain_ladder() {
    check_file("reserving_munich_r.csv");
}

#[test]
fn chainladder_python_munich_adjustment() {
    check_file("reserving_munich_python.csv");
}

#[test]
fn r_printed_summary() {
    // R ChainLadder 0.2.21: summary(MunichChainLadder(MCLpaid,
    // MCLincurred))$Totals, the default call, which on this data falls
    // back to Mack's sigma with a warning; prospicio's default
    // (log-linear) gives chainladder-python's figures instead.
    let fit = fit(SigmaInterpolation::Mack, Tail::default(), Tail::default());
    let close = |got: f64, want: f64| assert!((got - want).abs() < 1e-8, "{got} vs {want}");
    close(fit.total_ultimate_paid(), 32_121.496_994_989_8);
    close(fit.total_ultimate_incurred(), 32_719.512_501_042_9);
    close(
        fit.total_ultimate_paid() / fit.total_ultimate_incurred(),
        0.981_722_970_168_516,
    );
    // The latest P/I of the youngest origin is 0.407; Munich takes it to
    // 0.980, near the older origins'.
    close(fit.latest_ratio()[6], 0.407_009_159_697_332);
    close(fit.ultimate_ratio()[6], 0.980_301_200_225_115);
}

#[test]
fn munich_pulls_the_ratios_together() {
    // Separate chain ladders leave the youngest origin's P/I at 0.73; the
    // Munich projection brings every origin's within 0.98 +- 0.01.
    let fit = fit(SigmaInterpolation::Mack, Tail::default(), Tail::default());
    let separate: Vec<f64> = fit
        .paid
        .chain_ladder
        .ultimate
        .iter()
        .zip(&fit.incurred.chain_ladder.ultimate)
        .map(|(p, i)| p / i)
        .collect();
    let spread = |v: &[f64]| {
        let (lo, hi) = v
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                (a.min(x), b.max(x))
            });
        hi - lo
    };
    assert!(spread(&separate) > 0.2, "{separate:?}");
    assert!(spread(&fit.ultimate_ratio()) < 0.01);
}

#[test]
fn no_information_is_the_separate_chain_ladders() {
    // Incurred a fixed multiple of paid: every P/I is the same, so the
    // ratios' residuals are zero, lambda is 0, and the projections are the
    // separate chain ladders of each column.
    let long = mcl_long();
    let paid = column(&long, "paid");
    let incurred: Vec<f64> = paid.iter().map(|v| 1.25 * v).collect();
    let tri = with_values(&long, &paid, &incurred);
    let fit = MunichChainLadder::default()
        .fit(&tri, "paid", "incurred")
        .unwrap();
    assert_eq!((fit.lambda_paid, fit.lambda_incurred), (0.0, 0.0));
    let cp = ChainLadder::default().fit(&tri, "paid").unwrap();
    let ci = ChainLadder::default().fit(&tri, "incurred").unwrap();
    for o in 0..cp.ultimate.len() {
        assert!((fit.ultimate_paid[o] - cp.ultimate[o]).abs() < 1e-9);
        assert!((fit.ultimate_incurred[o] - ci.ultimate[o]).abs() < 1e-9);
    }
}

#[test]
fn a_missing_incurred_cell_is_refused() {
    let long = mcl_long();
    let mut incurred = column(&long, "incurred");
    // The 2004 origin's latest incurred, at 48 months.
    let row = long
        .origin
        .iter()
        .zip(&long.development)
        .position(|(o, &d)| o.year() == 2004 && d == 48)
        .unwrap();
    incurred[row] = f64::NAN;
    let tri = with_values(&long, &column(&long, "paid"), &incurred);
    let err = MunichChainLadder::default()
        .fit(&tri, "paid", "incurred")
        .unwrap_err();
    assert!(matches!(err, Error::Munich(_)), "{err}");
    assert!(err.to_string().contains("origin 2004 at age 48"), "{err}");
}
