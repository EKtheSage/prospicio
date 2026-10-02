//! Distribution parity: moments and quantiles against SciPy; limited
//! expected values and grid masses against 30-digit mpmath references.

use act_prob::{Distribution, Lognormal, Severity};
use act_validation::{check, reference};

#[test]
fn lognormal_matches_scipy() {
    let cases: Vec<_> = reference("distributions_scipy.csv")
        .into_iter()
        .filter(|c| c.get("distribution") == "lognormal")
        .collect();
    check(&cases, |c| {
        let d = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?;
        let arg = c.number("arg");
        match c.get("quantity") {
            "mean" => Some(d.mean()),
            "variance" => Some(d.variance()),
            "cdf" => Some(d.cdf(arg?)),
            "quantile" => d.quantile(arg?).ok(),
            _ => None,
        }
    });
}

#[test]
fn lognormal_severity_matches_integration() {
    let cases = reference("severity_mpmath.csv");
    check(&cases, |c| {
        let d = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?;
        let arg = c.number("arg")?;
        match c.get("quantity") {
            "lev" => Some(d.lev(arg)),
            "stop_loss" => Some(d.stop_loss(arg)),
            "layer" => Some(d.layer(arg, c.number("arg2")?)),
            _ => None,
        }
    });
}

#[test]
fn lognormal_grids_match_textbook_masses() {
    use act_prob::Grid;
    let cases = reference("grid_mpmath.csv");
    check(&cases, |c| {
        let d = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?;
        let step = c.number("step")?;
        let points = c.number("points")? as usize;
        let (grid, _) = match c.get("method") {
            "local_moment" => Grid::local_moment(&d, step, points),
            "rounding" => Grid::rounding(&d, step, points),
            "lower" => Grid::lower(&d, step, points),
            _ => return None,
        }
        .ok()?;
        grid.probs().get(c.number("index")? as usize).copied()
    });
}

#[test]
fn claim_counts_match_scipy() {
    use act_prob::{Counting, NegativeBinomial, Poisson};
    let cases: Vec<_> = reference("distributions_scipy.csv")
        .into_iter()
        .filter(|c| matches!(c.get("distribution"), "poisson" | "negative_binomial"))
        .collect();
    check(&cases, |c| {
        let n: Box<dyn Counting> = match c.get("distribution") {
            "poisson" => Box::new(Poisson::new(c.param("params", "lambda")).ok()?),
            _ => Box::new(
                NegativeBinomial::new(c.param("params", "r"), c.param("params", "beta")).ok()?,
            ),
        };
        let arg = c.number("arg");
        match c.get("quantity") {
            "mean" => Some(n.mean()),
            "variance" => Some(n.variance()),
            "pmf" => Some(n.pmf(arg? as u64)),
            "cdf" => Some(n.cdf(arg? as u64)),
            "quantile" => n.quantile(arg?).ok().map(|k| k as f64),
            _ => None,
        }
    });
}

#[test]
fn distortions_match_integration() {
    use act_prob::{Distortion, Grid};
    // The discrete distribution in validation/scripts/mpmath_distortion.py.
    let values = [0.0, 1.0, 2.0, 5.0, 10.0, 100.0];
    let probs = [0.5, 0.2, 0.15, 0.1, 0.0499999999, 1e-10];
    let cases = reference("distortion_mpmath.csv");
    check(&cases, |c| {
        let a = c.number("arg")?;
        let d = match c.get("quantity") {
            "tvar" => Distortion::tvar(a),
            "wang" => Distortion::wang(a),
            "proportional_hazard" => Distortion::proportional_hazard(a),
            "dual_power" => Distortion::dual_power(a),
            _ => return None,
        }
        .ok()?;
        match c.get("distribution") {
            "discrete" => Some(d.apply_discrete(&values, &probs)),
            "lognormal" => {
                let ln = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog"))
                    .ok()?;
                let step = c.param("params", "step");
                let points = c.param("params", "points") as usize;
                let (grid, _) = Grid::local_moment(&ln, step, points).ok()?;
                Some(grid.distortion(&d))
            }
            _ => None,
        }
    });
}
