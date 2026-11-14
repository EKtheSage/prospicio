//! The alpha-stable law's algebra against `aggregate` 2.1.0's `Stable`
//! (`validation/scripts/aggregate_stable_algebra.py`, #780): sums, shifts,
//! scaling (the log term at `alpha = 1`), negation, `t @ X`, the S0
//! location, the tail constants, `Pr(X > loc)` and the stable limit of a
//! compound Pareto.

use prospicio_prob::{Stable, StableParameterization};
use prospicio_validation::{check, reference};

#[test]
fn stable_algebra_matches_aggregate() {
    let cases = reference("stable_algebra_aggregate.csv");
    check(&cases, |c| {
        let v = |name: &str| c.param("params", name);
        let law = |prefix: &str| {
            Stable::new(
                v(&format!("{prefix}_alpha")),
                v(&format!("{prefix}_beta")),
                v(&format!("{prefix}_scale")),
                v(&format!("{prefix}_loc")),
                StableParameterization::S1,
            )
            .ok()
        };
        let st = match c.get("arg") {
            "identity" => law("x")?,
            "add" => law("x")?.add(&law("y")?).ok()?,
            "shift" => law("x")?.shift(v("c")),
            "neg" => law("x")?.scale_by(-1.0).ok()?,
            "scale" => law("x")?.scale_by(v("c")).ok()?,
            "copies" => law("x")?.copies(v("c")).ok()?,
            "s0" => Stable::new(
                v("x_alpha"),
                v("x_beta"),
                v("x_scale"),
                v("x_loc"),
                StableParameterization::S0,
            )
            .ok()?,
            "compound_pareto" => {
                Stable::from_compound_pareto(v("frequency"), v("alpha"), v("theta"), v("sev_mean"))
                    .ok()?
            }
            _ => return None,
        };
        Some(match c.get("quantity") {
            "alpha" => st.alpha(),
            "beta" => st.beta(),
            "scale" => st.scale(),
            "loc" => st.loc(),
            "loc0" => st.loc0(),
            "tail_constant" => st.tail_constant(),
            "left_tail_constant" => st.left_tail_constant(),
            "prob_above_loc" => st.prob_above_loc(),
            _ => return None,
        })
    });
}
