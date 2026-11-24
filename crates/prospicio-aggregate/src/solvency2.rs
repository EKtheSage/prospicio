//! The Solvency II standard formula's counterparty default module for type 1
//! exposures (reinsurers and other rated counterparties), as pure functions:
//! Commission Delegated Regulation (EU) 2015/35, as amended by 2016/467 and
//! 2019/981, Articles 192(2), 199, 200 and 201 (#980,
//! `docs/design/reinsurer-default.md` decision (k)).
//!
//! Every input is the caller's: the best estimate of the recoverables, the
//! risk-mitigating effect `RM_re` (Art 196, the underwriting SCR without the
//! arrangement less the SCR with it), the risk-adjusted value of the
//! collateral (Art 197) and the factor that applies to it. The library has
//! no standard formula underwriting SCR to compute `RM_re` from.
//!
//! ```
//! use prospicio_aggregate::solvency2;
//!
//! // Two reinsurers: credit quality step 2, and an unrated Solvency II
//! // undertaking with a solvency ratio of 150 %.
//! let p_a = solvency2::probability_by_step(2).unwrap();
//! let p_b = solvency2::probability_by_ratio(1.5).unwrap();
//! assert_eq!((p_a, p_b), (0.0005, 0.001));
//! // Their losses given default: recoverables of 100 and 80, risk-mitigating
//! // effects of 40 and 20, collateral of 30 held against the first.
//! let lgd_a = solvency2::reinsurance_lgd(100.0, 40.0, 30.0, 0.5, 1.0).unwrap();
//! let lgd_b = solvency2::reinsurance_lgd(80.0, 20.0, 0.0, 0.5, 1.0).unwrap();
//! assert_eq!((lgd_a, lgd_b), (30.0, 45.0));
//! let t = solvency2::type1(&[(p_a, lgd_a), (p_b, lgd_b)]).unwrap();
//! assert_eq!(t.total_lgd, 75.0);
//! // sigma is well under 7 % of the total, so the charge is 3 sigma.
//! assert!(t.sigma <= 0.07 * t.total_lgd);
//! assert!((t.scr - 3.0 * t.sigma).abs() < 1e-12);
//! ```

use prospicio_core::{Error, Result};

/// The probabilities of default of Art 199(2), by credit quality step 0
/// to 6.
const BY_STEP: [f64; 7] = [0.00002, 0.0001, 0.0005, 0.0024, 0.012, 0.042, 0.042];

/// The table of Art 199(3): solvency ratio (as a fraction, descending) and
/// probability of default.
const BY_RATIO: [(f64, f64); 8] = [
    (1.96, 0.0001),
    (1.75, 0.0005),
    (1.50, 0.001),
    (1.25, 0.002),
    (1.22, 0.0024),
    (1.00, 0.005),
    (0.95, 0.012),
    (0.75, 0.042),
];

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

/// The probability of default of a counterparty with an external credit
/// assessment, by its credit quality step (Art 199(2)): steps 0 to 6 give
/// 0.002 %, 0.01 %, 0.05 %, 0.24 %, 1.2 %, 4.2 % and 4.2 %.
///
/// # Errors
///
/// A step above 6.
///
/// ```
/// use prospicio_aggregate::solvency2::probability_by_step;
///
/// assert_eq!(probability_by_step(3).unwrap(), 0.0024);
/// assert!(probability_by_step(7).is_err());
/// ```
pub fn probability_by_step(step: u8) -> Result<f64> {
    BY_STEP.get(usize::from(step)).copied().ok_or_else(|| {
        invalid(
            "step",
            f64::from(step),
            "must be a credit quality step, 0 to 6",
        )
    })
}

/// The probability of default of an unrated insurance or reinsurance
/// undertaking that meets its MCR, by its solvency ratio (Art 199(3)), the
/// ratio given as a fraction (1.5 for 150 %).
///
/// Linear in the ratio between the table's points: 196 % (0.01 %), 175 %
/// (0.05 %), 150 % (0.1 %), 125 % (0.2 %), 122 % (0.24 %), 100 % (0.5 %),
/// 95 % (1.2 %) and 75 % (4.2 %); 0.01 % at or above 196 % and 4.2 % at or
/// below 75 %. A ratio given in percent (150) is far above 1.96 and so
/// gives 0.01 %. An undertaking that does not meet its MCR has 4.2 %
/// (Art 199(3)), and one whose first ratio is not yet disclosed takes its
/// credit quality step or a ratio of 100 % (Art 199(5)); both are the
/// caller's to apply.
///
/// # Errors
///
/// A ratio that is NaN or negative.
///
/// ```
/// use prospicio_aggregate::solvency2::probability_by_ratio;
///
/// assert_eq!(probability_by_ratio(1.25).unwrap(), 0.002);
/// // Half way from 150 % to 125 %.
/// assert!((probability_by_ratio(1.375).unwrap() - 0.0015).abs() < 1e-15);
/// assert_eq!(probability_by_ratio(2.5).unwrap(), 0.0001);
/// assert_eq!(probability_by_ratio(0.5).unwrap(), 0.042);
/// ```
pub fn probability_by_ratio(scr_ratio: f64) -> Result<f64> {
    if scr_ratio.is_nan() || scr_ratio < 0.0 {
        return Err(invalid(
            "scr_ratio",
            scr_ratio,
            "must be a non-negative fraction (1.5 for 150 %)",
        ));
    }
    let (top, p_top) = BY_RATIO[0];
    let (bottom, p_bottom) = BY_RATIO[BY_RATIO.len() - 1];
    if scr_ratio >= top {
        return Ok(p_top);
    }
    if scr_ratio <= bottom {
        return Ok(p_bottom);
    }
    for pair in BY_RATIO.windows(2) {
        let ((hi, p_hi), (lo, p_lo)) = (pair[0], pair[1]);
        if scr_ratio == lo {
            return Ok(p_lo);
        }
        if scr_ratio > lo {
            let w = (scr_ratio - lo) / (hi - lo);
            return Ok(p_lo + w * (p_hi - p_lo));
        }
    }
    unreachable!("the ratio is between the table's ends")
}

/// The loss given default of a reinsurance arrangement (Art 192(2)):
/// `max(lgd_share * (recoverables + 0.5 * risk_mitigation) - factor *
/// collateral, 0)`.
///
/// - `recoverables`: the best estimate of the amounts recoverable from the
///   arrangement and the corresponding debtors.
/// - `risk_mitigation`: `RM_re`, the arrangement's risk-mitigating effect on
///   underwriting risk (Art 196).
/// - `collateral`: the risk-adjusted value of the collateral (Art 197).
/// - `lgd_share`: 0.5, or 0.9 for a counterparty with 60 % or more of its
///   assets under collateral arrangements (Art 192(2), second
///   subparagraph).
/// - `factor`: the factor of Art 197(7) on the collateral, F (100 %, or
///   50 % when the undertaking's share of the insolvency estate above the
///   collateral takes account of receiving the collateral) in the 50 %
///   case, F' (100 % or 18 %, a figure EIOPA Q&A 3119 questions) in the
///   90 % case.
///
/// Any share and factor in [0, 1] is accepted, so that a later amendment or
/// a supervisor's reading needs no new version; the two regulatory shares
/// and the factors above are the caller's to choose.
///
/// # Errors
///
/// Recoverables or a risk-mitigating effect that are not finite, collateral
/// that is not finite and non-negative, or a share or factor outside
/// [0, 1].
///
/// ```
/// use prospicio_aggregate::solvency2::reinsurance_lgd;
///
/// // 50 % of (100 + 50 % of 40), less collateral of 30.
/// assert_eq!(reinsurance_lgd(100.0, 40.0, 30.0, 0.5, 1.0).unwrap(), 30.0);
/// // Collateral above the exposure: no loss given default.
/// assert_eq!(reinsurance_lgd(100.0, 40.0, 200.0, 0.5, 1.0).unwrap(), 0.0);
/// ```
pub fn reinsurance_lgd(
    recoverables: f64,
    risk_mitigation: f64,
    collateral: f64,
    lgd_share: f64,
    factor: f64,
) -> Result<f64> {
    if !recoverables.is_finite() {
        return Err(invalid("recoverables", recoverables, "must be finite"));
    }
    if !risk_mitigation.is_finite() {
        return Err(invalid(
            "risk_mitigation",
            risk_mitigation,
            "must be finite",
        ));
    }
    if !(collateral.is_finite() && collateral >= 0.0) {
        return Err(invalid(
            "collateral",
            collateral,
            "must be finite and non-negative",
        ));
    }
    let unit = |x: f64| (0.0..=1.0).contains(&x);
    if !unit(lgd_share) {
        return Err(invalid(
            "lgd_share",
            lgd_share,
            "must be in [0, 1] (0.5, or 0.9 when 60 % or more of the counterparty's assets are collateralised)",
        ));
    }
    if !unit(factor) {
        return Err(invalid("factor", factor, "must be in [0, 1]"));
    }
    Ok((lgd_share * (recoverables + 0.5 * risk_mitigation) - factor * collateral).max(0.0))
}

/// The capital requirement for type 1 exposures and its parts, from
/// [`type1`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Type1 {
    /// `V_inter` of Art 201(2), over every ordered pair of groups of equal
    /// probability of default.
    pub v_inter: f64,
    /// `V_intra` of Art 201(3), within each group.
    pub v_intra: f64,
    /// The standard deviation of the loss distribution, `sqrt(V_inter +
    /// V_intra)`.
    pub sigma: f64,
    /// The sum of the losses given default of all the type 1 exposures.
    pub total_lgd: f64,
    /// `SCR_def,1` of Art 200: `3 sigma` if `sigma` is at most 7 % of
    /// `total_lgd`, `5 sigma` if at most 20 %, and `total_lgd` otherwise.
    pub scr: f64,
}

/// The capital requirement for counterparty default risk on type 1
/// exposures (Arts 200 and 201), from each exposure's probability of default
/// and loss given default, `(pd, lgd)`.
///
/// The exposures are grouped by probability of default (exactly equal
/// values share a group); `TLGD_j` is the sum of the losses given default in
/// group `j` and `PD_j` its probability. Then
///
/// - `V_inter` is the sum over every ordered pair of groups `(j, k)`, the
///   diagonal included (so each pair of different groups counts twice), of
///   `PD_j (1 - PD_j) PD_k (1 - PD_k) / (1.25 (PD_j + PD_k) - PD_j PD_k)
///   TLGD_j TLGD_k`, a term whose two probabilities are both 0 being 0
///   (EIOPA Q&A 2530);
/// - `V_intra` is the sum over the groups of `1.5 PD_j (1 - PD_j) / (2.5 -
///   PD_j)` times the sum of the squared losses given default in group `j`;
/// - `SCR_def,1` is `3 sigma` if `sigma = sqrt(V_inter + V_intra)` is at
///   most 7 % of the total loss given default, `5 sigma` if at most 20 %,
///   and the total otherwise.
///
/// For one exposure `V_inter + V_intra = PD (1 - PD) LGD^2`. No exposures,
/// or only ones with a zero probability or a zero loss given default, give
/// `sigma = 0` and a charge of 0 (`3 sigma`, since `0 <= 7 %` of the total).
/// An exposure whose probability of default is 1 adds nothing to `sigma`
/// but counts in the total, so a list of those alone also gives a charge of 0.
///
/// # Errors
///
/// A probability of default that is not in [0, 1] or a loss given default
/// that is not finite and non-negative.
///
/// ```
/// use prospicio_aggregate::solvency2::type1;
///
/// // One exposure of 100 at 0.2 %: sigma = sqrt(0.002 * 0.998) * 100,
/// // under 7 % of 100, so the charge is 3 sigma.
/// let t = type1(&[(0.002, 100.0)]).unwrap();
/// assert!((t.sigma - (0.002_f64 * 0.998).sqrt() * 100.0).abs() < 1e-12);
/// assert!((t.scr - 3.0 * t.sigma).abs() < 1e-12);
/// ```
pub fn type1(exposures: &[(f64, f64)]) -> Result<Type1> {
    // (PD_j, TLGD_j, sum of LGD^2), in order of first appearance.
    let mut groups: Vec<(f64, f64, f64)> = Vec::new();
    for &(pd, lgd) in exposures {
        if !(pd.is_finite() && (0.0..=1.0).contains(&pd)) {
            return Err(invalid("pd", pd, "must be in [0, 1]"));
        }
        if !(lgd.is_finite() && lgd >= 0.0) {
            return Err(invalid("lgd", lgd, "must be finite and non-negative"));
        }
        match groups.iter_mut().find(|g| g.0 == pd) {
            Some(g) => {
                g.1 += lgd;
                g.2 += lgd * lgd;
            }
            None => groups.push((pd, lgd, lgd * lgd)),
        }
    }
    let mut v_inter = 0.0;
    for &(pj, tj, _) in &groups {
        for &(pk, tk, _) in &groups {
            if pj == 0.0 && pk == 0.0 {
                continue;
            }
            let u = pj * (1.0 - pj) * pk * (1.0 - pk) / (1.25 * (pj + pk) - pj * pk);
            v_inter += u * tj * tk;
        }
    }
    let v_intra: f64 = groups
        .iter()
        .map(|&(p, _, sq)| 1.5 * p * (1.0 - p) / (2.5 - p) * sq)
        .sum();
    let sigma = (v_inter + v_intra).sqrt();
    let total_lgd: f64 = exposures.iter().map(|&(_, lgd)| lgd).sum();
    let scr = if sigma <= 0.07 * total_lgd {
        3.0 * sigma
    } else if sigma <= 0.2 * total_lgd {
        5.0 * sigma
    } else {
        total_lgd
    };
    Ok(Type1 {
        v_inter,
        v_intra,
        sigma,
        total_lgd,
        scr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn every_step_has_its_probability() {
        let want = [0.00002, 0.0001, 0.0005, 0.0024, 0.012, 0.042, 0.042];
        for (step, p) in want.into_iter().enumerate() {
            assert_eq!(probability_by_step(step as u8).unwrap(), p);
        }
        assert!(probability_by_step(7).is_err());
        assert!(probability_by_step(u8::MAX).is_err());
    }

    #[test]
    fn the_ratio_table_is_hit_exactly_and_clamped() {
        for (ratio, p) in BY_RATIO {
            assert_eq!(probability_by_ratio(ratio).unwrap(), p, "{ratio}");
        }
        assert_eq!(probability_by_ratio(1.97).unwrap(), 0.0001);
        assert_eq!(probability_by_ratio(150.0).unwrap(), 0.0001);
        assert_eq!(probability_by_ratio(f64::INFINITY).unwrap(), 0.0001);
        assert_eq!(probability_by_ratio(0.74).unwrap(), 0.042);
        assert_eq!(probability_by_ratio(0.0).unwrap(), 0.042);
        // Linear between the points: a quarter of the way from 100 % to 95 %.
        close(probability_by_ratio(0.9875).unwrap(), 0.005 + 0.25 * 0.007);
        assert!(probability_by_ratio(f64::NAN).is_err());
        assert!(probability_by_ratio(-0.1).is_err());
    }

    #[test]
    fn the_ratio_is_monotone() {
        let mut last = f64::INFINITY;
        for i in 0..=300 {
            let p = probability_by_ratio(f64::from(i) / 100.0).unwrap();
            assert!(p <= last, "{i}");
            last = p;
        }
    }

    #[test]
    fn the_lgd_follows_article_192() {
        assert_eq!(reinsurance_lgd(100.0, 40.0, 30.0, 0.5, 1.0).unwrap(), 30.0);
        assert_eq!(reinsurance_lgd(100.0, 40.0, 30.0, 0.5, 0.5).unwrap(), 45.0);
        close(
            reinsurance_lgd(100.0, 40.0, 30.0, 0.9, 0.18).unwrap(),
            0.9 * 120.0 - 0.18 * 30.0,
        );
        assert_eq!(reinsurance_lgd(100.0, 40.0, 200.0, 0.5, 1.0).unwrap(), 0.0);
        assert_eq!(reinsurance_lgd(-10.0, 0.0, 0.0, 0.5, 1.0).unwrap(), 0.0);
        assert!(reinsurance_lgd(f64::NAN, 0.0, 0.0, 0.5, 1.0).is_err());
        assert!(reinsurance_lgd(1.0, f64::INFINITY, 0.0, 0.5, 1.0).is_err());
        assert!(reinsurance_lgd(1.0, 0.0, -1.0, 0.5, 1.0).is_err());
        assert!(reinsurance_lgd(1.0, 0.0, 0.0, 1.5, 1.0).is_err());
        assert!(reinsurance_lgd(1.0, 0.0, 0.0, 0.5, -0.1).is_err());
        assert!(reinsurance_lgd(1.0, 0.0, 0.0, 0.5, f64::NAN).is_err());
    }

    #[test]
    fn one_exposure_by_hand() {
        // PD 0.2 %, LGD 100: V_inter = p^2 (1-p)^2 / (2.5 p - p^2) 100^2,
        // V_intra = 1.5 p (1-p) / (2.5 - p) 100^2, and their sum
        // p (1 - p) 100^2 = 19.96.
        let p: f64 = 0.002;
        let t = type1(&[(p, 100.0)]).unwrap();
        close(t.v_inter, p * (1.0 - p) * (1.0 - p) / (2.5 - p) * 1e4);
        close(t.v_intra, 1.5 * p * (1.0 - p) / (2.5 - p) * 1e4);
        close(t.v_inter + t.v_intra, 19.96);
        close(t.sigma, 19.96_f64.sqrt());
        assert_eq!(t.total_lgd, 100.0);
        assert_eq!(t.scr, 3.0 * t.sigma);
    }

    #[test]
    fn the_three_branches_of_article_200() {
        // One exposure: sigma / LGD = sqrt(p (1 - p)).
        let three = type1(&[(0.002, 100.0)]).unwrap(); // 4.5 %
        assert_eq!(three.scr, 3.0 * three.sigma);
        let five = type1(&[(0.012, 50.0)]).unwrap(); // 10.9 %
        assert_eq!(five.scr, 5.0 * five.sigma);
        let all = type1(&[(0.042, 10.0)]).unwrap(); // 20.06 %
        assert_eq!(all.scr, 10.0);
    }

    #[test]
    fn groups_are_by_equal_probability() {
        // Splitting an exposure in two at the same probability keeps
        // V_inter (bilinear in the group's total) and lowers V_intra.
        let one = type1(&[(0.005, 80.0), (0.0024, 70.0)]).unwrap();
        let split = type1(&[(0.005, 30.0), (0.0024, 70.0), (0.005, 50.0)]).unwrap();
        close(split.v_inter, one.v_inter);
        assert!(split.v_intra < one.v_intra);
        let sq = 30.0_f64.powi(2) + 50.0_f64.powi(2);
        close(
            split.v_intra - one.v_intra,
            1.5 * 0.005 * 0.995 / 2.495 * (sq - 6400.0),
        );
    }

    #[test]
    fn zero_probabilities_add_no_variance() {
        let t = type1(&[(0.0, 10.0), (0.0, 20.0)]).unwrap();
        assert_eq!((t.v_inter, t.v_intra, t.sigma, t.scr), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(t.total_lgd, 30.0);
        // A zero group beside a positive one: the cross terms are 0.
        let with = type1(&[(0.0, 10.0), (0.01, 20.0)]).unwrap();
        let without = type1(&[(0.01, 20.0)]).unwrap();
        close(with.v_inter, without.v_inter);
        close(with.v_intra, without.v_intra);
        assert_eq!(with.total_lgd, 30.0);
        // Certain default has no variance either.
        let sure = type1(&[(1.0, 50.0)]).unwrap();
        assert_eq!((sure.sigma, sure.scr, sure.total_lgd), (0.0, 0.0, 50.0));
    }

    #[test]
    fn no_exposures_give_no_charge() {
        let t = type1(&[]).unwrap();
        assert_eq!(
            t,
            Type1 {
                v_inter: 0.0,
                v_intra: 0.0,
                sigma: 0.0,
                total_lgd: 0.0,
                scr: 0.0
            }
        );
    }

    #[test]
    fn bad_exposures_are_refused() {
        assert!(type1(&[(1.5, 1.0)]).is_err());
        assert!(type1(&[(-0.1, 1.0)]).is_err());
        assert!(type1(&[(f64::NAN, 1.0)]).is_err());
        assert!(type1(&[(0.01, -1.0)]).is_err());
        assert!(type1(&[(0.01, f64::INFINITY)]).is_err());
    }
}
