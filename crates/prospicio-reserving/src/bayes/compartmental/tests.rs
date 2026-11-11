//! Unit tests of the compartmental model.

use super::*;
use prospicio_bayes::nuts::{PosteriorDraws, Sampler};
use prospicio_core::{Grain, Lag, Period, StreamRng};
use prospicio_math::special::norm_cdf;
use prospicio_prob::{Distribution, KeyValue, PredictiveDistribution, Provenance};

use super::super::dual::{CHUNK, gradient};
use super::super::laplace::Posterior;
use super::super::one_year::{BayesOneYear, LightRefit, OneYearRefit};
use super::super::scalar::Scalar;
use super::super::{Stream, derive_seed};
use super::{data::*, fit::*, model::*, priors::*, results::*, settings::*};
use crate::bayes::laplace::tests::assert_gradient;
use crate::dependence::SegmentDependence;
use crate::error::Error;
use crate::triangle::Long;
use crate::triangle::Triangle;
use crate::{DevelopmentColumn, Month};
use prospicio_math::special::norm_pdf;

/// Company 337 workers' compensation, the papers' training triangle
/// (validation/data/clrd_wkcomp_337.csv, calendar years up to 1996):
/// cumulative paid and incurred (with bulk and IBNR), USD thousands.
pub(super) const PAID: [&[f64]; 9] = [
    &[
        9558., 22778., 33298., 40348., 45146., 48048., 49782., 50623., 51812.,
    ],
    &[
        7913., 19472., 29622., 36816., 40975., 43302., 44707., 45871.,
    ],
    &[8744., 24302., 35406., 43412., 48057., 50897., 52879.],
    &[13301., 32950., 47201., 56394., 61650., 65039.],
    &[11424., 29086., 42034., 50910., 56406.],
    &[11792., 27161., 38229., 46722.],
    &[11194., 26893., 38488.],
    &[12550., 31604.],
    &[13194.],
];
pub(super) const INCURRED: [&[f64]; 9] = [
    &[
        62679., 64000., 65607., 65292., 62250., 61185., 59387., 57138., 53473.,
    ],
    &[
        62058., 56660., 56598., 56831., 55294., 53481., 51379., 48446.,
    ],
    &[63955., 61523., 63166., 61402., 59474., 57613., 57161.],
    &[73918., 75094., 73188., 71199., 71056., 70831.],
    &[77143., 75133., 73284., 73155., 68284.],
    &[79925., 78263., 78163., 68546.],
    &[73628., 73554., 70736.],
    &[69521., 80281.],
    &[69720.],
];
/// Direct earned premium, 1988 to 1997.
pub(super) const PREMIUM: [f64; 10] = [
    104437., 88883., 85956., 99339., 104897., 119427., 110784., 77731., 63646., 48052.,
];

/// The training triangle with columns `paid`, `incurred`, `os`
/// (incurred - paid) and `premium`, and 1997 with its premium only when
/// `with_1997`.
pub(super) fn triangle(with_1997: bool) -> Triangle {
    let (mut origin, mut ages) = (vec![], vec![]);
    let (mut paid, mut inc, mut os, mut prem) = (vec![], vec![], vec![], vec![]);
    for (w, row) in PAID.iter().enumerate() {
        for (d, &p) in row.iter().enumerate() {
            origin.push(Month::january(1988 + w as i32));
            ages.push(12 * (d as Lag + 1));
            paid.push(p);
            inc.push(INCURRED[w][d]);
            os.push(INCURRED[w][d] - p);
            prem.push(PREMIUM[w]);
        }
    }
    if with_1997 {
        origin.push(Month::january(1997));
        ages.push(12);
        paid.push(f64::NAN);
        inc.push(f64::NAN);
        os.push(f64::NAN);
        prem.push(PREMIUM[9]);
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[
            ("paid", &paid),
            ("incurred", &inc),
            ("os", &os),
            ("premium", &prem),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

pub(super) fn preset() -> Compartmental {
    Compartmental::new(CompartmentalPriors::gesmann_morris())
}

pub(super) fn quick(seed: u64) -> Compartmental {
    Compartmental {
        sampler: Sampler {
            chains: 2,
            tune: 100,
            draws: 100,
            target_accept: 0.9,
            ..preset().sampler
        },
        n_sims: 500,
        seed,
        ..preset()
    }
}

pub(super) fn incurred() -> Outstanding<'static> {
    Outstanding::FromIncurred("incurred")
}

pub(super) fn model(ordered: bool) -> Model {
    model_with(
        CompartmentalPriors::gesmann_morris(),
        ordered,
        Response::MODEL_1,
    )
}

fn model_with(priors: CompartmentalPriors, ordered: bool, response: Response) -> Model {
    let tri = triangle(true);
    let data = Data::new(
        &tri,
        "paid",
        incurred(),
        "premium",
        Fitted::PaidAndOutstanding,
    )
    .unwrap();
    Model::new(&data, priors, ordered, response).unwrap()
}

/// Gesmann and Morris' model 2 on the training triangle.
pub(super) fn model_2(ordered: bool) -> Model {
    model_with(
        CompartmentalPriors::gesmann_morris_2(),
        ordered,
        Response::MODEL_2,
    )
}

pub(super) fn preset_2() -> Compartmental {
    Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
}

/// The one-year defaults (decision 20 of 2026-10-16, model 2's changed
/// by the user's decision of 2026-10-21): model 1 moment-matches, model
/// 2 (development effects) refits its flagged simulations with the
/// light refit, and reweights only with a warning when that is turned
/// off; a refit mode the settings choose is kept without one; both run
/// at most 2,000 one-year simulations unless told otherwise.
#[test]
fn the_one_year_defaults_follow_the_model() {
    let default = BayesOneYear::default();
    let (one, sims, warning) = preset().one_year_defaults(&default);
    assert_eq!(one.refit, Some(OneYearRefit::MomentMatchThenRefit));
    assert_eq!((sims, warning), (Compartmental::ONE_YEAR_SIMS, None));
    let (two, sims, warning) = preset_2().one_year_defaults(&default);
    assert_eq!(two.refit, Some(OneYearRefit::RefitFlagged));
    assert_eq!(two.light_refit, Some(LightRefit::default()));
    assert_eq!((sims, warning), (2_000, None));
    // Turning the light refit off reweights only, with the warning.
    let off = BayesOneYear {
        no_light_refit: true,
        ..default
    };
    let (two, _, warning) = preset_2().one_year_defaults(&off);
    assert_eq!(
        (two.refit, two.light_refit),
        (Some(OneYearRefit::ReweightOnly), None)
    );
    let warning = warning.expect("the opt-out view warns");
    assert!(warning.contains("approximate") && warning.contains("0.4 sd"));
    // Off with a refit chosen refits with the fit's own sampler.
    let (two, _, warning) = preset_2().one_year_defaults(&BayesOneYear {
        refit: Some(OneYearRefit::RefitFlagged),
        ..off
    });
    assert_eq!((two.light_refit, warning), (None, None));
    // Model 1 keeps its own sampler by default.
    assert_eq!(preset().one_year_defaults(&default).0.light_refit, None);
    // Refits stay a choice: an explicit mode, even the other models'
    // default, is used as it is and is not approximate.
    for refit in [
        OneYearRefit::MomentMatchThenRefit,
        OneYearRefit::RefitFlagged,
        OneYearRefit::ReweightOnly,
    ] {
        let chosen = BayesOneYear {
            refit: Some(refit),
            n_sims: Some(300),
            ..default
        };
        let (two, sims, warning) = preset_2().one_year_defaults(&chosen);
        assert_eq!((two.refit, sims, warning), (Some(refit), 300, None));
        assert_eq!(two.light_refit, None);
    }
    // A light refit is a choice to refit: model 2 refits its flagged
    // simulations, without the approximate warning (decision 24); model
    // 1 keeps moment matching.
    let light = BayesOneYear {
        light_refit: Some(LightRefit::default()),
        ..default
    };
    let (two, _, warning) = preset_2().one_year_defaults(&light);
    assert_eq!(
        (two.refit, warning),
        (Some(OneYearRefit::RefitFlagged), None)
    );
    let (one, _, warning) = preset().one_year_defaults(&light);
    assert_eq!(
        (one.refit, warning),
        (Some(OneYearRefit::MomentMatchThenRefit), None)
    );
    // A model with fewer simulations runs that many; more are capped.
    let small = Compartmental {
        n_sims: 500,
        ..preset_2()
    };
    assert_eq!(small.one_year_defaults(&default).1, 500);
    let large = Compartmental {
        n_sims: 50_000,
        ..preset()
    };
    assert_eq!(large.one_year_defaults(&default).1, 2_000);
    let more = BayesOneYear {
        n_sims: Some(20_000),
        ..default
    };
    assert_eq!(large.one_year_defaults(&more).1, 20_000);
}

/// Points around the paper's posterior (model 1's, or model 2's when
/// the model has effects by cell): `theta` at the printed means with a
/// deterministic spread.
pub(super) fn points(model: &Model) -> Vec<Vec<f64>> {
    let two = model.cellwise;
    (0..20)
        .map(|k| {
            let mut x = vec![0.0; model.dim()];
            let wobble = |j: usize| 0.3 * ((7 * k + 3 * j) as f64).sin();
            x[B_KER] = match (model.ordered, two) {
                (true, _) => 0.2 + wobble(0),
                (false, false) => -5.4 + wobble(0),
                (false, true) => -1.31 + wobble(0),
            };
            let (b, tau, rho, ls) = if two {
                ([-5.07, 1.54, -1.45], [0.69, 0.73], 0.37, [-1.78, -1.89])
            } else {
                ([-8.49, 1.44, -1.36], [0.64, 0.74], 0.6, [8.25, 6.66])
            };
            x[B_KP] = b[0] + wobble(1);
            x[B_RLR] = b[1] + wobble(2);
            x[B_RRF] = b[2] + wobble(3);
            x[LOG_TAU] = tau[0].ln() + wobble(4);
            x[LOG_TAU + 1] = tau[1].ln() + wobble(5);
            x[ATANH_RHO] = rho + wobble(6);
            let z0 = model.layout.z0;
            // Model 2's other effect sds: 0.26, 2.23 by origin, then
            // 0.27, 0.45, 0.35, 1.11 by development year (the printed
            // m2fit).
            let sds = [0.26f64, 2.23, 0.27, 0.45, 0.35, 1.11];
            for (j, l) in x[ATANH_RHO + 1..z0].iter_mut().enumerate() {
                *l = sds[j].ln() + wobble(7 + j);
            }
            let ls0 = model.log_sigma0();
            let spread = if two { 1.0 } else { 3.0 };
            for (j, z) in x[z0..ls0].iter_mut().enumerate() {
                *z = spread * wobble(13 + j);
            }
            for (s, paid) in model.response.sigma_types().enumerate() {
                x[ls0 + s] = ls[usize::from(paid)] + wobble(ls0 + s);
            }
            x
        })
        .collect()
}

#[test]
fn the_layout_on_the_papers_training_triangle() {
    let m = model(false);
    assert_eq!(m.dim(), 27);
    assert_eq!((m.cells.len(), m.n_outstanding), (90, 45));
    assert_eq!(m.level[9], None);
    assert_eq!(m.n_levels(), 9);
    assert_eq!(m.times.len(), 10);
    assert_eq!(m.n_observed, 9);
    let names = m.names();
    assert_eq!(names.len(), 6 + 18 + 9);
    assert_eq!(names[6], "RLR[1988]");
    assert_eq!(names[15], "RRF[1988]");
    // Outstanding from incurred less paid.
    assert_eq!(m.cells[0].y, 62679. - 9558.);
    assert!(m.cells[45].paid && m.cells[45].y == 9558.);
}

#[test]
fn the_dual_gradient_matches_central_differences() {
    for ordered in [false, true] {
        let m = model(ordered);
        for x in points(&m) {
            assert_gradient(|t, g| m.log_posterior(t, g), &x);
        }
        // Model 2 (87 parameters): fewer points, for the debug build's
        // time.
        let m = model_2(ordered);
        for x in points(&m).iter().step_by(4) {
            assert_gradient(|t, g| m.log_posterior(t, g), x);
        }
    }
    // Model 1's hierarchy with model 2's response (the fast path with
    // increments on the log scale).
    let m = model_with(
        CompartmentalPriors::gesmann_morris(),
        false,
        Response::MODEL_2,
    );
    for x in points(&m).iter().step_by(4) {
        let mut x = x.clone();
        let ls = m.log_sigma0();
        (x[ls], x[ls + 1]) = (-1.8, -1.9);
        assert_gradient(|t, g| m.log_posterior(t, g), &x);
    }
}

#[test]
fn the_density_is_the_model_written_out() {
    // At one point, against the model written out in f64 with the
    // paper's form of the closed form (away from k_er = k_p).
    let m = model(false);
    let x = &points(&m)[3];
    let half_t = |l: f64, s: f64| l - 5.5 * ((l.exp() / s).powi(2) / 10.0).ln_1p();
    let cauchy = |v: f64| -((v / 1000.0).powi(2)).ln_1p();
    let (a, b) = (3.0 * (0.1 * x[0]).exp(), (0.1 * x[1]).exp());
    let rho = x[6].tanh();
    let ls = m.log_sigma0();
    let mut f = -0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
        + half_t(x[4], 0.2)
        + half_t(x[5], 0.1)
        + (1.0 - rho * rho).ln()
        + cauchy(x[ls])
        + cauchy(x[ls + 1]);
    for z in &x[m.layout.z0..ls] {
        f -= 0.5 * z * z;
    }
    for c in &m.cells {
        let j = m.level[c.origin].unwrap();
        let (z1, z2) = (x[m.layout.z0 + 2 * j], x[m.layout.z0 + 2 * j + 1]);
        let u1 = x[4].exp() * z1;
        let u2 = x[5].exp() * (rho * z1 + (1.0 - rho * rho).sqrt() * z2);
        let rlr = 0.7 * (0.2 * (x[2] + u1)).exp();
        let rrf = 0.8 * (0.1 * (x[3] + u2)).exp();
        let t = c.t;
        let os = rlr * a / (a - b) * ((-b * t).exp() - (-a * t).exp());
        let pd = rlr * rrf / (a - b) * (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp()));
        let (mean, l) = if c.paid { (pd, x[ls + 1]) } else { (os, x[ls]) };
        let r = (c.y - PREMIUM[c.origin] * mean) / l.exp();
        f -= l + 0.5 * r * r;
    }
    let mut g = vec![0.0; m.dim()];
    let got = m.log_posterior(x, &mut g);
    assert!((got - f).abs() <= 1e-12 * f.abs(), "{got} vs {f}");
}

/// A model of the training triangle fitted to paid and incurred.
pub(super) fn model_incurred(
    priors: CompartmentalPriors,
    ordered: bool,
    response: Response,
) -> Model {
    let tri = triangle(true);
    let data = Data::new(&tri, "paid", incurred(), "premium", Fitted::PaidAndIncurred).unwrap();
    Model::new(
        &data,
        priors,
        ordered,
        Response {
            incurred: true,
            ..response
        },
    )
    .unwrap()
}

#[test]
fn the_incurred_cells_are_the_incurred_column_or_the_outstanding_plus_paid() {
    let tri = triangle(true);
    let direct = Data::new(&tri, "paid", incurred(), "premium", Fitted::PaidAndIncurred).unwrap();
    let summed = Data::new(
        &tri,
        "paid",
        Outstanding::Column("os"),
        "premium",
        Fitted::PaidAndIncurred,
    )
    .unwrap();
    assert_eq!(direct.outstanding.len(), 45);
    assert_eq!(direct.outstanding[0].2, 62679.);
    assert_eq!(direct.outstanding.len(), summed.outstanding.len());
    for (a, b) in direct.outstanding.iter().zip(&summed.outstanding) {
        assert_eq!((a.0, a.1), (b.0, b.1));
        assert!((a.2 - b.2).abs() < 1e-9 * a.2.abs(), "{a:?} {b:?}");
    }
}

#[test]
fn the_incurred_density_is_the_model_written_out() {
    // The same hand computation as for paid and outstanding, with the
    // second mean RLR (OS + RRF PD) times the premium.
    let m = model_incurred(
        CompartmentalPriors::gesmann_morris(),
        false,
        Response::MODEL_1,
    );
    assert!(m.response.incurred);
    assert_eq!(m.cells[0].y, 62679.);
    let x = &points(&m)[3];
    let half_t = |l: f64, s: f64| l - 5.5 * ((l.exp() / s).powi(2) / 10.0).ln_1p();
    let cauchy = |v: f64| -((v / 1000.0).powi(2)).ln_1p();
    let (a, b) = (3.0 * (0.1 * x[0]).exp(), (0.1 * x[1]).exp());
    let rho = x[6].tanh();
    let ls = m.log_sigma0();
    let mut f = -0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
        + half_t(x[4], 0.2)
        + half_t(x[5], 0.1)
        + (1.0 - rho * rho).ln()
        + cauchy(x[ls])
        + cauchy(x[ls + 1]);
    for z in &x[m.layout.z0..ls] {
        f -= 0.5 * z * z;
    }
    for c in &m.cells {
        let j = m.level[c.origin].unwrap();
        let (z1, z2) = (x[m.layout.z0 + 2 * j], x[m.layout.z0 + 2 * j + 1]);
        let u1 = x[4].exp() * z1;
        let u2 = x[5].exp() * (rho * z1 + (1.0 - rho * rho).sqrt() * z2);
        let rlr = 0.7 * (0.2 * (x[2] + u1)).exp();
        let rrf = 0.8 * (0.1 * (x[3] + u2)).exp();
        let t = c.t;
        let os = rlr * a / (a - b) * ((-b * t).exp() - (-a * t).exp());
        let pd = rlr * rrf / (a - b) * (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp()));
        let (mean, l) = if c.paid {
            (pd, x[ls + 1])
        } else {
            (os + pd, x[ls])
        };
        let r = (c.y - PREMIUM[c.origin] * mean) / l.exp();
        f -= l + 0.5 * r * r;
    }
    let mut g = vec![0.0; m.dim()];
    let got = m.log_posterior(x, &mut g);
    assert!((got - f).abs() <= 1e-12 * f.abs(), "{got} vs {f}");
}

#[test]
fn the_incurred_gradient_matches_central_differences() {
    for ordered in [false, true] {
        let m = model_incurred(
            CompartmentalPriors::gesmann_morris(),
            ordered,
            Response::MODEL_1,
        );
        for x in points(&m) {
            assert_gradient(|t, g| m.log_posterior(t, g), &x);
        }
        // Model 2's hierarchy and response on incurred.
        let m = model_incurred(
            CompartmentalPriors::gesmann_morris_2(),
            ordered,
            Response::MODEL_2,
        );
        for x in points(&m).iter().step_by(4) {
            assert_gradient(|t, g| m.log_posterior(t, g), x);
        }
    }
}

#[test]
fn the_incurred_cellwise_gradient_is_the_generic_densitys() {
    let m = model_incurred(
        CompartmentalPriors::gesmann_morris_2(),
        false,
        Response::MODEL_2,
    );
    assert!(m.cellwise);
    for x in points(&m).iter().step_by(3) {
        let mut want = vec![0.0; m.dim()];
        let f = gradient::<CHUNK>(x, &mut want, |t| m.log_density(t));
        let mut got = vec![0.0; m.dim()];
        let g = m.cellwise_gradient(x, &mut got);
        let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        assert!((g - f).abs() <= 1e-12 * f.abs());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-12 * scale);
        }
    }
}

#[test]
fn the_incurred_mean_is_the_outstanding_plus_the_paid_mean() {
    // Cell by cell: incurred = outstanding + cumulative paid, under both
    // paid responses (the paid cumulative even when paid is incremental).
    let (os_model, inc_model) = (
        model_with(
            CompartmentalPriors::gesmann_morris_2(),
            false,
            Response::MODEL_2,
        ),
        model_incurred(
            CompartmentalPriors::gesmann_morris_2(),
            false,
            Response::MODEL_2,
        ),
    );
    let c = os_model.cells[0];
    assert!(!c.paid);
    let th = [1.2, 0.3, 0.8, 0.9];
    let os = os_model.cell_mean(&c, th);
    let paid_cum = {
        let (_, pd) = one_os(th[KER], th[KP], c.t);
        c.w * th[RLR] * th[RRF] * pd
    };
    let inc = inc_model.cell_mean(&c, th);
    assert!((inc - (os + paid_cum)).abs() <= 1e-12 * inc);
}

/// A model of the training triangle fitted to one response alone
/// (`Fitted::PaidOnly` or `Fitted::IncurredOnly`), with `response`'s
/// likelihood, scale and paid response.
pub(super) fn model_one(
    fitted: Fitted,
    priors: CompartmentalPriors,
    ordered: bool,
    response: Response,
) -> Model {
    let tri = triangle(true);
    let data = Data::new(&tri, "paid", incurred(), "premium", fitted).unwrap();
    let settings = Compartmental {
        fitted,
        likelihood: response.likelihood,
        scale: if response.loss_ratio {
            Scale::LossRatio
        } else {
            Scale::Amount
        },
        paid_response: if response.incremental {
            PaidResponse::Incremental
        } else {
            PaidResponse::Cumulative
        },
        ..preset()
    };
    Model::new(&data, priors, ordered, settings.response()).unwrap()
}

#[test]
fn the_one_response_layout_on_the_papers_training_triangle() {
    let pr = CompartmentalPriors::gesmann_morris();
    let paid = model_one(Fitted::PaidOnly, pr, true, Response::MODEL_1);
    let inc = model_one(Fitted::IncurredOnly, pr, true, Response::MODEL_1);
    // One sigma fewer than paid and outstanding's 27.
    for m in [&paid, &inc] {
        assert_eq!(m.dim(), 26);
        assert_eq!(m.cells.len(), 45);
        assert_eq!(m.n_levels(), 9);
        assert_eq!(m.level[9], None);
        let names = m.names();
        assert_eq!(names.len(), 5 + 18 + 8);
        assert_eq!(names[5], "RLR[1988]");
        assert_eq!(names[14], "RRF[1988]");
        assert_eq!(m.sampled_names().len(), m.dim());
        assert_eq!(m.natural(&points(m)[0]).len(), names.len());
    }
    assert_eq!(paid.n_outstanding, 0);
    assert!(paid.cells.iter().all(|c| c.paid));
    assert_eq!(paid.names()[4], "sigma_pd");
    assert_eq!(paid.names().last().unwrap(), "log_sigma_pd");
    assert_eq!(inc.n_outstanding, 45);
    assert!(inc.cells.iter().all(|c| !c.paid));
    assert_eq!(inc.cells[0].y, 62679.);
    assert_eq!(inc.names()[4], "sigma_os");
    assert_eq!(inc.names().last().unwrap(), "log_sigma_os");
    assert!(inc.response.incurred);
    // Incurred alone keeps each origin's latest paid, from the paid
    // column, for the reserves.
    assert_eq!(inc.latest_paid[3], PAID[3][5]);
    assert_eq!(paid.latest_paid, inc.latest_paid);
    // Model 2's hierarchy with incremental paid alone, and incurred
    // alone under the log-normal likelihood.
    let pr2 = CompartmentalPriors::gesmann_morris_2();
    let paid_2 = model_one(Fitted::PaidOnly, pr2, true, Response::MODEL_2);
    assert!(paid_2.cellwise && paid_2.cells.iter().all(|c| c.paid));
    assert!(paid_2.cells.iter().filter(|c| c.t_before.is_some()).count() == 36);
    let inc_2 = model_one(Fitted::IncurredOnly, pr2, true, Response::MODEL_2);
    assert_eq!(inc_2.dim(), model_2(true).dim() - 1);
}

/// The log density of one response alone, against the model written out
/// in f64 under ordered rates: `k_p = e^(0.1 x_1)`, `k_er = k_p +
/// e^(x_0)` with `b_ker`'s normal prior and the Jacobian `x_0 - ln
/// k_er`, one `log σ` with its prior, and the cells of the one type.
#[test]
fn the_one_response_density_is_the_model_written_out() {
    let pr = CompartmentalPriors::gesmann_morris();
    for fitted in [Fitted::PaidOnly, Fitted::IncurredOnly] {
        let m = model_one(fitted, pr, true, Response::MODEL_1);
        let x = &points(&m)[3];
        let half_t = |l: f64, s: f64| l - 5.5 * ((l.exp() / s).powi(2) / 10.0).ln_1p();
        let cauchy = |v: f64| -((v / 1000.0).powi(2)).ln_1p();
        let b = (0.1 * x[1]).exp();
        let a = b + x[0].exp();
        let b_ker = (a / 3.0).ln() / 0.1;
        let rho = x[6].tanh();
        let ls = m.log_sigma0();
        assert_eq!(ls + 1, m.dim());
        let mut f = -0.5 * (b_ker * b_ker + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) + x[0]
            - a.ln()
            + half_t(x[4], 0.2)
            + half_t(x[5], 0.1)
            + (1.0 - rho * rho).ln()
            + cauchy(x[ls]);
        for z in &x[m.layout.z0..ls] {
            f -= 0.5 * z * z;
        }
        for c in &m.cells {
            let j = m.level[c.origin].unwrap();
            let (z1, z2) = (x[m.layout.z0 + 2 * j], x[m.layout.z0 + 2 * j + 1]);
            let u1 = x[4].exp() * z1;
            let u2 = x[5].exp() * (rho * z1 + (1.0 - rho * rho).sqrt() * z2);
            let rlr = 0.7 * (0.2 * (x[2] + u1)).exp();
            let rrf = 0.8 * (0.1 * (x[3] + u2)).exp();
            let t = c.t;
            let os = rlr * a / (a - b) * ((-b * t).exp() - (-a * t).exp());
            let pd =
                rlr * rrf / (a - b) * (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp()));
            let mean = if fitted == Fitted::PaidOnly {
                assert!(c.paid);
                pd
            } else {
                assert!(!c.paid);
                os + pd
            };
            let r = (c.y - PREMIUM[c.origin] * mean) / x[ls].exp();
            f -= x[ls] + 0.5 * r * r;
        }
        let mut g = vec![0.0; m.dim()];
        let got = m.log_posterior(x, &mut g);
        assert!(
            (got - f).abs() <= 1e-12 * f.abs(),
            "{fitted:?}: {got} vs {f}"
        );
    }
}

#[test]
fn the_one_response_gradient_matches_central_differences() {
    for fitted in [Fitted::PaidOnly, Fitted::IncurredOnly] {
        for ordered in [true, false] {
            let m = model_one(
                fitted,
                CompartmentalPriors::gesmann_morris(),
                ordered,
                Response::MODEL_1,
            );
            for x in points(&m).iter().step_by(2) {
                assert_gradient(|t, g| m.log_posterior(t, g), x);
            }
            // Model 2's hierarchy and response (effects by cell).
            let m = model_one(
                fitted,
                CompartmentalPriors::gesmann_morris_2(),
                ordered,
                Response::MODEL_2,
            );
            for x in points(&m).iter().step_by(5) {
                assert_gradient(|t, g| m.log_posterior(t, g), x);
                // The hand-written chain rule is the generic dual pass's.
                let mut want = vec![0.0; m.dim()];
                let f = gradient::<CHUNK>(x, &mut want, |t| m.log_density(t));
                let mut got = vec![0.0; m.dim()];
                let g = m.cellwise_gradient(x, &mut got);
                let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                assert!((g - f).abs() <= 1e-12 * f.abs());
                for (a, b) in got.iter().zip(&want) {
                    assert!((a - b).abs() <= 1e-12 * scale);
                }
            }
        }
    }
}

/// Under incurred alone the future paid cells of an origin with data are
/// their expected values given the draw, whatever the random numbers,
/// and its future incurred cells are drawn with the incurred sigma;
/// under paid alone the square has no future outstanding cell.
#[test]
fn the_one_response_simulations() {
    let pr = CompartmentalPriors::gesmann_morris();
    let tri = triangle(true);
    let inc = model_one(Fitted::IncurredOnly, pr, true, Response::MODEL_1);
    let data = Data::new(&tri, "paid", incurred(), "premium", Fitted::IncurredOnly).unwrap();
    let future = data.future_cells();
    assert_eq!(future.outstanding.len(), future.paid.len());
    let nat = inc.natural(&points(&inc)[0]);
    let width = future.paid.len() + future.outstanding.len();
    let (mut a, mut b) = (vec![0.0; width], vec![0.0; width]);
    inc.simulate(&nat, &future, &mut StreamRng::new(1, 0), &mut a);
    inc.simulate(&nat, &future, &mut StreamRng::new(2, 0), &mut b);
    let (k_er, k_p) = (nat[0], nat[1]);
    let mut unused = StreamRng::new(0, 0);
    for (j, &(i, k)) in future.paid.iter().enumerate() {
        if inc.level[i].is_none() {
            continue;
        }
        assert_eq!(a[j], b[j], "paid of origin {i} at age index {k}");
        let th = inc.origin_values(&nat, i, &mut unused);
        let want = PREMIUM[i] * th[RLR] * th[RRF] * one_os(k_er, k_p, inc.times[k]).1;
        assert!((a[j] - want).abs() <= 1e-9 * want, "{} vs {want}", a[j]);
    }
    let n_paid = future.paid.len();
    assert!(a[n_paid..].iter().zip(&b[n_paid..]).all(|(x, y)| x != y));
    let paid = Data::new(&tri, "paid", incurred(), "premium", Fitted::PaidOnly).unwrap();
    assert!(paid.future_cells().outstanding.is_empty());
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_paid_and_incurred_fit_from_either_second_column() {
    let tri = triangle(true);
    let m = Compartmental {
        fitted: Fitted::PaidAndIncurred,
        ..quick(1)
    };
    let fits: Vec<_> = [incurred(), Outstanding::Column("os")]
        .into_iter()
        .map(|second| m.fit(&tri, "paid", second, "premium").unwrap())
        .collect();
    for fit in &fits {
        assert_eq!(fit.model.n_outstanding, 45);
        assert_eq!(fit.reserves.n_components(), 10);
        // The future incurred cells are above the future paid in the
        // mean (the 1990 origin at age 10).
        let key = vec![Period::year(1990).into(), KeyValue::Int(120)];
        let paid = fit.paid.marginal(&key).unwrap().mean();
        let inc = fit
            .outstanding
            .as_ref()
            .unwrap()
            .marginal(&key)
            .unwrap()
            .mean();
        assert!(inc > 0.9 * paid, "{inc} {paid}");
        assert!(fit.reserves.total().mean() > 0.0);
    }
    // The incurred column, and the outstanding column plus paid, are
    // the same data.
    assert_eq!(fits[0].posterior, fits[1].posterior);
}

/// The outputs of a fit of one response: under paid alone no
/// outstanding distribution or table, no outstanding cell in the loo
/// and no `sigma_os`; under incurred alone the future incurred cells and
/// no paid cell in the loo; reserves either way, and the same under
/// `fit_segments`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_outputs_of_one_response() {
    let tri = triangle(true);
    for fitted in [Fitted::PaidOnly, Fitted::IncurredOnly] {
        let m = Compartmental {
            fitted,
            rate_order: RateOrder::Ordered,
            ..quick(3)
        };
        let fit = m.fit(&tri, "paid", incurred(), "premium").unwrap();
        assert_eq!(fit.reserves.n_components(), 10);
        assert!(fit.reserves.total().mean() > 0.0);
        assert_eq!(fit.rate_swaps, 0);
        let loo = fit.loo().unwrap();
        let names = fit.posterior.names();
        let segments = m.fit_segments(&tri, "paid", incurred(), "premium").unwrap();
        if fitted == Fitted::PaidOnly {
            assert!(fit.outstanding.is_none() && fit.outstanding_table().is_err());
            assert_eq!((loo.n_outstanding, loo.elpd_outstanding), (0, 0.0));
            assert!(!names.iter().any(|n| n.contains("sigma_os")));
            assert!(segments.outstanding.is_none() && segments.outstanding_table().is_err());
            // The provenance records that the second column was not read.
            assert!(
                fit.reserves
                    .provenance()
                    .parameters
                    .iter()
                    .any(|(k, v)| k == "outstanding" && v == "not read")
            );
        } else {
            let os = fit.outstanding.as_ref().unwrap();
            assert_eq!(os.n_components(), fit.paid.n_components());
            assert_eq!(
                fit.outstanding_table().unwrap().values[0].1.len(),
                os.n_components()
            );
            assert_eq!((loo.n_outstanding, loo.elpd_paid), (45, 0.0));
            assert!(!names.iter().any(|n| n.contains("sigma_pd")));
            assert!(segments.outstanding.is_some());
        }
        assert_eq!(
            segments.reserves.n_components(),
            fit.reserves.n_components()
        );
    }
}

#[test]
fn the_means_are_invariant_under_the_rate_swap() {
    let mut rng = StreamRng::new(7, 0);
    let mut worst: f64 = 0.0;
    for _ in 0..200 {
        let mut u = |lo: f64, hi: f64| lo + (hi - lo) * rng.next_open01();
        let (k_er, k_p, rlr, rrf, t) = (
            u(0.05, 6.0),
            u(0.05, 6.0),
            u(0.3, 1.5),
            u(0.5, 1.5),
            u(0.1, 15.0),
        );
        let (os, pd) = one_os(k_er, k_p, t);
        let (os2, pd2) = one_os(k_p, k_er, t);
        let (rlr2, rrf2) = (rlr * k_er / k_p, rrf * k_p / k_er);
        for (x, y) in [(rlr * os, rlr2 * os2), (rlr * rrf * pd, rlr2 * rrf2 * pd2)] {
            let err = (x - y).abs() / x.abs();
            worst = worst.max(err);
            assert!(err <= 1e-12, "{k_er} {k_p} {t}: {x} vs {y}");
        }
    }
    println!("swap invariance: worst relative difference {worst:.1e}");
}

#[test]
fn the_data_start_is_the_population_least_squares() {
    let m = model(false);
    let start = m.start(CompartmentalStart::Data);
    // On the paper's side of the swap and near its posterior (k_er about
    // 1.75 and k_p about 0.43, so b_ker -5.4 and b_kp -8.5; log sigma
    // 8.25 and 6.66). Without origin effects the population curve fits
    // paid worse, so its log sigma_PD starts higher (about 8.7).
    let ls = m.log_sigma0();
    assert!(start[B_KER] > -9.0 && start[B_KER] < 0.0, "{start:?}");
    assert!(start[B_KP] > -10.0 && start[B_KP] < -7.0, "{start:?}");
    assert!((start[ls] - 8.25).abs() < 1.5, "{start:?}");
    assert!((start[ls + 1] - 6.66).abs() < 2.5, "{start:?}");
    assert!(start[m.layout.z0..ls].iter().all(|&z| z == 0.0));
    let centre = m.start(CompartmentalStart::PriorCentre);
    assert_eq!(&centre[..4], &[0.0; 4]);
    let ordered = model(true);
    let s = ordered.start(CompartmentalStart::Data);
    let (k_er, k_p, _) = ordered.rates(&s);
    assert!(k_er > k_p);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_seed_replays_whatever_the_threads() {
    let tri = triangle(true);
    let fit = |m: &Compartmental| m.fit(&tri, "paid", incurred(), "premium").unwrap();
    let a = fit(&quick(3));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let b = pool.install(|| fit(&quick(3)));
    assert_eq!(a.posterior, b.posterior);
    assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
    assert_eq!(a.paid.draw_matrix(), b.paid.draw_matrix());
    let c = fit(&quick(4));
    assert_ne!(a.posterior, c.posterior);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_fit_and_its_tables() {
    let tri = triangle(true);
    let fit = quick(1)
        .fit(&tri, "paid", Outstanding::Column("os"), "premium")
        .unwrap();
    assert_eq!(fit.origins.len(), 10);
    assert_eq!(fit.population_origins, [Period::year(1997)]);
    assert_eq!((fit.latest_paid[0], fit.latest_paid[9]), (51812., 0.0));
    assert_eq!(fit.reserves.n_components(), 10);
    // Future paid: 1988 at age 10, 1989 at 9 and 10, ..., 1997 at all 10.
    assert_eq!(fit.paid.n_components(), (1..=9).sum::<usize>() + 10);
    assert_eq!(
        fit.outstanding.as_ref().unwrap().n_components(),
        fit.paid.n_components()
    );
    // The reserve is the paid at age 10 less the latest paid.
    let key = vec![Period::year(1990).into(), KeyValue::Int(120)];
    let at_10 = fit.paid.marginal(&key).unwrap();
    let reserve = fit
        .reserves
        .marginal(&vec![Period::year(1990).into()])
        .unwrap();
    assert!((at_10.mean() - 52879. - reserve.mean()).abs() < 1e-6 * at_10.mean());
    let long = fit.to_long();
    assert_eq!(long.n_rows(), 10);
    assert_eq!(long.column("premium").unwrap(), PREMIUM);
    let ulr = long.column("ultimate_loss_ratio").unwrap();
    assert!(ulr.iter().all(|&v| v > 0.3 && v < 1.5), "{ulr:?}");
    let totals = fit.totals();
    assert_eq!(totals.n_rows(), 1);
    let sum: f64 = long.column("mean").unwrap().iter().sum();
    assert!((totals.column("mean").unwrap()[0] - sum).abs() < 1e-6 * sum);
    let table = fit.paid_table().unwrap();
    assert_eq!(table.n_rows(), fit.paid.n_components());
    assert_eq!(table.age.as_ref().unwrap()[0], 120);
    let q = |n| table.column(n).unwrap()[0];
    assert!(q("q05") < q("q50") && q("q50") < q("q95"));
    assert_eq!(
        fit.outstanding_table().unwrap().n_rows(),
        fit.outstanding.as_ref().unwrap().n_components()
    );
    assert_eq!(fit.summary().unwrap().len(), 33);
    let loo = fit.loo().unwrap();
    assert_eq!(loo.n_outstanding, 45);
    assert!(loo.elpd_outstanding.is_finite() && loo.elpd_paid.is_finite());
    let output = |pd: &PredictiveDistribution| {
        pd.provenance()
            .parameters
            .iter()
            .filter(|(k, _)| k == "output")
            .map(|(_, v)| v.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(output(&fit.reserves), ["reserves"]);
    assert_eq!(output(&fit.paid), ["paid"]);
}

#[test]
fn the_pattern_is_the_odes_paid_curve() {
    // The pattern from the named natural-scale draws, built here from
    // one_os alone: model 1's PD(t) / PD(10) at the population's rates,
    // and model 2's running sum of incremental means at the population's
    // rates with each observed age's development effects (none at 120
    // months, which has no observed cell), over its value at 120 months.
    for two in [false, true] {
        let m = if two { model_2(false) } else { model(false) };
        let names = m.names();
        for theta in points(&m) {
            let nat = m.natural(&theta);
            let named = |n: &str| nat[names.iter().position(|x| x == n).unwrap()];
            let mut shares = vec![0.0; 10];
            m.pattern(&nat, &mut shares);
            let (k_er, k_p) = (named("k_er"), named("k_p"));
            let mut expected = Vec::with_capacity(10);
            if two {
                let mut cumulative = 0.0;
                for k in 0..10 {
                    // Each multiplier exp(s v), s the prior's log sd.
                    let dev = |short: &str, sd: f64| {
                        names
                            .iter()
                            .position(|x| *x == format!("dev_{short}[{}]", 12 * (k + 1)))
                            .map_or(1.0, |j| (sd * nat[j]).exp())
                    };
                    let (er, p) = (k_er * dev("ker", 0.1), k_p * dev("kp", 0.1));
                    let scale = dev("RLR", 0.2) * dev("RRF", 0.1);
                    let t = (k + 1) as f64;
                    cumulative += scale * (one_os(er, p, t).1 - one_os(er, p, t - 1.0).1);
                    expected.push(cumulative);
                }
            } else {
                expected.extend((1..=10).map(|t| one_os(k_er, k_p, f64::from(t)).1));
            }
            assert_eq!(shares[9], 1.0);
            for (k, share) in shares.iter().enumerate() {
                let e = expected[k] / expected[9];
                assert!(
                    (share - e).abs() < 1e-12,
                    "model 2 {two}, age {k}: {share} {e}"
                );
            }
            // The paid curve rises, and so does a running sum of
            // positive increments: the pattern never falls nor passes 1.
            assert!(shares[0] > 0.0);
            assert!(shares.windows(2).all(|w| w[0] <= w[1]), "{shares:?}");
        }
    }
    // Model 1's last share is a share of the paid curve's limit, 1: the
    // pattern ends at the square's last age, not at the curve's limit.
    let m = model(false);
    let nat = m.natural(&points(&m)[0]);
    assert!(one_os(nat[KER], nat[KP], 10.0).1 < 1.0);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn each_simulations_pattern_is_its_posterior_draws() {
    let tri = triangle(true);
    let two = Compartmental {
        sampler: Sampler {
            chains: 2,
            tune: 60,
            draws: 60,
            target_accept: 0.9,
            ..preset_2().sampler
        },
        n_sims: 500,
        seed: 21,
        ..preset_2()
    };
    for (name, compartmental) in [("model 1", quick(21)), ("model 2", two)] {
        let fit = compartmental
            .fit(&tri, "paid", incurred(), "premium")
            .unwrap();
        let n_sims = compartmental.n_sims;
        assert_eq!(fit.patterns.n_sims(), n_sims);
        let patterns = fit.patterns.segment(&[]).unwrap();
        // The square's ages: ten, to 120 months, one more than the
        // triangle's paid.
        assert_eq!(
            patterns.ages(),
            (1..=10).map(|k| 12 * k).collect::<Vec<_>>()
        );
        assert_eq!(patterns.origin_grain(), Grain::Year);
        // One predictive run with the pattern columns after the cells,
        // the pattern computed from the draw each simulation took: the
        // fit's pattern row i is simulation i's.
        let future = fit.data.future_cells();
        let n_paid = future.paid.len();
        let n_future = n_paid + future.outstanding.len();
        let replay = |with_patterns: bool| {
            let width = n_future + if with_patterns { 10 } else { 0 };
            fit.posterior
                .predictive(
                    vec!["column".into()],
                    (0..width).map(|j| vec![KeyValue::Int(j as i64)]).collect(),
                    n_sims,
                    derive_seed(compartmental.seed, Stream::Predictive),
                    Provenance::new("replay"),
                    |nat, rng: &mut StreamRng, row| {
                        let (cells, shares) = row.split_at_mut(n_future);
                        fit.model.simulate(nat, &future, rng, cells);
                        if with_patterns {
                            fit.model.pattern(nat, shares);
                        }
                    },
                )
                .unwrap()
        };
        let with = replay(true);
        for (i, row) in with.draw_matrix().chunks_exact(n_future + 10).enumerate() {
            assert_eq!(&row[n_future..], patterns.draw(i), "{name} simulation {i}");
        }
        // The same run without the pattern columns, as the fit drew it
        // before patterns: the fit's cells are bit for bit these, so
        // writing the patterns moved no random number.
        let without = replay(false);
        let os = fit.outstanding.as_ref().unwrap();
        let n_os = n_future - n_paid;
        for (i, row) in without.draw_matrix().chunks_exact(n_future).enumerate() {
            let paid = &fit.paid.draw_matrix()[i * n_paid..(i + 1) * n_paid];
            let outstanding = &os.draw_matrix()[i * n_os..(i + 1) * n_os];
            assert_eq!(&row[..n_paid], paid, "{name} simulation {i}");
            assert_eq!(&row[n_paid..], outstanding, "{name} simulation {i}");
        }
        assert_eq!(
            &without.draw_matrix()[..n_future],
            &with.draw_matrix()[..n_future]
        );
        // The ODE's paid curve rises, and model 2's incremental means are
        // positive: no share falls or passes 1, so the cash-flow
        // engine's default takes them.
        let falling = (0..n_sims)
            .filter(|&i| patterns.draw(i).windows(2).any(|w| w[1] < w[0]))
            .count();
        let above = (0..n_sims)
            .filter(|&i| patterns.draw(i).iter().any(|&x| x > 1.0))
            .count();
        let mean = patterns.mean();
        eprintln!("{name}: mean {mean:?}, falling {falling}, above 1 {above} of {n_sims}");
        assert_eq!((falling, above), (0, 0), "{name}");
        for i in 0..n_sims {
            let row = patterns.draw(i);
            assert!(row[0] > 0.0 && row[9] == 1.0, "{name} {row:?}");
        }
        let basis = crate::MonthlyBasis::default();
        let monthly = fit
            .patterns
            .monthly(&[], &basis, 120, crate::Monotone::Refuse)
            .unwrap();
        assert_eq!(monthly.len(), n_sims);
        assert!(monthly.iter().all(|row| row[119] == 1.0));
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn an_origin_observed_at_the_last_age_has_no_reserve() {
    // Without 1997 the square has 9 ages, and 1988 is observed at age 9.
    let tri = triangle(false);
    let fit = quick(2).fit(&tri, "paid", incurred(), "premium").unwrap();
    assert!(fit.population_origins.is_empty());
    let r = fit
        .reserves
        .marginal(&vec![Period::year(1988).into()])
        .unwrap();
    assert_eq!((r.mean(), r.std_dev()), (0.0, 0.0));
    assert_eq!(fit.paid.n_components(), (1..=8).sum::<usize>());
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_rate_swap_warns() {
    // Rate priors centred on the mirror image of the paper's posterior:
    // the posterior sits at k_er < k_p and the fit says so.
    let priors = CompartmentalPriors {
        k_er: PositivePrior::LogNormal {
            median: 0.43,
            sd: 0.05,
        },
        k_p: PositivePrior::LogNormal {
            median: 1.75,
            sd: 0.05,
        },
        ..CompartmentalPriors::gesmann_morris()
    };
    let tri = triangle(true);
    let fit = Compartmental { priors, ..quick(5) }
        .fit(&tri, "paid", incurred(), "premium")
        .unwrap();
    assert!(fit.rate_swaps > 0);
    assert!(
        fit.diagnostics
            .warnings
            .iter()
            .any(|w| w.contains("k_er <= k_p")),
        "{:?}",
        fit.diagnostics.warnings
    );
    let ordered = Compartmental {
        priors,
        rate_order: RateOrder::Ordered,
        ..quick(5)
    }
    .fit(&tri, "paid", incurred(), "premium")
    .unwrap();
    assert_eq!(ordered.rate_swaps, 0);
    assert!(
        !ordered
            .diagnostics
            .warnings
            .iter()
            .any(|w| w.contains("k_er <= k_p"))
    );
}

#[test]
fn the_ordered_prior_check_keeps_the_order() {
    let rates = |er: f64, p: f64, sd: f64| CompartmentalPriors {
        k_er: PositivePrior::LogNormal { median: er, sd },
        k_p: PositivePrior::LogNormal { median: p, sd },
        ..CompartmentalPriors::gesmann_morris()
    };
    // The swapped priors of the_rate_swap_warns put about 1e-88 of
    // their mass on k_er > k_p: the ordered prior check says so
    // instead of drawing rates that break the order.
    let swapped = rates(0.43, 1.75, 0.05);
    let (a, _) = rate_order_bound(swapped);
    assert!(norm_cdf(-a) < 1e-80, "{a}");
    let e = Compartmental {
        priors: swapped,
        rate_order: RateOrder::Ordered,
        ..preset()
    }
    .prior_predictive(&triangle(true), "premium", 100)
    .unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(why) if why.contains("almost no mass to k_er > k_p")),
        "{e}"
    );
    // About 2.2e-5 of the mass on the order (a = 4.09): every draw
    // keeps it, and the projection's mean is the truncated normal's,
    // φ(a) / Φ(-a), with the orthogonal one's mean 0.
    let pr = rates(1.0, 1.75, 0.0968);
    let (m_er, s_er) = lognormal(pr.k_er);
    let (m_p, s_p) = lognormal(pr.k_p);
    let (a, s) = rate_order_bound(pr);
    let mass = norm_cdf(-a);
    assert!((1e-6..1e-4).contains(&mass), "{mass}");
    let mut rng = StreamRng::new(7, 0);
    let n = 20_000;
    let (mut z_sum, mut y_sum) = (0.0, 0.0);
    for _ in 0..n {
        let (b_ker, b_kp) = ordered_rates(pr, &mut rng);
        assert!(m_er * (s_er * b_ker).exp() > m_p * (s_p * b_kp).exp());
        z_sum += (s_er * b_ker - s_p * b_kp) / s;
        y_sum += (s_p * b_ker + s_er * b_kp) / s;
    }
    let mills = norm_pdf(a) / mass;
    // Standard errors: under 1 / (a sqrt(n)) = 0.002 for z (the
    // truncated normal's sd is below 1 / a) and 1 / sqrt(n) = 0.007
    // for y.
    assert!(
        (z_sum / n as f64 - mills).abs() < 0.01,
        "{}",
        z_sum / n as f64
    );
    assert!((y_sum / n as f64).abs() < 0.03, "{}", y_sum / n as f64);
}

/// Paid alone under free rates is refused by the settings alone (#580,
/// `reserving-v02.md` decision 29): `validate` reads no data, and a fit
/// refuses the settings before it reads the triangle, so a triangle
/// without the named columns gives the same error, not one about the
/// data, and nothing is sampled.
#[test]
fn paid_alone_under_free_rates_is_refused_before_any_data_or_sampling() {
    let free = Compartmental {
        fitted: Fitted::PaidOnly,
        ..preset()
    };
    let e = free.check_supported().unwrap_err();
    assert!(
        matches!(
            e,
            Error::NotSupported {
                setting: "rate_order Free",
                method: "fitting paid alone (Fitted::PaidOnly)",
                ..
            }
        ),
        "{e:?}"
    );
    let fit = free
        .fit(
            &triangle(true),
            "no such paid",
            Outstanding::Column("no such column"),
            "no such premium",
        )
        .unwrap_err();
    assert_eq!(fit, e);
    assert_eq!(free.validate().unwrap_err(), e);
    // With ordered rates the same settings are valid.
    Compartmental {
        rate_order: RateOrder::Ordered,
        ..free
    }
    .check_supported()
    .unwrap();
}

/// One response alone needs ordered rates (decision 12): under free
/// rates the fit, the segments, the one-year view and the backtest's
/// fit are errors that say why; with ordered rates the settings are
/// valid.
#[test]
fn one_response_needs_ordered_rates() {
    let tri = triangle(true);
    for (fitted, what) in [
        (Fitted::PaidOnly, "paid alone"),
        (Fitted::IncurredOnly, "incurred alone"),
    ] {
        let free = Compartmental { fitted, ..preset() };
        let errors = [
            free.fit(&tri, "paid", incurred(), "premium").unwrap_err(),
            free.fit_segments(&tri, "paid", incurred(), "premium")
                .unwrap_err(),
            free.one_year(
                &tri,
                "paid",
                incurred(),
                "premium",
                &BayesOneYear::default(),
            )
            .unwrap_err(),
        ];
        for e in errors {
            assert!(e.is_not_supported(), "{e:?}");
            let e = e.to_string();
            assert!(
                e.contains(what) && e.contains("rate_order Ordered") && e.contains("swap"),
                "{e}"
            );
        }
        let ordered = Compartmental {
            rate_order: RateOrder::Ordered,
            ..free
        };
        assert!(ordered.validate().is_ok());
    }
    // Paid alone does not read the second column: a missing one is no
    // error.
    let paid_only = Data::new(
        &tri,
        "paid",
        Outstanding::Column("no such column"),
        "premium",
        Fitted::PaidOnly,
    )
    .unwrap();
    assert!(paid_only.outstanding.is_empty() && paid_only.future_cells().outstanding.is_empty());
    assert!(
        Data::new(
            &tri,
            "paid",
            Outstanding::Column("no such column"),
            "premium",
            Fitted::IncurredOnly,
        )
        .is_err()
    );
}

#[test]
fn bad_settings_are_invalid() {
    let tri = triangle(true);
    let fit = |m: &Compartmental| m.fit(&tri, "paid", incurred(), "premium").unwrap_err();
    let name = |e: Error| match e {
        Error::InvalidSetting { name, .. } => name,
        other => panic!("{other}"),
    };
    let with = |f: fn(&mut CompartmentalPriors)| {
        let mut priors = CompartmentalPriors::gesmann_morris();
        f(&mut priors);
        Compartmental { priors, ..preset() }
    };
    type Change = fn(&mut CompartmentalPriors);
    let cases: [(Change, &str); 5] = [
        (
            |p| {
                p.k_p = PositivePrior::LogNormal {
                    median: -1.0,
                    sd: 0.1,
                }
            },
            "k_p median",
        ),
        (
            |p| {
                p.rlr = PositivePrior::LogNormal {
                    median: 0.7,
                    sd: 0.0,
                }
            },
            "rlr sd",
        ),
        (
            |p| p.rrf_effect_sd = HalfPrior::Normal { scale: 0.0 },
            "rrf_effect_sd scale",
        ),
        (|p| p.lkj_eta = f64::NAN, "lkj_eta"),
        (
            |p| {
                p.log_sigma[1] = RealPrior::Normal {
                    mean: f64::INFINITY,
                    sd: 1.0,
                }
            },
            "log_sigma mean",
        ),
    ];
    for (f, expected) in cases {
        assert_eq!(name(fit(&with(f))), expected);
    }
    let m = Compartmental {
        n_sims: 0,
        ..preset()
    };
    assert_eq!(name(fit(&m)), "n_sims");
}

#[test]
fn bad_data_are_errors() {
    let tri = triangle(true);
    let err = |paid: &str, os: Outstanding<'_>, premium: &str| {
        preset()
            .fit(&tri, paid, os, premium)
            .unwrap_err()
            .to_string()
    };
    // Missing columns.
    assert!(err("nope", incurred(), "premium").contains("nope"));
    assert!(err("paid", Outstanding::Column("nope"), "premium").contains("nope"));
    assert!(err("paid", incurred(), "nope").contains("nope"));
    // An outstanding column with no cell, and too few cells on an
    // unequal grain (accepted since decision 13).
    let origin: Vec<Month> = (0..3).map(|w| Month::january(2000 + w)).collect();
    let small = |development_grain: Grain, ages: &[Lag], os: &[f64]| {
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(ages),
            values: &[
                ("paid", &[1.0, 2.0, 3.0]),
                ("os", os),
                ("premium", &[10.0, 10.0, 10.0]),
            ],
            origin_grain: Grain::Year,
            development_grain,
            cumulative: true,
        })
        .unwrap()
    };
    let none = small(Grain::Year, &[12, 12, 12], &[f64::NAN; 3]);
    let e = preset()
        .fit(&none, "paid", Outstanding::Column("os"), "premium")
        .unwrap_err()
        .to_string();
    assert!(e.contains("os"), "{e}");
    let quarterly = small(Grain::Quarter, &[3, 3, 3], &[1.0, 2.0, 3.0]);
    let e = preset()
        .fit(&quarterly, "paid", Outstanding::Column("os"), "premium")
        .unwrap_err()
        .to_string();
    assert!(e.contains("more cells than parameters"), "{e}");
    let few = small(Grain::Year, &[12, 12, 12], &[1.0, 2.0, 3.0]);
    let e = preset()
        .fit(&few, "paid", Outstanding::Column("os"), "premium")
        .unwrap_err()
        .to_string();
    assert!(e.contains("more cells than parameters"), "{e}");
    // An origin with paid but no premium.
    let gap = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&[12, 12, 12]),
        values: &[
            ("paid", &[1.0, 2.0, 3.0]),
            ("os", &[1.0, 2.0, 3.0]),
            ("premium", &[10.0, f64::NAN, 10.0]),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let e = preset()
        .fit(&gap, "paid", Outstanding::Column("os"), "premium")
        .unwrap_err()
        .to_string();
    assert!(e.contains("2001") && e.contains("premium"), "{e}");
}

#[test]
fn the_prior_predictive_paths() {
    let tri = triangle(true);
    let paths = preset().prior_predictive(&tri, "premium", 2_000).unwrap();
    assert_eq!(paths.paid.n_components(), 100);
    assert_eq!(paths.outstanding.n_components(), 100);
    assert_eq!(paths.paid.dims(), ["origin", "age"]);
    // Paid rises with age; at the preset's medians (k_er 3, k_p 1, RLR
    // 0.7, RRF 0.8) the paid at age 10 is about 0.56 of premium.
    let key = |a: i64| vec![Period::year(1988).into(), KeyValue::Int(a)];
    let at = |a| paths.paid.marginal(&key(a)).unwrap().mean();
    assert!(at(12) < at(60) && at(60) < at(120));
    let median = paths
        .paid
        .marginal(&key(120))
        .unwrap()
        .quantile(0.5)
        .unwrap();
    assert!((median / PREMIUM[0] - 0.56).abs() < 0.05, "{median}");
    let again = preset().prior_predictive(&tri, "premium", 2_000).unwrap();
    assert_eq!(paths.paid.draw_matrix(), again.paid.draw_matrix());
    let ordered = Compartmental {
        rate_order: RateOrder::Ordered,
        ..preset()
    }
    .prior_predictive(&tri, "premium", 500)
    .unwrap();
    assert!(
        ordered
            .paid
            .draw_matrix()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    );
    assert!(matches!(
        preset().prior_predictive(&tri, "premium", 0),
        Err(Error::InvalidSetting { name: "n_sims", .. })
    ));
}

/// Incurred alone under the paper's tight `k_p` prior: the incurred does
/// not move the payment rate, the fit says so, and paid alone (which
/// pins `k_p`) is silent.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn incurred_alone_warns_when_the_payment_rate_is_the_priors() {
    let tri = triangle(true);
    let fit = |fitted| {
        Compartmental {
            fitted,
            rate_order: RateOrder::Ordered,
            ..quick(5)
        }
        .fit(&tri, "paid", incurred(), "premium")
        .unwrap()
    };
    let warned = |f: &CompartmentalFit| {
        f.diagnostics
            .warnings
            .iter()
            .any(|w| w.contains("does not identify the payment rate"))
    };
    let incurred_alone = fit(Fitted::IncurredOnly);
    assert!(warned(&incurred_alone), "{:?}", incurred_alone.diagnostics);
    assert!(!warned(&fit(Fitted::PaidOnly)));
    assert!(!warned(&fit(Fitted::PaidAndIncurred)));
}

#[test]
fn incurred_alone_refuses_a_wide_payment_rate_prior() {
    let wide_kp = |sd: f64| CompartmentalPriors {
        k_p: PositivePrior::LogNormal { median: 1.0, sd },
        ..CompartmentalPriors::gesmann_morris()
    };
    let model = |fitted: Fitted, priors: CompartmentalPriors| Compartmental {
        fitted,
        rate_order: RateOrder::Ordered,
        ..Compartmental::new(priors)
    };
    // The diffuse preset (sd 1) is refused, with the reason in the message.
    let err = model(Fitted::IncurredOnly, CompartmentalPriors::diffuse())
        .validate()
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not identify the payment rate") && err.contains("at most 0.5"),
        "{err}"
    );
    // Just over the bound is refused; the bound itself, and the paper's
    // preset (0.1), are accepted.
    assert!(
        model(Fitted::IncurredOnly, wide_kp(0.5 + 1e-9))
            .validate()
            .is_err()
    );
    model(Fitted::IncurredOnly, wide_kp(MAX_INCURRED_ONLY_KP_SD))
        .validate()
        .unwrap();
    model(Fitted::IncurredOnly, CompartmentalPriors::gesmann_morris())
        .validate()
        .unwrap();
    // The rule is on k_p only, and only for incurred alone.
    let wide_rest = CompartmentalPriors {
        k_p: PositivePrior::LogNormal {
            median: 1.0,
            sd: 0.3,
        },
        ..CompartmentalPriors::diffuse()
    };
    model(Fitted::IncurredOnly, wide_rest).validate().unwrap();
    for fitted in [Fitted::PaidOnly, Fitted::PaidAndIncurred] {
        model(fitted, CompartmentalPriors::diffuse())
            .validate()
            .unwrap();
    }
    // The prior check still runs on a refused set, so the user can see
    // what it implies.
    model(Fitted::IncurredOnly, CompartmentalPriors::diffuse())
        .prior_predictive(&triangle(true), "premium", 100)
        .unwrap();
}

#[test]
fn the_diffuse_preset() {
    let d = CompartmentalPriors::diffuse();
    let g = CompartmentalPriors::gesmann_morris();
    // The paper's medians, a log-scale sd of 1 on the four, and
    // everything else as model 1's.
    for (diffuse, paper, median) in [
        (d.k_er, g.k_er, 3.0),
        (d.k_p, g.k_p, 1.0),
        (d.rlr, g.rlr, 0.7),
        (d.rrf, g.rrf, 0.8),
    ] {
        let PositivePrior::LogNormal { median: m, sd } = diffuse;
        let PositivePrior::LogNormal {
            median: pm,
            sd: psd,
        } = paper;
        assert_eq!((m, sd, pm), (median, 1.0, median));
        assert!(psd <= 0.2);
    }
    assert_eq!(
        (d.rlr_effect_sd, d.rrf_effect_sd, d.lkj_eta, d.log_sigma),
        (g.rlr_effect_sd, g.rrf_effect_sd, g.lkj_eta, g.log_sigma)
    );
    assert!(d.ker_effect_sd.is_none() && d.development_effect_sd.iter().all(Option::is_none));
    d.validate().unwrap();
    // The prior predictive check, free and ordered: paid at the last
    // age as a share of the 1988 premium, 5th, 50th and 95th percentiles. The
    // paper's preset gives 0.39, 0.56 and 0.81 (the same under both rate
    // orders); the diffuse one 0.06, 0.53 and 5.5, with the same median.
    // The diffuse one spans the plausible range and beyond without ruling
    // any of it out.
    let tri = triangle(true);
    let band = |priors, rate_order| {
        let paths = Compartmental {
            rate_order,
            ..Compartmental::new(priors)
        }
        .prior_predictive(&tri, "premium", 4_000)
        .unwrap();
        let key = vec![Period::year(1988).into(), KeyValue::Int(120)];
        let paid = paths.paid.marginal(&key).unwrap();
        let q = |p| paid.quantile(p).unwrap() / PREMIUM[0];
        (q(0.05), q(0.5), q(0.95))
    };
    for rate_order in [RateOrder::Free, RateOrder::Ordered] {
        let (lo, mid, hi) = band(g, rate_order);
        let (dlo, dmid, dhi) = band(d, rate_order);
        assert!(hi / lo < 2.5, "{lo} {hi}");
        assert!((mid - dmid).abs() < 0.1, "{mid} {dmid}");
        assert!(dlo < 0.2 && dhi > 1.5 && dhi / dlo > 15.0, "{dlo} {dhi}");
    }
}

#[test]
fn the_layout_of_model_2() {
    let m = model_2(false);
    // 4 intercepts, 8 sds, 1 correlation, 72 z and 2 log sigma.
    assert_eq!(m.dim(), 87);
    assert_eq!(m.sampled_names().len(), 87);
    assert_eq!((m.n_levels(), m.n_dev_levels()), (9, 9));
    assert_eq!(m.dev_level[9], None);
    assert_eq!(m.layout.z0, 13);
    assert_eq!(m.layout.z_rate, [Some(31), Some(40)]);
    assert_eq!(m.layout.z_dev, [Some(49), Some(58), Some(67), Some(76)]);
    assert_eq!(m.log_sigma0(), 85);
    let names = m.names();
    // 6 population values, 4 x 9 origin values, 4 x 9 development
    // effects, 6 intercepts and sds, 2 rate sds, rho, 4 development
    // sds and 2 log sigma.
    assert_eq!(names.len(), 6 + 36 + 36 + 6 + 2 + 1 + 4 + 2);
    assert_eq!(names[6], "RLR[1988]");
    assert_eq!(names[24], "k_er[1988]");
    assert_eq!(names[42], "dev_ker[12]");
    assert_eq!(names[77], "dev_RRF[108]");
    assert_eq!(
        &names[82..],
        [
            "tau_RLR",
            "tau_RRF",
            "tau_ker",
            "tau_kp",
            "rho",
            "tau_dev_ker",
            "tau_dev_kp",
            "tau_dev_RLR",
            "tau_dev_RRF",
            "log_sigma_os",
            "log_sigma_pd"
        ]
    );
    let nat = m.natural(&points(&m)[0]);
    assert_eq!(nat.len(), names.len());
    // Loss ratios, incremental paid.
    let c = &m.cells;
    assert!((c[0].y - (62679. - 9558.) / PREMIUM[0]).abs() < 1e-15);
    assert!(c[45].paid && c[45].y == 9558. / PREMIUM[0] && c[45].t_before.is_none());
    assert_eq!(c[46].y, (22778. - 9558.) / PREMIUM[0]);
    assert_eq!((c[46].t_before, c[46].w), (Some(1.0), 1.0));
    assert_eq!(c[46].ln_y, c[46].y.ln());
    assert_eq!(
        CompartmentalPriors::gesmann_morris_2().hierarchy(),
        Hierarchy::Model2
    );
    let some = CompartmentalPriors {
        kp_effect_sd: None,
        ..CompartmentalPriors::gesmann_morris_2()
    };
    assert_eq!(some.hierarchy(), Hierarchy::Other);
    let m = model_with(some, false, Response::MODEL_2);
    assert_eq!(m.dim(), 87 - 1 - 9);
}

/// The log posterior written out in `f64` with the paper's form of the
/// closed form, for either preset's hierarchy and any response, at `x`
/// in the documented layout.
fn written_out(two: bool, response: Response, x: &[f64]) -> f64 {
    let half_t = |l: f64, s: f64| l - 5.5 * ((l.exp() / s).powi(2) / 10.0).ln_1p();
    let (n_tau, z0, ls) = if two { (6, 13, 85) } else { (0, 7, 25) };
    let rho = x[6].tanh();
    let mut f =
        -0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) + (1.0 - rho * rho).ln();
    let scales = if two {
        f += half_t(x[4], 0.7) + half_t(x[5], 0.5);
        [0.3, 0.3, 0.3, 0.3, 0.7, 0.5]
    } else {
        f += half_t(x[4], 0.2) + half_t(x[5], 0.1);
        [0.0; 6]
    };
    for j in 0..n_tau {
        f += half_t(x[7 + j], scales[j]);
    }
    for z in &x[z0..ls] {
        f -= 0.5 * z * z;
    }
    for l in [x[ls], x[ls + 1]] {
        f += if two {
            -0.5 * ((l - 0.2f64.ln()) / 0.2).powi(2)
        } else {
            -((l / 1000.0).powi(2)).ln_1p()
        };
    }
    let tau = |j: usize| x[7 + j].exp();
    let pd = |a: f64, b: f64, rlr: f64, rrf: f64, t: f64| {
        rlr * rrf / (a - b) * (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp()))
    };
    for (w, row) in PAID.iter().enumerate() {
        for (d, &paid) in row.iter().enumerate() {
            let (z1, z2) = (x[z0 + 2 * w], x[z0 + 2 * w + 1]);
            let mut o = [
                x[0],
                x[1],
                x[2] + x[4].exp() * z1,
                x[3] + x[5].exp() * (rho * z1 + (1.0 - rho * rho).sqrt() * z2),
            ];
            if two {
                o[0] += tau(0) * x[31 + w];
                o[1] += tau(1) * x[40 + w];
                for (p, o) in o.iter_mut().enumerate() {
                    *o += tau(2 + p) * x[49 + 9 * p + d];
                }
            }
            let (a, b) = (3.0 * (0.1 * o[0]).exp(), (0.1 * o[1]).exp());
            let (rlr, rrf) = (0.7 * (0.2 * o[2]).exp(), 0.8 * (0.1 * o[3]).exp());
            let t = (d + 1) as f64;
            let os = rlr * a / (a - b) * ((-b * t).exp() - (-a * t).exp());
            let (paid_y, paid_mean) = if response.incremental && d > 0 {
                (
                    paid - row[d - 1],
                    pd(a, b, rlr, rrf, t) - pd(a, b, rlr, rrf, t - 1.0),
                )
            } else {
                (paid, pd(a, b, rlr, rrf, t))
            };
            let unit = if response.loss_ratio { 1.0 } else { PREMIUM[w] };
            for (y, mean, l) in [
                (INCURRED[w][d] - paid, os, x[ls]),
                (paid_y, paid_mean, x[ls + 1]),
            ] {
                let (y, mean) = (y / PREMIUM[w] * unit, mean * unit);
                let r = if response.log_normal() {
                    (y.ln() - mean.ln()) / l.exp()
                } else {
                    (y - mean) / l.exp()
                };
                f -= l + 0.5 * r * r;
            }
        }
    }
    f
}

#[test]
fn the_density_is_each_model_written_out() {
    // Both presets' hierarchies with every response, at one point.
    for two in [false, true] {
        for likelihood in [
            CompartmentalLikelihood::Normal,
            CompartmentalLikelihood::LogNormal,
        ] {
            for loss_ratio in [false, true] {
                for incremental in [false, true] {
                    let response = Response {
                        likelihood,
                        loss_ratio,
                        incremental,
                        incurred: false,
                        paid: true,
                        second: true,
                        levels: DevelopmentLevels::PerAge,
                    };
                    let priors = if two {
                        CompartmentalPriors::gesmann_morris_2()
                    } else {
                        CompartmentalPriors::gesmann_morris()
                    };
                    let m = model_with(priors, false, response);
                    let mut x = points(&m)[3].clone();
                    let ls = m.log_sigma0();
                    (x[ls], x[ls + 1]) = match (response.log_normal(), loss_ratio) {
                        (true, _) => (-1.8, -1.9),
                        (false, true) => (-3.0, -3.5),
                        (false, false) => (8.25, 6.66),
                    };
                    let want = written_out(two, response, &x);
                    let mut g = vec![0.0; m.dim()];
                    let got = m.log_posterior(&x, &mut g);
                    assert!(
                        (got - want).abs() <= 1e-12 * want.abs(),
                        "{two} {response:?}: {got} vs {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn responses_the_likelihood_cannot_take_are_errors() {
    let origin: Vec<Month> = [0, 0, 1].map(|w| Month::january(2000 + w)).to_vec();
    let tri = |ages: &[Lag], paid: &[f64], os: &[f64]| {
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(ages),
            values: &[("paid", paid), ("os", os), ("premium", &[10.0; 3])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    };
    let err = |m: &Compartmental, t: &Triangle| {
        m.fit(t, "paid", Outstanding::Column("os"), "premium")
            .unwrap_err()
            .to_string()
    };
    // A zero outstanding, and a paid that does not grow: no log.
    let zero_os = tri(&[12, 24, 12], &[1.0, 2.0, 1.0], &[3.0, 0.0, 2.0]);
    let e = err(&preset_2(), &zero_os);
    assert!(
        e.contains("2000") && e.contains("24 months") && e.contains("outstanding 0"),
        "{e}"
    );
    let flat = tri(&[12, 24, 12], &[1.0, 1.0, 1.0], &[3.0, 2.0, 2.0]);
    let e = err(&preset_2(), &flat);
    assert!(
        e.contains("incremental paid 0") && e.contains("positive"),
        "{e}"
    );
    // Incremental paid with no paid at the age before.
    let gap = tri(&[12, 36, 12], &[1.0, 2.0, 1.0], &[3.0, 2.0, 2.0]);
    let normal = Compartmental {
        paid_response: PaidResponse::Incremental,
        ..preset()
    };
    let e = err(&normal, &gap);
    assert!(
        e.contains("36 months") && e.contains("none at 24 months"),
        "{e}"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn model_2_fits_and_predicts_increments() {
    let tri = triangle(true);
    let fit = Compartmental {
        sampler: Sampler {
            chains: 2,
            tune: 60,
            draws: 60,
            target_accept: 0.9,
            ..preset_2().sampler
        },
        n_sims: 200,
        seed: 6,
        ..preset_2()
    }
    .fit(&tri, "paid", incurred(), "premium")
    .unwrap();
    assert_eq!(fit.population_origins, [Period::year(1997)]);
    assert_eq!(fit.posterior.dim(), 93);
    assert!(fit.posterior.names().iter().any(|n| n == "tau_dev_RRF"));
    let share = fit.cell_rate_swaps.unwrap();
    assert!((0.0..=1.0).contains(&share), "{share}");
    assert_eq!(fit.loo().unwrap().loo.pareto_k.len(), 90);
    assert_eq!(fit.reserves.provenance().model, "compartmental_gm2");
    // Log-normal increments are positive, so each simulation's paid
    // grows with age from the latest paid, and the reserve is the paid
    // at age 10 less the latest.
    let keys = fit.paid.components();
    let n = keys.len();
    for (row, reserves) in fit
        .paid
        .draw_matrix()
        .chunks_exact(n)
        .zip(fit.reserves.draw_matrix().as_chunks::<10>().0)
    {
        for j in 0..n {
            let (KeyValue::Period(o), KeyValue::Int(age)) = (&keys[j][0], &keys[j][1]) else {
                unreachable!("cell keys are (origin, age)")
            };
            let i = fit.origins.iter().position(|p| p == o).unwrap();
            let before = if j > 0 && keys[j - 1][0] == keys[j][0] {
                row[j - 1]
            } else {
                fit.latest_paid[i]
            };
            assert!(row[j] > before, "{o} {age}: {} after {before}", row[j]);
            if *age == 120 {
                assert!((row[j] - fit.latest_paid[i] - reserves[i]).abs() < 1e-9 * row[j]);
            }
        }
    }
    assert!(
        fit.ultimate_loss_ratio
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    );
}

#[test]
fn the_prior_predictive_paths_of_model_2() {
    let tri = triangle(true);
    let paths = preset_2().prior_predictive(&tri, "premium", 500).unwrap();
    assert_eq!(paths.paid.n_components(), 100);
    // The increments are positive, so cumulative paid grows with age in
    // every draw.
    let draws = paths.paid.draw_matrix();
    for row in draws.as_chunks::<100>().0 {
        for i in 0..10 {
            for k in 1..10 {
                assert!(row[10 * i + k] > row[10 * i + k - 1]);
            }
        }
    }
    let os = paths.outstanding.draw_matrix();
    assert!(os.iter().all(|v| v.is_finite() && *v > 0.0));
    let again = preset_2().prior_predictive(&tri, "premium", 500).unwrap();
    assert_eq!(draws, again.paid.draw_matrix());
    // The provenance records the paid response and the priors.
    let param = |pd: &PredictiveDistribution, key: &str| {
        pd.provenance()
            .parameters
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap()
    };
    let cumulative = Compartmental {
        paid_response: PaidResponse::Cumulative,
        ..preset_2()
    }
    .prior_predictive(&tri, "premium", 10)
    .unwrap();
    assert_ne!(
        param(&paths.paid, "paid_response"),
        param(&cumulative.paid, "paid_response")
    );
    let model_1 = preset().prior_predictive(&tri, "premium", 10).unwrap();
    assert_ne!(
        param(&paths.outstanding, "priors"),
        param(&model_1.outstanding, "priors")
    );
}

#[test]
fn the_cellwise_gradient_is_the_generic_densitys() {
    // The per-cell gradient against the chunked dual pass over the
    // generic density (the oracle), at 20 points around model 2's
    // posterior: both rate orders, a partial hierarchy, and amounts with
    // the normal likelihood.
    let partial = CompartmentalPriors {
        ker_effect_sd: None,
        development_effect_sd: [None, Some(HalfPrior::Normal { scale: 0.3 }), None, None],
        ..CompartmentalPriors::gesmann_morris_2()
    };
    let models = [
        model_2(false),
        model_2(true),
        model_with(partial, false, Response::MODEL_2),
        model_with(
            CompartmentalPriors::gesmann_morris_2(),
            false,
            Response::MODEL_1,
        ),
    ];
    let (mut worst_value, mut worst_grad): (f64, f64) = (0.0, 0.0);
    for (n, m) in models.iter().enumerate() {
        assert!(m.cellwise);
        for mut x in points(m) {
            if n == 3 {
                let ls = m.log_sigma0();
                (x[ls], x[ls + 1]) = (8.25, 6.66);
            }
            let mut want = vec![0.0; m.dim()];
            let f = gradient::<CHUNK>(&x, &mut want, |t| m.log_density(t));
            let mut got = vec![0.0; m.dim()];
            let g = m.cellwise_gradient(&x, &mut got);
            let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            worst_value = worst_value.max((g - f).abs() / f.abs());
            for (a, b) in got.iter().zip(&want) {
                worst_grad = worst_grad.max((a - b).abs() / scale);
            }
        }
    }
    println!("cellwise gradient: value {worst_value:.1e}, gradient {worst_grad:.1e}");
    assert!(worst_value <= 1e-12 && worst_grad <= 1e-12);
}

/// The cost of model 2's log posterior on the paper's training triangle
/// (87 parameters), release build: median time of the value and of the
/// value with its gradient.
#[test]
#[ignore = "a release benchmark; run with --release --ignored --nocapture"]
fn model_2_gradient_benchmark() {
    let median = |f: &mut dyn FnMut()| {
        let mut times: Vec<f64> = (0..41)
            .map(|_| {
                let start = std::time::Instant::now();
                for _ in 0..200 {
                    f();
                }
                start.elapsed().as_secs_f64() / 200.0 * 1e6
            })
            .collect();
        times.sort_by(f64::total_cmp);
        times[20]
    };
    for (label, m) in [("model 1", model(false)), ("model 2", model_2(false))] {
        let x = points(&m)[0].clone();
        let mut g = vec![0.0; m.dim()];
        let value = median(&mut || {
            std::hint::black_box(m.log_density::<f64>(std::hint::black_box(&x)));
        });
        let chunked = median(&mut || {
            std::hint::black_box(gradient::<CHUNK>(std::hint::black_box(&x), &mut g, |t| {
                m.log_density(t)
            }));
        });
        let grad = median(&mut || {
            std::hint::black_box(m.log_posterior(std::hint::black_box(&x), &mut g));
        });
        println!(
            "{label}, p = {}: value {value:.2} us, chunked dual gradient {chunked:.2} us \
             ({:.1}x), the sampler's gradient {grad:.2} us ({:.1}x)",
            m.dim(),
            chunked / value,
            grad / value
        );
    }
}

/// The training triangle's square of 1988 to 1992 at ages up to 60
/// months: every origin at the square's last age, so no year.
fn closed_triangle() -> Triangle {
    let (mut origin, mut ages) = (vec![], vec![]);
    let (mut paid, mut inc, mut prem) = (vec![], vec![], vec![]);
    for w in 0..5 {
        for d in 0..5 {
            origin.push(Month::january(1988 + w as i32));
            ages.push(12 * (d as Lag + 1));
            paid.push(PAID[w][d]);
            inc.push(INCURRED[w][d]);
            prem.push(PREMIUM[w]);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("incurred", &inc), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// A triangle with annual origins 1988 to 1992 and quarterly
/// development (decision 13), valued at the end of 1992 (1988 at 3 to
/// 60 months, 1992 at 3 to 12: 60 cells of each type), or, when
/// `closed`, every origin at 3 to 60 months (the closed square, no
/// year). Each cell is the closed form of model 1 at `k_er` 1.8, `k_p`
/// 0.45 and each origin's own `RLR` and `RRF`, times its premium, with
/// a deterministic wobble of up to 8% on each outstanding and each paid
/// increment, so both stay positive.
pub(super) fn quarterly_triangle(closed: bool) -> Triangle {
    quarterly_lines(&[""], closed)
}

/// [`quarterly_triangle`] once for each of `names`, with a key `lob`
/// unless `names` is `[""]`.
pub(super) fn quarterly_lines(names: &[&str], closed: bool) -> Triangle {
    quarterly_lines_with(names, closed, Lag::MAX, 0.08)
}

/// [`quarterly_lines`] with no cell after age `last_age` months, so the
/// oldest origins stop short of the square's last development year, and
/// a wobble of up to `amplitude` on each cell.
pub(super) fn quarterly_lines_with(
    names: &[&str],
    closed: bool,
    last_age: Lag,
    amplitude: f64,
) -> Triangle {
    let (mut lob, mut origin, mut ages) = (vec![], vec![], vec![]);
    let (mut paid, mut inc, mut prem) = (vec![], vec![], vec![]);
    for (w, name) in (0..5usize).flat_map(|w| names.iter().map(move |n| (w, *n))) {
        let rlr = 0.75 + 0.03 * (w as f64).sin();
        let rrf = 0.8 + 0.02 * (2.0 * w as f64).cos();
        let n_ages = (if closed { 20 } else { 4 * (5 - w) }).min((last_age / 3) as usize);
        let (mut p, mut before) = (0.0, 0.0);
        for d in 0..n_ages {
            let age = 3 * (d as Lag + 1);
            let (os, pd) = one_os(1.8, 0.45, f64::from(age) / 12.0);
            let wobble = |j: usize| 1.0 + amplitude * ((7 * w + 3 * d + j) as f64).sin();
            // The wobble on each increment, so every increment is
            // positive.
            p += PREMIUM[w] * rlr * rrf * (pd - before) * wobble(0);
            before = pd;
            lob.push(name);
            origin.push(Month::january(1988 + w as i32));
            ages.push(age);
            paid.push(p);
            inc.push(PREMIUM[w] * rlr * os * wobble(1) + p);
            prem.push(PREMIUM[w]);
        }
    }
    let key = [("lob", &lob[..])];
    Triangle::from_long(&Long {
        keys: if names == [""] { &[] } else { &key },
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("incurred", &inc), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Quarter,
        cumulative: true,
    })
    .unwrap()
}

/// The grains of the square (decision 13): the origin grain a whole
/// multiple of the development grain, else an error naming both.
#[test]
fn the_square_needs_whole_development_periods_in_an_origin() {
    use Grain::{Month, Quarter, Semester, Year};
    for (origin, development, per) in [
        (Year, Year, 1),
        (Year, Semester, 2),
        (Year, Quarter, 4),
        (Year, Month, 12),
        (Semester, Quarter, 2),
        (Quarter, Month, 3),
        (Quarter, Quarter, 1),
    ] {
        let g = SquareGrain::new(origin, development).unwrap();
        assert_eq!(g.per_origin(), per);
        assert_eq!(g.square_ages(10), 10 * per);
    }
    for (origin, development) in [(Quarter, Semester), (Quarter, Year), (Month, Quarter)] {
        let e = SquareGrain::new(origin, development)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(&format!("origin grain {origin}"))
                && e.contains(&format!("development grain {development}")),
            "{e}"
        );
    }
}

/// Ages map to their time in years on every grain (decision 13), so an
/// equal-grain fit keeps its times: annual ages 12 to 120 months are 1
/// to 10 years, quarterly ages 3 to 120 months are 0.25 to 10 years.
#[test]
fn ages_map_to_years_and_the_square_to_the_origin_periods() {
    let annual = SquareGrain::new(Grain::Year, Grain::Year).unwrap();
    let quarterly = SquareGrain::new(Grain::Year, Grain::Quarter).unwrap();
    let monthly = SquareGrain::new(Grain::Quarter, Grain::Month).unwrap();
    for k in 0..10 {
        assert_eq!(annual.age(k), 12 * (k as Lag + 1));
        assert_eq!(annual.time(k), (k + 1) as f64);
        assert_eq!(annual.age_index(annual.age(k)).unwrap(), k);
    }
    for k in 0..40 {
        assert_eq!(quarterly.age(k), 3 * (k as Lag + 1));
        assert_eq!(quarterly.time(k), (k + 1) as f64 / 4.0);
        assert_eq!(quarterly.age_index(quarterly.age(k)).unwrap(), k);
    }
    // Quarterly origins with monthly development: still years.
    assert_eq!(monthly.time(5), 0.5);
    assert!(quarterly.age_index(4).is_err());
    assert!(quarterly.age_index(0).is_err());
    // The data and the model of the quarterly triangle: 5 origins, so
    // the square has 5 years of quarterly ages, 20.
    let tri = quarterly_triangle(false);
    let data = Data::new(&tri, "paid", incurred(), "premium", Fitted::PaidAndIncurred).unwrap();
    assert_eq!(data.n_ages, 20);
    assert_eq!((data.paid.len(), data.outstanding.len()), (60, 60));
    assert_eq!(data.latest_paid[4].map(|l| l.0), Some(3));
    assert_eq!(data.age(19), 60);
    let model = Model::new(
        &data,
        CompartmentalPriors::gesmann_morris(),
        false,
        Response {
            incurred: true,
            ..Response::MODEL_1
        },
    )
    .unwrap();
    assert_eq!(model.times.len(), 20);
    assert_eq!(model.times[0], 0.25);
    assert_eq!(model.times[19], 5.0);
    // The future cells run to the square's last age, 60 months: 1988
    // is complete, 1992 has 16 future quarters.
    let future = data.future_cells();
    let count = |i: usize| future.paid.iter().filter(|c| c.0 == i).count();
    assert_eq!((0..5).map(count).collect::<Vec<_>>(), [0, 4, 8, 12, 16]);
    assert_eq!(future.last_paid[0], None);
    assert_eq!(future.paid[future.last_paid[4].unwrap()], (4, 19));
    // With a sixth origin (premium only) the square has 24 ages.
    let mut more = data.clone();
    more.origins.push(Period::year(1993));
    more.premium.push(PREMIUM[5]);
    assert_eq!(more.square_ages(), 24);
}

/// Incremental paid on quarterly development is the increment over one
/// quarter: each paid cell after the first age is its paid less that
/// of the quarter before, with `t_before` a quarter earlier.
#[test]
fn incremental_paid_is_the_increment_over_one_development_period() {
    let tri = quarterly_triangle(false);
    let data = Data::new(&tri, "paid", incurred(), "premium", Fitted::PaidAndIncurred).unwrap();
    let response = Response {
        incurred: true,
        ..Response::MODEL_2
    };
    let model = Model::new(
        &data,
        CompartmentalPriors::gesmann_morris_2(),
        false,
        response,
    )
    .unwrap();
    let paid_at = |i: usize, k: usize| {
        data.paid
            .iter()
            .find(|c| c.0 == i && c.1 == k)
            .map(|c| c.2)
            .unwrap()
    };
    let cells = &model.cells[model.n_outstanding..];
    assert_eq!(cells.len(), 60);
    for c in cells {
        let before = if c.k == 0 {
            assert_eq!(c.t_before, None);
            0.0
        } else {
            assert_eq!(c.t_before, Some(c.t - 0.25));
            paid_at(c.origin, c.k - 1)
        };
        let want = (paid_at(c.origin, c.k) - before) / PREMIUM[c.origin];
        assert!(
            (c.y - want).abs() <= 1e-12 * want.abs(),
            "{} vs {want}",
            c.y
        );
    }
    // One development level per quarterly age with a cell (20).
    assert_eq!(model.dev_labels.len(), 20);
    assert_eq!(model.dev_labels[1], "6");
}

/// The gradient on the quarterly triangle against central differences,
/// for model 1 and for model 2's hierarchy and response, each with
/// paid and incurred.
#[test]
fn the_unequal_grain_gradient_matches_central_differences() {
    let tri = quarterly_triangle(false);
    let data = Data::new(&tri, "paid", incurred(), "premium", Fitted::PaidAndIncurred).unwrap();
    for (priors, response, step) in [
        (CompartmentalPriors::gesmann_morris(), Response::MODEL_1, 2),
        (
            CompartmentalPriors::gesmann_morris_2(),
            Response::MODEL_2,
            5,
        ),
        // Levels by development year: 5 levels of 4 ages each.
        (
            CompartmentalPriors::gesmann_morris_2(),
            Response {
                levels: DevelopmentLevels::ByYear,
                ..Response::MODEL_2
            },
            5,
        ),
    ] {
        for ordered in [false, true] {
            let response = Response {
                incurred: true,
                ..response
            };
            let m = Model::new(&data, priors, ordered, response).unwrap();
            for x in points(&m).iter().step_by(step) {
                assert_gradient(|t, g| m.log_posterior(t, g), x);
            }
        }
    }
}

/// A triangle on the given grains with `origins` (start month, last
/// observed age in months), every age from `first` to it observed except
/// those in `skip`, with positive cumulative paid and outstanding and a
/// premium of 1,000.
pub(super) fn shaped(
    origin_grain: Grain,
    development_grain: Grain,
    origins: &[(Month, Lag)],
    first: Lag,
    skip: &[Lag],
) -> Triangle {
    let step = development_grain.months();
    let (mut origin, mut ages) = (vec![], vec![]);
    let (mut paid, mut os, mut prem) = (vec![], vec![], vec![]);
    for &(start, last) in origins {
        for age in (first..=last).step_by(step as usize) {
            if skip.contains(&age) {
                continue;
            }
            let a = f64::from(age);
            origin.push(start);
            ages.push(age);
            paid.push(100.0 * (1.0 - (-a / 30.0).exp()) + 1.0);
            os.push(80.0 * (-a / 40.0).exp() + 5.0);
            prem.push(1000.0);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("os", &os), ("premium", &prem)],
        origin_grain,
        development_grain,
        cumulative: true,
    })
    .unwrap()
}

/// Model 2's density on `tri` with cumulative log-normal paid (so that
/// a triangle with missing ages is readable) and the given levels.
pub(super) fn levels_model(tri: &Triangle, levels: DevelopmentLevels) -> Model {
    levels_model_on(
        tri,
        Outstanding::Column("os"),
        Fitted::PaidAndOutstanding,
        levels,
    )
}

/// [`levels_model`] with the second response named.
fn levels_model_on(
    tri: &Triangle,
    outstanding: Outstanding<'_>,
    fitted: Fitted,
    levels: DevelopmentLevels,
) -> Model {
    let data = Data::new(tri, "paid", outstanding, "premium", fitted).unwrap();
    let response = Response {
        incremental: false,
        incurred: fitted == Fitted::PaidAndIncurred,
        levels,
        ..Response::MODEL_2
    };
    Model::new(
        &data,
        CompartmentalPriors::gesmann_morris_2(),
        false,
        response,
    )
    .unwrap()
}

/// The development year of an age is `(age - 1) / 12`, from the
/// origin's own start: levels by year have one level per year with an
/// observed cell on the quarterly, monthly and semiannual grains, their
/// ages share it, and the density has 4 effects fewer per level fewer.
#[test]
fn levels_by_year_count_the_development_years() {
    use DevelopmentLevels::{ByYear, PerAge};
    // Quarterly development, annual origins: 5 years of 4 ages.
    let tri = quarterly_triangle(false);
    let on = |levels| levels_model_on(&tri, incurred(), Fitted::PaidAndIncurred, levels);
    let (per_age, by_year) = (on(PerAge), on(ByYear));
    assert_eq!((per_age.n_dev_levels(), by_year.n_dev_levels()), (20, 5));
    assert_eq!(by_year.dev_labels, ["3", "15", "27", "39", "51"]);
    assert_eq!(by_year.dev_level[3], Some(0), "age 12 is in the first year");
    assert_eq!(
        by_year.dev_level[4],
        Some(1),
        "age 15 is in the second year"
    );
    assert_eq!(by_year.dev_level[19], Some(4));
    assert_eq!(per_age.dim() - by_year.dim(), 4 * 15);
    // Monthly development, annual origins: 24 ages, 2 years.
    let tri = shaped(
        Grain::Year,
        Grain::Month,
        &[(Month::january(1990), 24), (Month::january(1991), 12)],
        1,
        &[],
    );
    let (per_age, by_year) = (levels_model(&tri, PerAge), levels_model(&tri, ByYear));
    assert_eq!((per_age.n_dev_levels(), by_year.n_dev_levels()), (24, 2));
    assert_eq!(by_year.dev_labels, ["1", "13"]);
    assert_eq!(by_year.dev_level[11], Some(0));
    assert_eq!(by_year.dev_level[12], Some(1));
    // Semiannual development: ages 6 and 12, then 18 and 24.
    let tri = shaped(
        Grain::Year,
        Grain::Semester,
        &[(Month::january(1990), 24), (Month::january(1991), 12)],
        6,
        &[],
    );
    let by_year = levels_model(&tri, ByYear);
    assert_eq!(by_year.dev_labels, ["6", "18"]);
    assert_eq!(by_year.dev_level, [Some(0), Some(0), Some(1), Some(1)]);
}

/// A year counts from each origin's start and not by calendar: the same
/// monthly square of quarterly origins starting in January or in April
/// has the same development levels.
#[test]
fn a_development_year_does_not_depend_on_the_calendar_month_of_the_origin() {
    let starts = |first_month: u32| -> Vec<(Month, Lag)> {
        (0..8u32)
            .map(|q| {
                let months = 3 * q + first_month - 1;
                (
                    Month::new(2000 + (months / 12) as i32, (months % 12 + 1) as u8).unwrap(),
                    3 * (8 - q),
                )
            })
            .collect()
    };
    let model = |first_month: u32| {
        let tri = shaped(Grain::Quarter, Grain::Month, &starts(first_month), 1, &[]);
        levels_model(&tri, DevelopmentLevels::ByYear)
    };
    let (january, april) = (model(1), model(4));
    assert_eq!(january.n_dev_levels(), 2);
    assert_eq!(january.dev_labels, ["1", "13"]);
    assert_eq!(january.dev_labels, april.dev_labels);
    assert_eq!(january.dev_level, april.dev_level);
    assert_eq!(january.dev_level[12], Some(1), "age 13 is the second year");
}

/// At the annual grain the two widths are the same model: the same
/// levels, parameters and density.
#[test]
fn at_the_annual_grain_the_widths_are_the_same_model() {
    let (per_age, by_year) = (
        model_with(
            CompartmentalPriors::gesmann_morris_2(),
            false,
            Response::MODEL_2,
        ),
        model_with(
            CompartmentalPriors::gesmann_morris_2(),
            false,
            Response {
                levels: DevelopmentLevels::ByYear,
                ..Response::MODEL_2
            },
        ),
    );
    assert_eq!(per_age.dev_level, by_year.dev_level);
    assert_eq!(per_age.dev_labels, by_year.dev_labels);
    assert_eq!(per_age.dim(), by_year.dim());
    for x in points(&per_age).iter().step_by(3) {
        let (mut g1, mut g2) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        let f1 = per_age.log_posterior(x, &mut g1);
        let f2 = by_year.log_posterior(x, &mut g2);
        assert_eq!((f1, g1), (f2, g2));
    }
}

/// Ages with no observed cell: a year with none takes one new effect
/// for all its ages (not one per age), and an age with no cell of its
/// own in an observed year takes the year's level.
#[test]
fn a_year_without_a_cell_shares_one_new_effect() {
    use DevelopmentLevels::{ByYear, PerAge};
    // Annual origins, quarterly development, ages 15 and later only,
    // and no cell at age 21 (in the second year with 15, 18 and 24).
    let origins: Vec<(Month, Lag)> = (0..4)
        .map(|w| (Month::january(1988 + w), 12 * (5 - w as Lag)))
        .collect();
    let tri = shaped(Grain::Year, Grain::Quarter, &origins, 15, &[21]);
    let (per_age, by_year) = (levels_model(&tri, PerAge), levels_model(&tri, ByYear));
    // Ages 3 to 12 (indices 0 to 3) have no cell in either.
    assert!(per_age.dev_level[..4].iter().all(Option::is_none));
    assert!(by_year.dev_level[..4].iter().all(Option::is_none));
    // The unobserved age 21 (index 6): none per age, the year's level
    // by year.
    assert_eq!(per_age.dev_level[6], None);
    assert_eq!(by_year.dev_level[6], by_year.dev_level[4]);
    assert_eq!(by_year.n_dev_levels(), 4);
    assert_eq!(per_age.n_dev_levels(), 16 - 1);
    // The multipliers of a draw: count the new effects asked for.
    let nat: Vec<f64> = (0..per_age.names().len())
        .map(|i| 0.01 * i as f64)
        .collect();
    let asked = |m: &Model| {
        let mut calls = 0;
        let mult = m.dev_multipliers_with(&nat, |_| {
            calls += 1;
            0.5 * f64::from(calls)
        });
        (calls, mult)
    };
    let (n_per_age, mult_per_age) = asked(&per_age);
    let (n_by_year, mult_by_year) = asked(&by_year);
    // 4 effects (k_er, k_p, RLR, RRF): one new draw for each of the 5
    // ages without a level per age (3, 6, 9, 12 and 21), one for the
    // one year without a level by year.
    assert_eq!((n_per_age, n_by_year), (4 * 5, 4));
    assert!(mult_by_year[..4].windows(2).all(|w| w[0] == w[1]));
    assert!(mult_per_age[..4].windows(2).all(|w| w[0] != w[1]));
    // Age 21 takes the level of its year, as ages 15 and 18 do.
    assert_eq!(mult_by_year[6], mult_by_year[4]);
    assert_eq!(mult_by_year[5], mult_by_year[4]);
}

/// Levels by year are refused where the priors have no development
/// effects (model 1), when the setting is checked and before any data
/// are read; model 2 takes them.
#[test]
fn levels_by_year_need_development_effects() {
    let by_year = |m: Compartmental| Compartmental {
        development_levels: DevelopmentLevels::ByYear,
        ..m
    };
    let err = by_year(preset()).check_supported().unwrap_err();
    assert!(
        matches!(&err, Error::NotSupported { setting, .. } if setting.contains("development_levels")),
        "{err:?}"
    );
    let tri = quarterly_triangle(false);
    let fit = by_year(preset()).fit(&tri, "paid", incurred(), "premium");
    assert!(matches!(fit, Err(Error::NotSupported { .. })));
    // Per age, and model 2 by year, are supported.
    preset().check_supported().unwrap();
    by_year(preset_2()).check_supported().unwrap();
    // Model 1 has no effects to group, so the prior check still runs.
    by_year(preset())
        .prior_predictive(&tri, "premium", 5)
        .unwrap();
}

/// The prior check draws the levels the fit has: under levels by year
/// the four quarters of a development year share one effect, so a step
/// within the year moves far less than one between years, and far less
/// than the same step per age. At annual grain the two settings are one
/// model and give the same draws, and the provenance names the setting.
#[test]
fn the_prior_check_draws_the_fits_levels() {
    let model = |levels| Compartmental {
        development_levels: levels,
        ..preset_2()
    };
    let tri = quarterly_triangle(false);
    let paths = |levels| {
        model(levels)
            .prior_predictive(&tri, "premium", 2_000)
            .unwrap()
    };
    // The sd over the simulations of the log step from age index `k` to
    // `k + 1` of the oldest origin's outstanding.
    let step_sd = |p: &CompartmentalPaths, k: usize| {
        let n = p.outstanding.n_components();
        let steps: Vec<f64> = p
            .outstanding
            .draw_matrix()
            .chunks_exact(n)
            .map(|row| (row[k + 1] / row[k]).ln())
            .collect();
        let mean = steps.iter().sum::<f64>() / steps.len() as f64;
        (steps.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / steps.len() as f64).sqrt()
    };
    let (per_age, by_year) = (
        paths(DevelopmentLevels::PerAge),
        paths(DevelopmentLevels::ByYear),
    );
    // Indices 0 to 3 are the first year's quarters, 4 the next year's.
    let (within, across) = (step_sd(&by_year, 1), step_sd(&by_year, 3));
    assert!(within < 0.5 * across, "{within} {across}");
    assert!(within < 0.5 * step_sd(&per_age, 1));
    let levels = |p: &CompartmentalPaths| {
        let params = &p.paid.provenance().parameters;
        let found = params.iter().find(|(k, _)| k == "development_levels");
        found.unwrap().1.clone()
    };
    assert_ne!(levels(&per_age), levels(&by_year));
    let annual = triangle(true);
    let draws = |levels| {
        model(levels)
            .prior_predictive(&annual, "premium", 50)
            .unwrap()
            .paid
            .draw_matrix()
            .to_vec()
    };
    assert_eq!(
        draws(DevelopmentLevels::PerAge),
        draws(DevelopmentLevels::ByYear)
    );
}

/// The warning of cumulative paid on cells finer than a year: on
/// quarterly, monthly and semiannual development, not on annual, not
/// under incremental paid, not when paid is not fitted.
#[test]
fn cumulative_paid_on_sub_annual_cells_warns() {
    let grain = |d| SquareGrain::new(Grain::Year, d).unwrap();
    let cumulative = Response::MODEL_1;
    let incremental = Response::MODEL_2;
    for d in [Grain::Quarter, Grain::Month, Grain::Semester] {
        let w = cumulative_sub_annual_warning(cumulative, grain(d)).unwrap();
        assert!(
            w.contains("cumulative paid") && w.contains("Check the residuals"),
            "{w}"
        );
        assert!(!w.contains("wrong"), "{w}");
        assert!(cumulative_sub_annual_warning(incremental, grain(d)).is_none());
    }
    assert!(cumulative_sub_annual_warning(cumulative, grain(Grain::Year)).is_none());
    let no_paid = Response {
        paid: false,
        ..cumulative
    };
    assert!(cumulative_sub_annual_warning(no_paid, grain(Grain::Quarter)).is_none());
}

/// The prior predictive check on the quarterly triangle draws the
/// square of 5 origins by 20 quarterly ages.
#[test]
fn the_prior_check_draws_the_quarterly_square() {
    let tri = quarterly_triangle(false);
    let paths = preset().prior_predictive(&tri, "premium", 50).unwrap();
    assert_eq!(paths.paid.components().len(), 5 * 20);
    assert_eq!(paths.outstanding.components().len(), 5 * 20);
    assert_eq!(
        paths.paid.components()[19],
        vec![Period::year(1988).into(), KeyValue::Int(60)]
    );
    // Every path starts at 0 and the paid rises along each origin.
    let draws = paths.paid.draw_matrix();
    for row in draws.as_chunks::<100>().0 {
        for i in 0..5 {
            let r = &row[i * 20..(i + 1) * 20];
            assert!(r.windows(2).all(|w| w[1] >= w[0]), "{r:?}");
        }
    }
}

/// A fit on the quarterly triangle: reserves to 60 months, future cells
/// keyed by their age in months, and the coming year of four quarters
/// of each type for every open origin.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_fit_on_unequal_grain() {
    use crate::bayes::one_year::YearModel;
    let tri = quarterly_triangle(false);
    let fit = Compartmental {
        fitted: Fitted::PaidAndIncurred,
        ..quick(4)
    }
    .fit(&tri, "paid", incurred(), "premium")
    .unwrap();
    let ages: std::collections::BTreeSet<i64> = fit
        .paid
        .components()
        .iter()
        .map(|k| match k[1] {
            KeyValue::Int(a) => a,
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(ages.first(), Some(&15));
    assert_eq!(ages.last(), Some(&60));
    assert_eq!(fit.paid.components().len(), 4 + 8 + 12 + 16);
    // Cumulative paid on quarterly cells warns, once, and the warning
    // says to check and does not call the fit wrong.
    let cumulative_warnings: Vec<&String> = fit
        .diagnostics
        .warnings
        .iter()
        .filter(|w| w.contains("cumulative paid is fitted on"))
        .collect();
    assert_eq!(
        cumulative_warnings.len(),
        1,
        "{:?}",
        fit.diagnostics.warnings
    );
    assert!(cumulative_warnings[0].contains("fitted on Q development"));
    // 1988 is at the square's last age.
    let r = fit
        .reserves
        .marginal(&vec![Period::year(1988).into()])
        .unwrap();
    assert_eq!(r.mean(), 0.0);
    let total = fit.reserves.total().mean();
    assert!(total > 0.0 && total.is_finite(), "{total}");
    let data = Data::new(&tri, "paid", incurred(), "premium", Fitted::PaidAndIncurred).unwrap();
    let m = quick(4);
    let year = year::CompartmentalYear::new(&m, &data).unwrap();
    // 1989 to 1992, four quarters each, paid and outstanding.
    assert_eq!(year.n_year(), 2 * 4 * 4);
}

/// `fit_segments` on two lines of the quarterly triangle: each segment
/// is fitted on its own square of 20 quarterly ages.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn segments_on_unequal_grain() {
    let tri = quarterly_lines(&["A", "B"], false);
    let fits = quick(2)
        .fit_segments(&tri, "paid", incurred(), "premium")
        .unwrap();
    assert_eq!(fits.segments.len(), 2);
    for s in &fits.segments.fits {
        assert_eq!(s.fit.paid.components().len(), 4 + 8 + 12 + 16);
        assert!(s.fit.reserves.total().mean() > 0.0);
    }
    assert_eq!(fits.reserves.n_components(), 2 * 5);
}

/// The one-year conformance suite on the quarterly triangle, models 1
/// and 2 (paid and incurred): the year holds four quarterly cells of
/// each type for each open origin, chained as increments under model
/// 2; every origin and age has a cell, so no level is added.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_on_unequal_grain_conforms() {
    for (name, preset) in [
        ("compartmental quarterly", preset()),
        ("compartmental model 2 quarterly", preset_2()),
    ] {
        one_year_conformance_on(
            name,
            Compartmental {
                fitted: Fitted::PaidAndIncurred,
                ..preset
            },
            &quarterly_triangle(false),
            &quarterly_triangle(true),
            0,
            2 * 4 * 4,
        );
    }
}

/// The one-year conformance suite on model 2 with levels by development
/// year, on the quarterly triangle. In the first, every year has an
/// observed cell, so no level is added. In the second the oldest origin
/// stops at 48 months, so its coming year (51 to 60 months) opens the
/// fifth development year: one new level, 4 effects added to each draw
/// (against 16 for the four ages under levels per age), and the year's
/// cells (5 origins, 4 quarters, paid and incurred) fall in observed
/// years and in the new one.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_by_development_year_conforms() {
    let by_year = || Compartmental {
        fitted: Fitted::PaidAndIncurred,
        development_levels: DevelopmentLevels::ByYear,
        ..preset_2()
    };
    one_year_conformance_on(
        "compartmental model 2 by year quarterly",
        by_year(),
        &quarterly_triangle(false),
        &quarterly_triangle(true),
        0,
        2 * 4 * 4,
    );
    one_year_conformance_on(
        "compartmental model 2 by year, a year opens a level",
        by_year(),
        &quarterly_lines_with(&[""], false, 48, 0.3),
        &quarterly_triangle(true),
        4,
        5 * 4 * 2,
    );
}

/// At the annual grain levels by year are the model itself: the same
/// posterior and the same reserves draw for draw at one seed.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn at_the_annual_grain_the_widths_fit_identically() {
    let tri = triangle(true);
    let fit = |levels| {
        Compartmental {
            sampler: Sampler {
                chains: 2,
                tune: 60,
                draws: 60,
                target_accept: 0.9,
                ..preset_2().sampler
            },
            n_sims: 200,
            seed: 6,
            development_levels: levels,
            ..preset_2()
        }
        .fit(&tri, "paid", incurred(), "premium")
        .unwrap()
    };
    let (per_age, by_year) = (
        fit(DevelopmentLevels::PerAge),
        fit(DevelopmentLevels::ByYear),
    );
    assert_eq!(per_age.posterior.values(), by_year.posterior.values());
    assert_eq!(per_age.posterior.names(), by_year.posterior.names());
    assert_eq!(
        per_age.reserves.draw_matrix(),
        by_year.reserves.draw_matrix()
    );
    assert_eq!(per_age.paid.draw_matrix(), by_year.paid.draw_matrix());
    assert_eq!(
        per_age.loo().unwrap().loo.estimate.elpd,
        by_year.loo().unwrap().loo.estimate.elpd
    );
    // The warning is not there at the annual grain, whichever the paid.
    let cumulative = quick(3).fit(&tri, "paid", incurred(), "premium").unwrap();
    for fit in [&per_age, &by_year, &cumulative] {
        assert!(
            !fit.diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("cumulative paid is fitted")),
            "{:?}",
            fit.diagnostics.warnings
        );
    }
}

/// Model 2 with levels by year on the quarterly triangle: a fit with
/// fewer parameters than per age (5 levels of 4 effects, not 20), each
/// level named by its first age, future cells to 60 months, and no
/// warning about cumulative paid (model 2 fits increments).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn model_2_by_development_year_fits_the_quarterly_triangle() {
    let tri = quarterly_triangle(false);
    let fit = |levels| {
        Compartmental {
            fitted: Fitted::PaidAndIncurred,
            development_levels: levels,
            sampler: Sampler {
                chains: 2,
                tune: 60,
                draws: 60,
                target_accept: 0.9,
                ..preset_2().sampler
            },
            n_sims: 200,
            seed: 4,
            ..preset_2()
        }
        .fit(&tri, "paid", incurred(), "premium")
        .unwrap()
    };
    let (per_age, by_year) = (
        fit(DevelopmentLevels::PerAge),
        fit(DevelopmentLevels::ByYear),
    );
    assert_eq!(per_age.posterior.dim() - by_year.posterior.dim(), 4 * 15);
    let names = by_year.posterior.names();
    assert!(names.iter().any(|n| n == "dev_ker[15]"), "{names:?}");
    assert!(!names.iter().any(|n| n == "dev_ker[18]"));
    assert_eq!(by_year.paid.components().len(), 4 + 8 + 12 + 16);
    let total = by_year.reserves.total().mean();
    assert!(total.is_finite() && total > 0.0, "{total}");
    assert!(by_year.loo().unwrap().loo.estimate.elpd.is_finite());
    for fit in [&per_age, &by_year] {
        assert!(
            !fit.diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("cumulative paid is fitted")),
            "{:?}",
            fit.diagnostics.warnings
        );
    }
}

#[test]
fn the_coming_year_of_the_training_triangle() {
    use crate::bayes::one_year::YearModel;
    for (preset, cells) in [(preset(), 20), (preset_2(), 20)] {
        let data = Data::new(
            &triangle(true),
            "paid",
            incurred(),
            "premium",
            Fitted::PaidAndOutstanding,
        )
        .unwrap();
        let year = year::CompartmentalYear::new(&preset, &data).unwrap();
        // Every origin's next diagonal, paid and outstanding: 1988 at
        // 120 months, ..., 1997 at 12 months.
        assert_eq!(year.n_year(), cells);
        let data = Data::new(
            &triangle(false),
            "paid",
            incurred(),
            "premium",
            Fitted::PaidAndOutstanding,
        )
        .unwrap();
        // Without 1997 the square has 9 ages and 1988 is at the last.
        assert_eq!(
            year::CompartmentalYear::new(&preset, &data)
                .unwrap()
                .n_year(),
            16
        );
        let data = Data::new(
            &closed_triangle(),
            "paid",
            incurred(),
            "premium",
            Fitted::PaidAndOutstanding,
        )
        .unwrap();
        assert_eq!(
            year::CompartmentalYear::new(&preset, &data)
                .unwrap()
                .n_year(),
            0
        );
    }
}

/// The one-year conformance suite (`one_year::conformance`) on
/// `preset` with a small sampler: the training triangle with 1997, and
/// the closed square of [`closed_triangle`], each sampled directly (the
/// closed square has fewer cells than model 2 has parameters, which
/// `fit` refuses but the posterior does not need). Then the light
/// refit's own check ([`light_refit_conforms`]).
fn one_year_conformance(name: &str, preset: Compartmental) {
    // 1997's two `z` (and, with development effects, age 10's) are
    // added to each draw; each fitted type's next diagonal has 10
    // cells.
    let added = if preset.priors.hierarchy() == Hierarchy::Model1 {
        2
    } else {
        4 + 4
    };
    let n_year = 10 * preset.response().n_sigma();
    one_year_conformance_on(
        name,
        preset,
        &triangle(true),
        &closed_triangle(),
        added,
        n_year,
    );
}

/// [`one_year_conformance`] on the triangle `open` and the closed
/// square `closed`, with `added` levels no cell informs and `n_year`
/// cells in the year.
fn one_year_conformance_on(
    name: &str,
    preset: Compartmental,
    open_triangle: &Triangle,
    closed_triangle: &Triangle,
    added: usize,
    n_year: usize,
) {
    use crate::bayes::one_year::YearModel;
    let m = Compartmental {
        sampler: Sampler {
            chains: 2,
            tune: 100,
            draws: 100,
            target_accept: 0.9,
            ..preset.sampler
        },
        ..preset
    };
    let fitted = |tri: &Triangle| {
        let data = Data::new(tri, "paid", incurred(), "premium", m.fitted).unwrap();
        let ordered = m.rate_order == RateOrder::Ordered;
        let model = Model::new(&data, m.priors, ordered, m.response()).unwrap();
        let (_, sampled, _) = m.sample(&model, 5).unwrap();
        (data, model, sampled)
    };
    let (open, closed) = (fitted(open_triangle), fitted(closed_triangle));
    let mut year = year::CompartmentalYear::new(&m, &open.0).unwrap();
    let opening = year.opening_draws(&open.1, &open.2, 9).unwrap();
    let closed_year = year::CompartmentalYear::new(&m, &closed.0).unwrap();
    let closed_opening = closed_year.opening_draws(&closed.1, &closed.2, 9).unwrap();
    assert_eq!(opening.dim(), open.2.dim() + added);
    assert_eq!(year.n_year(), n_year);
    crate::bayes::one_year::conformance::check(
        name,
        &year,
        &opening,
        &closed_year,
        &closed_opening,
    );
    light_refit_conforms(name, &m, &open.0, &mut year, &opening);
}

/// The light refit (decision 24) samples the posterior the fit's own
/// sampler does, on the same layout: one year drawn from the opening
/// posterior, refitted by the fit's sampler (4 chains of 300 and 500,
/// the reference) and by the default light sampler, has the opening's
/// column names both ways, closing reserves whose mean agrees within
/// 0.3 of their sd and whose sd agrees within 25% (the Monte Carlo
/// error of the two is about 0.1 sd and 5%), and light draws, mapped
/// back to the centred scale, whose score under `log_posterior` with
/// that year has mean zero (`conformance::score_z_at`). A warm start
/// mis-centred, mis-scaled or mapped back wrongly fails one of them.
/// A light refit of 2 draws warns.
fn light_refit_conforms(
    name: &str,
    m: &Compartmental,
    data: &Data,
    year: &mut year::CompartmentalYear,
    opening: &PosteriorDraws,
) {
    use crate::bayes::one_year::YearModel;
    use crate::bayes::one_year::conformance::{SCORE_Z, score_z_at};
    let mut cells = vec![0.0; year.n_year()];
    year.draw_year(opening.draw(0), &mut StreamRng::new(13, 0), &mut cells);
    let reference = Compartmental {
        sampler: Sampler {
            chains: 4,
            tune: 300,
            draws: 500,
            ..m.sampler
        },
        ..m.clone()
    };
    let (full, _) = year::CompartmentalYear::new(&reference, data)
        .unwrap()
        .refit(&cells, 21)
        .unwrap();
    year.warm_start(Some(LightRefit::default()), opening)
        .unwrap();
    let (light, _) = year.refit(&cells, 21).unwrap();
    assert_eq!(light.names(), opening.names(), "{name}: light refit");
    assert_eq!(full.names(), opening.names(), "{name}: full refit");
    let reserves = |draws: &PosteriorDraws| -> (f64, f64) {
        let mut row = vec![0.0; year.n_origins()];
        let t: Vec<f64> = (0..draws.n_draws())
            .map(|d| {
                year.closing(draws.draw(d), &cells, &mut row);
                row.iter().sum()
            })
            .collect();
        let n = t.len() as f64;
        let mean = t.iter().sum::<f64>() / n;
        let sd = (t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        (mean, sd)
    };
    let ((mf, sf), (ml, sl)) = (reserves(&full), reserves(&light));
    assert!(
        (ml - mf).abs() < 0.3 * sf && (sl / sf - 1.0).abs() < 0.25,
        "{name}: closing reserve {ml} (sd {sl}) light, {mf} (sd {sf}) full"
    );
    let z = score_z_at(&*year, &light, &cells);
    let worst = z.iter().copied().fold(0.0, |w: f64, v| w.max(v.abs()));
    assert!(
        worst < SCORE_Z,
        "{name}: the light refit's score has mean z {z:?}"
    );
    // The warning rule fires on a run too short for 100 effective
    // draws of the closing reserve.
    let tiny = LightRefit {
        chains: 1,
        tune: 1,
        draws: 2,
    };
    year.warm_start(Some(tiny), opening).unwrap();
    let (_, warned) = year.refit(&cells, 21).unwrap();
    assert!(warned, "{name}: a light refit of 2 draws did not warn");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_of_model_1_conforms() {
    one_year_conformance("compartmental model 1", preset());
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_of_paid_and_incurred_conforms() {
    one_year_conformance(
        "compartmental paid and incurred",
        Compartmental {
            fitted: Fitted::PaidAndIncurred,
            ..preset()
        },
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_of_paid_alone_conforms() {
    one_year_conformance(
        "compartmental paid alone",
        Compartmental {
            fitted: Fitted::PaidOnly,
            rate_order: RateOrder::Ordered,
            ..preset()
        },
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_of_incurred_alone_conforms() {
    one_year_conformance(
        "compartmental incurred alone",
        Compartmental {
            fitted: Fitted::IncurredOnly,
            rate_order: RateOrder::Ordered,
            ..preset()
        },
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_of_model_2_conforms() {
    one_year_conformance("compartmental model 2", preset_2());
}

/// The opening reserve, the posterior mean over the draws with 1997's
/// effects added, is the fit's mean reserve, whose simulations draw
/// them from the population, within Monte Carlo error; the tables have
/// one row per premium origin.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_opening_reserve_is_the_fits() {
    use crate::bayes::one_year::YearModel;
    let only = BayesOneYear {
        refit: Some(OneYearRefit::ReweightOnly),
        // Above the default cap of `ONE_YEAR_SIMS`.
        n_sims: Some(4000),
        ..BayesOneYear::default()
    };
    for preset in [preset(), preset_2()] {
        let m = Compartmental {
            sampler: Sampler {
                chains: 2,
                tune: 100,
                draws: 100,
                target_accept: 0.9,
                ..preset.sampler
            },
            n_sims: 4000,
            seed: 2,
            ..preset
        };
        let view = m
            .one_year(&triangle(true), "paid", incurred(), "premium", &only)
            .unwrap();
        let total = view.fit.reserves.total();
        let opening: f64 = view.opening_reserve.iter().sum();
        // The error of both means: the simulations', and the draws'
        // (each with its own 1997 effects) of the opening.
        let year = year::CompartmentalYear::new(&m, &view.fit.data).unwrap();
        let draws = year
            .opening_draws(&view.fit.model, &view.fit.sampled, 1)
            .unwrap();
        let mut row = vec![0.0; 10];
        let by_draw: Vec<f64> = (0..draws.n_draws())
            .map(|d| {
                year.opening(draws.draw(d), &mut row);
                row.iter().sum()
            })
            .collect();
        let s = by_draw.len() as f64;
        let mean_d = by_draw.iter().sum::<f64>() / s;
        let var_d = by_draw.iter().map(|v| (v - mean_d).powi(2)).sum::<f64>() / s;
        let se = (total.variance() / m.n_sims as f64 + var_d / s).sqrt();

        assert!(
            (opening - total.mean()).abs() < 4.0 * se,
            "opening {opening}, fit {} (se {se})",
            total.mean()
        );
        assert_eq!(view.cdr.n_components(), 10);
        assert_eq!(view.pareto_k.len(), 4000);
        assert!(view.refitted.is_empty() && view.moment_matched.is_empty());
        assert!(view.cdr.std_dev() < view.fit.reserves.std_dev());
        let long = view.to_long();
        assert_eq!(long.origin.as_ref().unwrap().len(), 10);
        let totals = view.totals();
        assert!(totals.values.iter().any(|(n, _)| n == "flagged_share"));
        assert!(view.cdr.provenance().model.ends_with("_one_year"));
    }
}

/// A light refit (decision 24) refits every simulation with a shorter,
/// warm-started sampler: the same years and `k̂` as the fit's own sampler,
/// the same simulations refitted, closing reserves of the same size,
/// reproducible, and recorded in the provenance; a bad setting is an
/// error before any sampling.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_light_refit_refits_every_simulation_on_a_warm_start() {
    let m = Compartmental {
        n_sims: 6,
        ..quick(4)
    };
    let with = |light_refit: Option<LightRefit>| {
        m.one_year(
            &triangle(true),
            "paid",
            incurred(),
            "premium",
            &BayesOneYear {
                refit: Some(OneYearRefit::RefitAll),
                light_refit,
                ..BayesOneYear::default()
            },
        )
        .unwrap()
    };
    let light = LightRefit {
        chains: 2,
        tune: 60,
        draws: 60,
    };
    let (full, a, b) = (with(None), with(Some(light)), with(Some(light)));
    assert_eq!(a.refitted, full.refitted);
    assert_eq!(a.refitted.len(), 6);
    assert_eq!(a.pareto_k, full.pareto_k);
    // Reproducible, and a different sampler than the fit's.
    assert_eq!(a.cdr.draw_matrix(), b.cdr.draw_matrix());
    assert_ne!(a.cdr.draw_matrix(), full.cdr.draw_matrix());
    // Both close the same reserves, to Monte Carlo error: each total
    // within a generous bound of the full refit's.
    let sd = full.cdr.std_dev();
    for (x, y) in a
        .cdr
        .draw_matrix()
        .chunks(10)
        .zip(full.cdr.draw_matrix().chunks(10))
    {
        let (x, y): (f64, f64) = (x.iter().sum(), y.iter().sum());
        assert!(x.is_finite() && (x - y).abs() < 3.0 * sd, "{x} against {y}");
    }
    assert!(
        a.cdr
            .provenance()
            .parameters
            .contains(&("light_refit".into(), "2x60+60".into()))
    );
    assert!(
        !full
            .cdr
            .provenance()
            .parameters
            .iter()
            .any(|(k, _)| k == "light_refit")
    );
    for bad in [(0, 60, 60), (2, 0, 60), (2, 60, 1)] {
        let settings = BayesOneYear {
            light_refit: Some(LightRefit {
                chains: bad.0,
                tune: bad.1,
                draws: bad.2,
            }),
            ..BayesOneYear::default()
        };
        assert!(matches!(
            m.one_year(&triangle(true), "paid", incurred(), "premium", &settings),
            Err(Error::InvalidSetting {
                name: "light_refit",
                ..
            })
        ));
    }
}

/// A refit budget refits the first simulations due for one, in
/// order, and reweights and lists the others; the years, the `k̂` and
/// the simulations within the budget are those without one, and those
/// beyond it are `ReweightOnly`'s, bit for bit.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_refit_budget_refits_the_first_and_lists_the_rest() {
    let m = Compartmental {
        n_sims: 12,
        ..quick(4)
    };
    let with = |refit: OneYearRefit, max_refits: Option<usize>| {
        m.one_year(
            &triangle(true),
            "paid",
            incurred(),
            "premium",
            &BayesOneYear {
                refit: Some(refit),
                max_refits,
                ..BayesOneYear::default()
            },
        )
        .unwrap()
    };
    let view = |max_refits: Option<usize>| with(OneYearRefit::RefitFlagged, max_refits);
    let all = view(None);
    assert!(all.refitted.len() >= 3, "{:?}", all.pareto_k);
    assert!(all.unrefitted.is_empty());
    let two = view(Some(2));
    assert_eq!(two.pareto_k, all.pareto_k);
    assert_eq!(two.refitted, all.refitted[..2]);
    assert_eq!(two.unrefitted, all.refitted[2..]);
    let n = 10;
    for (a, b) in two
        .cdr
        .draw_matrix()
        .chunks(n)
        .zip(all.cdr.draw_matrix().chunks(n))
        .enumerate()
        .filter(|(i, _)| !two.unrefitted.contains(i))
        .map(|(_, r)| r)
    {
        assert_eq!(a, b);
    }
    let none = view(Some(0));
    assert!(none.refitted.is_empty());
    assert_eq!(none.unrefitted, all.refitted);
    let only = with(OneYearRefit::ReweightOnly, None);
    assert_eq!(none.cdr.draw_matrix(), only.cdr.draw_matrix());
    for (a, b) in two
        .cdr
        .draw_matrix()
        .chunks(n)
        .zip(only.cdr.draw_matrix().chunks(n))
        .enumerate()
        .filter(|(i, _)| two.unrefitted.contains(i))
        .map(|(_, r)| r)
    {
        assert_eq!(a, b);
    }
    assert!(
        none.cdr
            .provenance()
            .parameters
            .contains(&("max_refits".into(), "0".into()))
    );
}

/// The first five origins of the training triangle, ages 1 to 5 up to
/// the diagonal, once for each of `names` (a key `lob` when `keyed`).
fn lines(names: &[&str], keyed: bool) -> Triangle {
    let (mut lob, mut origin, mut ages) = (vec![], vec![], vec![]);
    let (mut paid, mut inc, mut prem) = (vec![], vec![], vec![]);
    for name in names {
        for w in 0..5 {
            for d in 0..5 - w {
                lob.push(*name);
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as Lag + 1));
                paid.push(PAID[w][d]);
                inc.push(INCURRED[w][d]);
                prem.push(PREMIUM[w]);
            }
        }
    }
    let key = [("lob", &lob[..])];
    Triangle::from_long(&Long {
        keys: if keyed { &key } else { &[] },
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("incurred", &inc), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn segments_of(m: &Compartmental, tri: &Triangle) -> CompartmentalFits {
    m.fit_segments(tri, "paid", incurred(), "premium").unwrap()
}

/// Each line's total of `pd` (dimensions `lob`, `origin`), simulation by
/// simulation.
fn line_totals(pd: &PredictiveDistribution) -> Vec<Vec<f64>> {
    let by = pd.aggregate(&["lob"]).unwrap();
    let n = by.n_components();
    (0..n)
        .map(|j| by.draw_matrix().chunks(n).map(|r| r[j]).collect())
        .collect()
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let ranks = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![0.0; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (rx, ry) = (ranks(x), ranks(y));
    let m = (x.len() as f64 - 1.0) / 2.0;
    let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
    let sxx: f64 = rx.iter().map(|a| (a - m).powi(2)).sum();
    sxy / sxx
}

/// `Synchronized` and a bad matrix fail before any sampling (the test
/// would take minutes if either sampled), `Synchronized` as
/// `Error::Bayes` and a bad matrix as `InvalidSetting`.
#[test]
fn synchronized_and_a_bad_matrix_fail_before_sampling() {
    let tri = lines(&["A", "B"], true);
    let run = |dependence| {
        Compartmental {
            dependence,
            ..quick(1)
        }
        .fit_segments(&tri, "paid", incurred(), "premium")
        .unwrap_err()
    };
    assert!(matches!(
        run(SegmentDependence::Synchronized),
        Error::NotSupported {
            setting: "Synchronized dependence",
            ..
        }
    ));
    for spearman in [
        vec![1.0, 0.5],
        vec![1.0, 0.5, 0.4, 1.0],
        vec![2.0, 0.5, 0.5, 1.0],
        vec![1.0, 1.1, 1.1, 1.0],
    ] {
        assert!(
            matches!(
                run(SegmentDependence::RankCorrelation { spearman }),
                Error::InvalidSetting {
                    name: "spearman",
                    ..
                }
            ),
            "bad matrix"
        );
    }
}

/// Each segment samples and simulates from its own streams, `fit` does
/// not change, and a segment is `fit` on its own triangle with that
/// segment's streams.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn each_segment_has_its_own_seeds_and_fit_is_unchanged() {
    let tri = lines(&["A", "B"], true);
    let m = quick(1);
    let fits = segments_of(&m, &tri);
    assert_eq!(fits.segments.len(), 2);
    let [a, b] = [0, 1].map(|k| &fits.segments.fits[k].fit);
    // "B" is "A" under another name: with the same streams the posteriors
    // would agree draw by draw; with their own they do not.
    assert_ne!(a.posterior, b.posterior);
    let only = |lob: &str| tri.select(&[("lob", &[lob][..])]).unwrap();
    for (k, lob) in [(0, "A"), (1, "B")] {
        let alone = m
            .fit_streams(
                &only(lob),
                "paid",
                incurred(),
                "premium",
                Stream::SegmentChains(k),
                Stream::SegmentPredictive(k),
            )
            .unwrap();
        let seg = &fits.segments.fits[k].fit;
        assert_eq!(alone.posterior, seg.posterior);
        assert_eq!(alone.reserves.draw_matrix(), seg.reserves.draw_matrix());
        assert_eq!(alone.paid.draw_matrix(), seg.paid.draw_matrix());
        assert_eq!(
            alone.outstanding.as_ref().unwrap().draw_matrix(),
            seg.outstanding.as_ref().unwrap().draw_matrix()
        );
    }
    // `fit` is the base streams' fit.
    let base = m.fit(&only("A"), "paid", incurred(), "premium").unwrap();
    let again = m
        .fit_streams(
            &only("A"),
            "paid",
            incurred(),
            "premium",
            Stream::Chains,
            Stream::Predictive,
        )
        .unwrap();
    assert_eq!(base.posterior, again.posterior);
    assert_eq!(base.reserves.draw_matrix(), again.reserves.draw_matrix());
    assert_ne!(base.posterior, a.posterior);
    // The joint results.
    assert_eq!(fits.reserves.dims(), ["lob", "origin"]);
    assert_eq!(fits.reserves.n_components(), 10);
    assert_eq!(fits.paid.dims(), ["lob", "origin", "age"]);
    assert_eq!(fits.paid.n_components(), 2 * a.paid.n_components());
    assert_eq!(
        fits.outstanding.as_ref().unwrap().n_components(),
        2 * a.outstanding.as_ref().unwrap().n_components()
    );
    let long = fits.to_long();
    assert_eq!(long.keys[0].1.len(), 10);
    assert_eq!(long.n_rows(), 10);
    assert_eq!(fits.totals().n_rows(), 2);
    assert_eq!(
        fits.paid_table().unwrap().n_rows(),
        fits.paid.n_components()
    );
    assert_eq!(fits.outstanding_table().unwrap().keys[0].1[0], "A");
    let one = fits.segment(&[("lob", "B")]).unwrap();
    assert_eq!(one.reserves.n_components(), 5);
    assert_eq!(one.reserves.dims(), ["lob", "origin"]);
    let param = |pd: &PredictiveDistribution, key: &str| {
        pd.provenance()
            .parameters
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(param(&fits.reserves, "segments").as_deref(), Some("2"));
    assert_eq!(param(&fits.paid, "output").as_deref(), Some("paid"));
    assert_eq!(
        param(&fits.reserves, "dependence").as_deref(),
        Some("Independent")
    );
}

/// A triangle without keys is one segment, fitted with segment 0's
/// streams.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_one_segment_fit_is_fit_with_segment_zero_streams() {
    let tri = lines(&["A"], false);
    let m = quick(2);
    let fits = segments_of(&m, &tri);
    assert_eq!(fits.segments.len(), 1);
    let alone = m
        .fit_streams(
            &tri,
            "paid",
            incurred(),
            "premium",
            Stream::SegmentChains(0),
            Stream::SegmentPredictive(0),
        )
        .unwrap();
    let seg = &fits.segments.fits[0].fit;
    assert_eq!(alone.posterior, seg.posterior);
    assert_eq!(alone.reserves.draw_matrix(), seg.reserves.draw_matrix());
    assert_eq!(fits.reserves.draw_matrix(), seg.reserves.draw_matrix());
    assert_eq!(fits.paid.draw_matrix(), seg.paid.draw_matrix());
    assert_ne!(
        m.fit(&tri, "paid", incurred(), "premium")
            .unwrap()
            .posterior,
        seg.posterior
    );
}

/// The rank correlation reorders whole rows: each segment keeps its
/// simulations (as a set), its reserves stay its paid at the last age
/// less the latest paid in every simulation, and the totals reach the
/// target.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_joint_reorder_keeps_each_segments_rows_consistent() {
    let tri = lines(&["A", "B"], true);
    let m = quick(3);
    let independent = segments_of(&m, &tri);
    let ranked = segments_of(
        &Compartmental {
            dependence: SegmentDependence::RankCorrelation {
                spearman: vec![1.0, 0.8, 0.8, 1.0],
            },
            ..m.clone()
        },
        &tri,
    );
    let sorted = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v
    };
    let (ti, tr) = (
        line_totals(&independent.reserves),
        line_totals(&ranked.reserves),
    );
    for s in 0..2 {
        assert_eq!(sorted(ti[s].clone()), sorted(tr[s].clone()));
    }
    // The first segment's rows are shuffled, not left sorted.
    assert!(tr[0].windows(2).any(|w| w[0] > w[1]));
    let r = spearman(&ti[0], &ti[1]);
    assert!(r.abs() < 0.2, "independent lines: {r}");
    let r = spearman(&tr[0], &tr[1]);
    assert!((r - 0.8).abs() < 0.05, "ranked lines: {r}");
    // Every segment's own parts agree, simulation by simulation.
    for (s, seg) in ranked.segments.fits.iter().enumerate() {
        let f = &seg.fit;
        let n_sims = f.reserves.n_sims();
        let n = f.reserves.n_components();
        let paid_n = f.paid.n_components();
        for (i, origin) in f.origins.iter().enumerate() {
            let Some(last) = f
                .paid
                .components()
                .iter()
                .position(|k| k[0] == KeyValue::from(*origin) && k[1] == KeyValue::Int(60))
            else {
                continue;
            };
            for sim in 0..n_sims {
                let reserve = f.reserves.draw_matrix()[sim * n + i];
                let at_last = f.paid.draw_matrix()[sim * paid_n + last];
                assert!(
                    (at_last - f.latest_paid[i] - reserve).abs() < 1e-6 * at_last.abs().max(1.0),
                    "segment {s} origin {origin} simulation {sim}"
                );
            }
        }
        // The joint distributions hold the segment's parts.
        let joint_n = ranked.reserves.n_components();
        for sim in 0..n_sims {
            assert_eq!(
                &ranked.reserves.draw_matrix()[sim * joint_n + s * n..sim * joint_n + (s + 1) * n],
                &f.reserves.draw_matrix()[sim * n..(sim + 1) * n]
            );
        }
    }
}

fn reweight_only(n_sims: Option<usize>) -> BayesOneYear {
    BayesOneYear {
        refit: Some(OneYearRefit::ReweightOnly),
        n_sims,
        ..BayesOneYear::default()
    }
}

/// `Synchronized`, a bad matrix and a NaN threshold fail before any
/// sampling (the test would take minutes if any sampled).
#[test]
fn the_one_year_view_over_segments_fails_before_sampling() {
    let tri = lines(&["A", "B"], true);
    let run = |dependence, settings: &BayesOneYear| {
        Compartmental {
            dependence,
            ..quick(1)
        }
        .one_year_segments(&tri, "paid", incurred(), "premium", settings)
        .unwrap_err()
    };
    let default = BayesOneYear::default();
    assert!(matches!(
        run(SegmentDependence::Synchronized, &default),
        Error::NotSupported {
            setting: "Synchronized dependence",
            ..
        }
    ));
    for spearman in [
        vec![1.0, 0.5],
        vec![1.0, 0.5, 0.4, 1.0],
        vec![1.0, 1.1, 1.1, 1.0],
    ] {
        assert!(matches!(
            run(SegmentDependence::RankCorrelation { spearman }, &default),
            Error::InvalidSetting {
                name: "spearman",
                ..
            }
        ));
    }
    let nan = BayesOneYear {
        k_threshold: f64::NAN,
        ..default
    };
    assert!(matches!(
        run(SegmentDependence::Independent, &nan),
        Error::InvalidSetting {
            name: "k_threshold",
            ..
        }
    ));
}

/// Each segment's view is `one_year` of its own triangle on its own
/// streams, the segments share no random number, a one-segment triangle
/// is segment 0's view (and not `one_year`'s), `n_sims` and the tables
/// are laid out as for `fit_segments`, and the independent lines are
/// uncorrelated.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_over_segments_runs_each_segment_on_its_own_streams() {
    let tri = lines(&["A", "B"], true);
    let m = quick(4);
    let only = reweight_only(None);
    let views = m
        .one_year_segments(&tri, "paid", incurred(), "premium", &only)
        .unwrap();
    assert_eq!(views.cdr.dims(), ["lob", "origin"]);
    assert_eq!(views.cdr.n_components(), 10);
    assert_eq!(views.cdr.n_sims(), 500);
    assert_eq!(views.approximate, None);
    let lifetime = segments_of(&m, &tri);
    let (settings, n_sims, _) = m.one_year_defaults(&only);
    for (k, lob) in [(0, "A"), (1, "B")] {
        let seg = &views.segments.fits[k];
        // The posterior is the lifetime fit's, draw for draw.
        assert_eq!(
            seg.fit.fit.posterior,
            lifetime.segments.fits[k].fit.posterior
        );
        let one = tri.select(&[("lob", &[lob][..])]).unwrap();
        let (fit, year) = m
            .year_streams(
                &one,
                "paid",
                incurred(),
                "premium",
                &settings,
                n_sims,
                Stream::SegmentChains(k),
                Stream::SegmentPredictive(k),
            )
            .unwrap();
        assert_eq!(fit.posterior, seg.fit.fit.posterior);
        assert_eq!(year.pareto_k, seg.pareto_k);
        assert_eq!(year.opening, seg.opening_reserve);
        let range = views.range(k);
        let cdr: Vec<f64> = views
            .cdr
            .draw_matrix()
            .chunks(10)
            .flat_map(|r| r[range.clone()].iter().copied())
            .collect();
        assert_eq!(year.cdr, cdr);
    }
    // "B" is "A" under another name: its own streams, its own view.
    let (a, b) = (&views.segments.fits[0], &views.segments.fits[1]);
    assert_ne!(a.pareto_k, b.pareto_k);
    assert_ne!(a.opening_reserve, b.opening_reserve);
    // 1988 is at the square's last age: no cell in the year.
    assert!(
        views
            .cdr
            .draw_matrix()
            .chunks(10)
            .all(|r| r[0] == 0.0 && r[5] == 0.0)
    );
    let totals = line_totals(&views.cdr);
    let r = spearman(&totals[0], &totals[1]);
    assert!(r.abs() < 4.5 / 500f64.sqrt(), "independent lines: {r}");
    // The tables.
    let long = views.to_long();
    assert_eq!(long.n_rows(), 10);
    assert_eq!(long.keys[0].1[0], "A");
    assert!(long.column("opening_reserve").is_some() && long.column("cdr_std_dev").is_some());
    let totals = views.totals();
    assert_eq!(totals.n_rows(), 2);
    let share = totals.column("flagged_share").unwrap();
    assert_eq!(share[1], b.flagged_share());
    let cdr_mean = totals.column("cdr_mean").unwrap();
    let line_b: Vec<f64> = line_totals(&views.cdr)[1].clone();
    let mean_b = line_b.iter().sum::<f64>() / line_b.len() as f64;
    assert!((cdr_mean[1] - mean_b).abs() < 1e-9 * (1.0 + mean_b.abs()));
    let one = views.segment(&[("lob", "B")]).unwrap();
    assert_eq!(one.cdr.n_components(), 5);
    assert_eq!(one.segments.fits[0].pareto_k, b.pareto_k);
    assert_eq!(
        one.cdr.draw_matrix().chunks(5).next().unwrap(),
        &views.cdr.draw_matrix()[5..10]
    );
    let param = |name: &str| {
        views
            .cdr
            .provenance()
            .parameters
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(param("segments").as_deref(), Some("2"));
    assert_eq!(param("dependence").as_deref(), Some("Independent"));
    assert_eq!(param("one_year_sims").as_deref(), Some("500"));
    assert_eq!(
        views.cdr.provenance().model,
        format!("{}_one_year", lifetime.reserves.provenance().model)
    );
    // `max_refits` is a budget of each segment: none left, nothing refitted.
    let none = m
        .one_year_segments(
            &tri,
            "paid",
            incurred(),
            "premium",
            &BayesOneYear {
                refit: Some(OneYearRefit::RefitAll),
                max_refits: Some(0),
                n_sims: Some(40),
                ..BayesOneYear::default()
            },
        )
        .unwrap();
    for seg in &none.segments.fits {
        assert!(seg.refitted.is_empty());
        assert_eq!(seg.unrefitted.len(), 40);
    }
}

/// A triangle without keys is one segment, viewed with segment 0's
/// streams: `one_year_segments` agrees with the view of `year_streams`
/// and, being on other streams, not with `one_year`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_one_segment_view_is_the_view_with_segment_zero_streams() {
    let tri = lines(&["A"], false);
    let m = quick(5);
    let only = reweight_only(Some(300));
    let views = m
        .one_year_segments(&tri, "paid", incurred(), "premium", &only)
        .unwrap();
    assert_eq!(views.segments.len(), 1);
    let (settings, n_sims, _) = m.one_year_defaults(&only);
    let (fit, year) = m
        .year_streams(
            &tri,
            "paid",
            incurred(),
            "premium",
            &settings,
            n_sims,
            Stream::SegmentChains(0),
            Stream::SegmentPredictive(0),
        )
        .unwrap();
    let seg = &views.segments.fits[0];
    assert_eq!(fit.posterior, seg.fit.fit.posterior);
    assert_eq!(year.cdr, views.cdr.draw_matrix());
    assert_eq!(year.pareto_k, seg.pareto_k);
    assert_eq!(views.cdr.dims(), ["origin"]);
    let base = m
        .one_year(&tri, "paid", incurred(), "premium", &only)
        .unwrap();
    assert_ne!(base.fit.posterior, seg.fit.fit.posterior);
    assert_ne!(base.pareto_k, seg.pareto_k);
    // `one_year` itself is still the base streams' view.
    let (fit, year) = m
        .year_streams(
            &tri,
            "paid",
            incurred(),
            "premium",
            &settings,
            n_sims,
            Stream::Chains,
            Stream::Predictive,
        )
        .unwrap();
    assert_eq!(fit.posterior, base.fit.posterior);
    assert_eq!(year.cdr, base.cdr.draw_matrix());
}

/// The rank correlation reorders the segments' whole rows of claims
/// development results: each keeps its simulations as a set and the
/// totals reach the target.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_over_segments_reaches_the_rank_correlation() {
    let tri = lines(&["A", "B"], true);
    let m = quick(6);
    let only = reweight_only(None);
    let independent = m
        .one_year_segments(&tri, "paid", incurred(), "premium", &only)
        .unwrap();
    let ranked = Compartmental {
        dependence: SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.7, 0.7, 1.0],
        },
        ..m
    }
    .one_year_segments(&tri, "paid", incurred(), "premium", &only)
    .unwrap();
    let (ti, tr) = (line_totals(&independent.cdr), line_totals(&ranked.cdr));
    let sorted = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v
    };
    for s in 0..2 {
        assert_eq!(sorted(ti[s].clone()), sorted(tr[s].clone()));
    }
    let r = spearman(&tr[0], &tr[1]);
    assert!((r - 0.7).abs() < 0.05, "ranked lines: {r}");
    // The segments' own fits and Pareto k are untouched by the reorder.
    for (a, b) in independent.segments.fits.iter().zip(&ranked.segments.fits) {
        assert_eq!(a.pareto_k, b.pareto_k);
    }
}

/// Bit-exact on CI's Linux runner (#841; the doc of `bayes::pins` says what to
/// do when the hash moves): model 1 on company 337 (paid and incurred, 1997's
/// premium), two chains of 100 warm-up and 50 kept draws, 100 simulations;
/// the hash covers the posterior, the reserves and every simulated paid and
/// outstanding cell.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_fit_is_pinned_on_linux() {
    let model = Compartmental {
        sampler: Sampler {
            chains: 2,
            tune: 100,
            draws: 50,
            ..quick(841).sampler
        },
        n_sims: 100,
        ..quick(841)
    };
    crate::bayes::pins::check("Compartmental", 0x6956_4f17_9431_dcb1, || {
        let fit = model
            .fit(&triangle(true), "paid", incurred(), "premium")
            .unwrap();
        // The reserves read only the last paid cell, so the paid and
        // outstanding cells are hashed too: a swap of two of their draws
        // must move the hash.
        let h = crate::bayes::pins::fit_hash(&fit.posterior, &fit.reserves);
        let h = crate::bayes::pins::bits_hash_from(h, fit.paid.draw_matrix());
        match &fit.outstanding {
            Some(o) => crate::bayes::pins::bits_hash_from(h, o.draw_matrix()),
            None => h,
        }
    });
}

/// PSIS-LOO bit for bit (#842), of a fit like `the_fit_is_pinned_on_linux`'s
/// with 1,000 draws (2 chains of 500), so that each Pareto tail's length, `min(S/5, 3 sqrt(S / r_eff))`, moves with the draws' relative efficiency: the elpd
/// with its standard error, the pointwise terms and each cell's Pareto k,
/// and its outstanding and paid sums.
/// The Linux hash is still 0: CI's `bayes` job prints the value to pin
/// as `PIN Compartmental-loo actual 0x...`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_loo_is_pinned_on_linux() {
    let model = Compartmental {
        sampler: Sampler {
            chains: 2,
            tune: 100,
            draws: 500,
            ..quick(841).sampler
        },
        n_sims: 100,
        ..quick(841)
    };
    crate::bayes::pins::check("Compartmental-loo", 0, || {
        let fit = model
            .fit(&triangle(true), "paid", incurred(), "premium")
            .unwrap();
        let loo = fit.loo().unwrap();
        let h = crate::bayes::pins::loo_hash(&loo.loo);
        crate::bayes::pins::bits_hash_from(
            h,
            &[
                loo.n_outstanding as f64,
                loo.elpd_outstanding,
                loo.elpd_paid,
            ],
        )
    });
}
