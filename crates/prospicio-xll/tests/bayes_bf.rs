//! `PRO.BAYES.BF`, the Bayesian Bornhuetter-Ferguson with a normalised
//! pattern and a gamma prior on each origin's ultimate, sampled with NUTS,
//! reached through the catalogue as Excel reaches it:
//!
//! * every view is the library's `BayesBfNormalised` fit with the same
//!   settings, exactly, since the fit is reproducible by seed;
//! * a flat prior (`cv` `"flat"`) gives `PRO.BAYES.ODP`'s reserves within
//!   Monte Carlo error, the Bayesian chain ladder;
//! * a tight prior at the chain ladder's ultimates gives `PRO.BF`'s
//!   classical reserve;
//! * the cache returns the fit a refit gives, and errors come back as text.

mod common;

use common::{genins, n, num, t, text};
use prospicio_bayes::nuts::Sampler;
use prospicio_prob::Distribution;
use prospicio_reserving::ChainLadder;
use prospicio_reserving::bayes::BayesBfNormalised;
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// GenIns has ten origins: the exhibit's total is on row 11.
const TOTAL: usize = 11;

/// `PRO.BAYES.BF(GenIns, premium, apriori, cv, output, draws, 4, seed,
/// sims, , prior_sd, process)`, blanks where `None`.
#[allow(clippy::too_many_arguments)]
fn bf(
    exposure: Grid,
    apriori: f64,
    cv: Grid,
    output: &str,
    draws: f64,
    seed: f64,
    sims: f64,
    prior_sd: Option<f64>,
    process: Option<&str>,
) -> Grid {
    let (tri, _) = genins();
    call(
        "pro_bayes_bf",
        vec![
            tri,
            exposure,
            n(apriori),
            cv,
            t(output),
            n(draws),
            n(4.0),
            n(seed),
            n(sims),
            Grid::missing(),
            prior_sd.map_or_else(Grid::missing, n),
            process.map_or_else(Grid::missing, t),
        ],
    )
}

fn small(output: &str) -> Grid {
    bf(
        genins().1,
        0.65,
        n(0.2),
        output,
        300.0,
        5.0,
        2000.0,
        None,
        None,
    )
}

/// The library's fit of `small`'s settings.
fn small_fit() -> prospicio_reserving::bayes::BayesBfNormalisedFit {
    let (tri, premium) = genins();
    let premium: Vec<f64> = premium.cells.iter().map(num).collect();
    let tri = prospicio_xll::reserving::triangle_of(&tri, Some(&premium), true).unwrap();
    BayesBfNormalised {
        apriori: 0.65,
        cv: 0.2,
        sampler: Sampler {
            chains: 4,
            tune: 300,
            draws: 300,
            ..BayesBfNormalised::default().sampler
        },
        n_sims: 2000,
        seed: 5,
        ..BayesBfNormalised::default()
    }
    .fit(&tri, "values", "exposure")
    .unwrap()
}

#[test]
fn views_are_the_library_fit() {
    let fit = small_fit();

    let ex = small("reserves");
    assert_eq!((ex.rows, ex.columns), (12, 9), "{:?}", ex.as_scalar());
    assert_eq!(ex.get(TOTAL, 0), &Value::from("Total"));
    assert_eq!(num(ex.get(TOTAL, 2)), fit.reserves.mean());
    assert_eq!(num(ex.get(TOTAL, 3)), fit.reserves.std_dev());

    let ult = small("ultimates");
    assert_eq!((ult.rows, ult.columns), (12, 6));
    let posterior = fit.expected_ultimate();
    for w in 0..10 {
        assert_eq!(num(ult.get(w + 1, 1)), fit.exposure[w]);
        assert_eq!(num(ult.get(w + 1, 3)), fit.prior_ultimate[w]);
        assert_eq!(num(ult.get(w + 1, 4)), posterior[w]);
    }
    assert_eq!(
        num(ult.get(TOTAL, 3)),
        fit.prior_ultimate.iter().sum::<f64>()
    );

    let ps = small("params");
    let summary = fit.summary().unwrap();
    assert_eq!(ps.rows, summary.len() + 1);
    assert_eq!(ps.get(1, 0), &Value::from("alpha[1]"));
    for (r, p) in summary.iter().enumerate() {
        assert_eq!(ps.get(r + 1, 0), &Value::Text(p.name.clone()));
        assert_eq!(num(ps.get(r + 1, 1)), p.mean);
        assert_eq!(num(ps.get(r + 1, 6)), p.rhat);
    }

    let d = small("diagnostics");
    assert_eq!(num(d.get(0, 1)), fit.diagnostics.max_rhat);
    assert_eq!(num(d.get(3, 1)), fit.diagnostics.divergences as f64);
    assert_eq!(num(d.get(6, 1)), fit.scale);
}

/// The two limits of `reserving-bayes.md` decision 23, on the sheet.
#[test]
fn a_flat_prior_is_the_bayesian_chain_ladder_and_a_tight_one_is_classical_bf() {
    // Flat: PRO.BAYES.ODP's reserves within Monte Carlo error, as the
    // library's own test (mean within 0.1 SD, SD within 10%).
    let flat = bf(
        genins().1,
        1.0,
        t("flat"),
        "reserves",
        1000.0,
        1.0,
        10_000.0,
        None,
        None,
    );
    let odp = call(
        "pro_bayes_odp",
        vec![
            genins().0,
            Grid::missing(),
            n(1000.0),
            n(4.0),
            n(2.0),
            n(10_000.0),
        ],
    );
    for r in 2..=TOTAL {
        let (mean, sd) = (num(flat.get(r, 2)), num(flat.get(r, 3)));
        let (odp_mean, odp_sd) = (num(odp.get(r, 2)), num(odp.get(r, 3)));
        assert!(
            (mean - odp_mean).abs() < 0.1 * odp_sd,
            "row {r}: {mean} vs {odp_mean}"
        );
        assert!(
            (sd / odp_sd - 1.0).abs() < 0.10,
            "row {r}: sd {sd} vs {odp_sd}"
        );
    }

    // Tight at the chain ladder's ultimates: the classical reserve.
    let cl = ChainLadder::default()
        .fit(
            &prospicio_xll::reserving::triangle(&genins().0).unwrap(),
            "values",
        )
        .unwrap();
    let at_cl = Grid::from_rows(
        cl.ultimate
            .iter()
            .map(|&u| vec![Value::Number(u)])
            .collect(),
    );
    let classical = call("pro_bf", vec![genins().0, at_cl.clone(), n(1.0)]);
    let classical = num(classical.get(TOTAL, 6));
    assert!((classical / cl.total_reserve() - 1.0).abs() < 1e-9);
    let tight = bf(
        at_cl,
        1.0,
        n(1e-3),
        "reserves",
        1000.0,
        3.0,
        10_000.0,
        Some(100.0),
        Some("none"),
    );
    let total = num(tight.get(TOTAL, 2));
    assert!(
        (total / classical - 1.0).abs() < 5e-3,
        "{total} vs {classical}"
    );
}

#[test]
fn the_cache_returns_the_refit_and_a_new_seed_a_new_fit() {
    let a = small("reserves");
    assert_eq!(a, small("reserves"));
    let c = bf(
        genins().1,
        0.65,
        n(0.2),
        "reserves",
        300.0,
        6.0,
        2000.0,
        None,
        None,
    );
    assert_ne!(num(a.get(TOTAL, 2)), num(c.get(TOTAL, 2)));
    // A blank output is the reserves.
    let blank = bf(genins().1, 0.65, n(0.2), "", 300.0, 5.0, 2000.0, None, None);
    assert_eq!(a, blank);
}

#[test]
fn errors_come_back_as_text() {
    let (tri, premium) = genins();
    let e = |a: Vec<Grid>| text(&call("pro_bayes_bf", a)).to_owned();
    let base = || vec![tri.clone(), premium.clone(), n(0.65)];
    let with = |i: usize, g: Grid| {
        let mut a = base();
        a.resize(14, Grid::missing());
        a[i] = g;
        a
    };
    assert!(e(vec![tri.clone(), premium.clone()]).contains("apriori is required"));
    assert!(e(with(2, n(-1.0))).contains("apriori"));
    assert!(e(with(3, n(0.0))).contains("cv"));
    assert!(e(with(3, t("vague"))).contains("cv"));
    assert!(e(with(4, t("trace"))).contains("output must be"));
    assert!(e(with(5, n(0.0))).contains("draws"));
    assert!(e(with(6, n(9.0))).contains("chains"));
    assert!(e(with(10, n(-1.0))).contains("prior_sd"));
    assert!(e(with(13, n(0.0))).contains("common_cv"));
    let mut a = with(13, n(0.3));
    a[3] = t("flat");
    assert!(e(a).contains("finite cv"));
    let short = Grid::from_rows(vec![vec![Value::Number(1.0)]; 3]);
    assert!(e(vec![tri, short, n(0.65)]).contains("exposure"));
}
