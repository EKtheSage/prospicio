//! Selected development patterns (decision 12 of
//! `docs/design/reserving-v02.md`).
//!
//! * chainladder-python parity: full patterns through `DevelopmentConstant`
//!   (age-to-age, to ultimate with the level 1 at the oldest age, to
//!   ultimate with the level 1.03 and no tail, which both keep as the tail,
//!   and to ultimate followed by `TailConstant`), with the chain ladder on RAA and
//!   GenIns and the four expected-loss methods on the CAS workers'
//!   compensation triangle (`reference/reserving_patterns_python.csv`, from
//!   `scripts/reserving_patterns_python.py`). The percents developed of the
//!   same patterns are checked against the same rows. chainladder-python has
//!   no blend, so its blended rows are the script's own blend passed as a
//!   pattern: they check how blended factors reach each method.
//! * Mack against R ChainLadder: selecting the triangle's own factors fully
//!   leaves R's `Mack.ProcessRisk` and no parameter risk; with a uniform
//!   weight `w` the parameter risk is `(1 - w)` times R's
//!   (`reference/reserving_chainladder_r.csv`). R has no selected factors,
//!   so this is the reference where the selection equals the estimate.
//! * Merz and Wüthrich with a blend: the closed form's coming year is the
//!   first-order (linearised) CDR covariance under the model of item 7,
//!   computed exactly here, to rounding; every later year is checked
//!   against a simulation of that model over the full run-off; and Mack's
//!   one-year bootstrap with the same blend reconciles with it.

use std::collections::HashMap;

use prospicio_core::StreamRng;
use prospicio_reserving::{
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ChainLadderFit, Development,
    ExpectedLoss, Lag, Mack, MackBootstrap, MackFit, OneYearMethod, PatternKind, PatternWeight,
    Period, SelectedPattern, Tail, Triangle,
};
use prospicio_validation::{Case, check, reference, triangle, triangle_columns};

fn with(pattern: SelectedPattern) -> Development {
    Development {
        pattern: Some(pattern),
        ..Default::default()
    }
}

/// The settings of a reference `method`: the pattern's kind, the tail and
/// the weight.
fn style(method: &str) -> (PatternKind, Tail, PatternWeight) {
    let has = |p: &str| method.split(';').any(|x| x == p);
    let kind = if has("style=ldf") {
        PatternKind::AgeToAge
    } else if has("style=cdf") {
        PatternKind::AgeToUltimate
    } else {
        panic!("{method}: no style")
    };
    let tail = if has("tail=1.05") {
        1.05.into()
    } else {
        Tail::default()
    };
    let weight = if has("weight=0.6") {
        PatternWeight::All(0.6)
    } else {
        PatternWeight::All(1.0)
    };
    (kind, tail, weight)
}

/// The fitted chain ladder of a reference case, with the method's
/// ultimates.
struct Fitted {
    chain_ladder: ChainLadderFit,
    ultimate: Vec<f64>,
}

/// Fits `method` on `dataset` with `pattern` (of the method's kind, or its
/// percents developed when `percent`).
fn fit(dataset: &str, method: &str, values: &[(Lag, f64)], percent: bool) -> Fitted {
    let (kind, tail, weight) = style(method);
    let pattern = if percent {
        assert_eq!(kind, PatternKind::AgeToUltimate);
        SelectedPattern::new(
            PatternKind::PercentDeveloped,
            values.iter().map(|&(a, c)| (a, 1.0 / c)).collect(),
        )
    } else {
        SelectedPattern::new(kind, values.to_vec())
    }
    .with_weight(weight);
    let chain_ladder = ChainLadder {
        development: with(pattern),
        tail,
    };
    let fail = |e: prospicio_reserving::Error| -> ! { panic!("{dataset} {method}: {e}") };
    let param = |name: &str| {
        method
            .split(';')
            .find_map(|p| p.strip_prefix(name)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("{method}: no {name}"))
            .parse::<f64>()
            .unwrap()
    };
    if method.starts_with("chain_ladder") {
        let cl = chain_ladder
            .fit(&triangle(dataset), "values")
            .unwrap_or_else(|e| fail(e));
        return Fitted {
            ultimate: cl.ultimate.clone(),
            chain_ladder: cl,
        };
    }
    let tri = triangle_columns(dataset, &["paid", "premium"]);
    let fit = match method.split(';').next().unwrap() {
        "expected_loss" => ExpectedLoss {
            apriori: param("apriori").into(),
            chain_ladder,
        }
        .fit(&tri, "paid", "premium"),
        "bornhuetter_ferguson" => BornhuetterFerguson {
            apriori: param("apriori").into(),
            chain_ladder,
        }
        .fit(&tri, "paid", "premium"),
        "benktander" => Benktander {
            apriori: param("apriori").into(),
            n_iters: param("n_iters") as usize,
            chain_ladder,
        }
        .fit(&tri, "paid", "premium"),
        "cape_cod" => CapeCod {
            trend: param("trend"),
            decay: param("decay"),
            chain_ladder,
        }
        .fit(&tri, "paid", "premium")
        .map(|f| f.expected_loss),
        other => panic!("unknown method {other}"),
    }
    .unwrap_or_else(|e| fail(e));
    Fitted {
        ultimate: fit.ultimate,
        chain_ladder: fit.chain_ladder,
    }
}

/// Every reference row, with the pattern given as `percent` developed when
/// it is a cumulative one.
fn check_parity(percent: bool) {
    let cases = reference("reserving_patterns_python.csv");
    let mut inputs: HashMap<(String, String), Vec<(Lag, f64)>> = HashMap::new();
    for case in cases.iter().filter(|c| c.get("quantity") == "pattern") {
        let key = (case.get("dataset").into(), case.get("method").into());
        let age = case.number("arg").unwrap() as Lag;
        inputs
            .entry(key)
            .or_default()
            .push((age, case.number("expected").unwrap()));
    }
    let mut fits: HashMap<(String, String), Fitted> = HashMap::new();
    let selected: Vec<Case> = cases
        .into_iter()
        .filter(|c| !percent || style(c.get("method")).0 == PatternKind::AgeToUltimate)
        .collect();
    check(&selected, |case: &Case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let key = (dataset.to_string(), method.to_string());
        let f = fits
            .entry(key.clone())
            .or_insert_with(|| fit(dataset, method, &inputs[&key], percent));
        let cl = &f.chain_ladder;
        let age = || {
            let age = case.number("arg")? as Lag;
            cl.development.development.iter().position(|&a| a == age)
        };
        let origin = || {
            let year = case.number("arg")? as i32;
            cl.origins.iter().position(|&p| p == Period::year(year))
        };
        match case.get("quantity") {
            // An age-to-age pattern is read back from the selected factors,
            // a cumulative one from the factors to ultimate in use and the
            // rescale.
            "pattern" => match style(method).0 {
                PatternKind::AgeToAge => cl.development.selected[age()?],
                _ => Some(cl.cdf[age()?] * cl.pattern_rescale?),
            },
            "ldf" => cl.ldf().get(age()?).copied(),
            "cdf" => cl.cdf.get(age()?).copied(),
            "ultimate" => Some(f.ultimate[origin()?]),
            "total_ultimate" => Some(f.ultimate.iter().sum()),
            _ => None,
        }
    });
}

#[test]
fn matches_chainladder_python_development_constant() {
    check_parity(false);
}

#[test]
fn percents_developed_match_the_same_factors_to_ultimate() {
    check_parity(true);
}

#[test]
fn rescale_is_one_without_a_tail_and_the_tail_otherwise() {
    // A cumulative pattern to the oldest age with no tail is not rescaled:
    // at 100% it implies no tail, at the level 1.03 a tail of 1.03. With
    // the level 1.03 and a tail of 1.05, the level is replaced by the
    // tail: 1.03 / 1.05.
    let cases = reference("reserving_patterns_python.csv");
    for (method, want, implied) in [
        ("chain_ladder;style=cdf", 1.0, Some(1.0)),
        ("chain_ladder;style=cdf;tail=1.05", 1.03 / 1.05, None),
        ("chain_ladder;style=cdf;level=1.03", 1.0, Some(1.03)),
    ] {
        let values: Vec<(Lag, f64)> = cases
            .iter()
            .filter(|c| {
                (c.get("dataset"), c.get("method"), c.get("quantity")) == ("raa", method, "pattern")
            })
            .map(|c| {
                (
                    c.number("arg").unwrap() as Lag,
                    c.number("expected").unwrap(),
                )
            })
            .collect();
        let cl = fit("raa", method, &values, false).chain_ladder;
        let got = cl.pattern_rescale.unwrap();
        assert!((got - want).abs() < 1e-12, "{method}: {got} vs {want}");
        match (cl.pattern_tail, implied) {
            (Some(t), Some(want)) => assert!((t - want).abs() < 1e-12, "{method}: {t}"),
            (got, want) => assert_eq!(got, want, "{method}"),
        }
    }
}

/// The triangle's own estimated factors as an age-to-age pattern.
fn own(tri: &Triangle, weight: PatternWeight) -> SelectedPattern {
    let dev = Development::default().fit(tri, "values").unwrap();
    SelectedPattern::new(
        PatternKind::AgeToAge,
        dev.development.iter().copied().zip(dev.ldf).collect(),
    )
    .with_weight(weight)
}

fn mack(tri: &Triangle, weight: PatternWeight) -> MackFit {
    Mack {
        development: with(own(tri, weight)),
        ..Default::default()
    }
    .fit(tri, "values")
    .unwrap()
}

#[test]
fn mack_with_the_triangles_own_factors_matches_r() {
    let cases = reference("reserving_chainladder_r.csv");
    let r = |dataset: &str, quantity: &str, arg: &str| -> f64 {
        cases
            .iter()
            .find(|c| {
                (
                    c.get("dataset"),
                    c.get("method"),
                    c.get("quantity"),
                    c.get("arg"),
                ) == (dataset, "mack", quantity, arg)
            })
            .and_then(|c| c.number("expected"))
            .unwrap_or_else(|| panic!("{dataset} {quantity} {arg}"))
    };
    let mut failures = Vec::new();
    for dataset in ["raa", "genins", "abc"] {
        let tri = triangle(dataset);
        for w in [1.0, 0.4] {
            let fit = mack(&tri, PatternWeight::All(w));
            let mut compare = |what: String, got: f64, want: f64| {
                if (got - want).abs() > 1e-9 * want.abs().max(1.0) {
                    failures.push(format!("{dataset} w={w} {what}: {got} vs {want}"));
                }
            };
            for (o, p) in fit.chain_ladder.origins.iter().enumerate() {
                let year = p.to_string();
                let process = r(dataset, "process_risk", &year);
                let parameter = (1.0 - w) * r(dataset, "parameter_risk", &year);
                compare(format!("{year} process"), fit.process_risk[o], process);
                compare(
                    format!("{year} parameter"),
                    fit.parameter_risk[o],
                    parameter,
                );
                compare(
                    format!("{year} se"),
                    fit.standard_error[o],
                    process.hypot(parameter),
                );
            }
            let process = r(dataset, "total_process_risk", "");
            let parameter = (1.0 - w) * r(dataset, "total_parameter_risk", "");
            compare("total process".into(), fit.total_process_risk, process);
            compare(
                "total parameter".into(),
                fit.total_parameter_risk,
                parameter,
            );
            compare(
                "total se".into(),
                fit.total_standard_error,
                process.hypot(parameter),
            );
            // The ultimates are R's: the selection is the estimate.
            for (o, p) in fit.chain_ladder.origins.iter().enumerate() {
                let year = p.to_string();
                let cl = cases
                    .iter()
                    .find(|c| {
                        (
                            c.get("dataset"),
                            c.get("method"),
                            c.get("quantity"),
                            c.get("arg"),
                        ) == (dataset, "chain_ladder", "ultimate", year.as_str())
                    })
                    .and_then(|c| c.number("expected"))
                    .unwrap();
                compare(format!("{year} ultimate"), fit.chain_ladder.ultimate[o], cl);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Weights that differ by age, every link selected.
fn by_age(tri: &Triangle) -> PatternWeight {
    const W: [f64; 9] = [0.9, 0.2, 0.6, 1.0, 0.0, 0.5, 0.3, 0.8, 0.7];
    let ages = tri.development();
    PatternWeight::ByAge(ages.iter().copied().zip(W).collect())
}

/// The coming year's CDR covariance to first order under the model of item
/// 7, with the triangle's own factors selected (so the used factors and
/// their re-estimates' means are the estimate's, and the linearisation
/// point is the opening ultimates). Origin `o` (latest age `k`) has next
/// value `Z_o` with mean `f_k C` and variance `g_k^2 sigma_k^2 / S_k C^2 +
/// sigma_k^2 C`, the true factor's parameter error and the process; the
/// used factor at `k` next year is `w e_k + g_k (A_k + Z_o) / (S_k + C)`,
/// linear in `Z_o`; and each closing ultimate is a product of such factors,
/// each in one `Z`. Returns each origin's standard deviation, then the
/// total's.
fn linearised_one_year(tri: &Triangle, fit: &MackFit) -> Vec<f64> {
    let cl = &fit.chain_ladder;
    let dev = &cl.development;
    let (f, sigma, weight) = (&dev.ldf, &dev.sigma, &dev.weight);
    let n = cl.origins.len();
    let c = |o: usize, d: usize| tri.get(0, 0, o, d).unwrap();
    let mut ez = vec![0.0; n];
    let mut vz = vec![0.0; n];
    let mut ab = vec![(1.0, 0.0); n];
    for o in 1..n {
        let k = n - 1 - o;
        let s: f64 = (0..o).map(|i| c(i, k)).sum();
        let a: f64 = (0..o).map(|i| c(i, k + 1)).sum();
        let g = 1.0 - weight[k];
        let ci = c(o, k);
        ez[o] = f[k] * ci;
        vz[o] = g * g * sigma[k].powi(2) / s * ci * ci + sigma[k].powi(2) * ci;
        ab[o] = (weight[k] * f[k] + g * a / (s + ci), g / (s + ci));
    }
    let factor = |l: usize, o: usize| match o.cmp(&l) {
        std::cmp::Ordering::Equal => (0.0, 1.0),
        std::cmp::Ordering::Less if o >= 1 => ab[o],
        _ => (1.0, 0.0),
    };
    let mean = |l: usize| -> f64 {
        (1..n)
            .map(|o| {
                let (a, b) = factor(l, o);
                a + b * ez[o]
            })
            .product()
    };
    let cov = |l: usize, j: usize| -> f64 {
        (1..n)
            .map(|o| {
                let ((al, bl), (aj, bj)) = (factor(l, o), factor(j, o));
                mean(l) * bl / (al + bl * ez[o]) * mean(j) * bj / (aj + bj * ez[o]) * vz[o]
            })
            .sum()
    };
    let mut out: Vec<f64> = (0..n).map(|l| cov(l, l).sqrt()).collect();
    out.push(
        (0..n)
            .flat_map(|l| (0..n).map(move |j| (l, j)))
            .map(|(l, j)| cov(l, j))
            .sum::<f64>()
            .sqrt(),
    );
    out
}

#[test]
fn merz_wuthrich_coming_year_is_the_linearised_blend() {
    // The closed form of item 7 against the first-order covariance computed
    // directly, to rounding, for a uniform and an age-by-age blend. The
    // opening ultimates are the linearisation point of both.
    for dataset in ["raa", "genins", "abc"] {
        let tri = triangle(dataset);
        for weight in [PatternWeight::All(0.5), by_age(&tri)] {
            let fit = mack(&tri, weight.clone());
            let cdr = fit.claims_development_result().unwrap();
            let want = linearised_one_year(&tri, &fit);
            let got: Vec<f64> = cdr
                .one_year_standard_error
                .iter()
                .copied()
                .chain([cdr.total_one_year_standard_error])
                .collect();
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-9 * w.max(1.0),
                    "{dataset} {weight:?} [{k}]: closed form {g} vs linearised {w}"
                );
            }
        }
    }
}

/// A standard normal draw (Box-Muller, one of the pair).
fn normal(rng: &mut StreamRng) -> f64 {
    let (u, v) = (rng.next_open01(), rng.next_open01());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// Standard deviation of `x` and its Monte Carlo standard error,
/// `sd * sqrt((kurtosis - 1) / (4 n))`.
fn sd_and_error(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    let sd = m2.sqrt();
    (sd, sd * ((m4 / (m2 * m2) - 1.0) / (4.0 * n)).sqrt())
}

/// Simulates the model of item 7 over the full run-off of a regular
/// triangle: each true factor drawn once, `f_k + g_k sigma_k / sqrt(S_k)
/// N(0, 1)`, then every year each origin develops one age with Mack's
/// normal process, `phi_k C + sigma_k sqrt(C) N(0, 1)`, every factor is
/// re-estimated (volume-weighted over the pairs observed so far) and
/// blended, `w s + (1 - w) e'`, and each origin's CDR is its ultimate at the
/// year's start minus at its end. Returns the draws per year, per origin
/// then the total.
fn simulate_run_off(
    tri: &Triangle,
    fit: &MackFit,
    n_paths: usize,
    seed: u64,
) -> Vec<Vec<Vec<f64>>> {
    let cl = &fit.chain_ladder;
    let dev = &cl.development;
    let n = cl.origins.len();
    let n_links = dev.ldf.len();
    let blended = dev.blended_ldf();
    let selected: Vec<f64> = dev
        .selected
        .iter()
        .zip(&dev.ldf)
        .map(|(s, e)| s.unwrap_or(*e))
        .collect();
    let opening: Vec<Vec<f64>> = (0..n)
        .map(|o| {
            (0..=cl.latest_position[o])
                .map(|d| tri.get(0, 0, o, d).unwrap())
                .collect()
        })
        .collect();
    let mut draws = vec![vec![Vec::with_capacity(n_paths); n + 1]; n_links];
    let mut rng = StreamRng::new(seed, 0);
    for _ in 0..n_paths {
        let phi: Vec<f64> = (0..n_links)
            .map(|k| {
                let g = 1.0 - dev.weight[k];
                blended[k] + g * dev.sigma[k] / dev.volume[k].sqrt() * normal(&mut rng)
            })
            .collect();
        let mut cells = opening.clone();
        let ultimates = |cells: &[Vec<f64>]| -> Vec<f64> {
            let used: Vec<f64> = (0..n_links)
                .map(|k| {
                    let (mut num, mut den) = (0.0, 0.0);
                    for row in cells.iter().filter(|r| r.len() > k + 1) {
                        num += row[k + 1];
                        den += row[k];
                    }
                    let w = dev.weight[k];
                    w * selected[k] + (1.0 - w) * num / den
                })
                .collect();
            cells
                .iter()
                .map(|row| row.last().unwrap() * used[row.len() - 1..].iter().product::<f64>())
                .collect()
        };
        let mut before = ultimates(&cells);
        for year in draws.iter_mut() {
            for row in cells.iter_mut().filter(|r| r.len() <= n_links) {
                let k = row.len() - 1;
                let c = *row.last().unwrap();
                row.push(phi[k] * c + dev.sigma[k] * c.max(0.0).sqrt() * normal(&mut rng));
            }
            let after = ultimates(&cells);
            let mut total = 0.0;
            for o in 0..n {
                let cdr = before[o] - after[o];
                year[o].push(cdr);
                total += cdr;
            }
            year[n].push(total);
            before = after;
        }
    }
    draws
}

#[test]
fn merz_wuthrich_later_years_match_a_simulation_of_the_blend() {
    // Every calendar year of the closed form (each origin and the total)
    // against 40,000 simulated run-offs of its model on GenIns, within five
    // Monte Carlo standard errors (about 1.9%): without a pattern (Merz and
    // Wüthrich's own formula, which shows that the simulation and a
    // first-order closed form agree to Monte Carlo error) and with a
    // uniform and an age-by-age blend. On this seed the largest gaps are
    // 0.9%, 1.1% and 1.1%, the mean gaps 0.3%. The coming year's terms
    // carried over to later years unchanged (an earlier draft) are
    // 3% to 6% below the simulation from the second year on and fail here.
    let tri = triangle("genins");
    let mut failures = Vec::new();
    for (label, weight) in [
        ("w=0", PatternWeight::All(0.0)),
        ("w=0.5", PatternWeight::All(0.5)),
        ("by age", by_age(&tri)),
    ] {
        let fit = mack(&tri, weight);
        let cdr = fit.claims_development_result().unwrap();
        let draws = simulate_run_off(&tri, &fit, 40_000, 20_261_014);
        for (y, year) in draws.iter().enumerate() {
            for (o, x) in year.iter().enumerate() {
                let want = match cdr.by_calendar_year[y].get(o) {
                    Some(&v) => v,
                    None => cdr.total_by_calendar_year[y],
                };
                let (sd, error) = sd_and_error(x);
                if want == 0.0 {
                    if sd > 1e-6 {
                        failures.push(format!("{label} year {y} [{o}]: sd {sd}, want 0"));
                    }
                    continue;
                }
                if (sd - want).abs() > 5.0 * error {
                    failures.push(format!(
                        "{label} year {y} [{o}]: simulated {sd:.0} vs closed form {want:.0} \
                         ({:+.2}%, Monte Carlo SE {:.2}%)",
                        100.0 * (sd / want - 1.0),
                        100.0 * error / want
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn mack_bootstrap_one_year_reconciles_with_the_blended_closed_form() {
    // Mack's bootstrap re-reserving with the same blend: its one-year
    // standard deviations against the closed form, within five Monte Carlo
    // standard errors, as without a pattern (reserving_one_year_mack.rs).
    let tri = triangle("genins");
    let pattern = own(&tri, PatternWeight::All(0.5));
    let closed = mack(&tri, PatternWeight::All(0.5))
        .claims_development_result()
        .unwrap();
    let n_sims = 20_000;
    let fit = MackBootstrap {
        n_sims,
        seed: 20_261_014,
        development: with(pattern.clone()),
        ..Default::default()
    }
    .one_year(
        &tri,
        "values",
        &OneYearMethod::ChainLadder(ChainLadder {
            development: with(pattern),
            ..Default::default()
        }),
    )
    .unwrap();
    let n = fit.cdr.n_components();
    let rows: Vec<&[f64]> = fit.cdr.draw_matrix().chunks(n).collect();
    let mut failures = Vec::new();
    for o in 0..=n {
        let x: Vec<f64> = rows
            .iter()
            .map(|r| if o < n { r[o] } else { r.iter().sum() })
            .collect();
        let want = if o < n {
            closed.one_year_standard_error[o]
        } else {
            closed.total_one_year_standard_error
        };
        let (sd, error) = sd_and_error(&x);
        if want == 0.0 {
            assert_eq!(sd, 0.0);
            continue;
        }
        if (sd - want).abs() > 5.0 * error {
            failures.push(format!(
                "[{o}]: bootstrap {sd:.0} vs closed form {want:.0} ({:+.2} SE)",
                (sd - want) / error
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(rows.len(), n_sims);
}
