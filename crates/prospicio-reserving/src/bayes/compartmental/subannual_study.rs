//! Test-only: model 2's development effects by development year against one
//! level per age (`docs/design/reserving-compartmental.md` decision 14,
//! `knowledge/findings/compartmental-subannual.md`). The by-year form is the
//! built setting [`DevelopmentLevels::ByYear`], so the two arms differ in
//! that setting alone. The study's panels ran the density with its
//! age-to-level map changed and the predictive per age; the built setting
//! shares a year's effect in the predictive too, so a rerun can differ from
//! the recorded `grouped` rows in the reserve (not in the density, `loo` or
//! the sampler).
//!
//! The books come from `validation/tests/reserving_compartmental_subannual.rs`
//! (`export_books`), so that they are the panels' own:
//!
//! ```text
//! SUBANNUAL_EXPORT=m2y:aq,m2q:aq SUBANNUAL_BOOKS=30 cargo test --release \
//!   -p prospicio-validation --features slow-bayes \
//!   --test bayes reserving_compartmental_subannual::export_books -- --ignored
//! SUBANNUAL_GROUPED=m2y_aq,m2q_aq SUBANNUAL_BOOKS=30 cargo test --release \
//!   -p prospicio-reserving --features bayes --lib subannual_study \
//!   -- --ignored --nocapture
//! ```
//!
//! Each book's two fits share the seed `1,000 + book` and the sampler of the
//! study (`SUBANNUAL_SAMPLER=default` for the model's own). Results go to
//! `target/compartmental-subannual/grouped_<books>.csv`, a book already there
//! skipped.

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::time::Instant;

use prospicio_bayes::nuts::Sampler;
use prospicio_core::{Grain, Lag, Month};
use prospicio_prob::Distribution;

use super::{Compartmental, CompartmentalFit, CompartmentalPriors, DevelopmentLevels, Outstanding};
use crate::triangle::Long;
use crate::{DevelopmentColumn, Triangle};

struct Book {
    triangle: Triangle,
    outcome: f64,
    expected: f64,
    cells: usize,
}

fn grain(months: u32) -> Grain {
    match months {
        12 => Grain::Year,
        3 => Grain::Quarter,
        1 => Grain::Month,
        other => panic!("no grain of {other} months"),
    }
}

fn read_book(path: &std::path::Path) -> Book {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split(' ').collect();
    let (outcome, expected) = (head[2].parse().unwrap(), head[4].parse().unwrap());
    let g: Vec<u32> = lines
        .next()
        .unwrap()
        .split(' ')
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    let (mut origin, mut ages) = (vec![], vec![]);
    let (mut pd, mut os, mut premium) = (vec![], vec![], vec![]);
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        origin.push(Month::new(f[0].parse().unwrap(), f[1].parse().unwrap()).unwrap());
        ages.push(f[2].parse::<Lag>().unwrap());
        pd.push(f[3].parse::<f64>().unwrap());
        os.push(f[4].parse::<f64>().unwrap());
        premium.push(f[5].parse::<f64>().unwrap());
    }
    let triangle = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &pd), ("outstanding", &os), ("premium", &premium)],
        origin_grain: grain(g[0]),
        development_grain: grain(g[1]),
        cumulative: true,
    })
    .unwrap();
    Book {
        triangle,
        outcome,
        expected,
        cells: 2 * pd.len(),
    }
}

fn model(seed: u64, levels: DevelopmentLevels) -> Compartmental {
    let mut m = Compartmental::model_2(CompartmentalPriors::gesmann_morris_2());
    m.seed = seed;
    m.development_levels = levels;
    if std::env::var("SUBANNUAL_SAMPLER").as_deref() != Ok("default") {
        m.sampler = Sampler {
            tune: 500,
            draws: 500,
            target_accept: 0.99,
            ..m.sampler
        };
        m.n_sims = 4000;
    }
    m
}

fn fit(book: &Book, seed: u64, levels: DevelopmentLevels) -> CompartmentalFit {
    model(seed, levels)
        .fit(
            &book.triangle,
            "paid",
            Outstanding::Column("outstanding"),
            "premium",
        )
        .expect("fits")
}

fn line(form: &str, b: u64, book: &Book, fit: &CompartmentalFit, secs: f64) -> String {
    let d = &fit.diagnostics;
    let total = fit.reserves.total();
    let elpd = fit.loo().map_or(f64::NAN, |l| l.loo.estimate.elpd);
    let mut s = String::new();
    writeln!(
        s,
        "{form},{b},{},{},{secs:.2},{:.5},{:.1},{:.1},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},\
         {:.3},{elpd:.2}",
        book.cells,
        fit.posterior.dim(),
        d.max_rhat,
        d.min_ess_bulk,
        d.min_ess_tail,
        d.divergences,
        d.warnings.len(),
        book.outcome,
        book.expected,
        total.mean(),
        total.std_dev(),
        total.quantile(0.05).unwrap(),
        total.quantile(0.95).unwrap(),
        100.0 * total.cdf(book.outcome),
    )
    .unwrap();
    s
}

#[test]
#[ignore = "the study of decision 14, run by hand in release"]
fn grouped_against_per_age() {
    let spec = std::env::var("SUBANNUAL_GROUPED").expect("SUBANNUAL_GROUPED");
    let books: u64 = std::env::var("SUBANNUAL_BOOKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/compartmental-subannual"
    );
    for name in spec.split(',') {
        let path = format!("{dir}/grouped_{name}.csv");
        let done = std::fs::read_to_string(&path).unwrap_or_default();
        if done.is_empty() {
            std::fs::write(
                &path,
                "form,book,cells,draw_dim,seconds,max_rhat,ess_bulk,ess_tail,divergences,\
                 warnings,outcome,expected,mean,sd,q05,q95,percentile,elpd_loo\n",
            )
            .unwrap();
        }
        let seen: Vec<u64> = done
            .lines()
            .skip(1)
            .filter(|l| l.starts_with("grouped"))
            .filter_map(|l| l.split(',').nth(1)?.parse().ok())
            .collect();
        for b in (0..books).filter(|b| !seen.contains(b)) {
            let book = read_book(format!("{dir}/books/{name}_{b}.csv").as_ref());
            let start = Instant::now();
            let per_age = fit(&book, 1000 + b, DevelopmentLevels::PerAge);
            let per_age_secs = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let by_year = fit(&book, 1000 + b, DevelopmentLevels::ByYear);
            let by_year_secs = start.elapsed().as_secs_f64();
            let text = line("per_age", b, &book, &per_age, per_age_secs)
                + &line("grouped", b, &book, &by_year, by_year_secs);
            eprint!("{name} {text}");
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(text.as_bytes()).unwrap();
        }
    }
}
