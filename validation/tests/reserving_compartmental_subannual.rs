//! How well `Compartmental` fits sub-annual data: the study behind
//! `docs/design/reserving-compartmental.md` decision 14 and
//! `knowledge/findings/compartmental-subannual.md`. Nothing in CI runs it
//! but the cheap checks of the books at the end of this file; the panels are
//! ignored tests, run by hand in release:
//!
//! ```text
//! SUBANNUAL_ARMS=iid:aq:m1:preset:study SUBANNUAL_BOOKS=50 \
//!   cargo test --release -p prospicio-validation --features slow-bayes \
//!   --test bayes reserving_compartmental_subannual::panel -- --ignored --nocapture
//! ```
//!
//! One book is simulated once at the finest grain, every month to 120
//! months, and each grain reads it: an annual cell is the monthly cell at
//! month 12, so "the same book at annual grain" is literally that. An arm is
//! `world:grain:model:prior:sampler`.
//!
//! * Worlds (`World::parse`). `iid`: model 1 exactly, ten annual origins,
//!   independent normal noise on every cumulative cell. `persist`: the same
//!   means, noise that accumulates (independent monthly payment noise, an
//!   AR(1) on outstanding). `earned`: 40 quarterly origins, each model 1
//!   exactly; an annual origin is the sum of its four quarters, so the
//!   premium is earned through the year as it is in a real accident year.
//!   `conflict` and `fast`: `iid` with rates far from the preset. `season`
//!   and `seasonq`: `iid` and `earned` with a calendar-quarter factor on
//!   the paid increments. `season117` and `iid117` value the book at the
//!   end of the third calendar quarter, where the factors' cumulative
//!   deviation from flat is 0, and `season114` and `iid114` at the end of
//!   the second, where it is largest (0.3 of a quarter's payments).
//!   `m2y` and `m2q`: model 2's cell-level process
//!   with development effects by development year or by quarterly age,
//!   log-normal monthly increments; `m2season` is `m2y` with the seasonal
//!   factor; `m2qo` is `m2y` on 40 quarterly origins and `m2seasonq` is
//!   `m2qo` with the seasonal factor; `m2y114` and
//!   `m2season114` value `m2y` and `m2season` at the end of the second
//!   calendar quarter.
//! * Grains. `aa`, `aq`, `am`: annual origins with annual, quarterly or
//!   monthly development. `qq`, `qm`: quarterly origins.
//! * Models. `m1`, `m2`: `Compartmental::new` and `model_2`. `m1i`: model 1
//!   with `PaidResponse::Incremental`, a built setting.
//! * Priors (`priors`). `preset`, `diffuse`, `ratewide`, and for model 2
//!   `sigmawide`.
//! * Samplers (`model`). `default`: the model's own. `study`: a shorter run,
//!   stated in the note.
//!
//! The arms run in order and books run one after another (a fit's four
//! chains use four threads). `SUBANNUAL_BOOKS` and `SUBANNUAL_FIRST` choose
//! the books, so two processes can split an arm's range; a book already in
//! the arm's CSV is skipped. Results go to
//! `target/compartmental-subannual/<arm>.csv` (`SUBANNUAL_OUT` for another
//! directory), one line a fit, and
//! `validation/scripts/compartmental_subannual_summary.py` turns them into
//! the note's tables. Book `b` of a world always uses
//! `StreamRng::new(BOOK_SEED, world stream * 10,000 + b)`, and its fit seed is
//! `1,000 + b`, in every arm.

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use prospicio_bayes::nuts::Sampler;
use prospicio_core::StreamRng;
use prospicio_math::special::norm_quantile;
use prospicio_prob::{Distribution, KeyValue};
use prospicio_reserving::bayes::{
    Compartmental, CompartmentalFit, CompartmentalPriors, Outstanding, PaidResponse, PositivePrior,
    RealPrior,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Period, Triangle};

const BOOK_SEED: u64 = 20_261_105;
/// The last age, and the calendar month of a year-end valuation.
const LAST: usize = 120;
/// The seasonal factors by calendar quarter, averaging 1.
const SEASON: [f64; 4] = [1.3, 1.0, 0.7, 1.0];

fn outstanding(t: f64, k_er: f64, k_p: f64, rlr: f64) -> f64 {
    rlr * k_er * (((-k_p * t).exp() - (-k_er * t).exp()) / (k_er - k_p))
}

fn paid(t: f64, k_er: f64, k_p: f64, rlr: f64, rrf: f64) -> f64 {
    let g = ((-k_p * t).exp() - (-k_er * t).exp()) / (k_er - k_p);
    rlr * rrf * ((1.0 - (-k_p * t).exp()) - k_p * g)
}

/// The population of a model 1 world, on the preset's `o` scale.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Truth {
    b: [f64; 4],
    tau_rlr: f64,
    tau_rrf: f64,
    rho: f64,
    sigma_os: f64,
    sigma_pd: f64,
}

/// `validation/scripts/compartmental_grain_simulate.py`'s truth.
const BASE: Truth = Truth {
    b: [1.0, -2.0, 1.0, -1.0],
    tau_rlr: 0.4,
    tau_rrf: 0.2,
    rho: 0.3,
    sigma_os: 1500.0,
    sigma_pd: 500.0,
};

impl Truth {
    fn k_er(&self) -> f64 {
        3.0 * (0.1 * self.b[0]).exp()
    }
    fn k_p(&self) -> f64 {
        (0.1 * self.b[1]).exp()
    }
    fn rlr(&self, effect: f64) -> f64 {
        0.7 * (0.2 * (self.b[2] + effect)).exp()
    }
    fn rrf(&self, effect: f64) -> f64 {
        0.8 * (0.1 * (self.b[3] + effect)).exp()
    }
}

/// How the noise enters.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Noise {
    /// Independent normal noise on each cumulative cell: the model's own.
    Iid,
    /// Independent monthly payment noise with variance `PERSIST_PHI` times
    /// the month's mean payment, summed, and an AR(1) on outstanding.
    Persistent,
}

/// Payment noise per unit of expected paid: an sd of about 2% of a 60,000
/// ultimate at month 120.
const PERSIST_PHI: f64 = 25.0;
/// The monthly autocorrelation of the outstanding noise.
const PERSIST_RHO: f64 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq)]
enum World {
    Model1 {
        stream: u64,
        truth: Truth,
        noise: Noise,
        quarterly_origins: bool,
        season: bool,
        valuation: usize,
    },
    Model2 {
        stream: u64,
        /// Months per development level of the process (12 or 3).
        dev_unit: usize,
        season: bool,
        quarterly_origins: bool,
        valuation: usize,
    },
}

impl World {
    fn parse(name: &str) -> World {
        let m1 = |stream, truth, noise, quarterly_origins, season, valuation| World::Model1 {
            stream,
            truth,
            noise,
            quarterly_origins,
            season,
            valuation,
        };
        match name {
            "iid" => m1(1, BASE, Noise::Iid, false, false, LAST),
            "persist" => m1(2, BASE, Noise::Persistent, false, false, LAST),
            "earned" => m1(3, BASE, Noise::Iid, true, false, LAST),
            // Company 337's posterior (decision 2): k_er 1.75, k_p 0.43.
            "conflict" => m1(
                4,
                Truth {
                    b: [-5.39, -8.44, 1.44, -1.36],
                    ..BASE
                },
                Noise::Iid,
                false,
                false,
                LAST,
            ),
            // A short tail, resolved only inside the first year: k_er 8,
            // k_p 2.5.
            "fast" => m1(
                5,
                Truth {
                    b: [9.81, 9.16, 1.0, -1.0],
                    ..BASE
                },
                Noise::Iid,
                false,
                false,
                LAST,
            ),
            "season" => m1(1, BASE, Noise::Iid, false, true, LAST),
            "season117" => m1(1, BASE, Noise::Iid, false, true, LAST - 3),
            "iid117" => m1(1, BASE, Noise::Iid, false, false, LAST - 3),
            "season114" => m1(1, BASE, Noise::Iid, false, true, LAST - 6),
            "iid114" => m1(1, BASE, Noise::Iid, false, false, LAST - 6),
            "seasonq" => m1(3, BASE, Noise::Iid, true, true, LAST),
            "m2y" => World::Model2 {
                stream: 6,
                dev_unit: 12,
                season: false,
                quarterly_origins: false,
                valuation: LAST,
            },
            "m2q" => World::Model2 {
                stream: 7,
                dev_unit: 3,
                season: false,
                quarterly_origins: false,
                valuation: LAST,
            },
            "m2season" => World::Model2 {
                stream: 6,
                dev_unit: 12,
                season: true,
                quarterly_origins: false,
                valuation: LAST,
            },
            "m2qo" => World::Model2 {
                stream: 8,
                dev_unit: 12,
                season: false,
                quarterly_origins: true,
                valuation: LAST,
            },
            // `m2qo` with the seasonal factor: origins start in all four
            // calendar quarters, so the season is not the age.
            "m2seasonq" => World::Model2 {
                stream: 8,
                dev_unit: 12,
                season: true,
                quarterly_origins: true,
                valuation: LAST,
            },
            "m2y114" => World::Model2 {
                stream: 6,
                dev_unit: 12,
                season: false,
                quarterly_origins: false,
                valuation: LAST - 6,
            },
            "m2season114" => World::Model2 {
                stream: 6,
                dev_unit: 12,
                season: true,
                quarterly_origins: false,
                valuation: LAST - 6,
            },
            other => panic!("unknown world {other}"),
        }
    }

    fn stream(&self) -> u64 {
        match *self {
            World::Model1 { stream, .. } | World::Model2 { stream, .. } => stream,
        }
    }

    fn valuation(&self) -> usize {
        match *self {
            World::Model1 { valuation, .. } | World::Model2 { valuation, .. } => valuation,
        }
    }
}

fn normal(rng: &mut StreamRng) -> f64 {
    norm_quantile(rng.next_open01())
}

/// A simulated book at the finest grain: origins `origin_months` apart, each
/// origin's cumulative paid and outstanding at every month 1 to 120 (index
/// `month - 1`), noisy, and its mean cumulative paid.
#[derive(Debug, Clone)]
struct Book {
    origin_months: usize,
    premium: Vec<f64>,
    paid: Vec<Vec<f64>>,
    os: Vec<Vec<f64>>,
    paid_mean: Vec<Vec<f64>>,
}

impl Book {
    fn new(origin_months: usize) -> Self {
        Self {
            origin_months,
            premium: vec![],
            paid: vec![],
            os: vec![],
            paid_mean: vec![],
        }
    }
}

/// The seasonal factor of the month that ends `month` months after an
/// origin that starts at calendar month `start`.
fn season_factor(start: usize, month: usize) -> f64 {
    SEASON[((start + month - 1) / 3) % 4]
}

fn annual_premium(year: usize) -> f64 {
    (100_000.0 * 1.02f64.powi(year as i32)).round()
}

fn model_1_book(
    rng: &mut StreamRng,
    truth: Truth,
    noise: Noise,
    quarterly: bool,
    season: bool,
) -> Book {
    let mut book = Book::new(if quarterly { 3 } else { 12 });
    let n = LAST / book.origin_months;
    // Four quarters' independent noises add to one annual noise.
    let scale = if quarterly { 0.5 } else { 1.0 };
    let (k_er, k_p) = (truth.k_er(), truth.k_p());
    let chol = (1.0 - truth.rho * truth.rho).sqrt();
    for i in 0..n {
        let (year, per_year) = if quarterly { (i / 4, 4.0) } else { (i, 1.0) };
        let premium = annual_premium(year) / per_year;
        let (z1, z2) = (normal(rng), normal(rng));
        let rlr = truth.rlr(truth.tau_rlr * z1);
        let rrf = truth.rrf(truth.tau_rrf * (truth.rho * z1 + chol * z2));
        let (mut cum, mut before) = (0.0, 0.0);
        let (mut walk, mut ar) = (0.0, truth.sigma_os * scale * normal(rng));
        let (mut pd, mut os, mut mean) = (vec![], vec![], vec![]);
        for m in 1..=LAST {
            let t = m as f64 / 12.0;
            let curve = premium * paid(t, k_er, k_p, rlr, rrf);
            let factor = if season {
                season_factor(i * book.origin_months, m)
            } else {
                1.0
            };
            let increment = (curve - before) * factor;
            before = curve;
            cum += increment;
            let os_mean = premium * outstanding(t, k_er, k_p, rlr);
            let (e_pd, e_os) = match noise {
                Noise::Iid => (
                    truth.sigma_pd * scale * normal(rng),
                    truth.sigma_os * scale * normal(rng),
                ),
                Noise::Persistent => {
                    walk += (PERSIST_PHI * scale * scale * increment.max(0.0)).sqrt() * normal(rng);
                    ar = PERSIST_RHO * ar
                        + (1.0 - PERSIST_RHO * PERSIST_RHO).sqrt()
                            * truth.sigma_os
                            * scale
                            * normal(rng);
                    (walk, ar)
                }
            };
            pd.push(cum + e_pd);
            os.push(os_mean + e_os);
            mean.push(cum);
        }
        book.premium.push(premium);
        book.paid.push(pd);
        book.os.push(os);
        book.paid_mean.push(mean);
    }
    book
}

/// The model 2 process's population, on the `o` scale: origin effects on
/// all four parameters and development effects on all four.
const M2_ORIGIN_SD: [f64; 4] = [1.0, 1.0, 0.4, 0.2];
const M2_DEV_SD: [f64; 4] = [1.0, 1.0, 0.5, 0.5];
/// The log sd of a month's paid increment and of a cell's outstanding.
const M2_SIGMA_PD: f64 = 0.3;
const M2_SIGMA_OS: f64 = 0.1;

fn model_2_book(rng: &mut StreamRng, dev_unit: usize, season: bool, quarterly: bool) -> Book {
    let scale = [0.1, 0.1, 0.2, 0.1];
    let medians = [3.0, 1.0, 0.7, 0.8];
    let levels = LAST / dev_unit;
    let dev: Vec<[f64; 4]> = (0..levels)
        .map(|_| [0, 1, 2, 3].map(|p| M2_DEV_SD[p] * normal(rng)))
        .collect();
    let mut book = Book::new(if quarterly { 3 } else { 12 });
    for i in 0..LAST / book.origin_months {
        let premium = if quarterly {
            annual_premium(i / 4) / 4.0
        } else {
            annual_premium(i)
        };
        let u = [0, 1, 2, 3].map(|p| M2_ORIGIN_SD[p] * normal(rng));
        let (mut cum, mut cum_mean) = (0.0, 0.0);
        let (mut pd, mut os, mut mean) = (vec![], vec![], vec![]);
        for m in 1..=LAST {
            let d = dev[(m - 1) / dev_unit];
            let [k_er, k_p, rlr, rrf] =
                [0, 1, 2, 3].map(|p| medians[p] * (scale[p] * (BASE.b[p] + u[p] + d[p])).exp());
            let t = m as f64 / 12.0;
            let t0 = (m - 1) as f64 / 12.0;
            let factor = if season {
                season_factor(i * book.origin_months, m)
            } else {
                1.0
            };
            let increment =
                premium * (paid(t, k_er, k_p, rlr, rrf) - paid(t0, k_er, k_p, rlr, rrf)) * factor;
            // Mean-preserving, so that expected increments follow the closed
            // form on every grain.
            let e = (M2_SIGMA_PD * normal(rng) - 0.5 * M2_SIGMA_PD * M2_SIGMA_PD).exp();
            cum += increment * e;
            cum_mean += increment;
            pd.push(cum);
            os.push(premium * outstanding(t, k_er, k_p, rlr) * (M2_SIGMA_OS * normal(rng)).exp());
            mean.push(cum_mean);
        }
        book.premium.push(premium);
        book.paid.push(pd);
        book.os.push(os);
        book.paid_mean.push(mean);
    }
    book
}

fn simulate(world: World, book: u64) -> Book {
    let mut rng = StreamRng::new(BOOK_SEED, world.stream() * 10_000 + book);
    match world {
        World::Model1 {
            truth,
            noise,
            quarterly_origins,
            season,
            ..
        } => model_1_book(&mut rng, truth, noise, quarterly_origins, season),
        World::Model2 {
            dev_unit,
            season,
            quarterly_origins,
            ..
        } => model_2_book(&mut rng, dev_unit, season, quarterly_origins),
    }
}

/// A grain of the study: months between origins and between ages.
#[derive(Debug, Clone, Copy, PartialEq)]
struct View {
    origin: usize,
    development: usize,
}

impl View {
    fn parse(name: &str) -> View {
        let (origin, development) = match name {
            "aa" => (12, 12),
            "aq" => (12, 3),
            "am" => (12, 1),
            "qq" => (3, 3),
            "qm" => (3, 1),
            other => panic!("unknown grain {other}"),
        };
        View {
            origin,
            development,
        }
    }
}

/// An observed cell: origin year and month, age, paid, outstanding, premium.
type Row = (i32, u8, usize, f64, f64, f64);

/// A book read at a grain.
#[derive(Debug, Clone)]
struct Read {
    rows: Vec<Row>,
    view: View,
    /// The realised future paid to the square's last age.
    outcome: f64,
    /// The mean of paid at the last age less the latest paid. Only under
    /// the `iid` noises is this the outcome's mean given the latest paid.
    expected: f64,
}

fn grain(months: usize) -> Grain {
    match months {
        12 => Grain::Year,
        3 => Grain::Quarter,
        1 => Grain::Month,
        other => panic!("no grain of {other} months"),
    }
}

impl Book {
    /// The value of `of` at origin `i` of `origin_months` at age `age`: the
    /// book's own, or the sum of its quarterly origins in an annual origin.
    fn cell(&self, of: &[Vec<f64>], origin_months: usize, i: usize, age: usize) -> f64 {
        if origin_months == self.origin_months {
            return of[i][age - 1];
        }
        assert_eq!((origin_months, self.origin_months), (12, 3));
        (0..4)
            .filter(|q| age > 3 * q)
            .map(|q| of[4 * i + q][age - 3 * q - 1])
            .sum()
    }

    fn read(&self, view: View, valuation: usize) -> Read {
        let mut rows = vec![];
        let (mut outcome, mut expected) = (0.0, 0.0);
        for i in 0..LAST / view.origin {
            let (year, month) = (i * view.origin / 12, i * view.origin % 12);
            let premium = if view.origin == self.origin_months {
                self.premium[i]
            } else {
                (0..4).map(|q| self.premium[4 * i + q]).sum()
            };
            let latest = (valuation - i * view.origin) / view.development * view.development;
            let mut age = view.development;
            while age <= latest {
                rows.push((
                    2011 + year as i32,
                    1 + month as u8,
                    age,
                    self.cell(&self.paid, view.origin, i, age),
                    self.cell(&self.os, view.origin, i, age),
                    premium,
                ));
                age += view.development;
            }
            let observed = self.cell(&self.paid, view.origin, i, latest);
            outcome += self.cell(&self.paid, view.origin, i, LAST) - observed;
            expected += self.cell(&self.paid_mean, view.origin, i, LAST) - observed;
        }
        Read {
            rows,
            view,
            outcome,
            expected,
        }
    }
}

impl Read {
    fn cells(&self) -> usize {
        2 * self.rows.len()
    }

    fn triangle(&self) -> Triangle {
        let origin: Vec<Month> = self
            .rows
            .iter()
            .map(|r| Month::new(r.0, r.1).unwrap())
            .collect();
        let ages: Vec<Lag> = self.rows.iter().map(|r| r.2 as Lag).collect();
        let pd: Vec<f64> = self.rows.iter().map(|r| r.3).collect();
        let os: Vec<f64> = self.rows.iter().map(|r| r.4).collect();
        let premium: Vec<f64> = self.rows.iter().map(|r| r.5).collect();
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &pd), ("outstanding", &os), ("premium", &premium)],
            origin_grain: grain(self.view.origin),
            development_grain: grain(self.view.development),
            cumulative: true,
        })
        .unwrap()
    }
}

fn priors(name: &str, model_2: bool) -> CompartmentalPriors {
    let base = if model_2 {
        CompartmentalPriors::gesmann_morris_2()
    } else {
        CompartmentalPriors::gesmann_morris()
    };
    let wide = |median: f64| PositivePrior::LogNormal { median, sd: 1.0 };
    match name {
        "preset" => base,
        "diffuse" => CompartmentalPriors {
            k_er: wide(3.0),
            k_p: wide(1.0),
            rlr: wide(0.7),
            rrf: wide(0.8),
            ..base
        },
        "ratewide" => CompartmentalPriors {
            k_er: wide(3.0),
            k_p: wide(1.0),
            ..base
        },
        "sigmawide" => {
            let log_sigma = RealPrior::Normal {
                mean: 0.2f64.ln(),
                sd: 1.0,
            };
            CompartmentalPriors {
                log_sigma: [log_sigma, log_sigma],
                ..base
            }
        }
        other => panic!("unknown priors {other}"),
    }
}

/// The arm's model. `study` is a shorter, coarser run than the model's own:
/// 500 warm-up and 500 kept draws a chain, target acceptance 0.9 for model 1
/// and 0.99 for model 2 (against 0.99 and 0.999), 4,000 simulations. On books
/// 0 to 2 of `iid` the coarser target moved the total reserve's mean by at
/// most 0.1 sd and halved the time, with a few divergences that the default
/// sampler does not have, so the panels score reserves with it and the
/// `default` arms measure the sampler.
fn model(arm: &Arm, seed: u64) -> Compartmental {
    let model_2 = arm.model == "m2";
    let pr = priors(&arm.priors, model_2);
    let mut m = if model_2 {
        Compartmental::model_2(pr)
    } else {
        Compartmental::new(pr)
    };
    m.seed = seed;
    if arm.model == "m1i" {
        m.paid_response = PaidResponse::Incremental;
    }
    match arm.sampler.as_str() {
        "default" => {}
        "study" => {
            m.sampler = Sampler {
                tune: 500,
                draws: 500,
                target_accept: if model_2 { 0.99 } else { 0.9 },
                ..m.sampler
            };
            m.n_sims = 4000;
        }
        other => panic!("unknown sampler {other}"),
    }
    m
}

#[derive(Debug, Clone)]
struct Arm {
    label: String,
    world: String,
    grain: String,
    model: String,
    priors: String,
    sampler: String,
}

impl Arm {
    fn parse(spec: &str) -> Arm {
        let f: Vec<&str> = spec.split(':').collect();
        assert_eq!(f.len(), 5, "arm {spec}: world:grain:model:prior:sampler");
        Arm {
            label: f.join("_"),
            world: f[0].into(),
            grain: f[1].into(),
            model: f[2].into(),
            priors: f[3].into(),
            sampler: f[4].into(),
        }
    }
}

/// The parameters whose truth is scored, by the posterior's names. The
/// sigmas depend on the grain: the sum of four independent quarterly noises
/// is an annual noise, so the `earned` world's quarterly origins carry half
/// of it, and a cell of `n` months of model 2's log-normal months has the
/// log sd of their sum. The `persist` world's noise is not the model's, so it
/// has no true sigma.
fn scored(world: World, view: View) -> Vec<(&'static str, f64)> {
    match world {
        World::Model1 {
            truth,
            noise,
            quarterly_origins,
            ..
        } => {
            let scale = if quarterly_origins && view.origin == 3 {
                0.5
            } else {
                1.0
            };
            let mut out = vec![
                ("k_er", truth.k_er()),
                ("k_p", truth.k_p()),
                ("b_RLR", truth.b[2]),
                ("b_RRF", truth.b[3]),
                ("tau_RLR", truth.tau_rlr),
                ("tau_RRF", truth.tau_rrf),
                ("rho", truth.rho),
            ];
            if noise == Noise::Iid {
                out.push(("sigma_os", scale * truth.sigma_os));
                out.push(("sigma_pd", scale * truth.sigma_pd));
            }
            out
        }
        World::Model2 { .. } => {
            let n = view.development as f64;
            let cell = (1.0 + ((M2_SIGMA_PD * M2_SIGMA_PD).exp() - 1.0) / n)
                .ln()
                .sqrt();
            vec![
                ("b_ker", BASE.b[0]),
                ("b_kp", BASE.b[1]),
                ("b_RLR", BASE.b[2]),
                ("b_RRF", BASE.b[3]),
                ("tau_RLR", M2_ORIGIN_SD[2]),
                ("tau_RRF", M2_ORIGIN_SD[3]),
                ("tau_dev_kp", M2_DEV_SD[1]),
                ("sigma_os", M2_SIGMA_OS),
                ("sigma_pd", cell),
            ]
        }
    }
}

fn draws(fit: &CompartmentalFit, name: &str) -> Option<Vec<f64>> {
    let j = fit.posterior.names().iter().position(|n| n == name)?;
    Some(fit.posterior.parameter(j).concat())
}

/// The fit's line of an arm's CSV.
fn line(book: u64, read: &Read, fit: &CompartmentalFit, secs: f64, world: World) -> String {
    let d = &fit.diagnostics;
    let total = fit.reserves.total();
    let mut out = format!(
        "{book},{},{secs:.2},{:.5},{:.1},{:.1},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.3}",
        read.cells(),
        d.max_rhat,
        d.min_ess_bulk,
        d.min_ess_tail,
        d.divergences,
        d.warnings.len(),
        fit.rate_swaps,
        read.outcome,
        read.expected,
        total.mean(),
        total.std_dev(),
        total.quantile(0.05).unwrap(),
        total.quantile(0.95).unwrap(),
        100.0 * total.cdf(read.outcome),
    );
    for (name, truth) in scored(world, read.view) {
        let (mean, pct) = match draws(fit, name) {
            Some(v) => (
                v.iter().sum::<f64>() / v.len() as f64,
                100.0 * v.iter().filter(|&&x| x < truth).count() as f64 / v.len() as f64,
            ),
            None => (f64::NAN, f64::NAN),
        };
        write!(out, ",{mean:.5},{pct:.2}").unwrap();
    }
    out.push('\n');
    out
}

fn header(world: World, view: View) -> String {
    let mut h = String::from(
        "book,cells,seconds,max_rhat,ess_bulk,ess_tail,divergences,warnings,rate_swaps,\
         outcome,expected,mean,sd,q05,q95,percentile",
    );
    for (name, _) in scored(world, view) {
        write!(h, ",{name}_mean,{name}_pct").unwrap();
    }
    h.push('\n');
    h
}

fn out_dir() -> PathBuf {
    std::env::var_os("SUBANNUAL_OUT").map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../target/compartmental-subannual"
            ))
        },
        PathBuf::from,
    )
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Runs `SUBANNUAL_ARMS`, comma-separated, on books `SUBANNUAL_FIRST` up to
/// `SUBANNUAL_BOOKS`.
#[test]
#[ignore]
fn panel() {
    let spec = std::env::var("SUBANNUAL_ARMS").expect("SUBANNUAL_ARMS");
    let (first, books) = (env_u64("SUBANNUAL_FIRST", 0), env_u64("SUBANNUAL_BOOKS", 1));
    let dir = out_dir();
    std::fs::create_dir_all(&dir).unwrap();
    for arm in spec.split(',').map(Arm::parse) {
        let world = World::parse(&arm.world);
        let view = View::parse(&arm.grain);
        let path = dir.join(format!("{}.csv", arm.label));
        let done = std::fs::read_to_string(&path).unwrap_or_default();
        if done.is_empty() {
            std::fs::write(&path, header(world, view)).unwrap();
        }
        let seen: Vec<u64> = done
            .lines()
            .skip(1)
            .filter_map(|l| l.split(',').next()?.parse().ok())
            .collect();
        for b in first..books {
            if seen.contains(&b) {
                continue;
            }
            let read = simulate(world, b).read(view, world.valuation());
            let triangle = read.triangle();
            let start = Instant::now();
            let result = model(&arm, 1000 + b).fit(
                &triangle,
                "paid",
                Outstanding::Column("outstanding"),
                "premium",
            );
            let secs = start.elapsed().as_secs_f64();
            let text = match result {
                Ok(fit) => line(b, &read, &fit, secs, world),
                Err(e) => format!("{b},{},{secs:.2},error,{e}\n", read.cells()),
            };
            eprint!("{} {text}", arm.label);
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(text.as_bytes()).unwrap();
        }
    }
}

/// `prior_bands.csv`: the 5%, 50% and 95% of the mean paid and outstanding
/// over premium at months 1, 3, 6, 12, 24 and 120 of a monthly square,
/// under each model 1 prior set.
#[test]
#[ignore]
fn prior_bands() {
    let triangle = simulate(World::parse("iid"), 0)
        .read(View::parse("am"), LAST)
        .triangle();
    let mut out = String::from("priors,quantity,month,q05,q50,q95\n");
    for name in ["preset", "ratewide", "diffuse"] {
        let m = Compartmental {
            seed: 1,
            ..Compartmental::new(priors(name, false))
        };
        let paths = m
            .prior_predictive(&triangle, "premium", 4000)
            .expect("prior predictive");
        let premium = annual_premium(0);
        for (quantity, dist) in [("paid", &paths.paid), ("outstanding", &paths.outstanding)] {
            for month in [1i64, 3, 6, 12, 24, 120] {
                let key = vec![KeyValue::from(Period::year(2011)), KeyValue::Int(month)];
                let d = dist.marginal(&key).unwrap();
                let q = |p: f64| d.quantile(p).unwrap() / premium;
                writeln!(
                    out,
                    "{name},{quantity},{month},{:.4},{:.4},{:.4}",
                    q(0.05),
                    q(0.5),
                    q(0.95)
                )
                .unwrap();
            }
        }
    }
    eprint!("{out}");
    let dir = out_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("prior_bands.csv"), out).unwrap();
}

/// `zero_cells.csv`: under a Poisson number of payments spread over a
/// book's development by its mean increments, the share of `m2y` books
/// with an empty cell at each grain, which the log-normal likelihood on
/// increments refuses, and the share of cells that are empty.
#[test]
#[ignore]
fn zero_cells() {
    let mut out = String::from("payments_per_origin,grain,books,books_refused,cells_empty\n");
    for payments in [500.0, 2000.0, 10000.0] {
        for (name, months) in [("aa", 12usize), ("aq", 3), ("am", 1)] {
            let (mut refused, mut cells, mut empty) = (0, 0usize, 0usize);
            let books = 200u64;
            for b in 0..books {
                let mut rng = StreamRng::new(BOOK_SEED, 90_000 + b);
                let book = simulate(World::parse("m2y"), b);
                let mut any = false;
                for (i, paid_mean) in book.paid_mean.iter().enumerate().take(10) {
                    let ultimate = paid_mean[LAST - 1];
                    let mut before = 0.0;
                    for m in (months..=LAST - 12 * i).step_by(months) {
                        let mean = paid_mean[m - 1];
                        let lambda = payments * (mean - before) / ultimate;
                        before = mean;
                        cells += 1;
                        if rng.next_open01() < (-lambda).exp() {
                            empty += 1;
                            any = true;
                        }
                    }
                }
                refused += u64::from(any);
            }
            writeln!(
                out,
                "{payments},{name},{books},{refused},{:.5}",
                empty as f64 / cells as f64
            )
            .unwrap();
        }
    }
    eprint!("{out}");
    let dir = out_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("zero_cells.csv"), out).unwrap();
}

/// Writes the books of `SUBANNUAL_EXPORT` (`world:grain`, comma-separated)
/// for the crate's test-only comparison of development effects
/// (`bayes::compartmental::subannual_study`): one
/// `books/<world>_<grain>_<book>.csv` each, the first line
/// `# outcome <x> expected <y>`.
#[test]
#[ignore]
fn export_books() {
    let spec = std::env::var("SUBANNUAL_EXPORT").expect("SUBANNUAL_EXPORT");
    let books = env_u64("SUBANNUAL_BOOKS", 1);
    let dir = out_dir().join("books");
    std::fs::create_dir_all(&dir).unwrap();
    for item in spec.split(',') {
        let (w, g) = item.split_once(':').expect("world:grain");
        let world = World::parse(w);
        for b in 0..books {
            let read = simulate(world, b).read(View::parse(g), world.valuation());
            let mut out = format!(
                "# outcome {} expected {}\n# {} {}\n",
                read.outcome, read.expected, read.view.origin, read.view.development
            );
            for r in &read.rows {
                writeln!(out, "{},{},{},{},{},{}", r.0, r.1, r.2, r.3, r.4, r.5).unwrap();
            }
            std::fs::write(dir.join(format!("{w}_{g}_{b}.csv")), out).unwrap();
        }
    }
}

/// An annual cell is the monthly cell at the year end, on every grain.
#[test]
fn the_grains_read_one_book() {
    let book = simulate(World::parse("iid"), 3);
    let a = book.read(View::parse("aa"), LAST);
    let m = book.read(View::parse("am"), LAST);
    assert_eq!((a.cells(), m.cells()), (110, 1320));
    for r in &a.rows {
        let same = m
            .rows
            .iter()
            .find(|x| x.0 == r.0 && x.2 == r.2)
            .expect("the same cell");
        assert_eq!((same.3, same.4), (r.3, r.4));
    }
    assert_eq!((a.outcome, a.expected), (m.outcome, m.expected));
}

/// In the earned world an annual origin is the sum of its four quarters, so
/// the premium is earned through the year.
#[test]
fn an_annual_origin_is_its_quarters() {
    let book = simulate(World::parse("earned"), 0);
    assert_eq!(book.premium.len(), 40);
    let v = book.cell(&book.paid, 12, 1, 12);
    let by_hand = book.paid[4][11] + book.paid[5][8] + book.paid[6][5] + book.paid[7][2];
    assert_eq!(v, by_hand);
    assert_eq!(book.cell(&book.paid, 12, 1, 3), book.paid[4][2]);
    assert_eq!(book.read(View::parse("qq"), LAST).cells(), 1640);
}

/// Model 2's process keeps every paid increment and outstanding positive,
/// as its likelihood needs.
#[test]
fn model_2_books_are_positive() {
    let book = simulate(World::parse("m2y"), 0);
    for row in &book.paid {
        assert!(row.windows(2).all(|w| w[1] > w[0]));
    }
    assert!(book.os.iter().flatten().all(|&v| v > 0.0));
}
