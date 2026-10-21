//! Meyers' ultimate percentile test on 200 Schedule P triangles, run on
//! prospicio's reserving models (`docs/design/reserving-calibration.md`,
//! issue #392).
//!
//! Each of 50 triangles in commercial auto (ca), personal auto (pa),
//! workers' compensation (wc) and other liability (ol) is fitted on its
//! upper triangle (accident years 1988 to 1997 at the 1997 statement); the
//! percentile of the outcome, the sum of the ten origins' cumulative values
//! at age 10 (Meyers 2019, CAS Monograph 8, Table 3.1), is read under the
//! model's predictive distribution, and the 200 percentiles, and the 50 of
//! each line, are tested for uniformity (Meyers' D and the two-sided
//! Kolmogorov-Smirnov statistic against the `136 / sqrt(n)` bound).
//!
//! This file holds the harness (PRs 2 to 4 of the note): Mack on paid and
//! incurred and the ODP bootstrap on paid (PR 2); `BayesOdp` on paid and
//! Meyers' CRC (paid and incurred), CSR (paid) and CAY (incurred) at his
//! priors and the library's NUTS defaults (PR 3); Gesmann and Morris'
//! compartmental model 1 on paid, and model 2 as an optional arm, at the
//! paper's presets (PR 4). The study is
//! `#[ignore]`d and run by hand; nothing in CI runs it. The file's
//! `[[test]]` entry requires `slow-bayes`, so a plain `cargo test` does not
//! build it and CI's `bayes` job runs it without `--include-ignored`: the
//! unit tests below the study (the pure pieces on small inline cases, a
//! cut-down NUTS check and the wiring check on group 353) run there, in
//! release, and the study does not.
//!
//! Run (from the repository root; the panel is built from the pip package
//! `cas-schedule-p` at run time and is not stored in the repo):
//!
//! ```text
//! uv run --no-project --with "cas-schedule-p[screens]==2026.6.13" python -I validation/scripts/calibration_panel.py
//! cargo test --release -p prospicio-validation --features slow-bayes --test reserving_calibration_meyers -- --ignored --nocapture
//! ```
//!
//! The script writes `target/calibration/panel.csv` (set
//! `RISK_RS_CALIBRATION_PANEL` for another path); the study reads it and
//! writes `percentiles.csv`, `pp.csv` and `ks.csv` beside it. Options, all
//! environment variables read by the study itself:
//!
//! * `RISK_RS_CALIBRATION_LINES`: for example `ca,pa` (default all four);
//! * `RISK_RS_CALIBRATION_ARMS`: for example `mack,odp,crc` (default:
//!   `mack,odp,bayes_odp,crc,csr,cay,compartmental`; `crc` runs paid and
//!   incurred, `csr` paid, `cay` incurred, `bayes_odp` and `compartmental`
//!   paid; `compartmental_2`, model 2 on paid, takes about 59 s a fit
//!   against model 1's 2.1 s on the panel (25 to 30 times slower) and runs
//!   only when named);
//! * `RISK_RS_CALIBRATION_LIMIT`: the first n triangles of each line, for a
//!   timing probe (for example 2);
//! * `RISK_RS_CALIBRATION_SEED_SALT`: any text, mixed into every fit's seed
//!   (unset: the recorded run, bit for bit), to measure how far each arm's
//!   D moves with the seed alone. A salted run asserts no pins.
//!
//! The script also writes `panel_check.csv` (the diff against Table A.2,
//! with each code's rank in the screen) and `panel_meta.csv` (the package
//! versions and the panel's sha256), which the study prints.
//!
//! The study asserts the harness and the panel (decision 7): the panel's
//! shape, group 353 commercial auto's square against
//! `validation/data/clrd_comauto_353.csv`, and Mack's D near Meyers'
//! published values. It also asserts the pins (`PINS`, `TOLERANCES`): every
//! arm's D per line and on all 200, as the user reviewed it on 2026-10-21,
//! within a tolerance set from the seed noise, on the pinned panel, the
//! default seed and the whole panel (a narrowed run asserts only the cells
//! it ran; `RISK_RS_CALIBRATION_LIMIT` or a seed salt asserts no pins). The
//! unit tests `the_arms_reproduce_meyers_on_group_353` and
//! `the_bayesian_arms_reproduce_meyers_on_group_353` check the arms'
//! wiring on that square in CI's `bayes` job.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use prospicio_prob::{Distribution, Lognormal, PredictiveDistribution};
use prospicio_reserving::bayes::{
    Compartmental, CompartmentalFit, CompartmentalPriors, MeyersModel, Outstanding,
};
use prospicio_reserving::{
    BayesOdp, DevelopmentColumn, Diagnostics, Grain, Lag, Long, Mack, Meyers, Month, OdpBootstrap,
    Triangle,
};
use rayon::prelude::*;

/// Simulations of the ODP bootstrap (Meyers p.12 uses 10,000).
const ODP_SIMS: usize = 10_000;

/// The command that builds the panel, quoted in every failure message.
const PANEL_COMMAND: &str = "uv run --no-project --with \"cas-schedule-p[screens]==2026.6.13\" python -I validation/scripts/calibration_panel.py";

/// Target acceptance of a NUTS arm's first fit: the library's default for
/// `BayesOdp` and `Meyers` (0.95, above the sampler's 0.8 for the reasons
/// in their docs).
const FIRST_ACCEPT: f64 = 0.95;
/// Target acceptance of the refit of a fit whose diagnostics fail (the note,
/// decision 5; the remedy of the Meyers backtest sweep).
const REFIT_ACCEPT: f64 = 0.99;
/// R-hat above which a NUTS fit is refitted.
const RHAT_LIMIT: f64 = 1.05;

const ORIGINS: usize = 10;
const FIRST_ORIGIN: i32 = 1988;

// ---------------------------------------------------------------------------
// The panel
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Line {
    Ca,
    Pa,
    Wc,
    Ol,
}

impl Line {
    const ALL: [Line; 4] = [Line::Ca, Line::Pa, Line::Wc, Line::Ol];

    fn code(self) -> &'static str {
        match self {
            Line::Ca => "ca",
            Line::Pa => "pa",
            Line::Wc => "wc",
            Line::Ol => "ol",
        }
    }

    fn parse(code: &str) -> Option<Line> {
        Line::ALL.into_iter().find(|l| l.code() == code)
    }
}

/// What a model is fitted on: Meyers' paid, or his incurred (case
/// incurred, `incurred_loss - bulk_loss`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Basis {
    Paid,
    Incurred,
}

impl Basis {
    /// Also the measure's name in the training triangle.
    fn name(self) -> &'static str {
        match self {
            Basis::Paid => "paid",
            Basis::Incurred => "incurred",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    paid: f64,
    incurred: f64,
    premium: f64,
}

/// One company's full 10 x 10 square, `cells[origin][development]`: origin
/// 1988 to 1997, development 12 to 120 months.
#[derive(Clone, Debug)]
struct PanelTriangle {
    line: Line,
    group: String,
    cells: [[Cell; ORIGINS]; ORIGINS],
}

impl PanelTriangle {
    fn value(&self, basis: Basis, o: usize, d: usize) -> f64 {
        let c = &self.cells[o][d];
        match basis {
            Basis::Paid => c.paid,
            Basis::Incurred => c.incurred,
        }
    }

    /// Each origin's cumulative value at age 10.
    fn outcome_by_origin(&self, basis: Basis) -> [f64; ORIGINS] {
        std::array::from_fn(|o| self.value(basis, o, ORIGINS - 1))
    }

    /// Meyers' outcome: the sum of those over the ten origins.
    fn outcome(&self, basis: Basis) -> f64 {
        self.outcome_by_origin(basis).iter().sum()
    }

    /// The upper triangle (calendar years up to 1997): the 55 cells with
    /// `origin + development <= 9` (zero-based). No model can see the rest.
    fn training(&self) -> Triangle {
        let mut origin = Vec::new();
        let mut age: Vec<Lag> = Vec::new();
        let (mut paid, mut incurred, mut premium) = (Vec::new(), Vec::new(), Vec::new());
        for o in 0..ORIGINS {
            for d in 0..ORIGINS - o {
                let c = &self.cells[o][d];
                origin.push(Month::january(FIRST_ORIGIN + o as i32));
                age.push(12 * (d as Lag + 1));
                paid.push(c.paid);
                incurred.push(c.incurred);
                premium.push(c.premium);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&age),
            values: &[
                ("paid", &paid),
                ("incurred", &incurred),
                ("premium", &premium),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap_or_else(|e| panic!("{} {}: {e}", self.line.code(), self.group))
    }
}

const PANEL_HEADER: &str =
    "line,group_code,origin,development,paid,incurred,case_incurred,premium,premium_direct";

/// Parses the panel CSV written by `validation/scripts/calibration_panel.py`.
/// `incurred` in the triangle is the file's `case_incurred`. Every triangle
/// must be a complete 10 x 10 square without repeated cells.
fn parse_panel(text: &str) -> Result<Vec<PanelTriangle>, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or("the panel file is empty")?;
    if header.trim() != PANEL_HEADER {
        return Err(format!(
            "the panel's header is {header:?}, expected {PANEL_HEADER:?}"
        ));
    }
    let mut index: BTreeMap<(Line, String), usize> = BTreeMap::new();
    let mut triangles: Vec<(PanelTriangle, BTreeSet<(usize, usize)>)> = Vec::new();
    for (n, row) in lines.enumerate() {
        let f: Vec<&str> = row.split(',').collect();
        let row_no = n + 2;
        if f.len() != 9 {
            return Err(format!("row {row_no}: {} fields, expected 9", f.len()));
        }
        let line = Line::parse(f[0]).ok_or(format!("row {row_no}: unknown line {:?}", f[0]))?;
        let num = |i: usize| -> Result<f64, String> {
            f[i].parse::<f64>()
                .map_err(|_| format!("row {row_no}: {:?} is not a number", f[i]))
        };
        let origin = num(2)? as i32 - FIRST_ORIGIN;
        let months = num(3)? as i64;
        if !(0..ORIGINS as i32).contains(&origin)
            || months % 12 != 0
            || !(12..=120).contains(&months)
        {
            return Err(format!(
                "row {row_no}: origin {} / development {} is outside the 10 x 10 square",
                f[2], f[3]
            ));
        }
        let (o, d) = (origin as usize, (months / 12 - 1) as usize);
        let at = *index.entry((line, f[1].to_string())).or_insert_with(|| {
            triangles.push((
                PanelTriangle {
                    line,
                    group: f[1].to_string(),
                    cells: [[Cell::default(); ORIGINS]; ORIGINS],
                },
                BTreeSet::new(),
            ));
            triangles.len() - 1
        });
        let (t, seen) = &mut triangles[at];
        if !seen.insert((o, d)) {
            return Err(format!(
                "row {row_no}: {} {} repeats origin {} development {}",
                line.code(),
                f[1],
                f[2],
                f[3]
            ));
        }
        t.cells[o][d] = Cell {
            paid: num(4)?,
            incurred: num(6)?,
            premium: num(7)?,
        };
    }
    for (t, seen) in &triangles {
        if seen.len() != ORIGINS * ORIGINS {
            return Err(format!(
                "{} {} has {} cells, expected 100",
                t.line.code(),
                t.group,
                seen.len()
            ));
        }
    }
    Ok(triangles.into_iter().map(|(t, _)| t).collect())
}

/// Where the panel is read from: `RISK_RS_CALIBRATION_PANEL`, else
/// `target/calibration/panel.csv` under the workspace root.
fn panel_path() -> PathBuf {
    match std::env::var_os("RISK_RS_CALIBRATION_PANEL") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/calibration/panel.csv"),
    }
}

/// Reads and parses the panel at `path`. An absent file is an error that
/// holds the command that builds it and the path looked for; the study
/// never reaches for the network itself.
fn load_panel_at(path: &Path) -> Result<Vec<PanelTriangle>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "cannot read the calibration panel at {} ({e}). Build it from the repository root with\n    {PANEL_COMMAND}\n(set RISK_RS_CALIBRATION_PANEL to read another path)",
            path.display()
        )
    })?;
    parse_panel(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Group 353 commercial auto's full square from the committed
/// `validation/data/clrd_comauto_353.csv` (Meyers' Tables 2.1 to 2.3; its
/// `incurred` is already case incurred).
fn group_353() -> PanelTriangle {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/clrd_comauto_353.csv");
    let text = std::fs::read_to_string(&path).expect("validation/data/clrd_comauto_353.csv");
    let mut rows = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    assert_eq!(
        rows.next().map(str::trim),
        Some("origin,development,paid,incurred,premium")
    );
    let mut csv = String::from(PANEL_HEADER);
    csv.push('\n');
    for r in rows {
        let f: Vec<&str> = r.trim().split(',').collect();
        assert_eq!(f.len(), 5, "{r}");
        let _ = writeln!(csv, "ca,353,{},{},{},,{},{},", f[0], f[1], f[2], f[3], f[4]);
    }
    let mut panel = parse_panel(&csv).expect("group 353's square");
    assert_eq!(panel.len(), 1);
    panel.remove(0)
}

// ---------------------------------------------------------------------------
// Percentiles and the uniformity test
// ---------------------------------------------------------------------------

/// The percentile (0 to 100) of `x` under the log-normal with this mean and
/// standard deviation (Meyers p.7, following Mack's suggestion).
fn lognormal_percentile(mean: f64, sd: f64, x: f64) -> Result<f64, String> {
    if !(mean.is_finite() && sd.is_finite() && x.is_finite())
        || mean <= 0.0
        || sd <= 0.0
        || x <= 0.0
    {
        return Err(format!(
            "no log-normal for mean {mean}, sd {sd} at outcome {x}"
        ));
    }
    let s2 = (1.0 + (sd / mean).powi(2)).ln();
    let dist = Lognormal::new(mean.ln() - s2 / 2.0, s2.sqrt()).map_err(|e| e.to_string())?;
    Ok(100.0 * dist.cdf(x))
}

/// Meyers' D, in percentage points: `max_i |p_(i) - 100 i / n|` over the
/// sorted percentiles.
fn meyers_d(percentiles: &[f64]) -> f64 {
    let n = percentiles.len() as f64;
    sorted(percentiles)
        .iter()
        .enumerate()
        .map(|(i, p)| (p - 100.0 * (i as f64 + 1.0) / n).abs())
        .fold(0.0, f64::max)
}

/// The two-sided Kolmogorov-Smirnov statistic in percentage points,
/// `max_i max(100 i / n - p_(i), p_(i) - 100 (i - 1) / n)`: at least Meyers'
/// D and at most `100 / n` above it.
fn ks_two_sided(percentiles: &[f64]) -> f64 {
    let n = percentiles.len() as f64;
    sorted(percentiles)
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (lo, hi) = (100.0 * i as f64 / n, 100.0 * (i as f64 + 1.0) / n);
            (hi - p).max(p - lo)
        })
        .fold(0.0, f64::max)
}

/// The 5% bound `1.36 / sqrt(n)` in percentage points.
fn ks_bound(n: usize) -> f64 {
    136.0 / (n as f64).sqrt()
}

/// The p-p plot's points for the sorted percentiles: `(e_i, f_i, p_(i))`.
/// `e_i = 100 i / (n + 1)` is the expected percentile, Meyers' horizontal
/// axis (p.8); `f_i = 100 i / n` is the point his D and the
/// `136 / sqrt(n)` band are measured from (his footnote 6: plotting against
/// `e_i` is an approximation of that).
fn pp_points(percentiles: &[f64]) -> Vec<(f64, f64, f64)> {
    let n = percentiles.len() as f64;
    sorted(percentiles)
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let k = i as f64 + 1.0;
            (100.0 * k / (n + 1.0), 100.0 * k / n, p)
        })
        .collect()
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

/// A fixed, platform-independent seed per triangle, arm and basis
/// (FNV-1a), so a run is reproducible whatever its order or sharding.
fn seed_for(line: Line, group: &str, arm: &str, basis: Basis) -> u64 {
    let key = format!("{}|{group}|{arm}|{}", line.code(), basis.name());
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The seed with `RISK_RS_CALIBRATION_SEED_SALT` mixed in (FNV-1a over the
/// salt's bytes, continuing from the seed). No salt, or an empty one, gives
/// the seed unchanged, so a default run is the recorded run; a salt gives
/// every NUTS and bootstrap fit another stream, to measure how much each
/// arm's D moves with the seed alone (the note, "Pinning").
fn salted(seed: u64, salt: Option<&str>) -> u64 {
    match salt {
        Some(s) if !s.is_empty() => s.bytes().fold(seed, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        }),
        _ => seed,
    }
}

/// `RISK_RS_CALIBRATION_SEED_SALT`, if set and not empty.
fn seed_salt() -> Option<String> {
    std::env::var("RISK_RS_CALIBRATION_SEED_SALT")
        .ok()
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------
// Meyers' published D statistics (percentage points)
// ---------------------------------------------------------------------------

/// Meyers' D by line (ca, pa, wc, ol) and over all 200, read from the figure
/// text of the monograph: Mack incurred (Figure 3.2, p.10), Mack paid
/// (Figure 4.2, p.15), ODP bootstrap paid (Figure 4.1, p.14), CRC paid
/// (Figure 5.1, p.19) and incurred (Figure 5.2, p.20), CSR paid (Figure 7.4,
/// p.35), CAY incurred (Figure 8.5, p.42). `BayesOdp` has no counterpart.
fn meyers_published(arm: &str, basis: Basis, line: Option<Line>) -> Option<f64> {
    let row: [f64; 5] = match (arm, basis) {
        ("mack", Basis::Incurred) => [16.4, 16.6, 27.0, 14.2, 15.4],
        ("mack", Basis::Paid) => [24.6, 44.6, 30.4, 8.5, 23.1],
        ("odp", Basis::Paid) => [23.1, 44.9, 28.4, 6.8, 24.1],
        ("crc", Basis::Paid) => [22.8, 41.7, 30.5, 25.3, 25.5],
        ("crc", Basis::Incurred) => [10.2, 16.7, 14.4, 20.7, 11.9],
        ("csr", Basis::Paid) => [5.9, 18.5, 12.0, 10.4, 3.1],
        ("cay", Basis::Incurred) => [8.3, 15.9, 12.5, 20.8, 10.8],
        _ => return None,
    };
    Some(match line {
        Some(Line::Ca) => row[0],
        Some(Line::Pa) => row[1],
        Some(Line::Wc) => row[2],
        Some(Line::Ol) => row[3],
        None => row[4],
    })
}

/// How far Mack's D may be from Meyers' before the panel is rejected:
/// 8 points per line and 4 on all 200 (the note, decision 3). The
/// tolerance is empirical, not a bound. The rebuilt panel is known to miss
/// 26 of Table A.2's 175 printed codes (9 CA, 6 PA, 2 WC, 9 OL;
/// `panel_check.csv`), and 25 CA codes cannot be checked at all; a
/// triangle swapped can move a line's D by up to 100/50 = 2 points, so the
/// known differences alone could move it by more than 8. The first run was
/// within 3.7 points per line and 1.5 on all 200.
fn mack_tolerance(line: Option<Line>) -> f64 {
    if line.is_some() { 8.0 } else { 4.0 }
}

// ---------------------------------------------------------------------------
// The arms
// ---------------------------------------------------------------------------

/// One fit's result: the outcome's percentile and what the predictive
/// distribution of the total looked like.
#[derive(Clone, Debug)]
struct Scored {
    percentile: f64,
    mean: f64,
    sd: f64,
    outcome: f64,
    /// Each origin's percentile; NaN for the oldest, already at age 10.
    per_origin: [f64; ORIGINS],
    /// The NUTS diagnostics of the fit that gave this result (none for
    /// Mack and the ODP bootstrap).
    diag: Option<DiagSummary>,
    /// When the first fit failed its diagnostics and this result is the
    /// refit at a higher target acceptance: the first fit's percentile and
    /// diagnostics.
    first: Option<(f64, DiagSummary)>,
}

/// What the study keeps of a fit's [`Diagnostics`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct DiagSummary {
    max_rhat: f64,
    min_ess_bulk: f64,
    min_ess_tail: f64,
    divergences: usize,
    /// Posterior draws with the rates swapped (`k_er <= k_p`;
    /// Compartmental only, 0 for the other models). Recorded, not a refit
    /// trigger.
    rate_swaps: usize,
}

impl DiagSummary {
    fn new(d: &Diagnostics) -> Self {
        Self {
            max_rhat: d.max_rhat,
            min_ess_bulk: d.min_ess_bulk,
            min_ess_tail: d.min_ess_tail,
            divergences: d.divergences,
            rate_swaps: 0,
        }
    }

    /// R-hat above the limit, or NaN.
    fn rhat_fails(&self) -> bool {
        self.max_rhat.is_nan() || self.max_rhat > RHAT_LIMIT
    }

    /// The note's rule (decision 5): R-hat above 1.05 (or NaN), or any
    /// divergence.
    fn needs_refit(&self) -> bool {
        self.rhat_fails() || self.divergences > 0
    }

    fn text(&self) -> String {
        let swaps = if self.rate_swaps > 0 {
            format!(";swaps={}", self.rate_swaps)
        } else {
            String::new()
        };
        format!(
            "rhat={:.4};ess_bulk={:.0};ess_tail={:.0};div={}{swaps}",
            self.max_rhat, self.min_ess_bulk, self.min_ess_tail, self.divergences
        )
    }
}

/// Fits with `fit(target_accept)` and, when the first fit fails its
/// diagnostics, once more at [`REFIT_ACCEPT`]; the refit is the result and
/// carries the first fit's percentile and diagnostics. A refit that errors
/// leaves the first fit as the result.
fn with_refit(fit: impl Fn(f64) -> Result<Scored, String>) -> Result<Scored, String> {
    with_refit_at(FIRST_ACCEPT, REFIT_ACCEPT, fit)
}

/// [`with_refit`] with the two target acceptances given: Compartmental's
/// first fit is at the library's own (0.99 for model 1, 0.999 for model 2),
/// so its refit is a tenth of the remaining distance to 1 closer.
fn with_refit_at(
    first_accept: f64,
    refit_accept: f64,
    fit: impl Fn(f64) -> Result<Scored, String>,
) -> Result<Scored, String> {
    let first = fit(first_accept)?;
    let Some(diag) = first.diag.filter(DiagSummary::needs_refit) else {
        return Ok(first);
    };
    Ok(match fit(refit_accept) {
        Ok(mut second) => {
            second.first = Some((first.percentile, diag));
            second
        }
        Err(_) => first,
    })
}

/// Mack on `basis`: a log-normal matched to the total ultimate and its
/// standard error.
fn score_mack(t: &PanelTriangle, basis: Basis) -> Result<Scored, String> {
    let fit = Mack::default()
        .fit(&t.training(), basis.name())
        .map_err(|e| e.to_string())?;
    let cl = &fit.chain_ladder;
    if cl.ultimate.len() != ORIGINS {
        return Err(format!(
            "{} origins fitted, expected {ORIGINS}",
            cl.ultimate.len()
        ));
    }
    let outcome = t.outcome(basis);
    let (mean, sd) = (cl.total_ultimate(), fit.total_standard_error);
    let by_origin = t.outcome_by_origin(basis);
    let mut per_origin = [f64::NAN; ORIGINS];
    for o in 1..ORIGINS {
        if let Ok(p) = lognormal_percentile(cl.ultimate[o], fit.standard_error[o], by_origin[o]) {
            per_origin[o] = p;
        }
    }
    Ok(Scored {
        percentile: lognormal_percentile(mean, sd, outcome)?,
        mean,
        sd,
        outcome,
        per_origin,
        diag: None,
        first: None,
    })
}

/// The percentile and moments of the outcome under predictive draws of
/// `dist` (one component per origin) that are the quantity less `shift`
/// (an origin's latest value for reserves, zero for ultimates): the share
/// of simulated totals at or below the outcome less the total shift
/// (`Sampled::cdf`, Meyers' statistic).
fn score_draws(
    t: &PanelTriangle,
    basis: Basis,
    dist: &PredictiveDistribution,
    shift: &[f64],
) -> Result<Scored, String> {
    let (n_comp, n_sims) = (dist.n_components(), dist.n_sims());
    if n_comp != ORIGINS || shift.len() != ORIGINS {
        return Err(format!("{n_comp} components, expected {ORIGINS}"));
    }
    let shift_sum: f64 = shift.iter().sum();
    let outcome = t.outcome(basis);
    let total = dist.total();
    let by_origin = t.outcome_by_origin(basis);
    let draws = dist.draw_matrix();
    let mut per_origin = [f64::NAN; ORIGINS];
    for o in 1..ORIGINS {
        let x = by_origin[o] - shift[o];
        let below = (0..n_sims).filter(|&s| draws[s * n_comp + o] <= x).count();
        per_origin[o] = 100.0 * below as f64 / n_sims as f64;
    }
    Ok(Scored {
        percentile: 100.0 * total.cdf(outcome - shift_sum),
        mean: shift_sum + total.mean(),
        sd: total.variance().sqrt(),
        outcome,
        per_origin,
        diag: None,
        first: None,
    })
}

/// The ODP bootstrap on paid: the share of simulated totals at or below the
/// outcome less what is already paid.
fn score_odp(t: &PanelTriangle, seed: u64, n_sims: usize) -> Result<Scored, String> {
    let fit = OdpBootstrap {
        n_sims,
        seed,
        ..Default::default()
    }
    .fit(&t.training(), Basis::Paid.name())
    .map_err(|e| e.to_string())?;
    if fit.chain_ladder.latest.len() != ORIGINS {
        return Err(format!(
            "{} origins fitted, expected {ORIGINS}",
            fit.chain_ladder.latest.len()
        ));
    }
    score_draws(t, Basis::Paid, &fit.reserves, &fit.chain_ladder.latest)
}

/// `BayesOdp` at its defaults (priors, four chains of 1,000 warm-up and
/// 1,000 kept draws, 10,000 simulations) with this seed and target
/// acceptance.
fn bayes_odp_for(seed: u64, target_accept: f64) -> BayesOdp {
    let mut model = BayesOdp {
        seed,
        ..BayesOdp::default()
    };
    model.sampler.target_accept = target_accept;
    model
}

/// `BayesOdp` on paid, scored like the ODP bootstrap: the outcome less what
/// is paid among the simulated reserves.
fn score_bayes_odp(t: &PanelTriangle, model: &BayesOdp) -> Result<Scored, String> {
    let fit = model
        .fit(&t.training(), Basis::Paid.name())
        .map_err(|e| e.to_string())?;
    if fit.chain_ladder.latest.len() != ORIGINS {
        return Err(format!(
            "{} origins fitted, expected {ORIGINS}",
            fit.chain_ladder.latest.len()
        ));
    }
    let mut scored = score_draws(t, Basis::Paid, &fit.reserves, &fit.chain_ladder.latest)?;
    scored.diag = Some(DiagSummary::new(&fit.diagnostics));
    Ok(scored)
}

/// Meyers' model with his priors (the library's `Meyers` defaults: his
/// `N(-0.4, 3.162^2)` and `N(0, 3.162^2)` priors, `gamma` and `rho` as in
/// his Stan code; four chains of 1,000 warm-up and 2,000 kept draws;
/// 10,000 simulations) with this seed and target acceptance.
fn meyers_for(model: MeyersModel, seed: u64, target_accept: f64) -> Meyers {
    let mut m = Meyers {
        model,
        seed,
        ..Meyers::default()
    };
    m.sampler.target_accept = target_accept;
    m
}

/// A Meyers model on `basis`, with the premium from the panel: the
/// percentile of the outcome among the simulated ultimates (no shift).
fn score_meyers(t: &PanelTriangle, basis: Basis, model: &Meyers) -> Result<Scored, String> {
    let fit = model
        .fit(&t.training(), basis.name(), "premium")
        .map_err(|e| e.to_string())?;
    let mut scored = score_draws(t, basis, &fit.ultimates, &[0.0; ORIGINS])?;
    scored.diag = Some(DiagSummary::new(&fit.diagnostics));
    Ok(scored)
}

/// Gesmann and Morris' model 1 (or model 2) with the paper's preset priors
/// and the library's sampler, with this seed and target acceptance. The
/// presets were fitted to workers' compensation company 337, so on the
/// other lines the study partly tests the preset (the note, decision 1).
fn compartmental_for(model_2: bool, seed: u64, target_accept: f64) -> Compartmental {
    let mut m = if model_2 {
        Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
    } else {
        Compartmental::new(CompartmentalPriors::gesmann_morris())
    };
    m.seed = seed;
    m.sampler.target_accept = target_accept;
    m
}

/// The library's own target acceptance of the model (0.99 for model 1,
/// 0.999 for model 2), the first fit's.
fn compartmental_accept(model_2: bool) -> f64 {
    if model_2 {
        Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
    } else {
        Compartmental::new(CompartmentalPriors::gesmann_morris())
    }
    .sampler
    .target_accept
}

/// The refit's target acceptance: a tenth of the remaining distance to 1
/// closer than the first fit's (0.999 for model 1, 0.9999 for model 2).
fn compartmental_refit_accept(model_2: bool) -> f64 {
    1.0 - (1.0 - compartmental_accept(model_2)) / 10.0
}

/// Compartmental on paid, with outstanding from case incurred and the
/// panel's premium, scored like the ODP bootstrap: the outcome less what is
/// paid among the simulated reserves (the paid at the square's last age
/// less the latest paid).
fn score_compartmental(t: &PanelTriangle, model: &Compartmental) -> Result<Scored, String> {
    let fit = model
        .fit(
            &t.training(),
            Basis::Paid.name(),
            Outstanding::FromIncurred(Basis::Incurred.name()),
            "premium",
        )
        .map_err(|e| e.to_string())?;
    compartmental_scored(t, &fit)
}

fn compartmental_scored(t: &PanelTriangle, fit: &CompartmentalFit) -> Result<Scored, String> {
    let mut scored = score_draws(t, Basis::Paid, &fit.reserves, &fit.latest_paid)?;
    scored.diag = Some(DiagSummary {
        rate_swaps: fit.rate_swaps,
        ..DiagSummary::new(&fit.diagnostics)
    });
    Ok(scored)
}

/// An arm and basis the study runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ArmBasis {
    arm: &'static str,
    basis: Basis,
}

impl ArmBasis {
    fn label(self) -> String {
        format!("{}_{}", self.arm, self.basis.name())
    }

    fn seed(self, t: &PanelTriangle) -> u64 {
        salted(
            seed_for(t.line, &t.group, self.arm, self.basis),
            seed_salt().as_deref(),
        )
    }

    fn run(self, t: &PanelTriangle) -> Result<Scored, String> {
        match self.arm {
            "mack" => score_mack(t, self.basis),
            "odp" => score_odp(t, self.seed(t), ODP_SIMS),
            "bayes_odp" => with_refit(|ta| score_bayes_odp(t, &bayes_odp_for(self.seed(t), ta))),
            "crc" | "csr" | "cay" => {
                let kind = match self.arm {
                    "crc" => MeyersModel::Crc,
                    "csr" => MeyersModel::Csr,
                    _ => MeyersModel::Cay,
                };
                with_refit(|ta| score_meyers(t, self.basis, &meyers_for(kind, self.seed(t), ta)))
            }
            "compartmental" | "compartmental_2" => {
                let m2 = self.arm == "compartmental_2";
                with_refit_at(
                    compartmental_accept(m2),
                    compartmental_refit_accept(m2),
                    |ta| score_compartmental(t, &compartmental_for(m2, self.seed(t), ta)),
                )
            }
            other => Err(format!("unknown arm {other}")),
        }
    }
}

/// Every arm the study has.
const ALL_ARMS: &str = "mack,odp,bayes_odp,crc,csr,cay,compartmental,compartmental_2";

/// The default of `RISK_RS_CALIBRATION_ARMS`: all but Compartmental model 2
/// (about 59 s a fit on the panel against model 1's 2.1 s), which runs only
/// when named.
const DEFAULT_ARMS: &str = "mack,odp,bayes_odp,crc,csr,cay,compartmental";

/// The arms named by `RISK_RS_CALIBRATION_ARMS` (default [`DEFAULT_ARMS`]). An
/// arm runs on Meyers' bases: Mack and CRC on paid and incurred, the ODP
/// bootstrap, `BayesOdp`, CSR and Compartmental (both models) on paid, CAY
/// on incurred.
fn selected_arms(spec: Option<&str>) -> Vec<ArmBasis> {
    let spec = spec.unwrap_or(DEFAULT_ARMS);
    let mut out = Vec::new();
    for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let bases: &[Basis] = match name {
            "mack" | "crc" => &[Basis::Paid, Basis::Incurred],
            "odp" | "bayes_odp" | "csr" | "compartmental" | "compartmental_2" => &[Basis::Paid],
            "cay" => &[Basis::Incurred],
            other => panic!(
                "RISK_RS_CALIBRATION_ARMS: unknown arm {other:?} (known: {})",
                ALL_ARMS.replace(',', ", ")
            ),
        };
        let arm = ALL_ARMS
            .split(',')
            .find(|a| *a == name)
            .expect("a known arm");
        for &basis in bases {
            out.push(ArmBasis { arm, basis });
        }
    }
    assert!(!out.is_empty(), "RISK_RS_CALIBRATION_ARMS names no arm");
    out
}

/// The lines named by `RISK_RS_CALIBRATION_LINES` (default all four).
fn selected_lines(spec: Option<&str>) -> Vec<Line> {
    let Some(spec) = spec.filter(|s| !s.trim().is_empty()) else {
        return Line::ALL.to_vec();
    };
    spec.split(',')
        .map(|s| {
            Line::parse(s.trim())
                .unwrap_or_else(|| panic!("RISK_RS_CALIBRATION_LINES: unknown line {s:?}"))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pins: the recorded D of each arm, asserted by the study (decision 7)
// ---------------------------------------------------------------------------

/// The sha256 of the panel the pins were recorded on (`panel_meta.csv`,
/// `panel_sha256`). A rebuild with another hash is another panel, and the
/// pins say nothing about it.
const PINNED_PANEL_SHA256: &str =
    "075b392525b26c505c251ea1d94f1641b4c6ab4216ee085655e5ec678fff59d2";

/// One pinned cell: the triangles an arm and basis scored in a line (or on
/// all 200, `scope` `None`) and the Meyers D they gave, in percentage
/// points, in the reviewed run.
#[derive(Clone, Copy, Debug)]
struct Pin {
    arm: &'static str,
    basis: Basis,
    scope: Option<Line>,
    n: usize,
    d: f64,
}

/// What a run measured for one cell (the arm's own set of triangles).
#[derive(Clone, Copy, Debug)]
struct Observed {
    arm: &'static str,
    basis: Basis,
    scope: Option<Line>,
    n: usize,
    d: f64,
}

/// How far an arm's D may move from its pin, in percentage points, before
/// the study fails (`scope` `None` is all 200). The numbers are in
/// [`TOLERANCES`] and the note, "Pinning". An arm with no entry has none:
/// NaN is within no tolerance, so its cell fails loudly.
fn pin_tolerance(arm: &str, scope: Option<Line>) -> f64 {
    TOLERANCES
        .iter()
        .find(|(a, _, _)| *a == arm)
        .map_or(
            f64::NAN,
            |(_, per_line, all)| {
                if scope.is_some() { *per_line } else { *all }
            },
        )
}

/// Whether the study asserts its pins: only on the default seed and with
/// every triangle of each line, since a salted seed or a limit is another
/// experiment (the pins are for the recorded one).
fn pins_active(limit: Option<usize>, salt: Option<&str>) -> bool {
    limit.is_none() && salt.is_none_or(str::is_empty)
}

/// The panel hash the study ran on must be the pinned one. `meta` is
/// `panel_meta.csv` as `panel_meta` reads it.
fn check_panel_hash(meta: &[(String, String)]) -> Result<(), String> {
    let found = meta
        .iter()
        .find(|(k, _)| k == "panel_sha256")
        .map(|(_, v)| v.as_str());
    if found == Some(PINNED_PANEL_SHA256) {
        return Ok(());
    }
    Err(format!(
        "the panel's sha256 is {}, not the pinned {PINNED_PANEL_SHA256}: the D pins were recorded on another panel (cas-schedule-p 2026.6.13) and say nothing about this one. Rebuild the panel with\n    {PANEL_COMMAND}\nor, after a reviewed change of panel, re-pin (the note, \"Pinning\"). Set RISK_RS_CALIBRATION_LIMIT or RISK_RS_CALIBRATION_SEED_SALT to run without pins",
        found.unwrap_or("absent from panel_meta.csv")
    ))
}

/// The pins the run breaks. Only the cells the run covers are checked: an
/// arm or line that was not run is skipped, and so is `all` unless
/// `all_lines` (a run on fewer lines has another all-200 set). A cell fails
/// when it scored another number of triangles (its D is then another set's)
/// or when D is further from the pin than the tolerance. The verdict (D
/// above the 5% bound, `136 / sqrt(n)`) is held by the same test: a D within
/// the tolerance of a pin that is further than the tolerance from the bound
/// is on the pin's side of the bound, so only a pin within the tolerance of
/// the bound ([`near_the_bound`]) may be passed on either side of it. A
/// failure names a verdict that flipped.
fn check_pins(
    pins: &[Pin],
    observed: &[Observed],
    all_lines: bool,
    tolerance: impl Fn(&str, Option<Line>) -> f64,
) -> Vec<String> {
    let mut out = Vec::new();
    for p in pins {
        if p.scope.is_none() && !all_lines {
            continue;
        }
        let Some(o) = observed
            .iter()
            .find(|o| o.arm == p.arm && o.basis == p.basis && o.scope == p.scope)
        else {
            continue;
        };
        let cell = format!(
            "{} {} {}",
            p.arm,
            p.basis.name(),
            p.scope.map_or("all", Line::code)
        );
        if o.n != p.n {
            out.push(format!(
                "{cell}: scored {} triangles, pinned on {}",
                o.n, p.n
            ));
            continue;
        }
        let tol = tolerance(p.arm, p.scope);
        // A NaN D or tolerance is within no tolerance.
        let moved = (o.d - p.d).abs();
        if moved.is_nan() || tol.is_nan() || moved > tol {
            let bound = ks_bound(p.n);
            let flipped = if (o.d > bound) != (p.d > bound) {
                format!("; the verdict flipped (bound {bound:.2})")
            } else {
                String::new()
            };
            out.push(format!(
                "{cell}: D {:.2} is {moved:.2} from the pinned {:.2} (tolerance {tol}){flipped}",
                o.d, p.d
            ));
        }
    }
    out
}

/// Whether the pin is within `tol` of the 5% bound of its n, so that a run
/// within the tolerance of it may pass or fail there: its verdict is not
/// pinned. Every other pin's verdict is.
fn near_the_bound(p: &Pin, tol: f64) -> bool {
    (p.d - ks_bound(p.n)).abs() <= tol
}

/// The own-set cells of a run, to check against the pins or to re-pin from.
fn observed_cells(groups: &[Group]) -> Vec<Observed> {
    groups
        .iter()
        .filter(|g| g.set == "own")
        .map(|g| Observed {
            arm: g.arm.arm,
            basis: g.arm.basis,
            scope: g.line,
            n: g.percentiles.len(),
            d: if g.percentiles.is_empty() {
                f64::NAN
            } else {
                meyers_d(&g.percentiles)
            },
        })
        .collect()
}

/// A run's cells as source lines for [`PINS`], to paste after a reviewed
/// re-pin (the study prints them).
fn pins_source(observed: &[Observed]) -> String {
    let mut out = String::new();
    for o in observed.iter().filter(|o| o.n > 0) {
        let scope = o
            .scope
            .map_or("None".to_string(), |l| format!("Some(Line::{l:?})"));
        let _ = writeln!(
            out,
            "    pin(\"{}\", Basis::{:?}, {scope}, {}, {:.2}),",
            o.arm, o.basis, o.n, o.d
        );
    }
    out
}

const fn pin(arm: &'static str, basis: Basis, scope: Option<Line>, n: usize, d: f64) -> Pin {
    Pin {
        arm,
        basis,
        scope,
        n,
        d,
    }
}

/// Tolerances by arm: (arm, per line, all 200), in percentage points. Each
/// is twice the largest difference in D between two of three seeds (the
/// default and two salts, measured 2026-10-21 over the arm's bases, in a
/// line and on all 200), rounded up to 0.1, or a floor if that is larger:
/// 1.0 per line and 0.5 on all 200 for a NUTS arm, 0.1 for the
/// deterministic Mack (platform arithmetic only), 0.5 on all 200 for the
/// seeded ODP bootstrap. Compartmental model 2 was salted once, on
/// 6, 10, 27 and 5 triangles, so it takes 2.0 and 1.0. The numbers, the
/// measured differences and what the tolerance does not cover are in the
/// note, "Pinning".
const TOLERANCES: &[(&str, f64, f64)] = &[
    ("mack", 0.1, 0.1),
    ("odp", 1.7, 0.5),
    ("bayes_odp", 2.1, 0.5),
    ("crc", 3.5, 1.7),
    ("csr", 3.0, 2.5),
    ("cay", 2.5, 1.3),
    ("compartmental", 2.7, 0.9),
    ("compartmental_2", 2.0, 1.0),
];

/// The reviewed run's D (2026-10-21, Windows, the default seed, the panel of
/// `PINNED_PANEL_SHA256`; approved by the user on 2026-10-21), with the
/// number of triangles each arm scored in a line or on all 200.
const PINS: &[Pin] = &[
    pin("mack", Basis::Incurred, Some(Line::Ca), 50, 17.81),
    pin("mack", Basis::Incurred, Some(Line::Pa), 50, 14.69),
    pin("mack", Basis::Incurred, Some(Line::Wc), 50, 24.66),
    pin("mack", Basis::Incurred, Some(Line::Ol), 50, 16.20),
    pin("mack", Basis::Incurred, None, 200, 16.46),
    pin("mack", Basis::Paid, Some(Line::Ca), 50, 23.19),
    pin("mack", Basis::Paid, Some(Line::Pa), 50, 48.31),
    pin("mack", Basis::Paid, Some(Line::Wc), 50, 33.36),
    pin("mack", Basis::Paid, Some(Line::Ol), 49, 10.33),
    pin("mack", Basis::Paid, None, 199, 24.64),
    pin("odp", Basis::Paid, Some(Line::Ca), 50, 25.58),
    pin("odp", Basis::Paid, Some(Line::Pa), 50, 48.83),
    pin("odp", Basis::Paid, Some(Line::Wc), 50, 31.33),
    pin("odp", Basis::Paid, Some(Line::Ol), 50, 8.74),
    pin("odp", Basis::Paid, None, 200, 26.28),
    pin("bayes_odp", Basis::Paid, Some(Line::Ca), 14, 53.83),
    pin("bayes_odp", Basis::Paid, Some(Line::Pa), 19, 57.83),
    pin("bayes_odp", Basis::Paid, Some(Line::Wc), 40, 36.53),
    pin("bayes_odp", Basis::Paid, Some(Line::Ol), 14, 27.15),
    pin("bayes_odp", Basis::Paid, None, 87, 38.49),
    pin("crc", Basis::Incurred, Some(Line::Ca), 50, 8.61),
    pin("crc", Basis::Incurred, Some(Line::Pa), 50, 11.82),
    pin("crc", Basis::Incurred, Some(Line::Wc), 50, 17.32),
    pin("crc", Basis::Incurred, Some(Line::Ol), 50, 10.45),
    pin("crc", Basis::Incurred, None, 200, 10.29),
    pin("crc", Basis::Paid, Some(Line::Ca), 48, 20.74),
    pin("crc", Basis::Paid, Some(Line::Pa), 50, 42.97),
    pin("crc", Basis::Paid, Some(Line::Wc), 50, 32.80),
    pin("crc", Basis::Paid, Some(Line::Ol), 45, 21.57),
    pin("crc", Basis::Paid, None, 193, 26.40),
    pin("csr", Basis::Paid, Some(Line::Ca), 48, 5.53),
    pin("csr", Basis::Paid, Some(Line::Pa), 50, 25.69),
    pin("csr", Basis::Paid, Some(Line::Wc), 50, 10.92),
    pin("csr", Basis::Paid, Some(Line::Ol), 45, 12.75),
    pin("csr", Basis::Paid, None, 193, 4.62),
    pin("cay", Basis::Incurred, Some(Line::Ca), 50, 12.20),
    pin("cay", Basis::Incurred, Some(Line::Pa), 50, 10.14),
    pin("cay", Basis::Incurred, Some(Line::Wc), 50, 14.99),
    pin("cay", Basis::Incurred, Some(Line::Ol), 50, 10.65),
    pin("cay", Basis::Incurred, None, 200, 6.96),
    pin("compartmental", Basis::Paid, Some(Line::Ca), 50, 49.27),
    pin("compartmental", Basis::Paid, Some(Line::Pa), 50, 36.86),
    pin("compartmental", Basis::Paid, Some(Line::Wc), 50, 26.78),
    pin("compartmental", Basis::Paid, Some(Line::Ol), 50, 66.28),
    pin("compartmental", Basis::Paid, None, 200, 39.28),
    pin("compartmental_2", Basis::Paid, Some(Line::Ca), 6, 27.32),
    pin("compartmental_2", Basis::Paid, Some(Line::Pa), 10, 38.24),
    pin("compartmental_2", Basis::Paid, Some(Line::Wc), 27, 29.36),
    pin("compartmental_2", Basis::Paid, Some(Line::Ol), 5, 19.80),
    pin("compartmental_2", Basis::Paid, None, 48, 20.52),
];

// ---------------------------------------------------------------------------
// The study
// ---------------------------------------------------------------------------

struct Record<'a> {
    triangle: &'a PanelTriangle,
    arm: ArmBasis,
    seed: u64,
    wall_ms: f64,
    result: Result<Scored, String>,
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn num(x: f64) -> String {
    if x.is_nan() {
        String::new()
    } else {
        format!("{x}")
    }
}

fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Fits every triangle with every arm, in parallel. A panic in a fit is
/// recorded as an error, not propagated.
fn run_arms<'a>(triangles: &'a [PanelTriangle], arms: &[ArmBasis]) -> Vec<Record<'a>> {
    let jobs: Vec<(&PanelTriangle, ArmBasis)> = triangles
        .iter()
        .flat_map(|t| arms.iter().map(move |&a| (t, a)))
        .collect();
    let started = Instant::now();
    let done = AtomicUsize::new(0);
    let step = (jobs.len() / 40).max(1);
    jobs.par_iter()
        .map(|&(triangle, arm)| {
            let start = Instant::now();
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| arm.run(triangle)))
                    .unwrap_or_else(|_| Err("the fit panicked".to_string()));
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % step == 0 || n == jobs.len() {
                eprintln!(
                    "progress: {n} of {} fits, {:.0} s",
                    jobs.len(),
                    started.elapsed().as_secs_f64()
                );
            }
            Record {
                triangle,
                arm,
                seed: arm.seed(triangle),
                wall_ms: start.elapsed().as_secs_f64() * 1e3,
                result,
            }
        })
        .collect()
}

fn percentiles_csv(records: &[Record<'_>]) -> String {
    let mut out = String::from(
        "line,group,arm,basis,percentile,mean,sd,outcome,seed,platform,wall_ms,diagnostics,refit,error",
    );
    for o in 0..ORIGINS {
        let _ = write!(out, ",p_{}", FIRST_ORIGIN + o as i32);
    }
    out.push('\n');
    for r in records {
        let t = r.triangle;
        let _ = write!(
            out,
            "{},{},{},{},",
            t.line.code(),
            t.group,
            r.arm.arm,
            r.arm.basis.name()
        );
        let (scored, error) = match &r.result {
            Ok(s) => (Some(s), String::new()),
            Err(e) => (None, e.clone()),
        };
        let _ = write!(
            out,
            "{},{},{},{},{},{},{:.1},{},{},{}",
            scored.map_or(String::new(), |s| num(s.percentile)),
            scored.map_or(String::new(), |s| num(s.mean)),
            scored.map_or(String::new(), |s| num(s.sd)),
            scored.map_or(String::new(), |s| num(s.outcome)),
            r.seed,
            platform(),
            r.wall_ms,
            scored
                .and_then(|s| s.diag)
                .map_or(String::new(), |d| d.text()),
            scored
                .and_then(|s| s.first)
                .map_or(String::new(), |(p, d)| format!(
                    "first_fit_percentile={p:.4};{}",
                    d.text()
                )),
            csv_field(&error)
        );
        for o in 0..ORIGINS {
            let _ = write!(
                out,
                ",{}",
                scored.map_or(String::new(), |s| num(s.per_origin[o]))
            );
        }
        out.push('\n');
    }
    out
}

/// A set of percentiles under one heading of `ks.csv` and `pp.csv`.
struct Group {
    /// ca, pa, wc, ol or all.
    label: String,
    line: Option<Line>,
    arm: ArmBasis,
    /// own: every triangle this arm scored; common: those every arm scored.
    set: &'static str,
    percentiles: Vec<f64>,
    skipped: usize,
}

fn groups(records: &[Record<'_>], arms: &[ArmBasis], lines: &[Line]) -> Vec<Group> {
    // Triangles (line, group) that every arm scored.
    let mut ok_arms: BTreeMap<(Line, &str), usize> = BTreeMap::new();
    for r in records.iter().filter(|r| r.result.is_ok()) {
        *ok_arms
            .entry((r.triangle.line, &r.triangle.group))
            .or_default() += 1;
    }
    let common = |r: &Record<'_>| {
        ok_arms.get(&(r.triangle.line, r.triangle.group.as_str())) == Some(&arms.len())
    };
    let mut out = Vec::new();
    for &arm in arms {
        for set in ["own", "common"] {
            for scope in lines.iter().map(|&l| Some(l)).chain([None]) {
                let mine = records
                    .iter()
                    .filter(|r| r.arm == arm && scope.is_none_or(|l| r.triangle.line == l));
                let (mut percentiles, mut skipped) = (Vec::new(), 0);
                for r in mine {
                    match &r.result {
                        Ok(s) if set == "own" || common(r) => percentiles.push(s.percentile),
                        _ => skipped += 1,
                    }
                }
                out.push(Group {
                    label: scope.map_or("all", Line::code).to_string(),
                    line: scope,
                    arm,
                    set,
                    percentiles,
                    skipped,
                });
            }
        }
    }
    out
}

/// The diff of the panel against Table A.2 written by the script:
/// `line -> [both, chosen_only, printed_only, unchecked]` counts, and
/// `all`. `unchecked` are the commercial auto codes chosen but not printed:
/// Table A.2 prints only 25 of that line's 50, so they cannot be checked.
fn panel_check(dir: &Path) -> BTreeMap<String, [usize; 4]> {
    let Ok(text) = std::fs::read_to_string(dir.join("panel_check.csv")) else {
        return BTreeMap::new();
    };
    panel_check_counts(&text)
}

fn panel_check_counts(text: &str) -> BTreeMap<String, [usize; 4]> {
    let mut out: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    for row in text.lines().skip(1) {
        let f: Vec<&str> = row.split(',').collect();
        if f.len() < 3 {
            continue;
        }
        let slot = match f[2].trim() {
            "both" => 0,
            "chosen_only" => 1,
            "printed_only" => 2,
            "unchecked" => 3,
            _ => continue,
        };
        for key in [f[0], "all"] {
            out.entry(key.to_string()).or_default()[slot] += 1;
        }
    }
    out
}

/// The script's sidecar `panel_meta.csv` (`key,value` rows: the package
/// versions and the panel's sha256), printed in the study's header.
fn panel_meta(dir: &Path) -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(dir.join("panel_meta.csv")) else {
        return Vec::new();
    };
    text.lines()
        .skip(1)
        .filter_map(|l| l.split_once(','))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

/// One line on an arm's NUTS diagnostics, or none for an arm without them:
/// the fits that failed their first diagnostics (R-hat above 1.05 or a
/// divergence) and were refitted at target acceptance 0.99, those still
/// failing, and the worst and total values of the kept fits.
fn diagnostics_line(records: &[Record<'_>], arm: ArmBasis) -> Option<String> {
    let ds: Vec<(DiagSummary, Option<DiagSummary>)> = records
        .iter()
        .filter(|r| r.arm == arm)
        .filter_map(|r| r.result.as_ref().ok())
        .filter_map(|s| s.diag.map(|d| (d, s.first.map(|f| f.1))))
        .collect();
    if ds.is_empty() {
        return None;
    }
    let first_failed = ds
        .iter()
        .filter(|(d, first)| first.unwrap_or(*d).needs_refit())
        .count();
    let first_rhat = ds
        .iter()
        .filter(|(d, first)| {
            let f = first.unwrap_or(*d);
            f.rhat_fails()
        })
        .count();
    let first_div = ds
        .iter()
        .filter(|(d, first)| first.unwrap_or(*d).divergences > 0)
        .count();
    let refitted = ds.iter().filter(|(_, f)| f.is_some()).count();
    let still = ds.iter().filter(|(d, _)| d.needs_refit()).count();
    let max_rhat = ds.iter().map(|(d, _)| d.max_rhat).fold(0.0, f64::max);
    let total_div: usize = ds.iter().map(|(d, _)| d.divergences).sum();
    let min_bulk = ds
        .iter()
        .map(|(d, _)| d.min_ess_bulk)
        .fold(f64::INFINITY, f64::min);
    let min_tail = ds
        .iter()
        .map(|(d, _)| d.min_ess_tail)
        .fold(f64::INFINITY, f64::min);
    Some(format!(
        "{}: {} NUTS fits; first fit failed {first_failed} (R-hat above {RHAT_LIMIT} in {first_rhat}, a divergence in {first_div}), {refitted} refitted at a higher target acceptance ({REFIT_ACCEPT}; Compartmental a tenth of the way to 1), {still} still failing; kept fits: max R-hat {max_rhat:.4}, {total_div} divergences, min ESS bulk {min_bulk:.0} tail {min_tail:.0}",
        arm.label(),
        ds.len()
    ))
}

/// An arm's distinct error texts with their counts, most frequent first.
fn error_tally(records: &[Record<'_>], arm: ArmBasis) -> Vec<(String, usize)> {
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for r in records.iter().filter(|r| r.arm == arm) {
        if let Err(e) = &r.result {
            *tally.entry(e.chars().take(120).collect()).or_default() += 1;
        }
    }
    let mut out: Vec<_> = tally.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

fn ks_csv(groups: &[Group], check: &BTreeMap<String, [usize; 4]>) -> String {
    let mut out = String::from(
        "line,arm,basis,set,n,skipped,meyers_d,ks_d,bound,pass_d,pass_ks,meyers_published_d,panel_vs_table_a2\n",
    );
    for g in groups {
        let n = g.percentiles.len();
        let (d, ks, bound, pd, pk) = if n == 0 {
            (f64::NAN, f64::NAN, f64::NAN, String::new(), String::new())
        } else {
            let (d, ks, b) = (
                meyers_d(&g.percentiles),
                ks_two_sided(&g.percentiles),
                ks_bound(n),
            );
            (d, ks, b, (d <= b).to_string(), (ks <= b).to_string())
        };
        let published = if g.set == "own" {
            meyers_published(g.arm.arm, g.arm.basis, g.line).map_or(String::new(), num)
        } else {
            String::new()
        };
        let a2 = check
            .get(&g.label)
            .map_or(String::new(), |[b, c, p, u]| format!("{b}/{c}/{p}/{u}"));
        let _ = writeln!(
            out,
            "{},{},{},{},{n},{},{},{},{},{pd},{pk},{published},{a2}",
            g.label,
            g.arm.arm,
            g.arm.basis.name(),
            g.set,
            g.skipped,
            num(d),
            num(ks),
            num(bound),
        );
    }
    out
}

fn pp_csv(groups: &[Group]) -> String {
    let mut out = String::from("line,arm,basis,set,i,e_i,f_i,percentile\n");
    for g in groups {
        for (i, (e, f, p)) in pp_points(&g.percentiles).into_iter().enumerate() {
            let _ = writeln!(
                out,
                "{},{},{},{},{},{e},{f},{p}",
                g.label,
                g.arm.arm,
                g.arm.basis.name(),
                g.set,
                i + 1
            );
        }
    }
    out
}

/// The KS table as the test log prints it.
fn ks_table(groups: &[Group]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<5} {:<6} {:<9} {:<7} {:>4} {:>4} {:>8} {:>8} {:>7} {:>10}",
        "line", "arm", "basis", "set", "n", "skip", "D", "KS", "bound", "Meyers D"
    );
    for g in groups {
        let n = g.percentiles.len();
        if n == 0 {
            continue;
        }
        let d = meyers_d(&g.percentiles);
        let published = if g.set == "own" {
            meyers_published(g.arm.arm, g.arm.basis, g.line)
        } else {
            None
        };
        let _ = writeln!(
            out,
            "{:<5} {:<6} {:<9} {:<7} {:>4} {:>4} {:>7.1}{} {:>8.1} {:>7.1} {:>10}",
            g.label,
            g.arm.arm,
            g.arm.basis.name(),
            g.set,
            n,
            g.skipped,
            d,
            if d > ks_bound(n) { "*" } else { " " },
            ks_two_sided(&g.percentiles),
            ks_bound(n),
            published.map_or(String::new(), |p| format!("{p:.1}")),
        );
    }
    out
}

fn limit_per_line(triangles: Vec<PanelTriangle>, limit: Option<usize>) -> Vec<PanelTriangle> {
    let Some(limit) = limit else { return triangles };
    let mut taken: BTreeMap<Line, usize> = BTreeMap::new();
    triangles
        .into_iter()
        .filter(|t| {
            let n = taken.entry(t.line).or_default();
            *n += 1;
            *n <= limit
        })
        .collect()
}

/// The Meyers study on the rebuilt panel. Ignored: run by hand (see the
/// module header).
#[test]
#[ignore = "the 200-triangle study runs by hand, after validation/scripts/calibration_panel.py"]
fn calibration_study() {
    let started = Instant::now();
    let path = panel_path();
    let all = load_panel_at(&path).unwrap_or_else(|e| panic!("{e}"));
    let dir = path
        .parent()
        .expect("the panel path has a directory")
        .to_path_buf();

    // The panel's shape and group 353 (decision 7): 200 triangles, 50 per
    // line, each a full square (parse_panel checks the 100 cells).
    assert_eq!(
        all.len(),
        200,
        "the panel holds {} triangles, expected 200",
        all.len()
    );
    for line in Line::ALL {
        let n = all.iter().filter(|t| t.line == line).count();
        assert_eq!(n, 50, "{} has {n} triangles, expected 50", line.code());
    }
    // Group 353 commercial auto (Meyers' illustrative insurer; personal
    // auto and workers' compensation have a group 353 too, other insurers)
    // must be in the panel and equal the committed square cell by cell.
    let ca353 = all
        .iter()
        .find(|t| t.line == Line::Ca && t.group == "353")
        .expect("group 353 commercial auto is not in the panel");
    let committed = group_353();
    for o in 0..ORIGINS {
        for d in 0..ORIGINS {
            let (a, b) = (&ca353.cells[o][d], &committed.cells[o][d]);
            assert!(
                a.paid == b.paid && a.incurred == b.incurred && a.premium == b.premium,
                "group 353 commercial auto differs from validation/data/clrd_comauto_353.csv at origin {} age {}: {a:?} vs {b:?}",
                FIRST_ORIGIN + o as i32,
                12 * (d + 1)
            );
        }
    }
    println!("group 353 commercial auto equals validation/data/clrd_comauto_353.csv");
    for (k, v) in panel_meta(&dir) {
        println!("panel_meta {k}: {v}");
    }

    let lines = selected_lines(std::env::var("RISK_RS_CALIBRATION_LINES").ok().as_deref());
    let arms = selected_arms(std::env::var("RISK_RS_CALIBRATION_ARMS").ok().as_deref());
    let limit = std::env::var("RISK_RS_CALIBRATION_LIMIT")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok());
    let triangles = limit_per_line(
        all.into_iter()
            .filter(|t| lines.contains(&t.line))
            .collect(),
        limit,
    );
    println!(
        "panel {}: {} triangles, lines {:?}, arms {:?}, limit {:?}",
        path.display(),
        triangles.len(),
        lines.iter().map(|l| l.code()).collect::<Vec<_>>(),
        arms.iter().map(|a| a.label()).collect::<Vec<_>>(),
        limit
    );

    let records = run_arms(&triangles, &arms);
    let check = panel_check(&dir);
    let gs = groups(&records, &arms, &lines);
    std::fs::write(dir.join("percentiles.csv"), percentiles_csv(&records))
        .expect("percentiles.csv");
    std::fs::write(dir.join("pp.csv"), pp_csv(&gs)).expect("pp.csv");
    std::fs::write(dir.join("ks.csv"), ks_csv(&gs, &check)).expect("ks.csv");

    println!("\n{}", ks_table(&gs));
    for (key, [both, chosen_only, printed_only, unchecked]) in &check {
        println!(
            "panel vs Table A.2, {key}: {both} in both, {chosen_only} chosen only, {printed_only} printed only, {unchecked} chosen and not checkable (CA)"
        );
    }
    for a in &arms {
        let times: Vec<f64> = records
            .iter()
            .filter(|r| r.arm == *a)
            .map(|r| r.wall_ms)
            .collect();
        let errors = records
            .iter()
            .filter(|r| r.arm == *a && r.result.is_err())
            .count();
        println!(
            "{}: {} fits, {} errors, mean wall {:.1} ms (parallel runs inflate it), max {:.1} ms",
            a.label(),
            times.len(),
            errors,
            times.iter().sum::<f64>() / times.len().max(1) as f64,
            times.iter().copied().fold(0.0, f64::max)
        );
        for (text, n) in error_tally(&records, *a) {
            println!("    {n} x {text}");
        }
        if let Some(line) = diagnostics_line(&records, *a) {
            println!("    {line}");
        }
    }
    println!(
        "total {:.1} s; outputs in {}",
        started.elapsed().as_secs_f64(),
        dir.display()
    );

    // Mack's D near Meyers' (decision 3), only on the whole panel.
    if limit.is_none() && lines.len() == Line::ALL.len() {
        let mut off = Vec::new();
        for g in gs.iter().filter(|g| g.set == "own" && g.arm.arm == "mack") {
            if g.percentiles.is_empty() {
                continue;
            }
            let want = meyers_published("mack", g.arm.basis, g.line).expect("Mack is published");
            let got = meyers_d(&g.percentiles);
            if (got - want).abs() > mack_tolerance(g.line) {
                off.push(format!(
                    "Mack {} {}: D {got:.1}, Meyers {want:.1}, tolerance {}",
                    g.arm.basis.name(),
                    g.label,
                    mack_tolerance(g.line)
                ));
            }
        }
        assert!(
            off.is_empty(),
            "Mack's D is not near Meyers' on the rebuilt panel. The tolerance is empirical: the panel is known to differ from Table A.2 on at least 26 printed codes (panel_check.csv), so a miss can come from the panel, not a bug:\n{}",
            off.join("\n")
        );
    }

    // The pins (decision 7): the reviewed run's D per arm, basis and line,
    // on the default seed, the whole panel and the pinned panel.
    let salt = seed_salt();
    if pins_active(limit, salt.as_deref()) {
        let observed = observed_cells(&gs);
        println!(
            "\nthis run's cells as PINS source:\n{}",
            pins_source(&observed)
        );
        if let Err(e) = check_panel_hash(&panel_meta(&dir)) {
            panic!("{e}");
        }
        let all_lines = lines.len() == Line::ALL.len();
        let broken = check_pins(PINS, &observed, all_lines, pin_tolerance);
        assert!(
            broken.is_empty(),
            "{} pinned D value(s) moved. If the change is understood and the user approves it, re-pin from the source printed above (the note, \"Pinning\"):\n{}",
            broken.len(),
            broken.join("\n")
        );
        let checked = PINS
            .iter()
            .filter(|p| {
                arms.iter().any(|a| a.arm == p.arm && a.basis == p.basis)
                    && p.scope.map_or(all_lines, |l| lines.contains(&l))
            })
            .count();
        println!("pins: {checked} cells checked, all within tolerance");
        let near: Vec<String> = PINS
            .iter()
            .filter(|p| near_the_bound(p, pin_tolerance(p.arm, p.scope)))
            .map(|p| {
                format!(
                    "{} {} {}",
                    p.arm,
                    p.basis.name(),
                    p.scope.map_or("all", Line::code)
                )
            })
            .collect();
        println!(
            "pins within tolerance of the 5% bound, whose verdict is not pinned: {}",
            if near.is_empty() {
                "none".to_string()
            } else {
                near.join("; ")
            }
        );
    } else {
        println!(
            "pins not asserted (RISK_RS_CALIBRATION_LIMIT or RISK_RS_CALIBRATION_SEED_SALT is set)"
        );
    }
}

// ---------------------------------------------------------------------------
// Unit tests of the pure pieces (not ignored: they run in `cargo test`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::Sampled;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
    }

    /// A synthetic square: cumulative values with a development pattern and
    /// a wiggle so that every sigma is estimable.
    fn synthetic_csv(line: &str, group: &str) -> String {
        let mut s = String::from(PANEL_HEADER);
        s.push('\n');
        let cdf = [0.30, 0.55, 0.72, 0.84, 0.91, 0.95, 0.975, 0.99, 0.997, 1.0];
        for o in 0..ORIGINS {
            for (d, &cd) in cdf.iter().enumerate() {
                let wiggle = 1.0 + 0.04 * ((o * 7 + d * 3) as f64).sin();
                let paid = 1000.0 * (1.0 + 0.1 * o as f64) * cd * wiggle;
                let case = paid * 1.1;
                let _ = writeln!(
                    s,
                    "{line},{group},{},{},{paid},{},{case},{},{}",
                    FIRST_ORIGIN as usize + o,
                    12 * (d + 1),
                    case + 5.0,
                    1500.0 + o as f64,
                    2000.0
                );
            }
        }
        s
    }

    #[test]
    fn meyers_d_and_two_sided_ks_by_hand() {
        // Uniform midpoints: both are half a step, 100 / (2 n).
        let p: Vec<f64> = (0..4).map(|i| 12.5 + 25.0 * i as f64).collect();
        close(meyers_d(&p), 12.5, 1e-12);
        close(ks_two_sided(&p), 12.5, 1e-12);
        // Unsorted input: the order does not matter.
        close(meyers_d(&[90.0, 10.0, 50.0]), 100.0 / 3.0 - 10.0, 1e-12);
        close(meyers_d(&[90.0, 10.0, 50.0]), 23.333_333_333_333_33, 1e-9);
        // Two percentiles at 100: D = 50 at the first point, the KS statistic
        // sees the step from 0 to 100 and is 100, 100 / n above D.
        close(meyers_d(&[100.0, 100.0]), 50.0, 1e-12);
        close(ks_two_sided(&[100.0, 100.0]), 100.0, 1e-12);
        // KS is never below D and at most 100 / n above it.
        let q = [3.0, 8.0, 40.0, 41.0, 77.0, 99.0, 99.5];
        let (d, ks) = (meyers_d(&q), ks_two_sided(&q));
        assert!(ks >= d - 1e-12 && ks <= d + 100.0 / 7.0 + 1e-12, "{d} {ks}");
    }

    #[test]
    fn the_bound_is_136_over_root_n() {
        close(ks_bound(50), 19.233, 1e-3);
        close(ks_bound(200), 9.617, 1e-3);
    }

    #[test]
    fn pp_points_are_sorted_against_the_uniform() {
        let pts = pp_points(&[70.0, 10.0, 40.0, 95.0]);
        // e_i = 100 i / 5 (Meyers' axis), f_i = 100 i / 4 (his D's points).
        let want = [
            (20.0, 25.0, 10.0),
            (40.0, 50.0, 40.0),
            (60.0, 75.0, 70.0),
            (80.0, 100.0, 95.0),
        ];
        assert_eq!(pts.len(), 4);
        for (got, want) in pts.iter().zip(want) {
            close(got.0, want.0, 1e-12);
            close(got.1, want.1, 1e-12);
            close(got.2, want.2, 1e-12);
        }
    }

    #[test]
    fn lognormal_percentile_matches_meyers_group_353() {
        // Meyers' Table 3.1 and 4.1: incurred 38,914 (se 1,057) at 40,061 is
        // the 86.03th percentile, paid 39,177 (se 1,442) at 40,000 the 72.02th.
        close(
            lognormal_percentile(38_914.0, 1_057.0, 40_061.0).unwrap(),
            86.03,
            0.1,
        );
        close(
            lognormal_percentile(39_177.0, 1_442.0, 40_000.0).unwrap(),
            72.02,
            0.1,
        );
        // The median of the matched log-normal is mean / sqrt(1 + cv^2).
        let (m, s) = (100.0_f64, 30.0_f64);
        let median = m / (1.0 + (s / m) * (s / m)).sqrt();
        close(lognormal_percentile(m, s, median).unwrap(), 50.0, 1e-9);
        assert!(lognormal_percentile(0.0, 1.0, 1.0).is_err());
        assert!(lognormal_percentile(1.0, 0.0, 1.0).is_err());
        assert!(lognormal_percentile(1.0, 1.0, -1.0).is_err());
        assert!(lognormal_percentile(f64::NAN, 1.0, 1.0).is_err());
    }

    #[test]
    fn a_percentile_from_simulations_counts_draws_at_or_below() {
        let draws = Sampled::new((1..=10).map(f64::from).collect()).unwrap();
        close(100.0 * draws.cdf(7.0), 70.0, 1e-12);
        close(100.0 * draws.cdf(0.5), 0.0, 1e-12);
        close(100.0 * draws.cdf(10.0), 100.0, 1e-12);
    }

    #[test]
    fn seeds_are_fixed_and_differ() {
        let a = seed_for(Line::Ca, "353", "odp", Basis::Paid);
        assert_eq!(a, seed_for(Line::Ca, "353", "odp", Basis::Paid));
        assert_ne!(a, seed_for(Line::Pa, "353", "odp", Basis::Paid));
        assert_ne!(a, seed_for(Line::Ca, "353", "mack", Basis::Paid));
        assert_ne!(a, seed_for(Line::Ca, "353", "odp", Basis::Incurred));
        // The seed is pinned, so a platform or toolchain cannot drift it.
        assert_eq!(
            seed_for(Line::Wc, "1", "mack", Basis::Paid),
            0x46ce_8fc1_469f_912e_u64
        );
    }

    #[test]
    fn the_panel_parses_and_the_training_triangle_is_the_upper_half() {
        let mut text = synthetic_csv("wc", "7");
        text.push_str(
            &synthetic_csv("ca", "9")
                .lines()
                .skip(1)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        text.push('\n');
        let panel = parse_panel(&text).unwrap();
        assert_eq!(panel.len(), 2);
        assert_eq!((panel[0].line, panel[0].group.as_str()), (Line::Wc, "7"));
        assert_eq!((panel[1].line, panel[1].group.as_str()), (Line::Ca, "9"));
        let t = &panel[0];
        // The outcome is the sum of the age-10 column, premium kept apart.
        let by_origin = t.outcome_by_origin(Basis::Paid);
        close(t.outcome(Basis::Paid), by_origin.iter().sum::<f64>(), 1e-9);
        assert!(t.outcome(Basis::Incurred) > t.outcome(Basis::Paid));
        // The 55 cells: the chain ladder's latest values are the diagonal and
        // the oldest origin is at age 10.
        let fit = Mack::default().fit(&t.training(), "paid").unwrap();
        for (o, latest) in fit.chain_ladder.latest.iter().enumerate() {
            close(*latest, t.cells[o][ORIGINS - 1 - o].paid, 1e-9);
        }
        close(fit.chain_ladder.latest[0], by_origin[0], 1e-9);
    }

    #[test]
    fn a_malformed_panel_is_an_error_not_a_panic() {
        assert!(parse_panel("").is_err());
        assert!(parse_panel("a,b\n1,2\n").unwrap_err().contains("header"));
        let text = synthetic_csv("wc", "7");
        let short: String = text.lines().take(100).collect::<Vec<_>>().join("\n");
        assert!(parse_panel(&short).unwrap_err().contains("99 cells"));
        let mut repeated = text.clone();
        repeated.push_str(text.lines().nth(1).unwrap());
        assert!(parse_panel(&repeated).unwrap_err().contains("repeats"));
        let bad_line = text.replace("wc,", "xx,");
        assert!(parse_panel(&bad_line).unwrap_err().contains("unknown line"));
        let outside = text.replace(",1997,", ",2005,");
        assert!(parse_panel(&outside).unwrap_err().contains("outside"));
    }

    #[test]
    fn an_absent_panel_names_the_command_and_the_path() {
        let missing = Path::new(env!("CARGO_MANIFEST_DIR")).join("does-not-exist/panel.csv");
        let msg = load_panel_at(&missing).unwrap_err();
        assert!(msg.contains("calibration_panel.py"), "{msg}");
        assert!(msg.contains("cas-schedule-p"), "{msg}");
        assert!(msg.contains("does-not-exist"), "{msg}");
    }

    #[test]
    fn both_arms_score_a_synthetic_triangle() {
        let panel = parse_panel(&synthetic_csv("pa", "1")).unwrap();
        let t = &panel[0];
        for basis in [Basis::Paid, Basis::Incurred] {
            let s = score_mack(t, basis).unwrap();
            assert!(s.percentile > 0.0 && s.percentile < 100.0, "{s:?}");
            assert!(s.per_origin[0].is_nan() && s.per_origin[9].is_finite());
            close(s.outcome, t.outcome(basis), 1e-9);
        }
        let s = score_odp(t, 1, 2_000).unwrap();
        assert!(s.percentile >= 0.0 && s.percentile <= 100.0, "{s:?}");
        assert!(s.sd > 0.0 && s.mean > 0.0);
        // The same seed gives the same percentile.
        close(
            score_odp(t, 1, 2_000).unwrap().percentile,
            s.percentile,
            0.0,
        );
    }

    /// The arms on Meyers' own triangle, group 353 commercial auto, with its
    /// outcomes in the lower triangle: Mack gives his Table 3.1 and 4.1
    /// percentiles, and the ODP bootstrap's total is the reserve plus what
    /// is paid (a missing shift, or ultimates scored as reserves, fails).
    #[test]
    fn the_arms_reproduce_meyers_on_group_353() {
        let t = group_353();
        close(t.outcome(Basis::Paid), 40_000.0, 0.0);
        close(t.outcome(Basis::Incurred), 40_061.0, 0.0);
        let paid = score_mack(&t, Basis::Paid).unwrap();
        let incurred = score_mack(&t, Basis::Incurred).unwrap();
        close(paid.percentile, 72.02, 0.1);
        close(incurred.percentile, 86.03, 0.1);
        close(paid.mean, 39_177.0, 1.0);
        close(incurred.mean, 38_914.0, 1.0);
        // 2,000 simulations: the percentile's Monte Carlo sd is about one
        // point; the study's 10,000 gave 74.06.
        let seed = seed_for(Line::Ca, "353", "odp", Basis::Paid);
        let odp = score_odp(&t, seed, 2_000).unwrap();
        assert!(
            (odp.mean - paid.mean).abs() < 0.01 * paid.mean,
            "{} vs {}",
            odp.mean,
            paid.mean
        );
        assert!(
            (odp.sd - paid.sd).abs() < 0.2 * paid.sd,
            "{} vs {}",
            odp.sd,
            paid.sd
        );
        close(odp.percentile, 74.0, 5.0);
    }

    #[test]
    fn panel_check_counts_by_status_and_in_total() {
        let text = "line,group_code,status,rank,screen_pass\n\
                    ca,1,both,3,63\nca,2,unchecked,7,63\nca,3,printed_only,55,63\n\
                    pa,4,chosen_only,12,60\npa,5,both,1,60\n";
        let c = panel_check_counts(text);
        assert_eq!(c["ca"], [1, 0, 1, 1]);
        assert_eq!(c["pa"], [1, 1, 0, 0]);
        assert_eq!(c["all"], [2, 1, 1, 1]);
        // The old three-column file still counts.
        assert_eq!(
            panel_check_counts("line,group_code,status\nwc,9,both\n")["wc"],
            [1, 0, 0, 0]
        );
    }

    #[test]
    fn groups_split_own_and_common_sets_and_count_skips() {
        let panel = parse_panel(&synthetic_csv("ca", "1")).unwrap();
        let t = &panel[0];
        let arm_a = ArmBasis {
            arm: "mack",
            basis: Basis::Paid,
        };
        let arm_b = ArmBasis {
            arm: "odp",
            basis: Basis::Paid,
        };
        let ok = |p: f64| {
            Ok(Scored {
                percentile: p,
                mean: 0.0,
                sd: 0.0,
                outcome: 0.0,
                per_origin: [f64::NAN; ORIGINS],
                diag: None,
                first: None,
            })
        };
        let records = vec![
            Record {
                triangle: t,
                arm: arm_a,
                seed: 0,
                wall_ms: 0.0,
                result: ok(40.0),
            },
            Record {
                triangle: t,
                arm: arm_b,
                seed: 0,
                wall_ms: 0.0,
                result: Err("boom".into()),
            },
        ];
        let gs = groups(&records, &[arm_a, arm_b], &[Line::Ca]);
        let find = |arm: ArmBasis, set: &str, label: &str| {
            gs.iter()
                .find(|g| g.arm == arm && g.set == set && g.label == label)
                .unwrap()
        };
        assert_eq!(find(arm_a, "own", "ca").percentiles, vec![40.0]);
        assert!(find(arm_a, "common", "ca").percentiles.is_empty());
        assert_eq!(find(arm_a, "common", "ca").skipped, 1);
        assert!(find(arm_b, "own", "all").percentiles.is_empty());
        assert_eq!(find(arm_b, "own", "all").skipped, 1);
        let csv = ks_csv(&gs, &BTreeMap::new());
        assert!(csv.starts_with("line,arm,basis,set,n,skipped,meyers_d"));
        assert!(csv.contains("ca,mack,paid,own,1,0,"));
        let errors = percentiles_csv(&records);
        assert!(errors.contains("boom"));
    }

    #[test]
    fn the_selection_variables_parse() {
        // Mack (2), ODP, BayesOdp, CRC (2), CSR, CAY, Compartmental model 1;
        // model 2 only when named.
        assert_eq!(selected_arms(None).len(), 9);
        assert!(
            selected_arms(None)
                .iter()
                .all(|a| a.arm != "compartmental_2")
        );
        assert_eq!(
            selected_arms(Some("compartmental_2"))[0].label(),
            "compartmental_2_paid"
        );
        assert_eq!(
            selected_arms(Some("compartmental, compartmental_2")).len(),
            2
        );
        assert_eq!(selected_arms(Some("odp")).len(), 1);
        let bases = |spec: &str| -> Vec<Basis> {
            selected_arms(Some(spec)).iter().map(|a| a.basis).collect()
        };
        assert_eq!(bases("crc"), vec![Basis::Paid, Basis::Incurred]);
        assert_eq!(bases("csr"), vec![Basis::Paid]);
        assert_eq!(bases("cay"), vec![Basis::Incurred]);
        assert_eq!(bases("bayes_odp"), vec![Basis::Paid]);
        assert_eq!(selected_arms(Some("cay"))[0].label(), "cay_incurred");
        assert_eq!(selected_lines(Some("ca, wc")), vec![Line::Ca, Line::Wc]);
        assert_eq!(selected_lines(None).len(), 4);
        assert_eq!(meyers_published("mack", Basis::Paid, None), Some(23.1));
        assert_eq!(meyers_published("odp", Basis::Incurred, None), None);
        // Meyers' Figures 5.1, 5.2, 7.4 and 8.5; none for BayesOdp or the
        // bases he did not run.
        assert_eq!(
            meyers_published("crc", Basis::Paid, Some(Line::Ol)),
            Some(25.3)
        );
        assert_eq!(meyers_published("crc", Basis::Incurred, None), Some(11.9));
        assert_eq!(
            meyers_published("csr", Basis::Paid, Some(Line::Ca)),
            Some(5.9)
        );
        assert_eq!(
            meyers_published("cay", Basis::Incurred, Some(Line::Pa)),
            Some(15.9)
        );
        assert_eq!(meyers_published("csr", Basis::Incurred, None), None);
        assert_eq!(meyers_published("bayes_odp", Basis::Paid, None), None);
        assert_eq!(csv_field("a,b"), "\"a,b\"");
    }

    fn scored_with(percentile: f64, diag: DiagSummary) -> Scored {
        Scored {
            percentile,
            mean: 1.0,
            sd: 1.0,
            outcome: 1.0,
            per_origin: [f64::NAN; ORIGINS],
            diag: Some(diag),
            first: None,
        }
    }

    fn clean() -> DiagSummary {
        DiagSummary {
            max_rhat: 1.002,
            min_ess_bulk: 900.0,
            min_ess_tail: 800.0,
            divergences: 0,
            rate_swaps: 0,
        }
    }

    #[test]
    fn a_refit_is_for_rhat_above_1_05_nan_or_a_divergence() {
        assert!(!clean().needs_refit());
        assert!(
            !DiagSummary {
                max_rhat: 1.05,
                ..clean()
            }
            .needs_refit()
        );
        assert!(
            DiagSummary {
                max_rhat: 1.0501,
                ..clean()
            }
            .needs_refit()
        );
        assert!(
            DiagSummary {
                max_rhat: f64::NAN,
                ..clean()
            }
            .needs_refit()
        );
        assert!(
            DiagSummary {
                divergences: 1,
                ..clean()
            }
            .needs_refit()
        );
    }

    #[test]
    fn with_refit_refits_once_at_the_higher_target_and_keeps_both_results() {
        use std::cell::RefCell;
        let calls = RefCell::new(Vec::new());
        // Clean at once: one fit, no first-fit record.
        let r = with_refit(|ta| {
            calls.borrow_mut().push(ta);
            Ok(scored_with(40.0, clean()))
        })
        .unwrap();
        assert_eq!(*calls.borrow(), vec![FIRST_ACCEPT]);
        assert!(r.first.is_none());
        // A divergence at 0.95, clean at 0.99: the refit is the result and
        // carries the first fit's percentile and diagnostics.
        calls.borrow_mut().clear();
        let bad = DiagSummary {
            divergences: 3,
            ..clean()
        };
        let r = with_refit(|ta| {
            calls.borrow_mut().push(ta);
            Ok(if ta == FIRST_ACCEPT {
                scored_with(40.0, bad)
            } else {
                scored_with(55.0, clean())
            })
        })
        .unwrap();
        assert_eq!(*calls.borrow(), vec![FIRST_ACCEPT, REFIT_ACCEPT]);
        close(r.percentile, 55.0, 0.0);
        assert_eq!(r.diag, Some(clean()));
        assert_eq!(r.first, Some((40.0, bad)));
        // A refit that errors leaves the first fit.
        let r = with_refit(|ta| {
            if ta == FIRST_ACCEPT {
                Ok(scored_with(40.0, bad))
            } else {
                Err("no".into())
            }
        })
        .unwrap();
        close(r.percentile, 40.0, 0.0);
        assert!(r.first.is_none() && r.diag == Some(bad));
        // An error of the first fit is the error.
        assert_eq!(
            with_refit(|_| Err("first".to_string())).unwrap_err(),
            "first"
        );
        // Mack and the ODP bootstrap have no diagnostics and never refit.
        let n = RefCell::new(0);
        let _ = with_refit(|_| {
            *n.borrow_mut() += 1;
            Ok(Scored {
                diag: None,
                ..scored_with(1.0, clean())
            })
        });
        assert_eq!(*n.borrow(), 1);
    }

    /// Compartmental's first fit is at the library's target acceptance and
    /// its refit a tenth of the remaining distance to 1 closer; the refit
    /// carries the first fit, and the rate swaps print only when present.
    #[test]
    fn compartmental_refits_a_tenth_closer_to_1_and_prints_its_swaps() {
        use std::cell::RefCell;
        close(compartmental_refit_accept(false), 0.999, 1e-12);
        close(compartmental_refit_accept(true), 0.9999, 1e-12);
        for m2 in [false, true] {
            let (first, refit) = (compartmental_accept(m2), compartmental_refit_accept(m2));
            let calls = RefCell::new(Vec::new());
            let bad = DiagSummary {
                divergences: 2,
                rate_swaps: 7,
                ..clean()
            };
            let r = with_refit_at(first, refit, |ta| {
                calls.borrow_mut().push(ta);
                Ok(if ta == first {
                    scored_with(30.0, bad)
                } else {
                    scored_with(60.0, clean())
                })
            })
            .unwrap();
            assert_eq!(*calls.borrow(), vec![first, refit]);
            close(r.percentile, 60.0, 0.0);
            assert_eq!(r.first, Some((30.0, bad)));
        }
        assert!(!clean().text().contains("swaps"));
        let swapped = DiagSummary {
            rate_swaps: 7,
            ..clean()
        };
        assert!(
            swapped.text().ends_with(";div=0;swaps=7"),
            "{}",
            swapped.text()
        );
        // Swaps are recorded, not a refit trigger.
        assert!(!swapped.needs_refit());
    }

    #[test]
    fn the_diagnostics_reach_the_csv_and_the_log_line() {
        let panel = parse_panel(&synthetic_csv("ca", "1")).unwrap();
        let t = &panel[0];
        let arm = ArmBasis {
            arm: "crc",
            basis: Basis::Paid,
        };
        let bad = DiagSummary {
            max_rhat: 1.2,
            divergences: 2,
            ..clean()
        };
        let refit = Scored {
            first: Some((33.0, bad)),
            ..scored_with(44.0, clean())
        };
        let records = vec![
            Record {
                triangle: t,
                arm,
                seed: 1,
                wall_ms: 5.0,
                result: Ok(refit),
            },
            Record {
                triangle: t,
                arm,
                seed: 2,
                wall_ms: 5.0,
                result: Err("bad".into()),
            },
        ];
        let csv = percentiles_csv(&records);
        let row = csv.lines().nth(1).unwrap();
        assert!(
            row.contains("rhat=1.0020;ess_bulk=900;ess_tail=800;div=0"),
            "{row}"
        );
        assert!(
            row.contains("first_fit_percentile=33.0000;rhat=1.2000"),
            "{row}"
        );
        // Every row has the header's number of fields.
        let width = csv.lines().next().unwrap().split(',').count();
        for l in csv.lines() {
            assert_eq!(l.split(',').count(), width, "{l}");
        }
        let line = diagnostics_line(&records, arm).unwrap();
        assert!(line.contains("1 NUTS fits; first fit failed 1"), "{line}");
        assert!(
            line.contains("1 refitted") && line.contains("0 still failing"),
            "{line}"
        );
        assert_eq!(error_tally(&records, arm), vec![("bad".to_string(), 1)]);
        let mack = ArmBasis {
            arm: "mack",
            basis: Basis::Paid,
        };
        assert!(diagnostics_line(&records, mack).is_none());
    }

    /// The NUTS arms with the sampler cut down, on a synthetic square.
    #[test]
    fn the_bayesian_arms_score_a_synthetic_triangle() {
        // Increments with a wiggle, so that every incremental value is
        // positive (BayesOdp's log-linear mean needs positive column sums).
        let mut csv = String::from(PANEL_HEADER);
        csv.push('\n');
        let pattern = [
            0.30, 0.25, 0.17, 0.12, 0.07, 0.04, 0.025, 0.015, 0.007, 0.003,
        ];
        for o in 0..ORIGINS {
            let mut cum = 0.0;
            for (d, &p) in pattern.iter().enumerate() {
                cum += 1000.0
                    * (1.0 + 0.1 * o as f64)
                    * p
                    * (1.0 + 0.2 * ((o * 7 + d * 3) as f64).sin());
                let case = cum * 1.1;
                let _ = writeln!(
                    csv,
                    "pa,1,{},{},{cum},{},{case},{},{}",
                    FIRST_ORIGIN as usize + o,
                    12 * (d + 1),
                    case + 5.0,
                    1500.0 + o as f64,
                    2000.0
                );
            }
        }
        let panel = parse_panel(&csv).unwrap();
        let t = &panel[0];
        let cheap = |s: &mut prospicio_bayes::nuts::Sampler| {
            s.chains = 2;
            s.tune = 150;
            s.draws = 150;
        };
        let mut odp = bayes_odp_for(1, FIRST_ACCEPT);
        cheap(&mut odp.sampler);
        odp.n_sims = 500;
        let s = score_bayes_odp(t, &odp).unwrap();
        assert!(s.percentile >= 0.0 && s.percentile <= 100.0, "{s:?}");
        assert!(s.diag.is_some() && s.mean > 0.0 && s.sd > 0.0);
        // The reserves are shifted by what is paid: the mean is an
        // ultimate, near the data's age-10 total, not a reserve.
        assert!(s.mean > 0.5 * t.outcome(Basis::Paid), "{s:?}");
        close(
            score_bayes_odp(t, &odp).unwrap().percentile,
            s.percentile,
            0.0,
        );
        for (kind, basis) in [
            (MeyersModel::Crc, Basis::Paid),
            (MeyersModel::Csr, Basis::Paid),
            (MeyersModel::Cay, Basis::Incurred),
        ] {
            let mut m = meyers_for(kind, 1, FIRST_ACCEPT);
            cheap(&mut m.sampler);
            m.n_sims = 500;
            let s = score_meyers(t, basis, &m).unwrap();
            assert!(
                s.percentile >= 0.0 && s.percentile <= 100.0,
                "{kind:?} {s:?}"
            );
            assert!(s.diag.is_some() && s.sd > 0.0);
            assert!(s.per_origin[0].is_nan() && s.per_origin[9].is_finite());
            close(s.outcome, t.outcome(basis), 1e-9);
            // The ultimates are not shifted: the mean is near the data's
            // age-10 total, not near the reserve.
            assert!(s.mean > 0.5 * t.outcome(basis), "{kind:?} {s:?}");
        }
    }

    /// Compartmental on a synthetic square, with the sampler cut down: the
    /// reserves are shifted by what is paid (the mean is an ultimate), the
    /// rate swaps are carried and the run is reproducible by seed.
    #[test]
    fn the_compartmental_arm_scores_a_synthetic_triangle() {
        let mut csv = String::from(PANEL_HEADER);
        csv.push('\n');
        let pattern = [
            0.30, 0.25, 0.17, 0.12, 0.07, 0.04, 0.025, 0.015, 0.007, 0.003,
        ];
        for o in 0..ORIGINS {
            let mut cum = 0.0;
            for (d, &p) in pattern.iter().enumerate() {
                cum += 1000.0
                    * (1.0 + 0.1 * o as f64)
                    * p
                    * (1.0 + 0.2 * ((o * 7 + d * 3) as f64).sin());
                let case = cum * (1.0 + 1.5 * (1.0 - d as f64 / 10.0));
                let _ = writeln!(
                    csv,
                    "pa,1,{},{},{cum},{},{case},{},{}",
                    FIRST_ORIGIN as usize + o,
                    12 * (d + 1),
                    case + 5.0,
                    1500.0 + o as f64,
                    2000.0
                );
            }
        }
        let panel = parse_panel(&csv).unwrap();
        let t = &panel[0];
        let mut m = compartmental_for(false, 1, 0.9);
        m.sampler.chains = 2;
        m.sampler.tune = 200;
        m.sampler.draws = 200;
        m.n_sims = 500;
        let fit = m
            .fit(
                &t.training(),
                Basis::Paid.name(),
                Outstanding::FromIncurred(Basis::Incurred.name()),
                "premium",
            )
            .unwrap();
        let s = compartmental_scored(t, &fit).unwrap();
        // The fit's diagnostics, its rate swaps included, are carried.
        assert_eq!(
            s.diag,
            Some(DiagSummary {
                rate_swaps: fit.rate_swaps,
                ..DiagSummary::new(&fit.diagnostics)
            })
        );
        assert!(s.percentile >= 0.0 && s.percentile <= 100.0, "{s:?}");
        assert!(s.diag.is_some() && s.sd > 0.0);
        close(s.outcome, t.outcome(Basis::Paid), 0.0);
        assert!(s.per_origin[0].is_nan() && s.per_origin[9].is_finite());
        // An ultimate, near the data's age-10 total, not a reserve.
        assert!(s.mean > 0.5 * t.outcome(Basis::Paid), "{s:?}");
        close(
            score_compartmental(t, &m).unwrap().percentile,
            s.percentile,
            0.0,
        );
        // Model 2 differs by its settings, not its wiring.
        let m2 = compartmental_for(true, 1, 0.9);
        assert_eq!(
            m2.likelihood,
            prospicio_reserving::bayes::CompartmentalLikelihood::LogNormal
        );
        close(compartmental_accept(false), 0.99, 0.0);
        close(compartmental_accept(true), 0.999, 0.0);
    }

    /// Compartmental model 1 through the study's code path on Meyers' group
    /// 353 commercial auto (paid, outstanding from case incurred, the
    /// premium column). Meyers publishes no compartmental result, so the
    /// check is of the wiring, against the models of this file: the
    /// predictive mean ultimate agrees with the ODP bootstrap's and Mack's
    /// on the same square (the preset is a WC prior, so loosely), the
    /// percentile is a probability and the oldest origin is already at age
    /// 10. About 1.5 s in release. (The paper's company 337 is not a 10 x 10
    /// square, so its parity targets stay in `reserving_compartmental.rs`.)
    /// On this square the preset gives a mean of 35,803 against the
    /// bootstrap's 39,197 and the outcome 40,000: the percentile is 100.
    #[test]
    fn the_compartmental_arm_scores_group_353() {
        let t = group_353();
        let s = ArmBasis {
            arm: "compartmental",
            basis: Basis::Paid,
        }
        .run(&t)
        .unwrap();
        let boot = ArmBasis {
            arm: "odp",
            basis: Basis::Paid,
        }
        .run(&t)
        .unwrap();
        eprintln!(
            "compartmental paid: {:.2} mean {:.0} sd {:.0} {:?}; odp bootstrap: {:.2} mean {:.0} sd {:.0}",
            s.percentile, s.mean, s.sd, s.diag, boot.percentile, boot.mean, boot.sd
        );
        close(s.outcome, boot.outcome, 0.0);
        close(s.mean, boot.mean, 0.15 * boot.mean);
        assert!(s.per_origin[9].is_finite() && s.per_origin[0].is_nan());
        assert!((0.0..=100.0).contains(&s.percentile));
        assert!(s.diag.is_some_and(|d| !d.needs_refit()) || s.first.is_some());
    }

    /// The Bayesian arms on Meyers' own triangle, with the study's code
    /// path (seeds, refit rule): CRC, CSR and CAY give his printed outcome
    /// percentiles (Tables 5.2, 5.3, 7.2 and 8.2; the guide's tolerance of
    /// 5 points, `knowledge/references/meyers-monograph-8-reproduced.md`),
    /// and `BayesOdp` agrees with the ODP bootstrap run in the same test at
    /// the study's 10,000 simulations. The file builds only with
    /// `slow-bayes` (its `[[test]]` entry), so CI's `bayes` job runs this in
    /// release, about 2 s. Run with
    /// `cargo test --release -p prospicio-validation --features slow-bayes
    /// --test reserving_calibration_meyers`.
    #[test]
    fn the_bayesian_arms_reproduce_meyers_on_group_353() {
        let t = group_353();
        for (arm, basis, printed) in [
            ("crc", Basis::Paid, 51.88),
            ("crc", Basis::Incurred, 74.75),
            ("csr", Basis::Paid, 86.26),
            ("cay", Basis::Incurred, 73.24),
        ] {
            let s = ArmBasis { arm, basis }.run(&t).unwrap();
            eprintln!(
                "{arm} {}: {:.2} (Meyers {printed}) {:?}",
                basis.name(),
                s.percentile,
                s.diag
            );
            close(s.percentile, printed, 5.0);
            close(s.outcome, t.outcome(basis), 0.0);
            assert!(s.per_origin[9].is_finite() && s.per_origin[0].is_nan());
        }
        let odp = ArmBasis {
            arm: "bayes_odp",
            basis: Basis::Paid,
        }
        .run(&t)
        .unwrap();
        let boot = ArmBasis {
            arm: "odp",
            basis: Basis::Paid,
        }
        .run(&t)
        .unwrap();
        eprintln!(
            "bayes_odp paid: {:.2} mean {:.0} sd {:.0} {:?}; odp bootstrap: {:.2} mean {:.0} sd {:.0}",
            odp.percentile, odp.mean, odp.sd, odp.diag, boot.percentile, boot.mean, boot.sd
        );
        close(odp.outcome, boot.outcome, 0.0);
        close(odp.mean, boot.mean, 0.01 * boot.mean);
        // On this platform 69.75 against 74.06: BayesOdp's predictive mean
        // is 39,376 against the bootstrap's 39,197 (sd 1,399 and 1,381), so
        // the outcome 40,000 sits 0.45 sd above its mean against 0.58, about
        // 4.5 points at a normal. The tolerance leaves 1.7 points for the
        // other platforms' NUTS draws and the bootstrap's Monte Carlo error
        // (each about a point); a dropped shift or a wrong basis moves the
        // percentile by tens of points.
        const BAYES_ODP_VS_BOOTSTRAP: f64 = 6.0;
        close(odp.percentile, boot.percentile, BAYES_ODP_VS_BOOTSTRAP);
    }

    #[test]
    fn a_seed_salt_changes_the_seed_and_no_salt_keeps_it() {
        let s = seed_for(Line::Ca, "353", "crc", Basis::Paid);
        assert_eq!(salted(s, None), s);
        assert_eq!(salted(s, Some("")), s);
        assert_ne!(salted(s, Some("a")), s);
        assert_ne!(salted(s, Some("a")), salted(s, Some("b")));
        assert_eq!(salted(s, Some("a")), salted(s, Some("a")));
        // Another triangle keeps another seed under the same salt.
        let t = seed_for(Line::Pa, "353", "crc", Basis::Paid);
        assert_ne!(salted(s, Some("a")), salted(t, Some("a")));
    }

    fn cell(scope: Option<Line>, n: usize, d: f64) -> (Pin, Observed) {
        (
            pin("mack", Basis::Paid, scope, n, d),
            Observed {
                arm: "mack",
                basis: Basis::Paid,
                scope,
                n,
                d,
            },
        )
    }

    /// Tolerance 2 points per line and 1 on all 200.
    fn tol(_: &str, scope: Option<Line>) -> f64 {
        if scope.is_some() { 2.0 } else { 1.0 }
    }

    #[test]
    fn a_d_inside_the_tolerance_passes_and_outside_fails() {
        let (p, mut o) = cell(Some(Line::Wc), 50, 33.4);
        assert!(check_pins(&[p], &[o], true, tol).is_empty());
        o.d = 35.3; // inside by 0.1
        assert!(check_pins(&[p], &[o], true, tol).is_empty());
        o.d = 35.5;
        let broken = check_pins(&[p], &[o], true, tol);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].contains("mack paid wc"), "{broken:?}");
        o.d = 31.5;
        assert!(check_pins(&[p], &[o], true, tol).is_empty());
        o.d = 31.3;
        assert_eq!(check_pins(&[p], &[o], true, tol).len(), 1);
        // The all-200 cell has the tighter tolerance.
        let (p, mut o) = cell(None, 200, 24.6);
        o.d = 25.7;
        assert_eq!(check_pins(&[p], &[o], true, tol).len(), 1);
        o.d = 25.5;
        assert!(check_pins(&[p], &[o], true, tol).is_empty());
    }

    #[test]
    fn another_count_of_scored_triangles_or_no_score_fails() {
        let (p, mut o) = cell(Some(Line::Ol), 49, 10.3);
        o.n = 50;
        let broken = check_pins(&[p], &[o], true, tol);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].contains("scored 50 triangles, pinned on 49"));
        // A cell that scored nothing has no D: it fails, not passes.
        o.n = 49;
        o.d = f64::NAN;
        assert_eq!(check_pins(&[p], &[o], true, tol).len(), 1);
        // An arm without a tolerance fails loudly, too.
        let (p, o) = cell(Some(Line::Ca), 50, 20.0);
        assert_eq!(
            check_pins(&[p], &[o], true, |_, _| f64::NAN).len(),
            1,
            "NaN tolerance"
        );
    }

    #[test]
    fn the_verdict_may_flip_only_within_the_tolerance_of_the_bound() {
        // The bound for n = 50 is 136 / sqrt(50) = 19.23.
        let bound = ks_bound(50);
        // Pinned 21.0, 1.8 above the bound (within the tolerance of 2): a D
        // of 19.0 is within the tolerance of the pin and flips the verdict,
        // which is allowed.
        let (p, mut o) = cell(Some(Line::Ca), 50, bound + 1.8);
        o.d = bound - 0.1;
        assert!(check_pins(&[p], &[o], true, tol).is_empty());
        assert!(near_the_bound(&p, 2.0));
        // Pinned 2.5 above the bound (outside the tolerance of 2): a D that
        // crosses the bound is more than 2 points from the pin, so the cell
        // fails and the message names the flipped verdict.
        let (p, mut o) = cell(Some(Line::Ca), 50, bound + 2.5);
        assert!(!near_the_bound(&p, 2.0));
        o.d = bound - 0.5;
        let broken = check_pins(&[p], &[o], true, tol);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].contains("verdict flipped"), "{broken:?}");
        // Staying on the pinned side but too far fails without the flip.
        o.d = bound + 5.0;
        let broken = check_pins(&[p], &[o], true, tol);
        assert_eq!(broken.len(), 1);
        assert!(!broken[0].contains("verdict flipped"), "{broken:?}");
        // Staying on the pinned side within the tolerance passes.
        o.d = bound + 1.0;
        assert!(check_pins(&[p], &[o], true, tol).is_empty());
    }

    #[test]
    fn a_narrowed_run_asserts_only_what_it_ran() {
        let (p_ca, o_ca) = cell(Some(Line::Ca), 50, 20.0);
        let (p_wc, _) = cell(Some(Line::Wc), 50, 30.0);
        let (p_all, mut o_all) = cell(None, 200, 25.0);
        let pins = [p_ca, p_wc, p_all];
        // Only commercial auto ran: wc has no cell, all is skipped even
        // though its observed value is far off.
        o_all.d = 99.0;
        assert!(check_pins(&pins, &[o_ca, o_all], false, tol).is_empty());
        // With every line the all-200 cell counts again.
        assert_eq!(check_pins(&pins, &[o_ca, o_all], true, tol).len(), 1);
        // An arm that was not run is skipped.
        let other = Observed { arm: "odp", ..o_ca };
        assert!(check_pins(&[p_ca], &[other], true, tol).is_empty());
        // A cell far off on a line that ran fails whatever else was run.
        let mut off = o_ca;
        off.d = 50.0;
        assert_eq!(check_pins(&pins, &[off], false, tol).len(), 1);
    }

    #[test]
    fn pins_apply_only_to_the_default_seed_and_the_whole_panel() {
        assert!(pins_active(None, None));
        assert!(pins_active(None, Some("")));
        assert!(!pins_active(Some(2), None));
        assert!(!pins_active(None, Some("noise")));
    }

    #[test]
    fn another_panel_hash_fails_with_the_command_to_rebuild() {
        let ok = vec![("panel_sha256".to_string(), PINNED_PANEL_SHA256.to_string())];
        assert!(check_panel_hash(&ok).is_ok());
        let other = vec![("panel_sha256".to_string(), "abc".to_string())];
        let e = check_panel_hash(&other).unwrap_err();
        assert!(
            e.contains("abc")
                && e.contains(PINNED_PANEL_SHA256)
                && e.contains("calibration_panel.py")
        );
        assert!(check_panel_hash(&[]).unwrap_err().contains("absent"));
    }

    #[test]
    fn the_pins_are_well_formed() {
        for p in PINS {
            assert!(
                p.n >= 1 && p.n <= if p.scope.is_some() { 50 } else { 200 },
                "{p:?}"
            );
            assert!(p.d.is_finite() && (0.0..=100.0).contains(&p.d), "{p:?}");
            assert!(
                pin_tolerance(p.arm, p.scope) > 0.0,
                "{} has no tolerance",
                p.arm
            );
            assert!(
                ALL_ARMS.split(',').any(|a| a == p.arm),
                "{} is not an arm",
                p.arm
            );
        }
        for (arm, per_line, all) in TOLERANCES {
            assert!(ALL_ARMS.split(',').any(|a| a == *arm));
            assert!(*per_line > 0.0 && *all > 0.0);
        }
    }

    #[test]
    fn a_runs_cells_print_as_pins_source() {
        let o = Observed {
            arm: "odp",
            basis: Basis::Paid,
            scope: Some(Line::Pa),
            n: 50,
            d: 48.84,
        };
        let all = Observed {
            scope: None,
            n: 200,
            ..o
        };
        let text = pins_source(&[o, all]);
        assert!(text.contains("pin(\"odp\", Basis::Paid, Some(Line::Pa), 50, 48.84),"));
        assert!(text.contains("pin(\"odp\", Basis::Paid, None, 200, 48.84),"));
    }
}
