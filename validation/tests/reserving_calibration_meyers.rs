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
//! This file holds the harness (PR 2 of the note): Mack on paid and
//! incurred and the ODP bootstrap on paid. The study is `#[ignore]`d and
//! run by hand; nothing in CI runs it. The unit tests below it are not
//! ignored: they check the pure pieces on small inline cases and run with
//! `cargo test`.
//!
//! Run (from the repository root; the panel is built from the pip package
//! `cas-schedule-p` at run time and is not stored in the repo):
//!
//! ```text
//! uv run --no-project --with "cas-schedule-p[screens]==2026.6.13" python -I validation/scripts/calibration_panel.py
//! cargo test --release -p prospicio-validation --test reserving_calibration_meyers -- --ignored --nocapture
//! ```
//!
//! The script writes `target/calibration/panel.csv` (set
//! `RISK_RS_CALIBRATION_PANEL` for another path); the study reads it and
//! writes `percentiles.csv`, `pp.csv` and `ks.csv` beside it. Options, all
//! environment variables read by the study itself:
//!
//! * `RISK_RS_CALIBRATION_LINES`: for example `ca,pa` (default all four);
//! * `RISK_RS_CALIBRATION_ARMS`: for example `mack,odp` (the default);
//! * `RISK_RS_CALIBRATION_LIMIT`: the first n triangles of each line, for a
//!   timing probe (for example 2).
//!
//! The script also writes `panel_check.csv` (the diff against Table A.2,
//! with each code's rank in the screen) and `panel_meta.csv` (the package
//! versions and the panel's sha256), which the study prints.
//!
//! First runs assert only the harness and the panel (decision 7): the
//! panel's shape, group 353 commercial auto's square against
//! `validation/data/clrd_comauto_353.csv`, and Mack's D near Meyers'
//! published values. Each other arm's D is recorded, not asserted. The
//! unit test `the_arms_reproduce_meyers_on_group_353` checks the arms'
//! wiring on that square in every `cargo test`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use prospicio_prob::{Distribution, Lognormal};
use prospicio_reserving::{
    DevelopmentColumn, Grain, Lag, Long, Mack, Month, OdpBootstrap, Triangle,
};
use rayon::prelude::*;

/// Simulations of the ODP bootstrap (Meyers p.12 uses 10,000).
const ODP_SIMS: usize = 10_000;

/// The command that builds the panel, quoted in every failure message.
const PANEL_COMMAND: &str = "uv run --no-project --with \"cas-schedule-p[screens]==2026.6.13\" python -I validation/scripts/calibration_panel.py";

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

// ---------------------------------------------------------------------------
// Meyers' published D statistics (percentage points)
// ---------------------------------------------------------------------------

/// Meyers' D by line (ca, pa, wc, ol) and over all 200, read from the figure
/// text of the monograph: Mack incurred (Figure 3.2, p.10), Mack paid
/// (Figure 4.2, p.15), ODP bootstrap paid (Figure 4.1, p.14).
fn meyers_published(arm: &str, basis: Basis, line: Option<Line>) -> Option<f64> {
    let row: [f64; 5] = match (arm, basis) {
        ("mack", Basis::Incurred) => [16.4, 16.6, 27.0, 14.2, 15.4],
        ("mack", Basis::Paid) => [24.6, 44.6, 30.4, 8.5, 23.1],
        ("odp", Basis::Paid) => [23.1, 44.9, 28.4, 6.8, 24.1],
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
    })
}

/// The ODP bootstrap on paid: the share of simulated totals at or below the
/// outcome less what is already paid.
fn score_odp(t: &PanelTriangle, seed: u64, n_sims: usize) -> Result<Scored, String> {
    let basis = Basis::Paid;
    let fit = OdpBootstrap {
        n_sims,
        seed,
        ..Default::default()
    }
    .fit(&t.training(), basis.name())
    .map_err(|e| e.to_string())?;
    let latest = &fit.chain_ladder.latest;
    if latest.len() != ORIGINS {
        return Err(format!(
            "{} origins fitted, expected {ORIGINS}",
            latest.len()
        ));
    }
    let latest_sum: f64 = latest.iter().sum();
    let outcome = t.outcome(basis);
    let total = fit.reserves.total();
    let by_origin = t.outcome_by_origin(basis);
    let (n_comp, n_sims_got) = (fit.reserves.n_components(), fit.reserves.n_sims());
    if n_comp != ORIGINS {
        return Err(format!("{n_comp} components, expected {ORIGINS}"));
    }
    let draws = fit.reserves.draw_matrix();
    let mut per_origin = [f64::NAN; ORIGINS];
    for o in 1..ORIGINS {
        let x = by_origin[o] - latest[o];
        let below = (0..n_sims_got)
            .filter(|&s| draws[s * n_comp + o] <= x)
            .count();
        per_origin[o] = 100.0 * below as f64 / n_sims_got as f64;
    }
    Ok(Scored {
        percentile: 100.0 * total.cdf(outcome - latest_sum),
        mean: latest_sum + total.mean(),
        sd: total.variance().sqrt(),
        outcome,
        per_origin,
    })
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
        seed_for(t.line, &t.group, self.arm, self.basis)
    }

    fn run(self, t: &PanelTriangle) -> Result<Scored, String> {
        match self.arm {
            "mack" => score_mack(t, self.basis),
            "odp" => score_odp(t, self.seed(t), ODP_SIMS),
            other => Err(format!("unknown arm {other}")),
        }
    }
}

/// The arms named by `RISK_RS_CALIBRATION_ARMS` (default `mack,odp`).
fn selected_arms(spec: Option<&str>) -> Vec<ArmBasis> {
    let spec = spec.unwrap_or("mack,odp");
    let mut out = Vec::new();
    for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match name {
            "mack" => {
                out.push(ArmBasis {
                    arm: "mack",
                    basis: Basis::Paid,
                });
                out.push(ArmBasis {
                    arm: "mack",
                    basis: Basis::Incurred,
                });
            }
            "odp" => out.push(ArmBasis {
                arm: "odp",
                basis: Basis::Paid,
            }),
            other => panic!("RISK_RS_CALIBRATION_ARMS: unknown arm {other:?} (known: mack, odp)"),
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
    jobs.par_iter()
        .map(|&(triangle, arm)| {
            let start = Instant::now();
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| arm.run(triangle)))
                    .unwrap_or_else(|_| Err("the fit panicked".to_string()));
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
            "{},{},{},{},{},{},{:.1},,,{}",
            scored.map_or(String::new(), |s| num(s.percentile)),
            scored.map_or(String::new(), |s| num(s.mean)),
            scored.map_or(String::new(), |s| num(s.sd)),
            scored.map_or(String::new(), |s| num(s.outcome)),
            r.seed,
            platform(),
            r.wall_ms,
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
        assert_eq!(selected_arms(None).len(), 3);
        assert_eq!(selected_arms(Some("odp")).len(), 1);
        assert_eq!(selected_lines(Some("ca, wc")), vec![Line::Ca, Line::Wc]);
        assert_eq!(selected_lines(None).len(), 4);
        assert_eq!(meyers_published("mack", Basis::Paid, None), Some(23.1));
        assert_eq!(meyers_published("odp", Basis::Incurred, None), None);
        assert_eq!(csv_field("a,b"), "\"a,b\"");
    }
}
