//! A latest diagonal projected with a selected pattern (`PRO.DIAG.CL`,
//! `.EL`, `.BF`, `.BENKTANDER`, `.CAPECOD` and `PRO.PATTERN.AVERAGE`,
//! `docs/design/xll.md` decision 39), reached through the catalogue as
//! Excel reaches them:
//!
//! * the hand diagonal of `validation/tests/reserving_latest_diagonal.rs`
//!   under every method, by ages and by a valuation date;
//! * chainladder-python 0.10.1's `DevelopmentConstant` on GenIns and the
//!   CAS workers' compensation triangle
//!   (`validation/reference/reserving_latest_diagonal_python.csv`);
//! * ages off the pattern's grid, origins past the pattern's end, patterns
//!   averaged over origin months, a triangle's pattern frozen and applied
//!   to its own diagonal, and the errors.
//!
//! Every exhibit is also checked against the same fit made directly with
//! `prospicio_reserving`, to the bit.

mod common;

use std::collections::BTreeMap;

use common::{clrd_wkcomp, genins, n, num, raa, reference, t, text};
use prospicio_reserving::{
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ChainLadderFit, Development,
    ExpectedLoss, ExpectedLossFit, Grain, Lag, LatestDiagonal, Month, PastPatternEnd, PatternKind,
    SelectedPattern, Tail, TailCurve,
};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// A diagonal function's arguments by name; omitted ones are missing.
#[derive(Clone)]
struct Diag {
    origins: Grid,
    ages: Grid,
    latest: Grid,
    exposure: Grid,
    pattern: Grid,
    pattern_ages: Grid,
    kind: Grid,
    tail: Grid,
    grain: Grid,
    valuation: Grid,
    past_end: Grid,
    pattern_origin: Grid,
}

impl Diag {
    fn new(origins: &[f64], latest: &[f64], pattern: &[f64]) -> Self {
        Self {
            origins: col(origins),
            ages: Grid::missing(),
            latest: col(latest),
            exposure: Grid::missing(),
            pattern: row(pattern),
            pattern_ages: Grid::missing(),
            kind: Grid::missing(),
            tail: Grid::missing(),
            grain: Grid::missing(),
            valuation: Grid::missing(),
            past_end: Grid::missing(),
            pattern_origin: Grid::missing(),
        }
    }

    fn ages(mut self, ages: &[f64]) -> Self {
        self.ages = col(ages);
        self
    }

    fn valuation(mut self, serial: f64) -> Self {
        self.valuation = n(serial);
        self
    }

    fn exposure(mut self, exposure: &[f64]) -> Self {
        self.exposure = col(exposure);
        self
    }

    fn pattern_ages(mut self, ages: &[f64]) -> Self {
        self.pattern_ages = row(ages);
        self
    }

    fn kind(mut self, kind: &str) -> Self {
        self.kind = t(kind);
        self
    }

    fn tail(mut self, tail: Grid) -> Self {
        self.tail = tail;
        self
    }

    fn grain(mut self, grain: &str) -> Self {
        self.grain = t(grain);
        self
    }

    fn past_end(mut self, rule: &str) -> Self {
        self.past_end = t(rule);
        self
    }

    fn pattern_origin(mut self, basis: &str) -> Self {
        self.pattern_origin = t(basis);
        self
    }

    fn shared(&self) -> Vec<Grid> {
        vec![
            self.pattern_ages.clone(),
            self.kind.clone(),
            self.tail.clone(),
            self.grain.clone(),
            self.valuation.clone(),
            self.past_end.clone(),
            self.pattern_origin.clone(),
        ]
    }

    fn head(&self) -> Vec<Grid> {
        vec![self.origins.clone(), self.ages.clone(), self.latest.clone()]
    }

    fn cl(&self) -> Grid {
        let mut args = self.head();
        args.push(self.pattern.clone());
        args.extend(self.shared());
        call("pro_diag_cl", args)
    }

    fn el_like(&self, symbol: &str, apriori: f64) -> Grid {
        let mut args = self.head();
        args.extend([self.exposure.clone(), n(apriori), self.pattern.clone()]);
        args.extend(self.shared());
        call(symbol, args)
    }

    fn el(&self, apriori: f64) -> Grid {
        self.el_like("pro_diag_el", apriori)
    }

    fn bf(&self, apriori: f64) -> Grid {
        self.el_like("pro_diag_bf", apriori)
    }

    fn bk(&self, apriori: f64, iterations: f64) -> Grid {
        let mut args = self.head();
        args.extend([
            self.exposure.clone(),
            n(apriori),
            self.pattern.clone(),
            n(iterations),
        ]);
        args.extend(self.shared());
        call("pro_diag_benktander", args)
    }

    fn cc(&self, trend: f64, decay: f64) -> Grid {
        let mut args = self.head();
        args.extend([
            self.exposure.clone(),
            self.pattern.clone(),
            n(trend),
            n(decay),
        ]);
        args.extend(self.shared());
        call("pro_diag_capecod", args)
    }
}

fn col(values: &[f64]) -> Grid {
    Grid::column(values)
}

fn row(values: &[f64]) -> Grid {
    Grid::row(values)
}

/// An exhibit's column `name`, one value per origin (the header and the
/// total row left out).
fn column(g: &Grid, name: &str) -> Vec<f64> {
    let c = header(g, name);
    (1..g.rows - 1).map(|r| num(g.get(r, c))).collect()
}

fn header(g: &Grid, name: &str) -> usize {
    assert!(
        g.rows > 2,
        "not an exhibit: {:?}",
        g.as_scalar().map(|v| format!("{v:?}"))
    );
    (0..g.columns)
        .find(|&c| g.get(0, c) == &Value::from(name))
        .unwrap_or_else(|| panic!("no column {name}"))
}

/// The total row's value of column `name`.
fn total(g: &Grid, name: &str) -> f64 {
    num(g.get(g.rows - 1, header(g, name)))
}

/// The first column's labels, one per origin.
fn labels(g: &Grid) -> Vec<Value> {
    (1..g.rows - 1).map(|r| g.get(r, 0).clone()).collect()
}

fn error(g: &Grid) -> String {
    let s = text(g);
    assert!(s.starts_with("#PROSPICIO! "), "not an error: {s}");
    s.trim_start_matches("#PROSPICIO! ").to_owned()
}

fn close(found: f64, expected: f64, rel: f64) -> bool {
    (found - expected).abs() <= rel * expected.abs().max(1.0)
}

fn assert_rel(found: f64, expected: f64, rel: f64, what: &str) {
    assert!(
        close(found, expected, rel),
        "{what}: {found} against {expected}"
    );
}

fn assert_all(found: &[f64], expected: &[f64], rel: f64, what: &str) {
    assert_eq!(found.len(), expected.len(), "{what}: lengths");
    for (i, (f, e)) in found.iter().zip(expected).enumerate() {
        assert_rel(*f, *e, rel, &format!("{what}, origin {}", i + 1));
    }
}

/// The chain ladder exhibit `g` is the direct fit `fit`, to the bit.
fn same_cl(g: &Grid, fit: &ChainLadderFit) {
    assert_eq!(column(g, "Ultimate"), fit.ultimate);
    assert_eq!(column(g, "Latest"), fit.latest);
    let cdf: Vec<f64> = fit.latest_position.iter().map(|&d| fit.cdf[d]).collect();
    assert_eq!(column(g, "CDF"), cdf);
    assert_eq!(total(g, "Ultimate"), fit.total_ultimate());
}

/// The expected-loss exhibit `g` is the direct fit `fit`, to the bit.
fn same_el(g: &Grid, fit: &ExpectedLossFit) {
    assert_eq!(column(g, "Ultimate"), fit.ultimate);
    assert_eq!(column(g, "Exposure"), fit.exposure);
    assert_eq!(column(g, "A priori"), fit.apriori);
    assert_eq!(total(g, "Ultimate"), fit.total_ultimate());
    same_cl_part(g, &fit.chain_ladder);
}

fn same_cl_part(g: &Grid, cl: &ChainLadderFit) {
    let cdf: Vec<f64> = cl.latest_position.iter().map(|&d| cl.cdf[d]).collect();
    assert_eq!(column(g, "CDF"), cdf);
    assert_eq!(column(g, "Latest"), cl.latest);
}

fn selected(kind: PatternKind, values: Vec<(Lag, f64)>, tail: Tail) -> ChainLadder {
    ChainLadder {
        development: Development {
            pattern: Some(SelectedPattern::new(kind, values)),
            ..Default::default()
        },
        tail,
    }
}

fn years(from: i32, n: usize) -> Vec<f64> {
    (0..n as i32).map(|k| f64::from(from + k)).collect()
}

// The hand diagonal: accident years 2019 to 2024 at 31 December 2024.

const HAND_LATEST: [f64; 6] = [980.0, 860.0, 700.0, 520.0, 400.0, 150.0];
const HAND_AGES: [f64; 6] = [72.0, 60.0, 48.0, 36.0, 24.0, 12.0];
/// To ultimate at 12, 24, ..., 72 months.
const HAND_CDF: [f64; 6] = [12.0 / 7.0, 1.3, 1.15, 1.07, 1.02, 1.0];
const DEC_2024: f64 = 45657.0;

fn hand() -> Diag {
    Diag::new(&years(2019, 6), &HAND_LATEST, &HAND_CDF)
        .kind("cdf")
        .exposure(&[200.0; 6])
}

fn hand_direct() -> (LatestDiagonal, ChainLadder) {
    let origins: Vec<Month> = (2019..=2024).map(Month::january).collect();
    let d = LatestDiagonal::new(
        Grain::Year,
        &origins,
        &HAND_AGES.map(|a| a as Lag),
        &HAND_LATEST,
    )
    .unwrap()
    .with_exposure(&[200.0; 6])
    .unwrap();
    let values = (1..=6).map(|k| 12 * k).zip(HAND_CDF).collect();
    (
        d,
        selected(PatternKind::AgeToUltimate, values, Tail::default()),
    )
}

#[test]
fn the_hand_diagonal_under_every_method() {
    let (d, cl) = hand_direct();
    for sheet in [
        hand().ages(&HAND_AGES),
        hand().valuation(DEC_2024),
        hand().ages(&HAND_AGES).grain("year"),
    ] {
        let g = sheet.cl();
        assert_eq!(
            (g.rows, g.columns),
            (8, 6),
            "{:?}",
            g.as_scalar().map(|v| format!("{v:?}"))
        );
        assert_eq!(
            labels(&g),
            years(2019, 6)
                .into_iter()
                .map(Value::Number)
                .collect::<Vec<_>>()
        );
        assert_eq!(column(&g, "Age"), HAND_AGES);
        assert_eq!(g.get(7, 0), &Value::from("Total"));
        assert_eq!(g.get(7, 1), &Value::Empty);
        assert_rel(column(&g, "Ultimate")[5], 257.142857142857, 1e-12, "CL");
        assert_rel(total(&g, "Ultimate"), 3981.342857142857, 1e-12, "CL total");
        same_cl(&g, &cl.fit_diagonal(&d).unwrap());

        let bf = sheet.bf(0.7);
        assert_eq!((bf.rows, bf.columns), (8, 8));
        assert_eq!(column(&bf, "Age"), HAND_AGES);
        assert_rel(column(&bf, "Ultimate")[5], 208.333333333333, 1e-12, "BF");
        assert_rel(
            total(&bf, "Ultimate"),
            3730.8058717501317,
            1e-12,
            "BF total",
        );
        let direct = BornhuetterFerguson {
            apriori: 0.7.into(),
            chain_ladder: cl.clone(),
        };
        same_el(&bf, &direct.fit_diagonal(&d).unwrap());

        let bk = sheet.bk(0.7, 2.0);
        assert_rel(column(&bk, "Ultimate")[5], 236.80555555555554, 1e-12, "BK");
        assert_rel(total(&bk, "Ultimate"), 3930.086950714278, 1e-12, "BK total");
        let direct = Benktander {
            apriori: 0.7.into(),
            n_iters: 2,
            chain_ladder: cl.clone(),
        };
        same_el(&bk, &direct.fit_diagonal(&d).unwrap());

        let cc = sheet.cc(0.0, 1.0);
        assert_rel(column(&cc, "A priori")[0], 3.513654937852953, 1e-12, "ELR");
        assert_rel(column(&cc, "Ultimate")[5], 442.80457815441275, 1e-12, "CC");
        assert_rel(total(&cc, "Ultimate"), 4216.385925423543, 1e-12, "CC total");
        let direct = CapeCod {
            chain_ladder: cl.clone(),
            ..Default::default()
        };
        same_el(&cc, &direct.fit_diagonal(&d).unwrap().expected_loss);

        let el = sheet.el(0.7);
        assert_all(&column(&el, "Ultimate"), &[140.0; 6], 1e-14, "EL");
        assert_rel(total(&el, "Ultimate"), 840.0, 1e-14, "EL total");
        let direct = ExpectedLoss {
            apriori: 0.7.into(),
            chain_ladder: cl.clone(),
        };
        same_el(&el, &direct.fit_diagonal(&d).unwrap());
    }
    // Percents developed, 1 / cdf, give the chain ladder's ultimates.
    let percent = Diag::new(&years(2019, 6), &HAND_LATEST, &HAND_CDF.map(|c| 1.0 / c))
        .kind("percent")
        .ages(&HAND_AGES);
    assert_all(
        &column(&percent.cl(), "Ultimate"),
        &column(&hand().ages(&HAND_AGES).cl(), "Ultimate"),
        1e-14,
        "percent",
    );
    // The input order is free; the exhibit lists the origins oldest first.
    let mut reversed = hand();
    let rev = |v: &[f64]| v.iter().rev().copied().collect::<Vec<_>>();
    reversed.origins = col(&rev(&years(2019, 6)));
    reversed.ages = col(&rev(&HAND_AGES));
    reversed.latest = col(&rev(&HAND_LATEST));
    assert_eq!(
        column(&reversed.cl(), "Ultimate"),
        column(&hand().ages(&HAND_AGES).cl(), "Ultimate")
    );
}

/// The reference CSV's rows of `dataset`, keyed by method: the pattern
/// (age, value), each origin's ultimate and the total.
type Cases = BTreeMap<String, (Vec<(f64, f64)>, Vec<(f64, f64)>, f64)>;

fn parity_cases(dataset: &str) -> Cases {
    let mut cases = Cases::new();
    for r in reference("reserving_latest_diagonal_python.csv") {
        if r[0] != dataset {
            continue;
        }
        let entry = cases.entry(r[1].clone()).or_default();
        let expected: f64 = r[4].parse().unwrap();
        match r[2].as_str() {
            "pattern" => entry.0.push((r[3].parse().unwrap(), expected)),
            "ultimate" => entry.1.push((r[3].parse().unwrap(), expected)),
            "total_ultimate" => entry.2 = expected,
            other => panic!("unknown quantity {other}"),
        }
    }
    cases
}

/// A method spec's numeric setting: `apriori=0.6` in
/// `bornhuetter_ferguson;apriori=0.6;style=cdf`.
fn param(spec: &str, name: &str) -> f64 {
    spec.split(';')
        .find_map(|p| p.strip_prefix(&format!("{name}=")))
        .unwrap_or_else(|| panic!("{spec} has no {name}"))
        .parse()
        .unwrap()
}

fn check_parity(g: &Grid, ultimates: &[(f64, f64)], total_ultimate: f64, what: &str) {
    let found = column(g, "Ultimate");
    let origins: Vec<f64> = labels(g).iter().map(num).collect();
    assert_eq!(found.len(), ultimates.len(), "{what}");
    for &(year, expected) in ultimates {
        let o = origins.iter().position(|&y| y == year).unwrap();
        assert_rel(found[o], expected, 1e-9, &format!("{what} {year}"));
    }
    assert_rel(total(g, "Ultimate"), total_ultimate, 1e-9, what);
}

#[test]
fn genins_matches_chainladder_python() {
    let (tri, _) = genins();
    let latest: Vec<f64> = (0..tri.rows)
        .map(|r| {
            (0..tri.columns)
                .rev()
                .find_map(|c| match tri.get(r, c) {
                    Value::Number(x) => Some(*x),
                    _ => None,
                })
                .unwrap()
        })
        .collect();
    let cases = parity_cases("genins");
    assert_eq!(cases.len(), 2);
    for (method, (pattern, ultimates, total_ultimate)) in &cases {
        let kind = if method.contains("style=ldf") {
            "ldf"
        } else {
            "cdf"
        };
        let values: Vec<f64> = pattern.iter().map(|p| p.1).collect();
        let ages: Vec<f64> = pattern.iter().map(|p| p.0).collect();
        let sheet = Diag::new(&years(2001, 10), &latest, &values)
            .pattern_ages(&ages)
            .kind(kind)
            .valuation(40543.0);
        let g = sheet.cl();
        check_parity(&g, ultimates, *total_ultimate, method);
        assert_rel(total(&g, "Ultimate"), 53038945.61192431, 1e-9, method);
        // The default ages, 12, 24, ..., are the pattern's here.
        let mut by_cell = sheet.clone();
        by_cell.pattern_ages = Grid::missing();
        assert_eq!(by_cell.cl(), g);
    }
}

#[test]
fn clrd_wkcomp_matches_chainladder_python() {
    let (paid, premium) = clrd_wkcomp();
    let cases = parity_cases("clrd_wkcomp");
    assert_eq!(cases.len(), 3);
    for (method, (pattern, ultimates, total_ultimate)) in &cases {
        assert!(method.contains("style=cdf"));
        let values: Vec<f64> = pattern.iter().map(|p| p.1).collect();
        let ages: Vec<f64> = pattern.iter().map(|p| p.0).collect();
        let sheet = Diag::new(&years(1988, 10), &paid, &values)
            .pattern_ages(&ages)
            .kind("cdf")
            .valuation(35795.0)
            .exposure(&premium);
        let g = match method.split(';').next().unwrap() {
            "bornhuetter_ferguson" => sheet.bf(param(method, "apriori")),
            "benktander" => sheet.bk(param(method, "apriori"), param(method, "n_iters")),
            "cape_cod" => sheet.cc(param(method, "trend"), param(method, "decay")),
            other => panic!("unknown method {other}"),
        };
        check_parity(&g, ultimates, *total_ultimate, method);
    }
    let total_of = |m: &str| cases.iter().find(|c| c.0.starts_with(m)).unwrap().1.2;
    assert_rel(total_of("bornhuetter"), 13865202.236379728, 1e-12, "BF");
    assert_rel(total_of("benktander"), 13790662.769758541, 1e-12, "BK");
    assert_rel(total_of("cape_cod"), 13990057.87828532, 1e-12, "CC");
}

/// Factors to ultimate of 2, 1.25 and 1 at 12, 24 and 36 months.
const ANNUAL_CDF: [f64; 3] = [2.0, 1.25, 1.0];

#[test]
fn ages_off_the_patterns_grid_are_read_log_linearly() {
    // Half years at December 2024: 24, 18 and 12 months.
    let half = Diag::new(
        &[44927.0, 45108.0, 45292.0],
        &[500.0, 300.0, 100.0],
        &ANNUAL_CDF,
    )
    .kind("cdf")
    .grain("semiannual")
    .valuation(DEC_2024)
    .exposure(&[800.0; 3]);
    let cdf18 = (2.0f64 * 1.25).sqrt();
    let g = half.cl();
    assert_eq!(
        labels(&g),
        ["2023H1", "2023H2", "2024H1"].map(Value::from).to_vec()
    );
    assert_eq!(column(&g, "Age"), [24.0, 18.0, 12.0]);
    let cl_ultimates = [625.0, 300.0 * cdf18, 200.0];
    assert_all(&column(&g, "Ultimate"), &cl_ultimates, 1e-12, "half CL");
    let bf = half.bf(0.5);
    assert_all(
        &column(&bf, "Ultimate"),
        &[
            500.0 + 400.0 * (1.0 - 1.0 / 1.25),
            300.0 + 400.0 * (1.0 - 1.0 / cdf18),
            100.0 + 400.0 * 0.5,
        ],
        1e-12,
        "half BF",
    );
    let origins = [(2023, 1), (2023, 7), (2024, 1)].map(|(y, m)| Month::new(y, m).unwrap());
    let d = LatestDiagonal::at_valuation(
        Grain::Semester,
        &origins,
        Month::new(2024, 12).unwrap(),
        &[500.0, 300.0, 100.0],
    )
    .unwrap();
    let direct = selected(
        PatternKind::AgeToUltimate,
        vec![(12, 2.0), (24, 1.25), (36, 1.0)],
        Tail::default(),
    );
    same_cl(&g, &direct.fit_diagonal(&d).unwrap());
    // "semi-annual" is the same grain, in any case; "half" and
    // "semester" are not grain words (the user's decision of 2026-10-24).
    for word in ["semi-annual", "SemiAnnual"] {
        same_cl(
            &half.clone().grain(word).cl(),
            &direct.fit_diagonal(&d).unwrap(),
        );
    }
    for word in ["half", "semester"] {
        assert_eq!(
            error(&half.clone().grain(word).cl()),
            format!(
                "grain must be \"year\", \"semiannual\", \"quarter\" or \"month\", got {word:?}"
            )
        );
    }
    // Age-to-age factors 1.6 and 1.25 give the same ultimates, and a tail
    // of 1.05 multiplies them.
    let mut ldf = half.clone();
    ldf.pattern = row(&[1.6, 1.25]);
    ldf.kind = Grid::missing();
    assert_all(
        &column(&ldf.cl(), "Ultimate"),
        &cl_ultimates,
        1e-12,
        "half ldf",
    );
    let tailed = ldf.tail(n(1.05)).cl();
    assert_all(
        &column(&tailed, "Ultimate"),
        &cl_ultimates.map(|u| u * 1.05),
        1e-12,
        "half ldf tail",
    );

    // Accident quarters of 2024 at December 2024: 12, 9, 6 and 3 months.
    let quarters = Diag::new(
        &[45292.0, 45383.0, 45474.0, 45566.0],
        &[400.0, 300.0, 200.0, 100.0],
        &[4.0, 2.0, 1.25, 1.0],
    )
    .pattern_ages(&[3.0, 12.0, 24.0, 36.0])
    .kind("cdf")
    .grain("quarter")
    .valuation(DEC_2024);
    let g = quarters.cl();
    assert_eq!(
        labels(&g),
        ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
            .map(Value::from)
            .to_vec()
    );
    let b = |a: f64| 4.0 * 0.5f64.powf((a - 3.0) / 9.0);
    assert_all(
        &column(&g, "Ultimate"),
        &[800.0, 300.0 * b(9.0), 200.0 * b(6.0), 400.0],
        1e-12,
        "quarters",
    );

    // Accident years at January 2025: 25 and 13 months.
    let january = Diag::new(&[2023.0, 2024.0], &[500.0, 200.0], &ANNUAL_CDF)
        .kind("cdf")
        .valuation(45658.0);
    let g = january.cl();
    assert_eq!(column(&g, "Age"), [25.0, 13.0]);
    assert_all(
        &column(&g, "Ultimate"),
        &[
            500.0 * 1.25 * 0.8f64.powf(1.0 / 12.0),
            200.0 * 2.0 * 0.625f64.powf(1.0 / 12.0),
        ],
        1e-12,
        "January",
    );
    // An Excel date for an accident year is floored to its year.
    let mut dated = january.clone();
    dated.origins = col(&[44927.0 + 40.0, 45292.0 + 200.0]);
    assert_eq!(dated.cl(), g);
}

/// The Excel serial of the first of month `m` of year `y`.
fn serial(y: i32, m: u32) -> f64 {
    let (y, m) = (i64::from(y) - i64::from(m <= 2), i64::from(m));
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146_097 + doe - 719_468 + 25_569) as f64
}

/// Nine age-to-age factors of 1.1 at 12 to 108 months, ending at 120.
const TO_120: [f64; 9] = [1.1; 9];

/// Accident years from 2000 at `ages`, each latest value 100.
fn past(ages: &[f64]) -> Diag {
    Diag::new(&years(2000, ages.len()), &vec![100.0; ages.len()], &TO_120).ages(ages)
}

fn cdfs(g: &Grid) -> Vec<f64> {
    column(g, "CDF")
}

#[test]
fn excel_serials_of_month_starts() {
    assert_eq!(serial(2024, 1), 45292.0);
    assert_eq!(serial(2024, 10), 45566.0);
    assert_eq!(serial(2000, 1), 36526.0);
}

#[test]
fn origins_past_the_patterns_end() {
    let g = past(&[144.0, 132.0, 120.0, 108.0, 12.0])
        .tail(n(1.05))
        .past_end("tail")
        .cl();
    let c = cdfs(&g);
    assert_rel(c[0], 1.0132158418990078, 1e-14, "144");
    assert_rel(c[1], 1.0253798721500054, 1e-14, "132");
    assert_rel(c[2], 1.05, 1e-14, "120");
    assert_rel(c[3], 1.155, 1e-14, "108");
    let d = LatestDiagonal::new(
        Grain::Year,
        &(2000..2005).map(Month::january).collect::<Vec<_>>(),
        &[144, 132, 120, 108, 12],
        &[100.0; 5],
    )
    .unwrap()
    .with_past_pattern_end(PastPatternEnd::Tail);
    let ldf: Vec<(Lag, f64)> = (0..9).map(|k| (12 + 12 * k, 1.1)).collect();
    let direct = selected(PatternKind::AgeToAge, ldf.clone(), Tail::from(1.05));
    same_cl(&g, &direct.fit_diagonal(&d).unwrap());

    // Accident quarters at 123 months: one origin period past the end.
    let quarterly = Diag::new(
        &(0..41)
            .map(|k| serial(2000 + k / 4, 1 + 3 * (k % 4) as u32))
            .collect::<Vec<_>>(),
        &[100.0; 41],
        &[1.01; 39],
    )
    .pattern_ages(&(1..40).map(|k| f64::from(3 * k)).collect::<Vec<_>>())
    .ages(&(0..41).map(|k| f64::from(123 - 3 * k)).collect::<Vec<_>>())
    .grain("quarter")
    .tail(n(1.05))
    .past_end("tail");
    let g = quarterly.cl();
    assert_eq!(labels(&g)[0], Value::from("2000Q1"));
    assert_eq!(labels(&g)[40], Value::from("2010Q1"));
    assert_rel(cdfs(&g)[0], 1.0253798721500054, 1e-14, "quarter at 123");
    // Accident years valued at March, the oldest at 123 months.
    let march = past(&[123.0, 111.0, 99.0, 87.0, 75.0, 63.0, 51.0, 39.0, 27.0, 15.0])
        .tail(n(1.05))
        .past_end("tail")
        .cl();
    assert_rel(cdfs(&march)[0], 1.0437900950394003, 1e-14, "March at 123");

    // No development past the end.
    let none = past(&[132.0, 120.0, 108.0, 12.0]).past_end("none").cl();
    assert_eq!(column(&none, "Ultimate")[0], column(&none, "Latest")[0]);
    assert_rel(cdfs(&none)[2], 1.1, 1e-15, "108 under none");
    // A level of 95% at the end is rescaled to 100% under "none".
    let level = |rule: Option<&str>| {
        let mut d = Diag::new(&years(2000, 3), &[100.0; 3], &[0.4, 0.8, 0.95])
            .pattern_ages(&[12.0, 60.0, 120.0])
            .kind("percent")
            .ages(&[120.0, 108.0, 12.0]);
        if let Some(r) = rule {
            d = d.past_end(r);
        }
        column(&d.cl(), "Ultimate")[2]
    };
    assert_rel(level(Some("none")), 237.5, 1e-14, "95% under none");
    assert_rel(
        level(Some("no_development")),
        237.5,
        1e-14,
        "no_development",
    );
    assert_rel(level(None), 250.0, 1e-14, "95% under reject");
    assert_rel(level(Some("reject")), 250.0, 1e-14, "95% under reject");

    // The spread's floor at decay 0.5 is exp(-0.75) = 0.47237.
    let floor = |f: f64| past(&[132.0, 120.0, 12.0]).tail(n(f)).past_end("tail").cl();
    assert!(
        error(&floor(0.4723)).starts_with("tail"),
        "{:?}",
        floor(0.4723)
    );
    assert_rel(cdfs(&floor(0.4724))[1], 0.4724, 1e-14, "above the floor");

    // The expected-loss methods take the spread too.
    let el = Diag::new(&years(2019, 6), &HAND_LATEST, &TO_120)
        .ages(&[132.0, 108.0, 84.0, 60.0, 36.0, 12.0])
        .exposure(&[1000.0; 6])
        .tail(n(1.05))
        .past_end("tail");
    let bf = el.bf(0.7);
    assert_rel(
        column(&bf, "Ultimate")[0],
        980.0 + 700.0 * (1.0 - 1.0 / 1.0253798721500054),
        1e-14,
        "BF at 132",
    );
    let d = LatestDiagonal::new(
        Grain::Year,
        &(2019..=2024).map(Month::january).collect::<Vec<_>>(),
        &[132, 108, 84, 60, 36, 12],
        &HAND_LATEST,
    )
    .unwrap()
    .with_exposure(&[1000.0; 6])
    .unwrap()
    .with_past_pattern_end(PastPatternEnd::Tail);
    let chain_ladder = selected(PatternKind::AgeToAge, ldf, Tail::from(1.05));
    let direct = BornhuetterFerguson {
        apriori: 0.7.into(),
        chain_ladder: chain_ladder.clone(),
    };
    same_el(&bf, &direct.fit_diagonal(&d).unwrap());
    let direct = Benktander {
        apriori: 0.7.into(),
        n_iters: 2,
        chain_ladder: chain_ladder.clone(),
    };
    same_el(&el.bk(0.7, 2.0), &direct.fit_diagonal(&d).unwrap());
    let direct = CapeCod {
        trend: 0.02,
        decay: 0.75,
        chain_ladder,
    };
    same_el(
        &el.cc(0.02, 0.75),
        &direct.fit_diagonal(&d).unwrap().expected_loss,
    );
}

#[test]
fn the_past_end_errors_name_the_sheets_arguments() {
    let e = error(&past(&[132.0, 120.0, 108.0]).cl());
    assert_eq!(
        e,
        "latest diagonal: origin 2000 is at 132 months, outside the pattern's ages from 12 to 120 \
         months: an origin older than the pattern's end needs past_end \"none\" (a factor of 1) or \
         \"tail\" with a tail (a factor, or \"exponential\", \"inverse_power\" or \"bondy\" fitted \
         on the pattern's own factors)"
    );
    // The crate's own text, which the add-in rewrites from its ending: a
    // change in the reserving crate fails here on purpose.
    let d = LatestDiagonal::new(
        Grain::Year,
        &[2000, 2001, 2002].map(Month::january),
        &[132, 120, 108],
        &[100.0; 3],
    )
    .unwrap();
    let ldf: Vec<(Lag, f64)> = (0..9).map(|k| (12 + 12 * k, 1.1)).collect();
    let crate_text = selected(PatternKind::AgeToAge, ldf, Tail::default())
        .fit_diagonal(&d)
        .unwrap_err()
        .to_string();
    assert_eq!(
        crate_text,
        "latest diagonal: origin 2000 is at 132 months, outside the pattern's ages from 12 to 120 \
         months: an origin older than the pattern's end needs \
         LatestDiagonal::with_past_pattern_end with PastPatternEnd::NoDevelopment (a factor of 1), \
         or PastPatternEnd::Tail with an a-priori tail (a TailConstant) or a tail estimated on the \
         pattern's own factors (TailCurve, TailBondy, Tail::LogLinear)"
    );
    // A younger origin's error is the crate's, unchanged.
    let young = past(&[24.0, 6.0]).tail(n(1.05)).past_end("tail");
    assert_eq!(
        error(&young.cl()),
        "latest diagonal: origin 2001 is at 6 months, outside the pattern's ages from 12 to 120 months"
    );
    // The two settings.
    for tail in [n(1.05), t("exponential")] {
        assert_eq!(
            error(&past(&[120.0, 12.0]).tail(tail).past_end("none").cl()),
            "past_end \"none\" is a factor of 1 past the pattern's end: leave tail empty, or use \
             past_end \"tail\""
        );
    }
    for tail in [Grid::missing(), n(1.0), t("none")] {
        assert_eq!(
            error(&past(&[120.0, 12.0]).tail(tail).past_end("tail").cl()),
            "past_end \"tail\" needs a tail: for a factor of 1 use past_end \"none\""
        );
    }
    // A tail of 1 under "none" is no tail, as in the crate.
    let none = past(&[132.0, 12.0]).tail(n(1.0)).past_end("none").cl();
    assert_eq!(column(&none, "CDF")[0], 1.0);
}

/// t/12 at 1 to 12 months: an accident month reports evenly over a year.
fn linear_month() -> (Vec<f64>, Vec<f64>) {
    (
        (1..=12).map(f64::from).collect(),
        (1..=12).map(|k| f64::from(k) / 12.0).collect(),
    )
}

#[test]
fn a_pattern_averages_over_origin_months() {
    let (ages, values) = linear_month();
    let g = call(
        "pro_pattern_average",
        vec![
            row(&ages),
            row(&values),
            t("month"),
            t("year"),
            t("percent"),
        ],
    );
    assert_eq!(
        (0..3).map(|c| g.get(0, c).clone()).collect::<Vec<_>>(),
        ["Age", "Percent developed", "CDF"]
            .map(Value::from)
            .to_vec()
    );
    let at = |age: f64| {
        let r = (1..g.rows).find(|&r| num(g.get(r, 0)) == age).unwrap();
        (num(g.get(r, 1)), num(g.get(r, 2)))
    };
    assert_rel(at(12.0).0, 78.0 / 144.0, 1e-15, "at 12");
    assert_rel(at(12.0).1, 144.0 / 78.0, 1e-15, "CDF at 12");
    assert_rel(at(6.0).0, 21.0 / 144.0, 1e-15, "at 6");
    assert_eq!(num(g.get(g.rows - 1, 0)), 23.0);
    assert_eq!(num(g.get(g.rows - 1, 1)), 1.0);
    // The grain defaults to the year.
    assert_eq!(
        call(
            "pro_pattern_average",
            vec![
                row(&ages),
                row(&values),
                t("month"),
                Grid::missing(),
                t("percent")
            ]
        ),
        g
    );
    let direct = SelectedPattern::new(
        PatternKind::PercentDeveloped,
        (1..=12).map(|k| (k, f64::from(k) / 12.0)).collect(),
    )
    .averaged_over_origins(Grain::Month, Grain::Year)
    .unwrap();
    let sheet: Vec<f64> = (1..g.rows).map(|r| num(g.get(r, 1))).collect();
    let crate_values: Vec<f64> = direct.values.iter().map(|v| v.1).collect();
    assert_eq!(sheet, crate_values);
}

#[test]
fn a_monthly_pattern_reaches_accident_years() {
    // An accident month's 1 - 0.9^t at 1 to 60 months; accident years
    // 2015 to 2020 valued at 30 June 2020, each month's ultimate 100.
    let p = |t: i64| 1.0 - 0.9f64.powi(t.clamp(0, 60) as i32);
    let june = Month::new(2020, 6).unwrap();
    let origins: Vec<Month> = (2015..=2020).map(Month::january).collect();
    let latest: Vec<f64> = origins
        .iter()
        .map(|&o| {
            (0..12)
                .map(|j: i64| 100.0 * p(june.months_since(o.add_months(j)) + 1))
                .sum()
        })
        .collect();
    let ages: Vec<f64> = (1..=60).map(f64::from).collect();
    let values: Vec<f64> = (1..=60).map(|t| p(i64::from(t))).collect();
    let sheet = Diag::new(&years(2015, 6), &latest, &values)
        .pattern_ages(&ages)
        .kind("percent")
        .valuation(44012.0)
        .pattern_origin("month");
    let g = sheet.cl();
    assert_eq!(column(&g, "Age"), [66.0, 54.0, 42.0, 30.0, 18.0, 6.0]);
    assert_all(&column(&g, "Ultimate"), &[1200.0; 6], 1e-12, "June");
    let averaged = SelectedPattern::new(
        PatternKind::PercentDeveloped,
        (1..=60).map(|t| (t, p(i64::from(t)))).collect(),
    )
    .averaged_over_origins(Grain::Month, Grain::Year)
    .unwrap();
    let d = LatestDiagonal::at_valuation(Grain::Year, &origins, june, &latest).unwrap();
    let direct = ChainLadder {
        development: Development {
            pattern: Some(averaged),
            ..Default::default()
        },
        tail: Tail::default(),
    };
    same_cl(&g, &direct.fit_diagonal(&d).unwrap());
    // PRO.PATTERN.AVERAGE's output fed back as percents developed gives the
    // same exhibit, to the bit.
    let avg = call(
        "pro_pattern_average",
        vec![
            row(&ages),
            row(&values),
            t("month"),
            t("year"),
            t("percent"),
        ],
    );
    let fed_ages: Vec<f64> = (1..avg.rows).map(|r| num(avg.get(r, 0))).collect();
    let fed: Vec<f64> = (1..avg.rows).map(|r| num(avg.get(r, 1))).collect();
    let mut fed_back = sheet.clone().pattern_ages(&fed_ages);
    fed_back.pattern = row(&fed);
    fed_back.pattern_origin = Grid::missing();
    assert_eq!(fed_back.cl(), g);
    // The CDF column fed back as factors to ultimate gives the same
    // ultimates within rounding.
    let cdf: Vec<f64> = (1..avg.rows).map(|r| num(avg.get(r, 2))).collect();
    let mut by_cdf = fed_back.clone().kind("cdf");
    by_cdf.pattern = row(&cdf);
    assert_all(
        &column(&by_cdf.cl(), "Ultimate"),
        &[1200.0; 6],
        1e-12,
        "by cdf",
    );
}

/// `PRO.CL.LDF`'s row on `tri` with `tail`.
fn ldf_row(tri: &Grid, tail: &Grid) -> Vec<f64> {
    let g = call(
        "pro_cl_ldf",
        vec![tri.clone(), Grid::missing(), tail.clone()],
    );
    (0..g.columns).map(|c| num(g.get(0, c))).collect()
}

#[test]
fn a_triangles_pattern_frozen_on_the_sheet_reproduces_its_chain_ladder() {
    let (genins_tri, _) = genins();
    for (tri, valuation) in [(raa(), 40543.0), (genins_tri, 40543.0)] {
        let latest_g = call("pro_tri_latest", vec![tri.clone()]);
        let latest: Vec<f64> = latest_g.cells.iter().map(num).collect();
        for (tail, direct_tail, rel) in [
            (Grid::missing(), Tail::default(), 0.0),
            (n(1.05), Tail::from(1.05), 1e-14),
            (t("exponential"), Tail::from(TailCurve::default()), 1e-14),
        ] {
            let ldf = ldf_row(&tri, &tail);
            let k = ldf.len() - 1;
            let cl = call("pro_cl", vec![tri.clone(), Grid::missing(), tail.clone()]);
            let cl_ult = column(&cl, "Ultimate");
            // Exact: the factors at 12 to 108 months and the tail factor.
            let exact = Diag::new(&years(2001, 10), &latest, &ldf[..k])
                .pattern_ages(&(1..=k).map(|j| 12.0 * j as f64).collect::<Vec<_>>())
                .valuation(valuation)
                .tail(n(ldf[k]));
            let g = exact.cl();
            if rel == 0.0 {
                assert_eq!(column(&g, "Ultimate"), cl_ult);
            } else {
                assert_all(&column(&g, "Ultimate"), &cl_ult, rel, "frozen");
            }
            // ChainLadderFit::to_chain_ladder on the triangle's diagonal.
            let triangle = prospicio_xll::reserving::triangle(&tri).unwrap();
            let tri_fit = ChainLadder {
                development: Development::default(),
                tail: direct_tail,
            }
            .fit(&triangle, "values")
            .unwrap();
            let d = LatestDiagonal::from_triangle(&triangle, "values", None).unwrap();
            let frozen = tri_fit.to_chain_ladder().fit_diagonal(&d).unwrap();
            same_cl(&g, &frozen);
            // Simpler: the whole row at 12 to 120 months and no tail.
            let simple = Diag::new(&years(2001, 10), &latest, &ldf).valuation(valuation);
            assert_all(
                &column(&simple.cl(), "Ultimate"),
                &cl_ult,
                1e-14,
                "the whole row",
            );
        }
    }
}

#[test]
fn input_errors_name_their_argument() {
    let base = || hand().ages(&HAND_AGES);
    let e = |g: Grid| error(&g);
    let mut short = base();
    short.latest = col(&HAND_LATEST[..5]);
    assert_eq!(
        e(short.cl()),
        "latest must have 6 values, one per origin, got 5"
    );
    let mut short = base();
    short.ages = col(&HAND_AGES[..5]);
    assert_eq!(
        e(short.cl()),
        "ages must have 6 values, one per origin, got 5"
    );
    assert_eq!(
        e(base().exposure(&[200.0; 5]).bf(0.7)),
        "exposure must have 6 values, one per origin, got 5"
    );
    assert_eq!(
        e(base().ages(&[72.0, 60.0, 48.0, 36.0, 24.0, 0.0]).cl()),
        "ages 6 must be a whole number of months from 1, got 0"
    );
    assert_eq!(
        e(hand().valuation(45291.0).cl()),
        "origin 6 starts after the valuation"
    );
    let mut fraction = base();
    fraction.origins = col(&[2019.0, 2020.0, 2021.0, 2022.0, 2023.0, 2024.5]);
    assert_eq!(
        e(fraction.cl()),
        "origins 6 must be a year or an Excel date, got 2024.5"
    );
    let mut blank = base();
    blank.latest = Grid::from_rows(
        [
            Value::Number(1.0),
            Value::Empty,
            Value::Number(1.0),
            Value::Number(1.0),
            Value::Number(1.0),
            Value::Number(1.0),
        ]
        .map(|v| vec![v])
        .to_vec(),
    );
    assert_eq!(
        e(blank.cl()),
        "latest 2 is blank: give 0 for an origin with nothing reported"
    );
    let mut no_age = base();
    no_age.pattern_ages = Grid::from_rows(vec![vec![
        Value::Number(12.0),
        Value::Number(24.0),
        Value::Empty,
        Value::Number(48.0),
        Value::Number(60.0),
        Value::Number(72.0),
    ]]);
    assert_eq!(
        e(no_age.cl()),
        "pattern_ages 3 must be a whole number of months from 1, got a blank"
    );
    // A pair with a blank value is skipped, its age with it.
    let mut skipped = base();
    skipped.pattern = Grid::from_rows(vec![vec![
        Value::Number(12.0 / 7.0),
        Value::Number(1.3),
        Value::Number(1.15),
        Value::Number(1.07),
        Value::Number(1.02),
        Value::Empty,
        Value::Empty,
    ]]);
    skipped.pattern_ages = row(&[12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]);
    // 72 months is past the pattern's end now.
    assert!(e(skipped.cl()).contains("outside the pattern's ages from 12 to 60 months"));
    assert_eq!(
        e(base()
            .exposure(&[200.0, 200.0, 0.0, 200.0, 200.0, 200.0])
            .bf(0.7)),
        "exposure 3 must be a positive number, got 0"
    );
    let mut no_exposure = base();
    no_exposure.exposure = Grid::missing();
    assert!(e(no_exposure.bf(0.7)).starts_with("exposure is required"));
    assert_eq!(
        e(base().grain("week").cl()),
        "grain must be \"year\", \"semiannual\", \"quarter\" or \"month\", got \"week\""
    );
    assert_eq!(
        e(base().past_end("extend").cl()),
        "past_end must be \"reject\", \"none\" or \"tail\", got \"extend\""
    );
    assert_eq!(
        e(base().kind("factor").cl()),
        "kind must be \"ldf\", \"cdf\" or \"percent\", got \"factor\""
    );
    assert_eq!(
        e(base().pattern_origin("day").cl()),
        "pattern_origin must be \"year\", \"semiannual\", \"quarter\" or \"month\", got \"day\""
    );
    // An origin grain that is not a whole multiple of the pattern's.
    assert_eq!(
        e(base().grain("quarter").pattern_origin("semiannual").cl()),
        "selected pattern: the origin grain must be a whole multiple of the pattern's origin grain"
    );
    let mut twice = base();
    twice.origins = col(&[2019.0, 2020.0, 2021.0, 2022.0, 2024.0, 2024.0]);
    assert_eq!(e(twice.cl()), "latest diagonal: origin 2024 is given twice");
    let skip = Diag::new(&[2022.0, 2024.0], &[1.0, 1.0], &ANNUAL_CDF)
        .kind("cdf")
        .ages(&[24.0, 12.0]);
    assert_eq!(e(skip.cl()), "origin 2023 has no observed values");
    let young = Diag::new(&[2024.0], &[1.0], &ANNUAL_CDF)
        .kind("cdf")
        .pattern_ages(&[24.0, 36.0, 48.0])
        .ages(&[12.0]);
    assert_eq!(
        e(young.cl()),
        "latest diagonal: origin 2024 is at 12 months, outside the pattern's ages from 24 to 48 months"
    );
    assert_eq!(
        e(base().valuation(DEC_2024).cl()),
        "give ages or valuation, not both"
    );
    let mut neither = base();
    neither.ages = Grid::missing();
    assert!(e(neither.cl()).starts_with("ages or valuation is required"));
    assert!(e(hand().valuation(2024.0).cl()).starts_with("valuation must be one Excel date"));
    let mut no_pattern = base();
    no_pattern.pattern = Grid::missing();
    assert!(e(no_pattern.cl()).starts_with("pattern is required"));
    assert_eq!(
        e(call(
            "pro_pattern_average",
            vec![
                row(&[1.0, 2.0]),
                row(&[0.5, 1.0]),
                t("year"),
                t("year"),
                t("percent")
            ]
        )),
        "pattern_origin and grain are the same: the pattern is used as given, with nothing to average"
    );
}

/// Under kind "ldf" an age-to-age factor runs from its age to the next age
/// given (`reserving-v02.md` decisions 16 and 18 item 4), so an interior
/// blank widens the factor before it: 1.5, blank, 1.2 at 12, 24, 36 months
/// is 1.5 from 12 to 36, the same as ages {12, 36}, not a factor of 1 from
/// 24 to 36 (which the pattern 1.5, 1, 1.2 gives).
#[test]
fn an_interior_blank_widens_the_age_to_age_factor_before_it() {
    let base = Diag::new(&years(2022, 3), &[100.0; 3], &[1.0]).ages(&[36.0, 24.0, 12.0]);
    let mut blank = base.clone();
    blank.pattern = Grid::from_rows(vec![vec![
        Value::Number(1.5),
        Value::Empty,
        Value::Number(1.2),
    ]]);
    let mut sparse = base.clone();
    sparse.pattern = row(&[1.5, 1.2]);
    sparse.pattern_ages = row(&[12.0, 36.0]);
    let mut unit = base;
    unit.pattern = row(&[1.5, 1.0, 1.2]);
    let (blank, sparse, unit) = (blank.cl(), sparse.cl(), unit.cl());
    assert_eq!(blank, sparse);
    let (b, u) = (cdfs(&blank), cdfs(&unit));
    // 36 months: 1.2 to ultimate either way; 12 months: 1.5 x 1.2.
    assert_rel(b[0], 1.2, 1e-12, "36 months");
    assert_rel(b[2], 1.8, 1e-12, "12 months");
    assert_rel(u[2], 1.8, 1e-12, "12 months, factor of 1");
    // 24 months: inside the widened 12-to-36 factor, so above 1.2; with a
    // factor of 1 from 24 to 36 it is 1.2.
    assert_rel(u[1], 1.2, 1e-12, "24 months, factor of 1");
    assert!(b[1] > 1.2 + 1e-6, "24 months, widened: {}", b[1]);
}
