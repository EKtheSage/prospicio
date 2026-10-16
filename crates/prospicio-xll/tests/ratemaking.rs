//! Ratemaking, part one (`PRO.YEARFRACTION`, `PRO.ONLEVEL`,
//! `PRO.RATELEVEL`, `PRO.AVGDATE`, `PRO.TREND.FIT`), reached through the
//! catalogue as Excel reaches it, against Werner and Modlin, *Basic
//! Ratemaking* (5th ed., CAS, 2016): the parallelogram method of chapter
//! 5 and Appendices A and C (shares, average levels and on-level factors,
//! with the books' printed levels), the trend lengths of chapters 5 and 6,
//! and Table 6.14's exponential trends.

mod common;

use common::{n, num, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn column(v: &[f64]) -> Grid {
    Grid::from_rows(v.iter().map(|&x| vec![Value::Number(x)]).collect())
}

/// Within half a unit of the last printed digit.
fn printed(x: f64, book: f64, dp: i32) {
    let tol = 0.5 * 10f64.powi(-dp) + 1e-12;
    assert!((x - book).abs() <= tol, "{x} is not {book} at {dp} dp");
}

fn round(x: f64, dp: i32) -> f64 {
    let s = 10f64.powi(dp);
    (x * s).round() / s
}

/// Excel dates: 1 July 2010, 1 January 2011, 1 April 2012.
const CHAPTER_5_DATES: [f64; 3] = [40_360.0, 40_544.0, 40_995.0];

#[test]
fn excel_dates_on_the_half_month_grid() {
    // 30 June 2013, 15 November 2011, 31 December 2015, 1 July 2010.
    let d = Grid::from_rows(vec![vec![
        Value::Number(41_455.0),
        Value::Number(40_862.0),
        Value::Number(42_369.0),
        Value::Number(40_360.0),
    ]]);
    let y = call("pro_yearfraction", vec![d]);
    let want = [2013.5, 2011.875, 2016.0, 2010.5];
    for (k, w) in want.iter().enumerate() {
        assert_eq!(num(&y.cells[k]), *w);
    }
    assert!(text(&call("pro_yearfraction", vec![n(2011.5)])).contains("Excel date"));
}

#[test]
fn parallelogram_chapter_5() {
    // +5% on 7/1/10, +10% on 1/1/11, -1% on 4/1/12, given as Excel dates.
    let eff = column(&CHAPTER_5_DATES);
    let changes = column(&[0.05, 0.10, -0.01]);
    let ol = |from: f64, to: f64, period: &str, term: f64| {
        num(call(
            "pro_onlevel",
            vec![
                eff.clone(),
                changes.clone(),
                n(from),
                n(to),
                t(period),
                n(term),
            ],
        )
        .as_scalar()
        .unwrap())
    };
    // Figures 5.17 to 5.21, the exact chain of changes.
    assert!((ol(2011.0, 2012.0, "calendar", 1.0) - 1.0430559).abs() < 1e-7);
    assert!((ol(2011.0, 2012.0, "calendar", 0.5) - 1.0130233).abs() < 1e-7);
    assert!((ol(2012.0, 2013.0, "policy", 1.0) - 0.9974811).abs() < 1e-7);

    // Calendar year 2011's cells: 1/8 at 1.05, 3/8 at 1.155, 1/2 ... the
    // shares 0.125, 0.375, 0.5 and the average level 1.09625.
    let rl = call(
        "pro_ratelevel",
        vec![eff.clone(), changes.clone(), n(2011.0), n(2012.0)],
    );
    assert_eq!((rl.rows, rl.columns), (7, 4), "{:?}", rl.as_scalar());
    for (k, s) in [0.125, 0.375, 0.5].iter().enumerate() {
        assert!((num(rl.get(k + 1, 3)) - s).abs() < 1e-14);
    }
    assert_eq!(rl.get(4, 0), &Value::from("Average level"));
    assert!((num(rl.get(4, 2)) - 1.09625).abs() < 1e-12);
    assert!((num(rl.get(6, 2)) - 1.0430559).abs() < 1e-7);

    // A column of calendar years against the printed levels (Table 5.19):
    // the book's factors 1.0431 and, for six-month policies, 1.0130.
    let levels = column(&[1.05, 1.155, 1.1435]);
    let years = column(&[2011.0]);
    let book = call(
        "pro_onlevel",
        vec![
            eff.clone(),
            levels.clone(),
            years.clone(),
            n(2012.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            t("levels"),
        ],
    );
    let rl = call(
        "pro_ratelevel",
        vec![
            eff,
            levels,
            n(2011.0),
            n(2012.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            t("levels"),
        ],
    );
    printed(num(rl.get(4, 2)), 1.0963, 4);
    printed(1.1435 / round(num(rl.get(4, 2)), 4), 1.0431, 4);
    assert!(num(book.as_scalar().unwrap()) > 1.04);
}

#[test]
fn parallelogram_appendices() {
    // Appendix A, Current Rate Level-1 and -2: six-month policies, the
    // book's printed levels, calendar years 2011 to 2015 as a column.
    let eff = column(&[2011.25, 2012.5, 2013.75, 2014.5, 2015.75, 2016.0]);
    let levels = column(&[0.9500, 1.0450, 1.0973, 1.0754, 1.1292, 1.1857]);
    let from = column(&[2011.0, 2012.0, 2013.0, 2014.0, 2015.0]);
    let to = column(&[2012.0, 2013.0, 2014.0, 2015.0, 2016.0]);
    let book = [1.2161, 1.2176, 1.1311, 1.0892, 1.0991];
    let averages = [0.9750, 0.9738, 1.0483, 1.0886, 1.0788];
    let f = call(
        "pro_onlevel",
        vec![
            eff.clone(),
            levels.clone(),
            from,
            to,
            t("calendar"),
            n(0.5),
            t("written"),
            t("levels"),
        ],
    );
    assert_eq!((f.rows, f.columns), (5, 1));
    for k in 0..5 {
        // The book divides the current level by the average rounded to 4 dp.
        let average = 1.1857 / num(f.get(k, 0));
        printed(average, averages[k], 4);
        printed(1.1857 / averages[k], book[k], 4);
    }

    // Appendix C, Current Rate Level-1 and -2: annual policies, 1/10/11,
    // 1/3/14 and 1/7/15 as Excel dates.
    let eff = column(&[40_817.0, 41_699.0, 42_186.0]);
    let levels = column(&[0.9240, 1.0589, 1.2029]);
    let years = [2010.0, 2011.0, 2012.0, 2013.0, 2014.0, 2015.0];
    let book = [1.2029, 1.2058, 1.2724, 1.3018, 1.2391, 1.1190];
    let f = call(
        "pro_onlevel",
        vec![
            eff,
            levels,
            column(&years),
            column(&years.map(|y| y + 1.0)),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            t("levels"),
        ],
    );
    for (k, &factor) in book.iter().enumerate() {
        let average = round(1.2029 / num(f.get(k, 0)), 4);
        printed(1.2029 / average, factor, 4);
    }
}

#[test]
fn trend_lengths() {
    let avg = |from: f64, to: f64, period: &str, term: f64, basis: &str| {
        num(call(
            "pro_avgdate",
            vec![n(from), n(to), t(period), n(term), t(basis)],
        )
        .as_scalar()
        .unwrap())
    };
    // CY2011 premium to rates effective 1/1/13 for a year: 2.5 years
    // (annual), 2.25 (six-month); PY2011: 2.0.
    let to = avg(2013.0, 2014.0, "policy", 1.0, "written");
    assert_eq!(to - avg(2011.0, 2012.0, "calendar", 1.0, "written"), 2.5);
    let to6 = avg(2013.0, 2014.0, "policy", 0.5, "written");
    assert_eq!(to6 - avg(2011.0, 2012.0, "calendar", 0.5, "written"), 2.25);
    assert_eq!(to - avg(2011.0, 2012.0, "policy", 1.0, "written"), 2.0);
    // Losses: AY2011 to rates effective 1/1/15: 4.5 years; PY2011: 4.0.
    let to = avg(2015.0, 2016.0, "policy", 1.0, "earned");
    assert_eq!(
        to - avg(2011.0, 2012.0, "calendar", 1.0, Default::default()),
        4.5
    );
    assert_eq!(to - avg(2011.0, 2012.0, "policy", 1.0, "earned"), 4.0);
    // Appendix C, rates effective 5/1/2016 (an Excel date) from CY2010.
    let to = num(
        call("pro_avgdate", vec![n(42_491.0), n(42_856.0), t("policy")])
            .as_scalar()
            .unwrap(),
    );
    let from = num(call("pro_avgdate", vec![n(2010.0), n(2011.0)])
        .as_scalar()
        .unwrap());
    assert!((to - from - (6.0 + 10.0 / 12.0)).abs() < 1e-12);
}

#[test]
fn trend_fits_table_6_14() {
    let freq = [
        0.0587, 0.0587, 0.0593, 0.0587, 0.0582, 0.0578, 0.0565, 0.0555, 0.0552, 0.0553, 0.0556,
        0.0560, 0.0559, 0.0554, 0.0551, 0.0549, 0.0550, 0.0549, 0.0547, 0.0544,
    ];
    let sev = [
        1061.45, 1076.56, 1085.56, 1098.02, 1102.46, 1107.52, 1117.71, 1123.70, 1127.63, 1124.01,
        1116.42, 1109.48, 1105.38, 1102.61, 1096.67, 1093.77, 1100.67, 1108.79, 1122.13, 1126.46,
    ];
    type Fits = [(usize, f64); 6];
    let book: [(&[f64], Fits); 2] = [
        (
            &freq,
            [
                (20, -1.7),
                (16, -1.3),
                (12, -0.7),
                (8, -1.2),
                (6, -0.9),
                (4, -1.5),
            ],
        ),
        (
            &sev,
            [
                (20, 0.5),
                (16, -0.1),
                (12, -0.2),
                (8, 1.2),
                (6, 2.5),
                (4, 3.3),
            ],
        ),
    ];
    for (values, fits) in book {
        for (points, pct) in fits {
            let v = &values[values.len() - points..];
            let times: Vec<f64> = (0..points).map(|i| 2010.0 + i as f64 / 4.0).collect();
            let fit = call("pro_trend_fit", vec![column(&times), column(v)]);
            assert_eq!(fit.get(1, 0), &Value::from("Annual change"));
            let got = 100.0 * num(fit.get(1, 1));
            assert!(
                (got - pct).abs() <= 0.05 + 1e-9,
                "{points} points: {got} vs {pct}"
            );
            assert_eq!(num(fit.get(5, 1)), points as f64);
        }
    }
    // A linear fit of a straight line, with a blank pair skipped.
    let times = Grid::from_rows(vec![vec![
        Value::Number(0.0),
        Value::Number(1.0),
        Value::Empty,
        Value::Number(3.0),
    ]]);
    let values = Grid::from_rows(vec![vec![
        Value::Number(10.0),
        Value::Number(12.0),
        Value::Number(99.0),
        Value::Number(16.0),
    ]]);
    let fit = call("pro_trend_fit", vec![times, values, t("linear")]);
    assert!((num(fit.get(1, 1)) - 2.0).abs() < 1e-12);
    assert!((num(fit.get(4, 1)) - 1.0).abs() < 1e-12);
    assert_eq!(num(fit.get(5, 1)), 3.0);
}

#[test]
fn errors_come_back_as_text() {
    let e = |f: &str, a: Vec<Grid>| text(&call(f, a)).to_owned();
    let eff = column(&[2011.0, 2012.0]);
    assert!(
        e(
            "pro_onlevel",
            vec![eff.clone(), column(&[0.1]), n(2011.0), n(2012.0)]
        )
        .contains("one value per effective date")
    );
    assert!(
        e(
            "pro_onlevel",
            vec![
                column(&[2012.0, 2011.0]),
                column(&[0.1, 0.1]),
                n(2011.0),
                n(2012.0)
            ]
        )
        .contains("order")
    );
    assert!(
        e(
            "pro_onlevel",
            vec![eff.clone(), column(&[0.1, 0.1]), n(2012.0), n(2011.0)]
        )
        .contains("end after")
    );
    assert!(
        e(
            "pro_onlevel",
            vec![
                eff.clone(),
                column(&[0.1, 0.1]),
                n(2011.0),
                n(2012.0),
                t("fiscal")
            ]
        )
        .contains("period must be")
    );
    assert!(
        e(
            "pro_onlevel",
            vec![
                eff,
                column(&[0.1, 0.1]),
                n(2011.0),
                n(2012.0),
                Grid::missing(),
                Grid::missing(),
                t("sometimes")
            ]
        )
        .contains("applies must be")
    );
    assert!(
        e(
            "pro_trend_fit",
            vec![column(&[0.0, 1.0]), column(&[1.0, -1.0])]
        )
        .contains("positive")
    );
    assert!(
        e("pro_trend_fit", vec![column(&[0.0, 1.0]), column(&[1.0])])
            .contains("same number of cells")
    );
}
