//! Catastrophe model output on a sheet (`PRO.YLT.STATS`, `PRO.YLT.OEP`,
//! `PRO.ELT.STATS`, `PRO.ELT.OEP`, `docs/design/xll.md` decision 44),
//! reached through the catalogue as Excel reaches it:
//!
//! * Alder Mutual's year loss table and event loss table
//!   (`examples/data/cat_book_*.csv`) against the numbers the guide
//!   `docs/guides/cat-2-risk-metrics.md` prints, which CI checks against
//!   the Python package: the average annual loss, SD, VaR and TVaR under
//!   each quantile rule, the OEP and AEP at amounts, raw and under a
//!   72-hour clause, the PMLs by return period, and the ELT's rate,
//!   expected annual loss, OEP and occurrence PMLs;
//! * the ELT's SD against its simulated years, and the errors.

mod common;

use common::{n, num, t, text, validation};
use prospicio_aggregate::cat::EventLossTable;
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

/// Alder's rows of the year loss table: years, losses and times.
fn alder() -> (Grid, Grid, Grid) {
    let file = validation("../examples/data/cat_book_yelt.csv");
    let mut lines = file.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |name: &str| head.iter().position(|h| *h == name).unwrap();
    let (y, d, h, c, l) = (
        col("year"),
        col("day"),
        col("hour"),
        col("cedent"),
        col("loss"),
    );
    let (mut years, mut losses, mut times) = (vec![], vec![], vec![]);
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        if f[c] != "alder" {
            continue;
        }
        years.push(f[y].parse::<f64>().unwrap());
        losses.push(f[l].parse::<f64>().unwrap());
        let (day, hour): (f64, f64) = (f[d].parse().unwrap(), f[h].parse().unwrap());
        times.push(((day - 1.0) * 24.0 + hour) / 8760.0);
    }
    (
        Grid::column(&years),
        Grid::column(&losses),
        Grid::column(&times),
    )
}

/// Alder's event loss table: rates, means, the two SDs and exposures.
fn alder_elt() -> [Grid; 5] {
    let file = validation("../examples/data/cat_book_elt_alder.csv");
    let mut lines = file.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
    [
        "rate",
        "mean",
        "sd_independent",
        "sd_correlated",
        "exposure",
    ]
    .map(|name| {
        let k = head.iter().position(|h| *h == name).unwrap();
        Grid::column(
            &rows
                .iter()
                .map(|r| r[k].parse().unwrap())
                .collect::<Vec<f64>>(),
        )
    })
}

/// The value beside `label` in a two-column exhibit.
fn get(g: &Grid, label: &str) -> f64 {
    (0..g.rows)
        .find(|&r| g.get(r, 0) == &Value::from(label))
        .map(|r| num(g.get(r, 1)))
        .unwrap_or_else(|| panic!("no row {label:?} in {g:?}"))
}

/// `x` rounds to `printed` at its decimals.
fn printed(x: f64, printed: f64, decimals: i32) {
    let half = 0.5 * 10f64.powi(-decimals) + 1e-12;
    assert!(
        (x - printed).abs() <= half,
        "{x} does not print as {printed}"
    );
}

#[test]
fn ylt_stats_match_the_guide() {
    let (years, losses, _) = alder();
    let st = call(
        "pro_ylt_stats",
        vec![
            years.clone(),
            losses.clone(),
            n(2000.0),
            Grid::row(&[0.9, 0.99]),
        ],
    );
    assert_eq!(get(&st, "Years"), 2000.0);
    assert_eq!(get(&st, "Losses"), 5391.0);
    assert_eq!(get(&st, "Occurrences"), 5391.0);
    printed(get(&st, "Mean"), 16.778, 3);
    printed(get(&st, "SD"), 35.158, 3);
    printed(get(&st, "VaR 90%"), 42.244, 3);
    printed(get(&st, "VaR 99%"), 176.824, 3);
    printed(get(&st, "TVaR 99%"), 255.969, 3);
    assert!((get(&st, "CV") - get(&st, "SD") / get(&st, "Mean")).abs() < 1e-15);

    let linear = call(
        "pro_ylt_stats",
        vec![years.clone(), losses.clone(), none(), n(0.99), t("linear")],
    );
    printed(get(&linear, "VaR 99%"), 176.836, 3);
    printed(get(&linear, "TVaR 99%"), 246.942, 3);
    let kth = call(
        "pro_ylt_stats",
        vec![years, losses, none(), n(0.99), t("kth_largest")],
    );
    printed(get(&kth, "VaR 99%"), 177.979, 3);
    printed(get(&kth, "TVaR 99%"), 255.969, 3);
}

#[test]
fn ylt_exceedance_at_amounts_raw_and_grouped() {
    let (years, losses, times) = alder();
    let amounts = Grid::column(&[30.0, 70.0, 110.0]);
    let raw = call(
        "pro_ylt_oep",
        vec![years.clone(), losses.clone(), none(), amounts.clone()],
    );
    assert_eq!(raw.get(0, 1), &Value::from("OEP"));
    for (r, (oep, aep)) in [(0.1070, 0.1465), (0.0345, 0.0535), (0.0190, 0.0235)]
        .into_iter()
        .enumerate()
    {
        printed(num(raw.get(r + 1, 1)), oep, 4);
        printed(num(raw.get(r + 1, 2)), aep, 4);
        assert!((num(raw.get(r + 1, 3)) * num(raw.get(r + 1, 1)) - 1.0).abs() < 1e-12);
    }
    let grouped = call(
        "pro_ylt_oep",
        vec![
            years.clone(),
            losses.clone(),
            none(),
            amounts,
            none(),
            none(),
            times.clone(),
            n(72.0),
        ],
    );
    for (r, oep) in [0.1150, 0.0375, 0.0215].into_iter().enumerate() {
        printed(num(grouped.get(r + 1, 1)), oep, 4);
        // The clause leaves each year's total as it was.
        assert_eq!(grouped.get(r + 1, 2), raw.get(r + 1, 2));
    }
    let st = call(
        "pro_ylt_stats",
        vec![years, losses, none(), none(), none(), times, n(72.0)],
    );
    assert_eq!(get(&st, "Occurrences"), 4238.0);
    assert_eq!(get(&st, "Losses"), 5391.0);
}

#[test]
fn ylt_pmls_by_return_period() {
    let (years, losses, times) = alder();
    let rp = Grid::column(&[10.0, 50.0, 100.0, 250.0]);
    let curve = call(
        "pro_ylt_oep",
        vec![
            years.clone(),
            losses.clone(),
            rp,
            none(),
            none(),
            none(),
            times,
            n(72.0),
        ],
    );
    assert_eq!(curve.rows, 5);
    for (r, (aep, oep)) in [
        (42.244, 33.971),
        (125.246, 120.464),
        (176.824, 163.418),
        (232.741, 223.450),
    ]
    .into_iter()
    .enumerate()
    {
        printed(num(curve.get(r + 1, 2)), oep, 3);
        printed(num(curve.get(r + 1, 3)), aep, 3);
        assert!(num(curve.get(r + 1, 4)) >= num(curve.get(r + 1, 2)));
        assert!(num(curve.get(r + 1, 5)) >= num(curve.get(r + 1, 3)));
    }
    // The 100-year aggregate TVaR is STATS' TVaR 99%.
    let st = call(
        "pro_ylt_stats",
        vec![years.clone(), losses.clone(), none(), n(0.99)],
    );
    let plain = call("pro_ylt_oep", vec![years.clone(), losses.clone(), n(100.0)]);
    assert_eq!(num(plain.get(1, 5)), get(&st, "TVaR 99%"));
    // The defaults stop at the table's 2,000 years, so 2 to 1000 all stay.
    let defaults = call("pro_ylt_oep", vec![years.clone(), losses.clone()]);
    assert_eq!(defaults.rows, 10);
    assert_eq!(num(defaults.get(9, 0)), 1000.0);
    let short = call(
        "pro_ylt_oep",
        vec![
            Grid::column(&[1.0, 3.0]),
            Grid::column(&[5.0, 7.0]),
            none(),
            none(),
            n(40.0),
        ],
    );
    assert_eq!(short.rows, 5);
    assert_eq!(num(short.get(4, 0)), 25.0);
}

#[test]
fn elt_matches_the_guide() {
    let [rates, means, sdi, sdc, exposures] = alder_elt();
    let st = call(
        "pro_elt_stats",
        vec![
            rates.clone(),
            means.clone(),
            sdi.clone(),
            sdc.clone(),
            exposures.clone(),
        ],
    );
    assert_eq!(get(&st, "Events"), 299.0);
    printed(get(&st, "Events a year"), 2.3365, 4);
    printed(get(&st, "Expected annual loss"), 16.056, 3);
    let elt = vec![rates, means, sdi, sdc, exposures];
    let mut at = elt.clone();
    at.extend([none(), Grid::column(&[30.0, 70.0, 110.0])]);
    let at = call("pro_elt_oep", at);
    for (r, p) in [0.1136, 0.0374, 0.0180].into_iter().enumerate() {
        printed(num(at.get(r + 1, 2)), p, 4);
        let f = num(at.get(r + 1, 1));
        assert!((num(at.get(r + 1, 2)) - (1.0 - (-f).exp())).abs() < 1e-15);
    }
    let mut by = elt.clone();
    by.push(Grid::column(&[10.0, 50.0, 100.0, 250.0]));
    let by = call("pro_elt_oep", by);
    for (r, pml) in [33.648, 103.432, 147.938, 199.470].into_iter().enumerate() {
        printed(num(by.get(r + 1, 2)), pml, 3);
        // The exceedance frequency at the PML gives the probability back.
        let f = num(by.get(r + 1, 3));
        assert!(((1.0 - (-f).exp()) - num(by.get(r + 1, 1))).abs() < 1e-9);
    }
}

#[test]
fn elt_sd_agrees_with_simulated_years() {
    let rates = [0.005, 0.02, 0.1, 0.4];
    let means = [20_000.0, 6000.0, 1500.0, 200.0];
    let sdi = [6000.0, 2000.0, 600.0, 100.0];
    let sdc = [2000.0, 1000.0, 200.0, 0.0];
    let exposures = [60_000.0, 25_000.0, 8000.0, 2000.0];
    let st = call(
        "pro_elt_stats",
        [&rates[..], &means, &sdi, &sdc, &exposures]
            .map(Grid::column)
            .to_vec(),
    );
    let elt = EventLossTable::new(&rates, &means, &sdi, &sdc, &exposures).unwrap();
    let (events, _) = elt.simulate(200_000, 5).unwrap();
    let totals: Vec<f64> = (0..events.n_sims())
        .map(|y| events.events(y).iter().sum())
        .collect();
    let m = totals.iter().sum::<f64>() / totals.len() as f64;
    let var = totals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / totals.len() as f64;
    let sd = get(&st, "SD");
    // The sample variance's standard error from the fourth moment.
    let m4 = totals.iter().map(|x| (x - m).powi(4)).sum::<f64>() / totals.len() as f64;
    let se = ((m4 - var * var) / totals.len() as f64).sqrt();
    assert!((var - sd * sd).abs() < 4.0 * se, "{var} vs {}", sd * sd);
    assert_eq!(get(&st, "Expected annual loss"), elt.expected_annual_loss());
    assert_eq!(get(&st, "Largest exposure"), 60_000.0);
    // Without the standard deviations each event loses its mean.
    let point = call(
        "pro_elt_stats",
        vec![
            Grid::column(&rates),
            Grid::column(&means),
            none(),
            none(),
            Grid::column(&exposures),
        ],
    );
    let v: f64 = rates.iter().zip(&means).map(|(r, m)| r * m * m).sum();
    assert!((get(&point, "SD") - v.sqrt()).abs() < 1e-9);
}

#[test]
fn small_tables_and_errors() {
    // Years 1 and 3 of a 4-year table; year 3 has two losses, timed a day
    // apart, so a 72-hour clause joins them.
    let years = Grid::column(&[1.0, 3.0, 3.0]);
    let losses = Grid::column(&[5.0, 7.0, 2.0]);
    let times = Grid::column(&[0.5, 0.2, 0.2 + 24.0 / 8760.0]);
    let st = call(
        "pro_ylt_stats",
        vec![years.clone(), losses.clone(), n(4.0), n(0.5)],
    );
    assert_eq!(get(&st, "Years"), 4.0);
    assert_eq!(get(&st, "Mean"), 14.0 / 4.0);
    assert_eq!(get(&st, "Years with a loss"), 2.0);
    assert_eq!(get(&st, "Largest occurrence"), 7.0);
    let g = call(
        "pro_ylt_stats",
        vec![
            years.clone(),
            losses.clone(),
            n(4.0),
            n(0.5),
            none(),
            times.clone(),
            n(72.0),
        ],
    );
    assert_eq!(get(&g, "Occurrences"), 2.0);
    assert_eq!(get(&g, "Largest occurrence"), 9.0);
    // Negative losses, as some files store them.
    let neg = call(
        "pro_ylt_stats",
        vec![
            years.clone(),
            Grid::column(&[-5.0, -7.0, -2.0]),
            n(4.0),
            n(0.5),
            none(),
            none(),
            none(),
            Grid::scalar(Value::Bool(true)),
        ],
    );
    assert_eq!(neg, st);
    // A blank row at the end of the range is skipped.
    let padded = call(
        "pro_ylt_stats",
        vec![
            Grid::from_rows(vec![
                vec![v(1.0)],
                vec![v(3.0)],
                vec![v(3.0)],
                vec![Value::Empty],
            ]),
            Grid::from_rows(vec![
                vec![v(5.0)],
                vec![v(7.0)],
                vec![v(2.0)],
                vec![Value::Empty],
            ]),
            n(4.0),
            n(0.5),
        ],
    );
    assert_eq!(padded, st);

    let error = |args: Vec<Grid>, symbol: &str, needle: &str| {
        let out = call(symbol, args);
        let msg = text(&out);
        assert!(msg.contains(needle), "{msg:?} lacks {needle:?}");
    };
    error(
        vec![years.clone(), Grid::column(&[5.0, 7.0])],
        "pro_ylt_stats",
        "2 cells for 3 years",
    );
    error(
        vec![Grid::column(&[1.0, 0.0, 3.0]), losses.clone()],
        "pro_ylt_stats",
        "row 2",
    );
    error(
        vec![years.clone(), Grid::column(&[5.0, -7.0, 2.0])],
        "pro_ylt_stats",
        "not finite and non-negative",
    );
    error(
        vec![
            years.clone(),
            losses.clone(),
            none(),
            none(),
            none(),
            none(),
            n(72.0),
        ],
        "pro_ylt_stats",
        "needs each loss's time",
    );
    error(
        vec![years.clone(), losses.clone(), n(2.0)],
        "pro_ylt_stats",
        "below the largest year",
    );
    error(
        vec![years.clone(), losses.clone(), none(), n(1.5)],
        "pro_ylt_stats",
        "probabilities must be in (0, 1)",
    );
    error(
        vec![years.clone(), losses.clone(), none(), none(), t("type7")],
        "pro_ylt_stats",
        "quantile must be",
    );
    error(
        vec![years.clone(), losses.clone(), n(10.0)],
        "pro_ylt_oep",
        "the table has 3 years",
    );
    error(
        vec![years.clone(), losses.clone(), n(2.0), n(5.0)],
        "pro_ylt_oep",
        "not both",
    );
    let one = |x: f64| Grid::column(&[x]);
    error(
        vec![one(0.5), one(100.0), one(200.0), none(), one(500.0)],
        "pro_elt_stats",
        "no beta on [0, 500]",
    );
    error(
        vec![one(0.5), one(100.0), none(), none(), one(500.0), n(1.2)],
        "pro_elt_oep",
        "probability 0.8333",
    );
    // The defaults stop where the chance of any event does: 1 - e^-0.5 is
    // about 0.39, so the 2-year point is left out.
    let curve = call(
        "pro_elt_oep",
        vec![one(0.5), one(100.0), none(), none(), one(500.0)],
    );
    assert_eq!(curve.rows, 9);
    assert_eq!(num(curve.get(1, 0)), 5.0);
}

fn v(x: f64) -> Value {
    Value::Number(x)
}
