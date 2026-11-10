//! Recoveries, ALAE and ULAE (`docs/design/xll.md`, decision 53) against
//! Friedland's printed exhibits, as `validation/tests/reserving_expense.rs`
//! checks the library: chapter 14 Exhibit I (salvage and subrogation),
//! chapter 16 Exhibit I (ALAE) and chapter 17 Exhibits I, II and IV
//! (ULAE). The ultimate claims are chained from `PRO.RES.CL.ULTIMATE` as a
//! sheet would, on the printed factors to ultimate.

use crate::common;

use common::{cell, data_window, fails, n, ok, row, t};
use prospicio_xll::sheet::{Grid, Value};

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (tolerance {tol})"
    );
}

fn close_all(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, tol, &format!("{what}[{k}]"));
    }
}

fn r(x: f64, digits: i32) -> f64 {
    let s = 10f64.powi(digits);
    (x * s).round() / s
}

fn data(file: &str, column: &str) -> Grid {
    data_window(file, column, 1998, 2008, 132).0
}

/// The mean of the chain ladders on reported and paid claims, each on the
/// book's printed factors to ultimate.
fn selected_claims(file: &str, reported_cdf: &[f64], paid_cdf: &[f64]) -> Vec<f64> {
    let ult = |column: &str, cdf: &[f64]| {
        let g = ok(
            "pro_res_cl_ultimate",
            vec![
                data(file, column),
                Grid::missing(),
                Grid::missing(),
                row(cdf),
                t("cdf"),
            ],
        );
        (0..g.rows)
            .map(|r| cell(&g, r, 0).unwrap())
            .collect::<Vec<_>>()
    };
    let (a, b) = (ult("reported", reported_cdf), ult("paid", paid_cdf));
    a.iter().zip(&b).map(|(x, y)| (x + y) / 2.0).collect()
}

fn table_column(table: &Grid, name: &str) -> Vec<f64> {
    let c = common::column_of(table, name);
    (1..table.rows - 1)
        .map(|r| cell(table, r, c).unwrap())
        .collect()
}

fn table_total(table: &Grid, name: &str) -> f64 {
    cell(table, table.rows - 1, common::column_of(table, name)).unwrap()
}

/// A labelled row of a `"factors"` output, from its first link.
fn factor_row(g: &Grid, label: &str) -> Vec<Option<f64>> {
    let r = (0..g.rows)
        .find(|&r| *g.get(r, 0) == Value::from(label))
        .unwrap_or_else(|| panic!("no row {label}"));
    (1..g.columns).map(|c| cell(g, r, c)).collect()
}

fn average_row(kind: &str, latest: f64, drop: f64) -> Grid {
    let mut cells = vec![Value::from(kind), Value::Number(latest)];
    if drop > 0.0 {
        cells.extend([Value::Number(drop), Value::Number(drop)]);
    }
    Grid::from_rows(vec![cells])
}

/// A column of `n` blanks with `values` at their rows.
fn per_origin_last(values: &[(usize, f64)], n: usize) -> Grid {
    Grid::from_rows(
        (0..n)
            .map(|k| {
                vec![
                    values
                        .iter()
                        .find(|(i, _)| *i == k)
                        .map_or(Value::Empty, |(_, x)| Value::Number(*x)),
                ]
            })
            .collect(),
    )
}

const PHYSICAL: &str = "friedland_auto_physical_damage";
const PROPERTY: &str = "friedland_auto_property_damage";
const SS_RATIO_CDF: [f64; 11] = [1.499, 1.009, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
const ALAE_RATIO_CDF: [f64; 11] = [
    1.332, 1.201, 1.140, 1.086, 1.050, 1.021, 1.007, 1.003, 1.002, 1.0, 1.0,
];

fn ss_claims() -> Vec<f64> {
    selected_claims(
        PHYSICAL,
        &[1.115, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[1.279, 1.005, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
}

fn alae_claims() -> Vec<f64> {
    selected_claims(
        PROPERTY,
        &[
            1.101, 0.990, 0.989, 0.991, 0.993, 0.998, 0.999, 0.999, 1.0, 1.0, 1.0,
        ],
        &[
            1.584, 1.029, 1.007, 1.004, 1.002, 1.001, 1.001, 1.001, 1.0, 1.0, 1.0,
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn ratio(
    file: &str,
    numerator: &str,
    claims: &[f64],
    average: Grid,
    tail: Grid,
    selected: Grid,
    kind: Grid,
    factors: Grid,
    ultimate_ratios: Grid,
    output: &str,
) -> Grid {
    ok(
        "pro_res_ratio",
        vec![
            data(file, numerator),
            data(file, "paid"),
            Grid::column(claims),
            average,
            tail,
            selected,
            kind,
            factors,
            ultimate_ratios,
            Grid::missing(),
            t(output),
        ],
    )
}

/// Chapter 14, Exhibit I, sheets 6 to 9: received salvage and subrogation
/// over paid claims.
#[test]
fn friedland_14_salvage_and_subrogation() {
    let claims = ss_claims();
    close_all(
        &claims,
        &[
            2864.0, 4697.0, 7902.0, 10319.0, 11137.0, 12527.0, 14536.0, 16837.0, 16952.0, 16893.0,
            16453.0,
        ],
        1.0,
        "selected claims",
    );
    // Sheet 7: the medial average of the latest five.
    let medial = ratio(
        PHYSICAL,
        "received_ss",
        &claims,
        average_row("simple", 5.0, 1.0),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        "factors",
    );
    let averages: Vec<f64> = factor_row(&medial, "Average")[..6]
        .iter()
        .map(|x| x.unwrap())
        .collect();
    close_all(
        &averages,
        &[1.486, 1.009, 1.000, 1.000, 1.000, 1.000],
        5e-4,
        "medial average",
    );
    let cdf: Vec<f64> = factor_row(&medial, "To ultimate")
        .iter()
        .map(|x| x.unwrap())
        .collect();
    close_all(&cdf, &SS_RATIO_CDF, 1.5e-3, "CDF");
    let ratios = ratio(
        PHYSICAL,
        "received_ss",
        &claims,
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        "ratios",
    );
    let first: Vec<f64> = (0..11)
        .map(|o| r(cell(&ratios, o, 0).unwrap(), 3))
        .collect();
    assert_eq!(
        first,
        [
            0.157, 0.198, 0.166, 0.264, 0.257, 0.220, 0.235, 0.243, 0.233, 0.217, 0.210
        ]
    );
    // Sheets 8 and 9: the printed factors, 0.345 selected for 2008.
    let table = ratio(
        PHYSICAL,
        "received_ss",
        &claims,
        Grid::missing(),
        Grid::missing(),
        row(&SS_RATIO_CDF),
        t("cdf"),
        Grid::missing(),
        per_origin_last(&[(10, 0.345)], 11),
        "ultimate",
    );
    let projected: Vec<f64> = table_column(&table, "Projected ratio")
        .iter()
        .map(|x| r(*x, 3))
        .collect();
    assert_eq!(
        projected,
        [
            0.277, 0.290, 0.306, 0.352, 0.367, 0.348, 0.355, 0.340, 0.334, 0.357, 0.315
        ]
    );
    assert_eq!(table_column(&table, "Selected ratio")[10], 0.345);
    close_all(
        &table_column(&table, "Ultimate"),
        &[
            793.0, 1360.0, 2421.0, 3637.0, 4090.0, 4365.0, 5165.0, 5731.0, 5658.0, 6036.0, 5676.0,
        ],
        1.0,
        "projected S&S",
    );
    close(table_total(&table, "Ultimate"), 44_934.0, 2.0, "total");
    close(table_total(&table, "Latest"), 41_879.0, 1e-9, "received");
    close_all(
        &table_column(&table, "Unpaid"),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.0, 79.0, 2966.0],
        1.0,
        "recoverable",
    );
    close(
        table_total(&table, "Unpaid"),
        3_054.0,
        2.0,
        "total recoverable",
    );
}

/// Chapter 16, Exhibit I, sheets 7, 8 and 11: paid ALAE over paid claims,
/// multiplicative, 0.0077 selected for 2007 and 2008.
#[test]
fn friedland_16_alae_multiplicative() {
    let claims = alae_claims();
    let averaged = ratio(
        PROPERTY,
        "paid_alae",
        &claims,
        average_row("simple", 3.0, 0.0),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        "factors",
    );
    let selected: Vec<f64> = factor_row(&averaged, "Selected")[..10]
        .iter()
        .map(|x| x.unwrap())
        .collect();
    close_all(
        &selected,
        &[
            1.109, 1.053, 1.050, 1.034, 1.029, 1.013, 1.004, 1.001, 1.002, 1.000,
        ],
        5e-4,
        "selected",
    );
    let ratios = per_origin_last(&[(9, 0.0077), (10, 0.0077)], 11);
    let table = ratio(
        PROPERTY,
        "paid_alae",
        &claims,
        Grid::missing(),
        Grid::missing(),
        row(&ALAE_RATIO_CDF),
        t("cdf"),
        t("multiplicative"),
        ratios,
        "ultimate",
    );
    close_all(
        &table_column(&table, "Ultimate"),
        &[
            1084.0, 1045.0, 861.0, 1001.0, 1106.0, 1108.0, 1040.0, 1072.0, 1302.0, 1360.0, 1505.0,
        ],
        1.0,
        "projected ALAE",
    );
    close(table_total(&table, "Ultimate"), 12_485.0, 2.0, "total");
    close_all(
        &table_column(&table, "Unpaid"),
        &[
            0.0, 0.0, 0.0, 3.0, 8.0, 23.0, 51.0, 85.0, 170.0, -94.0, 553.0,
        ],
        1.0,
        "unpaid",
    );
    close(table_total(&table, "Unpaid"), 800.0, 2.0, "total unpaid");
}

/// Chapter 16, Exhibit I, sheets 9 to 11: additive development.
#[test]
fn friedland_16_alae_additive() {
    let claims = alae_claims();
    let args = |output: &str| {
        ratio(
            PROPERTY,
            "paid_alae",
            &alae_claims(),
            average_row("simple", 3.0, 0.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            t("additive"),
            Grid::missing(),
            output,
        )
    };
    let factors = args("factors");
    let selected: Vec<f64> = factor_row(&factors, "Selected")[..10]
        .iter()
        .map(|x| r(x.unwrap(), 4))
        .collect();
    assert_eq!(
        selected,
        [
            0.0007, 0.0003, 0.0003, 0.0002, 0.0002, 0.0001, 0.0, 0.0, 0.0, 0.0
        ]
    );
    let cdf: Vec<f64> = factor_row(&factors, "To ultimate")
        .iter()
        .map(|x| r(x.unwrap(), 4))
        .collect();
    assert_eq!(
        cdf,
        [
            0.0019, 0.0012, 0.0009, 0.0006, 0.0004, 0.0002, 0.0001, 0.0, 0.0, 0.0, 0.0
        ]
    );
    let table = args("ultimate");
    close_all(
        &table_column(&table, "Ultimate"),
        &[
            1084.0, 1045.0, 862.0, 1003.0, 1108.0, 1112.0, 1046.0, 1077.0, 1300.0, 1709.0, 1870.0,
        ],
        1.0,
        "projected ALAE",
    );
    close(table_total(&table, "Ultimate"), 13_215.0, 2.0, "total");
    close(table_total(&table, "Unpaid"), 1_530.0, 2.0, "total unpaid");
    assert_eq!(table_column(&table, "Ultimate claims"), claims);
    // An additive selection takes no kind.
    let message = fails(
        "pro_res_ratio",
        vec![
            data(PROPERTY, "paid_alae"),
            data(PROPERTY, "paid"),
            Grid::column(&claims),
            Grid::missing(),
            Grid::missing(),
            row(&[0.001]),
            t("cdf"),
            t("additive"),
        ],
    );
    assert!(message.contains("kind"), "{message}");
}

const XYZ_ULAE: [f64; 5] = [14.352e6, 15.321e6, 16.870e6, 17.112e6, 17.331e6];
const XYZ_PAID: [f64; 5] = [333e6, 358e6, 334e6, 347e6, 391e6];
const XYZ_INCURRED: [f64; 5] = [535.213e6, 492.265e6, 435.985e6, 432.966e6, 475.300e6];

fn ulae(
    technique: &str,
    ratio: f64,
    weights: Grid,
    reported: Grid,
    pure: Grid,
    output: &str,
) -> Grid {
    ok(
        "pro_res_ulae",
        vec![
            row(&XYZ_ULAE),
            row(&XYZ_PAID),
            n(603e6),
            n(316e6),
            t(technique),
            n(ratio),
            weights,
            reported,
            Grid::missing(),
            pure,
            Grid::missing(),
            Grid::missing(),
            t(output),
        ],
    )
}

fn summary(g: &Grid, label: &str) -> f64 {
    let r = (0..g.rows)
        .find(|&r| *g.get(r, 0) == Value::from(label))
        .unwrap_or_else(|| panic!("no row {label}"));
    cell(g, r, 1).unwrap()
}

/// Chapter 17, Exhibits I and II: XYZ Insurer, the classical technique and
/// Kittel's refinement, with and without a pure IBNR.
#[test]
fn friedland_17_classical_and_kittel() {
    let classical = ulae(
        "classical",
        0.045,
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        "summary",
    );
    close(
        summary(&classical, "Unpaid ULAE"),
        27_787_500.0,
        1e-6,
        "classical",
    );
    assert_eq!(r(summary(&classical, "Total ratio"), 3), 0.046);
    let years = ulae(
        "classical",
        0.045,
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        "years",
    );
    let ratios: Vec<f64> = (1..6).map(|y| r(cell(&years, y, 2).unwrap(), 3)).collect();
    assert_eq!(ratios, [0.043, 0.043, 0.051, 0.049, 0.044]);
    let pure = ulae(
        "classical",
        0.045,
        Grid::missing(),
        Grid::missing(),
        n(0.05 * 380e6),
        "summary",
    );
    close(
        summary(&pure, "Unpaid ULAE"),
        21_105_000.0,
        1e-6,
        "classical pure",
    );

    let kittel = ulae(
        "kittel",
        0.040,
        Grid::missing(),
        row(&XYZ_INCURRED),
        Grid::missing(),
        "years",
    );
    let basis: Vec<f64> = (1..6).map(|y| cell(&kittel, y, 1).unwrap()).collect();
    close_all(
        &basis,
        &[
            434_106_500.0,
            425_132_500.0,
            384_992_500.0,
            389_983_000.0,
            433_150_000.0,
        ],
        1e-6,
        "Kittel basis",
    );
    let kittel = ulae(
        "kittel",
        0.040,
        Grid::missing(),
        row(&XYZ_INCURRED),
        Grid::missing(),
        "summary",
    );
    close(
        summary(&kittel, "Unpaid ULAE"),
        24_700_000.0,
        1e-6,
        "Kittel",
    );
    let pure = ulae(
        "kittel",
        0.040,
        Grid::missing(),
        row(&XYZ_INCURRED),
        n(19e6),
        "summary",
    );
    close(
        summary(&pure, "Unpaid ULAE"),
        18_760_000.0,
        1e-6,
        "Kittel pure",
    );
    // A Kittel run takes no weights.
    assert!(
        fails(
            "pro_res_ulae",
            vec![
                row(&XYZ_ULAE),
                row(&XYZ_PAID),
                n(603e6),
                n(316e6),
                t("kittel"),
                Grid::missing(),
                n(0.5)
            ]
        )
        .contains("weights")
    );
}

const PQR_ULAE: [f64; 6] = [1978.0, 4820.0, 8558.0, 12039.0, 13143.0, 15286.0];
const PQR_PAID: [f64; 6] = [4590.0, 14600.0, 38390.0, 58297.0, 86074.0, 105466.0];
const PQR_ULT_REPORTED: [f64; 6] = [27200.0, 76700.0, 106900.0, 154300.0, 163100.0, 176400.0];
const PQR_ULTIMATE: [f64; 6] = [28600.0, 79200.0, 108400.0, 156700.0, 163400.0, 177100.0];

/// Chapter 17, Exhibit IV, sheets 3 and 4: Conger and Nolibos' generalized
/// approach on PQR Insurer at 60/40 and 70/30, with its three total forms.
#[test]
fn friedland_17_conger_nolibos() {
    for (opening, basis_total, bf, development) in [
        (0.6, 545_727.0, 16_767.0, 17_152.0),
        (0.7, 585_445.0, 12_795.0, 12_201.0),
    ] {
        let args = |output: &str| {
            vec![
                row(&PQR_ULAE),
                row(&PQR_PAID),
                n(292_130.0),
                n(113_853.0),
                t("conger_nolibos"),
                n(0.100),
                row(&[opening, 1.0 - opening]),
                row(&PQR_ULT_REPORTED),
                Grid::missing(),
                n(713_400.0 - 704_600.0),
                Grid::missing(),
                row(&PQR_ULTIMATE),
                t(output),
            ]
        };
        let s = ok("pro_res_ulae", args("summary"));
        close(
            summary(&s, "Expected claims form"),
            15_516.0,
            0.5,
            "expected",
        );
        close(summary(&s, "Bornhuetter-Ferguson form"), bf, 0.5, "BF");
        close(
            summary(&s, "Development form"),
            development,
            0.5,
            "development",
        );
        close(summary(&s, "Unpaid ULAE"), bf, 0.5, "unpaid");
        let years = ok("pro_res_ulae", args("years"));
        close(cell(&years, 7, 1).unwrap(), basis_total, 0.5, "total basis");
        let rows = ok("pro_res_ulae", args("rows"));
        close(
            cell(&rows, 0, 0).unwrap(),
            summary(&s, "Unpaid ULAE"),
            1e-9,
            "rows",
        );
    }
}

/// Rows by accident year sum to the total, the formula being linear.
#[test]
fn ulae_by_accident_year_sums_to_the_total() {
    let ibnr = [257.0, 1742.0, 5095.0, 16140.0, 34477.0, 56141.0];
    let total: f64 = ibnr.iter().sum();
    let case: Vec<f64> = ibnr.iter().map(|i| 292_130.0 * i / total).collect();
    let run = |case: Grid, ibnr: Grid, output: &str| {
        ok(
            "pro_res_ulae",
            vec![
                row(&PQR_ULAE),
                row(&PQR_PAID),
                case,
                ibnr,
                Grid::missing(),
                n(0.160),
                Grid::missing(),
                Grid::missing(),
                Grid::missing(),
                Grid::missing(),
                Grid::missing(),
                Grid::missing(),
                t(output),
            ],
        )
    };
    let rows = run(Grid::column(&case), Grid::column(&ibnr), "rows");
    assert_eq!(rows.rows, 6);
    let sum: f64 = (0..6).map(|r| cell(&rows, r, 0).unwrap()).sum();
    let whole = run(n(292_130.0), n(total), "summary");
    close(sum, summary(&whole, "Unpaid ULAE"), 1e-6, "sum of years");
}
