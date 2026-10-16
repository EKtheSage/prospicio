//! Ratemaking, part two (`PRO.LR.EXPERIENCE`, `PRO.PP.EXPERIENCE`,
//! `PRO.INDICATED.CHANGE`, `PRO.INDICATED.RATE`, `PRO.PERMISSIBLE.LR`,
//! `PRO.TRENDED.PRESENT`, `PRO.CRED.WEIGHTED`, `PRO.EXPENSES`), reached
//! through the catalogue as Excel reaches them, against Werner and Modlin,
//! *Basic Ratemaking* (5th ed., CAS, 2016): chapters 7, 8 and 12 and the
//! indications of Appendices A (loss ratio), B (pure premium) and C (loss
//! ratio on a net trend), end to end, at the printed digits.

mod common;

use common::{n, num, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn column(v: &[f64]) -> Grid {
    Grid::from_rows(v.iter().map(|&x| vec![Value::Number(x)]).collect())
}

fn printed(x: f64, book: f64, dp: i32) {
    let tol = 0.5 * 10f64.powi(-dp) + 1e-9;
    assert!((x - book).abs() <= tol, "{x} is not {book} at {dp} dp");
}

fn round(x: f64, dp: i32) -> f64 {
    let s = 10f64.powi(dp);
    (x * s).round() / s
}

fn one(f: &str, args: Vec<Grid>) -> f64 {
    let g = call(f, args);
    num(g
        .as_scalar()
        .unwrap_or_else(|| panic!("{f}: {} x {}", g.rows, g.columns)))
}

#[test]
fn chapters_7_8_and_12() {
    printed(
        one(
            "pro_indicated_rate",
            vec![n(180.0), n(0.15), n(0.05), n(20.0)],
        ),
        250.0,
        2,
    );
    // Table 7.3: a column of losses gives a column of rates.
    let rates = call(
        "pro_indicated_rate",
        vec![column(&[135.0, 180.0, 225.0]), n(0.15), n(0.05), n(20.0)],
    );
    for (k, r) in [193.75, 250.0, 306.25].iter().enumerate() {
        printed(num(rates.get(k, 0)), *r, 2);
    }
    // Table 7.4: 5.9% general expenses, 75% fixed.
    let e = call("pro_expenses", vec![n(0.059), n(0.75)]);
    printed(100.0 * num(e.get(0, 1)), 4.4, 1);
    printed(100.0 * num(e.get(1, 1)), 1.5, 1);
    // Chapter 8: $500 and +10%.
    printed(
        one(
            "pro_indicated_rate",
            vec![n(300.0), n(0.25), n(0.10), n(25.0)],
        ),
        500.0,
        2,
    );
    printed(
        one(
            "pro_indicated_change",
            vec![n(0.65), n(0.25), n(0.10), n(0.065)],
        ),
        0.10,
        4,
    );
    // Chapter 12: Z 0.30 on $200 against $300; trended present rates $229.
    let z = one("pro_cred_z", vec![n(100.0), n(1082.0)]);
    printed(z, 0.30, 2);
    printed(
        one(
            "pro_cred_weighted",
            vec![n(200.0), n(round(z, 2)), n(300.0)],
        ),
        270.0,
        2,
    );
    let c = one(
        "pro_trended_present",
        vec![n(0.10), n(0.06), n(1.05f64.powi(2))],
    );
    printed(200.0 * (1.0 + c), 229.0, 0);
}

#[test]
fn appendix_a_loss_ratio() {
    let ep = [
        1_122_372.0,
        1_154_508.0,
        1_280_545.0,
        1_369_976.0,
        1_397_750.0,
    ];
    let crl = [1.2161, 1.2176, 1.1311, 1.0892, 1.0991];
    let pt = [1.1342, 1.1116, 1.0879, 1.0663, 1.0452];
    let losses = [856_495.0, 867_184.0, 835_120.0, 821_509.0, 797_866.0];
    let ldf = [1.0000, 0.9799, 1.0003, 1.0282, 1.0966];
    let lt = [0.9912, 0.9962, 1.0012, 1.0062, 1.0113];
    let premium_book = [
        1_548_088.0,
        1_562_608.0,
        1_575_741.0,
        1_591_109.0,
        1_605_706.0,
    ];
    let loss_book = [970_359.0, 967_578.0, 955_974.0, 971_450.0, 1_011_357.0];
    let ex = call(
        "pro_lr_experience",
        vec![
            column(&ep),
            column(&losses),
            column(&crl),
            column(&pt),
            column(&ldf),
            column(&lt),
            n(1.143),
        ],
    );
    assert_eq!((ex.rows, ex.columns), (7, 3), "{:?}", ex.as_scalar());
    for k in 0..5 {
        printed(num(ex.get(k + 1, 0)), premium_book[k], 0);
        printed(num(ex.get(k + 1, 1)), loss_book[k], 0);
    }
    assert!((num(ex.get(6, 0)) - 7_883_253.19).abs() < 0.01);
    printed(num(ex.get(6, 1)), 4_876_718.0, 0);
    printed(100.0 * num(ex.get(6, 2)), 61.9, 1);
    // Expense-1, each category split and rounded to 0.1%: 11.3% and 17.0%.
    let split = |r: f64, s: f64| {
        let e = call("pro_expenses", vec![n(r), n(s)]);
        (
            round(100.0 * num(e.get(0, 1)), 1),
            round(100.0 * num(e.get(1, 1)), 1),
        )
    };
    let parts = [
        split(0.063, 0.75),
        split(0.085, 0.75),
        split(0.002, 1.0),
        split(0.112, 0.0),
        split(0.021, 0.0),
    ];
    let (f, v) = parts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    printed(f, 11.3, 1);
    printed(v, 17.0, 1);
    printed(
        100.0 * one("pro_permissible_lr", vec![n(0.170), n(0.05)]),
        78.0,
        1,
    );
    let indicated = one(
        "pro_indicated_change",
        vec![n(0.619), n(0.170), n(0.05), n(0.113)],
    );
    printed(100.0 * indicated, -6.2, 1);
    let complement = one(
        "pro_trended_present",
        vec![n(0.132), n(0.05), n(1.005 / 1.02)],
    );
    printed(100.0 * complement, 6.2, 1);
    printed(
        100.0
            * one(
                "pro_cred_weighted",
                vec![n(indicated), n(1.0), n(complement)],
            ),
        -6.2,
        1,
    );
}

#[test]
fn appendix_b_pure_premium() {
    let exposure = [12_760.0, 12_766.0, 12_805.0, 12_834.0, 13_411.0];
    let losses = [
        5_161_624.0,
        4_820_968.0,
        4_112_172.0,
        5_052_052.0,
        6_559_224.0,
    ];
    let ldf = [1.0000, 1.0012, 1.0055, 1.0186, 1.0555];
    let lt = [1.1939, 1.1705, 1.1476, 1.1251, 1.1030];
    let book = [
        6_236_412.0,
        5_717_511.0,
        4_802_025.0,
        5_859_265.0,
        7_727_993.0,
    ];
    let ex = call(
        "pro_pp_experience",
        vec![
            column(&exposure),
            column(&losses),
            column(&ldf),
            column(&lt),
            n(1.012),
        ],
    );
    for (k, b) in book.iter().enumerate() {
        printed(num(ex.get(k + 1, 1)), *b, 0);
    }
    printed(num(ex.get(6, 1)), 30_343_206.0, 0);
    printed(num(ex.get(6, 2)), 469.88, 2);
    let z = one("pro_cred_z", vec![n(683.0), n(1_082.0)]);
    printed(z, 0.795, 3);
    printed(
        one(
            "pro_cred_weighted",
            vec![n(469.88), n(round(z, 3)), n(585.75)],
        ),
        493.63,
        2,
    );
    printed(
        one(
            "pro_indicated_rate",
            vec![
                n(493.63 + 29.11 + 74.57 + 15.68),
                n(0.138),
                n(0.05),
                n(77.83),
            ],
        ),
        850.76,
        2,
    );
}

#[test]
fn appendix_c_loss_ratio_net_trend() {
    let ep = [
        14_904_664.0,
        14_494_543.0,
        14_442_449.0,
        14_834_605.0,
        18_265_093.0,
    ];
    let crl = [1.2029, 1.2058, 1.2724, 1.3018, 1.2391];
    let losses = [
        11_673_500.0,
        11_200_835.0,
        6_290_368.0,
        18_254_793.0,
        23_371_444.0,
    ];
    let nt = [1.7902, 1.6439, 1.5095, 1.3862, 1.2729];
    let ex = call(
        "pro_lr_experience",
        vec![
            column(&ep),
            column(&losses),
            column(&crl),
            Grid::missing(),
            Grid::missing(),
            column(&nt),
        ],
    );
    printed(num(ex.get(6, 0)), 95_726_878.0, 0);
    printed(num(ex.get(6, 1)), 103_860_568.0, 0);
    printed(100.0 * num(ex.get(6, 2)), 108.5, 1);
    let e = call(
        "pro_expenses",
        vec![column(&[0.064, 0.026, 0.021, 0.212, 0.024])],
    );
    assert_eq!(num(e.get(0, 1)), 0.0);
    let variable = num(e.get(1, 1));
    printed(100.0 * variable, 34.7, 1);
    printed(
        100.0 * one("pro_permissible_lr", vec![n(variable), n(-0.05)]),
        70.3,
        1,
    );
    let indicated = one(
        "pro_indicated_change",
        vec![n(1.085), n(variable), n(-0.05)],
    );
    printed(100.0 * indicated, 54.3, 1);
    let z = one("pro_cred_z", vec![n(283.0), n(683.0)]);
    printed(z, 0.644, 3);
    printed(
        100.0
            * one(
                "pro_cred_weighted",
                vec![n(round(indicated, 3)), n(round(z, 3)), n(0.185)],
            ),
        41.6,
        1,
    );
}

#[test]
fn errors_come_back_as_text() {
    let e = |f: &str, a: Vec<Grid>| text(&call(f, a)).to_owned();
    assert!(
        e(
            "pro_lr_experience",
            vec![column(&[1.0, 2.0]), column(&[1.0])]
        )
        .contains("one value per year")
    );
    assert!(
        e(
            "pro_lr_experience",
            vec![
                column(&[1.0, 2.0]),
                column(&[1.0, 1.0]),
                column(&[1.0, 1.0, 1.0])
            ]
        )
        .contains("one per year")
    );
    assert!(
        e("pro_lr_experience", vec![column(&[-1.0]), column(&[1.0])]).contains("earned_premium")
    );
    assert!(e("pro_indicated_change", vec![n(0.6), n(0.7), n(0.4)]).starts_with("#PROSPICIO!"));
    assert!(e("pro_cred_weighted", vec![n(1.0), n(1.5), n(2.0)]).contains("[0, 1]"));
    assert!(
        e(
            "pro_expenses",
            vec![column(&[0.1, 0.2]), column(&[0.5, 0.5, 0.5])]
        )
        .contains("one per ratio")
    );
}
