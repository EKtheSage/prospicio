//! `PRO.PROFILE`, an exposure-rated risk profile, reached through the
//! catalogue as Excel reaches it, against `prospicio_pricing`'s
//! `RiskProfile` (the engine behind Python's `RiskProfile` and R's) on the
//! same bands, and against closed forms: the whole loss is the expected
//! loss, a surplus cedes `min(SI - R, lines R) / SI` of each band, and a
//! per-band curve is each band's own.

mod common;

use common::{n, num, t, text};
use prospicio_pricing::exposure::Mbbefd;
use prospicio_pricing::profile::{Band, RiskProfile};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn column(v: &[f64]) -> Grid {
    Grid::from_rows(v.iter().map(|&x| vec![Value::Number(x)]).collect())
}

const SI: [f64; 4] = [1e6, 3e6, 8e6, 20e6];
const EL: [f64; 4] = [1.2e6, 0.9e6, 0.6e6, 0.3e6];

fn profile(rest: Vec<Grid>) -> Grid {
    let mut args = vec![column(&SI), column(&EL)];
    args.extend(rest);
    call("pro_profile", args)
}

#[test]
fn matches_the_engine_and_closed_forms() {
    let ex = profile(vec![n(5e6), n(2e6), t("swiss_re"), n(3.0)]);
    assert_eq!((ex.rows, ex.columns), (6, 7), "{:?}", ex.as_scalar());
    assert_eq!(ex.get(0, 5), &Value::from("Layer loss"));
    let c3 = Mbbefd::swiss_re(3.0).unwrap();
    let bands: Vec<Band> = SI
        .iter()
        .zip(EL)
        .map(|(&s, e)| Band::from_expected_loss(s, 1.0, e, c3).unwrap())
        .collect();
    let engine = RiskProfile::new(bands).unwrap();
    let total = engine.expected_layer_loss(5e6, 2e6, None).unwrap();
    assert!((num(ex.get(5, 5)) - total).abs() <= 1e-9 * total);
    assert!((num(ex.get(5, 3)) - engine.expected_claims()).abs() < 1e-9);
    // The 1m band never reaches 2m: no layer loss.
    assert_eq!(num(ex.get(1, 5)), 0.0);
    // Band 3's share is the curve's share of 5m xs 2m on an 8m risk.
    let share = num(ex.get(3, 6));
    use prospicio_pricing::exposure::ExposureCurve;
    assert!((share - c3.layer_share(5e6, 2e6, 8e6).unwrap()).abs() < 1e-12);
    // No surplus: the column is blank.
    assert_eq!(ex.get(5, 4), &Value::Empty);

    // The whole loss is the expected loss.
    let all = profile(vec![t("unlimited"), Grid::missing(), t("swiss_re"), n(3.0)]);
    assert_eq!(all.rows, 6, "{:?}", all.as_scalar());
    assert!((num(all.get(5, 5)) - 3.0e6).abs() < 1e-6);
    assert!((num(all.get(5, 6)) - 1.0).abs() < 1e-12);

    // A surplus of 3 lines on 1m: band k cedes min(SI - 1m, 3m) / SI.
    let net = profile(vec![
        n(5e6),
        n(2e6),
        t("swiss_re"),
        n(3.0),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        n(1e6),
        n(3.0),
    ]);
    assert_eq!(net.rows, 6, "{:?}", net.as_scalar());
    for (k, (&s, e)) in SI.iter().zip(EL).enumerate() {
        let want = (s - 1e6).clamp(0.0, 3e6) / s * e;
        assert!((num(net.get(k + 1, 4)) - want).abs() < 1e-6, "band {k}");
    }
    let want = engine
        .expected_layer_loss(5e6, 2e6, Some((1e6, 3.0)))
        .unwrap();
    assert!((num(net.get(5, 5)) - want).abs() <= 1e-9 * want);
}

#[test]
fn per_band_curves_and_bounds() {
    let cs = [2.0, 3.0, 4.0, 5.0];
    let ex = profile(vec![n(5e6), n(2e6), t("swiss_re"), column(&cs)]);
    let bands: Vec<Band> = SI
        .iter()
        .zip(EL)
        .zip(cs)
        .map(|((&s, e), c)| {
            Band::from_expected_loss(s, 1.0, e, Mbbefd::swiss_re(c).unwrap()).unwrap()
        })
        .collect();
    let engine = RiskProfile::new(bands.clone()).unwrap();
    let want = engine.expected_layer_loss(5e6, 2e6, None).unwrap();
    assert!((num(ex.get(5, 5)) - want).abs() <= 1e-9 * want);

    // Bounds on the last two bands, tilted to keep their sums insured.
    let lower = Grid::from_rows(vec![
        vec![Value::Empty],
        vec![Value::Empty],
        vec![Value::Number(5e6)],
        vec![Value::Number(10e6)],
    ]);
    let upper = Grid::from_rows(vec![
        vec![Value::Empty],
        vec![Value::Empty],
        vec![Value::Number(10e6)],
        vec![Value::Number(50e6)],
    ]);
    let tilted = profile(vec![
        n(5e6),
        n(2e6),
        t("swiss_re"),
        column(&cs),
        Grid::missing(),
        Grid::missing(),
        lower,
        upper,
        t("tilted"),
    ]);
    let mut b = bands;
    b[2] = b[2].clone().with_tilted_bounds(5e6, 10e6).unwrap();
    b[3] = b[3].clone().with_tilted_bounds(10e6, 50e6).unwrap();
    let engine = RiskProfile::new(b).unwrap();
    let want = engine.expected_layer_loss(5e6, 2e6, None).unwrap();
    assert!((num(tilted.get(5, 5)) - want).abs() <= 1e-9 * want);
    // Tilted keeps the mean sum insured.
    assert!((num(tilted.get(4, 1)) - 20e6).abs() < 1.0);
}

#[test]
fn errors_come_back_as_text() {
    let e = |g: Grid| text(&g).to_owned();
    assert!(
        e(call(
            "pro_profile",
            vec![
                column(&SI),
                column(&EL[..3]),
                n(1.0),
                n(0.0),
                t("swiss_re"),
                n(3.0)
            ]
        ))
        .contains("one value per band")
    );
    assert!(
        e(profile(vec![
            n(1.0),
            n(0.0),
            t("swiss_re"),
            column(&[1.0, 2.0])
        ]))
        .contains("one per band")
    );
    let mut surplus = vec![n(1.0), n(0.0), t("swiss_re"), n(3.0)];
    surplus.extend(vec![Grid::missing(); 5]);
    surplus.push(n(1e6));
    assert!(e(profile(surplus)).contains("both retention and lines"));
    assert!(
        e(profile(vec![
            n(1.0),
            n(0.0),
            t("swiss_re"),
            n(3.0),
            Grid::missing(),
            Grid::missing(),
            column(&[1.0, 1.0, 1.0, 1.0])
        ]))
        .contains("both bounds")
    );
}
