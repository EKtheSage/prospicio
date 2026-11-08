//! Recoveries, ALAE and ULAE (`docs/design/xll.md`, decision 53):
//! `prospicio_reserving`'s `RatioDevelopment` (Friedland's ratio approach
//! for salvage and subrogation, chapter 14, and for ALAE, chapter 16) and
//! `Ulae` (the classical technique, Kittel's refinement and Conger and
//! Nolibos' generalized approach, chapter 17).

use prospicio_core::Lag;
use prospicio_reserving::{Average, RatioDevelopment, RatioFactors, Ulae, UlaeInputs, UlaeWeights};

use crate::counts::measures;
use crate::reserving::{development_with, link_labels, origin_labels, tail, text_option};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

const NUMERATOR: &str = "numerator";
const DENOMINATOR: &str = "denominator";

fn cell(x: f64) -> Value {
    if x.is_finite() {
        Value::number(x)
    } else {
        Value::Empty
    }
}

/// A row or column of numbers; `name` is required.
fn numbers(grid: &Grid, name: &str) -> Result<Vec<f64>> {
    if grid.is_blank() {
        return Err(format!("{name} is required"));
    }
    if grid.rows != 1 && grid.columns != 1 {
        return Err(format!("{name} must be a row or a column"));
    }
    grid.cells
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            Value::Number(x) => Ok(*x),
            other => Err(format!("{name} {} must be a number, got {other:?}", i + 1)),
        })
        .collect()
}

/// Optional numbers: empty when omitted.
fn numbers_or_empty(grid: &Grid, name: &str) -> Result<Vec<f64>> {
    if grid.is_blank() {
        Ok(Vec::new())
    } else {
        numbers(grid, name)
    }
}

/// `PRO.RES.RATIO(numerator, denominator, ultimate_claims, [average],
/// [tail], [selected], [kind], [factors], [ultimate_ratios], [origins],
/// [output])`: Friedland's ratio approach. The ratio of two triangles
/// (received salvage and subrogation over paid claims, paid ALAE over paid
/// claims) is developed to ultimate, by its age-to-age factors or, with
/// `factors` `"additive"`, by its age-to-age differences (Wiser), and
/// applied to the ultimate claims. `average` is the simple average when
/// omitted, as Friedland's; `ultimate_ratios` replaces the projected ratio
/// where given. `output` is `"ultimate"` (default: the table per origin),
/// `"factors"` (the averages, selected factors and factors to ultimate, a
/// column per link) or `"ratios"` (the ratio triangle).
pub fn ratio(args: &[Grid]) -> Result<Grid> {
    let tri = measures(&[(NUMERATOR, &args[0]), (DENOMINATOR, &args[1])], &[])?;
    let origins = tri.origins();
    let n = origins.len();
    let claims = numbers(&args[2], "ultimate_claims")?;
    if claims.len() != n {
        return Err(format!(
            "ultimate_claims must have {n} values, one per triangle row, got {}",
            claims.len()
        ));
    }
    let additive = match text_option(&args[7], "factors")?.as_deref() {
        None | Some("multiplicative") => false,
        Some("additive") => true,
        Some(other) => {
            return Err(format!(
                "factors must be \"multiplicative\" (default) or \"additive\", got {other:?}"
            ));
        }
    };
    let (development, factors, tail) = if additive {
        if !args[6].is_blank() {
            return Err(
                "kind applies to multiplicative factors; additive selections are differences"
                    .into(),
            );
        }
        let mut selected = Vec::new();
        if !args[5].is_blank() {
            if args[5].rows != 1 && args[5].columns != 1 {
                return Err(
                    "selected must be a row or a column, one cell per triangle column".into(),
                );
            }
            for (k, v) in args[5].cells.iter().enumerate() {
                match v {
                    Value::Number(x) => selected.push((12 * (k as Lag + 1), *x)),
                    v if v.is_blank() => {}
                    other => {
                        return Err(format!(
                            "selected {} must be a number, got {other:?}",
                            k + 1
                        ));
                    }
                }
            }
        }
        let addition = crate::reserving::number_or(&args[4], "tail", 0.0)?;
        let mut dev = development_with(
            &args[3],
            [
                &Grid::missing(),
                &Grid::missing(),
                &Grid::missing(),
                &Grid::missing(),
            ],
            &Grid::missing(),
        )?;
        if args[3].is_blank() {
            dev.average = Average::Simple;
        }
        (
            dev,
            RatioFactors::Additive {
                selected,
                tail: addition,
            },
            prospicio_reserving::Tail::default(),
        )
    } else {
        let mut dev = development_with(
            &args[3],
            [&args[5], &args[6], &Grid::missing(), &Grid::missing()],
            &Grid::missing(),
        )?;
        if args[3].is_blank() {
            dev.average = Average::Simple;
        }
        (dev, RatioFactors::Multiplicative, tail(&args[4])?)
    };
    let selected = if args[8].is_blank() {
        Vec::new()
    } else {
        if args[8].cells.len() != n || (args[8].rows != 1 && args[8].columns != 1) {
            return Err(format!(
                "ultimate_ratios must be a row or column of {n} cells, one per triangle row (blank to keep the projection)"
            ));
        }
        let mut s = Vec::new();
        for (o, v) in args[8].cells.iter().enumerate() {
            match v {
                Value::Number(x) => s.push((origins[o], *x)),
                v if v.is_blank() => {}
                other => {
                    return Err(format!(
                        "ultimate_ratios {} must be a number or blank, got {other:?}",
                        o + 1
                    ));
                }
            }
        }
        s
    };
    let fit = RatioDevelopment {
        development,
        tail,
        factors,
        selected,
    }
    .fit(&tri, NUMERATOR, DENOMINATOR, &claims)
    .map_err(|e| e.to_string())?;
    match text_option(&args[10], "output")?.as_deref() {
        None | Some("ultimate") => {
            let labels = origin_labels(&args[9], n)?;
            let unpaid = fit.unpaid();
            let mut rows = vec![
                [
                    "Origin",
                    "Latest ratio",
                    "Projected ratio",
                    "Selected ratio",
                    "Ultimate claims",
                    "Latest",
                    "Ultimate",
                    "Unpaid",
                ]
                .map(Value::from)
                .to_vec(),
            ];
            for o in 0..n {
                rows.push(vec![
                    labels[o].clone(),
                    cell(fit.latest_ratio[o]),
                    cell(fit.projected_ratio[o]),
                    cell(fit.selected_ratio[o]),
                    cell(fit.ultimate_claims[o]),
                    cell(fit.latest[o]),
                    cell(fit.ultimate[o]),
                    cell(unpaid[o]),
                ]);
            }
            let sum = |v: &[f64]| cell(v.iter().sum());
            rows.push(vec![
                Value::from("Total"),
                Value::Empty,
                Value::Empty,
                Value::Empty,
                sum(&fit.ultimate_claims),
                sum(&fit.latest),
                sum(&fit.ultimate),
                sum(&unpaid),
            ]);
            Ok(Grid::from_rows(rows))
        }
        Some("factors") => {
            let ages = fit.ages.len();
            let mut header = vec![Value::Empty];
            header.extend(link_labels(ages));
            let mut average = vec![Value::from("Average")];
            average.extend(fit.averages.iter().map(|&x| cell(x)));
            average.push(Value::Empty);
            let mut selected = vec![Value::from("Selected")];
            selected.extend(fit.factors.iter().map(|&x| cell(x)));
            // The tail: the last age's development to ultimate.
            selected.push(cell(fit.cdf[ages - 1]));
            let mut cdf = vec![Value::from("To ultimate")];
            cdf.extend(fit.cdf.iter().map(|&x| cell(x)));
            Ok(Grid::from_rows(vec![header, average, selected, cdf]))
        }
        Some("ratios") => Ok(Grid::from_rows(
            fit.ratios
                .iter()
                .map(|r| r.iter().map(|v| v.map_or(Value::Empty, cell)).collect())
                .collect(),
        )),
        Some(other) => Err(format!(
            "output must be \"ultimate\", \"factors\" or \"ratios\", got {other:?}"
        )),
    }
}

/// `PRO.RES.ULAE(paid_ulae, paid, case_outstanding, ibnr, [technique],
/// [ratio], [weights], [reported], [closed], [pure_ibnr], [unclosed],
/// [ultimate], [output])`: unpaid ULAE by Friedland's chapter 17. The
/// calendar years' paid ULAE over a claims basis gives a ratio (the ratio
/// of the totals unless one is selected), applied to what remains of the
/// claims, a row per valuation row (one row of totals, or one per accident
/// year). `technique` is `"classical"` (default; `weights` the share spent
/// opening a claim, 0.5 by default), `"kittel"` (`reported` the calendar
/// incurred claims) or `"conger_nolibos"` (`weights` U1 and U2, or U1, U2
/// and U3). Mango and Allen's refinement is the classical technique on
/// expected paid claims. `output` is `"summary"` (default), `"years"` (paid
/// ULAE, basis and ratio by calendar year) or `"rows"` (unpaid ULAE by
/// valuation row).
pub fn ulae(args: &[Grid]) -> Result<Grid> {
    let ratio = match args[5].as_scalar() {
        Some(v) if v.is_blank() => None,
        _ => Some(crate::reserving::number(&args[5], "ratio")?),
    };
    let weights = numbers_or_empty(&args[6], "weights")?;
    let technique = match text_option(&args[4], "technique")?.as_deref() {
        None | Some("classical") => match weights.as_slice() {
            [] => Ulae::classical(ratio),
            [opening] => Ulae::Classical {
                ratio,
                opening: *opening,
            },
            _ => return Err(
                "weights for the classical technique is one share, the part spent opening a claim"
                    .into(),
            ),
        },
        Some("kittel") => {
            if !weights.is_empty() {
                return Err(
                    "Kittel's refinement takes no weights: it averages paid and incurred".into(),
                );
            }
            Ulae::kittel(ratio)
        }
        Some("conger_nolibos") | Some("conger-nolibos") | Some("generalized") => {
            let w = match weights.as_slice() {
                [u1, u2] => UlaeWeights::new(*u1, *u2),
                [u1, u2, u3] => UlaeWeights {
                    opening: *u1,
                    maintaining: *u2,
                    closing: *u3,
                },
                _ => {
                    return Err(
                        "weights for Conger and Nolibos are U1 and U2, or U1, U2 and U3".into(),
                    );
                }
            };
            Ulae::conger_nolibos(w, ratio)
        }
        Some(other) => {
            return Err(format!(
                "technique must be \"classical\" (default), \"kittel\" or \"conger_nolibos\", got {other:?}"
            ));
        }
    };
    let paid_ulae = numbers(&args[0], "paid_ulae")?;
    let paid = numbers(&args[1], "paid")?;
    let case = numbers(&args[2], "case_outstanding")?;
    let ibnr = numbers(&args[3], "ibnr")?;
    let reported = numbers_or_empty(&args[7], "reported")?;
    let closed = numbers_or_empty(&args[8], "closed")?;
    let pure = numbers_or_empty(&args[9], "pure_ibnr")?;
    let unclosed = numbers_or_empty(&args[10], "unclosed")?;
    let ultimate = numbers_or_empty(&args[11], "ultimate")?;
    let fit = technique
        .estimate(&UlaeInputs {
            paid_ulae: &paid_ulae,
            paid: &paid,
            reported: &reported,
            closed: &closed,
            case_outstanding: &case,
            ibnr: &ibnr,
            pure_ibnr: (!pure.is_empty()).then_some(pure.as_slice()),
            unclosed: &unclosed,
            ultimate: &ultimate,
        })
        .map_err(|e| e.to_string())?;
    match text_option(&args[12], "output")?.as_deref() {
        None | Some("summary") => {
            let mut rows = vec![
                vec![Value::from("Total ratio"), cell(fit.total_ratio)],
                vec![Value::from("Ratio applied"), cell(fit.ratio)],
                vec![Value::from("Unpaid ULAE"), cell(fit.total())],
            ];
            for (label, v) in [
                ("Expected claims form", fit.expected_claims),
                ("Bornhuetter-Ferguson form", fit.bornhuetter_ferguson),
                ("Development form", fit.development),
            ] {
                if let Some(x) = v {
                    rows.push(vec![Value::from(label), cell(x)]);
                }
            }
            Ok(Grid::from_rows(rows))
        }
        Some("years") => {
            let mut rows = vec![["Paid ULAE", "Basis", "Ratio"].map(Value::from).to_vec()];
            for ((m, b), w) in paid_ulae.iter().zip(&fit.basis).zip(&fit.ratios) {
                rows.push(vec![cell(*m), cell(*b), cell(*w)]);
            }
            rows.push(vec![
                cell(paid_ulae.iter().sum()),
                cell(fit.basis.iter().sum()),
                cell(fit.total_ratio),
            ]);
            Ok(Grid::from_rows(rows))
        }
        Some("rows") => Ok(Grid::column(&fit.unpaid)),
        Some(other) => Err(format!(
            "output must be \"summary\", \"years\" or \"rows\", got {other:?}"
        )),
    }
}
