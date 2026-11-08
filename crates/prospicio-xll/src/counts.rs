//! Methods on claim counts (`docs/design/xll.md`, decision 52): Friedland's
//! frequency-severity approaches #1 and #2, the disposal rate method and
//! case outstanding development (chapters 11 and 12), the exponential
//! severity trend by age, and the Berquist-Sherman case adequacy and
//! settlement rate adjustments (chapter 13), over
//! `prospicio_reserving`'s `FrequencySeverity`, `Disposal`,
//! `CaseOutstanding`, `trend_by_age`, `CaseAdequacy`, `SettlementRate`
//! and `BerquistSherman`.

use prospicio_core::{Lag, Period};
use prospicio_reserving::{
    BerquistSherman, CaseAdequacy, CaseOutstanding, CaseTrend, ChainLadder, CountProjection,
    DevelopmentColumn, Disposal, FrequencySeverity, Grain, Interpolation, Long, OpenCounts,
    Selection, SettlementRate, SeverityProjection, Trended, Triangle, UltimateCounts, pooled_trend,
    trend_by_age,
};

use crate::reserving::{COLUMN, development_with, origin_labels, tail, text_option, triangle};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;
type Rows = Vec<Vec<Option<f64>>>;

/// Several sheet triangles observed at the same cells, as one triangle
/// with a measure per name; `per_origin` adds measures with one value per
/// origin on every observed cell of its row (an exposure).
pub fn measures(parts: &[(&str, &Grid)], per_origin: &[(&str, &[f64])]) -> Result<Triangle> {
    let (first_name, first) = parts[0];
    let mut longs = Vec::new();
    for (name, grid) in parts {
        if (grid.rows, grid.columns) != (first.rows, first.columns) {
            return Err(format!(
                "{name} must be the same size as {first_name}, {} x {}, got {} x {}",
                first.rows, first.columns, grid.rows, grid.columns
            ));
        }
        let long = triangle(grid)
            .map(|t| t.to_long())
            .map_err(|e| format!("{name}: {e}"))?;
        if let Some(l0) = longs.first() {
            let l0: &prospicio_reserving::LongTable = l0;
            if l0.origin != long.origin || l0.development != long.development {
                return Err(format!(
                    "{name} must be observed at the same cells as {first_name}"
                ));
            }
        }
        longs.push(long);
    }
    let values: Vec<Vec<f64>> = longs
        .iter()
        .map(|l| {
            l.values
                .iter()
                .find(|(n, _)| n == COLUMN)
                .map(|(_, v)| v.clone())
                .expect("a sheet triangle has its one column")
        })
        .collect();
    let origin = &longs[0].origin;
    let firsts: Vec<_> = {
        let mut o = origin.clone();
        o.dedup();
        o
    };
    let mut extra: Vec<(&str, Vec<f64>)> = Vec::new();
    for (name, v) in per_origin {
        if v.len() != firsts.len() {
            return Err(format!(
                "{name} must have {} values, one per triangle row, got {}",
                firsts.len(),
                v.len()
            ));
        }
        extra.push((
            name,
            origin
                .iter()
                .map(|o| v[firsts.iter().position(|f| f == o).expect("an origin")])
                .collect(),
        ));
    }
    let mut columns: Vec<(&str, &[f64])> = parts
        .iter()
        .zip(&values)
        .map(|((n, _), v)| (*n, v.as_slice()))
        .collect();
    columns.extend(extra.iter().map(|(n, v)| (*n, v.as_slice())));
    Triangle::from_long(&Long {
        keys: &[],
        origin,
        development: DevelopmentColumn::Age(&longs[0].development),
        values: &columns,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .map_err(|e| e.to_string())
}

/// The cells of `column` as a sheet range, origins down and ages across,
/// blank where unobserved.
fn range(tri: &Triangle, column: &str) -> Result<Grid> {
    let c = tri.column_position(column).map_err(|e| e.to_string())?;
    let [_, _, no, nd] = tri.shape();
    Ok(Grid::from_rows(
        (0..no)
            .map(|o| {
                (0..nd)
                    .map(|d| tri.get(0, c, o, d).map_or(Value::Empty, Value::number))
                    .collect()
            })
            .collect(),
    ))
}

/// Rows of optional numbers as a sheet range.
fn square(rows: &Rows) -> Grid {
    Grid::from_rows(
        rows.iter()
            .map(|r| {
                r.iter()
                    .map(|v| {
                        v.filter(|x| x.is_finite())
                            .map_or(Value::Empty, Value::number)
                    })
                    .collect()
            })
            .collect(),
    )
}

/// A row of numbers, blank where not finite.
fn row(values: &[f64]) -> Grid {
    Grid::from_rows(vec![
        values
            .iter()
            .map(|&x| {
                if x.is_finite() {
                    Value::number(x)
                } else {
                    Value::Empty
                }
            })
            .collect(),
    ])
}

/// A row or column of `n` numbers, one per origin.
fn per_origin(grid: &Grid, name: &str, n: usize) -> Result<Vec<f64>> {
    if grid.is_blank() {
        return Err(format!("{name} is required: one value per triangle row"));
    }
    if grid.cells.len() != n || (grid.rows != 1 && grid.columns != 1) {
        return Err(format!(
            "{name} must be a row or column of {n} values, one per triangle row"
        ));
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

/// Optional per-origin factors: empty when omitted.
fn factors(grid: &Grid, name: &str, n: usize) -> Result<Vec<f64>> {
    if grid.is_blank() {
        Ok(Vec::new())
    } else {
        per_origin(grid, name, n)
    }
}

/// An optional number with a default.
fn number_or(grid: &Grid, name: &str, default: f64) -> Result<f64> {
    crate::reserving::number_or(grid, name, default)
}

/// How values are selected at each age:
///
/// - blank or `"all"`: the simple average of every origin;
/// - a number `n` or `"latest n"`: the simple average of the latest `n`;
/// - `"medial n"` (one dropped each side) or `"medial nxk"` (`k` each side);
/// - `"diagonal"`: the latest diagonal's value, as Berquist and Sherman;
/// - a row or column: the values themselves, one per triangle column from
///   12 months (blank where not given).
pub(crate) fn selection(grid: &Grid, name: &str) -> Result<Selection> {
    if grid.is_blank() {
        return Ok(Selection::default());
    }
    if let Some(v) = grid.as_scalar() {
        return match v {
            Value::Number(x) => latest(*x, name),
            Value::Text(s) => selection_text(s, name),
            other => Err(format!(
                "{name} must be a number, text or a row, got {other:?}"
            )),
        };
    }
    if grid.rows != 1 && grid.columns != 1 {
        return Err(format!(
            "{name} must be a row or column of values, one per triangle column"
        ));
    }
    let mut values = Vec::new();
    for (k, v) in grid.cells.iter().enumerate() {
        match v {
            Value::Number(x) => values.push((12 * (k as Lag + 1), *x)),
            v if v.is_blank() => {}
            other => {
                return Err(format!("{name} {} must be a number, got {other:?}", k + 1));
            }
        }
    }
    Ok(Selection::Given(values))
}

fn latest(x: f64, name: &str) -> Result<Selection> {
    if x < 1.0 || x.fract() != 0.0 {
        return Err(format!(
            "{name} must be a whole number of origins of at least 1, got {x}"
        ));
    }
    Ok(Selection::latest(x as usize))
}

fn selection_text(s: &str, name: &str) -> Result<Selection> {
    let s = s.trim().to_ascii_lowercase();
    let whole = |t: &str| -> Result<usize> {
        t.trim()
            .parse::<usize>()
            .ok()
            .filter(|n| *n >= 1)
            .ok_or_else(|| format!("{name}: {t:?} is not a whole number of at least 1"))
    };
    match s.as_str() {
        "all" | "average" => Ok(Selection::default()),
        "diagonal" | "latest diagonal" => Ok(Selection::LatestDiagonal),
        _ => {
            if let Some(n) = s.strip_prefix("latest ") {
                Ok(Selection::latest(whole(n)?))
            } else if let Some(rest) = s.strip_prefix("medial ") {
                let (n, k) = match rest.split_once('x') {
                    Some((n, k)) => (
                        whole(n)?,
                        k.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("{name}: {k:?} is not a whole number"))?,
                    ),
                    None => (whole(rest)?, 1),
                };
                Ok(Selection::Average {
                    n_periods: Some(n),
                    drop_high: k,
                    drop_low: k,
                })
            } else {
                Err(format!(
                    "{name} must be \"all\", \"latest n\", \"medial n\", \"medial nxk\", \"diagonal\", a number or the values by age, got {s:?}"
                ))
            }
        }
    }
}

/// The origin a base label names among the origin labels; the latest
/// origin when omitted.
fn base(grid: &Grid, labels: &[Value], origins: &[Period]) -> Result<Period> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(*origins.last().expect("a triangle has origins")),
        Some(v) => labels
            .iter()
            .position(|l| l == v)
            .map(|k| origins[k])
            .ok_or_else(|| {
                format!("base {v:?} is not one of the origins (their labels, or 1, 2, ... when origins is omitted)")
            }),
        None => Err("base must be a single origin label".into()),
    }
}

fn origins_of(tri: &Triangle) -> Vec<Period> {
    tri.origins()
}

/// The rows of a summary table with a total row: `columns` holds each
/// column's per-origin values and whether it is totalled.
fn table(labels: &[Value], header: &[&str], columns: &[(&[f64], bool)]) -> Grid {
    let mut rows = vec![
        std::iter::once(Value::from("Origin"))
            .chain(header.iter().map(|h| Value::from(*h)))
            .collect::<Vec<_>>(),
    ];
    let cell = |x: f64| {
        if x.is_finite() {
            Value::number(x)
        } else {
            Value::Empty
        }
    };
    for (o, label) in labels.iter().enumerate() {
        let mut r = vec![label.clone()];
        r.extend(columns.iter().map(|(v, _)| cell(v[o])));
        rows.push(r);
    }
    let mut total = vec![Value::from("Total")];
    total.extend(columns.iter().map(|(v, sum)| {
        if *sum {
            cell(v.iter().filter(|x| x.is_finite()).sum())
        } else {
            Value::Empty
        }
    }));
    rows.push(total);
    Grid::from_rows(rows)
}

fn err<E: ToString>(e: E) -> String {
    e.to_string()
}

const AMOUNTS: &str = "amounts";
const COUNTS: &str = "counts";
const EXPOSURE: &str = "exposure";

// ---------------------------------------------------------------------------
// Frequency-severity.

/// `PRO.RES.FREQSEV(amounts, counts, ultimate_counts, [average], [tail],
/// [selected], [kind], [origins])`: Friedland's approach #1, the ultimate
/// counts given per origin times the ultimate severity, the chain ladder
/// on the severities (amounts over counts, cell by cell) with its settings.
pub fn frequency_severity(args: &[Grid]) -> Result<Grid> {
    let tri = measures(&[(AMOUNTS, &args[0]), (COUNTS, &args[1])], &[])?;
    let n = origins_of(&tri).len();
    let counts = per_origin(&args[2], "ultimate_counts", n)?;
    let development = development_with(
        &args[3],
        [&args[5], &args[6], &Grid::missing(), &Grid::missing()],
        &Grid::missing(),
    )?;
    let method = FrequencySeverity {
        amounts: AMOUNTS.into(),
        counts: COUNTS.into(),
        ultimate_counts: CountProjection::Counts(UltimateCounts::Given(counts)),
        severity: SeverityProjection::Development(ChainLadder {
            development,
            tail: tail(&args[4])?,
        }),
    };
    severity_table(&method, &tri, &args[7])
}

/// `PRO.RES.FREQSEV.TRENDED(amounts, counts, exposure, frequency, severity,
/// [base], [frequency_trend], [severity_trend], [exposure_factors],
/// [frequency_factors], [severity_factors], [origins])`: Friedland's
/// approach #2, a frequency and a severity selected at the base origin's
/// level and trended to the others; the counts are the exposure times the
/// trended frequency.
pub fn frequency_severity_trended(args: &[Grid]) -> Result<Grid> {
    let rows = args[0].rows;
    let exposure = per_origin(&args[2], "exposure", rows)?;
    let tri = measures(
        &[(AMOUNTS, &args[0]), (COUNTS, &args[1])],
        &[(EXPOSURE, &exposure)],
    )?;
    let origins = origins_of(&tri);
    let labels = origin_labels(&args[11], origins.len())?;
    let base = base(&args[5], &labels, &origins)?;
    let trended = |selected: &Grid, name: &str, trend: &Grid, f: &Grid, fname: &str| {
        Ok::<_, String>(Trended {
            selected: crate::reserving::number(selected, name)?,
            base,
            trend: number_or(trend, &format!("{name}_trend"), 0.0)?,
            factors: factors(f, fname, origins.len())?,
        })
    };
    let method = FrequencySeverity {
        amounts: AMOUNTS.into(),
        counts: COUNTS.into(),
        ultimate_counts: CountProjection::Frequency {
            exposure: EXPOSURE.into(),
            exposure_factors: factors(&args[8], "exposure_factors", origins.len())?,
            frequency: trended(
                &args[3],
                "frequency",
                &args[6],
                &args[9],
                "frequency_factors",
            )?,
        },
        severity: SeverityProjection::Trended(trended(
            &args[4],
            "severity",
            &args[7],
            &args[10],
            "severity_factors",
        )?),
    };
    severity_table(&method, &tri, &args[11])
}

fn severity_table(method: &FrequencySeverity, tri: &Triangle, origins: &Grid) -> Result<Grid> {
    let fit = method.fit(tri).map_err(err)?;
    let labels = origin_labels(origins, fit.origins.len())?;
    let reserves = fit.reserves();
    Ok(table(
        &labels,
        &[
            "Latest",
            "Latest severity",
            "Ultimate counts",
            "Ultimate severity",
            "Ultimate",
            "Reserve",
        ],
        &[
            (&fit.latest, true),
            (&fit.latest_severity, false),
            (&fit.ultimate_counts, true),
            (&fit.ultimate_severity, false),
            (&fit.ultimate, true),
            (&reserves, true),
        ],
    ))
}

// ---------------------------------------------------------------------------
// The disposal rate method.

/// `PRO.RES.DISPOSAL(closed, paid, ultimate_counts, [rates], [severities],
/// [tail_severity_from], [trend], [base], [factors], [origins], [output])`:
/// Friedland's disposal rate method. `output` is `"ultimate"` (default:
/// the table per origin), `"rates"` (the selected disposal rates, a row
/// under the triangle's columns), `"severities"` (the selected severities
/// at the base level, the step to ultimate last), `"counts"` (incremental
/// closed counts, observed then projected) or `"payments"` (incremental
/// paid, observed then projected).
pub fn disposal(args: &[Grid]) -> Result<Grid> {
    let tri = measures(&[("closed", &args[0]), ("paid", &args[1])], &[])?;
    let origins = origins_of(&tri);
    let n = origins.len();
    let labels = origin_labels(&args[9], n)?;
    let tail_from = match args[5].as_scalar() {
        Some(v) if v.is_blank() => None,
        Some(Value::Number(x)) if *x > 0.0 && x.fract() == 0.0 => Some(*x as Lag),
        _ => return Err("tail_severity_from must be an age in months, such as 84".into()),
    };
    let base = if args[7].is_blank() {
        None
    } else {
        Some(base(&args[7], &labels, &origins)?)
    };
    let fit = Disposal {
        closed_counts: "closed".into(),
        paid: "paid".into(),
        ultimate_counts: UltimateCounts::Given(per_origin(&args[2], "ultimate_counts", n)?),
        disposal_rates: selection(&args[3], "rates")?,
        severities: selection(&args[4], "severities")?,
        tail_severity_from: tail_from,
        trend: number_or(&args[6], "trend", 0.0)?,
        base,
        factors: factors(&args[8], "factors", n)?,
    }
    .fit(&tri)
    .map_err(err)?;
    match text_option(&args[10], "output")?.as_deref() {
        None | Some("ultimate") => {
            let reserves = fit.reserves();
            Ok(table(
                &labels,
                &["Ultimate counts", "Paid", "Ultimate", "Unpaid"],
                &[
                    (&fit.ultimate_counts, true),
                    (&fit.latest, true),
                    (&fit.ultimate, true),
                    (&reserves, true),
                ],
            ))
        }
        Some("rates") => Ok(row(&fit.selected_rates)),
        Some("severities") => Ok(row(&fit.selected_severities)),
        Some("counts") => Ok(square(&fit.incremental_counts)),
        Some("payments") => Ok(square(&fit.incremental_paid)),
        Some(other) => Err(format!(
            "output must be \"ultimate\", \"rates\", \"severities\", \"counts\" or \"payments\", got {other:?}"
        )),
    }
}

// ---------------------------------------------------------------------------
// Case outstanding development.

/// `PRO.RES.CASE.OS(paid, incurred, [paid_on_case], [case_on_case],
/// [paid_to_ultimate], [origins], [output])`: Friedland's case outstanding
/// development (chapter 12, approach #1). `output` is `"ultimate"`
/// (default: the table per origin), `"paid_on_case"` or `"case_on_case"`
/// (the selected ratios, a row under the triangle's columns), `"case"` (the
/// case outstanding, observed then projected) or `"payments"`
/// (incremental paid, observed then projected).
pub fn case_outstanding(args: &[Grid]) -> Result<Grid> {
    let tri = measures(&[("paid", &args[0]), ("incurred", &args[1])], &[])?;
    let n = origins_of(&tri).len();
    let labels = origin_labels(&args[5], n)?;
    let fit = CaseOutstanding {
        paid: "paid".into(),
        incurred: "incurred".into(),
        paid_on_case: selection(&args[2], "paid_on_case")?,
        case_on_case: selection(&args[3], "case_on_case")?,
        paid_to_ultimate: number_or(&args[4], "paid_to_ultimate", 1.0)?,
    }
    .fit(&tri)
    .map_err(err)?;
    match text_option(&args[6], "output")?.as_deref() {
        None | Some("ultimate") => {
            let (ibnr, unpaid) = (fit.ibnr(), fit.reserves());
            Ok(table(
                &labels,
                &["Paid", "Incurred", "Ultimate", "IBNR", "Unpaid"],
                &[
                    (&fit.latest_paid, true),
                    (&fit.latest_incurred, true),
                    (&fit.ultimate, true),
                    (&ibnr, true),
                    (&unpaid, true),
                ],
            ))
        }
        Some("paid_on_case") => Ok(row(&fit.selected_paid_on_case)),
        Some("case_on_case") => Ok(row(&fit.selected_case_on_case)),
        Some("case") => Ok(square(&fit.case_outstanding)),
        Some("payments") => Ok(square(&fit.incremental_paid)),
        Some(other) => Err(format!(
            "output must be \"ultimate\", \"paid_on_case\", \"case_on_case\", \"case\" or \"payments\", got {other:?}"
        )),
    }
}

// ---------------------------------------------------------------------------
// Severity trends.

/// `PRO.RES.TREND.BY.AGE(amounts, [counts])`: the exponential trend
/// across origins at each age of a triangle of severities (or of `amounts`
/// over `counts`, cell by cell), as Friedland's chapter 13 fits the average
/// case and the average paid claim, then the trend pooled over every age:
/// a row per age with the annual rate, R-squared and the points fitted.
pub fn trend_by_age_table(args: &[Grid]) -> Result<Grid> {
    let tri = if args[1].is_blank() {
        triangle(&args[0])?
    } else {
        let t = measures(&[(AMOUNTS, &args[0]), (COUNTS, &args[1])], &[])?;
        prospicio_reserving::ratio_column(&t, AMOUNTS, COUNTS, COLUMN).map_err(err)?
    };
    let by_age = trend_by_age(&tri, COLUMN).map_err(err)?;
    let pooled = pooled_trend(&tri, COLUMN).map_err(err)?;
    let mut rows = vec![
        ["Age", "Rate", "R-squared", "Points"]
            .map(Value::from)
            .to_vec(),
    ];
    let fit_row = |label: Value, fit: Option<prospicio_reserving::TrendFit>| {
        let mut r = vec![label];
        match fit {
            Some(f) => r.extend([
                Value::number(f.rate),
                Value::number(f.r_squared),
                Value::number(f.n as f64),
            ]),
            None => r.extend([Value::Empty, Value::Empty, Value::Empty]),
        }
        r
    };
    for (age, fit) in by_age {
        rows.push(fit_row(Value::number(f64::from(age)), fit));
    }
    rows.push(fit_row(Value::from("All ages"), pooled));
    Ok(Grid::from_rows(rows))
}

// ---------------------------------------------------------------------------
// Berquist-Sherman.

/// The case trend argument: an annual rate, or `"average_case"` for the
/// trend of the average case outstanding fitted over every age.
fn case_trend(grid: &Grid) -> Result<CaseTrend> {
    match grid.as_scalar() {
        Some(Value::Number(x)) => Ok(CaseTrend::Given(*x)),
        Some(Value::Text(s))
            if matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "average_case" | "average case"
            ) =>
        {
            Ok(CaseTrend::AverageCase)
        }
        Some(v) if v.is_blank() => {
            Err("trend is required: an annual rate such as 0.05, or \"average_case\"".into())
        }
        _ => Err("trend must be an annual rate such as 0.05, or \"average_case\"".into()),
    }
}

fn max_age(grid: &Grid) -> Result<Option<Lag>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(Value::Number(x)) if *x > 0.0 && x.fract() == 0.0 => Ok(Some(*x as Lag)),
        _ => Err("max_age must be an age in months, such as 84".into()),
    }
}

fn interpolation(grid: &Grid) -> Result<Interpolation> {
    match text_option(grid, "interpolation")?.as_deref() {
        None | Some("exponential") => Ok(Interpolation::Exponential),
        Some("linear") => Ok(Interpolation::Linear),
        Some(other) => Err(format!(
            "interpolation must be \"exponential\" (default) or \"linear\", got {other:?}"
        )),
    }
}

/// The rates argument of the settlement adjustment: the latest diagonal's
/// unless given otherwise.
fn settlement_rates(grid: &Grid) -> Result<Selection> {
    if grid.is_blank() {
        Ok(Selection::LatestDiagonal)
    } else {
        selection(grid, "rates")
    }
}

/// `PRO.RES.BS.CASE(incurred, paid, open, trend, [max_age], [output])`:
/// the Berquist-Sherman case adequacy adjustment. The average case
/// outstanding of the latest diagonal is carried back to earlier origins by
/// `trend`, then times the open counts plus paid; ages past `max_age` keep
/// their data. `output` is `"adjusted"` (default: the adjusted incurred
/// triangle) or `"average_case"` (the observed average case outstanding).
pub fn bs_case(args: &[Grid]) -> Result<Grid> {
    let tri = measures(
        &[
            ("incurred", &args[0]),
            ("paid", &args[1]),
            ("open", &args[2]),
        ],
        &[],
    )?;
    let case = CaseAdequacy {
        incurred: "incurred".into(),
        paid: "paid".into(),
        open_counts: OpenCounts::Column("open".into()),
        trend: case_trend(&args[3])?,
        max_age: max_age(&args[4])?,
    };
    match text_option(&args[5], "output")?.as_deref() {
        None | Some("adjusted") => range(&case.adjust(&tri).map_err(err)?, "incurred"),
        Some("average_case") => range(&case.average_case(&tri).map_err(err)?, "average_case"),
        Some(other) => Err(format!(
            "output must be \"adjusted\" or \"average_case\", got {other:?}"
        )),
    }
}

/// `PRO.RES.BS.SETTLEMENT(paid, closed, ultimate_counts, [rates],
/// [interpolation], [output])`: the Berquist-Sherman settlement rate
/// adjustment. Closed counts are restated at the selected disposal rates
/// (the latest diagonal's by default) times the ultimate counts, and paid
/// claims interpolated between the observed points. `output` is `"paid"`
/// (default) or `"closed"`: the adjusted triangle.
pub fn bs_settlement(args: &[Grid]) -> Result<Grid> {
    let tri = measures(&[("paid", &args[0]), ("closed", &args[1])], &[])?;
    let n = origins_of(&tri).len();
    let adjusted = SettlementRate {
        paid: "paid".into(),
        closed_counts: "closed".into(),
        ultimate_counts: UltimateCounts::Given(per_origin(&args[2], "ultimate_counts", n)?),
        disposal_rates: settlement_rates(&args[3])?,
        interpolation: interpolation(&args[4])?,
    }
    .adjust(&tri)
    .map_err(err)?;
    match text_option(&args[5], "output")?.as_deref() {
        None | Some("paid") => range(&adjusted, "paid"),
        Some("closed") => range(&adjusted, "closed"),
        Some(other) => Err(format!(
            "output must be \"paid\" or \"closed\", got {other:?}"
        )),
    }
}

/// `PRO.RES.BS(incurred, paid, reported_counts, closed, ultimate_counts,
/// trend, [max_age], [rates], [interpolation], [output])`: both
/// Berquist-Sherman adjustments, Friedland's third projection of XYZ
/// Insurer: paid and closed counts from the settlement adjustment, and
/// incurred as the adjusted average case times reported less adjusted
/// closed counts plus adjusted paid. `output` is `"incurred"` (default),
/// `"paid"` or `"closed"`.
pub fn bs_both(args: &[Grid]) -> Result<Grid> {
    let tri = measures(
        &[
            ("incurred", &args[0]),
            ("paid", &args[1]),
            ("reported", &args[2]),
            ("closed", &args[3]),
        ],
        &[],
    )?;
    let n = origins_of(&tri).len();
    let adjusted = BerquistSherman {
        case_adequacy: CaseAdequacy {
            incurred: "incurred".into(),
            paid: "paid".into(),
            open_counts: OpenCounts::ReportedLessClosed {
                reported: "reported".into(),
                closed: "closed".into(),
            },
            trend: case_trend(&args[5])?,
            max_age: max_age(&args[6])?,
        },
        settlement_rate: SettlementRate {
            paid: "paid".into(),
            closed_counts: "closed".into(),
            ultimate_counts: UltimateCounts::Given(per_origin(&args[4], "ultimate_counts", n)?),
            disposal_rates: settlement_rates(&args[7])?,
            interpolation: interpolation(&args[8])?,
        },
    }
    .adjust(&tri)
    .map_err(err)?;
    match text_option(&args[9], "output")?.as_deref() {
        None | Some("incurred") => range(&adjusted, "incurred"),
        Some("paid") => range(&adjusted, "paid"),
        Some("closed") => range(&adjusted, "closed"),
        Some(other) => Err(format!(
            "output must be \"incurred\", \"paid\" or \"closed\", got {other:?}"
        )),
    }
}
