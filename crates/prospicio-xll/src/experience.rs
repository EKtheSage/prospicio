//! Experience rating from large losses (`docs/design/xll.md`): the
//! burning cost of a per-risk or per-occurrence program, after Clark's
//! *Basics of Reinsurance Pricing*, over `prospicio_pricing::experience`.
//! Each loss is trended to the prospective period, each accident year's
//! trended losses go through the layers, the layer totals are developed
//! and divided by the year's on-level premium.

use prospicio_aggregate::{Layer, Tower};
use prospicio_pricing::experience::{Experience, LargeLoss, Trend, YearBasis};

use crate::pricing::{Result, amount, scalar};
use crate::ratemaking::decimal_year;
use crate::sheet::{Grid, Value};

/// Every cell of a range as a number, `"unlimited"` allowed when asked.
fn cells(g: &Grid, name: &str, unlimited: bool) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(g.cells.len());
    for r in 0..g.rows {
        for c in 0..g.columns {
            match amount(g.get(r, c), name, r, c, unlimited)? {
                Some(x) => out.push(x),
                None => return Err(format!("{name} must have a value in every cell")),
            }
        }
    }
    Ok(out)
}

/// One value for all `n`, one per `unit`, or `default` when omitted.
fn each(g: &Grid, name: &str, n: usize, unit: &str, default: Option<f64>) -> Result<Vec<f64>> {
    if g.is_blank() {
        return default
            .map(|d| vec![d; n])
            .ok_or(format!("{name} is required"));
    }
    let v = cells(g, name, false)?;
    match v.len() {
        1 => Ok(vec![v[0]; n]),
        k if k == n => Ok(v),
        k => Err(format!(
            "{name} must be one value, or one per {unit} ({n}), got {k}"
        )),
    }
}

/// The burning cost by accident year:
/// `PRO.BURNINGCOST(accident, amounts, limits, attachments, years,
/// premiums, [development], [trend], [trend_to], [threshold],
/// [agg_deductible], [agg_limit])`. Each year gets its premium, its count
/// of losses through the layers and its development factor, then each
/// layer's trended layer loss, developed loss and loss cost; the total row
/// is the developed losses over the premium summed, and the last row each
/// layer's loss cost averaged over the years with equal weight.
pub fn burning_cost(args: &[Grid]) -> Result<Grid> {
    let accident: Vec<f64> = cells(&args[0], "accident", false)?
        .into_iter()
        .map(decimal_year)
        .collect::<Result<_>>()?;
    let amounts = cells(&args[1], "amounts", false)?;
    let n = accident.len();
    if amounts.len() != n {
        return Err(format!(
            "amounts must have one value per accident date ({n}), got {}",
            amounts.len()
        ));
    }
    let limits = cells(&args[2], "limits", true)?;
    let attachments = each(&args[3], "attachments", limits.len(), "limit", None)?;
    let years = cells(&args[4], "years", false)?;
    let m = years.len();
    let premiums = each(&args[5], "premiums", m, "year", None)?;
    let development = each(&args[6], "development", m, "year", Some(1.0))?;
    let trend = if args[7].cells.len() > 1 {
        let f = cells(&args[7], "trend", false)?;
        if f.len() != n {
            return Err(format!(
                "trend as factors needs one per loss ({n}), got {}",
                f.len()
            ));
        }
        Trend::Factors(f)
    } else {
        let rate = if args[7].is_blank() {
            0.0
        } else {
            scalar(&args[7], "trend")?
        };
        if rate == 0.0 && args[8].is_blank() {
            Trend::Factors(vec![1.0; n])
        } else {
            let to = decimal_year(scalar(&args[8], "trend_to")?)?;
            Trend::Rate { rate, to }
        }
    };
    let layers = limits.len();
    let aad = each(&args[10], "agg_deductible", layers, "layer", Some(0.0))?;
    let aal = each(&args[11], "agg_limit", layers, "layer", Some(f64::INFINITY))?;

    let e = |e: prospicio_core::Error| e.to_string();
    let losses = accident
        .iter()
        .zip(&amounts)
        .enumerate()
        .map(|(k, (&accident, &amount))| LargeLoss {
            id: format!("{}", k + 1),
            accident,
            amount,
        })
        .collect();
    let mut experience = Experience::new(losses).map_err(e)?;
    if !args[9].is_blank() {
        experience = experience
            .with_threshold(scalar(&args[9], "threshold")?)
            .map_err(e)?;
    }
    let prepared = experience.prepare(&trend).map_err(e)?;
    let mut names = Vec::with_capacity(layers);
    let mut tower = Vec::with_capacity(layers);
    for j in 0..layers {
        let name = layer_name(limits[j], attachments[j]);
        let mut layer = Layer::xol(name.clone(), limits[j], attachments[j]).map_err(e)?;
        if aad[j] > 0.0 {
            layer = layer.aggregate_deductible(aad[j]).map_err(e)?;
        }
        if aal[j].is_finite() {
            layer = layer.aggregate_limit(aal[j]).map_err(e)?;
        }
        names.push(name);
        tower.push(layer);
    }
    let tower = Tower::new(tower).map_err(e)?;
    let basis: Vec<YearBasis> = (0..m)
        .map(|k| {
            let y = years[k];
            if y.fract() != 0.0 {
                return Err(format!("years must be whole accident years, got {y}"));
            }
            Ok(YearBasis {
                year: y as i32,
                premium: premiums[k],
                development: development[k],
            })
        })
        .collect::<Result<_>>()?;
    let b = prepared.burning_cost(&vec![tower; m], &basis).map_err(e)?;

    let mut header = ["Year", "Premium", "Losses", "Development"]
        .map(Value::from)
        .to_vec();
    for name in &names {
        for what in ["layer loss", "developed", "loss cost"] {
            header.push(Value::Text(format!("{name} {what}")));
        }
    }
    let mut rows = vec![header];
    for y in &b.years {
        let mut row = vec![
            Value::Number(f64::from(y.year)),
            Value::number(y.premium),
            Value::Number(y.losses as f64),
            Value::number(y.development),
        ];
        for j in 0..layers {
            row.extend([
                Value::number(y.layer_loss[j]),
                Value::number(y.developed[j]),
                Value::number(y.loss_cost[j]),
            ]);
        }
        rows.push(row);
    }
    let mut total = vec![
        Value::from("Total"),
        Value::number(b.years.iter().map(|y| y.premium).sum()),
        Value::Number(b.years.iter().map(|y| y.losses).sum::<usize>() as f64),
        Value::Empty,
    ];
    let mut mean = vec![
        Value::from("Mean of years"),
        Value::Empty,
        Value::Empty,
        Value::Empty,
    ];
    for j in 0..layers {
        total.extend([
            Value::number(b.years.iter().map(|y| y.layer_loss[j]).sum()),
            Value::number(b.years.iter().map(|y| y.developed[j]).sum()),
            Value::number(b.loss_cost[j]),
        ]);
        mean.extend([
            Value::Empty,
            Value::Empty,
            Value::number(b.mean_loss_cost[j]),
        ]);
    }
    rows.push(total);
    rows.push(mean);
    Ok(Grid::from_rows(rows))
}

/// `"400,000 xs 100,000"`, or `"unlimited xs 1,000,000"`.
fn layer_name(limit: f64, attachment: f64) -> String {
    let fmt = |x: f64| {
        if x.is_infinite() {
            return "unlimited".to_owned();
        }
        let s = format!("{}", x.round() as i64);
        let mut out = String::new();
        for (k, ch) in s.chars().enumerate() {
            if k > 0 && (s.len() - k) % 3 == 0 && ch != '-' {
                out.push(',');
            }
            out.push(ch);
        }
        out
    };
    format!("{} xs {}", fmt(limit), fmt(attachment))
}

#[cfg(test)]
mod tests {
    #[test]
    fn layer_names() {
        assert_eq!(
            super::layer_name(400_000.0, 100_000.0),
            "400,000 xs 100,000"
        );
        assert_eq!(
            super::layer_name(f64::INFINITY, 1e6),
            "unlimited xs 1,000,000"
        );
        assert_eq!(super::layer_name(500.0, 0.0), "500 xs 0");
    }
}
