//! The state of an IFRS 17 group as a JSON document
//! (`reserving-ifrs17-measurement.md`, PR 5): a group rolled forward this
//! quarter is saved and rolled forward next quarter from where it stopped.
//!
//! [`Group`], [`PaaGroup`] and [`ReinsuranceHeld`] each write everything
//! their next roll-forward reads: the locked-in curve and the options, the
//! periods rolled, the last view of the cash flows, the balances and what
//! is accumulated to date. The documents follow the fits' conventions
//! (`fit_json.rs`): `"format"` and `"format_version"`, numbers bit for bit,
//! a non-finite number as a string, an absent option as `null`.
//!
//! ```json
//! {"format": "risk_rs.ifrs17_group", "format_version": 1,
//!  "curve": {"maturities": [1.0], "rates": [0.05]},
//!  "options": {...}, "period": 1, "view": {...}, "balances": {...}, ...}
//! ```

use serde_json::{Value, json};

use super::paa::{Lic, PaaBalances, PaaGroup, PaaOptions};
use super::reinsurance::{
    CededCashFlows, NonPerformance, ReinsuranceBalances, ReinsuranceHeld, ReinsuranceOptions,
};
use super::{Balances, CashFlowTiming, ExpectedCashFlows, Group, GroupOptions};
use crate::discount::DiscountCurve;
use crate::error::{Error, Result};
use crate::fit_json::{count, document, field, floats, numbers};
use crate::triangle::json::{number_from_json, number_to_json};

/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

type Read<T> = std::result::Result<T, String>;

fn data_error(what: &str, msg: String) -> Error {
    Error::Core(prospicio_core::Error::Data(format!("{what} JSON: {msg}")))
}

fn finish(mut doc: Value, format: &str) -> String {
    let obj = doc.as_object_mut().expect("an object");
    obj.insert("format".into(), json!(format));
    obj.insert("format_version".into(), json!(FORMAT_VERSION));
    serde_json::to_string(&doc).expect("a JSON value serializes")
}

fn number(v: &Value, name: &str) -> Read<f64> {
    number_from_json(field(v, name)?).map_err(|e| format!("{name:?}: {e}"))
}

fn flag(v: &Value, name: &str) -> Read<bool> {
    field(v, name)?
        .as_bool()
        .ok_or_else(|| format!("{name:?} is not a boolean"))
}

fn optional_floats(v: &Value, name: &str) -> Read<Option<Vec<f64>>> {
    match field(v, name)? {
        Value::Null => Ok(None),
        _ => floats(v, name).map(Some),
    }
}

fn curve_value(c: &DiscountCurve) -> Value {
    json!({"maturities": numbers(c.maturities()), "rates": numbers(c.rates())})
}

fn curve_from_value(v: &Value) -> Read<DiscountCurve> {
    DiscountCurve::spot(&floats(v, "maturities")?, &floats(v, "rates")?)
        .map_err(|e| format!("a curve: {e}"))
}

fn optional_curve(v: &Value, name: &str) -> Read<Option<DiscountCurve>> {
    match field(v, name)? {
        Value::Null => Ok(None),
        c => curve_from_value(c).map(Some),
    }
}

fn timing_name(t: CashFlowTiming) -> &'static str {
    match t {
        CashFlowTiming::StartOfPeriod => "start",
        CashFlowTiming::MidPeriod => "mid",
        CashFlowTiming::EndOfPeriod => "end",
    }
}

fn timing(v: &Value, name: &str) -> Read<CashFlowTiming> {
    match field(v, name)?.as_str() {
        Some("start") => Ok(CashFlowTiming::StartOfPeriod),
        Some("mid") => Ok(CashFlowTiming::MidPeriod),
        Some("end") => Ok(CashFlowTiming::EndOfPeriod),
        _ => Err(format!("{name:?} is not \"start\", \"mid\" or \"end\"")),
    }
}

fn period(v: &Value) -> Read<usize> {
    count(field(v, "period")?, "\"period\"")
}

fn view_value(c: &ExpectedCashFlows) -> Value {
    json!({
        "premiums": numbers(&c.premiums),
        "claims": numbers(&c.claims),
        "expenses": numbers(&c.expenses),
        "acquisition": numbers(&c.acquisition),
        "investment_component": numbers(&c.investment_component),
        "premium_timing": timing_name(c.premium_timing),
        "claim_timing": timing_name(c.claim_timing),
        "acquisition_timing": timing_name(c.acquisition_timing),
    })
}

fn view_from_value(v: &Value) -> Read<ExpectedCashFlows> {
    let view = ExpectedCashFlows {
        premiums: floats(v, "premiums")?,
        claims: floats(v, "claims")?,
        expenses: floats(v, "expenses")?,
        acquisition: floats(v, "acquisition")?,
        investment_component: floats(v, "investment_component")?,
        premium_timing: timing(v, "premium_timing")?,
        claim_timing: timing(v, "claim_timing")?,
        acquisition_timing: timing(v, "acquisition_timing")?,
    };
    view.validate().map_err(|e| e.to_string())?;
    Ok(view)
}

fn balances_value(b: &Balances) -> Value {
    json!({
        "present_value": number_to_json(b.present_value),
        "present_value_locked_in": number_to_json(b.present_value_locked_in),
        "risk_adjustment": number_to_json(b.risk_adjustment),
        "csm": number_to_json(b.csm),
        "loss_component": number_to_json(b.loss_component),
    })
}

fn balances_from_value(v: &Value) -> Read<Balances> {
    Ok(Balances {
        present_value: number(v, "present_value")?,
        present_value_locked_in: number(v, "present_value_locked_in")?,
        risk_adjustment: number(v, "risk_adjustment")?,
        csm: number(v, "csm")?,
        loss_component: number(v, "loss_component")?,
    })
}

fn group_value(g: &Group) -> Value {
    let o = &g.options;
    json!({
        "curve": curve_value(&g.curve),
        "options": {
            "coverage_units": o.coverage_units.as_deref().map(numbers),
            "discount_coverage_units": o.discount_coverage_units,
            "period_years": number_to_json(o.period_years),
            "oci": o.oci,
            "accretion_curve": o.accretion_curve.as_ref().map(curve_value),
        },
        "period": g.period,
        "view": view_value(&g.view),
        "balances": balances_value(&g.balances),
        "coverage_units": numbers(&g.coverage_units),
        "acquisition_paid": number_to_json(g.acquisition_paid),
        "acquisition_recovered": number_to_json(g.acquisition_recovered),
        "recognised": g.recognised.as_ref().map(balances_value),
    })
}

fn group_from_value(v: &Value) -> Read<Group> {
    let o = field(v, "options")?;
    let options = GroupOptions {
        coverage_units: optional_floats(o, "coverage_units")?,
        discount_coverage_units: flag(o, "discount_coverage_units")?,
        period_years: number(o, "period_years")?,
        oci: flag(o, "oci")?,
        accretion_curve: optional_curve(o, "accretion_curve")?,
    };
    let recognised = match field(v, "recognised")? {
        Value::Null => None,
        b => Some(balances_from_value(b)?),
    };
    Ok(Group {
        curve: curve_from_value(field(v, "curve")?)?,
        options,
        period: period(v)?,
        view: view_from_value(field(v, "view")?)?,
        balances: balances_from_value(field(v, "balances")?)?,
        coverage_units: floats(v, "coverage_units")?,
        acquisition_paid: number(v, "acquisition_paid")?,
        acquisition_recovered: number(v, "acquisition_recovered")?,
        recognised,
    })
}

fn paa_value(g: &PaaGroup) -> Value {
    let o = &g.options;
    let b = &g.balances;
    json!({
        "options": {
            "revenue_pattern": numbers(&o.revenue_pattern),
            "expense_acquisition": o.expense_acquisition,
            "lic_curve": o.lic_curve.as_ref().map(curve_value),
            "period_years": number_to_json(o.period_years),
        },
        "expected_premiums": number_to_json(g.expected_premiums),
        "acquisition": number_to_json(g.acquisition),
        "period": g.period,
        "revenue_to_date": number_to_json(g.revenue_to_date),
        "amortised_to_date": number_to_json(g.amortised_to_date),
        "balances": {
            "lrc_excluding_loss_component": number_to_json(b.lrc_excluding_loss_component),
            "loss_component": number_to_json(b.loss_component),
            "lic_present_value": number_to_json(b.lic.present_value),
            "lic_risk_adjustment": number_to_json(b.lic.risk_adjustment),
        },
    })
}

fn paa_from_value(v: &Value) -> Read<PaaGroup> {
    let o = field(v, "options")?;
    let b = field(v, "balances")?;
    Ok(PaaGroup {
        options: PaaOptions {
            revenue_pattern: floats(o, "revenue_pattern")?,
            expense_acquisition: flag(o, "expense_acquisition")?,
            lic_curve: optional_curve(o, "lic_curve")?,
            period_years: number(o, "period_years")?,
        },
        expected_premiums: number(v, "expected_premiums")?,
        acquisition: number(v, "acquisition")?,
        period: period(v)?,
        revenue_to_date: number(v, "revenue_to_date")?,
        amortised_to_date: number(v, "amortised_to_date")?,
        balances: PaaBalances {
            lrc_excluding_loss_component: number(b, "lrc_excluding_loss_component")?,
            loss_component: number(b, "loss_component")?,
            lic: Lic {
                present_value: number(b, "lic_present_value")?,
                risk_adjustment: number(b, "lic_risk_adjustment")?,
            },
        },
    })
}

fn non_performance_value(np: &Option<NonPerformance>) -> Value {
    match np {
        Some(n) => json!({
            "default_probability": number_to_json(n.default_probability),
            "loss_given_default": number_to_json(n.loss_given_default),
        }),
        None => Value::Null,
    }
}

fn non_performance_from_value(v: &Value, name: &str) -> Read<Option<NonPerformance>> {
    match field(v, name)? {
        Value::Null => Ok(None),
        n => NonPerformance::new(
            number(n, "default_probability")?,
            number(n, "loss_given_default")?,
        )
        .map(Some)
        .map_err(|e| e.to_string()),
    }
}

fn ceded_balances_value(b: &ReinsuranceBalances) -> Value {
    json!({
        "present_value": number_to_json(b.present_value),
        "present_value_locked_in": number_to_json(b.present_value_locked_in),
        "non_performance": number_to_json(b.non_performance),
        "risk_adjustment": number_to_json(b.risk_adjustment),
        "csm": number_to_json(b.csm),
        "loss_recovery": number_to_json(b.loss_recovery),
    })
}

fn ceded_balances_from_value(v: &Value) -> Read<ReinsuranceBalances> {
    Ok(ReinsuranceBalances {
        present_value: number(v, "present_value")?,
        present_value_locked_in: number(v, "present_value_locked_in")?,
        non_performance: number(v, "non_performance")?,
        risk_adjustment: number(v, "risk_adjustment")?,
        csm: number(v, "csm")?,
        loss_recovery: number(v, "loss_recovery")?,
    })
}

fn reinsurance_value(r: &ReinsuranceHeld) -> Value {
    let o = &r.options;
    json!({
        "curve": curve_value(&r.curve),
        "options": {
            "coverage_units": o.coverage_units.as_deref().map(numbers),
            "period_years": number_to_json(o.period_years),
            "retroactive": o.retroactive,
            "non_performance": non_performance_value(&o.non_performance),
        },
        "period": r.period,
        "view": {
            "premiums": numbers(&r.view.premiums),
            "recoveries": numbers(&r.view.recoveries),
            "premium_timing": timing_name(r.view.premium_timing),
            "recovery_timing": timing_name(r.view.recovery_timing),
        },
        "non_performance": non_performance_value(&r.non_performance),
        "balances": ceded_balances_value(&r.balances),
        "coverage_units": numbers(&r.coverage_units),
        "shares": numbers(&r.shares),
        "recognised_losses": numbers(&r.recognised_losses),
        "recognised": r.recognised.as_ref().map(|(b, past)| json!({
            "balances": ceded_balances_value(b),
            "past_events_cost": number_to_json(*past),
        })),
    })
}

fn reinsurance_from_value(v: &Value) -> Read<ReinsuranceHeld> {
    let o = field(v, "options")?;
    let view = field(v, "view")?;
    let recognised = match field(v, "recognised")? {
        Value::Null => None,
        r => Some((
            ceded_balances_from_value(field(r, "balances")?)?,
            number(r, "past_events_cost")?,
        )),
    };
    let shares = floats(v, "shares")?;
    let recognised_losses = floats(v, "recognised_losses")?;
    if shares.len() != recognised_losses.len() {
        return Err("\"shares\" and \"recognised_losses\" differ in length".into());
    }
    let ceded = CededCashFlows {
        premiums: floats(view, "premiums")?,
        recoveries: floats(view, "recoveries")?,
        premium_timing: timing(view, "premium_timing")?,
        recovery_timing: timing(view, "recovery_timing")?,
    };
    if ceded.premiums.len() != ceded.recoveries.len() {
        return Err("the view's premiums and recoveries differ in length".into());
    }
    Ok(ReinsuranceHeld {
        curve: curve_from_value(field(v, "curve")?)?,
        options: ReinsuranceOptions {
            coverage_units: optional_floats(o, "coverage_units")?,
            period_years: number(o, "period_years")?,
            retroactive: flag(o, "retroactive")?,
            non_performance: non_performance_from_value(o, "non_performance")?,
        },
        period: period(v)?,
        view: ceded,
        non_performance: non_performance_from_value(v, "non_performance")?,
        balances: ceded_balances_from_value(field(v, "balances")?)?,
        coverage_units: floats(v, "coverage_units")?,
        shares,
        recognised_losses,
        recognised,
    })
}

macro_rules! state_json {
    ($ty:ty, $format:literal, $what:literal, $value:ident, $from:ident) => {
        impl $ty {
            /// The state as a JSON document (format
            #[doc = concat!("`", $format, "`),")]
            /// from which [`from_json`](Self::from_json) rebuilds it to roll
            /// forward again.
            pub fn to_json(&self) -> String {
                finish($value(self), $format)
            }

            /// Reads a document written by [`to_json`](Self::to_json). Fails on
            /// malformed JSON, another format, a newer format version, or
            /// parts that do not fit together.
            pub fn from_json(text: &str) -> Result<Self> {
                document(text, $format)
                    .and_then(|doc| $from(&doc))
                    .map_err(|msg| data_error($what, msg))
            }
        }
    };
}

state_json!(
    Group,
    "risk_rs.ifrs17_group",
    "IFRS 17 group",
    group_value,
    group_from_value
);
state_json!(
    PaaGroup,
    "risk_rs.ifrs17_paa_group",
    "IFRS 17 PAA group",
    paa_value,
    paa_from_value
);
state_json!(
    ReinsuranceHeld,
    "risk_rs.ifrs17_reinsurance_held",
    "IFRS 17 reinsurance held",
    reinsurance_value,
    reinsurance_from_value
);

#[cfg(test)]
mod tests {
    use super::super::PeriodInputs;
    use super::super::paa::PaaInputs;
    use super::super::reinsurance::ReinsuranceInputs;
    use super::*;

    #[test]
    fn a_group_saved_after_a_period_rolls_forward_as_the_original() {
        let cash = ExpectedCashFlows::new(3)
            .with_premiums(vec![900.0, 0.0, 0.0])
            .with_claims(vec![200.0; 3]);
        let options = GroupOptions {
            accretion_curve: Some(DiscountCurve::flat(0.04).unwrap()),
            ..Default::default()
        };
        let mut g =
            Group::recognise(cash, 120.0, DiscountCurve::flat(0.05).unwrap(), options).unwrap();
        let restored = Group::from_json(&g.to_json()).unwrap();
        assert_eq!(restored, g);
        g.roll_forward(&PeriodInputs::new(
            g.as_expected(),
            ExpectedCashFlows::new(2).with_claims(vec![200.0; 2]),
            80.0,
        ))
        .unwrap();
        let mut h = Group::from_json(&g.to_json()).unwrap();
        assert_eq!(h, g);
        let next = |g: &mut Group| {
            g.roll_forward(&PeriodInputs::new(
                g.as_expected(),
                ExpectedCashFlows::new(1).with_claims(vec![210.0]),
                40.0,
            ))
            .unwrap()
        };
        assert_eq!(next(&mut h), next(&mut g));
        assert!(Group::from_json(&g.to_json().replace("ifrs17_group", "other")).is_err());
        assert!(Group::from_json("{").is_err());
    }

    #[test]
    fn a_paa_group_and_a_reinsurance_group_round_trip() {
        let mut paa = PaaGroup::recognise(1_000.0, 100.0, PaaOptions::evenly(2)).unwrap();
        paa.roll_forward(&PaaInputs::new(
            1_000.0,
            100.0,
            300.0,
            200.0,
            Lic::new(100.0, 10.0),
        ))
        .unwrap();
        assert_eq!(PaaGroup::from_json(&paa.to_json()).unwrap(), paa);

        let options = ReinsuranceOptions {
            non_performance: Some(NonPerformance::new(0.02, 0.5).unwrap()),
            ..Default::default()
        };
        let mut held = ReinsuranceHeld::recognise(
            CededCashFlows::new(vec![300.0, 0.0, 0.0], vec![100.0; 3]),
            15.0,
            DiscountCurve::flat(0.03).unwrap(),
            options,
            &[],
        )
        .unwrap();
        assert_eq!(ReinsuranceHeld::from_json(&held.to_json()).unwrap(), held);
        held.roll_forward(
            &ReinsuranceInputs::new(
                held.as_expected(),
                CededCashFlows::new(vec![0.0; 2], vec![100.0; 2]),
                10.0,
            ),
            &[],
        )
        .unwrap();
        assert_eq!(ReinsuranceHeld::from_json(&held.to_json()).unwrap(), held);
    }
}
