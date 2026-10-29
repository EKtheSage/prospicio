//! A per-risk excess of loss with an event limit, inuring to a cat cover
//! (#585).
//!
//! A catastrophe programme often sits above a per-risk excess of loss: the
//! per-risk layer applies to each risk's loss within an event, an event
//! limit caps the sum of its recoveries from one event, and the cat layer
//! sees each event net of those recoveries. An [`EventSet`] holds one loss
//! per event, so the risks of an event are given here as years of events
//! of risk losses, and [`per_risk_inuring`] returns the per-risk layer's
//! cession and the event set net of it, one loss per event, for any tower
//! of cat layers ([`Tower::apply`](crate::Tower::apply)) to take as its
//! gross.

use prospicio_core::{Error, Result};

use crate::monte_carlo::EventSet;
use crate::reinsurance::{Basis, Layer};

/// The per-risk layer's cession and what it leaves for the cat stage.
#[derive(Debug, Clone, PartialEq)]
pub struct PerRiskInuring {
    /// Each year's ceded loss to the per-risk layer, at its share.
    pub ceded: Vec<f64>,
    /// Each year's ceded loss by event, at its share, in the events' order.
    pub ceded_by_event: Vec<Vec<f64>>,
    /// Each event's loss net of the per-risk cession: the cat stage's
    /// gross, with the events' times when they were given.
    pub net: EventSet,
}

/// Applies the per-risk excess of loss `layer` to years of events of risk
/// losses (`years[i][e]` the losses of the risks hit by event `e` of year
/// `i`, events in order), with `event_limit` capping the sum of one
/// event's per-risk recoveries (at 100%, before the layer's annual terms),
/// and returns its cession and the net events.
///
/// Each risk's recovery is the layer's occurrence terms on its loss; the
/// event's recoveries are summed and capped at the event limit; the
/// layer's annual terms (deductible, corridor, annual limit, at its share)
/// then use up the events' capped recoveries in order, as they use up a
/// year's occurrences ([`Layer::ceded_by_event`]). The net event is the
/// event's gross less its cession. `times`, when given, are each event's
/// time in the year and pass to the net event set.
///
/// The layer must be on the loss basis without reinstatements pro rata as
/// to time, an nth-event or top-and-drop cover or a scope, which read the
/// order or labels of whole events; an event limit must be at least 0.
///
/// ```
/// use prospicio_aggregate::per_risk::per_risk_inuring;
/// use prospicio_aggregate::{Layer, Tower};
///
/// // One event hits three risks of 12, 8 and 15; a 5 xs 5 per-risk layer
/// // recovers 5, 3 and 5, capped at 10 for the event.
/// let per_risk = Layer::xol("PR", 5.0, 5.0).unwrap();
/// let r = per_risk_inuring(&per_risk, Some(10.0), &[vec![vec![12.0, 8.0, 15.0]]], None, 0)
///     .unwrap();
/// assert_eq!(r.ceded, [10.0]);
/// assert_eq!(r.net.events(0), [25.0]);
/// // The cat layer above sees the event net of the per-risk cession.
/// let cat = Tower::new(vec![Layer::xol("Cat", 20.0, 10.0).unwrap()]).unwrap();
/// assert_eq!(cat.ceded(r.net.events(0)), [15.0]);
/// ```
pub fn per_risk_inuring(
    layer: &Layer,
    event_limit: Option<f64>,
    years: &[Vec<Vec<f64>>],
    times: Option<&[Vec<f64>]>,
    seed: u64,
) -> Result<PerRiskInuring> {
    if layer.basis != Basis::Loss {
        return Err(Error::Data(format!(
            "per-risk layer {:?} must be on the loss basis",
            layer.name
        )));
    }
    if layer.needs_times() || layer.needs_event_order() || layer.scope.is_some() {
        return Err(Error::Data(format!(
            "per-risk layer {:?} reads whole events (reinstatements pro rata as to time, an \
             nth-event or top-and-drop cover, or a scope); give a plain per-risk layer",
            layer.name
        )));
    }
    if let Some(l) = event_limit
        && (l.is_nan() || l < 0.0)
    {
        return Err(Error::InvalidParameter {
            name: "event_limit",
            value: l,
            reason: "must be at least 0",
        });
    }
    if let Some(t) = times
        && (t.len() != years.len() || t.iter().zip(years).any(|(t, y)| t.len() != y.len()))
    {
        return Err(Error::Data(
            "give one time for each event of each year".into(),
        ));
    }
    let cap = event_limit.unwrap_or(f64::INFINITY);
    let mut ceded = Vec::with_capacity(years.len());
    let mut by_event = Vec::with_capacity(years.len());
    let mut net = Vec::with_capacity(years.len());
    for year in years {
        let mut recovery = 0.0;
        let mut before = 0.0;
        let mut row = Vec::with_capacity(year.len());
        let mut net_row = Vec::with_capacity(year.len());
        for risks in year {
            if let Some(&x) = risks.iter().find(|x| !(x.is_finite() && **x >= 0.0)) {
                return Err(Error::InvalidParameter {
                    name: "losses",
                    value: x,
                    reason: "must be finite and non-negative",
                });
            }
            let event: f64 = risks.iter().map(|&x| layer.recovery(x)).sum();
            recovery += event.min(cap);
            let after = layer.share * layer.after_terms(recovery);
            let c = after - before;
            before = after;
            row.push(c);
            net_row.push(risks.iter().sum::<f64>() - c);
        }
        ceded.push(before);
        by_event.push(row);
        net.push(net_row);
    }
    let mut events = EventSet::from_years(net, seed)?;
    if let Some(t) = times {
        events = events.with_times(t.iter().flatten().copied().collect())?;
    }
    Ok(PerRiskInuring {
        ceded,
        ceded_by_event: by_event,
        net: events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tower;
    use prospicio_core::StreamRng;

    /// Years of events of risk losses, a few risks an event.
    fn years(n: usize, risks: usize, seed: u64) -> Vec<Vec<Vec<f64>>> {
        (0..n as u64)
            .map(|i| {
                let mut rng = StreamRng::new(seed, i);
                let events = (rng.next_open01() * 4.0) as usize;
                (0..events)
                    .map(|_| {
                        let k = 1 + (rng.next_open01() * risks as f64) as usize;
                        (0..k).map(|_| -20.0 * rng.next_open01().ln()).collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn one_risk_an_event_is_an_occurrence_layer_inuring() {
        let ys = years(2_000, 1, 3);
        let per_risk = Layer::xol("PR", 30.0, 10.0)
            .unwrap()
            .aggregate_limit(60.0)
            .unwrap();
        let cat = Layer::xol("Cat", 40.0, 25.0).unwrap();
        let r = per_risk_inuring(&per_risk, None, &ys, None, 0).unwrap();
        let tower = Tower::inuring(vec![vec![per_risk.clone()], vec![cat.clone()]]).unwrap();
        let cat_alone = Tower::new(vec![cat]).unwrap();
        for (i, y) in ys.iter().enumerate() {
            let flat: Vec<f64> = y.iter().map(|e| e[0]).collect();
            let both = tower.ceded(&flat);
            assert!((r.ceded[i] - both[0]).abs() < 1e-9);
            let c = cat_alone.ceded(r.net.events(i))[0];
            assert!((c - both[1]).abs() < 1e-9, "{i}: {c} {}", both[1]);
        }
    }

    #[test]
    fn an_event_limit_of_zero_leaves_the_gross() {
        let ys = years(500, 4, 5);
        let per_risk = Layer::xol("PR", 30.0, 10.0).unwrap();
        let r = per_risk_inuring(&per_risk, Some(0.0), &ys, None, 0).unwrap();
        for (i, y) in ys.iter().enumerate() {
            assert_eq!(r.ceded[i], 0.0);
            let gross: Vec<f64> = y.iter().map(|e| e.iter().sum()).collect();
            assert_eq!(r.net.events(i), gross.as_slice());
        }
        // No event limit cedes each risk's recovery.
        let open = per_risk_inuring(&per_risk, None, &ys, None, 0).unwrap();
        for (i, y) in ys.iter().enumerate() {
            let want: f64 = y.iter().flatten().map(|&x| per_risk.recovery(x)).sum();
            assert!((open.ceded[i] - want).abs() < 1e-9);
        }
    }

    #[test]
    fn the_event_limit_caps_each_event_before_the_annual_terms() {
        let per_risk = Layer::xol("PR", 5.0, 5.0)
            .unwrap()
            .aggregate_limit(16.0)
            .unwrap();
        // Recoveries 5 + 3 + 5 = 13, capped at 10; then 5 + 5 = 10; the
        // annual limit of 16 leaves 6 for the second event.
        let ys = vec![vec![vec![12.0, 8.0, 15.0], vec![20.0, 11.0]]];
        let t = vec![vec![0.2, 0.7]];
        let r = per_risk_inuring(&per_risk, Some(10.0), &ys, Some(&t), 0).unwrap();
        assert_eq!(r.ceded_by_event[0], [10.0, 6.0]);
        assert_eq!(r.net.events(0), [25.0, 25.0]);
        assert_eq!(r.net.times(0), Some(&[0.2, 0.7][..]));
        // Checks.
        assert!(per_risk_inuring(&per_risk, Some(-1.0), &ys, None, 0).is_err());
        assert!(per_risk_inuring(&per_risk, None, &[vec![vec![-1.0]]], None, 0).is_err());
        assert!(per_risk_inuring(&per_risk, None, &ys, Some(&[vec![0.1]]), 0).is_err());
        let surplus = Layer::surplus("S", 1.0, 2.0).unwrap();
        assert!(per_risk_inuring(&surplus, None, &ys, None, 0).is_err());
        let nth = Layer::xol("N", 5.0, 5.0)
            .unwrap()
            .nth_event(2, 5.0)
            .unwrap();
        assert!(per_risk_inuring(&nth, None, &ys, None, 0).is_err());
    }
}
