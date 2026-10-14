//! Aggregate lane: the P&L ledger (`prospicio_aggregate::pnl`) for R's
//! `pnl.R`. R evaluates each leg on the source's values and hands the legs
//! over flat: a group index, a side, a label, a kind and `n` values each.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::pnl::{
    Group, LADDER, Layout, Ledger, Leg, LegKind, Role, RowKind, Source, snap,
};

use crate::discrete::DiscreteDist;
use crate::to_r;

/// A P&L ledger.
#[extendr]
pub(crate) struct PnlLedger {
    inner: Ledger,
}

fn kind_of(name: &str) -> Result<Option<LegKind>> {
    Ok(match name {
        "" => None,
        "premium" => Some(LegKind::Premium),
        "loss" => Some(LegKind::Loss),
        "expense" => Some(LegKind::Expense),
        "recovery" => Some(LegKind::Recovery),
        "commission" => Some(LegKind::Commission),
        other => {
            return Err(Error::Other(format!(
                "a leg's kind is premium, loss, expense, recovery or commission, not {other:?}"
            )));
        }
    })
}

fn na(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

#[extendr]
impl PnlLedger {
    /// `leg_values` holds each leg's `length(values)` values in turn;
    /// `leg_side` is "consideration" or "obligation"; `leg_group` counts
    /// from 1; an empty `margin_labels` entry means the default.
    #[allow(clippy::too_many_arguments)]
    fn new(
        values: &[f64],
        probs: &[f64],
        group_labels: Vec<String>,
        group_roles: Vec<String>,
        margin_labels: Vec<String>,
        leg_group: &[f64],
        leg_side: Vec<String>,
        leg_label: Vec<String>,
        leg_kind: Vec<String>,
        leg_values: &[f64],
        name: &str,
        result_name: &str,
        span_label: Vec<String>,
        span_lo: &[f64],
        span_hi: &[f64],
        force_tower: bool,
    ) -> Result<Self> {
        let source = Source::new(values.to_vec(), probs.to_vec()).map_err(to_r)?;
        let n = values.len();
        let mut groups: Vec<Group> = Vec::with_capacity(group_labels.len());
        for (k, label) in group_labels.iter().enumerate() {
            let role = match group_roles[k].as_str() {
                "sell" => Role::Sell,
                "buy" => Role::Buy,
                other => {
                    return Err(Error::Other(format!(
                        "a group's role is sell or buy, not {other:?}"
                    )));
                }
            };
            let mut g = Group::new(label.clone(), role);
            if !margin_labels[k].is_empty() {
                g = g.margin_label(margin_labels[k].clone());
            }
            groups.push(g);
        }
        if leg_values.len() != leg_label.len() * n {
            return Err(Error::Other(
                "each leg needs one value per source value".into(),
            ));
        }
        for (i, label) in leg_label.iter().enumerate() {
            let gi = leg_group[i] as usize;
            if gi < 1 || gi > groups.len() {
                return Err(Error::Other(format!("leg {label:?} names no group")));
            }
            let mut leg = Leg::values(label.clone(), leg_values[i * n..(i + 1) * n].to_vec());
            if let Some(k) = kind_of(&leg_kind[i])? {
                leg = leg.kind(k);
            }
            let g = groups.remove(gi - 1);
            let g = if leg_side[i] == "consideration" {
                g.consideration(leg)
            } else {
                g.obligation(leg)
            };
            groups.insert(gi - 1, g);
        }
        let mut layout = Layout::new(name).result_name(result_name);
        for (k, label) in span_label.iter().enumerate() {
            layout = layout.span(label.clone(), span_lo[k] as usize, span_hi[k] as usize);
        }
        if force_tower {
            layout = layout.force_tower();
        }
        Ok(Self {
            inner: Ledger::new(&source, groups, layout).map_err(to_r)?,
        })
    }

    fn rows(&self) -> Vec<String> {
        self.inner.rows().iter().map(|r| r.label.clone()).collect()
    }

    fn result(&self) -> DiscreteDist {
        DiscreteDist {
            inner: self.inner.result().distribution().clone(),
        }
    }

    fn row(&self, label: &str) -> Result<DiscreteDist> {
        self.inner
            .row(label)
            .map(|r| DiscreteDist {
                inner: r.distribution().clone(),
            })
            .ok_or_else(|| Error::Other(format!("no row labelled {label:?}")))
    }

    fn prob_eq_0(&self) -> f64 {
        self.inner.prob_eq_0()
    }

    fn is_tower(&self) -> bool {
        self.inner.is_tower()
    }

    /// `list(step, side, label, row, EX, SD, CV, Skew, k01, …, k99)`.
    fn sheet(&self) -> Result<List> {
        let rows = self.inner.rows();
        let mut names = vec!["step", "side", "label", "row", "EX", "SD", "CV", "Skew"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let mut cols: Vec<Robj> = vec![
            rows.iter()
                .map(|r| r.step.clone())
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| r.side.name().to_string())
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| r.display.clone())
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| r.label.clone())
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| snap(r.mean()))
                .collect::<Vec<_>>()
                .into(),
            rows.iter().map(|r| snap(r.sd())).collect::<Vec<_>>().into(),
            rows.iter()
                .map(|r| na(r.cv().map(snap)))
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| na(r.skew().map(snap)))
                .collect::<Vec<_>>()
                .into(),
        ];
        for q in LADDER {
            names.push(format!("k{:02.0}", q * 100.0));
            cols.push(self.inner.scenarios(q).map_err(to_r)?.into());
        }
        List::from_names_and_values(names, cols)
    }

    /// `list(step, side, label, row, P01, …, P99)`.
    fn marginal(&self) -> Result<List> {
        let rows = self.inner.rows();
        let mut names = vec!["step", "side", "label", "row"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let mut cols: Vec<Robj> = vec![
            rows.iter()
                .map(|r| r.step.clone())
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| r.side.name().to_string())
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| r.display.clone())
                .collect::<Vec<_>>()
                .into(),
            rows.iter()
                .map(|r| r.label.clone())
                .collect::<Vec<_>>()
                .into(),
        ];
        for q in LADDER {
            names.push(format!("P{:02.0}", q * 100.0));
            let col: Result<Vec<f64>> = rows
                .iter()
                .map(|r| r.quantile(q).map(snap).map_err(to_r))
                .collect();
            cols.push(col?.into());
        }
        List::from_names_and_values(names, cols)
    }

    fn card(&self) -> Result<List> {
        let card = self.inner.card().map_err(to_r)?;
        Ok(list!(
            step = card.iter().map(|r| r.step.clone()).collect::<Vec<_>>(),
            side = card.iter().map(|r| r.side.clone()).collect::<Vec<_>>(),
            EX = card.iter().map(|r| r.ex).collect::<Vec<_>>(),
            SD = card.iter().map(|r| r.sd).collect::<Vec<_>>(),
            CV = card.iter().map(|r| na(r.cv)).collect::<Vec<_>>(),
            Skew = card.iter().map(|r| na(r.skew)).collect::<Vec<_>>(),
            P01 = card.iter().map(|r| r.p01).collect::<Vec<_>>(),
            Median = card.iter().map(|r| r.median).collect::<Vec<_>>(),
            P99 = card.iter().map(|r| r.p99).collect::<Vec<_>>()
        ))
    }

    fn legs(&self) -> List {
        let legs: Vec<_> = self
            .inner
            .rows()
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Leg { .. }))
            .collect();
        list!(
            step = legs.iter().map(|r| r.step.clone()).collect::<Vec<_>>(),
            side = legs
                .iter()
                .map(|r| r.side.name().to_string())
                .collect::<Vec<_>>(),
            label = legs.iter().map(|r| r.label.clone()).collect::<Vec<_>>(),
            kind = legs
                .iter()
                .map(|r| r.leg_kind.map_or(String::new(), |k| k.name().to_string()))
                .collect::<Vec<_>>(),
            EX = legs.iter().map(|r| r.mean()).collect::<Vec<_>>(),
            SD = legs.iter().map(|r| snap(r.sd())).collect::<Vec<_>>()
        )
    }
}

extendr_module! {
    mod pnl;
    impl PnlLedger;
}
