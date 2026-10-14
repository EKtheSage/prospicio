//! Aggregate lane: the P&L ledger (`prospicio_aggregate::pnl`) for R's
//! `pnl.R`. R evaluates each leg on the source's values and hands the legs
//! over flat: a group index, a side, a label, a kind and `n` values each.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::pnl::{
    Book, Expense, ExpenseBasis, Face, Group, LADDER, Layout, Ledger, Leg, LegKind, RatioRow, Role,
    RowKind, Source, derive_premium, snap,
};

use crate::discrete::DiscreteDist;
use crate::reinsurance::ReinsuranceTower;
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
    /// from 1; an empty `margin_labels` entry means the default; a NaN
    /// `leg_step` leaves the leg on its own values.
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
        leg_step: &[f64],
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
            if !leg_step[i].is_nan() {
                leg = leg.step(leg_step[i]);
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

    /// A book's ledger: expenses flattened to one term per entry, with
    /// `exp_index` (from 1) grouping the terms into legs and an empty
    /// `exp_label` leaving a leg unlabelled; `tower` is NULL or a
    /// `ReinsuranceTower`; an empty label means the default.
    #[allow(clippy::too_many_arguments)]
    fn book(
        values: &[f64],
        probs: &[f64],
        name: &str,
        premium: f64,
        exp_index: &[f64],
        exp_basis: Vec<String>,
        exp_value: &[f64],
        exp_label: Vec<String>,
        premium_label: &str,
        loss_label: &str,
        step_label: &str,
        tower: Robj,
        face: &str,
    ) -> Result<Self> {
        let source = Source::new(values.to_vec(), probs.to_vec()).map_err(to_r)?;
        let mut book = Book::new(name, premium);
        for e in expenses(exp_index, &exp_basis, exp_value, &exp_label)? {
            book = book.expense(e);
        }
        if !premium_label.is_empty() {
            book = book.premium_label(premium_label);
        }
        if !loss_label.is_empty() {
            book = book.loss_label(loss_label);
        }
        if !step_label.is_empty() {
            book = book.step_label(step_label);
        }
        let face = match face {
            "consolidated" => Face::Consolidated,
            "walk" => Face::Walk,
            "peel-top-down" => Face::PeelTopDown,
            "peel-bottom-up" => Face::PeelBottomUp,
            other => {
                return Err(Error::Other(format!(
                    "a face is consolidated, walk, peel-top-down or peel-bottom-up, not {other:?}"
                )));
            }
        };
        let tower = if tower.is_null() {
            None
        } else {
            Some(
                <&ReinsuranceTower>::try_from(&tower)
                    .map_err(|_| Error::Other("tower must be a reinsurance_tower".into()))?,
            )
        };
        Ok(Self {
            inner: book
                .ledger(&source, tower.map(|t| &t.inner), face)
                .map_err(to_r)?,
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

    fn validation(&self) -> List {
        let v = self.inner.validation();
        list!(
            label = v.iter().map(|r| r.label.clone()).collect::<Vec<_>>(),
            EX = v.iter().map(|r| r.ex).collect::<Vec<_>>(),
            Est = v.iter().map(|r| r.est).collect::<Vec<_>>(),
            abs_err = v.iter().map(|r| r.abs_err).collect::<Vec<_>>(),
            rel_err = v.iter().map(|r| r.rel_err).collect::<Vec<_>>()
        )
    }

    fn ratios(&self) -> List {
        let r = self.inner.ratios();
        let col = |f: fn(&RatioRow) -> Option<f64>| r.iter().map(|r| na(f(r))).collect::<Vec<_>>();
        list!(
            step = r.iter().map(|r| r.step.clone()).collect::<Vec<_>>(),
            P = col(|r| Some(r.p)),
            L = col(|r| Some(r.l)),
            E = col(|r| Some(r.e)),
            M = col(|r| Some(r.m)),
            SD = col(|r| r.sd),
            LR = col(|r| r.lr),
            ER = col(|r| r.er),
            CR = col(|r| r.cr),
            E_LR = col(|r| r.e_lr),
            E_ER = col(|r| r.e_er),
            E_CR = col(|r| r.e_cr),
            P_share = col(|r| r.p_share),
            M_share = col(|r| r.m_share)
        )
    }

    /// `c(EX, SD, CV, Skew, P01, …, P99)` of the result, NA where undefined.
    fn result_stats(&self) -> Result<Vec<f64>> {
        let s = self.inner.result_stats().map_err(to_r)?;
        let mut out = vec![s.ex, s.sd, na(s.cv), na(s.skew)];
        out.extend(s.ladder);
        Ok(out)
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

/// The gross premium that leaves `technical` after the expenses (flattened
/// as in `PnlLedger::book`).
#[extendr]
fn pnl_derive_premium(
    technical: f64,
    exp_index: &[f64],
    exp_basis: Vec<String>,
    exp_value: &[f64],
    exp_label: Vec<String>,
) -> Result<f64> {
    let e = expenses(exp_index, &exp_basis, exp_value, &exp_label)?;
    derive_premium(technical, &e).map_err(to_r)
}

/// Expense legs from flattened terms.
fn expenses(
    index: &[f64],
    basis: &[String],
    value: &[f64],
    label: &[String],
) -> Result<Vec<Expense>> {
    let mut out: Vec<Expense> = Vec::new();
    for (k, &i) in index.iter().enumerate() {
        let b = match basis[k].as_str() {
            "fixed" => ExpenseBasis::Fixed,
            "premium" => ExpenseBasis::Premium,
            "loss" => ExpenseBasis::Loss,
            other => {
                return Err(Error::Other(format!(
                    "an expense basis is fixed, premium or loss, not {other:?}"
                )));
            }
        };
        let i = i as usize;
        if i == out.len() + 1 {
            let mut e = Expense::new(b, value[k]);
            if !label[k].is_empty() {
                e = e.label(label[k].clone());
            }
            out.push(e);
        } else if i == out.len() && i > 0 {
            let last = out.pop().unwrap();
            out.push(last.and(b, value[k]));
        } else {
            return Err(Error::Other(
                "expense indices must run 1, 2, … in order".into(),
            ));
        }
    }
    Ok(out)
}

extendr_module! {
    mod pnl;
    impl PnlLedger;
    fn pnl_derive_premium;
}
