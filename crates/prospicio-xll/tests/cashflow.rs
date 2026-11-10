//! A collateralised deal's cash flows on a sheet (`PRO.CAP.CASHFLOW`,
//! `docs/design/xll.md` decisions 42 and 46), reached through the
//! catalogue as Excel reaches it:
//!
//! * Excel's own worked examples of `XNPV`, `XIRR` and `IRR` (Microsoft's
//!   function reference) through `prospicio_pricing::returns`, the
//!   discounting and root search `PRO.CAP.CASHFLOW` reports with;
//! * a deal through the terms block equal to the same deal built in
//!   `prospicio_pricing::cashflow`, to the bit, with every rule set: its
//!   flows, totals, IRR, MOIC and trace; and the crate's documented
//!   three-month deal;
//! * the errors.

use crate::common;

use common::{n, num, t, text};
use prospicio_pricing::cashflow::{
    Adjustment, AdjustmentBase, Base, CashflowTerms, Dates, FeeBasis, FundingRule, FundingSource,
    InterestBasis, MonthlyPatterns, ObligationReference, PerformanceFee, Requirement, Term, Timing,
};
use prospicio_pricing::returns::{self, IrrOptions, TieredFee, moic};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

/// A block of rows, the short ones padded with blanks as a sheet range is.
fn rows(mut cells: Vec<Vec<Value>>) -> Grid {
    let width = cells.iter().map(Vec::len).max().unwrap_or(0);
    for row in &mut cells {
        row.resize(width, Value::Empty);
    }
    Grid::from_rows(cells)
}

#[test]
fn excel_s_worked_examples() {
    // Microsoft's XNPV and XIRR example: five flows on 1 January 2008, 1
    // March 2008, 30 October 2008, 15 February 2009 and 1 April 2009,
    // discounted by days from the first over 365.
    let flows = [-10000.0, 2750.0, 4250.0, 3250.0, 2750.0];
    let dates = [39448.0, 39508.0, 39751.0, 39859.0, 39904.0];
    let years: Vec<f64> = dates.iter().map(|d| (d - dates[0]) / 365.0).collect();
    // XNPV(0.09, ...) = 2086.65 and XIRR(...) = 0.373362535 in the
    // reference; to more digits, the root of the same sum.
    let v = returns::npv(&flows, &years, 0.09).unwrap();
    assert!((v - 2086.6476020315363).abs() < 1e-9, "{v}");
    assert_eq!((v * 100.0).round() / 100.0, 2086.65);
    let r = returns::irr(&flows, &years, &IrrOptions::default())
        .unwrap()
        .rate;
    assert!((r - 0.37336253351883164).abs() < 1e-9, "{r}");

    // IRR on yearly flows: 8.7% over five years, -2.1% over four.
    let flows = [-70000.0, 12000.0, 15000.0, 18000.0, 21000.0, 26000.0];
    let yearly: Vec<f64> = (0..flows.len()).map(|t| t as f64).collect();
    let five = returns::irr(&flows, &yearly, &IrrOptions::default())
        .unwrap()
        .rate;
    assert!((five - 0.0866309480365316).abs() < 1e-9, "{five}");
    let four = returns::irr(&flows[..5], &yearly[..5], &IrrOptions::default())
        .unwrap()
        .rate;
    assert!((four + 0.021244848273411054).abs() < 1e-9, "{four}");
}

fn label(name: &str, values: &[Value]) -> Vec<Value> {
    let mut row = vec![Value::from(name)];
    row.extend(values.iter().cloned());
    row
}

fn nv(x: f64) -> Value {
    Value::Number(x)
}

/// Monthly cumulative shares over `len` months: written at once, earned
/// evenly over 12 months, paid evenly over `paid_months`.
fn ramps(len: usize, paid_months: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ramp = |m: usize| -> Vec<f64> {
        (0..len)
            .map(|t| ((t + 1) as f64 / m as f64).min(1.0))
            .collect()
    };
    (vec![1.0; len], ramp(12), ramp(paid_months))
}

fn pattern_grid(w: &[f64], e: &[f64], p: &[f64]) -> Grid {
    let mut cells = vec![vec![
        Value::from("written"),
        Value::from("earned"),
        Value::from("paid"),
    ]];
    for k in 0..w.len() {
        cells.push(vec![nv(w[k]), nv(e[k]), nv(p[k])]);
    }
    rows(cells)
}

#[test]
fn the_crate_s_three_month_deal() {
    let patterns = pattern_grid(
        &[1.0; 3],
        &[1.0 / 3.0, 2.0 / 3.0, 1.0],
        &[1.0 / 3.0, 2.0 / 3.0, 1.0],
    );
    let terms = rows(vec![
        label("premium", &[nv(100.0)]),
        label("Commutation month", &[nv(3.0)]),
    ]);
    let requirement = rows(vec![label("reserve", &[nv(1.0)])]);
    let commutation = rows(vec![label("outstanding", &[nv(1.0)])]);
    let args = |u: &[f64], output: &str| {
        vec![
            Grid::column(u),
            terms.clone(),
            patterns.clone(),
            requirement.clone(),
            none(),
            t(output),
            none(),
            commutation.clone(),
        ]
    };
    let flows = call("pro_cap_cashflow", args(&[0.6, 1.5], "flows"));
    assert_eq!(flows.get(0, 0), &Value::from("Month 0"));
    let round = |v: &Value| (num(v) * 1e9).round() / 1e9;
    let got: Vec<f64> = flows.cells[3..].iter().map(round).collect();
    assert_eq!(got, [80.0, -20.0, -20.0, 50.0, -50.0, -50.0]);
    let sims = call("pro_cap_cashflow", args(&[1.5], "simulations"));
    assert_eq!(sims.get(0, 3), &Value::from("Funded"));
    assert_eq!(round(sims.get(1, 3)), 100.0);
    assert_eq!(sims.get(1, 9), &Value::Bool(true), "paid before paying in");
}

/// A deal with every rule set, as a terms block and as the crate's terms.
fn full_deal() -> (Grid, Grid, Grid, Grid, CashflowTerms) {
    let len = 40;
    let (w, e, p) = ramps(len, 30);
    let terms_grid = rows(vec![
        label("premium", &[nv(1000.0)]),
        label("commutation_month", &[nv(36.0)]),
        label("collected_rate", &[nv(0.9)]),
        label("deduct_commission", &[Value::Bool(true)]),
        label("lag", &[nv(1.0)]),
        label("adjustment", &[nv(18.0), nv(0.3), Value::Bool(true)]),
        label(
            "adjustment",
            &[
                nv(24.0),
                nv(0.1),
                Value::Bool(false),
                Value::from("to-date"),
            ],
        ),
        label("funding_dates", &[nv(3.0), nv(1.0)]),
        label("funding_rule", &[Value::from("obligation increase")]),
        label("premium_offset", &[nv(0.5)]),
        label("reference", &[Value::from("running-max")]),
        label("funding_source", &[Value::from("operating-first")]),
        label("collateral_release", &[Value::from("quarterly")]),
        label("annual_yield", &[nv(0.04)]),
        label("interest_basis", &[Value::from("opening")]),
        label("opening_fixed", &[nv(50.0)]),
        label("opening_share", &[nv(0.1)]),
        label("expense_rate", &[nv(0.01)]),
        label("buffer_months", &[nv(2.0)]),
        label("buffer_paid", &[Value::Bool(true)]),
        label("buffer_expenses", &[Value::Bool(true)]),
        label(
            "equity_release",
            &[Value::from("months"), nv(12.0), nv(24.0)],
        ),
        label("fee_breakpoints", &[nv(0.0), nv(100.0)]),
        label("fee_slopes", &[nv(0.1), nv(0.2)]),
        label("fee_basis", &[Value::from("profit")]),
    ]);
    let requirement_grid = rows(vec![
        label("outstanding", &[nv(1.0)]),
        label("unearned-premium", &[nv(0.3)]),
        label("floor", &[nv(20.0)]),
        label("limit", &[nv(1.2), nv(1.5)]),
    ]);
    let commutation_grid = rows(vec![label("outstanding", &[nv(1.1)])]);
    let patterns = MonthlyPatterns::new(w.clone(), e.clone(), p.clone()).unwrap();
    let requirement = Requirement::new(vec![
        Term::new(Base::Outstanding, 1.0),
        Term::new(Base::UnearnedPremium, 0.3),
    ])
    .with_floor(20.0)
    .with_limit(1.2, 1.5);
    let mut terms = CashflowTerms::new(1000.0, 36, patterns, requirement).unwrap();
    terms.commutation_requirement = Requirement::new(vec![Term::new(Base::Outstanding, 1.1)]);
    terms.premium_flow.collected_rate = 0.9;
    terms.premium_flow.deduct_commission = true;
    terms.premium_flow.lag = 1;
    terms.premium_flow.adjustments = vec![
        Adjustment {
            month: 18,
            written_rate: 0.3,
            less_commission: true,
            base: AdjustmentBase::Ultimate,
        },
        Adjustment {
            month: 24,
            written_rate: 0.1,
            less_commission: false,
            base: AdjustmentBase::ToDate,
        },
    ];
    terms.funding.dates = Dates::Every {
        months: 3,
        first: 1,
    };
    terms.funding.rule = FundingRule::ObligationIncrease {
        premium_offset: 0.5,
        reference: ObligationReference::RunningMax,
    };
    terms.funding.source = FundingSource::OperatingFirst;
    terms.collateral_release = Timing::At(Dates::quarterly());
    terms.interest.annual_yield = 0.04;
    terms.interest.basis = InterestBasis::Opening;
    terms.operating.opening_fixed = 50.0;
    terms.operating.opening_share = 0.1;
    terms.operating.expense_rate = 0.01;
    terms.operating.buffer.months = 2;
    terms.operating.buffer.paid = true;
    terms.operating.buffer.expenses = true;
    terms.operating.release = Timing::At(Dates::Months(vec![12, 24]));
    terms.operating.fee = Some(PerformanceFee {
        tiers: TieredFee::new(vec![0.0, 100.0], vec![0.1, 0.2]).unwrap(),
        basis: FeeBasis::Profit,
    });
    (
        terms_grid,
        pattern_grid(&w, &e, &p),
        requirement_grid,
        commutation_grid,
        terms,
    )
}

#[test]
fn a_deal_with_every_rule_is_the_crate_s() {
    let (terms_grid, patterns, requirement, commutation, terms) = full_deal();
    let u = [0.4, 0.65, 0.9, 1.3, 2.0];
    let c = [0.25, 0.2, 0.15, 0.1, 0.1];
    let args = |output: &str, rate: Grid| {
        vec![
            Grid::column(&u),
            terms_grid.clone(),
            patterns.clone(),
            requirement.clone(),
            Grid::column(&c),
            t(output),
            rate,
            commutation.clone(),
        ]
    };
    let expected = terms.run(&u, &c).unwrap();
    let months = expected.months();
    assert_eq!(months, 36);

    let flows = call("pro_cap_cashflow", args("flows", none()));
    assert_eq!((flows.rows, flows.columns), (6, 36), "{flows:?}");
    for (got, want) in flows.cells[36..].iter().zip(expected.flows()) {
        assert_eq!(num(got), *want);
    }

    let sims = call("pro_cap_cashflow", args("simulations", n(0.08)));
    assert_eq!((sims.rows, sims.columns), (6, 11), "{sims:?}");
    assert_eq!(sims.get(0, 2), &Value::from("NPV at 8%"));
    let irr = expected.irr(&IrrOptions::default()).unwrap();
    let npv = expected.npv(0.08).unwrap();
    for j in 0..u.len() {
        let r = j + 1;
        assert_eq!(num(sims.get(r, 0)), u[j]);
        match sims.get(r, 1) {
            Value::Number(x) => assert_eq!(*x, irr[j].rate),
            v => assert!(v.is_blank() && irr[j].rate.is_nan()),
        }
        assert_eq!(num(sims.get(r, 2)), npv[j]);
        let row = &expected.flows()[j * months..(j + 1) * months];
        let m = moic(
            row.iter().sum(),
            expected.funded()[j],
            expected.opening()[j],
        )
        .unwrap();
        assert_eq!(num(sims.get(r, 3)), m);
        let totals = [
            expected.funded()[j],
            expected.opening()[j],
            expected.released()[j],
            expected.distributed()[j],
            expected.fee()[j],
            expected.commutation()[j],
        ];
        for (k, x) in totals.iter().enumerate() {
            assert_eq!(num(sims.get(r, 4 + k)), *x, "row {r} column {}", 4 + k);
        }
        assert_eq!(sims.get(r, 10), &Value::Bool(expected.paid_first()[j]));
    }
    // Something happened: a fee, capital called, an IRR found.
    assert!(expected.fee().iter().any(|f| *f > 0.0));
    assert!(expected.funded().iter().any(|f| *f > 0.0));
    assert!(irr.iter().any(|r| r.found));

    let trace = call("pro_cap_cashflow", args("trace", none()));
    let want = terms.trace(u[0], c[0]).unwrap();
    assert_eq!(trace.get(0, 13), &Value::from("Investor"));
    assert_eq!(trace.rows, 1 + 36 + 1 + 3);
    for m in 0..36 {
        assert_eq!(num(trace.get(1 + m, 0)), m as f64);
        assert_eq!(num(trace.get(1 + m, 1)), want.requirement[m]);
        assert_eq!(num(trace.get(1 + m, 13)), want.investor[m]);
    }
    assert_eq!(trace.get(39, 0), &Value::from("Fee"));
    assert_eq!(num(trace.get(39, 1)), want.fee);
    assert!(num(trace.get(40, 1)).abs() < 1e-9, "money is conserved");
}

#[test]
fn errors() {
    let (terms_grid, patterns, requirement, _, _) = full_deal();
    let run = |terms: Grid, patterns: Grid, requirement: Grid| {
        text(&call(
            "pro_cap_cashflow",
            vec![n(0.6), terms, patterns, requirement],
        ))
        .to_owned()
    };
    let with = |extra: Vec<Value>| {
        let mut g = terms_grid.clone();
        let mut cells: Vec<Vec<Value>> = (0..g.rows)
            .map(|r| (0..g.columns).map(|c| g.get(r, c).clone()).collect())
            .collect();
        let mut extra = extra;
        extra.resize(g.columns, Value::Empty);
        cells.push(extra);
        g = rows(cells);
        g
    };
    assert!(
        run(
            with(vec![Value::from("premum"), nv(1.0)]),
            patterns.clone(),
            requirement.clone()
        )
        .contains("unknown label \"premum\"")
    );
    assert!(
        run(
            with(vec![Value::from("lag"), nv(2.0)]),
            patterns.clone(),
            requirement.clone()
        )
        .contains("two lag rows")
    );
    assert!(
        run(
            with(vec![
                Value::from("collateral_release"),
                Value::from("weekly")
            ]),
            patterns.clone(),
            requirement.clone()
        )
        .contains("two collateral_release rows")
    );
    let only = rows(vec![label("commutation_month", &[nv(36.0)])]);
    assert!(run(only, patterns.clone(), requirement.clone()).contains("needs a premium row"));
    assert!(
        run(
            terms_grid.clone(),
            patterns.clone(),
            rows(vec![label("reserves", &[nv(1.0)])])
        )
        .contains("unknown base \"reserves\"")
    );
    assert!(
        run(
            terms_grid.clone(),
            Grid::column(&[1.0, 1.0]),
            requirement.clone()
        )
        .contains("three columns")
    );
    let short = pattern_grid(&[1.0; 10], &[1.0; 10], &[1.0; 10]);
    assert!(!run(terms_grid.clone(), short, requirement.clone()).is_empty());
    let mut bad_rule = terms_grid.clone();
    let r = (0..bad_rule.rows)
        .find(|&r| bad_rule.get(r, 0) == &Value::from("funding_rule"))
        .unwrap();
    bad_rule.cells[r * bad_rule.columns + 1] = Value::from("whenever");
    assert!(
        run(bad_rule, patterns.clone(), requirement.clone())
            .contains("shortfall or obligation-increase")
    );
    assert!(
        text(&call(
            "pro_cap_cashflow",
            vec![
                Grid::column(&[0.5, 0.6]),
                terms_grid,
                patterns,
                requirement,
                Grid::column(&[0.1, 0.1, 0.1])
            ]
        ))
        .contains("one rate or one per loss ratio")
    );
}
