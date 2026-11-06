use super::*;

fn two_books() -> Ledger {
    let source = Source::new(vec![0.0, 10.0, 20.0, 50.0], vec![0.4, 0.3, 0.2, 0.1]).unwrap();
    let gross = Group::sell("Gross")
        .consideration(Leg::of("Premium", |x| (5.0 + 1.1 * x).clamp(12.0, 40.0)))
        .obligation(Leg::of("Loss", |x| x))
        .obligation(Leg::of("LAE", |x| 0.05 * x));
    let xl = Group::buy("XL")
        .consideration(Leg::constant("XL premium", 4.0))
        .obligation(Leg::of("XL recovery", |x| (x - 15.0).clamp(0.0, 25.0)))
        .obligation(Leg::constant("XL commission", 0.4));
    Ledger::new(
        &source,
        vec![gross, xl],
        Layout::new("Ex2").result_name("margin"),
    )
    .unwrap()
}

#[test]
fn rows_foot_and_scenarios_add_down() {
    let pnl = two_books();
    let labels: Vec<&str> = pnl.rows().iter().map(|r| r.label.as_str()).collect();
    assert_eq!(
        labels,
        vec![
            "Premium",
            "Loss",
            "LAE",
            "Gross total obligation",
            "Gross result",
            "XL premium",
            "XL recovery",
            "XL commission",
            "XL total obligation",
            "XL result",
            "net through XL",
            "total consideration",
            "total obligation",
            "margin",
            "total impact",
        ]
    );
    assert!((pnl.result().mean() - 6.3).abs() < 1e-12);
    assert!((pnl.row("Gross result").unwrap().mean() - 6.4).abs() < 1e-12);
    for q in LADDER {
        let k = pnl.scenarios(q).unwrap();
        // margin = total consideration + total obligation, in every state.
        assert!((k[13] - (k[11] + k[12])).abs() < 1e-12);
        assert_eq!(k[13], pnl.result().quantile(q).unwrap());
    }
    let card = pnl.card().unwrap();
    assert_eq!(card.last().unwrap().side, "Impact");
    assert!((card.last().unwrap().ex - (6.3 - 6.4)).abs() < 1e-12);
}

#[test]
fn a_level_set_and_a_constant() {
    let source = Source::new(vec![0.0, 10.0, 20.0, 50.0], vec![0.4, 0.3, 0.2, 0.1]).unwrap();
    let pnl = Ledger::single(
        &source,
        "Ex2c",
        Role::Sell,
        vec![Leg::of("Prem", |x| if x == 20.0 { 30.0 } else { 10.0 })],
        vec![Leg::of("Loss", |x| x)],
        "result",
    )
    .unwrap();
    // Results 10, 0, 10, -40: the median state is both atoms at 10.
    let k = pnl.scenarios(0.5).unwrap();
    assert!((k[0] - 50.0 / 3.0).abs() < 1e-12);
    assert!((k[2] - 10.0).abs() < 1e-12);
    let premium_sd = Ledger::single(
        &source,
        "c",
        Role::Sell,
        vec![Leg::constant("P", 3.0)],
        vec![Leg::of("L", |x| x)],
        "r",
    )
    .unwrap();
    assert_eq!(premium_sd.rows()[0].sd(), 0.0);
    assert_eq!(premium_sd.rows()[0].skew(), None);
}

#[test]
fn bad_ledgers_are_refused() {
    let source = Source::new(vec![1.0, 2.0], vec![0.5, 0.5]).unwrap();
    assert!(Ledger::new(&source, vec![], Layout::new("x")).is_err());
    assert!(Ledger::new(&source, vec![Group::sell("A")], Layout::new("x")).is_err());
    let dup = vec![
        Group::sell("A").consideration(Leg::constant("P", 1.0)),
        Group::buy("B").consideration(Leg::constant("P", 1.0)),
    ];
    assert!(Ledger::new(&source, dup, Layout::new("x")).is_err());
    let bad = Group::sell("A").obligation(Leg::values("L", vec![1.0]));
    assert!(Ledger::new(&source, vec![bad], Layout::new("x")).is_err());
    let span = Layout::new("x").span("All", 0, 3);
    assert!(
        Ledger::new(
            &source,
            vec![Group::sell("A").consideration(Leg::constant("P", 1.0))],
            span
        )
        .is_err()
    );
    assert!(Source::new(vec![1.0], vec![0.0]).is_err());
}

#[test]
fn extending_a_ledger_keeps_its_rows() {
    let pnl = two_books();
    let source = Source::new(vec![0.0, 10.0, 20.0, 50.0], vec![0.4, 0.3, 0.2, 0.1]).unwrap();
    let xl2 = Ledger::new(
        &source,
        vec![Group::buy("XL2").consideration(Leg::constant("XL2 premium", 1.0))],
        Layout::new("XL2"),
    )
    .unwrap();
    let both = pnl
        .extend(&xl2, Layout::new("Both").result_name("margin"))
        .unwrap();
    assert!((both.result().mean() - 5.3).abs() < 1e-12);
    assert!(
        (both.row("XL recovery").unwrap().mean() - pnl.row("XL recovery").unwrap().mean()).abs()
            < 1e-15
    );
}
