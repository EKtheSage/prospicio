use super::*;

fn gross() -> Source {
    Source::new(
        vec![2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        vec![0.25, 0.3, 0.09, 0.2, 0.12, 0.04],
    )
    .unwrap()
}

#[test]
fn expense_names_follow_aggregate() {
    let book = Book::new("B", 10.0)
        .expense(Expense::premium(0.1))
        .expense(Expense::loss(0.05))
        .expense(Expense::fixed(1.0))
        .expense(Expense::premium(0.02));
    let l = book.ledger(&gross(), None, Face::Consolidated).unwrap();
    for label in [
        "premium expense",
        "loss expense",
        "fixed expense",
        "premium expense (2)",
    ] {
        assert!(l.row(label).is_some(), "{label}");
    }
    let mixed = Book::new("B", 10.0)
        .expense(Expense::premium(0.1).and(ExpenseBasis::Loss, 0.05))
        .ledger(&gross(), None, Face::Consolidated)
        .unwrap();
    assert!((mixed.row("expense").unwrap().mean() + 1.19).abs() < 1e-12);
}

#[test]
fn consolidated_loss_expense_at_each_atom() {
    let book = Book::new("B", 10.0).expense(Expense::loss(0.1));
    let cover = Tower::new(vec![
        Layer::stop_loss("SL", 3.0, 5.0)
            .unwrap()
            .deposit_premium(1.0)
            .unwrap(),
    ])
    .unwrap();
    let c = book
        .ledger(&gross(), Some(&cover), Face::Consolidated)
        .unwrap();
    let w = book.ledger(&gross(), Some(&cover), Face::Walk).unwrap();
    // The consolidated margin is the walk's net margin at every level.
    for q in [0.01, 0.25, 0.5, 0.9, 0.99] {
        let (a, b) = (
            c.result().quantile(q).unwrap(),
            w.result().quantile(q).unwrap(),
        );
        assert!((a - b).abs() < 1e-12, "{q}: {a} {b}");
    }
    assert!((c.result().sd() - w.result().sd()).abs() < 1e-12);
    // At the expected loss: the same mean, a constant leg.
    let m = book
        .clone()
        .expense_at_expected_loss()
        .ledger(&gross(), Some(&cover), Face::Consolidated)
        .unwrap();
    assert!((m.result().mean() - c.result().mean()).abs() < 1e-12);
    assert_eq!(m.row("expense").unwrap().sd(), 0.0);
    assert!(c.row("expense").unwrap().sd() > 0.0);
}

/// The cedant's ledger books a deal's commission, on its sections'
/// loss ratio together, with the cover that buys its last section:
/// two 50% quota shares in one deal are one 100% quota share with the
/// deal's scale as its own.
#[test]
fn a_deal_commission_is_booked_with_its_last_section() {
    let scale = vec![(0.35, 0.3), (0.15, 0.9)];
    let half = |name: &str| {
        Layer::quota_share(name, 0.5)
            .unwrap()
            .premium_rate(1.0, 10.0)
            .unwrap()
    };
    let deal = Tower::new(vec![half("A"), half("B")])
        .unwrap()
        .deal_scale(
            crate::DealScale::new("deal", ["A", "B"])
                .unwrap()
                .sliding_scale(scale.clone())
                .unwrap()
                .brokerage(0.02)
                .unwrap(),
        )
        .unwrap();
    let whole = Tower::new(vec![
        Layer::quota_share("QS", 1.0)
            .unwrap()
            .premium_rate(1.0, 10.0)
            .unwrap()
            .sliding_scale(scale)
            .unwrap(),
    ])
    .unwrap();
    let book = Book::new("B", 10.0);
    let walk = book.ledger(&gross(), Some(&deal), Face::Walk).unwrap();
    // At each atom, the commission apply_annual gives the deal.
    let sims = deal.apply_annual(gross().values()).unwrap();
    let key = vec!["ceding_commission".into(), "deal".into()];
    let expected = prospicio_prob::Empirical::draws(&sims.marginal(&key).unwrap()).to_vec();
    let row = walk.row("deal commission").unwrap();
    assert!(
        row.values()
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    assert!(row.values()[0] != row.values()[5]);
    // Brokerage comes out of what the reinsurer receives: not booked.
    assert!(walk.rows().iter().all(|r| !r.label.contains("brokerage")));
    // The peel books it on the last section too; every face agrees.
    let w = book.ledger(&gross(), Some(&whole), Face::Walk).unwrap();
    for face in [
        Face::Walk,
        Face::PeelTopDown,
        Face::PeelBottomUp,
        Face::Consolidated,
    ] {
        let l = book.ledger(&gross(), Some(&deal), face).unwrap();
        for q in [0.01, 0.3, 0.6, 0.99] {
            let (a, b) = (
                l.result().quantile(q).unwrap(),
                w.result().quantile(q).unwrap(),
            );
            assert!((a - b).abs() < 1e-12, "{face:?} {q}: {a} {b}");
        }
    }
}

#[test]
fn built_where_aggregate_refuses() {
    // A walk over a retro book: the one gross group (aggregate's xpnl
    // refuses retro; prospicio's walk has nothing to walk through).
    let retro = Book::new("R", 2.0)
        .retro(LossSensitivePremium::new(2.0, 1.1, Some(3.0), Some(9.0)).unwrap());
    let w = retro.ledger(&gross(), None, Face::Walk).unwrap();
    let c = retro.ledger(&gross(), None, Face::Consolidated).unwrap();
    assert!((w.result().mean() - c.result().mean()).abs() < 1e-12);
    // A peel of a single reinstated occurrence layer is its walk.
    let joint = Source::joint(
        vec![2.0, 4.0, 6.0],
        vec![1.0, 2.0, 4.0],
        vec![0.5, 0.3, 0.2],
    )
    .unwrap();
    let occ = Tower::new(vec![
        Layer::xol("XL", 2.0, 2.0)
            .unwrap()
            .paid_reinstatements(1.0, vec![1.0])
            .unwrap(),
    ])
    .unwrap();
    let b = Book::new("B", 10.0);
    let walk = b.occurrence_ledger(&joint, &occ, None, Face::Walk).unwrap();
    let peel = b
        .occurrence_ledger(&joint, &occ, None, Face::PeelTopDown)
        .unwrap();
    assert_eq!(walk.rows().len(), peel.rows().len());
    for (r, s) in walk.rows().iter().zip(peel.rows()) {
        assert_eq!(r.label, s.label);
        assert!((r.mean() - s.mean()).abs() < 1e-12);
    }
}

#[test]
fn stitched_peel() {
    use prospicio_prob::Poisson;
    let sev = Grid::new(1.0, vec![0.0, 0.4, 0.3, 0.0, 0.2, 0.0, 0.1]).unwrap();
    let occ = Tower::new(vec![
        Layer::xol("a", 1.0, 1.0)
            .unwrap()
            .deposit_premium(1.0)
            .unwrap(),
        Layer::xol("b", 2.0, 2.0)
            .unwrap()
            .deposit_premium(1.0)
            .unwrap(),
    ])
    .unwrap();
    let n = Poisson::new(2.0).unwrap();
    let book = Book::new("S", 20.0);
    let peel = book
        .occurrence_peel(&n, &sev, &occ, None, Face::PeelBottomUp, 64)
        .unwrap();
    assert!(peel.is_stitched());
    // The mean column foots: the grand result is the gross less the net
    // cost of the covers.
    let gross = peel.row("Gross result").unwrap().mean();
    let grand = peel.result().mean();
    let covers: f64 = ["a result", "b result"]
        .iter()
        .map(|l| peel.row(l).unwrap().mean())
        .sum();
    assert!((grand - gross - covers).abs() < 1e-9);
    assert!(peel.scenarios(0.5).is_err());
    assert!(peel.extend(&peel, Layout::new("x")).is_err());
    let impact = peel.row("total impact").unwrap();
    assert!(impact.is_delta());
    assert!(
        (impact.sd() - (peel.result().sd() - peel.row("Gross result").unwrap().sd())).abs() < 1e-15
    );
    // Refusals: a walk, fixed terms only on the occurrence layers.
    assert!(
        book.occurrence_peel(&n, &sev, &occ, None, Face::Walk, 64)
            .is_err()
    );
    let capped = Tower::new(vec![
        Layer::xol("a", 1.0, 1.0)
            .unwrap()
            .aggregate_limit(2.0)
            .unwrap(),
    ])
    .unwrap();
    assert!(
        book.occurrence_peel(&n, &sev, &capped, None, Face::PeelTopDown, 64)
            .is_err()
    );
    assert!(
        book.occurrence_peel(&n, &sev, &occ, None, Face::PeelTopDown, 4)
            .is_err()
    );
}

#[test]
fn occurrence_refusals() {
    let book = Book::new("B", 10.0);
    let occ = Tower::new(vec![Layer::xol("a", 1.0, 1.0).unwrap()]).unwrap();
    // Not a joint source.
    assert!(
        book.occurrence_ledger(&gross(), &occ, None, Face::Walk)
            .is_err()
    );
    let joint = Source::joint(vec![2.0, 4.0], vec![1.0, 2.0], vec![0.5, 0.5]).unwrap();
    assert!(
        book.occurrence_ledger(&joint, &occ, None, Face::Walk)
            .is_ok()
    );
    // Annual terms on a single occurrence layer apply to the joint's
    // annual cession; on a tier of several they cannot.
    let capped = Layer::xol("a", 1.0, 1.0)
        .unwrap()
        .aggregate_limit(1.5)
        .unwrap();
    let one = Tower::new(vec![capped.clone()]).unwrap();
    let l = book
        .occurrence_ledger(&joint, &one, None, Face::Walk)
        .unwrap();
    // Ceded 1 and 2, capped at 1.5: a recovery of 1.25.
    assert!((l.row("a recovery").unwrap().mean() - 1.25).abs() < 1e-12);
    let annual = Tower::new(vec![capped, Layer::xol("b", 2.0, 2.0).unwrap()]).unwrap();
    assert!(
        book.occurrence_ledger(&joint, &annual, None, Face::Walk)
            .is_err()
    );
    // A peel through two occurrence layers.
    let two = Tower::new(vec![
        Layer::xol("a", 1.0, 1.0).unwrap(),
        Layer::xol("b", 2.0, 2.0).unwrap(),
    ])
    .unwrap();
    assert!(
        book.occurrence_ledger(&joint, &two, None, Face::Walk)
            .is_ok()
    );
    assert!(
        book.occurrence_ledger(&joint, &two, None, Face::PeelTopDown)
            .is_err()
    );
}

#[test]
fn refusals() {
    let book = Book::new("B", 10.0);
    assert!(book.ledger(&gross(), None, Face::PeelTopDown).is_err());
    let pc = Layer::stop_loss("SL", 3.0, 5.0)
        .unwrap()
        .deposit_premium(1.0)
        .unwrap()
        .profit_commission(0.2, 0.1)
        .unwrap();
    let t = Tower::new(vec![pc]).unwrap();
    assert!(book.ledger(&gross(), Some(&t), Face::Walk).is_ok());
    // Reinstatement premiums, and retro with a cover.
    let reinstated = Layer::xol("XL", 2.0, 2.0)
        .unwrap()
        .paid_reinstatements(1.0, vec![1.0])
        .unwrap();
    let r = Tower::new(vec![reinstated]).unwrap();
    assert!(book.ledger(&gross(), Some(&r), Face::Walk).is_err());
    let retro = Book::new("R", 2.0)
        .retro(LossSensitivePremium::new(2.0, 1.1, Some(3.0), Some(9.0)).unwrap());
    assert!(retro.ledger(&gross(), Some(&t), Face::Walk).is_err());
    let plain = retro.ledger(&gross(), None, Face::Consolidated).unwrap();
    assert!((plain.row("Premium").unwrap().mean() - 6.108).abs() < 1e-12);
    assert!(derive_premium(1.0, &[Expense::loss(0.1)]).is_err());
    assert!(derive_premium(1.0, &[Expense::premium(1.0)]).is_err());
}
