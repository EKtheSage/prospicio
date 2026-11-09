use super::*;

fn insco() -> Portfolio {
    let rows = [
        [15.0, 7.0, 0.0],
        [15.0, 13.0, 0.0],
        [5.0, 20.0, 11.0],
        [7.0, 33.0, 0.0],
        [13.0, 20.0, 7.0],
        [5.0, 27.0, 8.0],
        [15.0, 16.0, 9.0],
        [26.0, 19.0, 10.0],
        [17.0, 8.0, 40.0],
        [16.0, 20.0, 64.0],
    ];
    let rows: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    Portfolio::from_rows(vec!["A".into(), "B".into(), "C".into()], &rows, None).unwrap()
}

#[test]
fn allocations_add_up() {
    let port = insco();
    for g in [
        Distortion::ccoc(0.15).unwrap(),
        Distortion::proportional_hazard(0.72).unwrap(),
        Distortion::wang(0.34).unwrap(),
        Distortion::tvar(0.27).unwrap(),
    ] {
        for a in [100.0, 65.0, 50.0, 40.0, 30.0, 10.0] {
            for method in [Allocation::Linear, Allocation::Lifted] {
                let p = port.price(&g, a, method).unwrap();
                let sum = |f: fn(&Pentagon) -> f64| p.allocated.iter().map(f).sum::<f64>();
                let t = p.total;
                assert!(
                    (sum(|u| u.loss) - t.loss).abs() < 1e-10,
                    "{g:?} {a} {method:?}"
                );
                assert!(
                    (sum(|u| u.premium) - t.premium).abs() < 1e-10,
                    "{g:?} {a} {method:?}"
                );
                assert!(
                    (sum(|u| u.capital) - t.capital).abs() < 1e-9,
                    "{g:?} {a} {method:?}: {} vs {}",
                    sum(|u| u.capital),
                    t.capital
                );
            }
        }
    }
}

#[test]
fn independent_units_match_brute_force() {
    let a = Grid::new(
        1.0,
        vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25],
    )
    .unwrap();
    let mut bp = vec![0.0; 91];
    bp[0] = 0.5;
    bp[1] = 0.25;
    bp[90] = 0.25;
    let b = Grid::new(1.0, bp).unwrap();
    let c = Grid::new(1.0, vec![0.2, 0.3, 0.5]).unwrap();
    let port = Portfolio::from_independent(
        vec!["a".into(), "b".into(), "c".into()],
        &[a.clone(), b.clone(), c.clone()],
    )
    .unwrap();
    // Every combination, by hand.
    let mut rows = Vec::new();
    let mut probs = Vec::new();
    for (i, pa) in a.probs().iter().enumerate() {
        for (j, pb) in b.probs().iter().enumerate() {
            for (k, pc) in c.probs().iter().enumerate() {
                if pa * pb * pc > 0.0 {
                    rows.push(vec![i as f64, j as f64, k as f64]);
                    probs.push(pa * pb * pc);
                }
            }
        }
    }
    let brute = Portfolio::from_rows(
        vec!["a".into(), "b".into(), "c".into()],
        &rows,
        Some(&probs),
    )
    .unwrap();
    assert_eq!(port.totals(), brute.totals());
    for (x, y) in port.probs().iter().zip(brute.probs()) {
        assert!((x - y).abs() < 1e-15);
    }
    for i in 0..3 {
        for (x, y) in port.kappa(i).iter().zip(brute.kappa(i)) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }
}

#[test]
fn epd_and_bodoff() {
    let port = insco();
    let (total, units) = port.epd(100.0);
    assert_eq!(total, 0.0);
    assert!(units.iter().all(|u| u.abs() < 1e-15));
    // With assets 65 only the 100 year defaults, by 35 with p 0.1.
    let (total, _) = port.epd(65.0);
    assert!((total - 3.5 / 46.6).abs() < 1e-15);
    let a = port.assets_for_epd(3.5 / 46.6).unwrap();
    assert!((a - 65.0).abs() < 1e-9);
    let b = port.bodoff(100.0);
    assert!((b.iter().sum::<f64>() - 100.0).abs() < 1e-10);
    assert_eq!(port.assets(1.0).unwrap(), 100.0);
    assert_eq!(port.assets(0.85).unwrap(), 65.0);
    assert_eq!(port.assets(0.7).unwrap(), 40.0);
}

#[test]
fn pentagon_solves_any_determined_triple() {
    use Quantity::*;
    let want = Pentagon::new(46.6, 53.565217391304344, 100.0);
    let all = [
        (Loss, want.loss),
        (Margin, want.margin),
        (Premium, want.premium),
        (Capital, want.capital),
        (Assets, want.assets),
        (LossRatio, want.loss_ratio()),
        (PremiumToCapital, want.premium_to_capital()),
        (ReturnOnCapital, want.return_on_capital()),
    ];
    let mut solved = 0;
    for i in 0..8 {
        for j in i + 1..8 {
            for k in j + 1..8 {
                if let Ok(p) = Pentagon::solve([all[i], all[j], all[k]]) {
                    solved += 1;
                    for (got, w) in [
                        (p.loss, want.loss),
                        (p.premium, want.premium),
                        (p.assets, want.assets),
                    ] {
                        assert!((got - w).abs() < 1e-9 * w, "{:?}", [all[i], all[j], all[k]]);
                    }
                }
            }
        }
    }
    // 46 of the 56 triples determine the pentagon. The rest are three
    // ratios, or a relation given twice: {L, M, P}, {P, Q, a},
    // {L, P, LR}, {P, Q, PQ}, {M, Q, ι}, and the like.
    assert_eq!(solved, 46);
    assert!(Pentagon::solve([(Loss, 1.0), (Margin, 1.0), (Premium, 2.0)]).is_err());
}

#[test]
fn premium_bounds_match_aggregate() {
    // aggregate 2.1.0 AllocationBounds on InsCo (1.0.1 the same), to its printed digits.
    let port = insco();
    let b = port.premium_bounds(53.565217391304344, 100.0).unwrap();
    let want = [
        (13.097826, 15.032744),
        (17.465726, 20.411685),
        (19.254738, 22.098038),
    ];
    for (got, (lo, hi)) in b.iter().zip(want) {
        assert!(
            (got.lower - lo).abs() < 1e-6 && (got.upper - hi).abs() < 1e-6,
            "{got:?}"
        );
    }
    let b = port.premium_bounds(47.0, 65.0).unwrap();
    let want = [
        (12.374586, 13.881867),
        (16.335616, 19.292994),
        (15.057851, 17.671233),
    ];
    for (got, (lo, hi)) in b.iter().zip(want) {
        assert!(
            (got.lower - lo).abs() < 1e-6 && (got.upper - hi).abs() < 1e-6,
            "{got:?}"
        );
    }
    // Each bound's distortion prices the total at the premium.
    let capped: Vec<f64> = port.totals().iter().map(|x| x.min(65.0)).collect();
    for u in &b {
        for d in [&u.lower_distortion, &u.upper_distortion] {
            assert!((d.apply_discrete(&capped, port.probs()) - 47.0).abs() < 1e-9);
        }
    }
}

/// The bounds by pricing the book at every knot and trying every pair,
/// as `premium_bounds` did before its prefix sums and hulls.
fn bounds_by_every_pair(port: &Portfolio, premium: f64, a: f64) -> Vec<(f64, f64)> {
    let capped: Vec<f64> = port.x.iter().map(|x| x.min(a)).collect();
    let tvar = |p: f64| {
        Distortion::tvar(p)
            .unwrap()
            .apply_discrete(&capped, &port.p)
    };
    let p_star = prospicio_math::roots::bisect(0.0, 1.0, |p| tvar(p) < premium);
    let mut knots = vec![0.0, p_star];
    let mut cum = 0.0;
    for (&x, &pk) in port.x.iter().zip(&port.p) {
        cum += pk;
        if x < a && cum < 1.0 {
            knots.push(cum);
        }
    }
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let t: Vec<f64> = knots.iter().map(|&p| tvar(p)).collect();
    let units: Vec<Vec<f64>> = knots
        .iter()
        .map(|&p| {
            let price = port
                .price(&Distortion::tvar(p).unwrap(), a, Allocation::Linear)
                .unwrap();
            price.allocated.iter().map(|u| u.premium).collect()
        })
        .collect();
    let tol = 1e-12 * premium.abs().max(1.0);
    let mut out = vec![(f64::INFINITY, f64::NEG_INFINITY); port.units.len()];
    for i in 0..knots.len() {
        for j in i..knots.len() {
            if t[i] > premium + tol || t[j] < premium - tol {
                continue;
            }
            let w = if t[j] - t[i] > tol {
                (premium - t[i]) / (t[j] - t[i])
            } else {
                0.0
            };
            if !(0.0..=1.0).contains(&w) {
                continue;
            }
            for (u, o) in out.iter_mut().enumerate() {
                let v = (1.0 - w) * units[i][u] + w * units[j][u];
                *o = (o.0.min(v), o.1.max(v));
            }
        }
    }
    out
}

#[test]
fn premium_bounds_match_every_pair() {
    // A book with ties and an atom at the assets, by a fixed LCG.
    let mut state = 12345_u64;
    let mut next = |k: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % k) as f64
    };
    let rows: Vec<Vec<f64>> = (0..80)
        .map(|_| vec![next(20), next(7) * next(5), next(3) * 10.0])
        .collect();
    let units = vec!["A".into(), "B".into(), "C".into()];
    let book = Portfolio::from_rows(units, &rows, None).unwrap();
    for port in [insco(), book] {
        let x = port.totals().to_vec();
        for a in [x[x.len() / 2], x[x.len() * 3 / 4], x[x.len() - 1]] {
            let (loss, _) = port.limited_expected(a);
            for f in [0.1, 0.4, 0.8] {
                let premium = loss + f * (a - loss);
                let got = port.premium_bounds(premium, a).unwrap();
                let want = bounds_by_every_pair(&port, premium, a);
                let capped: Vec<f64> = x.iter().map(|v| v.min(a)).collect();
                for (g, (lo, hi)) in got.iter().zip(want) {
                    let near = |u: f64, v: f64| (u - v).abs() < 1e-9 * v.abs().max(1.0);
                    assert!(
                        near(g.lower, lo) && near(g.upper, hi),
                        "{g:?} vs {lo}, {hi}"
                    );
                    for d in [&g.lower_distortion, &g.upper_distortion] {
                        assert!(near(d.apply_discrete(&capped, port.probs()), premium));
                    }
                }
            }
        }
    }
}

#[test]
fn calibration_targets() {
    let port = insco();
    let ccoc = port
        .calibrate(Family::Ccoc, 100.0, Target::ReturnOnCapital(0.15))
        .unwrap();
    assert!(matches!(ccoc, Distortion::Ccoc(r) if (r - 0.15).abs() < 1e-12));
    let ph = port
        .calibrate(Family::ProportionalHazard, 65.0, Target::LossRatio(0.9))
        .unwrap();
    let p = port.price(&ph, 65.0, Allocation::Linear).unwrap();
    assert!((p.total.loss_ratio() - 0.9).abs() < 1e-9);
}
