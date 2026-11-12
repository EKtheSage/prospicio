//! Nested Archimedean copulas (decision 64), reached through the
//! catalogue as Excel reaches them: the distribution function against
//! mpmath at 30 digits (`validation/reference/nested_copula_mpmath.csv`,
//! written by `validation/scripts/mpmath_nested_copula.py`), simulated
//! uniforms equal to `prospicio_prob`'s draws (and so to Python's
//! `NestedArchimedeanCopula.sample`), each pair's Kendall's tau that of
//! the node where the pair meets, and every malformed tree named.

use crate::common;

use common::{n, num, reference, t, text};
use prospicio_core::StreamRng;
use prospicio_prob::copula::{Archimedean, Copula, NestedArchimedeanCopula, NestedNode};
use prospicio_prob::vine_fit::kendall_tau;
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn s(x: &str) -> Value {
    Value::Text(x.into())
}

/// A node table from (node, parent, theta) rows; "" is a blank parent.
fn nodes(rows: &[(Value, Value, f64)]) -> Grid {
    Grid::from_rows(
        rows.iter()
            .map(|(a, b, th)| vec![a.clone(), b.clone(), Value::Number(*th)])
            .collect(),
    )
}

fn column(values: Vec<Value>) -> Grid {
    Grid::from_rows(values.into_iter().map(|v| vec![v]).collect())
}

/// The three trees of the mpmath script, as a sheet holds them: the
/// node table and the members column.
fn tree(name: &str) -> (Grid, Grid) {
    let e = Value::Empty;
    let num = Value::Number;
    match name {
        // 0 alone at 1.5; {1, 2} at 3; {3, 4} at 2.
        "gumbel5" => (
            nodes(&[
                (s("root"), e, 1.5),
                (s("a"), s("root"), 3.0),
                (s("b"), s("root"), 2.0),
            ]),
            column(["root", "a", "a", "b", "b"].map(s).to_vec()),
        ),
        // Three levels, the names numbers: 0 at 1, 1 at 2.5, {2, 3} at 4.
        "clayton4" => (
            nodes(&[
                (num(1.0), e, 1.0),
                (num(2.0), num(1.0), 2.5),
                (num(3.0), num(2.0), 4.0),
            ]),
            column([1.0, 2.0, 3.0, 3.0].map(num).to_vec()),
        ),
        // Two groups and no dimension at the root; members as a row.
        "clayton4b" => (
            nodes(&[
                (s("all"), e, 0.5),
                (s("x"), s("all"), 2.0),
                (s("y"), s("all"), 3.0),
            ]),
            Grid::from_rows(vec![["x", "x", "y", "y"].map(s).to_vec()]),
        ),
        _ => unreachable!(),
    }
}

#[test]
fn cdfs_match_mpmath() {
    let mut checked = 0;
    for row in reference("nested_copula_mpmath.csv").into_iter().skip(1) {
        let (nodes, members) = tree(&row[0]);
        let point: Vec<f64> = row[2].split(';').map(|x| x.parse().unwrap()).collect();
        let got = num(&call(
            "pro_cap_nested_cdf",
            vec![Grid::row(&point), t(&row[1]), nodes, members],
        )
        .cells[0]);
        let (want, abs): (f64, f64) = (row[3].parse().unwrap(), row[4].parse().unwrap());
        assert!(
            (got - want).abs() <= abs,
            "{} at {point:?}: {got} against {want}",
            row[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 12);
}

#[test]
fn points_are_rows() {
    let (nodes, members) = tree("clayton4");
    let points = Grid::from_rows(vec![
        [0.5, 0.5, 0.5, 0.5].map(Value::Number).to_vec(),
        [0.0, 0.5, 0.5, 0.5].map(Value::Number).to_vec(),
        [1.0, 1.0, 1.0, 0.3].map(Value::Number).to_vec(),
    ]);
    let out = call("pro_cap_nested_cdf", vec![points, t("Clayton"), nodes, members]);
    assert_eq!((out.rows, out.columns), (3, 1));
    assert!((num(&out.cells[0]) - 0.262_623_949_120_066_9).abs() < 1e-14);
    assert_eq!(num(&out.cells[1]), 0.0);
    // Every other margin at 1 leaves the last one's own value.
    assert!((num(&out.cells[2]) - 0.3).abs() < 1e-14);
}

#[test]
fn draws_are_the_crates() {
    let (nodes, members) = tree("gumbel5");
    let out = call(
        "pro_cap_nested_sim",
        vec![t("gumbel"), nodes, members, n(500.0), n(7.0)],
    );
    assert_eq!((out.rows, out.columns), (500, 5));
    let c = NestedArchimedeanCopula::new(
        Archimedean::Gumbel,
        NestedNode::new(
            1.5,
            vec![0],
            vec![
                NestedNode::new(3.0, vec![1, 2], vec![]),
                NestedNode::new(2.0, vec![3, 4], vec![]),
            ],
        ),
    )
    .unwrap();
    let mut u = [0.0; 5];
    for i in 0..500 {
        c.sample(&mut StreamRng::new(7, i as u64), &mut u);
        for (j, &x) in u.iter().enumerate() {
            assert_eq!(num(out.get(i, j)), x, "row {i} column {j}");
        }
    }
    // The default is 10,000 rows from seed 0.
    let (nodes, members) = tree("gumbel5");
    let all = call("pro_cap_nested_sim", vec![t("gumbel"), nodes, members]);
    assert_eq!((all.rows, all.columns), (10_000, 5));
}

#[test]
fn each_pair_has_the_tau_of_the_node_where_it_meets() {
    let (nodes, members) = tree("clayton4");
    let out = call(
        "pro_cap_nested_sim",
        vec![t("clayton"), nodes, members, n(4000.0), n(11.0)],
    );
    let col = |j: usize| (0..out.rows).map(|i| num(out.get(i, j))).collect::<Vec<_>>();
    // Clayton's tau is theta / (theta + 2).
    let tau = |theta: f64| theta / (theta + 2.0);
    for (a, b, theta) in [
        (0, 1, 1.0),
        (0, 3, 1.0),
        (1, 2, 2.5),
        (1, 3, 2.5),
        (2, 3, 4.0),
    ] {
        let got = kendall_tau(&col(a), &col(b)).unwrap();
        assert!(
            (got - tau(theta)).abs() < 0.03,
            "dimensions {a} and {b}: tau {got} against {}",
            tau(theta)
        );
    }
}

#[test]
fn malformed_trees_are_named() {
    let e = Value::Empty;
    let one = |rows: &[(Value, Value, f64)], members: &[&str]| {
        let m = column(members.iter().map(|x| s(x)).collect());
        call(
            "pro_cap_nested_cdf",
            vec![Grid::row(&vec![0.5; members.len()]), t("clayton"), nodes(rows), m],
        )
    };
    let has = |got: &str, want: &str| assert!(got.contains(want), "{got}");
    has(
        text(&one(&[(s("a"), s("b"), 1.0), (s("b"), s("a"), 2.0)], &["a", "b"])),
        "one root",
    );
    has(
        text(&one(&[(s("a"), e.clone(), 1.0), (s("b"), e.clone(), 2.0)], &["a", "b"])),
        "2 roots",
    );
    has(
        text(&one(&[(s("a"), e.clone(), 1.0), (s("b"), s("c"), 2.0)], &["a", "b"])),
        "parent c is not in nodes",
    );
    has(
        text(&one(&[(s("a"), e.clone(), 1.0), (s("a"), s("a"), 2.0)], &["a", "a"])),
        "appears twice",
    );
    // A loop below the root: b and c are each other's parent.
    has(
        text(&one(
            &[(s("a"), e.clone(), 1.0), (s("b"), s("c"), 2.0), (s("c"), s("b"), 2.0)],
            &["a", "a", "b", "c"],
        )),
        "loops",
    );
    has(
        text(&one(&[(s("a"), e.clone(), 1.0), (s("b"), s("a"), 2.0)], &["a", "a", "b"])),
        "node b joins 1 member",
    );
    has(
        text(&one(&[(s("a"), e.clone(), 2.0), (s("b"), s("a"), 1.0)], &["a", "b", "b"])),
        "at least its parent's",
    );
    has(
        text(&one(&[(s("a"), e.clone(), 1.0)], &["a", "z"])),
        "dimension 2 joins node z",
    );
    let (nodes, members) = tree("gumbel5");
    has(
        text(&call(
            "pro_cap_nested_cdf",
            vec![Grid::row(&[0.5; 4]), t("gumbel"), nodes.clone(), members.clone()],
        )),
        "points has 4 columns but members has 5",
    );
    has(
        text(&call(
            "pro_cap_nested_sim",
            vec![t("frank"), nodes.clone(), members.clone()],
        )),
        "family must be",
    );
    has(
        text(&call(
            "pro_cap_nested_sim",
            vec![
                t("gumbel"),
                nodes,
                Grid::from_rows(vec![vec![s("a"), s("a")], vec![s("b"), s("b")]]),
            ],
        )),
        "a row or a column",
    );
    has(
        text(&call(
            "pro_cap_nested_sim",
            vec![t("gumbel"), Grid::row(&[1.0, 2.0]), members],
        )),
        "three columns",
    );
}
