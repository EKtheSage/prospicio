//! Taylor's arithmetic separation method (`reserving-inflation.md`, PR 3,
//! #979) against the Poisson GLM on development and calendar factors that
//! it equals (statsmodels; `scripts/reserving_separation_statsmodels.py`)
//! on GenIns and ABC, with and without a normaliser, and against Taylor's
//! (1977) printed Example 1 (validation target 2).

use std::collections::HashMap;

use prospicio_reserving::{
    FutureInflation, Period, Separation, SeparationFit, SeparationKind, Triangle,
};
use prospicio_validation::{check, reference, triangle, triangle_columns};

fn fit(dataset: &str, method: &str) -> SeparationFit {
    let (tri, column, normaliser): (Triangle, &str, Option<Vec<f64>>) = match method {
        "arithmetic" => (triangle(dataset), "values", None),
        "arithmetic_premium" => {
            let tri = triangle_columns("genins_premium", &["paid", "premium"]);
            let premium = tri
                .latest_diagonal()
                .values(0, tri.column_position("premium").unwrap());
            (tri, "paid", Some(premium))
        }
        other => panic!("unknown method {other}"),
    };
    Separation {
        kind: SeparationKind::Arithmetic,
        normaliser,
        future: FutureInflation::Rate(0.05),
    }
    .fit(&tri, column)
    .unwrap()
}

#[test]
fn arithmetic_separation_matches_the_poisson_glm() {
    let mut fits: HashMap<(String, String), SeparationFit> = HashMap::new();
    check(&reference("reserving_separation_statsmodels.csv"), |case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let fit = fits
            .entry((dataset.into(), method.into()))
            .or_insert_with(|| fit(dataset, method));
        let origin = |year: f64| {
            fit.origins
                .iter()
                .position(|&p| p == Period::year(year as i32))
        };
        let age = |months: f64| fit.ages.iter().position(|&a| f64::from(a) == months);
        match case.get("quantity") {
            "proportion" => Some(fit.proportions[age(case.number("arg")?)?]),
            "index" => {
                let year = Period::year(case.number("arg")? as i32);
                let h = fit.calendar.iter().position(|&p| p == year)?;
                Some(fit.indices[h])
            }
            "fitted" => {
                let (o, a) = (origin(case.param("arg", "origin"))?, age(case.param("arg", "age"))?);
                fit.fitted[o][a]
            }
            "future" => {
                let (o, a) = (origin(case.param("arg", "origin"))?, case.param("arg", "age"));
                fit.future
                    .cells
                    .iter()
                    .find(|c| c.origin == o && f64::from(c.age) == a)
                    .map(|c| c.value)
            }
            "reserve" => Some(fit.reserves[origin(case.number("arg")?)?]),
            "total_reserve" => Some(fit.total_reserve()),
            _ => None,
        }
    });
}

/// Taylor (1977), Example 1 (a motor account), p. 228: the per-claim
/// incremental triangle, its printed `r_j` and `lambda_h` (4 decimals and
/// 1), the fitted triangle, the projection at 10% a year, and the 3 by 3
/// sub-triangle's `r_j` and `lambda_h`.
#[test]
fn taylor_1977_example_1() {
    let rows: [&[f64]; 4] = [
        &[50.4, 78.6, 87.6, 92.4],
        &[58.0, 87.2, 96.9],
        &[59.5, 92.7],
        &[66.2],
    ];
    let build = |rows: &[&[f64]]| {
        let (mut origin, mut age, mut value) = (vec![], vec![], vec![]);
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                origin.push(prospicio_reserving::Month::january(2001 + i as i32));
                age.push(12 * (j as u32 + 1));
                value.push(*v);
            }
        }
        Triangle::from_long(&prospicio_reserving::Long {
            keys: &[],
            origin: &origin,
            development: prospicio_reserving::DevelopmentColumn::Age(&age),
            values: &[("s", &value)],
            origin_grain: prospicio_reserving::Grain::Year,
            development_grain: prospicio_reserving::Grain::Year,
            cumulative: true,
        })
        .unwrap()
    };
    let sep = Separation {
        kind: SeparationKind::Arithmetic,
        normaliser: None,
        future: FutureInflation::Rate(0.10),
    };
    let fit = sep.fit(&build(&rows), "s").unwrap();
    for (got, want) in fit.proportions.iter().zip([0.5835, 0.2878, 0.0866, 0.0421]) {
        assert!((got - want).abs() <= 5e-5, "r {got} {want}");
    }
    for (got, want) in fit.indices.iter().zip([86.4, 98.9, 102.0, 113.9]) {
        assert!((got - want).abs() <= 0.05, "lambda {got} {want}");
    }
    let printed = [[50.4, 28.5, 8.8, 4.8], [57.7, 29.4, 9.9, 0.0], [59.5, 32.8, 0.0, 0.0], [66.5, 0.0, 0.0, 0.0]];
    for (i, row) in fit.fitted.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            if let Some(v) = cell {
                assert!((v - printed[i][j]).abs() <= 0.06, "fitted {i} {j}: {v}");
            }
        }
    }
    let future: Vec<f64> = fit.future.cells.iter().map(|c| c.value).collect();
    for (got, want) in future.iter().zip([5.3, 10.8, 5.8, 36.1, 11.9, 6.4]) {
        assert!((got - want).abs() <= 0.06, "future {got} {want}");
    }

    let sub = sep
        .fit(&build(&[&[50.4, 78.6, 87.6], &[58.0, 87.2], &[59.5]]), "s")
        .unwrap();
    // Taylor prints 0.6101 where the recursion gives 0.60993 (his
    // intermediate rounding, presumably), so 2e-4 here.
    for (got, want) in sub.proportions.iter().zip([0.6101, 0.2980, 0.0921]) {
        assert!((got - want).abs() <= 2e-4, "r {got} {want}");
    }
    for (got, want) in sub.indices.iter().zip([82.6, 94.9, 97.7]) {
        assert!((got - want).abs() <= 0.05, "lambda {got} {want}");
    }
}
