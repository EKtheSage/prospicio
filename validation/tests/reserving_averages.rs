//! Averages over the latest periods, excluding high and low link ratios, and
//! the geometric average (decision 13 of `docs/design/reserving-v02.md`).
//!
//! * chainladder-python parity: `Development(n_periods, drop_high,
//!   drop_low)` on RAA and GenIns for the volume, simple and regression
//!   averages (factors, sigmas, standard errors, the number of link ratios
//!   behind each factor and the chain ladder's ultimates), and the
//!   Bornhuetter-Ferguson and Cape Cod totals on the CAS workers'
//!   compensation triangle (`reference/reserving_averages_python.csv`, from
//!   `scripts/reserving_averages_python.py`).
//! * The books: Friedland's U.S. Industry Auto averages over the latest
//!   periods (simple, medial, volume and geometric, to the three decimals
//!   printed) and XYZ Insurer's volume average over the latest 2 with its
//!   reported pattern below 1; Werner and Modlin's Appendix A straight
//!   average excluding high and low.
//! * Mack's standard errors under the exclusions against R ChainLadder's
//!   `MackChainLadder(weights = W)` (`reference/reserving_averages_r.csv`,
//!   from `scripts/reserving_averages_r.R`).

use prospicio_reserving::{
    Average, BornhuetterFerguson, CapeCod, ChainLadder, Development, DevelopmentColumn, Grain, Lag,
    Long, Mack, Month, PatternKind, Period, SelectedPattern, Triangle,
};
use prospicio_validation::{Case, check, reference, triangle, triangle_columns};

/// The settings of a reference `method` such as
/// `average=volume;n_periods=3;drop_high=1`.
fn development(method: &str) -> Development {
    let get = |name: &str| {
        method
            .split(';')
            .find_map(|p| p.strip_prefix(name)?.strip_prefix('='))
    };
    Development {
        average: match get("average") {
            Some("volume") | None => Average::Volume,
            Some("simple") => Average::Simple,
            Some("regression") => Average::Regression,
            Some(other) => panic!("unknown average {other}"),
        },
        n_periods: get("n_periods").map(|v| v.parse().unwrap()),
        drop_high: get("drop_high").map_or(0, |v| v.parse().unwrap()),
        drop_low: get("drop_low").map_or(0, |v| v.parse().unwrap()),
        ..Default::default()
    }
}

#[test]
fn matches_chainladder_python_development() {
    let cases = reference("reserving_averages_python.csv");
    let mut fits = std::collections::HashMap::new();
    check(&cases, |case: &Case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let key = (dataset.to_string(), method.to_string());
        let quantity = case.get("quantity");
        if dataset == "clrd_wkcomp" {
            let chain_ladder = ChainLadder {
                development: development(method),
                ..Default::default()
            };
            let tri = triangle_columns(dataset, &["paid", "premium"]);
            return match method.split(';').next().unwrap() {
                "bornhuetter_ferguson" => BornhuetterFerguson {
                    apriori: case.param("method", "apriori"),
                    chain_ladder,
                }
                .fit(&tri, "paid", "premium")
                .ok()
                .map(|f| f.ultimate.iter().sum()),
                "cape_cod" => CapeCod {
                    trend: case.param("method", "trend"),
                    decay: case.param("method", "decay"),
                    chain_ladder,
                }
                .fit(&tri, "paid", "premium")
                .ok()
                .map(|f| f.expected_loss.ultimate.iter().sum()),
                other => panic!("unknown method {other}"),
            };
        }
        let fit = fits.entry(key).or_insert_with(|| {
            ChainLadder {
                development: development(method),
                ..Default::default()
            }
            .fit(&triangle(dataset), "values")
            .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
        });
        let dev = &fit.development;
        let age = || {
            let age = case.number("arg")? as Lag;
            dev.development.iter().position(|&a| a == age)
        };
        match quantity {
            "ldf" => dev.ldf.get(age()?).copied(),
            "sigma" => dev.sigma.get(age()?).copied(),
            "std_err" => dev.std_err.get(age()?).copied(),
            "n_links" => dev.n_links.get(age()?).map(|&n| n as f64),
            "ultimate" => {
                let year = case.number("arg")? as i32;
                let o = fit.origins.iter().position(|&p| p == Period::year(year))?;
                Some(fit.ultimate[o])
            }
            "total_ultimate" => Some(fit.ultimate.iter().sum()),
            _ => None,
        }
    });
}

/// A triangle from rows of cumulative values, `(first origin year, first
/// age in months, values)`, one row per origin year.
fn rows_triangle(rows: &[(i32, Lag, &[f64])]) -> Triangle {
    let (mut origin, mut age, mut value) = (Vec::new(), Vec::new(), Vec::new());
    for &(year, first_age, values) in rows {
        for (k, &v) in values.iter().enumerate() {
            origin.push(Month::january(year));
            age.push(first_age + 12 * k as Lag);
            value.push(v);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", &value)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// Friedland, *Estimating Unpaid Claims Using Basic Techniques*, chapter 7,
/// U.S. Industry Auto reported claims, accident years 1998 to 2007.
fn us_reported() -> Triangle {
    let rows: [&[f64]; 10] = [
        &[
            37017487.0, 43169009.0, 45568919.0, 46784558.0, 47337318.0, 47533264.0, 47634419.0,
            47689655.0, 47724678.0, 47742304.0,
        ],
        &[
            38954484.0, 46045718.0, 48882924.0, 50219672.0, 50729292.0, 50926779.0, 51069285.0,
            51163540.0, 51185767.0,
        ],
        &[
            41155776.0, 49371478.0, 52358476.0, 53780322.0, 54303086.0, 54582950.0, 54742188.0,
            54837929.0,
        ],
        &[
            42394069.0, 50584112.0, 53704296.0, 55150118.0, 55895583.0, 56156727.0, 56299562.0,
        ],
        &[
            44755243.0, 52971643.0, 56102312.0, 57703851.0, 58363564.0, 58592712.0,
        ],
        &[45163102.0, 52497731.0, 55468551.0, 57015411.0, 57565344.0],
        &[45417309.0, 52640322.0, 55553673.0, 56976657.0],
        &[46360869.0, 53790061.0, 56786410.0],
        &[46582684.0, 54641339.0],
        &[48853563.0],
    ];
    let rows: Vec<(i32, Lag, &[f64])> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (1998 + i as i32, 12, *r))
        .collect();
    rows_triangle(&rows)
}

/// XYZ Insurer's reported (`paid = false`) or paid auto bodily injury
/// claims, accident years 1998 to 2008; 1998 starts at 36 months and 1999
/// at 24.
fn xyz(paid: bool) -> Triangle {
    let reported: [&[f64]; 11] = [
        &[
            11171.0, 12380.0, 13216.0, 14067.0, 14688.0, 16366.0, 16163.0, 15835.0, 15822.0,
        ],
        &[
            13255.0, 16405.0, 19639.0, 22473.0, 23764.0, 25094.0, 24795.0, 25071.0, 25107.0,
        ],
        &[
            15676.0, 18749.0, 21900.0, 27144.0, 29488.0, 34458.0, 36949.0, 37505.0, 37246.0,
        ],
        &[
            11827.0, 16004.0, 21022.0, 26578.0, 34205.0, 37136.0, 38541.0, 38798.0,
        ],
        &[
            12811.0, 20370.0, 26656.0, 37667.0, 44414.0, 48701.0, 48169.0,
        ],
        &[9651.0, 16995.0, 30354.0, 40594.0, 44231.0, 44373.0],
        &[16995.0, 40180.0, 58866.0, 71707.0, 70288.0],
        &[28674.0, 47432.0, 70340.0, 70655.0],
        &[27066.0, 46783.0, 48804.0],
        &[19477.0, 31732.0],
        &[18632.0],
    ];
    let paid_rows: [&[f64]; 11] = [
        &[
            6309.0, 8521.0, 10082.0, 11620.0, 13242.0, 14419.0, 15311.0, 15764.0, 15822.0,
        ],
        &[
            4666.0, 9861.0, 13971.0, 18127.0, 22032.0, 23511.0, 24146.0, 24592.0, 24817.0,
        ],
        &[
            1302.0, 6513.0, 12139.0, 17828.0, 24030.0, 28853.0, 33222.0, 35902.0, 36782.0,
        ],
        &[
            1539.0, 5952.0, 12319.0, 18609.0, 24387.0, 31090.0, 37070.0, 38519.0,
        ],
        &[2318.0, 7932.0, 13822.0, 22095.0, 31945.0, 40629.0, 44437.0],
        &[1743.0, 6240.0, 12683.0, 22892.0, 34505.0, 39320.0],
        &[2221.0, 9898.0, 25950.0, 43439.0, 52811.0],
        &[3043.0, 12219.0, 27073.0, 40026.0],
        &[3531.0, 11778.0, 22819.0],
        &[3529.0, 11865.0],
        &[3409.0],
    ];
    let source = if paid { &paid_rows } else { &reported };
    let rows: Vec<(i32, Lag, &[f64])> = source
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let first_age = match i {
                0 => 36,
                1 => 24,
                _ => 12,
            };
            (1998 + i as i32, first_age, *r)
        })
        .collect();
    rows_triangle(&rows)
}

fn three_decimals(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|v| (v * 1000.0).round() / 1000.0)
        .collect()
}

#[test]
fn friedland_us_industry_auto_averages_over_the_latest_periods() {
    let tri = us_reported();
    let ldf = |average, n_periods, drop_high, drop_low| {
        let dev = Development {
            average,
            n_periods,
            drop_high,
            drop_low,
            ..Default::default()
        };
        three_decimals(&dev.fit(&tri, "values").unwrap().ldf)
    };
    // Chapter 7, Exhibit I, reported claims, rows (5) to (10) of the
    // averages; the printed values are those of
    // `docs/guides/reserving-friedland.md`.
    let simple5 = [
        1.168, 1.058, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
    ];
    let simple3 = [
        1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000,
    ];
    let medial = [
        1.165, 1.057, 1.027, 1.010, 1.004, 1.003, 1.002, 1.001, 1.000,
    ];
    let volume5 = simple5;
    let volume3 = simple3;
    let geometric4 = [
        1.164, 1.057, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
    ];
    assert_eq!(ldf(Average::Simple, Some(5), 0, 0), simple5);
    assert_eq!(ldf(Average::Simple, Some(3), 0, 0), simple3);
    assert_eq!(ldf(Average::Simple, Some(5), 1, 1), medial);
    assert_eq!(ldf(Average::Volume, Some(5), 0, 0), volume5);
    assert_eq!(ldf(Average::Volume, Some(3), 0, 0), volume3);
    assert_eq!(ldf(Average::Geometric, Some(4), 0, 0), geometric4);
    // "Latest 5" at the oldest ages uses the links there are: the last
    // factor rests on one link ratio, the one before on two.
    let dev = Development {
        n_periods: Some(5),
        ..Default::default()
    };
    let n_links = dev.fit(&tri, "values").unwrap().n_links;
    assert_eq!(n_links, vec![5, 5, 5, 5, 5, 4, 3, 2, 1]);
    // The medial average of two values is their simple average and of one
    // value the value itself.
    let dev = Development {
        n_periods: Some(5),
        drop_high: 1,
        drop_low: 1,
        ..Default::default()
    };
    assert_eq!(
        dev.fit(&tri, "values").unwrap().n_links,
        vec![3, 3, 3, 3, 3, 4 - 2, 3 - 2, 2, 1]
    );
}

#[test]
fn friedland_xyz_volume_latest_2_and_a_reported_pattern_below_1() {
    let volume2 = Development {
        n_periods: Some(2),
        ..Default::default()
    };
    // Chapter 7, Exhibit II, sheets 1 and 2, the selected factors.
    let reported = [
        1.687, 1.265, 1.102, 1.020, 1.050, 1.010, 1.011, 1.000, 0.993, 0.999,
    ];
    let paid = [
        3.349, 2.079, 1.574, 1.316, 1.203, 1.136, 1.059, 1.022, 1.017, 1.004,
    ];
    let r = xyz(false);
    let p = xyz(true);
    assert_eq!(
        three_decimals(&volume2.fit(&r, "values").unwrap().ldf),
        reported
    );
    assert_eq!(
        three_decimals(&volume2.fit(&p, "values").unwrap().ldf),
        paid
    );

    // Sheet 3, column (7): the reported pattern has 0.992 at 96 and 108
    // months and is a factor to ultimate (or, as the book prints it, a
    // percent developed above 1).
    let ages: Vec<Lag> = (1..=11).map(|k| 12 * k).collect();
    let cdf = [
        2.551, 1.512, 1.196, 1.085, 1.064, 1.013, 1.003, 0.992, 0.992, 0.999, 1.000,
    ];
    let values: Vec<(Lag, f64)> = ages.iter().copied().zip(cdf).collect();
    let book = [
        15822.0, 25082.0, 36948.0, 38487.0, 48313.0, 44950.0, 74787.0, 76661.0, 58370.0, 47979.0,
        47530.0,
    ];
    let fit_with = |kind: PatternKind, values: Vec<(Lag, f64)>| {
        ChainLadder {
            development: Development {
                pattern: Some(SelectedPattern::new(kind, values)),
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(&r, "values")
        .unwrap()
    };
    let as_cdf = fit_with(PatternKind::AgeToUltimate, values.clone());
    let percent: Vec<(Lag, f64)> = values.iter().map(|&(a, c)| (a, 1.0 / c)).collect();
    let as_percent = fit_with(PatternKind::PercentDeveloped, percent);
    for (i, want) in book.iter().enumerate() {
        // The book rounds its ultimates to whole numbers.
        assert!(
            (as_cdf.ultimate[i] - want).abs() <= 1.0,
            "{i}: {}",
            as_cdf.ultimate[i]
        );
        assert!((as_percent.ultimate[i] - as_cdf.ultimate[i]).abs() < 1e-6);
    }
    // The pattern's factors to ultimate are the ones printed.
    for (got, want) in as_cdf.cdf.iter().zip(cdf) {
        assert!((got - want).abs() < 1e-12, "{got} {want}");
    }
}

/// Werner and Modlin, *Basic Ratemaking*, Appendix A, Loss Development
/// exhibit: reported losses and paid ALAE, accident years 2009 to 2015 at
/// 15, 27, 39, 51 and 63 months.
fn werner_modlin_appendix_a() -> Triangle {
    let columns: [&[f64]; 5] = [
        &[
            705088.0, 712475.0, 714196.0, 764101.0, 774384.0, 785068.0, 797866.0,
        ],
        &[725592.0, 753295.0, 763913.0, 861114.0, 846167.0, 821509.0],
        &[738686.0, 782248.0, 855150.0, 884498.0, 835120.0],
        &[753027.0, 800258.0, 874106.0, 867184.0],
        &[732239.0, 813949.0, 856495.0],
    ];
    let (mut origin, mut age, mut value) = (Vec::new(), Vec::new(), Vec::new());
    for (a, column) in [15, 27, 39, 51, 63].into_iter().zip(columns) {
        for (i, &v) in column.iter().enumerate() {
            origin.push(Month::january(2009 + i as i32));
            age.push(a);
            value.push(v);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("values", &value)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn werner_modlin_straight_average_excluding_high_and_low() {
    let tri = werner_modlin_appendix_a();
    let dev = Development {
        average: Average::Simple,
        drop_high: 1,
        drop_low: 1,
        ..Default::default()
    };
    let fit = dev.fit(&tri, "values").unwrap();
    // The book rounds each link ratio to four decimals before averaging, so
    // exact link ratios can differ from its factors by up to 1e-4 (the
    // rounding of the ratios, then of the average); on its rounded link
    // ratios the same rule gives the factors printed.
    let book = [1.0665, 1.0279, 1.0208, 0.9799];
    for (got, want) in fit.ldf.iter().zip(book) {
        assert!((got - want).abs() < 1e-4, "{got} {want}");
    }
    assert_eq!(fit.n_links, vec![4, 3, 2, 1]);
    let rounded = |x: f64| (x * 1e4).round() / 1e4;
    let rows: Vec<Vec<f64>> = (0..4)
        .map(|j| {
            let col_from = [
                [705088.0, 712475.0, 714196.0, 764101.0, 774384.0, 785068.0].as_slice(),
                [725592.0, 753295.0, 763913.0, 861114.0, 846167.0].as_slice(),
                [738686.0, 782248.0, 855150.0, 884498.0].as_slice(),
                [753027.0, 800258.0, 874106.0].as_slice(),
            ][j];
            let col_to = [
                [725592.0, 753295.0, 763913.0, 861114.0, 846167.0, 821509.0].as_slice(),
                [738686.0, 782248.0, 855150.0, 884498.0, 835120.0].as_slice(),
                [753027.0, 800258.0, 874106.0, 867184.0].as_slice(),
                [732239.0, 813949.0, 856495.0].as_slice(),
            ][j];
            col_from
                .iter()
                .zip(col_to)
                .map(|(a, b)| rounded(b / a))
                .collect()
        })
        .collect();
    let on_rounded: Vec<f64> = rows
        .iter()
        .map(|c| {
            let max = c.iter().copied().fold(f64::MIN, f64::max);
            let min = c.iter().copied().fold(f64::MAX, f64::min);
            rounded((c.iter().sum::<f64>() - max - min) / (c.len() - 2) as f64)
        })
        .collect();
    assert_eq!(on_rounded, book);
}

#[test]
fn matches_r_mack_chain_ladder_with_weights() {
    let cases = reference("reserving_averages_r.csv");
    let mut fits = std::collections::HashMap::new();
    check(&cases, |case: &Case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let fit = fits
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                Mack {
                    development: development(method),
                    ..Default::default()
                }
                .fit(&triangle(dataset), "values")
                .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
            });
        let dev = &fit.chain_ladder.development;
        let k = || case.number("arg").map(|k| k as usize);
        match case.get("quantity") {
            "ldf" => dev.ldf.get(k()?).copied(),
            "sigma" => dev.sigma.get(k()?).copied(),
            "se" => {
                let year = case.number("arg")? as i32;
                let o = fit
                    .chain_ladder
                    .origins
                    .iter()
                    .position(|&p| p == Period::year(year))?;
                Some(fit.standard_error[o])
            }
            "total_standard_error" => Some(fit.total_standard_error),
            _ => None,
        }
    });
}
