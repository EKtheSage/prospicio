//! Benchmark of the Bayesian GLM's NUTS sampling (#969): every leapfrog
//! step evaluates the log posterior and its gradient over all rows.
//!
//! `cargo run --release -p prospicio-bayes --example bayes_glm_bench`
//! prints the median wall time.
//! Set `BENCH_QUICK=1` for one timed run.

use prospicio_bayes::glm::BayesGlm;
use prospicio_bayes::nuts::Sampler;
use prospicio_models::{Design, Family, Link, Model};
use std::hint::black_box;

/// SplitMix64, so the benchmark needs no dependency.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn poisson(&mut self, mean: f64) -> f64 {
        let u = self.uniform();
        let (mut k, mut p) = (0.0, (-mean).exp());
        let mut cdf = p;
        while u > cdf && k < 1000.0 {
            k += 1.0;
            p *= mean / k;
            cdf += p;
        }
        k
    }
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("prospicio-bayes benchmarks (#969), {threads} hardware threads");
    let quick = std::env::var_os("BENCH_QUICK").is_some();

    // 20,000 rows, 20 columns (an intercept and 19 continuous factors).
    let (n, p) = (20_000, 20);
    let mut rng = Rng(969);
    let mut columns = vec![vec![1.0; n]; p];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut eta = -1.0;
        for (c, col) in columns.iter_mut().enumerate().skip(1) {
            col[i] = rng.uniform() * 2.0 - 1.0;
            eta += 0.1 * col[i] / c as f64;
        }
        y[i] = rng.poisson(eta.exp());
    }
    let names = (0..p).map(|j| format!("x{j}")).collect();
    let design = Design::new(names, columns).unwrap();
    let model = BayesGlm::new(Family::Poisson, Link::Log).sampler(Sampler {
        chains: 4,
        tune: 300,
        draws: 300,
        ..Sampler::default()
    });
    model.fit(&design, &y).unwrap();
    let mut times: Vec<f64> = (0..if quick { 1 } else { 3 })
        .map(|_| {
            let t = std::time::Instant::now();
            black_box(model.fit(&design, &y).unwrap());
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    println!(
        "{:<40} {:>12.3} ms",
        "bayes_glm_nuts_poisson_n20000_p20_4x600",
        times[times.len() / 2]
    );
}
