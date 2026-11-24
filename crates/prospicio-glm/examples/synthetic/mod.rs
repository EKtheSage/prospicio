//! Seeded synthetic portfolios shared by the GLM benchmarks (#969).

/// A small deterministic generator (SplitMix64), so the benchmarks need no
/// dependency and give the same data on every platform.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on `(0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Poisson by inversion (fine for the small means used here).
    pub fn poisson(&mut self, mean: f64) -> f64 {
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

/// A frequency portfolio shaped like freMTPL2: an intercept, `numeric`
/// continuous rating factors and one-hot dummies for the rest of the `p`
/// columns (each row sets one dummy per factor of 6 levels), a log
/// exposure offset and Poisson claim counts.
pub struct Portfolio {
    pub names: Vec<String>,
    pub columns: Vec<Vec<f64>>,
    pub offset: Vec<f64>,
    pub y: Vec<f64>,
}

pub fn portfolio(n: usize, p: usize, numeric: usize, seed: u64) -> Portfolio {
    assert!(p > numeric);
    let mut rng = Rng::new(seed);
    let mut columns = vec![vec![0.0; n]; p];
    columns[0].iter_mut().for_each(|v| *v = 1.0);
    let dummies = p - 1 - numeric;
    let levels = 5; // dummies per factor (the 6th level is the base)
    let mut offset = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut eta = -2.5;
        for (c, col) in columns.iter_mut().enumerate().take(numeric + 1).skip(1) {
            let x = rng.uniform() * 2.0 - 1.0;
            col[i] = x;
            eta += 0.2 * x / c as f64;
        }
        let mut first = 0;
        while first < dummies {
            let width = levels.min(dummies - first);
            let level = (rng.next_u64() % (width as u64 + 1)) as usize;
            if level < width {
                let c = 1 + numeric + first + level;
                columns[c][i] = 1.0;
                eta += 0.05 * ((c % 7) as f64 - 3.0);
            }
            first += width;
        }
        let exposure = 0.1 + 0.9 * rng.uniform();
        offset[i] = exposure.ln();
        y[i] = rng.poisson((eta + offset[i]).exp());
    }
    let names = (0..p).map(|j| format!("x{j}")).collect();
    Portfolio {
        names,
        columns,
        offset,
        y,
    }
}

/// Median wall time in milliseconds of `runs` calls of `f`, after one
/// warm-up call.
pub fn time_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}
