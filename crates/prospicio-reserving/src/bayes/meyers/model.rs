//! The Meyers log posterior: the observed `Cell`, the `Model` density and its `Posterior` impl.

use super::*;

/// An observed cell: origin and development positions, `log C_wd` and
/// `log P_w`, and (CAY only, after the first origin) the index of the
/// previous origin's cell at the same age, which comes earlier.
#[derive(Debug, Clone, Copy)]
pub(super) struct Cell {
    pub(super) w: usize,
    pub(super) d: usize,
    pub(super) y: f64,
    pub(super) offset: f64,
    pub(super) prev: Option<usize>,
}

/// Smallest step `s_i` of `sigma_d^2`, `1 - exp(-1e-5)`: Meyers bounds
/// `a_ig` above by `1e5` in his Stan code, and `s_i = 1 - exp(-1 / a_ig)`.
pub(crate) const S_FLOOR: f64 = 9.999_950_000_166_667e-6;

/// The log posterior on the unconstrained
/// `theta = (logelr, alpha_2..alpha_n, beta_1..beta_(m-1), [gamma | v], u_1..u_m)`
/// with `s_i = S_FLOOR + (1 - S_FLOOR) logistic(u_i)` and (CAY)
/// `rho = tanh(v)`.
#[derive(Debug, Clone)]
pub(super) struct Model {
    pub(super) n_origins: usize,
    pub(super) n_dev: usize,
    pub(super) kind: MeyersModel,
    pub(super) cells: Vec<Cell>,
    pub(super) logelr_mean: f64,
    pub(super) logelr_sd: f64,
    pub(super) origin_sd: f64,
    pub(super) development_sd: f64,
    pub(super) gamma_sd: f64,
    pub(super) rho_shape: f64,
    /// Without priors, the likelihood alone (for the tests).
    pub(super) priors: bool,
}

/// `log(1 + exp(x))` without overflow.
pub(super) fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `1 / (1 + exp(-u))` without overflow.
pub(super) fn logistic(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    }
}

impl Model {
    fn csr(&self) -> bool {
        self.kind == MeyersModel::Csr
    }

    pub(super) fn cay(&self) -> bool {
        self.kind == MeyersModel::Cay
    }

    /// Position of `beta_1` in `theta`.
    pub(super) fn beta0(&self) -> usize {
        self.n_origins
    }

    /// Position of `gamma` (CSR), of `v` or `rho` (CAY) or of `u_1` (CRC).
    pub(super) fn after_beta(&self) -> usize {
        self.n_origins + self.n_dev - 1
    }

    /// Position of `u_1`.
    pub(super) fn u0(&self) -> usize {
        self.after_beta() + usize::from(self.kind != MeyersModel::Crc)
    }

    /// The names of the unconstrained parameters: `v` is `atanh(rho)` and
    /// `u[d]` the logit of `sigma[d]^2`'s step.
    pub(super) fn sampled_names(&self) -> Vec<String> {
        let mut names = self.names();
        if self.cay() {
            names[self.after_beta()] = "v".into();
        }
        for (d, name) in names[self.u0()..].iter_mut().enumerate() {
            *name = format!("u[{}]", d + 1);
        }
        names
    }

    pub(super) fn names(&self) -> Vec<String> {
        let mut names = vec!["logelr".to_string()];
        names.extend((2..=self.n_origins).map(|w| format!("alpha[{w}]")));
        names.extend((1..self.n_dev).map(|d| format!("beta[{d}]")));
        match self.kind {
            MeyersModel::Crc => {}
            MeyersModel::Csr => names.push("gamma".into()),
            MeyersModel::Cay => names.push("rho".into()),
        }
        names.extend((1..=self.n_dev).map(|d| format!("sigma[{d}]")));
        names
    }

    /// `sigma_d^2` for each age from `theta`'s `u`, and each
    /// `l_i = logistic(u_i)` and `1 - l_i`, with
    /// `s_i = S_FLOOR + (1 - S_FLOOR) l_i`.
    pub(super) fn variances(&self, theta: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = self.n_dev;
        let u = &theta[self.u0()..self.u0() + m];
        let l: Vec<f64> = u.iter().map(|&u| logistic(u)).collect();
        let lc: Vec<f64> = u.iter().map(|&u| logistic(-u)).collect();
        let mut var = vec![0.0; m];
        let mut acc = 0.0;
        for d in (0..m).rev() {
            acc += S_FLOOR + (1.0 - S_FLOOR) * l[d];
            var[d] = acc;
        }
        (var, l, lc)
    }

    /// `theta` on the natural scale: `logelr`, `alpha`, `beta`, `gamma`
    /// or `rho = tanh(v)`, and `sigma_d = sqrt(sigma_d^2)`.
    pub(super) fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let (var, _, _) = self.variances(theta);
        let mut out = theta[..self.u0()].to_vec();
        if self.cay() {
            let k = self.after_beta();
            out[k] = out[k].tanh();
        }
        out.extend(var.iter().map(|v| v.sqrt()));
        out
    }

    /// `g_w` and its derivative in `gamma`, for 0-based origin `w`.
    fn speedup(&self, gamma: f64, w: usize) -> (f64, f64) {
        if !self.csr() || w == 0 {
            return (1.0, 0.0);
        }
        let k = w as i32;
        let base = 1.0 - gamma;
        (base.powi(k), -(w as f64) * base.powi(k - 1))
    }

    /// `mu_wd` without CAY's `rho` term, from natural-scale or
    /// unconstrained `theta` (both hold `logelr`, `alpha`, `beta` and
    /// `gamma` at the same positions).
    pub(super) fn mu(&self, theta: &[f64], cell: &Cell) -> f64 {
        let alpha = if cell.w > 0 { theta[cell.w] } else { 0.0 };
        let beta = if cell.d + 1 < self.n_dev {
            theta[self.beta0() + cell.d]
        } else {
            0.0
        };
        let gamma = if self.csr() {
            theta[self.after_beta()]
        } else {
            0.0
        };
        cell.offset + theta[0] + alpha + beta * self.speedup(gamma, cell.w).0
    }

    /// The residual `log C_wd - mu_wd` of each cell, in `residuals`, with
    /// CAY's `rho` (0 otherwise): each cell's mean takes `rho` times the
    /// previous origin's residual, which comes earlier.
    fn residuals(&self, theta: &[f64], rho: f64, residuals: &mut Vec<f64>) {
        residuals.clear();
        for c in &self.cells {
            let mut r = c.y - self.mu(theta, c);
            if let Some(p) = c.prev {
                r -= rho * residuals[p];
            }
            residuals.push(r);
        }
    }

    /// Meyers' pointwise log likelihood, `normal_lpdf(log C | mu, sigma_d)`,
    /// of every cell from a natural-scale draw, appended to `out`
    /// (`residuals` is scratch).
    pub(super) fn log_densities(&self, nat: &[f64], residuals: &mut Vec<f64>, out: &mut Vec<f64>) {
        let rho = if self.cay() {
            nat[self.after_beta()]
        } else {
            0.0
        };
        self.residuals(nat, rho, residuals);
        for (c, r) in self.cells.iter().zip(residuals.iter()) {
            let sigma = nat[self.u0() + c.d];
            let z = r / sigma;
            out.push(-0.5 * (std::f64::consts::TAU.ln() + z * z) - sigma.ln());
        }
    }

    /// One simulation of the ultimate (the value at age `m`) of each origin
    /// into `row`, given a natural-scale draw: a closed origin keeps its
    /// latest value, any other is log-normal with mean
    /// `log P_w + logelr + alpha_w` (`beta_m = 0`, so neither beta nor
    /// gamma enters) and sd `sigma_m`. In CAY each mean also takes `rho`
    /// times the previous origin's residual at age `m`, observed for a
    /// closed origin and drawn otherwise, origin by origin (Meyers 2019,
    /// sec. 8). Returns the newest origin's residual at age `m` in CAY,
    /// which the next origin's mean takes `rho` times of
    /// ([`MeyersFit::new_year`]), and 0 in CRC and CSR.
    pub(super) fn draw_ultimates(
        &self,
        nat: &[f64],
        premiums: &[f64],
        closed: &[bool],
        latest: &[f64],
        rng: &mut StreamRng,
        row: &mut [f64],
    ) -> f64 {
        // The natural-scale draw ends with sigma[m].
        let sigma_m = nat[self.dim() - 1];
        if !self.cay() {
            for (w, ultimate) in row.iter_mut().enumerate() {
                *ultimate = if closed[w] {
                    latest[w]
                } else {
                    let alpha = if w > 0 { nat[w] } else { 0.0 };
                    let z = norm_quantile(rng.next_open01());
                    (premiums[w].ln() + nat[0] + alpha + sigma_m * z).exp()
                };
            }
            return 0.0;
        }
        let rho = nat[self.after_beta()];
        let mut residual = 0.0;
        for (w, ultimate) in row.iter_mut().enumerate() {
            let alpha = if w > 0 { nat[w] } else { 0.0 };
            let mean = premiums[w].ln() + nat[0] + alpha + rho * residual;
            if closed[w] {
                *ultimate = latest[w];
                residual = latest[w].ln() - mean;
            } else {
                residual = sigma_m * norm_quantile(rng.next_open01());
                *ultimate = (mean + residual).exp();
            }
        }
        residual
    }

    /// `mu_wd` without CAY's `rho` term at any origin and age, from a
    /// natural-scale draw and each origin's premium.
    pub(super) fn mu_at(&self, nat: &[f64], premiums: &[f64], w: usize, d: usize) -> f64 {
        let cell = Cell {
            w,
            d,
            y: 0.0,
            offset: premiums[w].ln(),
            prev: None,
        };
        self.mu(nat, &cell)
    }

    /// The development pattern of a natural-scale draw into `shares`, one
    /// per age: `exp(beta_d g)`, the median value at age `d` over the
    /// median at the oldest age, where the share is exactly 1 as
    /// `beta_m = 0`, of an origin with nothing yet observed, the one after
    /// the triangle's newest (`reserving-v02.md` decision 28 item 7). `g` is
    /// that origin's speed-up `(1 - gamma)^n` in CSR (0-based origin `n`,
    /// the number of origins) and 1 in CRC and CAY; CAY's `rho` correlates
    /// origins and does not enter.
    pub(super) fn pattern(&self, nat: &[f64], shares: &mut [f64]) {
        let gamma = if self.csr() {
            nat[self.after_beta()]
        } else {
            0.0
        };
        let g = self.speedup(gamma, self.n_origins).0;
        let m = self.n_dev;
        for (d, share) in shares.iter_mut().enumerate() {
            *share = if d + 1 < m {
                (nat[self.beta0() + d] * g).exp()
            } else {
                1.0
            };
        }
    }

    /// CAY's residuals of the observed cells from a natural-scale draw, in
    /// `residuals`; empty for CRC and CSR, which do not use them.
    pub(super) fn cay_residuals(&self, nat: &[f64], residuals: &mut Vec<f64>) {
        if self.cay() {
            self.residuals(nat, nat[self.after_beta()], residuals);
        } else {
            residuals.clear();
        }
    }

    /// The mean and variance of `log C_wd` for a cell that is not observed,
    /// given a natural-scale draw: `mu_wd` and `sigma_d^2`; in CAY, plus
    /// `rho` times the previous origin's residual at age `d` when that cell
    /// is observed (`at`, with `residuals` from
    /// [`cay_residuals`](Self::cay_residuals)), and with that residual,
    /// `N(0, sigma_d^2)`, integrated out when it is not, which makes the
    /// variance `(1 + rho^2) sigma_d^2`.
    pub(super) fn log_moments(
        &self,
        nat: &[f64],
        residuals: &[f64],
        at: &[Vec<Option<usize>>],
        premiums: &[f64],
        w: usize,
        d: usize,
    ) -> (f64, f64) {
        let sigma = nat[self.u0() + d];
        let mut m = self.mu_at(nat, premiums, w, d);
        let mut v = sigma * sigma;
        if self.cay() && w > 0 {
            let rho = nat[self.after_beta()];
            match at[w - 1][d] {
                Some(i) => m += rho * residuals[i],
                None => v *= 1.0 + rho * rho,
            }
        }
        (m, v)
    }

    /// One simulation of the test increments of `rebuilt` into `row`, given
    /// a natural-scale draw: each origin's cumulative values along its path
    /// (log-normal, independent given the draw in CRC and CSR), and each
    /// test row's increment the difference of consecutive values on it, the
    /// first from the latest train value. In CAY the ages of the paths run
    /// in order and, within an age, the origins: each mean takes `rho`
    /// times the previous origin's residual at that age, observed for a
    /// train cell and drawn otherwise (also for a cell on no path, which
    /// integrates it out). `at` indexes the model's cells and `ages` are
    /// [`Rebuilt::path_ages`].
    pub(super) fn draw_paths(
        &self,
        nat: &[f64],
        rebuilt: &Rebuilt,
        at: &[Vec<Option<usize>>],
        ages: &[usize],
        rng: &mut StreamRng,
        row: &mut [f64],
    ) {
        let cay = self.cay();
        let rho = if cay { nat[self.after_beta()] } else { 0.0 };
        let mut residuals = Vec::new();
        self.cay_residuals(nat, &mut residuals);
        let no = rebuilt.origins.len();
        // Origin w's value at age d is path[w][d - latest[w] - 1].
        let mut path: Vec<Vec<f64>> = (0..no)
            .map(|w| vec![0.0; rebuilt.path_end[w].map_or(0, |e| e - rebuilt.latest[w])])
            .collect();
        for &d in ages {
            let sigma = nat[self.u0() + d];
            // The previous origin's residual at age d (0 before the first).
            let mut previous = 0.0;
            for (w, values) in path.iter_mut().enumerate() {
                let on_path = rebuilt.in_path(w, d);
                previous = match at[w][d] {
                    Some(i) if cay => residuals[i],
                    Some(_) => 0.0,
                    None if on_path || cay => {
                        let e = sigma * norm_quantile(rng.next_open01());
                        if on_path {
                            let mu = self.mu_at(nat, &rebuilt.premiums, w, d);
                            values[d - rebuilt.latest[w] - 1] = (mu + rho * previous + e).exp();
                        }
                        e
                    }
                    None => 0.0,
                };
            }
        }
        for (x, &(w, d)) in row.iter_mut().zip(&rebuilt.test) {
            let k = d - rebuilt.latest[w] - 1;
            let before = if k == 0 {
                rebuilt.last_value[w]
            } else {
                path[w][k - 1]
            };
            *x = path[w][k] - before;
        }
    }

    /// The Newton start: least squares of `log C - log P` on `logelr`,
    /// `alpha` and `beta` (CRC, `gamma = 0`), then each age's mean squared
    /// residual made non-increasing from the oldest age, its steps as
    /// `s_i` within `[0.001, 0.5]`.
    pub(super) fn start(&self) -> Vec<f64> {
        let k = self.after_beta();
        let (n, m) = (self.n_origins, self.n_dev);
        let columns = |c: &Cell| {
            let mut x = vec![0];
            if c.w > 0 {
                x.push(c.w);
            }
            if c.d + 1 < m {
                x.push(n + c.d);
            }
            x
        };
        let mut xtx = vec![0.0; k * k];
        let mut xty = vec![0.0; k];
        for c in &self.cells {
            let x = columns(c);
            for &i in &x {
                xty[i] += c.y - c.offset;
                for &j in &x {
                    xtx[i * k + j] += 1.0;
                }
            }
        }
        for j in 0..k {
            xtx[j * k + j] += 1e-8;
        }
        let mut theta = vec![0.0; self.dim()];
        if let Some(l) = cholesky(&xtx, k) {
            theta[..k].copy_from_slice(&cholesky_solve(&l, &xty));
        } else {
            theta[0] = self.logelr_mean;
        }
        let mut sum = vec![0.0; m];
        let mut count = vec![0.0; m];
        for c in &self.cells {
            let r = c.y - self.mu(&theta, c);
            sum[c.d] += r * r;
            count[c.d] += 1.0;
        }
        let mut var: Vec<f64> = sum
            .iter()
            .zip(&count)
            .map(|(s, n)| if *n > 0.0 { s / n } else { 0.0 })
            .collect();
        for d in (0..m.saturating_sub(1)).rev() {
            var[d] = var[d].max(var[d + 1]);
        }
        let u0 = self.u0();
        for d in 0..m {
            let next = if d + 1 < m { var[d + 1] } else { 0.0 };
            let s = (var[d] - next).clamp(1e-3, 0.5);
            let l = (s - S_FLOOR) / (1.0 - S_FLOOR);
            theta[u0 + d] = (l / (1.0 - l)).ln();
        }
        theta
    }
}

impl Posterior for Model {
    fn dim(&self) -> usize {
        self.u0() + self.n_dev
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        grad.fill(0.0);
        let (n, m) = (self.n_origins, self.n_dev);
        let mut f = 0.0;
        if self.priors {
            let mut prior = |j: usize, mean: f64, sd: f64| {
                let r = theta[j] - mean;
                f -= 0.5 * r * r / (sd * sd);
                grad[j] -= r / (sd * sd);
            };
            prior(0, self.logelr_mean, self.logelr_sd);
            for j in 1..n {
                prior(j, 0.0, self.origin_sd);
            }
            for j in n..n + m - 1 {
                prior(j, 0.0, self.development_sd);
            }
            if self.csr() {
                prior(self.after_beta(), 0.0, self.gamma_sd);
            }
            if self.cay() {
                // (1 - rho) / 2 = logistic(-2 v) ~ Beta(a, a), with its
                // Jacobian: a (log logistic(2 v) + log logistic(-2 v)).
                let (k, a) = (self.after_beta(), self.rho_shape);
                let v = theta[k];
                f -= a * (softplus(2.0 * v) + softplus(-2.0 * v));
                grad[k] -= 2.0 * a * v.tanh();
            }
        }
        let (var, s, sc) = self.variances(theta);
        let gamma = if self.csr() {
            theta[self.after_beta()]
        } else {
            0.0
        };
        let rho = if self.cay() {
            theta[self.after_beta()].tanh()
        } else {
            0.0
        };
        let mut r = Vec::with_capacity(self.cells.len());
        self.residuals(theta, rho, &mut r);
        // CAY: h_i = d f / d mu_i at fixed rho, through every later
        // residual of its age's column: h_i = r_i / v_i - rho h_(next i),
        // collected backwards.
        let mut h = Vec::new();
        if self.cay() {
            h = vec![0.0; self.cells.len()];
            for (i, c) in self.cells.iter().enumerate().rev() {
                h[i] += r[i] / var[c.d];
                if let Some(p) = c.prev {
                    h[p] -= rho * h[i];
                }
            }
        }
        let mut drho = 0.0;
        // d f / d sigma_d^2.
        let mut dvar = vec![0.0; m];
        for (i, c) in self.cells.iter().enumerate() {
            let v = var[c.d];
            let r_i = r[i];
            f -= 0.5 * (v.ln() + r_i * r_i / v);
            dvar[c.d] += 0.5 * (r_i * r_i / v - 1.0) / v;
            let e = if self.cay() { h[i] } else { r_i / v };
            if let Some(p) = c.prev {
                drho += e * r[p];
            }
            grad[0] += e;
            if c.w > 0 {
                grad[c.w] += e;
            }
            if c.d + 1 < m {
                let (g, dg) = self.speedup(gamma, c.w);
                let beta = theta[self.beta0() + c.d];
                grad[self.beta0() + c.d] += e * g;
                if self.csr() {
                    grad[self.after_beta()] += e * beta * dg;
                }
            }
        }
        if self.cay() {
            grad[self.after_beta()] += drho * (1.0 - rho * rho);
        }
        // sigma_d^2 = sum over i >= d of s_i, so d f / d s_i sums dvar
        // over d <= i, and ds_i / du_i = (1 - S_FLOOR) l_i (1 - l_i); the
        // U(S_FLOOR, 1) prior on s_i is l_i (1 - l_i) in u_i.
        let u0 = self.u0();
        let mut acc = 0.0;
        for i in 0..m {
            acc += dvar[i];
            let u = theta[u0 + i];
            f -= softplus(-u) + softplus(u);
            grad[u0 + i] = (1.0 - S_FLOOR) * s[i] * sc[i] * acc + (sc[i] - s[i]);
        }
        if f.is_finite() { f } else { f64::NAN }
    }
}

/// The log posterior written once over [`Scalar`](crate::bayes::scalar::Scalar),
/// the oracle of the compartmental note's decision 4: test-only, beside
/// the hand-written [`Posterior::log_posterior`], which stays the fast
/// path. It computes each cell's residual forward, as the value does, and
/// lets the dual carry CAY's chain of residuals.
#[cfg(test)]
impl crate::bayes::autodiff::GenericDensity for Model {
    fn n_params(&self) -> usize {
        self.dim()
    }

    fn log_density<S: crate::bayes::scalar::Scalar>(&self, theta: &[S]) -> S {
        use crate::bayes::scalar::{logistic, softplus};
        let (n, m) = (self.n_origins, self.n_dev);
        let zero = S::from_f64(0.0);
        let mut f = zero;
        if self.priors {
            let prior = |x: S, mean: f64, sd: f64| {
                let r = x - mean;
                r * r * (0.5 / (sd * sd))
            };
            f -= prior(theta[0], self.logelr_mean, self.logelr_sd);
            for &a in &theta[1..n] {
                f -= prior(a, 0.0, self.origin_sd);
            }
            for &b in &theta[n..n + m - 1] {
                f -= prior(b, 0.0, self.development_sd);
            }
            if self.csr() {
                f -= prior(theta[self.after_beta()], 0.0, self.gamma_sd);
            }
            if self.cay() {
                let v = theta[self.after_beta()] * 2.0;
                f -= (softplus(v) + softplus(-v)) * self.rho_shape;
            }
        }
        let u0 = self.u0();
        let mut var = vec![zero; m];
        let mut acc = zero;
        for d in (0..m).rev() {
            acc += logistic(theta[u0 + d]) * (1.0 - S_FLOOR) + S_FLOOR;
            var[d] = acc;
        }
        for &u in &theta[u0..u0 + m] {
            f -= softplus(-u) + softplus(u);
        }
        let k = self.after_beta();
        let rho = if self.cay() { theta[k].tanh() } else { zero };
        let mut r: Vec<S> = Vec::with_capacity(self.cells.len());
        for c in &self.cells {
            let mut mu = theta[0] + c.offset;
            if c.w > 0 {
                mu += theta[c.w];
            }
            if c.d + 1 < m {
                let beta = theta[self.beta0() + c.d];
                mu += if self.csr() {
                    beta * (-theta[k] + 1.0).powi(c.w as i32)
                } else {
                    beta
                };
            }
            let mut res = -mu + c.y;
            if let Some(p) = c.prev {
                res -= rho * r[p];
            }
            r.push(res);
            let v = var[c.d];
            f -= (v.ln() + res * res / v) * 0.5;
        }
        f
    }
}
