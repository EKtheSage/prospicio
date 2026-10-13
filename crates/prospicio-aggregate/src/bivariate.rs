//! Bivariate compound distributions: two lines driven by one claim count,
//! and clash.
//!
//! Each of `N` shared events gives a pair of losses `(X_k, Y_k)`, drawn from
//! a joint per-event severity; the annual losses are `A = X_1 + … + X_N` and
//! `B = Y_1 + … + Y_N`. Because the pgf of `N` acts elementwise on a
//! transform, the joint law of `(A, B)` is the one-dimensional compound FFT
//! with the transforms made two-dimensional:
//! `density = iFFT2(P_N(FFT2(S)))`, where `S` is the per-event severity on a
//! grid with a step on each axis. A per-event severity may put mass on 0 on
//! either axis, so an event can hit one line, the other, or both.
//!
//! [`Clash`] is the usual model of a catastrophe hitting two lines (Mildenhall's
//! `aggregate`, `clash`): within each event line A is hit with probability
//! `p_A` and line B with `p_B`, independently, so the dependence between the
//! lines comes only from the shared count. It is parametrised by expected
//! event counts: A only, B only and both (the clash count). See
//! `docs/design/aggregate.md`.

use prospicio_core::{Error, Result};
use prospicio_prob::copula::Copula;
use prospicio_prob::{Counting, Grid};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// A joint distribution of two losses on a grid: `P(A = i h_A, B = j h_B)`,
/// with a step `h_A` on axis 0 (rows) and `h_B` on axis 1 (columns).
///
/// Probabilities are stored row by row. They sum to 1, so a grid that
/// stops short of the support carries the mass beyond its last row on the
/// last row, and beyond its last column on the last column, as a one-
/// dimensional [`Grid`] carries its tail on the last point.
#[derive(Debug, Clone, PartialEq)]
pub struct JointGrid {
    steps: [f64; 2],
    shape: [usize; 2],
    probs: Vec<f64>,
}

impl JointGrid {
    /// A joint grid with steps `steps`, `shape[0]` rows and `shape[1]`
    /// columns, and probabilities `probs` row by row.
    ///
    /// Fails if a step is not finite and positive, the shape is empty or does
    /// not match `probs`, a probability is negative or not finite, or they do
    /// not sum to 1 within `1e-9`.
    pub fn new(steps: [f64; 2], shape: [usize; 2], probs: Vec<f64>) -> Result<Self> {
        for &step in &steps {
            if !step.is_finite() || step <= 0.0 {
                return Err(Error::InvalidParameter {
                    name: "steps",
                    value: step,
                    reason: "must be finite and positive",
                });
            }
        }
        if shape[0] == 0 || shape[1] == 0 {
            return Err(Error::InvalidParameter {
                name: "shape",
                value: 0.0,
                reason: "must have at least one row and one column",
            });
        }
        if probs.len() != shape[0] * shape[1] {
            return Err(Error::InvalidParameter {
                name: "probs",
                value: probs.len() as f64,
                reason: "must hold rows × columns probabilities",
            });
        }
        if let Some(&bad) = probs.iter().find(|p| !p.is_finite() || **p < 0.0) {
            return Err(Error::InvalidParameter {
                name: "probs",
                value: bad,
                reason: "must all be finite and non-negative",
            });
        }
        let total: f64 = probs.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(Error::InvalidParameter {
                name: "probs",
                value: total,
                reason: "must sum to 1",
            });
        }
        Ok(Self {
            steps,
            shape,
            probs,
        })
    }

    /// The joint law of two independent losses: `P(i, j) = a_i b_j`.
    ///
    /// ```
    /// use prospicio_aggregate::bivariate::JointGrid;
    /// use prospicio_prob::Grid;
    ///
    /// let a = Grid::new(1.0, vec![0.5, 0.5]).unwrap();
    /// let b = Grid::new(2.0, vec![0.25, 0.75]).unwrap();
    /// let joint = JointGrid::independent(&a, &b);
    /// assert_eq!(joint.prob(1, 1), 0.375);
    /// assert!(joint.correlation().abs() < 1e-15);
    /// ```
    pub fn independent(a: &Grid, b: &Grid) -> Self {
        let probs = a
            .probs()
            .iter()
            .flat_map(|&p| b.probs().iter().map(move |&q| p * q))
            .collect();
        Self {
            steps: [a.step(), b.step()],
            shape: [a.len(), b.len()],
            probs,
        }
    }

    /// The joint per-event severity of two losses with per-event marginals
    /// `a` (rows) and `b` (columns), coupled by a two-dimensional `copula`:
    /// the copula's rectangle probabilities over the marginals' cdfs,
    /// `S[i, j] = C(F_A(i), F_B(j)) − C(F_A(i − 1), F_B(j))
    /// − C(F_A(i), F_B(j − 1)) + C(F_A(i − 1), F_B(j − 1))`, as `aggregate`'s
    /// copula `bivariate`. The marginals come back exactly, for any copula,
    /// since `C(u, 1) = u`; mass at 0 on an axis (see [`triggered`]) is an
    /// event that misses that line. A difference of nearly equal cdf values
    /// can come out a little below 0 (round-off, to about `1e-15`); it is
    /// kept, as `aggregate` keeps it, because setting it to 0 adds mass to
    /// every row and column and moves the second moments by `1e-9`
    /// relative. The compound's [`fft_bivariate`] sets its result's
    /// round-off to 0.
    ///
    /// Fails unless the copula is two-dimensional and has a distribution
    /// function ([`Copula::cdf`]).
    ///
    /// ```
    /// use prospicio_aggregate::bivariate::JointGrid;
    /// use prospicio_prob::Grid;
    /// use prospicio_prob::copula::{Archimedean, ArchimedeanCopula};
    ///
    /// let a = Grid::new(1.0, vec![0.25, 0.5, 0.25]).unwrap();
    /// let b = Grid::new(1.0, vec![0.5, 0.5]).unwrap();
    /// let gumbel = ArchimedeanCopula::new(Archimedean::Gumbel, 2.0, 2).unwrap();
    /// let joint = JointGrid::from_copula(&a, &b, &gumbel).unwrap();
    /// let m = joint.marginal(0).unwrap();
    /// assert!((m.probs()[1] - 0.5).abs() < 1e-15);
    /// assert!(joint.correlation() > 0.0);
    /// ```
    pub fn from_copula(a: &Grid, b: &Grid, copula: &dyn Copula) -> Result<Self> {
        if copula.dim() != 2 {
            return Err(Error::InvalidParameter {
                name: "copula",
                value: copula.dim() as f64,
                reason: "must be two-dimensional",
            });
        }
        let cumulative = |g: &Grid| -> Vec<f64> {
            let mut total = 0.0;
            std::iter::once(0.0)
                .chain(g.probs().iter().map(|p| {
                    total += p;
                    total.min(1.0)
                }))
                .collect()
        };
        let (fa, fb) = (cumulative(a), cumulative(b));
        // The last point of each axis reaches 1 exactly, so C(u, 1) = u
        // keeps the marginals.
        let (na, nb) = (fa.len(), fb.len());
        let mut c = vec![0.0; na * nb];
        for i in 1..na {
            let u = if i == na - 1 { 1.0 } else { fa[i] };
            for j in 1..nb {
                let v = if j == nb - 1 { 1.0 } else { fb[j] };
                c[i * nb + j] = copula.cdf(&[u, v])?;
            }
        }
        let mut probs = Vec::with_capacity((na - 1) * (nb - 1));
        for i in 1..na {
            for j in 1..nb {
                let mass = c[i * nb + j] - c[(i - 1) * nb + j] - c[i * nb + j - 1]
                    + c[(i - 1) * nb + j - 1];
                if mass < -1e-12 {
                    return Err(Error::Data(format!(
                        "the copula's rectangle probability at ({}, {}) is {mass}",
                        i - 1,
                        j - 1
                    )));
                }
                probs.push(mass);
            }
        }
        // Only round-off is below 0, so the checks of `new` apply to the
        // grid with it set to 0; the signed values are kept.
        let checked: Vec<f64> = probs.iter().map(|p| p.max(0.0)).collect();
        Self::new([a.step(), b.step()], [na - 1, nb - 1], checked)?;
        Ok(Self {
            steps: [a.step(), b.step()],
            shape: [na - 1, nb - 1],
            probs,
        })
    }

    /// The step on each axis.
    pub fn steps(&self) -> [f64; 2] {
        self.steps
    }

    /// The number of rows (axis 0) and columns (axis 1).
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// The probabilities, row by row.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// `P(A = i h_A, B = j h_B)`. Panics if `(i, j)` is outside the grid.
    pub fn prob(&self, i: usize, j: usize) -> f64 {
        assert!(
            i < self.shape[0] && j < self.shape[1],
            "cell outside the grid"
        );
        self.probs[i * self.shape[1] + j]
    }

    /// The marginal distribution of axis `axis` (0 or 1) as a grid.
    pub fn marginal(&self, axis: usize) -> Result<Grid> {
        let [rows, cols] = self.shape;
        let probs = match axis {
            0 => self.probs.chunks(cols).map(|r| r.iter().sum()).collect(),
            1 => {
                let mut out = vec![0.0; cols];
                for row in self.probs.chunks(cols) {
                    for (o, p) in out.iter_mut().zip(row) {
                        *o += p;
                    }
                }
                out
            }
            _ => return Err(bad_axis(axis)),
        };
        debug_assert_eq!(rows, self.shape[0]);
        Grid::new(self.steps[axis], probs)
    }

    /// The distribution of the total `A + B`.
    ///
    /// When the axes share a step, the total is exact on that step (the sum
    /// over each anti-diagonal). Otherwise each cell's mass is split between
    /// the two points of the larger step around its total, in proportion to
    /// how near each is, which keeps the mean (as `aggregate`'s
    /// `BivariateAggregate.total`).
    pub fn total(&self) -> Result<Grid> {
        let [rows, cols] = self.shape;
        let [ha, hb] = self.steps;
        if ha == hb {
            let mut out = vec![0.0; rows + cols - 1];
            for (i, row) in self.probs.chunks(cols).enumerate() {
                for (o, p) in out[i..i + cols].iter_mut().zip(row) {
                    *o += p;
                }
            }
            return Grid::new(ha, out);
        }
        let h = ha.max(hb);
        let top = (rows - 1) as f64 * ha + (cols - 1) as f64 * hb;
        let n = (top / h).ceil() as usize + 1;
        let mut out = vec![0.0; n];
        for (i, row) in self.probs.chunks(cols).enumerate() {
            for (j, &p) in row.iter().enumerate() {
                let u = (i as f64 * ha + j as f64 * hb) / h;
                let k = (u.floor() as usize).min(n - 1);
                let f = (u - k as f64).clamp(0.0, 1.0);
                out[k] += p * (1.0 - f);
                out[(k + 1).min(n - 1)] += p * f;
            }
        }
        Grid::new(h, out)
    }

    /// The raw mixed moment `E[A^p B^q]`.
    pub fn moment(&self, p: u32, q: u32) -> f64 {
        let [_, cols] = self.shape;
        let [ha, hb] = self.steps;
        let bq: Vec<f64> = (0..cols).map(|j| (j as f64 * hb).powi(q as i32)).collect();
        self.probs
            .chunks(cols)
            .enumerate()
            .map(|(i, row)| {
                let ap = (i as f64 * ha).powi(p as i32);
                ap * row.iter().zip(&bq).map(|(x, y)| x * y).sum::<f64>()
            })
            .sum()
    }

    /// The mean of axis `axis` (0 or 1). Panics on another axis.
    pub fn mean(&self, axis: usize) -> f64 {
        match axis {
            0 => self.moment(1, 0),
            1 => self.moment(0, 1),
            _ => panic!("axis must be 0 or 1"),
        }
    }

    /// The variance of axis `axis` (0 or 1). Panics on another axis.
    pub fn variance(&self, axis: usize) -> f64 {
        let m = self.mean(axis);
        let second = match axis {
            0 => self.moment(2, 0),
            _ => self.moment(0, 2),
        };
        second - m * m
    }

    /// `Cov(A, B)`.
    pub fn covariance(&self) -> f64 {
        self.moment(1, 1) - self.mean(0) * self.mean(1)
    }

    /// Pearson's correlation of `A` and `B`; NaN if either is constant.
    pub fn correlation(&self) -> f64 {
        let denom = self.variance(0) * self.variance(1);
        if denom <= 0.0 {
            return f64::NAN;
        }
        self.covariance() / denom.sqrt()
    }
}

fn bad_axis(axis: usize) -> Error {
    Error::InvalidParameter {
        name: "axis",
        value: axis as f64,
        reason: "must be 0 or 1",
    }
}

/// What a bivariate compound calculation produced and the error it
/// introduced.
#[derive(Debug, Clone, PartialEq)]
pub struct BivariateReport {
    /// Points on each axis of the result.
    pub points: [usize; 2],
    /// `P(A > (n_A - 1) h_A)` and `P(B > (n_B - 1) h_B)`: the mass beyond the
    /// last row and column, which is carried on them.
    pub tail_mass: [f64; 2],
    /// On each axis, the aggregate mass at or past the end of the FFT
    /// buffer, measured on that axis's marginal with a buffer twice as long.
    /// A finite FFT wraps this mass around onto the start, so their sum
    /// bounds the total probability misplaced in the result. When it is not
    /// negligible, request more points.
    pub aliasing_mass: [f64; 2],
    /// `E[N] E[X]` and `E[N] E[Y]` on the severity grid: the means before
    /// truncation.
    pub expected_means: [f64; 2],
    /// The means of the result.
    pub grid_means: [f64; 2],
}

/// The joint distribution of `(A, B) = Σ_{k ≤ N} (X_k, Y_k)` by a
/// two-dimensional FFT, on the severity's steps with `points` points on each
/// axis.
///
/// `severity` is the joint law of one event's pair of losses; mass at 0 on
/// an axis is an event that misses that line. The FFT buffer on each axis is
/// the power of two at least twice the larger of the points and the
/// severity's length (as [`fft`](crate::fft())); mass beyond it wraps around,
/// and [`BivariateReport::aliasing_mass`] measures how much. Each marginal of
/// the result is the one-dimensional compound of `N` with that axis's
/// per-event marginal, and the cost is `O(L_A L_B log(L_A L_B))`.
///
/// ```
/// use prospicio_aggregate::bivariate::{JointGrid, fft_bivariate};
/// use prospicio_aggregate::fft;
/// use prospicio_prob::{Grid, Poisson};
///
/// // Each event hits line A with 1 or 2, and line B with 0 or 1.
/// let a = Grid::new(1.0, vec![0.0, 0.6, 0.4]).unwrap();
/// let b = Grid::new(1.0, vec![0.5, 0.5]).unwrap();
/// let n = Poisson::new(2.0).unwrap();
/// let (joint, report) = fft_bivariate(&n, &JointGrid::independent(&a, &b), [64, 64]).unwrap();
/// let (alone, _) = fft(&n, &a, 64).unwrap();
/// for (x, y) in joint.marginal(0).unwrap().probs().iter().zip(alone.probs()) {
///     assert!((x - y).abs() < 1e-14);
/// }
/// // The shared count is the only link: Cov = Var(N) E[X] E[Y].
/// assert!((joint.covariance() - 2.0 * 1.4 * 0.5).abs() < 1e-9);
/// assert!(report.aliasing_mass.iter().all(|&m| m < 1e-15));
/// ```
pub fn fft_bivariate<N: Counting + ?Sized>(
    frequency: &N,
    severity: &JointGrid,
    points: [usize; 2],
) -> Result<(JointGrid, BivariateReport)> {
    if points.contains(&0) {
        return Err(Error::InvalidParameter {
            name: "points",
            value: 0.0,
            reason: "must be positive on both axes",
        });
    }
    let len = [0, 1].map(|a| (2 * points[a].max(severity.shape[a])).next_power_of_two());
    let full = compound_2d(frequency, severity, len);

    // Carry the mass past the last row and column on them.
    let [na, nb] = points;
    let mut probs = vec![0.0; na * nb];
    for (i, row) in full.chunks(len[1]).enumerate() {
        let r = i.min(na - 1);
        for (j, &p) in row.iter().enumerate() {
            probs[r * nb + j.min(nb - 1)] += p;
        }
    }
    // Round-off can leave tiny negatives. What the probabilities lack of 1
    // (round-off, and a severity summing to 1 only within 1e-9) stays
    // missing: put on the far corner it would move the moments.
    probs.iter_mut().for_each(|p| *p = p.max(0.0));
    let joint = JointGrid::new(severity.steps, points, probs)?;

    let marginals = [severity.marginal(0)?, severity.marginal(1)?];
    let mut tail_mass = [0.0; 2];
    let mut aliasing_mass = [0.0; 2];
    for a in 0..2 {
        tail_mass[a] = (points[a]..len[a])
            .map(|k| full_marginal_at(&full, len, a, k))
            .sum::<f64>()
            .max(0.0);
        let long = crate::fft::compound_buffer(frequency, marginals[a].probs(), 2 * len[a]);
        aliasing_mass[a] = long[len[a]..].iter().sum::<f64>().max(0.0);
    }
    let report = BivariateReport {
        points,
        tail_mass,
        aliasing_mass,
        expected_means: [
            frequency.mean() * severity.mean(0),
            frequency.mean() * severity.mean(1),
        ],
        grid_means: [joint.mean(0), joint.mean(1)],
    };
    Ok((joint, report))
}

/// The mass of the full buffer's marginal on axis `axis` at index `k`.
fn full_marginal_at(full: &[f64], len: [usize; 2], axis: usize, k: usize) -> f64 {
    match axis {
        0 => full[k * len[1]..(k + 1) * len[1]].iter().sum(),
        _ => full.chunks(len[1]).map(|row| row[k]).sum(),
    }
}

/// The compound pmf on a circular `len[0] × len[1]` buffer, row by row.
fn compound_2d<N: Counting + ?Sized>(
    frequency: &N,
    severity: &JointGrid,
    len: [usize; 2],
) -> Vec<f64> {
    let [rows, cols] = len;
    let mut buffer = vec![Complex64::new(0.0, 0.0); rows * cols];
    // A severity longer than the buffer wraps around, like the aggregate.
    for (i, row) in severity.probs.chunks(severity.shape[1]).enumerate() {
        for (j, &p) in row.iter().enumerate() {
            buffer[(i % rows) * cols + j % cols].re += p;
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    // A 2-D transform is a transform of every row, then of every column.
    planner.plan_fft_forward(cols).process(&mut buffer);
    let mut columns = transpose(&buffer, rows, cols);
    planner.plan_fft_forward(rows).process(&mut columns);
    // The pgf acts on each frequency alone, so the layout does not matter.
    for z in &mut columns {
        let (re, im) = frequency.pgf_complex((z.re, z.im));
        *z = Complex64::new(re, im);
    }
    planner.plan_fft_inverse(rows).process(&mut columns);
    let mut buffer = transpose(&columns, cols, rows);
    planner.plan_fft_inverse(cols).process(&mut buffer);
    // rustfft does not normalize. Round-off is left signed, so that it
    // cancels in the sums over rows and columns instead of adding up.
    let scale = (rows * cols) as f64;
    buffer.iter().map(|z| z.re / scale).collect()
}

/// The transpose of a `rows × cols` matrix stored row by row.
fn transpose(m: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = m[i * cols + j];
        }
    }
    out
}

/// Two lines hit by the same events, each line independently within an
/// event: the clash model of Mildenhall's `aggregate`.
///
/// It is given by the expected number of events a year that hit line A
/// only, line B only, and both (the clash count). Independence within an
/// event fixes the events that hit neither, `n_0 = n_A n_B / n_C`, so that
/// the 2 × 2 table of expected counts has independent rows and columns.
/// Then there are `n = n_A + n_B + n_C + n_0` events a year, and each hits
/// line A with probability `p_A = (n_A + n_C) / n` and line B with
/// `p_B = (n_B + n_C) / n`.
///
/// With a Poisson count the events hitting A only, B only and both are
/// independent Poisson counts with those means. With a mixed count (a
/// negative binomial) they move together, which adds dependence.
///
/// ```
/// use prospicio_aggregate::bivariate::Clash;
///
/// let clash = Clash::new(2.0, 3.0, 1.0).unwrap();
/// assert_eq!(clash.neither(), 6.0);
/// assert_eq!(clash.events(), 12.0);
/// assert_eq!(clash.trigger_a(), 0.25);
/// assert!((clash.trigger_b() - 1.0 / 3.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clash {
    only_a: f64,
    only_b: f64,
    both: f64,
}

impl Clash {
    /// A clash model with `only_a` events a year hitting line A only,
    /// `only_b` hitting line B only, and `both` hitting both.
    ///
    /// Fails if `both` is not finite and positive, or `only_a` or `only_b`
    /// is negative or not finite.
    pub fn new(only_a: f64, only_b: f64, both: f64) -> Result<Self> {
        if !both.is_finite() || both <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "both",
                value: both,
                reason: "must be finite and positive",
            });
        }
        for (name, value) in [("only_a", only_a), ("only_b", only_b)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidParameter {
                    name,
                    value,
                    reason: "must be finite and non-negative",
                });
            }
        }
        Ok(Self {
            only_a,
            only_b,
            both,
        })
    }

    /// Expected events a year hitting line A only.
    pub fn only_a(&self) -> f64 {
        self.only_a
    }

    /// Expected events a year hitting line B only.
    pub fn only_b(&self) -> f64 {
        self.only_b
    }

    /// Expected events a year hitting both lines: the clash count.
    pub fn both(&self) -> f64 {
        self.both
    }

    /// Expected events a year hitting neither line, `n_A n_B / n_C`.
    pub fn neither(&self) -> f64 {
        self.only_a * self.only_b / self.both
    }

    /// Expected events a year, the mean of the shared count.
    pub fn events(&self) -> f64 {
        self.only_a + self.only_b + self.both + self.neither()
    }

    /// The probability that an event hits line A.
    pub fn trigger_a(&self) -> f64 {
        (self.only_a + self.both) / self.events()
    }

    /// The probability that an event hits line B.
    pub fn trigger_b(&self) -> f64 {
        (self.only_b + self.both) / self.events()
    }

    /// The joint per-event severity: line A loses `a` with probability
    /// [`trigger_a`](Self::trigger_a) and nothing otherwise, line B loses `b`
    /// with probability [`trigger_b`](Self::trigger_b), independently.
    /// `a` and `b` are the loss to each line given that the event hits it.
    pub fn severity(&self, a: &Grid, b: &Grid) -> Result<JointGrid> {
        Ok(JointGrid::independent(
            &thin(a, self.trigger_a())?,
            &thin(b, self.trigger_b())?,
        ))
    }

    /// The joint annual loss to the two lines by [`fft_bivariate`], with
    /// `frequency` the shared count of events, whose mean must be
    /// [`events`](Self::events) (to `1e-9` relative).
    ///
    /// ```
    /// use prospicio_aggregate::bivariate::Clash;
    /// use prospicio_prob::{Grid, Poisson};
    ///
    /// let clash = Clash::new(2.0, 3.0, 1.0).unwrap();
    /// let a = Grid::new(1.0, vec![0.0, 0.5, 0.5]).unwrap();
    /// let b = Grid::new(1.0, vec![0.0, 1.0]).unwrap();
    /// let n = Poisson::new(clash.events()).unwrap();
    /// let (joint, _) = clash.aggregate(&n, &a, &b, [128, 128]).unwrap();
    /// // Poisson: Cov(A, B) = n_C E[X] E[Y].
    /// assert!((joint.covariance() - 1.0 * 1.5 * 1.0).abs() < 1e-9);
    /// ```
    pub fn aggregate<N: Counting + ?Sized>(
        &self,
        frequency: &N,
        a: &Grid,
        b: &Grid,
        points: [usize; 2],
    ) -> Result<(JointGrid, BivariateReport)> {
        let mean = frequency.mean();
        let gap = (mean / self.events() - 1.0).abs();
        if gap.is_nan() || gap > 1e-9 {
            return Err(Error::InvalidParameter {
                name: "frequency",
                value: mean,
                reason: "its mean must be the clash model's expected events",
            });
        }
        fft_bivariate(frequency, &self.severity(a, b)?, points)
    }
}

/// `(1 − p) δ_0 + p g`: a loss from `g` with probability `p`, else
/// nothing. The per-event marginal of a line an event hits with probability
/// `p`, for [`JointGrid::from_copula`].
///
/// ```
/// use prospicio_aggregate::bivariate::triggered;
/// use prospicio_prob::Grid;
///
/// let g = triggered(&Grid::new(1.0, vec![0.0, 1.0]).unwrap(), 0.25).unwrap();
/// assert_eq!(g.probs(), &[0.75, 0.25]);
/// ```
pub fn triggered(g: &Grid, p: f64) -> Result<Grid> {
    if !(0.0..=1.0).contains(&p) {
        return Err(Error::InvalidParameter {
            name: "p",
            value: p,
            reason: "must be in [0, 1]",
        });
    }
    thin(g, p)
}

/// `(1 - p) δ_0 + p g`: a loss of `g` with probability `p`, else nothing.
fn thin(g: &Grid, p: f64) -> Result<Grid> {
    let mut probs: Vec<f64> = g.probs().iter().map(|q| p * q).collect();
    probs[0] += 1.0 - p;
    Grid::new(g.step(), probs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft;
    use prospicio_prob::{Distribution, Gamma, Lognormal, NegativeBinomial, Poisson};

    fn severities() -> (Grid, Grid) {
        let (a, _) = Grid::rounding(&Gamma::from_mean_cv(10.0, 1.5).unwrap(), 1.0, 2048).unwrap();
        let (b, _) = Grid::rounding(&Lognormal::from_mean_cv(5.0, 0.5).unwrap(), 0.5, 256).unwrap();
        (a, b)
    }

    #[test]
    fn marginals_are_the_one_dimensional_compounds() {
        // Each marginal is the compound of N with that axis's per-event
        // marginal, a check independent of `aggregate`.
        let (a, b) = severities();
        let clash = Clash::new(2.0, 3.0, 1.0).unwrap();
        for n in [
            &Poisson::new(clash.events()).unwrap() as &dyn Counting,
            &NegativeBinomial::from_mean_variance(clash.events(), 24.0).unwrap(),
        ] {
            let (joint, report) = clash.aggregate(n, &a, &b, [2048, 512]).unwrap();
            let sev = clash.severity(&a, &b).unwrap();
            for axis in 0..2 {
                let (alone, r1) =
                    fft(n, &sev.marginal(axis).unwrap(), report.points[axis]).unwrap();
                let m = joint.marginal(axis).unwrap();
                // The 1-D compound's buffer is twice as long, so the two
                // differ by at most the mass this one wrapped around.
                let bound = report.aliasing_mass[axis] + 1e-12;
                assert!(report.aliasing_mass[axis] < 1e-10);
                for (x, y) in m.probs().iter().zip(alone.probs()) {
                    assert!((x - y).abs() <= bound, "axis {axis}");
                }
                assert!((report.tail_mass[axis] - r1.tail_mass).abs() <= bound);
            }
        }
    }

    #[test]
    fn covariance_comes_from_the_shared_count() {
        // Cov(A, B) = E[N] Cov(X, Y) + Var(N) E[X] E[Y], and Cov(X, Y) = 0
        // for independent triggers.
        let (a, b) = severities();
        let clash = Clash::new(2.0, 3.0, 1.0).unwrap();
        let (ea, eb) = (clash.trigger_a() * a.mean(), clash.trigger_b() * b.mean());
        let poisson = Poisson::new(clash.events()).unwrap();
        let (joint, _) = clash.aggregate(&poisson, &a, &b, [2048, 512]).unwrap();
        let cov = clash.events() * ea * eb;
        assert!((joint.covariance() / cov - 1.0).abs() < 1e-10);
        // Poisson: the clash count alone carries it.
        assert!((cov - clash.both() * a.mean() * b.mean()).abs() < 1e-9);

        let nb = NegativeBinomial::from_mean_variance(clash.events(), 24.0).unwrap();
        let (joint, _) = clash.aggregate(&nb, &a, &b, [2048, 512]).unwrap();
        assert!((joint.covariance() / (nb.variance() * ea * eb) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn total_is_the_sum() {
        let a = Grid::new(1.0, vec![0.2, 0.5, 0.3]).unwrap();
        let b = Grid::new(1.0, vec![0.4, 0.6]).unwrap();
        let n = Poisson::new(1.5).unwrap();
        let (joint, _) = fft_bivariate(&n, &JointGrid::independent(&a, &b), [64, 64]).unwrap();
        let total = joint.total().unwrap();
        let mean = joint.mean(0) + joint.mean(1);
        assert!((total.mean() - mean).abs() < 1e-12);
        let var = joint.variance(0) + joint.variance(1) + 2.0 * joint.covariance();
        assert!((total.variance() - var).abs() < 1e-10);
        // Unequal steps: the split keeps the mean.
        let b2 = Grid::new(0.5, vec![0.4, 0.6]).unwrap();
        let (joint, _) = fft_bivariate(&n, &JointGrid::independent(&a, &b2), [64, 64]).unwrap();
        let total = joint.total().unwrap();
        assert_eq!(total.step(), 1.0);
        assert!((total.mean() - joint.mean(0) - joint.mean(1)).abs() < 1e-12);
    }

    #[test]
    fn short_grids_carry_the_tail_on_the_edges() {
        let a = Grid::new(1.0, vec![0.0, 0.5, 0.5]).unwrap();
        let n = Poisson::new(2.0).unwrap();
        let sev = JointGrid::independent(&a, &a);
        let (joint, report) = fft_bivariate(&n, &sev, [6, 4]).unwrap();
        assert!((joint.probs().iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert!(report.tail_mass.iter().all(|&t| t > 1e-3));
        for axis in 0..2 {
            // The 1-D compound wraps less (its buffer is twice as long), so
            // the two differ by at most the mass this one wrapped around.
            let (alone, r1) = fft(&n, &a, report.points[axis]).unwrap();
            let bound = report.aliasing_mass[axis] + 1e-15;
            let m = joint.marginal(axis).unwrap();
            for (x, y) in m.probs().iter().zip(alone.probs()) {
                assert!((x - y).abs() <= bound);
            }
            assert!((report.tail_mass[axis] - r1.tail_mass).abs() <= bound);
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Clash::new(1.0, 1.0, 0.0).is_err());
        assert!(Clash::new(-1.0, 1.0, 1.0).is_err());
        let a = Grid::new(1.0, vec![0.5, 0.5]).unwrap();
        let clash = Clash::new(1.0, 1.0, 1.0).unwrap();
        let wrong = Poisson::new(2.0).unwrap();
        assert!(clash.aggregate(&wrong, &a, &a, [8, 8]).is_err());
        let sev = JointGrid::independent(&a, &a);
        assert!(fft_bivariate(&wrong, &sev, [0, 8]).is_err());
        assert!(JointGrid::new([1.0, 1.0], [2, 2], vec![0.5, 0.5]).is_err());
        assert!(sev.marginal(2).is_err());
    }

    #[test]
    fn a_copula_keeps_the_marginals_and_independence_is_the_product() {
        use prospicio_prob::copula::{
            Archimedean, ArchimedeanCopula, GaussianCopula, StudentTCopula,
        };
        let a = triggered(&Grid::new(1.0, vec![0.1, 0.3, 0.4, 0.2]).unwrap(), 0.5).unwrap();
        let b = Grid::new(2.0, vec![0.2, 0.5, 0.3]).unwrap();
        for copula in [
            Box::new(ArchimedeanCopula::new(Archimedean::Clayton, 3.0, 2).unwrap())
                as Box<dyn Copula>,
            Box::new(ArchimedeanCopula::new(Archimedean::Frank, 5.0, 2).unwrap()),
            Box::new(GaussianCopula::new(&[1.0, -0.6, -0.6, 1.0], 2).unwrap()),
        ] {
            let joint = JointGrid::from_copula(&a, &b, copula.as_ref()).unwrap();
            for (axis, g) in [(0, &a), (1, &b)] {
                for (got, want) in joint.marginal(axis).unwrap().probs().iter().zip(g.probs()) {
                    assert!((got - want).abs() < 1e-15, "{axis}: {got} vs {want}");
                }
            }
        }
        // A Gaussian at rho 0 is the independent product.
        let zero = GaussianCopula::new(&[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let joint = JointGrid::from_copula(&a, &b, &zero).unwrap();
        let product = JointGrid::independent(&a, &b);
        for (got, want) in joint.probs().iter().zip(product.probs()) {
            assert!((got - want).abs() < 1e-15);
        }
        // Positive dependence raises the covariance; negative lowers it.
        let up = ArchimedeanCopula::new(Archimedean::Gumbel, 3.0, 2).unwrap();
        let down = GaussianCopula::new(&[1.0, -0.6, -0.6, 1.0], 2).unwrap();
        assert!(JointGrid::from_copula(&a, &b, &up).unwrap().covariance() > 0.0);
        assert!(JointGrid::from_copula(&a, &b, &down).unwrap().covariance() < 0.0);
        let three = ArchimedeanCopula::new(Archimedean::Gumbel, 3.0, 3).unwrap();
        assert!(JointGrid::from_copula(&a, &b, &three).is_err());
        let t = StudentTCopula::new(&[1.0, 0.5, 0.5, 1.0], 2, 4.0).unwrap();
        assert!(JointGrid::from_copula(&a, &b, &t).is_err());
        assert!(triggered(&b, 1.5).is_err());
    }
}
