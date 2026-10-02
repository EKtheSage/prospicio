//! What every compound (aggregate) calculation reports.

/// How a compound distribution was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundMethod {
    Panjer,
    Fft,
}

/// What a compound calculation produced and the error it introduced.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundReport {
    pub method: CompoundMethod,
    /// Number of points in the aggregate grid.
    pub points: usize,
    /// `P(S > (n - 1)h)`, the aggregate mass above the last point, which is
    /// lumped onto it.
    pub tail_mass: f64,
    /// Largest change in any returned probability when the FFT buffer is
    /// doubled: a measured bound on wrap-around (aliasing) error. Zero for
    /// Panjer, which does not alias.
    pub aliasing_error: f64,
    /// `E[N] * E[X]` on the severity grid: the aggregate mean before
    /// truncation.
    pub expected_mean: f64,
    /// Mean of the aggregate grid.
    pub grid_mean: f64,
}

impl CompoundReport {
    /// `grid_mean - expected_mean`, from truncating the aggregate grid.
    pub fn mean_error(&self) -> f64 {
        self.grid_mean - self.expected_mean
    }
}

/// Puts everything above the last point onto it, so the probabilities sum
/// to 1, and returns the mass that moved there from beyond the grid.
pub(crate) fn lump_tail(mut g: Vec<f64>) -> (Vec<f64>, f64) {
    let n = g.len();
    let below_last: f64 = g[..n - 1].iter().sum();
    let tail_mass = (1.0 - below_last - g[n - 1]).max(0.0);
    g[n - 1] = (1.0 - below_last).max(0.0);
    (g, tail_mass)
}
