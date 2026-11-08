//! The diagonal backtest: the frame rebuilt as Meyers' cumulative cells and `TriangleModel for Meyers`.

use super::*;

/// The train rows of a [`TriangleFrame`] rebuilt as Meyers' cumulative
/// cells, and the test rows as increments of each origin's path.
#[derive(Debug, Clone)]
pub(super) struct Rebuilt {
    /// The frame's origin position of each model origin `w`.
    pub(super) origins: Vec<usize>,
    /// Number of ages, the oldest train age plus one.
    n_dev: usize,
    /// Each origin's latest train age.
    pub(super) latest: Vec<usize>,
    /// Each origin's cumulative value at its latest train age.
    pub(super) last_value: Vec<f64>,
    /// Each origin's premium (the frame's `exposure`).
    pub(super) premiums: Vec<f64>,
    /// The cumulative cells `(w, d, C_wd)`, origin by origin.
    values: Vec<(usize, usize, f64)>,
    /// `(w, d)` of each test row, in order.
    pub(super) test: Vec<(usize, usize)>,
    /// Each origin's oldest test age, if it has a test row: its path runs
    /// from its latest train age to there.
    pub(super) path_end: Vec<Option<usize>>,
}

impl Rebuilt {
    pub(super) fn new(cells: &TriangleFrame, train: &[usize], test: &[usize]) -> Result<Self> {
        let response = cells.response();
        let exposure = match cells.frame().column("exposure") {
            Some(Column::Numeric(v)) => v,
            _ => unreachable!("TriangleFrame::new builds a numeric exposure column"),
        };
        let mut by_origin: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for &r in train {
            by_origin
                .entry(cells.origin_of(r))
                .or_default()
                .push((cells.development_of(r), r));
        }
        let origins: Vec<usize> = by_origin.keys().copied().collect();
        let label = |o: usize| cells.origins()[o];
        if let Some(gap) = origins.windows(2).find(|p| p[1] != p[0] + 1) {
            return Err(Error::Bayes(format!(
                "origin {} has no train row but origins {} and {} have; Meyers' models need \
                 the train origins without a gap",
                label(gap[0] + 1),
                label(gap[0]),
                label(gap[1])
            )));
        }
        let (mut latest, mut last_value, mut premiums, mut values) =
            (vec![], vec![], vec![], vec![]);
        for (w, (&o, rows)) in by_origin.iter_mut().enumerate() {
            rows.sort_unstable();
            let mut cumulative = 0.0;
            for (k, &(d, r)) in rows.iter().enumerate() {
                if d != k {
                    return Err(Error::Bayes(format!(
                        "origin {} has no train row at development age {} months but has one \
                         later; Meyers' models rebuild each origin's cumulative values from \
                         its first age",
                        label(o),
                        cells.ages()[k]
                    )));
                }
                cumulative += response[r];
                values.push((w, d, cumulative));
            }
            latest.push(rows.len() - 1);
            last_value.push(cumulative);
            premiums.push(exposure[rows[0].1]);
        }
        let n_dev = latest.iter().max().map_or(0, |&l| l + 1);
        let mut path_end = vec![None; origins.len()];
        let test = test
            .iter()
            .map(|&r| {
                let (o, d) = (cells.origin_of(r), cells.development_of(r));
                let w = origins.binary_search(&o).map_err(|_| {
                    Error::Bayes(format!("test row {r} has an origin without a train row"))
                })?;
                if d >= n_dev {
                    return Err(Error::Bayes(format!(
                        "test row {r} has an age without a train row"
                    )));
                }
                if d <= latest[w] {
                    return Err(Error::Bayes(format!(
                        "test row {r} is not after its origin's latest train row; Meyers' \
                         models forecast each origin's cumulative path past it"
                    )));
                }
                let end: &mut Option<usize> = &mut path_end[w];
                *end = Some(end.map_or(d, |e| e.max(d)));
                Ok((w, d))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            origins,
            n_dev,
            latest,
            last_value,
            premiums,
            values,
            test,
            path_end,
        })
    }

    /// Whether origin `w`'s path holds age `d`.
    pub(super) fn in_path(&self, w: usize, d: usize) -> bool {
        d > self.latest[w] && self.path_end[w].is_some_and(|e| d <= e)
    }

    /// The ages some path holds, in order.
    pub(super) fn path_ages(&self) -> Vec<usize> {
        (0..self.n_dev)
            .filter(|&d| (0..self.origins.len()).any(|w| self.in_path(w, d)))
            .collect()
    }
}

/// The diagonal backtest's view of a [`Meyers`] (decision 8). The frame's
/// rows are increments, and Meyers models cumulative values with premium,
/// so the model is fitted to the train rows rebuilt as cumulative cells:
///
/// * **Cumulative values.** Each origin's train rows must run from the
///   first age without a gap (an [`Error::Bayes`] otherwise), and their
///   running sums are its cumulative values, which must be positive. The
///   train origins must have no gap between them either (CAY's previous
///   origin is the one before). The model's ages are the first age to the
///   oldest train age, which takes `beta_m = 0`.
/// * **Premium** is the frame's `exposure` column (the origin's exposure,
///   given to [`TriangleFrame::new`]). A frame built without exposure has
///   1 there, which the model fits but which puts its `logelr` far from
///   Meyers' prior on a loss ratio; give the premium.
/// * **Test rows** must be after their origin's latest train row, at an
///   age with train rows. Each origin's cumulative value is simulated
///   along one path, from the age after its latest train row to its oldest
///   test row, with one posterior draw per simulation; a test row's
///   increment is the difference of consecutive values on that path (the
///   first from the latest train value), so an origin's increments sum to
///   its last simulated value less its latest train value. Given the
///   parameters the values are independent log-normals (in CAY, a moving
///   average down each age, the previous origin's residual observed for a
///   train cell and drawn otherwise), so a simulated increment can be
///   negative, as Meyers' reserves can.
/// * **Mean** is each test row's posterior mean increment: the mean of
///   the log-normal `E[C_wd]` less the latest train value (or less
///   `E[C_w,d-1]` past the first age of the path), over every posterior
///   draw.
/// * **Log density.** When every test row follows its origin's latest
///   train row (always so in [`diagonal_backtest`](crate::diagonal_backtest)),
///   a test increment `y` is `C_wd` less a known value, so its density is
///   the log-normal density of `C_wd` at `y` plus the latest train value,
///   per unit of the response, and the log density is the log of its mean
///   over every posterior draw (the log predictive density, as
///   [`BayesOdp`](crate::bayes::BayesOdp) gives it); minus infinity when that sum
///   is not positive. In CAY a test row whose previous origin has no train
///   cell at its age takes that origin's residual integrated out (variance
///   `(1 + rho^2) sigma_d^2`). When some test row is further out, its
///   increment is a difference of two random log-normals, with no closed
///   density, and the log density is `None`.
/// * **Seeds.** The chains (and the pilot run) and the simulations take
///   seeds derived from the backtest's `seed` as [`Meyers::fit`] does from
///   [`Meyers::seed`]; the backtest's `n_sims` replaces [`Meyers::n_sims`].
///   The fit's diagnostics come back in [`CellForecast::diagnostics`] and
///   [`crate::Backtest::diagnostics`], and a fit with warnings is still scored.
///   A held-out triangle is smaller than the whole: on Meyers'
///   group 353 the default target acceptance 0.95 failed (R-hat above
///   1.05) in 21 of 3,960 backtest fits on its 28- to 45-cell training
///   triangles, and 0.99 in 4 of 5,400; raise [`Meyers::sampler`]'s
///   `target_accept` for small triangles.
///
/// Its name in the backtest is `"meyers_crc"`, `"meyers_csr"` or
/// `"meyers_cay"`.
impl TriangleModel for Meyers {
    fn name(&self) -> &str {
        match self.model {
            MeyersModel::Crc => "meyers_crc",
            MeyersModel::Csr => "meyers_csr",
            MeyersModel::Cay => "meyers_cay",
        }
    }

    fn forecast(
        &self,
        cells: &TriangleFrame,
        train: &[usize],
        test: &[usize],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast> {
        self.validate()?;
        let rebuilt = Rebuilt::new(cells, train, test)?;
        let no = rebuilt.origins.len();
        let origins: Vec<Period> = rebuilt
            .origins
            .iter()
            .map(|&o| cells.origins()[o])
            .collect();
        let model = self.model_from(
            no,
            rebuilt.n_dev,
            &rebuilt.values,
            &rebuilt.premiums,
            &origins,
            cells.ages(),
        )?;
        let (posterior, _, diagnostics) = self.sample(&model, derive_seed(seed, Stream::Chains))?;
        let at = cell_index(&model.cells, no, rebuilt.n_dev);
        let ages = rebuilt.path_ages();

        let n_draws = posterior.n_draws();
        let response = cells.response();
        let closed = rebuilt
            .test
            .iter()
            .all(|&(w, d)| d == rebuilt.latest[w] + 1);
        let mut mean = vec![0.0; test.len()];
        let mut ln_f = vec![Vec::with_capacity(n_draws); test.len()];
        let mut residuals = Vec::new();
        for s in 0..n_draws {
            let nat = posterior.draw(s);
            model.cay_residuals(nat, &mut residuals);
            let moments = |w: usize, d: usize| {
                model.log_moments(nat, &residuals, &at, &rebuilt.premiums, w, d)
            };
            for (j, (&r, &(w, d))) in test.iter().zip(&rebuilt.test).enumerate() {
                let (m, v) = moments(w, d);
                let before = if d == rebuilt.latest[w] + 1 {
                    rebuilt.last_value[w]
                } else {
                    let (m0, v0) = moments(w, d - 1);
                    (m0 + 0.5 * v0).exp()
                };
                mean[j] += ((m + 0.5 * v).exp() - before) / n_draws as f64;
                if closed {
                    let c = response[r] + rebuilt.last_value[w];
                    ln_f[j].push(if c > 0.0 && c.is_finite() {
                        let z = c.ln() - m;
                        -c.ln() - 0.5 * (std::f64::consts::TAU * v).ln() - 0.5 * z * z / v
                    } else {
                        f64::NEG_INFINITY
                    });
                }
            }
        }
        let log_density = closed.then(|| ln_f.iter().map(|f| log_mean_exp(f)).collect());

        let distribution = posterior.predictive(
            vec!["row".into()],
            test.iter().map(|&r| vec![(r as i64).into()]).collect(),
            n_sims,
            derive_seed(seed, Stream::Predictive),
            self.provenance("backtest", "exposure", InputHasher::new())
                .param("output", "test increments"),
            |nat, rng: &mut StreamRng, row| {
                model.draw_paths(nat, &rebuilt, &at, &ages, rng, row);
            },
        )?;
        Ok(CellForecast {
            mean,
            distribution,
            log_density,
            diagnostics: Some(diagnostics),
        })
    }
}
