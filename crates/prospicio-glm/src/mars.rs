//! MARS knot selection (Friedman, 1991) for additive hinge terms, as R's
//! `earth` does it with `degree = 1`.
//!
//! [`Mars::fit`] runs the forward pass: from the intercept it adds at each
//! step the predictor and knot `k` that most reduce the residual sum of
//! squares, as the pair `max(0, x − k)`, `max(0, k − x)` the first time `x`
//! enters, as the single hinge `max(0, x − k)` after that, or as `x` itself
//! when no knot beats the linear term. Knots are tried every `min_span`
//! sorted cases, `end_span` cases from each end, and the pass stops on
//! `earth`'s rules: no improving term, an `R²` gain below `threshold` (the
//! last term dropped), or `max_terms` reached. An optional backward pass
//! drops terms one at a time by least increase in RSS and keeps the size
//! with the lowest GCV (`earth`'s `pmethod = "backward"`).
//!
//! The candidate rules follow `earth`'s C code, so its knots come out
//! exactly: unweighted, a knot is scored by the closed form of
//! `FindKnot` (a hinge nearly collinear with the basis is skipped, and a
//! gain over ten times the previous step's is refused); with case weights,
//! by the regression RSS as `FindWeightedKnot` (a gain must exceed
//! `1e-10`), and a pair's lower hinge is kept or dropped by `earth`'s test
//! against a basis whose intercept is the constant, not `√w`, a column that
//! is then dropped from the result if the weighted basis does not need it
//! (`earth`'s "fixed rank deficient bx"). `validation/tests/models.rs`
//! checks the knots, RSS and GCV against `earth` 5.3 on the Chalk portfolio.
//!
//! With [`Mars::glm`] the search runs on a GLM's deviance instead: each
//! step scores the candidates by weighted least squares on the current
//! fit's IRLS working response and weights (the score approximation to the
//! deviance drop), adds the best and refits the GLM, so the steps' `rss`
//! are deviances. `earth`'s own `glm` argument fits a GLM after a
//! least-squares search, so this criterion has no `earth` counterpart.
//!
//! The selected knots feed [`Terms::hinges`]: [`MarsFit::add_terms`] adds
//! each predictor as `x` and hinges at its knots, which spans the pair
//! `max(0, x − k)`, `max(0, k − x)` with the intercept.

use prospicio_core::{Error, Result};
use prospicio_models::{Design, Family, Link, Terms};

use crate::{Glm, GlmFit};

/// The form of one basis term in a predictor `x` at knot `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `max(0, x − k)`, `earth`'s direction `1`.
    Up,
    /// `max(0, k − x)`, direction `-1`.
    Down,
    /// `x` itself, direction `2`; `k` is the smallest `x`.
    Linear,
}

/// A basis term: a predictor, a knot and a [`Direction`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarsTerm {
    /// Index of the predictor, in the order given to [`Mars::fit`].
    pub variable: usize,
    pub knot: f64,
    pub direction: Direction,
}

impl MarsTerm {
    /// The term at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        match self.direction {
            Direction::Up => (x - self.knot).max(0.0),
            Direction::Down => (self.knot - x).max(0.0),
            Direction::Linear => x,
        }
    }
}

/// One step of the forward pass: the term or pair it added and the fit
/// after it.
#[derive(Debug, Clone, PartialEq)]
pub struct MarsStep {
    /// The terms added: one, or a pair `Up` and `Down` at the same knot.
    pub terms: Vec<MarsTerm>,
    /// The residual sum of squares (weighted), or the deviance with
    /// [`Mars::glm`].
    pub rss: f64,
    /// `1 − rss / rss₀`, against the intercept-only fit.
    pub rsq: f64,
    /// The GCV: `rss / (n (1 − c / n)²)`, `c = m + penalty (m − 1) / 2`
    /// with `m` the terms in use, the intercept included.
    pub gcv: f64,
}

/// Why the forward pass stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// No candidate reduced the RSS.
    NoImprovement,
    /// The last step's `R²` gain was below the threshold; it was dropped.
    Threshold,
    /// The last step's GCV `R²` fell below −10; it was dropped.
    Gcv,
    /// `R²` reached `1 − threshold`.
    Rsq,
    /// The number of terms reached `max_terms`.
    MaxTerms,
}

/// MARS's forward pass, and optionally its backward pass, with `earth`'s
/// defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mars {
    /// `earth`'s `nk`, the most terms the forward pass may hold with the
    /// intercept: by default `min(200, max(20, 2p)) + 1` for `p`
    /// predictors.
    pub max_terms: Option<usize>,
    /// The forward pass stops when a step adds less than this to `R²`;
    /// `0` turns the `R²` and GCV rules off. Default `0.001`.
    pub threshold: f64,
    /// GCV's cost per knot, `2` for additive terms.
    pub penalty: f64,
    /// Cases between knots; by default Friedman's
    /// `(2.9702 + ln(p n)) / 1.7329`, rounded down.
    pub min_span: Option<usize>,
    /// Cases kept free of knots at each end; by default
    /// `7.32193 + log₂ p`, rounded down.
    pub end_span: Option<usize>,
    /// Run the backward pass by GCV (`earth`'s `pmethod = "backward"`);
    /// off by default (`pmethod = "none"`).
    pub prune: bool,
    /// Search on this GLM's deviance instead of least squares.
    pub glm: Option<(Family, Link)>,
}

impl Default for Mars {
    fn default() -> Self {
        Self::new()
    }
}

impl Mars {
    /// `earth`'s defaults with `degree = 1` and `pmethod = "none"`.
    pub fn new() -> Self {
        Self {
            max_terms: None,
            threshold: 0.001,
            penalty: 2.0,
            min_span: None,
            end_span: None,
            prune: false,
            glm: None,
        }
    }

    /// The same search with at most `n` terms, the intercept included.
    pub fn max_terms(mut self, n: usize) -> Self {
        self.max_terms = Some(n);
        self
    }

    /// The same search with the `R²` threshold `t`.
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// The same search with GCV cost `penalty` per knot.
    pub fn penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    /// The same search with `n` cases between knots.
    pub fn min_span(mut self, n: usize) -> Self {
        self.min_span = Some(n);
        self
    }

    /// The same search with `n` cases free of knots at each end.
    pub fn end_span(mut self, n: usize) -> Self {
        self.end_span = Some(n);
        self
    }

    /// The same search, then the backward pass by GCV when `prune`.
    pub fn prune(mut self, prune: bool) -> Self {
        self.prune = prune;
        self
    }

    /// The same search on the deviance of a GLM with `family` and `link`.
    pub fn glm(mut self, family: Family, link: Link) -> Self {
        self.glm = Some((family, link));
        self
    }

    /// Selects terms on the predictor columns `x` (named by `names`) for
    /// the response `y`, with optional case weights (exposure or prior
    /// weights; `earth`'s `weights`). Weights all equal are treated as
    /// none, as `earth` does.
    pub fn fit(
        &self,
        names: &[&str],
        x: &[Vec<f64>],
        y: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<MarsFit> {
        let n = y.len();
        let p = x.len();
        if p == 0 || names.len() != p {
            return Err(Error::Data(format!(
                "{} names for {p} predictor columns",
                names.len()
            )));
        }
        if let Some(c) = x.iter().find(|c| c.len() != n) {
            return Err(Error::Data(format!(
                "a predictor column has {} rows for {n} responses",
                c.len()
            )));
        }
        if n < 4 {
            return Err(Error::Data(format!("{n} cases; MARS needs at least 4")));
        }
        if x.iter().flatten().chain(y).any(|v| !v.is_finite()) {
            return Err(Error::Data("predictors and response must be finite".into()));
        }
        let w = match weights {
            Some(w) if w.len() != n => {
                return Err(Error::Data(format!("{} weights for {n} cases", w.len())));
            }
            Some(w) if w.iter().any(|&v| !(v >= 0.0 && v.is_finite())) => {
                return Err(Error::Data(
                    "weights must be finite and not negative".into(),
                ));
            }
            Some(w) => w.to_vec(),
            None => vec![1.0; n],
        };
        if !(self.threshold >= 0.0 && self.threshold < 1.0) {
            return Err(Error::InvalidParameter {
                name: "threshold",
                value: self.threshold,
                reason: "must be in [0, 1)",
            });
        }
        if !(self.penalty >= 0.0 && self.penalty.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "penalty",
                value: self.penalty,
                reason: "must be finite and not negative",
            });
        }
        let weighted = self.glm.is_some() || w.iter().any(|&v| v != w[0]);
        let max_terms = self.max_terms.unwrap_or_else(|| (2 * p).clamp(20, 200) + 1);
        if max_terms < 3 {
            return Err(Error::InvalidParameter {
                name: "max_terms",
                value: max_terms as f64,
                reason: "must be at least 3",
            });
        }
        let used = w.iter().filter(|&&v| v > 0.0).count();
        let spans = self.spans(n, p, used);
        let order: Vec<Vec<usize>> = x
            .iter()
            .map(|c| {
                let mut o: Vec<usize> = (0..n).collect();
                o.sort_by(|&a, &b| c[a].total_cmp(&c[b]));
                o
            })
            .collect();
        let state = Search {
            x,
            order: &order,
            spans,
            weighted,
            penalty: self.penalty,
        };
        let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        match self.glm {
            None => self.least_squares(&state, names, y, &w, max_terms),
            Some((family, link)) => self.deviance(&state, names, y, &w, max_terms, family, link),
        }
    }

    /// `earth`'s `GetSpanParams`: (min span, end span, start span), with
    /// its rounded constants (`0.69315` for `ln 2`), which set the knots.
    #[allow(clippy::approx_constant)]
    fn spans(&self, n: usize, p: usize, used: usize) -> (usize, usize, usize) {
        let end = self
            .end_span
            .unwrap_or_else(|| (7.32193 + (p as f64).ln() / 0.69315) as usize)
            .min((n / 2).saturating_sub(1))
            .max(1);
        let min = self
            .min_span
            .unwrap_or_else(|| ((2.9702 + ((p * used) as f64).ln()) / 1.7329) as usize)
            .max(1);
        let avail = n.saturating_sub(2 * end);
        let mut start = avail / 2;
        if avail > min {
            let div = avail / min;
            start = if avail == div * min {
                min / 2
            } else {
                (avail - div * min) / 2
            };
        }
        (min, end, (end + start).max(1))
    }

    fn least_squares(
        &self,
        s: &Search,
        names: Vec<String>,
        y: &[f64],
        w: &[f64],
        max_terms: usize,
    ) -> Result<MarsFit> {
        let n = y.len();
        let mut basis = Basis::new(w, s.weighted);
        let mut r = basis.residual(y);
        let rss_null = dot(&r, &r, w);
        if rss_null <= 0.0 {
            return Err(Error::Data("the response is constant".into()));
        }
        // `earth` scales y to unit sd: its absolute tolerances are in those units.
        let scale = sample_var(y);
        let mut rss = rss_null;
        let mut previous_delta = rss_null;
        let mut used = 1;
        let mut slots = 1;
        let mut steps = Vec::new();
        let gcv = |rss: f64, m: usize| gcv(rss, m, n, s.penalty);
        let gcv_null = gcv(rss_null, 1);
        let termination = loop {
            let max_legal = (1.01 * rss).min(10.0 * previous_delta);
            let Some(c) = s.best(&basis, &r, slots, max_legal, 1e-10 * scale) else {
                break Termination::NoImprovement;
            };
            let added = basis.add(s, &c);
            r = basis.residual(y);
            let new_rss = dot(&r, &r, w).max(0.0);
            let new_used = used + added.counted;
            let rsq_delta = ((1.0 - new_rss / rss_null) - (1.0 - rss / rss_null)).max(0.0);
            if self.threshold != 0.0 && rsq_delta < self.threshold {
                basis.pop(added.columns);
                break Termination::Threshold;
            }
            if self.threshold != 0.0 && 1.0 - gcv(new_rss, new_used) / gcv_null < -10.0 {
                basis.pop(added.columns);
                break Termination::Gcv;
            }
            previous_delta = rss - new_rss;
            rss = new_rss;
            used = new_used;
            slots += 2;
            steps.push(MarsStep {
                terms: added.terms,
                rss,
                rsq: 1.0 - rss / rss_null,
                gcv: gcv(rss, used),
            });
            if 1.0 - rss / rss_null >= 1.0 - self.threshold {
                break Termination::Rsq;
            }
            if slots + 1 >= max_terms {
                break Termination::MaxTerms;
            }
        };
        let forward = basis.independent_terms();
        let columns: Vec<Vec<f64>> = forward.iter().map(|t| s.column(t)).collect();
        let keep = if self.prune {
            let gram = Gram::new(&columns, y, w);
            gram.prune(n, s.penalty)
        } else {
            (0..forward.len()).collect()
        };
        let gram = Gram::new(&columns, y, w);
        let (coefficients, rss) = gram.solve(&keep)?;
        let terms: Vec<MarsTerm> = keep.iter().map(|&j| forward[j]).collect();
        Ok(MarsFit {
            names,
            gcv: gcv(rss, terms.len() + 1),
            rsq: 1.0 - rss / rss_null,
            grsq: 1.0 - gcv(rss, terms.len() + 1) / gcv_null,
            rss,
            coefficients,
            terms,
            forward,
            steps,
            termination,
            glm: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn deviance(
        &self,
        s: &Search,
        names: Vec<String>,
        y: &[f64],
        w: &[f64],
        max_terms: usize,
        family: Family,
        link: Link,
    ) -> Result<MarsFit> {
        let n = y.len();
        let glm = Glm::new(family, link);
        let fit_terms = |terms: &[MarsTerm], start: Option<&[f64]>| -> Result<GlmFit> {
            let design = s.design(terms, w)?;
            glm.fit_from(&design, y, start)
        };
        let mut terms: Vec<MarsTerm> = Vec::new();
        let mut fit = fit_terms(&terms, None)?;
        let dev_null = fit.deviance();
        if dev_null <= 0.0 {
            return Err(Error::Data("the intercept-only GLM fits exactly".into()));
        }
        let gcv = |dev: f64, m: usize| gcv(dev, m, n, s.penalty);
        let gcv_null = gcv(dev_null, 1);
        let mut dev = dev_null;
        let mut used = 1;
        let mut slots = 1;
        let mut steps = Vec::new();
        let termination = loop {
            let (z, v) = working(&fit, link, family, y, w);
            let mut basis = Basis::new(&v, true);
            for t in &terms {
                basis.push(s, *t);
            }
            let r = basis.residual(&z);
            let scale = dot(&r, &r, &v) / n as f64;
            let Some(c) = s.best(&basis, &r, slots, f64::INFINITY, 1e-10 * scale) else {
                break Termination::NoImprovement;
            };
            let added = basis.add(s, &c);
            let independent = basis.independent_terms();
            let Ok(new_fit) = fit_terms(&independent, Some(fit.fitted())) else {
                break Termination::NoImprovement;
            };
            let new_dev = new_fit.deviance();
            let new_used = used + added.counted;
            let rsq_delta = ((dev - new_dev) / dev_null).max(0.0);
            if self.threshold != 0.0 && rsq_delta < self.threshold {
                break Termination::Threshold;
            }
            if self.threshold != 0.0 && 1.0 - gcv(new_dev, new_used) / gcv_null < -10.0 {
                break Termination::Gcv;
            }
            terms = independent;
            fit = new_fit;
            dev = new_dev;
            used = new_used;
            slots += 2;
            steps.push(MarsStep {
                terms: added.terms,
                rss: dev,
                rsq: 1.0 - dev / dev_null,
                gcv: gcv(dev, used),
            });
            if 1.0 - dev / dev_null >= 1.0 - self.threshold {
                break Termination::Rsq;
            }
            if slots + 1 >= max_terms {
                break Termination::MaxTerms;
            }
        };
        let forward = terms;
        let mut selected = forward.clone();
        if self.prune && !forward.is_empty() {
            // The backward pass on the final fit's working response and
            // weights, the deviance's quadratic approximation there.
            let (z, v) = working(&fit, link, family, y, w);
            let columns: Vec<Vec<f64>> = forward.iter().map(|t| s.column(t)).collect();
            let gram = Gram::new(&columns, &z, &v);
            let (_, full) = gram.solve(&(0..forward.len()).collect::<Vec<_>>())?;
            let keep = gram.prune_with(n, s.penalty, |rss| dev + (rss - full));
            selected = keep.iter().map(|&j| forward[j]).collect();
            if selected.len() != forward.len() {
                fit = fit_terms(&selected, Some(fit.fitted()))?;
            }
        }
        let dev = fit.deviance();
        Ok(MarsFit {
            names,
            gcv: gcv(dev, selected.len() + 1),
            rsq: 1.0 - dev / dev_null,
            grsq: 1.0 - gcv(dev, selected.len() + 1) / gcv_null,
            rss: dev,
            coefficients: fit.coefficients().to_vec(),
            terms: selected,
            forward,
            steps,
            termination,
            glm: Some(fit),
        })
    }
}

/// The IRLS working response and weights of `fit`.
fn working(fit: &GlmFit, link: Link, family: Family, y: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mu = fit.fitted();
    let mut z = Vec::with_capacity(y.len());
    let mut v = Vec::with_capacity(y.len());
    for i in 0..y.len() {
        let eta = link.link(mu[i]);
        let d = link.mu_eta(eta);
        z.push(eta + (y[i] - mu[i]) / d);
        v.push(w[i] * d * d / family.variance(mu[i]));
    }
    (z, v)
}

fn gcv(rss: f64, used: usize, n: usize, penalty: f64) -> f64 {
    let m = used as f64;
    let c = m + penalty * (m - 1.0) / 2.0;
    let n = n as f64;
    rss / (n * (1.0 - c / n).powi(2))
}

fn dot(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    a.iter().zip(b).zip(w).map(|((a, b), w)| w * a * b).sum()
}

fn sample_var(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let m = y.iter().sum::<f64>() / n;
    y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Removes from `c` its projections on the orthonormal columns `q` (under
/// the weights `w`), twice for accuracy; returns the remaining sum of
/// squares.
fn orthogonalize(c: &mut [f64], q: &[Vec<f64>], w: &[f64]) -> f64 {
    for _ in 0..2 {
        for qj in q {
            let b = dot(qj, c, w);
            if b != 0.0 {
                c.iter_mut().zip(qj).for_each(|(c, q)| *c -= b * q);
            }
        }
    }
    dot(c, c, w)
}

/// `earth`'s `MIN_BX_SOS`: a column whose part outside the basis has a
/// smaller sum of squares adds nothing.
const MIN_BX_SOS: f64 = 0.01;

/// The terms in the model and an orthonormal basis for them under the
/// weights; with `weighted`, also `earth`'s check basis, the columns
/// scaled by `√w` but orthogonalized against the constant (its weighted
/// code's `bxOrth`), which decides whether a pair keeps its lower hinge.
struct Basis<'a> {
    w: &'a [f64],
    weighted: bool,
    terms: Vec<MarsTerm>,
    /// For each term, whether it adds to the weighted span.
    independent: Vec<bool>,
    q: Vec<Vec<f64>>,
    check: Vec<Vec<f64>>,
}

struct Added {
    terms: Vec<MarsTerm>,
    /// Columns pushed, to undo.
    columns: usize,
    /// Terms `earth` counts in its GCV.
    counted: usize,
}

impl<'a> Basis<'a> {
    fn new(w: &'a [f64], weighted: bool) -> Self {
        let n = w.len();
        let total: f64 = w.iter().sum();
        let q0 = vec![1.0 / total.sqrt(); n];
        let check = if weighted {
            vec![vec![1.0 / (n as f64).sqrt(); n]]
        } else {
            Vec::new()
        };
        Self {
            w,
            weighted,
            terms: Vec::new(),
            independent: Vec::new(),
            q: vec![q0],
            check,
        }
    }

    fn residual(&self, y: &[f64]) -> Vec<f64> {
        let mut r = y.to_vec();
        orthogonalize(&mut r, &self.q, self.w);
        r
    }

    /// Adds `term`'s column; returns whether its part outside the span
    /// survives (`earth`'s QR tolerance, relative `1e-8` on the norm).
    fn push(&mut self, s: &Search, term: MarsTerm) -> bool {
        let mut c = s.column(&term);
        let norm = dot(&c, &c, self.w);
        let ss = orthogonalize(&mut c, &self.q, self.w);
        let good = norm > 0.0 && ss > 1e-16 * norm;
        if good {
            let len = ss.sqrt();
            c.iter_mut().for_each(|v| *v /= len);
            self.q.push(c);
        } else {
            self.q.push(vec![0.0; self.w.len()]);
        }
        if self.weighted {
            let mut e: Vec<f64> = s
                .column(&term)
                .iter()
                .zip(self.w)
                .map(|(v, w)| v * w.sqrt())
                .collect();
            let ones = vec![1.0; self.w.len()];
            let ss = orthogonalize(&mut e, &self.check, &ones);
            if ss > 0.0 {
                let len = ss.sqrt();
                e.iter_mut().for_each(|v| *v /= len);
                self.check.push(e);
            } else {
                self.check.push(vec![0.0; self.w.len()]);
            }
        }
        self.terms.push(term);
        self.independent.push(good);
        good
    }

    fn pop(&mut self, columns: usize) {
        for _ in 0..columns {
            self.terms.pop();
            self.independent.pop();
            self.q.pop();
            if self.weighted {
                self.check.pop();
            }
        }
    }

    /// Whether `earth`'s weighted check keeps `term`: its `√w`-scaled
    /// column has more than [`MIN_BX_SOS`] outside the check basis.
    fn passes_check(&self, s: &Search, term: &MarsTerm) -> bool {
        let mut e: Vec<f64> = s
            .column(term)
            .iter()
            .zip(self.w)
            .map(|(v, w)| v * w.sqrt())
            .collect();
        let ones = vec![1.0; self.w.len()];
        orthogonalize(&mut e, &self.check, &ones) > MIN_BX_SOS
    }

    fn add(&mut self, s: &Search, c: &Candidate) -> Added {
        let k = c.knot;
        let v = c.variable;
        if c.linear {
            let t = MarsTerm {
                variable: v,
                knot: k,
                direction: Direction::Linear,
            };
            self.push(s, t);
            return Added {
                terms: vec![t],
                columns: 1,
                counted: 1,
            };
        }
        let up = MarsTerm {
            variable: v,
            knot: k,
            direction: Direction::Up,
        };
        let down = MarsTerm {
            direction: Direction::Down,
            ..up
        };
        self.push(s, up);
        let pair = c.new_form && (!self.weighted || self.passes_check(s, &down));
        if pair {
            self.push(s, down);
            Added {
                terms: vec![up, down],
                columns: 2,
                counted: 2,
            }
        } else {
            Added {
                terms: vec![up],
                columns: 1,
                counted: 1,
            }
        }
    }

    /// The terms that add to the weighted span, in order.
    fn independent_terms(&self) -> Vec<MarsTerm> {
        self.terms
            .iter()
            .zip(&self.independent)
            .filter(|(_, g)| **g)
            .map(|(t, _)| *t)
            .collect()
    }

    /// `earth`'s `GetNewFormFlag` for additive terms: true unless every
    /// term in the model is on `variable`.
    fn new_form_flag(&self, variable: usize) -> bool {
        self.terms.is_empty() || self.terms.iter().any(|t| t.variable != variable)
    }
}

struct Candidate {
    variable: usize,
    knot: f64,
    linear: bool,
    new_form: bool,
}

struct Search<'a> {
    x: &'a [Vec<f64>],
    order: &'a [Vec<usize>],
    /// (min span, end span, start span).
    spans: (usize, usize, usize),
    weighted: bool,
    penalty: f64,
}

impl Search<'_> {
    fn column(&self, t: &MarsTerm) -> Vec<f64> {
        self.x[t.variable].iter().map(|&x| t.eval(x)).collect()
    }

    /// A design of the intercept and `terms`, with case weights `w`.
    fn design(&self, terms: &[MarsTerm], w: &[f64]) -> Result<Design> {
        let n = w.len();
        let mut names = vec!["(Intercept)".to_string()];
        let mut columns = vec![vec![1.0; n]];
        for (j, t) in terms.iter().enumerate() {
            names.push(format!("t{j}"));
            columns.push(self.column(t));
        }
        Design::new(names, columns)?.with_weights(w.to_vec())
    }

    /// The best term to add to `basis` for the response `y` with residual
    /// `r`, by `earth`'s rules (`FindPredGivenParent` unweighted,
    /// `FindWeightedPredGivenParent` weighted).
    fn best(
        &self,
        basis: &Basis,
        r: &[f64],
        slots: usize,
        max_legal: f64,
        almost_zero: f64,
    ) -> Option<Candidate> {
        let w = basis.w;
        let n = w.len();
        let (min_span, end_span, start_span) = self.spans;
        let total: f64 = w.iter().sum();
        let mut best_delta = 0.0;
        let mut best = None;
        for (p, xp) in self.x.iter().enumerate() {
            let order = &self.order[p];
            let flag = basis.new_form_flag(p);
            // The linear candidate, orthogonalized on the basis.
            let mut q = Vec::new();
            let mut lin = 0.0;
            let mut rr = r.to_vec();
            // Unweighted, `earth` clears the flag when `x` is in the span;
            // weighted, it keeps it (the pair's lower hinge is then tested
            // in `Basis::add`) and its QR drops a collinear `x`.
            let mut new_form = flag && self.weighted;
            if flag {
                let mut c = xp.clone();
                let norm = dot(&c, &c, w);
                let ss = orthogonalize(&mut c, &basis.q, w);
                let enters = if self.weighted {
                    ss > 1e-16 * norm
                } else {
                    ss > MIN_BX_SOS
                };
                new_form |= enters;
                if enters {
                    let len = ss.sqrt();
                    c.iter_mut().for_each(|v| *v /= len);
                    let b = dot(&c, r, w);
                    lin = b * b;
                    rr.iter_mut().zip(&c).for_each(|(r, c)| *r -= b * c);
                    q.push(c);
                }
            }
            if !self.weighted && new_form && lin > best_delta {
                best_delta = lin;
                best = Some(Candidate {
                    variable: p,
                    knot: xp[order[0]],
                    linear: true,
                    new_form,
                });
            }
            // The columns a hinge is orthogonalized on, past the constant.
            let cols: Vec<&[f64]> = basis.q[1..]
                .iter()
                .chain(&q)
                .filter(|c| c.iter().any(|&v| v != 0.0))
                .map(|c| c.as_slice())
                .collect();
            let m = cols.len();
            let new_col = if new_form { slots + 1 } else { slots };
            let tol = if new_col < 15 { 0.01 } else { 1e-5 };
            // Running sums over the cases above the knot: T for u, H for
            // ⟨h, u⟩ with h = max(0, x − x0).
            let mut t_u = vec![0.0; m];
            let mut h_u = vec![0.0; m];
            let (mut t_r, mut h_r) = (0.0, 0.0);
            let (mut t_1, mut h_1, mut hh) = (0.0, 0.0, 0.0);
            let mut span = start_span as i64;
            let mut pair_delta = lin;
            let mut pair_case = None;
            if n < 2 {
                continue;
            }
            let mut i = n - 2;
            while i >= end_span {
                let ix1 = order[i + 1];
                let x0 = xp[order[i]];
                let delta = xp[ix1] - x0;
                let wi = w[ix1];
                t_1 += wi;
                hh += 2.0 * delta * h_1 + delta * delta * t_1;
                h_1 += delta * t_1;
                t_r += wi * rr[ix1];
                h_r += delta * t_r;
                for j in 0..m {
                    t_u[j] += wi * cols[j][ix1];
                    h_u[j] += delta * t_u[j];
                }
                let centered = hh - h_1 * h_1 / total;
                let counts = wi > 0.0 && (self.weighted || centered > 0.0);
                if counts {
                    span -= 1;
                }
                if counts && span == 0 {
                    span = min_span as i64;
                    let s = centered - h_u.iter().map(|h| h * h).sum::<f64>();
                    let hinge = if self.weighted {
                        if s > 1e-12 * hh { h_r * h_r / s } else { 0.0 }
                    } else if centered > 0.0 && s / centered > tol {
                        h_r * h_r / s
                    } else {
                        0.0
                    };
                    let d = lin + hinge;
                    let better = if self.weighted {
                        d > pair_delta + almost_zero
                    } else {
                        d > pair_delta && d < max_legal
                    };
                    if better {
                        pair_delta = d;
                        pair_case = Some(i);
                    }
                }
                if i == 0 {
                    break;
                }
                i -= 1;
            }
            match pair_case {
                Some(i) if pair_delta > best_delta => {
                    best_delta = pair_delta;
                    best = Some(Candidate {
                        variable: p,
                        knot: xp[order[i]],
                        linear: false,
                        new_form,
                    });
                }
                None if self.weighted && new_form && lin > best_delta => {
                    best_delta = lin;
                    best = Some(Candidate {
                        variable: p,
                        knot: xp[order[0]],
                        linear: true,
                        new_form,
                    });
                }
                _ => {}
            }
        }
        best
    }
}

/// Weighted cross products of the intercept and `columns` with `y`.
struct Gram {
    /// Row-major, intercept first.
    g: Vec<f64>,
    b: Vec<f64>,
    yy: f64,
    k: usize,
}

impl Gram {
    fn new(columns: &[Vec<f64>], y: &[f64], w: &[f64]) -> Self {
        let ones = vec![1.0; y.len()];
        let cols: Vec<&[f64]> = std::iter::once(ones.as_slice())
            .chain(columns.iter().map(|c| c.as_slice()))
            .collect();
        let k = cols.len();
        let mut g = vec![0.0; k * k];
        for a in 0..k {
            for c in a..k {
                let v = dot(cols[a], cols[c], w);
                g[a * k + c] = v;
                g[c * k + a] = v;
            }
        }
        let b = cols.iter().map(|c| dot(c, y, w)).collect();
        Self {
            g,
            b,
            yy: dot(y, y, w),
            k,
        }
    }

    /// Least squares on the intercept and the terms `keep` (indices into
    /// the columns): (coefficients, RSS).
    fn solve(&self, keep: &[usize]) -> Result<(Vec<f64>, f64)> {
        let idx: Vec<usize> = std::iter::once(0)
            .chain(keep.iter().map(|j| j + 1))
            .collect();
        let m = idx.len();
        let mut a = vec![0.0; m * m];
        for (r, &i) in idx.iter().enumerate() {
            for (c, &j) in idx.iter().enumerate() {
                a[r * m + c] = self.g[i * self.k + j];
            }
        }
        let rhs: Vec<f64> = idx.iter().map(|&i| self.b[i]).collect();
        let l = prospicio_math::linalg::cholesky(&a, m)
            .ok_or_else(|| Error::Data("the selected MARS terms are collinear".into()))?;
        let beta = prospicio_math::linalg::cholesky_solve(&l, &rhs);
        let fitted: f64 = beta.iter().zip(&rhs).map(|(b, r)| b * r).sum();
        Ok((beta, (self.yy - fitted).max(0.0)))
    }

    /// `earth`'s backward pass: drop the term whose removal least raises
    /// the RSS until only the intercept is left, then keep the subset with
    /// the lowest GCV (the smallest on ties).
    fn prune(&self, n: usize, penalty: f64) -> Vec<usize> {
        self.prune_with(n, penalty, |rss| rss)
    }

    /// [`prune`](Self::prune), scoring the GCV on `criterion(rss)`.
    fn prune_with(&self, n: usize, penalty: f64, criterion: impl Fn(f64) -> f64) -> Vec<usize> {
        let rss = |s: &[usize]| self.solve(s).map_or(f64::INFINITY, |(_, r)| r);
        let mut current: Vec<usize> = (0..self.k - 1).collect();
        let mut path = vec![(current.clone(), rss(&current))];
        while !current.is_empty() {
            let (drop, best) = (0..current.len())
                .map(|d| {
                    let mut s = current.clone();
                    s.remove(d);
                    (d, rss(&s))
                })
                .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
            current.remove(drop);
            path.push((current.clone(), best));
        }
        path.into_iter()
            .map(|(s, r)| {
                let g = gcv(criterion(r), s.len() + 1, n, penalty);
                (s, g)
            })
            .fold(
                (Vec::new(), f64::INFINITY),
                |a, b| if b.1 <= a.1 { b } else { a },
            )
            .0
    }
}

/// The terms MARS selected, with the fit on them.
#[derive(Debug, Clone)]
pub struct MarsFit {
    names: Vec<String>,
    terms: Vec<MarsTerm>,
    forward: Vec<MarsTerm>,
    steps: Vec<MarsStep>,
    termination: Termination,
    coefficients: Vec<f64>,
    rss: f64,
    gcv: f64,
    rsq: f64,
    grsq: f64,
    glm: Option<GlmFit>,
}

impl MarsFit {
    /// The predictors' names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The selected terms, in the order the forward pass added them,
    /// without the intercept.
    pub fn terms(&self) -> &[MarsTerm] {
        &self.terms
    }

    /// The forward pass's terms, before the backward pass, with any column
    /// the weighted basis does not need left out.
    pub fn forward_terms(&self) -> &[MarsTerm] {
        &self.forward
    }

    /// The forward pass, step by step.
    pub fn steps(&self) -> &[MarsStep] {
        &self.steps
    }

    /// Why the forward pass stopped.
    pub fn termination(&self) -> Termination {
        self.termination
    }

    /// The intercept and each selected term's coefficient (least squares,
    /// or the GLM's).
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The (weighted) residual sum of squares of the selected terms, or
    /// the GLM's deviance.
    pub fn rss(&self) -> f64 {
        self.rss
    }

    /// The GCV of the selected terms.
    pub fn gcv(&self) -> f64 {
        self.gcv
    }

    /// `1 − rss / rss₀`.
    pub fn rsq(&self) -> f64 {
        self.rsq
    }

    /// `1 − gcv / gcv₀`.
    pub fn grsq(&self) -> f64 {
        self.grsq
    }

    /// The GLM fitted on the selected terms, with [`Mars::glm`].
    pub fn glm_fit(&self) -> Option<&GlmFit> {
        self.glm.as_ref()
    }

    fn index(&self, name: &str) -> Result<usize> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| Error::Data(format!("no predictor named {name}")))
    }

    /// The sorted knots of `name`'s selected hinges, for [`Terms::hinges`].
    pub fn knots(&self, name: &str) -> Result<Vec<f64>> {
        let v = self.index(name)?;
        let mut k: Vec<f64> = self
            .terms
            .iter()
            .filter(|t| t.variable == v && t.direction != Direction::Linear)
            .map(|t| t.knot)
            .collect();
        k.sort_by(f64::total_cmp);
        k.dedup();
        Ok(k)
    }

    /// Whether `name` enters linearly besides its hinges: it has a linear
    /// term or a `max(0, k − x)`, which is `max(0, x − k) − x + k`.
    pub fn linear(&self, name: &str) -> Result<bool> {
        let v = self.index(name)?;
        Ok(self
            .terms
            .iter()
            .any(|t| t.variable == v && t.direction != Direction::Up))
    }

    /// `terms` with each selected predictor added, in the order of
    /// [`names`](Self::names): `x` when [`linear`](Self::linear) and
    /// hinges at its [`knots`](Self::knots). Their columns span the
    /// selected terms; without the backward pass the span is the same.
    pub fn add_terms(&self, mut terms: Terms) -> Terms {
        for name in &self.names {
            if self.linear(name).unwrap_or(false) {
                terms = terms.numeric(name);
            }
            let knots = self.knots(name).unwrap_or_default();
            if !knots.is_empty() {
                terms = terms.hinges(name, &knots);
            }
        }
        terms
    }

    /// The selected terms' columns at the predictor columns `x`, in the
    /// order of [`terms`](Self::terms).
    pub fn basis(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if x.len() != self.names.len() {
            return Err(Error::Data(format!(
                "{} predictor columns for {} names",
                x.len(),
                self.names.len()
            )));
        }
        Ok(self
            .terms
            .iter()
            .map(|t| x[t.variable].iter().map(|&v| t.eval(v)).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_models::Model;

    fn grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    /// A noise-free broken line is found at its knot, with no residual.
    #[test]
    fn a_broken_line_is_found_at_its_knot() {
        let x = grid(60);
        let y: Vec<f64> = x
            .iter()
            .map(|&v| 1.0 + 0.5 * v + 3.0 * (v - 31.0).max(0.0))
            .collect();
        let fit = Mars::new()
            .fit(&["x"], std::slice::from_ref(&x), &y, None)
            .unwrap();
        assert_eq!(fit.knots("x").unwrap(), vec![31.0]);
        assert!(fit.linear("x").unwrap());
        assert!(fit.rss() < 1e-18 * y.iter().map(|v| v * v).sum::<f64>());
        assert_eq!(fit.termination(), Termination::Rsq);
    }

    /// The `Terms` from `add_terms` span the selected terms: a least
    /// squares fit on them has the same RSS.
    #[test]
    fn add_terms_spans_the_selected_terms() {
        let n = 200;
        let x1: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 53) % 89) as f64 / 10.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                (x1[i] - 40.0).max(0.0)
                    + 2.0 * (6.0 - x2[i]).max(0.0)
                    + ((i * 7919) % 13) as f64 / 13.0
            })
            .collect();
        let fit = Mars::new()
            .threshold(0.0)
            .max_terms(9)
            .fit(&["a", "b"], &[x1.clone(), x2.clone()], &y, None)
            .unwrap();
        let frame = prospicio_models::Frame::new(vec![
            ("a".into(), prospicio_models::Column::Numeric(x1)),
            ("b".into(), prospicio_models::Column::Numeric(x2)),
        ])
        .unwrap();
        let design = fit
            .add_terms(Terms::new().intercept())
            .fit(&frame)
            .unwrap()
            .design(&frame)
            .unwrap();
        let ls = Glm::new(Family::Gaussian, Link::Identity)
            .fit(&design, &y)
            .unwrap();
        assert!((ls.deviance() - fit.rss()).abs() < 1e-8 * fit.rss());
    }

    /// The backward pass never keeps a subset with a higher GCV than the
    /// forward pass's.
    #[test]
    fn pruning_lowers_the_gcv() {
        let n = 300;
        let x: Vec<f64> = (0..n).map(|i| ((i * 97) % 151) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (x[i] - 70.0).max(0.0) + ((i * 7919) % 31) as f64)
            .collect();
        let full = Mars::new()
            .threshold(0.0)
            .max_terms(11)
            .fit(&["x"], std::slice::from_ref(&x), &y, None)
            .unwrap();
        let pruned = Mars::new()
            .threshold(0.0)
            .max_terms(11)
            .prune(true)
            .fit(&["x"], &[x], &y, None)
            .unwrap();
        assert!(pruned.gcv() <= full.gcv() + 1e-12);
        assert!(pruned.terms().len() <= full.terms().len());
    }

    /// The deviance search finds a Poisson rate's knot, and the deviance it
    /// reports is the GLM's on the selected terms.
    #[test]
    fn the_deviance_search_finds_a_poisson_knot() {
        let n = 400;
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 97) as f64).collect();
        let mu: Vec<f64> = x
            .iter()
            .map(|&v| (0.5 + 0.04 * (v - 50.0).max(0.0)).exp())
            .collect();
        let y: Vec<f64> = mu
            .iter()
            .enumerate()
            .map(|(i, m)| (m + ((i * 7919) % 7) as f64 / 3.0 - 1.0).round().max(0.0))
            .collect();
        let fit = Mars::new()
            .glm(Family::Poisson, Link::Log)
            .fit(&["x"], &[x], &y, None)
            .unwrap();
        let knots = fit.knots("x").unwrap();
        assert!(knots.iter().any(|&k| (k - 50.0).abs() <= 8.0), "{knots:?}");
        let glm = fit.glm_fit().unwrap();
        assert_eq!(fit.rss(), glm.deviance());
        assert!(fit.steps().windows(2).all(|s| s[1].rss < s[0].rss));
    }

    #[test]
    fn bad_input_is_an_error() {
        let x = grid(10);
        assert!(
            Mars::new()
                .fit(&["x", "y"], std::slice::from_ref(&x), &x, None)
                .is_err()
        );
        assert!(
            Mars::new()
                .fit(&["x"], std::slice::from_ref(&x), &[1.0; 10], None)
                .is_err()
        );
        assert!(
            Mars::new()
                .fit(&["x"], std::slice::from_ref(&x), &x, Some(&[1.0; 3]))
                .is_err()
        );
        assert!(
            Mars::new()
                .threshold(1.0)
                .fit(&["x"], std::slice::from_ref(&x), &x, None)
                .is_err()
        );
    }
}
