//! Reinsurer default on a tower's simulated years (#980,
//! `docs/design/reinsurer-default.md`): a [`Panel`] of [`Reinsurer`]s and
//! their signed lines on the tower's layers, applied to a tower's one-year
//! result ([`Tower::apply`], [`Tower::apply_aggregate`],
//! [`Tower::apply_annual`]) with [`Panel::apply`], or to a term's paths
//! ([`Tower::apply_term`]) with [`Panel::apply_term`], where a default is
//! final for the rest of the path and the collateral is used up.
//!
//! The model and the order of the random draws are documented on [`Panel`].

use prospicio_core::StreamRng;
use prospicio_math::special::{norm_cdf, norm_quantile};

use super::tower::fill_rows;
use super::*;

/// Year `i` draws from stream `DEFAULT_STREAM + i`, apart from the events'
/// streams (the year's own index, and `TIME_STREAM = 1 << 63` plus it).
const DEFAULT_STREAM: u64 = 3 << 62;

/// The largest `f64` below 1, the cap on a year's rank `U`.
const BELOW_ONE: f64 = 1.0 - f64::EPSILON / 2.0;

/// How far a layer's lines may sum from 1.
const LINES_TOLERANCE: f64 = 1e-9;

/// The kinds a panel adds to a tower's result.
const CREDIT_LOSS: &str = "credit_loss";
const RECOVERED: &str = "recovered";
const NET_AFTER_DEFAULT: &str = "net_after_default";

/// A reinsurer on a [`Panel`]: its one-year probability of default, the
/// recovery rate on what it owes when it defaults, the collateral held
/// against it and its loading on the systematic factor.
///
/// ```
/// use prospicio_aggregate::Reinsurer;
///
/// let r = Reinsurer::new("Re A", 0.002, 0.4)
///     .unwrap()
///     .collateral(1e6)
///     .unwrap()
///     .correlation(0.3)
///     .unwrap();
/// assert_eq!(r.collateral, 1e6);
/// assert!(Reinsurer::new("Re B", 1.5, 0.4).is_err());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Reinsurer {
    /// The reinsurer's name, unique on a panel.
    pub name: String,
    /// One-year probability of default, in [0, 1].
    pub probability: f64,
    /// Share of the uncollateralised exposure recovered on default, in [0, 1].
    pub recovery: f64,
    /// Collateral held against this reinsurer, across all its layers, >= 0.
    pub collateral: f64,
    /// Loading on the systematic factor, in [0, 1); 0 is independent default.
    pub correlation: f64,
}

impl Reinsurer {
    /// A reinsurer named `name` with a one-year `probability` of default
    /// and a `recovery` rate, both in [0, 1], no collateral and independent
    /// default (correlation 0).
    pub fn new(name: &str, probability: f64, recovery: f64) -> Result<Self> {
        let r = Self {
            name: name.to_string(),
            probability,
            recovery,
            collateral: 0.0,
            correlation: 0.0,
        };
        r.check()?;
        Ok(r)
    }

    /// Collateral held against this reinsurer (a trust fund or letter of
    /// credit), netted once across all its layers: finite and non-negative.
    pub fn collateral(mut self, amount: f64) -> Result<Self> {
        self.collateral = amount;
        self.check()?;
        Ok(self)
    }

    /// The loading `rho` on the systematic factor, in [0, 1): 0 is
    /// independent default, and a larger `rho` moves the defaults into the
    /// years with the largest driver without changing how often the
    /// reinsurer defaults.
    pub fn correlation(mut self, rho: f64) -> Result<Self> {
        self.correlation = rho;
        self.check()?;
        Ok(self)
    }

    fn check(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::Data("a reinsurer needs a name".into()));
        }
        let unit = |x: f64| x.is_finite() && (0.0..=1.0).contains(&x);
        if !unit(self.probability) {
            return Err(invalid(
                "probability",
                self.probability,
                "must be in [0, 1]",
            ));
        }
        if !unit(self.recovery) {
            return Err(invalid("recovery", self.recovery, "must be in [0, 1]"));
        }
        if !(self.collateral.is_finite() && self.collateral >= 0.0) {
            return Err(invalid(
                "collateral",
                self.collateral,
                "must be finite and non-negative",
            ));
        }
        if !(self.correlation.is_finite() && (0.0..1.0).contains(&self.correlation)) {
            return Err(invalid(
                "correlation",
                self.correlation,
                "must be in [0, 1)",
            ));
        }
        Ok(())
    }

    /// The probability of default given the factor `z`, with `q =
    /// Phi^{-1}(p)`: exactly `p` when `p` is 0 or 1 or the correlation is
    /// 0, so no infinite quantile or factor makes it NaN.
    fn given(&self, q: f64, z: f64) -> f64 {
        let (p, rho) = (self.probability, self.correlation);
        if p == 0.0 || p == 1.0 || rho == 0.0 {
            return p;
        }
        norm_cdf((q - rho.sqrt() * z) / (1.0 - rho).sqrt())
    }
}

/// One signed line: reinsurer `reinsurer` (its index on the panel) writes
/// `line` of layer `layer`'s ceded loss.
#[derive(Debug, Clone, PartialEq)]
struct Line {
    reinsurer: usize,
    layer: String,
    line: f64,
}

/// The reinsurers on a tower and their signed lines, applied to the
/// tower's simulated years to draw which reinsurers default
/// (`docs/design/reinsurer-default.md`).
///
/// A line is a share of the layer's ceded loss (after
/// [`Layer::share`](crate::Layer)), and the lines of a layer the panel
/// names must sum to 1 (to 1e-9), checked when the panel is applied; a
/// layer the panel does not name carries no credit risk.
///
/// # Model
///
/// Reinsurer `r`'s exposure in year `i` is `E = sum over l of s_rl X_l`,
/// with `X_l` layer `l`'s ceded loss and `s_rl` its line; if it defaults,
/// the cedant loses `(1 - R_r) (E - C_r)^+`, with `R_r` its recovery rate
/// and `C_r` its collateral. Defaults follow Vasicek's one-factor model:
/// given the year's factor `z`, reinsurer `r` defaults with probability
/// `p_r(z) = Phi((Phi^{-1}(p_r) - sqrt(rho_r) z) / sqrt(1 - rho_r))`, and
/// the factor is `z = -Phi^{-1}(U_i)`, where `U_i` is year `i`'s rank by
/// its driver (the gross loss, or [`Panel::driver`]), so the worst years
/// have the highest probability of default (wrong-way risk). With `b_i`
/// years strictly below year `i`'s driver and `t_i` tied with it (itself
/// included), `U_i = (b_i + v_i t_i) / n`, so ties are broken at random
/// and `U` is uniform over the years.
///
/// # Random numbers
///
/// Year `i` draws from its own stream, `StreamRng::new(seed,
/// DEFAULT_STREAM + i)` with `DEFAULT_STREAM = 3 << 62`, apart from the
/// events' streams (`i` for the losses, `1 << 63` plus `i` for event
/// times) even when the caller passes the events' seed. The year draws, in
/// this order, `v_i` for the rank, then one uniform per reinsurer in panel
/// order, drawn whether or not the reinsurer has an exposure that year;
/// reinsurer `r` defaults when its uniform is below `p_r(z)`. The ranks
/// are computed once before the years, and each year depends only on its
/// index and its rank, so the result has the same bits on any number of
/// threads. Over a term ([`Panel::apply_term`]) the stream is the path's,
/// `DEFAULT_STREAM + j` for path `j`, and its years draw in order, each as
/// a one-year result's year does, a reinsurer already in default drawing
/// too. The result keeps the tower result's provenance (so `join` sees
/// that it shares the events' streams) and adds the parameters
/// `default_seed`, `default_streams` and `default_driver` (and
/// `default_term_years` over a term).
///
/// # Example
///
/// ```
/// use prospicio_aggregate::{Layer, Panel, Reinsurer, Tower};
/// use prospicio_prob::KeyValue;
///
/// let tower = Tower::new(vec![Layer::xol("10x10", 10.0, 10.0).unwrap()]).unwrap();
/// let years = tower.apply_annual(&[5.0, 15.0, 30.0]).unwrap();
/// // Re A certainly defaults and pays back 40% of what it owes.
/// let panel = Panel::new(vec![
///     Reinsurer::new("Re A", 1.0, 0.4).unwrap(),
///     Reinsurer::new("Re B", 0.0, 0.4).unwrap(),
/// ])
/// .unwrap()
/// .line("Re A", "10x10", 0.5)
/// .unwrap()
/// .line("Re B", "10x10", 0.5)
/// .unwrap();
/// let after = panel.apply(&years, 7).unwrap();
/// let key = vec![KeyValue::from("credit_loss"), KeyValue::from("10x10")];
/// let credit = prospicio_prob::Empirical::draws(&after.marginal(&key).unwrap()).to_vec();
/// assert_eq!(credit, [0.0, 1.5, 3.0]); // 60% of half the ceded loss
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Panel {
    reinsurers: Vec<Reinsurer>,
    lines: Vec<Line>,
    driver: Option<Vec<f64>>,
}

impl Panel {
    /// A panel of `reinsurers`, with no lines yet: fails if there are none,
    /// two share a name, or one has a field out of range.
    pub fn new(reinsurers: Vec<Reinsurer>) -> Result<Self> {
        if reinsurers.is_empty() {
            return Err(Error::Data("a panel needs at least one reinsurer".into()));
        }
        for (k, r) in reinsurers.iter().enumerate() {
            r.check()?;
            if reinsurers[..k].iter().any(|o| o.name == r.name) {
                return Err(Error::Data(format!(
                    "two reinsurers are named {:?}: names must be unique",
                    r.name
                )));
            }
        }
        Ok(Self {
            reinsurers,
            lines: Vec::new(),
            driver: None,
        })
    }

    /// Reinsurer `reinsurer` writes `line`, in (0, 1], of layer `layer`'s
    /// ceded loss. Fails on a reinsurer not on the panel or a line given
    /// twice; whether `layer` is in the tower, and whether its lines sum to
    /// 1, is checked when the panel is applied.
    pub fn line(mut self, reinsurer: &str, layer: &str, line: f64) -> Result<Self> {
        let Some(r) = self.reinsurers.iter().position(|o| o.name == reinsurer) else {
            return Err(Error::Data(format!(
                "no reinsurer {reinsurer:?} on the panel"
            )));
        };
        if layer.is_empty() {
            return Err(Error::Data("a line needs a layer name".into()));
        }
        if !(line.is_finite() && line > 0.0 && line <= 1.0) {
            return Err(invalid("line", line, "must be in (0, 1]"));
        }
        if self
            .lines
            .iter()
            .any(|l| l.reinsurer == r && l.layer == layer)
        {
            return Err(Error::Data(format!(
                "{reinsurer:?} already has a line on {layer:?}"
            )));
        }
        self.lines.push(Line {
            reinsurer: r,
            layer: layer.to_string(),
            line,
        });
        Ok(self)
    }

    /// Ranks the years by `by_year` instead of the gross loss (an industry
    /// loss, say), one finite value per simulated year in the order of the
    /// tower's result; over a term ([`Panel::apply_term`]), one value per
    /// path-year, path-major (year `t` of path `j` at `j * n_years + t`,
    /// `t` from 0). Its length is checked when the panel is applied.
    pub fn driver(mut self, by_year: Vec<f64>) -> Result<Self> {
        if by_year.is_empty() {
            return Err(Error::Data("a driver needs one value per year".into()));
        }
        if let Some(&bad) = by_year.iter().find(|x| !x.is_finite()) {
            return Err(invalid("driver", bad, "must all be finite"));
        }
        self.driver = Some(by_year);
        Ok(self)
    }

    /// The panel's reinsurers, in panel order (the order they draw in).
    pub fn reinsurers(&self) -> &[Reinsurer] {
        &self.reinsurers
    }

    /// A tower's one-year result ([`Tower::apply`],
    /// [`Tower::apply_aggregate`], [`Tower::apply_annual`], labelled or
    /// not) after reinsurer default, with `seed` for the default draws.
    ///
    /// The result has the input's components, copied bit for bit, then
    /// `("credit_loss", l)` for every layer `l` of the input, in the
    /// input's order (zero on a layer the panel does not name), then
    /// `("recovered", l)`, layer `l`'s ceded loss less its credit loss
    /// (the ceded loss itself on a layer the panel does not name), and
    /// `("net_after_default", "retained")`, the net loss plus the year's
    /// credit loss. A reinsurer's credit loss is split over its layers in
    /// proportion to its exposure on each, so the layers' credit losses sum
    /// to the reinsurers'. On a labelled result the exposures sum each
    /// layer's ceded loss over the label combinations, and the new
    /// components get `""` in every label dimension: collateral nets
    /// across labels, so the credit loss has no split by label.
    ///
    /// Fails on a result that is no tower's (dims other than `["kind",
    /// "layer", ...]`, no ceded, net or, without a driver, gross
    /// components), one a panel was already applied to, a term's paths
    /// (dimension `"year"` with integer years; use
    /// [`apply_term`](Self::apply_term)), a line on a layer the result does
    /// not have, a named layer whose lines do not sum to 1, or a driver of
    /// the wrong length.
    pub fn apply(
        &self,
        tower: &PredictiveDistribution,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        Overlay::new(self, tower, seed, None)?.after_default()
    }

    /// A term's paths ([`Tower::apply_term`],
    /// [`Tower::apply_aggregate_term`], [`Tower::apply_timeline`]) of
    /// `n_years` years after reinsurer default, with `seed` for the default
    /// draws.
    ///
    /// Each row is a path, and a default is final: a reinsurer that
    /// defaults in year `t` of a path is in default in every later year of
    /// it, and each of those years' exposure is lost at `1 - R` too. The
    /// collateral is held once per reinsurer and path and is used up: in
    /// each year in default the collateral still held is set against that
    /// year's exposure, and what it covered is gone for the later years.
    /// Each path-year's factor comes from its driver ranked among all
    /// `n_paths * n_years` path-years (the gross loss, or a driver given
    /// path-major: year `t` of path `j` at `j * n_years + t`). Path `j`
    /// draws from stream `DEFAULT_STREAM + j`, its years in order, each
    /// year `v` for the rank then one uniform per reinsurer in panel order,
    /// a reinsurer already in default drawing too; so one year's term gives
    /// the same draws as [`apply`](Self::apply) on that year.
    ///
    /// The result has the input's components, copied bit for bit, then for
    /// each year in turn the components [`apply`](Self::apply) adds, with
    /// the year (`1..=n_years`) as the last key, as the term's own
    /// components have it. With a constant probability `p` and
    /// independent default (`rho = 0`), a reinsurer is in default by year
    /// `t` with probability `1 - (1 - p)^t`. With `rho > 0` this holds only
    /// when the years' drivers are independent and identically
    /// distributed: the ranks are taken over all path-years together, so a
    /// driver that trends or changes in spread from year to year moves
    /// defaults between years.
    ///
    /// Fails as [`apply`](Self::apply) does, and on a result whose dims
    /// are not `["kind", "layer", "year"]` or whose years are not exactly
    /// `1..=n_years` (a one-year result among them), and on a driver
    /// without `n_paths * n_years` values.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Panel, Reinsurer, Tower};
    /// use prospicio_prob::KeyValue;
    ///
    /// let tower = Tower::new(vec![Layer::xol("10x10", 10.0, 10.0).unwrap()]).unwrap();
    /// // One path of two years, ceding 5 and then 10.
    /// let term = tower
    ///     .apply_term(&[
    ///         EventSet::from_years(vec![vec![15.0]], 0).unwrap(),
    ///         EventSet::from_years(vec![vec![25.0]], 0).unwrap(),
    ///     ])
    ///     .unwrap();
    /// // Re A certainly defaults, recovers nothing and has collateral of 8:
    /// // year 1 uses 5 of it, year 2 the 3 left.
    /// let panel = Panel::new(vec![
    ///     Reinsurer::new("Re A", 1.0, 0.0).unwrap().collateral(8.0).unwrap(),
    /// ])
    /// .unwrap()
    /// .line("Re A", "10x10", 1.0)
    /// .unwrap();
    /// let after = panel.apply_term(&term, 2, 7).unwrap();
    /// let credit = |year: i64| {
    ///     let key = vec![KeyValue::from("credit_loss"), KeyValue::from("10x10"), KeyValue::Int(year)];
    ///     prospicio_prob::Empirical::draws(&after.marginal(&key).unwrap()).to_vec()
    /// };
    /// assert_eq!(credit(1), [0.0]);
    /// assert_eq!(credit(2), [7.0]);
    /// ```
    pub fn apply_term(
        &self,
        tower: &PredictiveDistribution,
        n_years: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        Overlay::new(self, tower, seed, Some(n_years))?.after_default()
    }

    /// Each reinsurer's credit loss, exposure and default in each year,
    /// from the same draws as [`apply`](Self::apply) with the same `seed`:
    /// dims `["kind", "reinsurer"]`, with `("credit_loss", r)` for every
    /// reinsurer `r` in panel order, then `("exposure", r)`, its exposure
    /// after collateral `(E - C)^+`, then `("default", r)`, 1 in a year it
    /// defaults and 0 otherwise. Fails as `apply` does.
    pub fn by_reinsurer(
        &self,
        tower: &PredictiveDistribution,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        Overlay::new(self, tower, seed, None)?.by_reinsurer()
    }

    /// [`by_reinsurer`](Self::by_reinsurer) over a term's paths, from the
    /// same draws as [`apply_term`](Self::apply_term) with the same `seed`:
    /// dims `["kind", "reinsurer", "year"]`, `by_reinsurer`'s components
    /// for each year in turn with the year as the last key. `("exposure",
    /// r, t)` is the exposure after the collateral still held at the start
    /// of year `t`, and `("default", r, t)` is 1 in every year from the one
    /// it defaults in. Fails as `apply_term` does.
    pub fn by_reinsurer_term(
        &self,
        tower: &PredictiveDistribution,
        n_years: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        Overlay::new(self, tower, seed, Some(n_years))?.by_reinsurer()
    }

    /// The provenance parameter describing reinsurer `r` and its lines.
    fn terms(&self, r: usize) -> String {
        let re = &self.reinsurers[r];
        let mut text = format!(
            "probability {}, recovery {}, collateral {}, correlation {}",
            re.probability, re.recovery, re.collateral, re.correlation
        );
        for l in self.lines.iter().filter(|l| l.reinsurer == r) {
            text += &format!(", line {} on {}", l.line, l.layer);
        }
        text
    }
}

/// One path's draws and amounts (a one-year result's year is a path of one
/// year), reused from path to path; every field is reset or refilled at
/// the start of each path.
#[derive(Debug, Default)]
struct PathDraws {
    /// Each year's ceded loss by layer, summed over the label combinations,
    /// at `t * n_layers + l`.
    x: Vec<f64>,
    /// Each year's exposure `E` by reinsurer, before collateral, at
    /// `t * n_reinsurers + r`.
    exposure: Vec<f64>,
    /// Each year's exposure after the collateral still held, by reinsurer.
    uncovered: Vec<f64>,
    /// Whether each reinsurer is in default in each year: from its first
    /// default on, for the rest of the path.
    default: Vec<bool>,
    /// Each reinsurer's credit loss in each year, 0 when not in default.
    credit: Vec<f64>,
    /// Whether each reinsurer has defaulted so far on the path.
    in_default: Vec<bool>,
    /// The collateral each reinsurer still has on the path.
    left: Vec<f64>,
    /// One year's credit loss by layer.
    by_layer: Vec<f64>,
}

/// What a panel reads from a tower's one-year result or term's paths,
/// checked, and the path-years' ranks.
struct Overlay<'a> {
    panel: &'a Panel,
    tower: &'a PredictiveDistribution,
    seed: u64,
    /// `Some(T)` for a term's paths of `T` years, `None` for a one-year
    /// result (a path of one year).
    term: Option<usize>,
    /// The input's layers, in the order of their first ceded component.
    layers: Vec<KeyValue>,
    /// Each year's ceded components by layer (one per label combination),
    /// `ceded[t][l]`.
    ceded: Vec<Vec<Vec<usize>>>,
    /// Each year's net components.
    net: Vec<Vec<usize>>,
    /// Each reinsurer's lines, as `(layer index, line)`.
    lines: Vec<Vec<(usize, f64)>>,
    /// `Phi^{-1}` of each reinsurer's probability of default.
    quantiles: Vec<f64>,
    /// Each path-year's `(b, t)`, path-major: the path-years strictly
    /// below its driver, and those tied with it, itself included.
    ranks: Vec<(usize, usize)>,
}

impl<'a> Overlay<'a> {
    fn new(
        panel: &'a Panel,
        tower: &'a PredictiveDistribution,
        seed: u64,
        term: Option<usize>,
    ) -> Result<Self> {
        let dims = tower.dims();
        if dims.len() < 2 || dims[0] != "kind" || dims[1] != "layer" {
            return Err(Error::Data(format!(
                "a panel applies to a tower's result, with dims [\"kind\", \"layer\", ...], \
                 not {dims:?}"
            )));
        }
        let year_dim = dims.iter().position(|d| d == "year");
        let has_int_years = year_dim.is_some_and(|t| {
            tower
                .components()
                .iter()
                .any(|k| matches!(k[t], KeyValue::Int(_)))
        });
        let n_years = match term {
            None if has_int_years => {
                return Err(Error::Data(
                    "this looks like a term's paths (apply_term): Panel::apply takes a one-year \
                     result; use Panel::apply_term for a term"
                        .into(),
                ));
            }
            None => 1,
            Some(0) => return Err(Error::Data("a term needs at least one year".into())),
            Some(_) if !has_int_years => {
                return Err(Error::Data(
                    "Panel::apply_term takes a term's paths (Tower::apply_term), with dims \
                     [\"kind\", \"layer\", \"year\"] and integer years; use Panel::apply for a \
                     one-year result"
                        .into(),
                ));
            }
            Some(_) if dims.len() != 3 || dims[2] != "year" => {
                return Err(Error::Data(format!(
                    "a term's paths have dims [\"kind\", \"layer\", \"year\"], not {dims:?}"
                )));
            }
            Some(n) => n,
        };
        // Each year needs at least one component, so a term longer than the
        // result's component count cannot match it; refusing it here also
        // keeps a huge `n_years` from reaching the allocations below.
        if term.is_some() && n_years > tower.n_components() {
            return Err(Error::Data(format!(
                "a term of {n_years} years needs components in every year, but the result \
                 has only {} components",
                tower.n_components()
            )));
        }
        // The year index of each component: 0 for a one-year result.
        let mut seen = vec![false; n_years];
        let mut year_of = Vec::with_capacity(tower.n_components());
        for key in tower.components() {
            if term.is_none() {
                year_of.push(0);
                continue;
            }
            match key[2] {
                KeyValue::Int(y) if y >= 1 && (y as u64) <= n_years as u64 => {
                    seen[y as usize - 1] = true;
                    year_of.push(y as usize - 1);
                }
                KeyValue::Int(y) => {
                    return Err(Error::Data(format!(
                        "the term has a component in year {y}, outside 1..={n_years}"
                    )));
                }
                ref y => {
                    return Err(Error::Data(format!(
                        "the term has a component whose year {y} is not an integer"
                    )));
                }
            }
        }
        if term.is_some()
            && let Some(t) = seen.iter().position(|s| !s)
        {
            return Err(Error::Data(format!(
                "the term has no components in year {} of the {n_years} asked for",
                t + 1
            )));
        }
        let mut layers: Vec<KeyValue> = Vec::new();
        let mut ceded_at: Vec<(usize, usize, usize)> = Vec::new();
        let mut gross = vec![Vec::new(); n_years];
        let mut net = vec![Vec::new(); n_years];
        for (j, key) in tower.components().iter().enumerate() {
            let t = year_of[j];
            match text(&key[0]) {
                Some("ceded") => {
                    let l = match layers.iter().position(|l| *l == key[1]) {
                        Some(l) => l,
                        None => {
                            layers.push(key[1].clone());
                            layers.len() - 1
                        }
                    };
                    ceded_at.push((t, l, j));
                }
                Some("gross") => gross[t].push(j),
                Some("net") => net[t].push(j),
                Some(kind @ (CREDIT_LOSS | RECOVERED | NET_AFTER_DEFAULT)) => {
                    return Err(Error::Data(format!(
                        "the result already has {kind:?} components: a panel applies once, \
                         to a tower's result"
                    )));
                }
                _ => {}
            }
        }
        if ceded_at.is_empty() || net.iter().any(Vec::is_empty) {
            return Err(Error::Data(
                "a panel needs the tower's ceded and net components".into(),
            ));
        }
        if panel.driver.is_none() && gross.iter().any(Vec::is_empty) {
            return Err(Error::Data(
                "a panel ranks the years by the gross loss: the result has no gross \
                 component (give a driver with Panel::driver)"
                    .into(),
            ));
        }
        let mut ceded = vec![vec![Vec::new(); layers.len()]; n_years];
        for (t, l, j) in ceded_at {
            ceded[t][l].push(j);
        }
        // Each reinsurer's lines by layer index, and each named layer's
        // total.
        let mut lines = vec![Vec::new(); panel.reinsurers.len()];
        let mut totals = vec![0.0; layers.len()];
        for line in &panel.lines {
            let Some(l) = layers.iter().position(|k| text(k) == Some(&line.layer)) else {
                return Err(Error::Data(format!(
                    "the panel has a line on {:?}, which is not a layer of the tower's result",
                    line.layer
                )));
            };
            lines[line.reinsurer].push((l, line.line));
            totals[l] += line.line;
        }
        for (l, &total) in totals.iter().enumerate() {
            if total != 0.0 && (total - 1.0).abs() > LINES_TOLERANCE {
                return Err(Error::Data(format!(
                    "the lines on {} sum to {total}: a named layer's lines must sum to 1",
                    layers[l]
                )));
            }
        }
        let n = tower.n_sims() * n_years;
        let ranks = match &panel.driver {
            Some(d) if d.len() != n => {
                return Err(Error::Data(if term.is_some() {
                    format!(
                        "the driver has {} values for {} paths of {n_years} years: give one per \
                         path-year, path-major",
                        d.len(),
                        tower.n_sims()
                    )
                } else {
                    format!("the driver has {} values for {n} simulated years", d.len())
                }));
            }
            Some(d) => ranks(d),
            None => {
                let mut by_year = Vec::with_capacity(n);
                for j in 0..tower.n_sims() {
                    let row = tower.row(j).expect("a row of the input");
                    by_year.extend(gross.iter().map(|g| g.iter().map(|&c| row[c]).sum::<f64>()));
                }
                ranks(&by_year)
            }
        };
        Ok(Self {
            panel,
            tower,
            seed,
            term,
            layers,
            ceded,
            net,
            lines,
            quantiles: panel
                .reinsurers
                .iter()
                .map(|r| norm_quantile(r.probability))
                .collect(),
            ranks,
        })
    }

    /// The number of years in a path: 1 for a one-year result.
    fn n_years(&self) -> usize {
        self.term.unwrap_or(1)
    }

    /// A new component's key: `kind` and `name`, then `""` in every label
    /// dimension of a one-year result, or year `t` (from 0) of a term.
    fn key(&self, kind: &str, name: &KeyValue, t: usize) -> ComponentKey {
        let mut key = vec![KeyValue::from(kind), name.clone()];
        match self.term {
            Some(_) => key.push(KeyValue::Int(t as i64 + 1)),
            None => key.extend((2..self.tower.dims().len()).map(|_| KeyValue::from(""))),
        }
        key
    }

    /// Path `j`'s ceded losses by layer, the reinsurers' exposures and
    /// whether each is in default, year by year, into `w`.
    fn path(&self, j: usize, w: &mut PathDraws) {
        let row = self.tower.row(j).expect("a row of the input");
        let reinsurers = &self.panel.reinsurers;
        let (n_years, n_layers) = (self.n_years(), self.layers.len());
        w.x.clear();
        for ceded in &self.ceded {
            w.x.extend(
                ceded
                    .iter()
                    .map(|cols| cols.iter().map(|&c| row[c]).sum::<f64>()),
            );
        }
        w.in_default.clear();
        w.in_default.resize(reinsurers.len(), false);
        w.left.clear();
        w.left.extend(reinsurers.iter().map(|r| r.collateral));
        w.exposure.clear();
        w.uncovered.clear();
        w.default.clear();
        w.credit.clear();
        let mut rng = StreamRng::new(self.seed, DEFAULT_STREAM + j as u64);
        for t in 0..n_years {
            let x = &w.x[t * n_layers..(t + 1) * n_layers];
            let (b, tied) = self.ranks[j * n_years + t];
            let v = rng.next_open01();
            // `next_open01` can return exactly 1.0 (`k + 0.5` rounds up for
            // `k >= 2^52`), and `(b + v t) / n` can round to 1: keep `U`
            // below 1 so the factor stays finite.
            let u = ((b as f64 + v * tied as f64) / self.ranks.len() as f64).min(BELOW_ONE);
            let z = -norm_quantile(u);
            for (r, re) in reinsurers.iter().enumerate() {
                // Every reinsurer draws, in default or not, so the draws
                // do not depend on the path's history.
                let draw = rng.next_open01();
                // A probability of 1 defaults even on a draw of exactly 1.0.
                let pz = re.given(self.quantiles[r], z);
                let defaults = w.in_default[r] || pz >= 1.0 || draw < pz;
                let e: f64 = self.lines[r].iter().map(|&(l, s)| s * x[l]).sum();
                let held = w.left[r];
                let uncovered = (e - held).max(0.0);
                w.exposure.push(e);
                w.uncovered.push(uncovered);
                w.default.push(defaults);
                if defaults {
                    w.in_default[r] = true;
                    // The collateral that covered this year's exposure is
                    // gone for the later years.
                    w.left[r] = (held - e.max(0.0)).max(0.0);
                    w.credit.push((1.0 - re.recovery) * uncovered);
                } else {
                    w.credit.push(0.0);
                }
            }
        }
    }

    /// The input with each year's credit loss and recovered by layer and
    /// net after default added ([`Panel::apply`], [`Panel::apply_term`]).
    fn after_default(&self) -> Result<PredictiveDistribution> {
        let tower = self.tower;
        let (n_years, n_layers) = (self.n_years(), self.layers.len());
        let n_reinsurers = self.panel.reinsurers.len();
        let mut components = tower.components().to_vec();
        for t in 0..n_years {
            for kind in [CREDIT_LOSS, RECOVERED] {
                components.extend(self.layers.iter().map(|l| self.key(kind, l, t)));
            }
            components.push(self.key(NET_AFTER_DEFAULT, &KeyValue::from("retained"), t));
        }
        let draws = fill_rows(
            tower.n_sims(),
            components.len(),
            |w: &mut PathDraws, row, j| {
                self.path(j, w);
                let input = tower.row(j).expect("a row of the input");
                row.extend_from_slice(input);
                for t in 0..n_years {
                    let x = &w.x[t * n_layers..(t + 1) * n_layers];
                    let exposure = &w.exposure[t * n_reinsurers..(t + 1) * n_reinsurers];
                    let credit = &w.credit[t * n_reinsurers..(t + 1) * n_reinsurers];
                    // Each reinsurer's credit loss over its layers, in
                    // proportion to its exposure on each.
                    w.by_layer.clear();
                    w.by_layer.resize(n_layers, 0.0);
                    for (r, lines) in self.lines.iter().enumerate() {
                        let (e, c) = (exposure[r], credit[r]);
                        if c == 0.0 || e <= 0.0 {
                            continue;
                        }
                        for &(l, s) in lines {
                            w.by_layer[l] += c * (s * x[l] / e);
                        }
                    }
                    row.extend_from_slice(&w.by_layer);
                    row.extend(x.iter().zip(&w.by_layer).map(|(x, c)| x - c));
                    let net: f64 = self.net[t].iter().map(|&c| input[c]).sum();
                    row.push(net + credit.iter().sum::<f64>());
                }
            },
        );
        PredictiveDistribution::from_draws(
            tower.dims().to_vec(),
            components,
            draws,
            self.provenance(),
        )
    }

    /// Each reinsurer's credit loss, exposure after collateral and default
    /// by year ([`Panel::by_reinsurer`], [`Panel::by_reinsurer_term`]).
    fn by_reinsurer(&self) -> Result<PredictiveDistribution> {
        let reinsurers = &self.panel.reinsurers;
        let (n_years, n_reinsurers) = (self.n_years(), reinsurers.len());
        let mut components = Vec::new();
        for t in 0..n_years {
            for kind in [CREDIT_LOSS, "exposure", "default"] {
                components.extend(reinsurers.iter().map(|r| {
                    let mut key = vec![KeyValue::from(kind), KeyValue::from(r.name.as_str())];
                    if self.term.is_some() {
                        key.push(KeyValue::Int(t as i64 + 1));
                    }
                    key
                }));
            }
        }
        let draws = fill_rows(
            self.tower.n_sims(),
            components.len(),
            |w: &mut PathDraws, row, j| {
                self.path(j, w);
                for t in 0..n_years {
                    let year = t * n_reinsurers..(t + 1) * n_reinsurers;
                    row.extend_from_slice(&w.credit[year.clone()]);
                    row.extend_from_slice(&w.uncovered[year.clone()]);
                    row.extend(w.default[year].iter().map(|&d| if d { 1.0 } else { 0.0 }));
                }
            },
        );
        let mut dims = vec!["kind".into(), "reinsurer".into()];
        if self.term.is_some() {
            dims.push("year".into());
        }
        PredictiveDistribution::from_draws(dims, components, draws, self.provenance())
    }

    /// The tower result's provenance with the panel's seed, stream block,
    /// driver and reinsurers.
    fn provenance(&self) -> Provenance {
        let mut p = self
            .tower
            .provenance()
            .clone()
            .param("default_seed", self.seed)
            .param(
                "default_streams",
                if self.term.is_some() {
                    "3 << 62 plus the path"
                } else {
                    "3 << 62 plus the year"
                },
            )
            .param(
                "default_driver",
                match (&self.panel.driver, self.term) {
                    (Some(_), None) => "given by year",
                    (Some(_), Some(_)) => "given by path-year, path-major",
                    (None, _) => "gross loss",
                },
            );
        if let Some(n) = self.term {
            p = p.param("default_term_years", n.to_string());
        }
        for (r, re) in self.panel.reinsurers.iter().enumerate() {
            p = p.param(format!("reinsurer:{}", re.name), self.panel.terms(r));
        }
        p
    }
}

/// The text of a key value, `None` for a number or a period.
fn text(k: &KeyValue) -> Option<&str> {
    match k {
        KeyValue::Text(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Each year's `(b, t)` by `driver`: the years strictly below it, and the
/// years tied with it (itself included; `-0.0` ties with `0.0`).
fn ranks(driver: &[f64]) -> Vec<(usize, usize)> {
    let n = driver.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| driver[a].total_cmp(&driver[b]));
    let mut out = vec![(0, 0); n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && driver[order[end]] == driver[order[start]] {
            end += 1;
        }
        for &k in &order[start..end] {
            out[k] = (start, end - start);
        }
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventSet, Layer, Tower};
    use prospicio_prob::Empirical;

    fn column(pd: &PredictiveDistribution, key: &[&str]) -> Vec<f64> {
        let key: ComponentKey = key.iter().map(|&k| KeyValue::from(k)).collect();
        Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
    }

    fn tower() -> Tower {
        Tower::new(vec![
            Layer::xol("a", 10.0, 10.0).unwrap(),
            Layer::xol("b", 20.0, 20.0).unwrap(),
        ])
        .unwrap()
    }

    /// Annual losses from 0 to 60, many of them 0.
    fn losses(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    (i * 37 % 61) as f64
                }
            })
            .collect()
    }

    fn panel(p: f64, recovery: f64, collateral: f64, rho: f64) -> Panel {
        Panel::new(vec![
            Reinsurer::new("X", p, recovery)
                .unwrap()
                .collateral(collateral)
                .unwrap()
                .correlation(rho)
                .unwrap(),
            Reinsurer::new("Y", 0.0, 0.5).unwrap(),
        ])
        .unwrap()
        .line("X", "a", 0.6)
        .unwrap()
        .line("Y", "a", 0.4)
        .unwrap()
        .line("X", "b", 1.0)
        .unwrap()
    }

    #[test]
    fn no_probability_of_default_gives_no_credit_loss() {
        let years = tower().apply_annual(&losses(300)).unwrap();
        let after = panel(0.0, 0.3, 0.0, 0.4).apply(&years, 1).unwrap();
        for l in ["a", "b"] {
            assert!(
                column(&after, &["credit_loss", l])
                    .iter()
                    .all(|&c| c == 0.0)
            );
            assert_eq!(
                column(&after, &["recovered", l]),
                column(&years, &["ceded", l])
            );
        }
        assert_eq!(
            column(&after, &["net_after_default", "retained"]),
            column(&years, &["net", "retained"])
        );
    }

    #[test]
    fn certain_default_loses_the_uncollateralised_share_exactly() {
        let years = tower().apply_annual(&losses(300)).unwrap();
        let (recovery, collateral) = (0.35, 4.0);
        let by = panel(1.0, recovery, collateral, 0.2)
            .by_reinsurer(&years, 3)
            .unwrap();
        let (a, b) = (
            column(&years, &["ceded", "a"]),
            column(&years, &["ceded", "b"]),
        );
        let credit = column(&by, &["credit_loss", "X"]);
        let exposure = column(&by, &["exposure", "X"]);
        for i in 0..a.len() {
            let e = 0.6 * a[i] + 1.0 * b[i];
            let after = (e - collateral).max(0.0);
            assert_eq!(exposure[i], after);
            assert_eq!(credit[i], (1.0 - recovery) * after);
        }
        assert!(column(&by, &["default", "X"]).iter().all(|&d| d == 1.0));
        assert!(column(&by, &["default", "Y"]).iter().all(|&d| d == 0.0));
        assert!(credit.iter().any(|&c| c > 0.0));
    }

    #[test]
    fn full_recovery_or_full_collateral_gives_no_credit_loss() {
        let years = tower().apply_annual(&losses(300)).unwrap();
        for panel in [panel(1.0, 1.0, 0.0, 0.0), panel(1.0, 0.0, 26.0, 0.0)] {
            let by = panel.by_reinsurer(&years, 5).unwrap();
            assert!(column(&by, &["credit_loss", "X"]).iter().all(|&c| c == 0.0));
            let after = panel.apply(&years, 5).unwrap();
            assert!(
                column(&after, &["credit_loss", "a"])
                    .iter()
                    .all(|&c| c == 0.0)
            );
        }
    }

    #[test]
    fn independent_default_draws_the_documented_stream() {
        let n = 500;
        let years = tower().apply_annual(&losses(n)).unwrap();
        let ps = [0.2, 0.05, 0.5];
        let panel = Panel::new(
            ps.iter()
                .enumerate()
                .map(|(k, &p)| Reinsurer::new(&format!("R{k}"), p, 0.4).unwrap())
                .collect(),
        )
        .unwrap()
        .line("R0", "a", 0.5)
        .unwrap()
        .line("R2", "a", 0.5)
        .unwrap();
        let seed = 42;
        let by = panel.by_reinsurer(&years, seed).unwrap();
        for i in 0..n {
            let mut rng = StreamRng::new(seed, (3 << 62) + i as u64);
            let _v = rng.next_open01();
            for (k, &p) in ps.iter().enumerate() {
                let want = f64::from(u8::from(rng.next_open01() < p));
                assert_eq!(by.row(i).unwrap()[2 * ps.len() + k], want, "year {i}");
            }
        }
        // R1 has no line yet still draws, so R2's draws are its own.
        let total: f64 = column(&by, &["default", "R1"]).iter().sum();
        assert!(total > 0.0);
    }

    #[test]
    fn credit_loss_by_layer_sums_to_credit_loss_by_reinsurer() {
        let years = tower().apply_annual(&losses(2_000)).unwrap();
        let panel = Panel::new(vec![
            Reinsurer::new("X", 0.3, 0.2)
                .unwrap()
                .collateral(3.0)
                .unwrap(),
            Reinsurer::new("Y", 0.4, 0.5)
                .unwrap()
                .correlation(0.5)
                .unwrap(),
        ])
        .unwrap()
        .line("X", "a", 0.25)
        .unwrap()
        .line("Y", "a", 0.75)
        .unwrap()
        .line("X", "b", 0.5)
        .unwrap()
        .line("Y", "b", 0.5)
        .unwrap();
        let after = panel.apply(&years, 9).unwrap();
        let by = panel.by_reinsurer(&years, 9).unwrap();
        let layers: Vec<f64> = (0..years.n_sims())
            .map(|i| {
                column(&after, &["credit_loss", "a"])[i] + column(&after, &["credit_loss", "b"])[i]
            })
            .collect();
        let (x, y) = (
            column(&by, &["credit_loss", "X"]),
            column(&by, &["credit_loss", "Y"]),
        );
        let net = column(&years, &["net", "retained"]);
        let after_net = column(&after, &["net_after_default", "retained"]);
        for i in 0..years.n_sims() {
            assert!((layers[i] - (x[i] + y[i])).abs() <= 1e-12 * (1.0 + x[i] + y[i]));
            assert_eq!(after_net[i], net[i] + (x[i] + y[i]));
        }
        assert!(x.iter().any(|&c| c > 0.0) && y.iter().any(|&c| c > 0.0));
    }

    #[test]
    fn the_input_components_are_copied_bit_for_bit() {
        let years = tower().apply_annual(&losses(400)).unwrap();
        let after = panel(0.3, 0.4, 2.0, 0.3).apply(&years, 11).unwrap();
        let m = years.n_components();
        assert_eq!(&after.components()[..m], years.components());
        assert_eq!(after.n_components(), m + 2 * 2 + 1);
        for i in 0..years.n_sims() {
            let (a, b) = (after.row(i).unwrap(), years.row(i).unwrap());
            assert!(
                a[..m]
                    .iter()
                    .zip(b)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            );
        }
        // A layer the panel does not name is recovered in full.
        let only_b = Panel::new(vec![Reinsurer::new("X", 1.0, 0.0).unwrap()])
            .unwrap()
            .line("X", "b", 1.0)
            .unwrap()
            .apply(&years, 1)
            .unwrap();
        assert!(
            column(&only_b, &["credit_loss", "a"])
                .iter()
                .all(|&c| c == 0.0)
        );
        assert_eq!(
            column(&only_b, &["recovered", "a"]),
            column(&years, &["ceded", "a"])
        );
        assert!(
            column(&only_b, &["recovered", "b"])
                .iter()
                .all(|&c| c == 0.0)
        );
        let p = after.provenance();
        assert_eq!(p.seed, years.provenance().seed);
        assert!(p.parameters.contains(&("default_seed".into(), "11".into())));
    }

    #[test]
    fn a_labelled_result_nets_across_labels() {
        let years: Vec<Vec<f64>> = (0..300)
            .map(|i| vec![(i % 17) as f64 * 2.0, (i % 11) as f64 * 3.0])
            .collect();
        let labels = (0..600)
            .map(|k| ["wind", "flood"][k % 2].to_string())
            .collect();
        let events = EventSet::from_years(years, 3)
            .unwrap()
            .with_labels(vec![("peril".into(), labels)])
            .unwrap();
        let tower = Tower::new(vec![
            Layer::xol("a", 10.0, 10.0).unwrap(),
            Layer::xol("b", 20.0, 5.0).unwrap(),
        ])
        .unwrap();
        let labelled = tower.apply(&events).unwrap();
        let flat = labelled.aggregate(&["kind", "layer"]).unwrap();
        let panel = panel(0.3, 0.4, 2.0, 0.3);
        let after = panel.apply(&labelled, 4).unwrap();
        assert_eq!(after.dims(), labelled.dims());
        let blank = |kind: &str, layer: &str| -> ComponentKey {
            vec![kind.into(), layer.into(), "".into()]
        };
        assert!(after.component_index(&blank("credit_loss", "a")).is_some());
        assert!(
            after
                .component_index(&blank("net_after_default", "retained"))
                .is_some()
        );
        let by_labelled = panel.by_reinsurer(&labelled, 4).unwrap();
        let by_flat = panel.by_reinsurer(&flat, 4).unwrap();
        assert_eq!(
            column(&by_labelled, &["default", "X"]),
            column(&by_flat, &["default", "X"])
        );
        let (cl, cf) = (
            column(&by_labelled, &["credit_loss", "X"]),
            column(&by_flat, &["credit_loss", "X"]),
        );
        assert!(
            cl.iter()
                .zip(&cf)
                .all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + b))
        );
        assert!(cl.iter().any(|&c| c > 0.0));
        // The credit loss on the labelled result is the flat one's.
        let total = |pd: &PredictiveDistribution| -> Vec<f64> {
            let a = column(pd, &["credit_loss", "a", ""]);
            let b = column(pd, &["credit_loss", "b", ""]);
            a.iter().zip(&b).map(|(x, y)| x + y).collect()
        };
        let flat_after = panel.apply(&flat, 4).unwrap();
        let fl: Vec<f64> = (0..flat.n_sims())
            .map(|i| {
                column(&flat_after, &["credit_loss", "a"])[i]
                    + column(&flat_after, &["credit_loss", "b"])[i]
            })
            .collect();
        assert!(
            total(&after)
                .iter()
                .zip(&fl)
                .all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + b))
        );
    }

    #[test]
    fn the_mean_frequency_of_default_is_kept_with_ties_and_correlation() {
        // 70% of the years tie at zero.
        let n = 100_000;
        let losses: Vec<f64> = (0..n)
            .map(|i| {
                if i % 10 < 7 {
                    0.0
                } else {
                    (i % 997) as f64 / 10.0
                }
            })
            .collect();
        let years = tower().apply_annual(&losses).unwrap();
        let p = 0.1;
        let panel = Panel::new(vec![
            Reinsurer::new("X", p, 0.4)
                .unwrap()
                .correlation(0.6)
                .unwrap(),
        ])
        .unwrap()
        .line("X", "a", 1.0)
        .unwrap();
        let defaults = column(&panel.by_reinsurer(&years, 17).unwrap(), &["default", "X"]);
        let mean = defaults.iter().sum::<f64>() / n as f64;
        let se = (p * (1.0 - p) / n as f64).sqrt();
        assert!((mean - p).abs() < 4.0 * se, "mean {mean}");
        // Wrong-way risk: the defaults gather in the worst years.
        let worst: Vec<usize> = (0..n).filter(|&i| losses[i] > 70.0).collect();
        let in_worst = worst.iter().map(|&i| defaults[i]).sum::<f64>() / worst.len() as f64;
        let zero: Vec<usize> = (0..n).filter(|&i| losses[i] == 0.0).collect();
        let in_zero = zero.iter().map(|&i| defaults[i]).sum::<f64>() / zero.len() as f64;
        assert!(in_worst > 2.0 * p && in_zero < p, "{in_worst} {in_zero}");
    }

    #[test]
    fn a_driver_ranks_the_years_instead_of_the_gross_loss() {
        let n = 20_000;
        let years = tower().apply_annual(&losses(n)).unwrap();
        // The driver runs against the year index; the defaults follow it.
        let driver: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let panel = Panel::new(vec![
            Reinsurer::new("X", 0.1, 0.4)
                .unwrap()
                .correlation(0.8)
                .unwrap(),
        ])
        .unwrap()
        .line("X", "a", 1.0)
        .unwrap()
        .driver(driver)
        .unwrap();
        let d = column(&panel.by_reinsurer(&years, 2).unwrap(), &["default", "X"]);
        let low: f64 = d[..n / 2].iter().sum();
        let high: f64 = d[n / 2..].iter().sum();
        assert!(high > 3.0 * low, "{low} {high}");
        let short = panel.clone().driver(vec![1.0; 3]).unwrap();
        assert!(short.apply(&years, 2).is_err());
    }

    #[test]
    fn ranks_break_ties_by_group() {
        assert_eq!(
            ranks(&[3.0, 0.0, -0.0, 5.0, 0.0, 3.0]),
            [(3, 2), (0, 3), (0, 3), (5, 1), (0, 3), (3, 2)]
        );
    }

    #[test]
    fn bad_inputs_are_refused() {
        assert!(Reinsurer::new("", 0.1, 0.4).is_err());
        assert!(Reinsurer::new("X", -0.1, 0.4).is_err());
        assert!(Reinsurer::new("X", f64::NAN, 0.4).is_err());
        assert!(Reinsurer::new("X", 0.1, 1.1).is_err());
        let x = Reinsurer::new("X", 0.1, 0.4).unwrap();
        assert!(x.clone().collateral(-1.0).is_err());
        assert!(x.clone().correlation(1.0).is_err());
        assert!(x.clone().correlation(-0.1).is_err());
        assert!(Panel::new(vec![]).is_err());
        assert!(Panel::new(vec![x.clone(), x.clone()]).is_err());
        let mut bad = x.clone();
        bad.probability = 2.0;
        assert!(Panel::new(vec![bad]).is_err());
        let p = Panel::new(vec![x]).unwrap();
        assert!(p.clone().line("Z", "a", 1.0).is_err());
        assert!(p.clone().line("X", "a", 0.0).is_err());
        assert!(p.clone().line("X", "a", 1.5).is_err());
        assert!(
            p.clone()
                .line("X", "a", 0.5)
                .unwrap()
                .line("X", "a", 0.5)
                .is_err()
        );
        assert!(p.clone().driver(vec![]).is_err());
        assert!(p.clone().driver(vec![f64::INFINITY]).is_err());

        let years = tower().apply_annual(&losses(30)).unwrap();
        // An unknown layer, and lines that do not sum to 1.
        assert!(
            p.clone()
                .line("X", "c", 1.0)
                .unwrap()
                .apply(&years, 1)
                .is_err()
        );
        assert!(
            p.clone()
                .line("X", "a", 0.9)
                .unwrap()
                .apply(&years, 1)
                .is_err()
        );
        let ok = p.clone().line("X", "a", 1.0).unwrap();
        // Applied twice.
        let once = ok.apply(&years, 1).unwrap();
        assert!(ok.apply(&once, 1).is_err());
        // Not a tower's result, or without net components.
        assert!(
            ok.by_reinsurer(&ok.by_reinsurer(&years, 1).unwrap(), 1)
                .is_err()
        );
        let ceded_only = years.select("kind", &["ceded".into()]).unwrap();
        assert!(ok.apply(&ceded_only, 1).is_err());
        // A term's paths.
        let term = tower()
            .apply_term(&[
                EventSet::from_years(vec![vec![15.0], vec![30.0]], 1).unwrap(),
                EventSet::from_years(vec![vec![25.0], vec![5.0]], 2).unwrap(),
            ])
            .unwrap();
        assert!(ok.apply(&term, 1).is_err());
    }

    /// A term of `n_years` years over `n` paths, one event a year, each
    /// year's losses shifted so the years differ.
    fn term(n: usize, n_years: usize) -> PredictiveDistribution {
        let years: Vec<EventSet> = (0..n_years)
            .map(|t| {
                let l = losses(n + 7 * t);
                EventSet::from_years(l[7 * t..].iter().map(|&x| vec![x]).collect(), t as u64)
                    .unwrap()
            })
            .collect();
        tower().apply_term(&years).unwrap()
    }

    fn term_column(pd: &PredictiveDistribution, kind: &str, name: &str, year: i64) -> Vec<f64> {
        let key = vec![
            KeyValue::from(kind),
            KeyValue::from(name),
            KeyValue::Int(year),
        ];
        Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
    }

    fn bits(pd: &PredictiveDistribution) -> Vec<u64> {
        pd.draw_matrix().iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn a_term_of_one_year_draws_as_apply_does() {
        let l = losses(400);
        let events = EventSet::from_years(l.iter().map(|&x| vec![x]).collect(), 3).unwrap();
        let one = tower().apply(&events).unwrap();
        let term = tower().apply_term(std::slice::from_ref(&events)).unwrap();
        let panel = panel(0.3, 0.4, 2.0, 0.5);
        assert_eq!(
            bits(&panel.apply(&one, 8).unwrap()),
            bits(&panel.apply_term(&term, 1, 8).unwrap())
        );
        assert_eq!(
            bits(&panel.by_reinsurer(&one, 8).unwrap()),
            bits(&panel.by_reinsurer_term(&term, 1, 8).unwrap())
        );
    }

    #[test]
    fn certain_default_uses_the_collateral_up_year_by_year() {
        let (n, n_years) = (300, 4);
        let term = term(n, n_years);
        let (recovery, collateral) = (0.35, 12.0);
        let panel = panel(1.0, recovery, collateral, 0.2);
        let by = panel.by_reinsurer_term(&term, n_years, 3).unwrap();
        let after = panel.apply_term(&term, n_years, 3).unwrap();
        let mut left = vec![collateral; n];
        for t in 1..=n_years as i64 {
            let (a, b) = (
                term_column(&term, "ceded", "a", t),
                term_column(&term, "ceded", "b", t),
            );
            let credit = term_column(&by, "credit_loss", "X", t);
            let exposure = term_column(&by, "exposure", "X", t);
            let by_layer: Vec<f64> = term_column(&after, "credit_loss", "a", t)
                .iter()
                .zip(term_column(&after, "credit_loss", "b", t))
                .map(|(x, y)| x + y)
                .collect();
            for j in 0..n {
                let e = 0.6 * a[j] + 1.0 * b[j];
                let uncovered = (e - left[j]).max(0.0);
                assert_eq!(exposure[j], uncovered, "path {j} year {t}");
                assert_eq!(credit[j], (1.0 - recovery) * uncovered, "path {j} year {t}");
                assert!((by_layer[j] - credit[j]).abs() <= 1e-12 * (1.0 + credit[j]));
                left[j] = (left[j] - e).max(0.0);
            }
            assert!(
                term_column(&by, "default", "X", t)
                    .iter()
                    .all(|&d| d == 1.0)
            );
            assert!(
                term_column(&by, "default", "Y", t)
                    .iter()
                    .all(|&d| d == 0.0)
            );
        }
        // Some paths used their collateral up part way, and some never did.
        let used_up = left.iter().filter(|&&c| c == 0.0).count();
        assert!(used_up > 0 && used_up < n, "{used_up}");
    }

    #[test]
    fn no_probability_full_recovery_or_full_collateral_lose_nothing_over_a_term() {
        let (n, n_years) = (200, 3);
        let term = term(n, n_years);
        // X's exposure is at most 26 a year, 78 over the term.
        for panel in [
            panel(0.0, 0.3, 0.0, 0.4),
            panel(1.0, 1.0, 0.0, 0.0),
            panel(1.0, 0.0, 78.0, 0.0),
        ] {
            let after = panel.apply_term(&term, n_years, 5).unwrap();
            for t in 1..=n_years as i64 {
                for l in ["a", "b"] {
                    assert!(
                        term_column(&after, "credit_loss", l, t)
                            .iter()
                            .all(|&c| c == 0.0)
                    );
                    assert_eq!(
                        term_column(&after, "recovered", l, t),
                        term_column(&term, "ceded", l, t)
                    );
                }
                assert_eq!(
                    term_column(&after, "net_after_default", "retained", t),
                    term_column(&term, "net", "retained", t)
                );
            }
        }
    }

    #[test]
    fn a_default_is_final_for_the_rest_of_the_path() {
        let (n, n_years) = (2_000, 5);
        let term = term(n, n_years);
        let panel = panel(0.2, 0.4, 3.0, 0.5);
        let by = panel.by_reinsurer_term(&term, n_years, 21).unwrap();
        let after = panel.apply_term(&term, n_years, 21).unwrap();
        let defaults: Vec<Vec<f64>> = (1..=n_years as i64)
            .map(|t| term_column(&by, "default", "X", t))
            .collect();
        let mut later = 0;
        for pair in defaults.windows(2) {
            for (j, (before, after)) in pair[0].iter().zip(&pair[1]).enumerate() {
                assert!(after >= before, "path {j}");
                later += usize::from(after > before);
            }
        }
        assert!(later > 0);
        // recovered = ceded - credit loss by layer and year, and the net
        // after default adds the year's credit loss.
        for t in 1..=n_years as i64 {
            let credit_x = term_column(&by, "credit_loss", "X", t);
            let net = term_column(&term, "net", "retained", t);
            let after_net = term_column(&after, "net_after_default", "retained", t);
            for l in ["a", "b"] {
                let ceded = term_column(&term, "ceded", l, t);
                let credit = term_column(&after, "credit_loss", l, t);
                let recovered = term_column(&after, "recovered", l, t);
                for j in 0..n {
                    assert_eq!(recovered[j], ceded[j] - credit[j]);
                }
            }
            for j in 0..n {
                // Y never defaults, so the year's credit loss is X's.
                assert_eq!(after_net[j], net[j] + (credit_x[j] + 0.0));
            }
        }
        // The input's components are copied bit for bit.
        let m = term.n_components();
        assert_eq!(&after.components()[..m], term.components());
        assert_eq!(after.n_components(), m + n_years * (2 * 2 + 1));
        for j in 0..n {
            let (a, b) = (after.row(j).unwrap(), term.row(j).unwrap());
            assert!(
                a[..m]
                    .iter()
                    .zip(b)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            );
        }
        let p = after.provenance();
        assert!(
            p.parameters
                .contains(&("default_term_years".into(), "5".into()))
        );
    }

    #[test]
    fn a_path_draws_its_years_in_order_from_its_own_stream() {
        let (n, n_years) = (300, 3);
        let term = term(n, n_years);
        let ps = [0.2, 0.05, 0.5];
        let panel = Panel::new(
            ps.iter()
                .enumerate()
                .map(|(k, &p)| Reinsurer::new(&format!("R{k}"), p, 0.4).unwrap())
                .collect(),
        )
        .unwrap()
        .line("R0", "a", 0.5)
        .unwrap()
        .line("R2", "a", 0.5)
        .unwrap();
        let seed = 42;
        let by = panel.by_reinsurer_term(&term, n_years, seed).unwrap();
        for j in 0..n {
            let mut rng = StreamRng::new(seed, (3 << 62) + j as u64);
            let mut in_default = [false; 3];
            for t in 0..n_years {
                let _v = rng.next_open01();
                for (k, &p) in ps.iter().enumerate() {
                    in_default[k] |= rng.next_open01() < p;
                    let at = t * 3 * ps.len() + 2 * ps.len() + k;
                    let want = f64::from(u8::from(in_default[k]));
                    assert_eq!(by.row(j).unwrap()[at], want, "path {j} year {t}");
                }
            }
        }
    }

    #[test]
    fn a_term_driver_is_path_major() {
        let (n, n_years) = (4_000, 2);
        let term = term(n, n_years);
        // The driver is high in year 2 of every path and low in year 1.
        let driver: Vec<f64> = (0..n * n_years).map(|k| (k % n_years) as f64).collect();
        let panel = Panel::new(vec![
            Reinsurer::new("X", 0.1, 0.4)
                .unwrap()
                .correlation(0.8)
                .unwrap(),
        ])
        .unwrap()
        .line("X", "a", 1.0)
        .unwrap()
        .driver(driver)
        .unwrap();
        let by = panel.by_reinsurer_term(&term, n_years, 2).unwrap();
        let first: f64 = term_column(&by, "default", "X", 1).iter().sum();
        let second: f64 = term_column(&by, "default", "X", 2).iter().sum();
        assert!(second - first > 3.0 * first, "{first} {second}");
        let short = panel.clone().driver(vec![1.0; n]).unwrap();
        assert!(short.apply_term(&term, n_years, 2).is_err());
    }

    #[test]
    fn apply_term_refuses_what_is_no_term() {
        let ok = Panel::new(vec![Reinsurer::new("X", 0.1, 0.4).unwrap()])
            .unwrap()
            .line("X", "a", 1.0)
            .unwrap();
        let years = tower().apply_annual(&losses(30)).unwrap();
        let term = term(30, 2);
        assert!(ok.apply_term(&years, 1, 1).is_err());
        assert!(ok.by_reinsurer_term(&years, 1, 1).is_err());
        assert!(ok.apply(&term, 1).is_err());
        assert!(ok.by_reinsurer(&term, 1).is_err());
        // The years must be exactly 1..=n_years.
        assert!(ok.apply_term(&term, 0, 1).is_err());
        assert!(ok.apply_term(&term, 1, 1).is_err());
        assert!(ok.apply_term(&term, 3, 1).is_err());
        assert!(ok.apply_term(&term, 2, 1).is_ok());
        // A term far longer than the result is refused before anything is
        // allocated for its years, not aborted.
        for huge in [1 << 40, usize::MAX] {
            let err = ok.apply_term(&term, huge, 1).unwrap_err().to_string();
            assert!(err.contains("needs components in every year"), "{err}");
            assert!(ok.by_reinsurer_term(&term, huge, 1).is_err());
        }
        // Year 2 of the term alone has years outside 1..=1.
        let year2 = term.select("year", &[KeyValue::Int(2)]).unwrap();
        assert!(ok.apply_term(&year2, 1, 1).is_err());
        // Applied twice.
        let once = ok.apply_term(&term, 2, 1).unwrap();
        assert!(ok.apply_term(&once, 2, 1).is_err());
        // Without net components.
        let ceded_only = term.select("kind", &["ceded".into()]).unwrap();
        assert!(ok.apply_term(&ceded_only, 2, 1).is_err());
    }
}
