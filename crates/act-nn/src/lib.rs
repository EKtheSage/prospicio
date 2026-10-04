//! Neural networks on Burn, behind the `act-models` interface
//! (`docs/design/models.md`).
//!
//! [`Cann`] is the Combined Actuarial Neural Network of Wüthrich and Merz:
//! a GLM's linear predictor, carried in the design's offset, plus a
//! feed-forward network of the design's columns,
//!
//! ```text
//! η = offset + NN(x),    μ = g⁻¹(η)
//! ```
//!
//! The network's output layer starts at zero, so training starts exactly
//! at the GLM and learns only what the GLM misses. Training minimizes the
//! family's weighted deviance by Adam on mini-batches, on the CPU
//! (`ndarray` backend, `f64`), and is reproducible from its seed.

pub mod attention;

pub use attention::{AttentionCann, AttentionCannFit, TokenAttention};

use act_core::{Error, Result, StreamRng};
use act_models::{Design, Family, Fitted, Link, Model};
use act_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Initializer, Module, ModuleVisitor, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

pub(crate) type Cpu = NdArray<f64>;
pub(crate) type Train = Autodiff<Cpu>;

/// The feed-forward correction: hidden ReLU layers, then a linear output
/// initialized to zero.
#[derive(Module, Debug)]
struct Net<B: Backend> {
    hidden: Vec<Linear<B>>,
    out: Linear<B>,
}

impl<B: Backend> Net<B> {
    fn new(inputs: usize, hidden: &[usize], device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(hidden.len());
        let mut width = inputs;
        for &h in hidden {
            layers.push(LinearConfig::new(width, h).init(device));
            width = h;
        }
        let out = LinearConfig::new(width, 1)
            .with_initializer(Initializer::Zeros)
            .init(device);
        Self {
            hidden: layers,
            out,
        }
    }

    /// One value per row of `x`.
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let n = x.dims()[0];
        let mut h = x;
        for layer in &self.hidden {
            h = relu(layer.forward(h));
        }
        self.out.forward(h).reshape([n])
    }
}

/// A CANN specification.
///
/// Supported families: Poisson, gamma and Tweedie with the log link, and
/// Gaussian with the identity link.
#[derive(Debug, Clone, PartialEq)]
pub struct Cann {
    pub family: Family,
    pub link: Link,
    /// Widths of the hidden layers.
    pub hidden: Vec<usize>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Seeds the initial weights and the mini-batch order.
    pub seed: u64,
    /// Hold out rows and keep the best epoch; off by default.
    pub early_stopping: Option<EarlyStopping>,
}

impl Cann {
    /// A CANN with two hidden layers of 16 and 8 units, 200 epochs,
    /// batches of 64, learning rate `1e-3` and seed 0.
    pub fn new(family: Family, link: Link) -> Self {
        Self {
            family,
            link,
            hidden: vec![16, 8],
            epochs: 200,
            batch_size: 64,
            learning_rate: 1e-3,
            seed: 0,
            early_stopping: None,
        }
    }

    fn check(&self) -> Result<()> {
        check_family(self.family, self.link)?;
        if self.batch_size == 0 || self.learning_rate.is_nan() || self.learning_rate <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "batch_size",
                value: self.batch_size as f64,
                reason: "batch size and learning rate must be positive",
            });
        }
        Ok(())
    }
}

/// The families and links the networks support: Poisson, gamma and
/// Tweedie with the log link, Gaussian with the identity link.
pub(crate) fn check_family(family: Family, link: Link) -> Result<()> {
    family.validate()?;
    let ok = matches!(
        (family, link),
        (
            Family::Poisson | Family::Gamma | Family::Tweedie { .. },
            Link::Log
        ) | (Family::Gaussian, Link::Identity)
    );
    if !ok {
        return Err(Error::Data(format!(
            "the networks support Poisson, gamma and Tweedie with the log link and Gaussian \
             with the identity link, not {} with {:?}",
            family.name(),
            link
        )));
    }
    Ok(())
}

/// Draws each row's response from the family at its mean (process
/// uncertainty only), components keyed `row = 0, 1, …`.
pub(crate) fn process_draws(
    family: Family,
    dispersion: f64,
    mu: &[f64],
    weights: &[f64],
    n_sims: usize,
    seed: u64,
    provenance: Provenance,
) -> Result<PredictiveDistribution> {
    let components: Vec<ComponentKey> = (0..mu.len())
        .map(|i| vec![KeyValue::from(i as i64)])
        .collect();
    PredictiveDistribution::simulate(
        vec!["row".into()],
        components,
        n_sims,
        seed,
        provenance,
        |rng, row| {
            for (i, out) in row.iter_mut().enumerate() {
                *out = family
                    .draw(mu[i], dispersion, weights[i], rng.next_open01())
                    .unwrap_or(f64::NAN);
            }
        },
    )
}

/// Early stopping: hold out a share of the training rows, score the
/// network on them after every epoch by the family's mean deviance, and
/// keep the epoch that scored best, stopping once `patience` epochs pass
/// without improvement. The returned network is that best one, trained on
/// the remaining rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyStopping {
    /// Share of rows held out for validation, in `(0, 1)`.
    pub validation_share: f64,
    /// Epochs without improvement before stopping.
    pub patience: usize,
}

impl EarlyStopping {
    /// A fifth of the rows held out, patience 10 epochs.
    pub fn new() -> Self {
        Self {
            validation_share: 0.2,
            patience: 10,
        }
    }

    fn check(&self) -> Result<()> {
        if !(self.validation_share > 0.0 && self.validation_share < 1.0) || self.patience == 0 {
            return Err(Error::InvalidParameter {
                name: "validation_share",
                value: self.validation_share,
                reason: "early stopping needs a share in (0, 1) and a positive patience",
            });
        }
        Ok(())
    }
}

impl Default for EarlyStopping {
    fn default() -> Self {
        Self::new()
    }
}

/// The settings every network trains with.
pub(crate) struct Training<'a> {
    pub(crate) family: Family,
    pub(crate) link: Link,
    pub(crate) epochs: usize,
    pub(crate) batch_size: usize,
    pub(crate) learning_rate: f64,
    pub(crate) seed: u64,
    pub(crate) early_stopping: Option<EarlyStopping>,
    pub(crate) y: &'a [f64],
    pub(crate) weights: &'a [f64],
    pub(crate) offset: &'a [f64],
}

/// A trained network with its early-stopping record.
pub(crate) struct Trained<M> {
    pub(crate) net: M,
    /// Epoch (from 1) whose network was kept; `None` without early stopping.
    pub(crate) best_epoch: Option<usize>,
    /// Validation mean deviance after each epoch run.
    pub(crate) history: Vec<f64>,
}

/// Adam on mini-batches of the family's weighted deviance. `forward` gives
/// the network's output for rows during training; `evaluate` gives it for
/// rows from the inference copy, for validation.
pub(crate) fn train<M, F, V>(
    t: &Training<'_>,
    mut net: M,
    forward: F,
    evaluate: V,
) -> Result<Trained<M::InnerModule>>
where
    M: AutodiffModule<Train>,
    F: Fn(&M, &[usize]) -> Tensor<Train, 1>,
    V: Fn(&M::InnerModule, &[usize]) -> Result<Vec<f64>>,
{
    let n = t.y.len();
    // Rows: shuffled once on a stream of its own to hold some out.
    let (mut order, holdout) = match t.early_stopping {
        None => ((0..n).collect::<Vec<usize>>(), Vec::new()),
        Some(e) => {
            e.check()?;
            let mut all: Vec<usize> = (0..n).collect();
            let mut rng = StreamRng::new(t.seed, u64::MAX);
            for i in (1..n).rev() {
                let j = ((rng.next_open01() * (i + 1) as f64) as usize).min(i);
                all.swap(i, j);
            }
            let k = ((e.validation_share * n as f64).round() as usize).clamp(1, n - 1);
            let mut holdout = all.split_off(n - k);
            holdout.sort_unstable();
            all.sort_unstable();
            (all, holdout)
        }
    };
    let held_weight: f64 = holdout.iter().map(|&i| t.weights[i]).sum();
    let device = Default::default();
    let mut optimizer = AdamConfig::new().init();
    let mut best: Option<(usize, f64, M::InnerModule)> = None;
    let mut history = Vec::new();
    for epoch in 0..t.epochs {
        // Fisher–Yates on stream `epoch` of the seed.
        let mut rng = StreamRng::new(t.seed, epoch as u64);
        for i in (1..order.len()).rev() {
            let j = ((rng.next_open01() * (i + 1) as f64) as usize).min(i);
            order.swap(i, j);
        }
        for batch in order.chunks(t.batch_size) {
            let m = batch.len();
            let pick = |v: &[f64]| batch.iter().map(|&i| v[i]).collect::<Vec<f64>>();
            let ob = Tensor::<Train, 1>::from_data(TensorData::new(pick(t.offset), [m]), &device);
            let yb = Tensor::<Train, 1>::from_data(TensorData::new(pick(t.y), [m]), &device);
            let wb = Tensor::<Train, 1>::from_data(TensorData::new(pick(t.weights), [m]), &device);
            let eta = ob + forward(&net, batch);
            let l = loss(t.family, eta, yb, wb);
            let grads = GradientsParams::from_grads(l.backward(), &net);
            net = optimizer.step(t.learning_rate, net, grads);
        }
        let Some(e) = t.early_stopping else {
            continue;
        };
        let current = net.valid();
        let out = evaluate(&current, &holdout)?;
        let score = holdout
            .iter()
            .zip(&out)
            .map(|(&i, c)| {
                let mu = t.link.inverse(t.offset[i] + c);
                t.weights[i] * t.family.unit_deviance(t.y[i], mu)
            })
            .sum::<f64>()
            / held_weight;
        history.push(score);
        let improved = best.as_ref().is_none_or(|(_, s, _)| score < *s);
        if improved {
            best = Some((epoch + 1, score, current));
        } else if epoch + 1 - best.as_ref().map_or(0, |b| b.0) >= e.patience {
            break;
        }
    }
    Ok(match best {
        Some((epoch, _, net)) => Trained {
            net,
            best_epoch: Some(epoch),
            history,
        },
        None => Trained {
            net: net.valid(),
            best_epoch: None,
            history,
        },
    })
}

/// Builds a network with the backend's generator seeded by `seed`.
///
/// Burn's generator is global to the backend and its parameters are
/// initialized lazily, on first use. So that a fit is reproducible however
/// many run in parallel, a lock makes seed, build and initialization of
/// every parameter one atomic step. Training itself draws nothing (no
/// dropout).
pub(crate) fn seeded<B: Backend, M: Module<B>>(
    device: &B::Device,
    seed: u64,
    init: impl FnOnce() -> M,
) -> M {
    static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = INIT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    B::seed(device, seed);
    let module = init();
    module.visit(&mut Materialize);
    module
}

/// Initializes every float parameter it visits.
struct Materialize;

impl<B: Backend> ModuleVisitor<B> for Materialize {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let _ = param.val();
    }
}

/// Per-column centring and scaling learned on the training design.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Scaling {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl Scaling {
    pub(crate) fn fit(design: &Design) -> Self {
        let n = design.n_rows() as f64;
        let (mut mean, mut scale) = (Vec::new(), Vec::new());
        for j in 0..design.n_cols() {
            let c = design.column(j);
            let m = c.iter().sum::<f64>() / n;
            let s = (c.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n).sqrt();
            mean.push(m);
            scale.push(if s > 0.0 { s } else { 1.0 });
        }
        Self { mean, scale }
    }

    /// Row-major `n × p` standardized features.
    pub(crate) fn apply(&self, design: &Design) -> Vec<f64> {
        let (n, p) = (design.n_rows(), design.n_cols());
        let mut out = vec![0.0; n * p];
        for j in 0..p {
            for (i, x) in design.column(j).iter().enumerate() {
                out[i * p + j] = (x - self.mean[j]) / self.scale[j];
            }
        }
        out
    }
}

/// The training loss: the family's deviance up to terms constant in `η`,
/// weighted, averaged over the batch.
pub(crate) fn loss<B: Backend>(
    family: Family,
    eta: Tensor<B, 1>,
    y: Tensor<B, 1>,
    w: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let per_row = match family {
        // 2 (μ - y η)
        Family::Poisson => (eta.clone().exp() - y * eta).mul_scalar(2.0),
        // 2 (y e^-η + η)
        Family::Gamma => (y * eta.clone().neg().exp() + eta).mul_scalar(2.0),
        // 2 (-y e^((1-p)η) / (1-p) + e^((2-p)η) / (2-p))
        Family::Tweedie { power: p } => {
            let a = (y * eta.clone().mul_scalar(1.0 - p).exp()).div_scalar(1.0 - p);
            let b = eta.mul_scalar(2.0 - p).exp().div_scalar(2.0 - p);
            (b - a).mul_scalar(2.0)
        }
        // (y - η)²
        _ => (y - eta).powi_scalar(2),
    };
    (per_row * w).mean()
}

impl Model for Cann {
    type Fitted = CannFit;

    fn fit(&self, design: &Design, y: &[f64]) -> Result<CannFit> {
        self.check()?;
        let (n, p) = (design.n_rows(), design.n_cols());
        if y.len() != n {
            return Err(Error::Data(format!(
                "{} responses for {n} design rows",
                y.len()
            )));
        }
        if let Some(bad) = y.iter().find(|&&v| !self.family.valid_y(v)) {
            return Err(Error::InvalidParameter {
                name: "y",
                value: *bad,
                reason: "is outside the family's range",
            });
        }
        let device = Default::default();
        let scaling = Scaling::fit(design);
        let x = scaling.apply(design);
        let net: Net<Train> =
            seeded::<Train, _>(&device, self.seed, || Net::new(p, &self.hidden, &device));
        let (offset, w) = (design.offset(), design.weights());
        let rows_of = |rows: &[usize]| -> Vec<f64> {
            rows.iter()
                .flat_map(|&i| x[i * p..(i + 1) * p].to_vec())
                .collect()
        };
        let training = Training {
            family: self.family,
            link: self.link,
            epochs: self.epochs,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            seed: self.seed,
            early_stopping: self.early_stopping,
            y,
            weights: w,
            offset,
        };
        let trained = train(
            &training,
            net,
            |net: &Net<Train>, rows| {
                let xb = TensorData::new(rows_of(rows), [rows.len(), p]);
                net.forward(Tensor::from_data(xb, &device))
            },
            |net: &Net<Cpu>, rows| {
                let xb = TensorData::new(rows_of(rows), [rows.len(), p]);
                net.forward(Tensor::from_data(xb, &device))
                    .into_data()
                    .to_vec::<f64>()
                    .map_err(|e| Error::Data(format!("network output: {e:?}")))
            },
        )?;
        let mut fit = CannFit {
            spec: self.clone(),
            names: design.names().to_vec(),
            scaling,
            net: trained.net,
            best_epoch: trained.best_epoch,
            validation_history: trained.history,
            dispersion: 1.0,
            fitted: Vec::new(),
        };
        fit.fitted = fit.predict(design)?;
        if !fit.spec.family.unit_dispersion() {
            fit.dispersion = (0..n)
                .map(|i| {
                    let m = fit.fitted[i];
                    w[i] * (y[i] - m).powi(2) / fit.spec.family.variance(m)
                })
                .sum::<f64>()
                / n as f64;
        }
        Ok(fit)
    }
}

/// A fitted CANN.
#[derive(Debug)]
pub struct CannFit {
    spec: Cann,
    names: Vec<String>,
    scaling: Scaling,
    net: Net<Cpu>,
    best_epoch: Option<usize>,
    validation_history: Vec<f64>,
    dispersion: f64,
    fitted: Vec<f64>,
}

impl CannFit {
    /// The network's correction to the linear predictor, per row.
    pub fn correction(&self, design: &Design) -> Result<Vec<f64>> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(
                "design columns do not match the fitted CANN".into(),
            ));
        }
        let (n, p) = (design.n_rows(), design.n_cols());
        let device = Default::default();
        let x = Tensor::<Cpu, 2>::from_data(
            TensorData::new(self.scaling.apply(design), [n, p]),
            &device,
        );
        self.net
            .forward(x)
            .into_data()
            .to_vec::<f64>()
            .map_err(|e| Error::Data(format!("network output: {e:?}")))
    }

    /// Fitted means on the training data.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// With early stopping, the epoch (from 1) whose network was kept.
    pub fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    /// With early stopping, the validation mean deviance after each epoch.
    pub fn validation_history(&self) -> &[f64] {
        &self.validation_history
    }

    /// Dispersion: 1 for the Poisson, otherwise Pearson's estimate on the
    /// training data (`Σ w (y - μ)² / V(μ) / n`).
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }
}

impl Fitted for CannFit {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        let correction = self.correction(design)?;
        Ok(design
            .offset()
            .iter()
            .zip(&correction)
            .map(|(o, c)| self.spec.link.inverse(o + c))
            .collect())
    }

    /// Process uncertainty only: each row's response is drawn from the
    /// family at the fitted mean. The network's parameter uncertainty is
    /// not included (it needs refits on bootstrap samples or an ensemble),
    /// so rows are independent and intervals are too narrow by that much.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let mu = self.predict(design)?;
        let provenance = Provenance::new("cann")
            .version("act-nn", env!("CARGO_PKG_VERSION"))
            .param("family", self.spec.family.name())
            .param("hidden", format!("{:?}", self.spec.hidden))
            .param("epochs", self.spec.epochs)
            .param("training_seed", self.spec.seed);
        process_draws(
            self.spec.family,
            self.dispersion,
            &mu,
            design.weights(),
            n_sims,
            seed,
            provenance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_models::metrics::mean_deviance;

    /// Poisson counts whose log rate is a hump in x, which a GLM linear in
    /// x cannot follow.
    fn data(n: usize, seed: u64) -> (Design, Vec<f64>, Vec<f64>) {
        let mut rng = StreamRng::new(seed, 0);
        let x: Vec<f64> = (0..n).map(|_| rng.next_open01() * 4.0 - 2.0).collect();
        let truth: Vec<f64> = x.iter().map(|v| (1.0 - v * v).exp()).collect();
        let y: Vec<f64> = truth
            .iter()
            .map(|&m| {
                act_prob::Counting::quantile(&act_prob::Poisson::new(m).unwrap(), rng.next_open01())
                    .unwrap() as f64
            })
            .collect();
        let d = Design::new(vec!["x".into()], vec![x]).unwrap();
        (d, y, truth)
    }

    #[test]
    fn learns_what_the_glm_misses_and_replays() {
        let (d, y, _) = data(800, 1);
        // The "GLM" here is a constant log rate: the offset is its η.
        let base = (y.iter().sum::<f64>() / y.len() as f64).ln();
        let d = d.with_offset(vec![base; 800]).unwrap();
        let mut spec = Cann::new(Family::Poisson, Link::Log);
        spec.epochs = 60;
        spec.learning_rate = 1e-2;
        let fit = spec.fit(&d, &y).unwrap();
        let glm_dev = mean_deviance(Family::Poisson, &y, &vec![base.exp(); 800], None).unwrap();
        let cann_dev = mean_deviance(Family::Poisson, &y, fit.fitted(), None).unwrap();
        assert!(cann_dev < 0.8 * glm_dev, "{cann_dev} vs {glm_dev}");
        // Holdout data from the same process.
        let (h, hy, _) = data(400, 2);
        let h = h.with_offset(vec![base; 400]).unwrap();
        let pred = fit.predict(&h).unwrap();
        let hold = mean_deviance(Family::Poisson, &hy, &pred, None).unwrap();
        let hold_glm = mean_deviance(Family::Poisson, &hy, &vec![base.exp(); 400], None).unwrap();
        assert!(hold < 0.85 * hold_glm, "{hold} vs {hold_glm}");
        // Same seed, same network.
        let again = spec.fit(&d, &y).unwrap();
        assert_eq!(fit.fitted(), again.fitted());
    }

    #[test]
    fn early_stopping_keeps_the_best_epoch() {
        let (d, y, _) = data(300, 4);
        let base = (y.iter().sum::<f64>() / y.len() as f64).ln();
        let d = d.with_offset(vec![base; 300]).unwrap();
        let mut spec = Cann::new(Family::Poisson, Link::Log);
        spec.hidden = vec![32, 32];
        spec.epochs = 400;
        spec.learning_rate = 3e-2;
        spec.batch_size = 16;
        spec.early_stopping = Some(EarlyStopping {
            validation_share: 0.3,
            patience: 5,
        });
        let fit = spec.fit(&d, &y).unwrap();
        let best = fit.best_epoch().unwrap();
        let history = fit.validation_history();
        // Stopped early, `patience` epochs after the best, which scored lowest.
        assert!(history.len() < 400);
        assert_eq!(history.len(), best + 5);
        let low = history.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(history[best - 1], low);
        // Without early stopping there is no record.
        spec.early_stopping = None;
        spec.epochs = 2;
        let plain = spec.fit(&d, &y).unwrap();
        assert!(plain.best_epoch().is_none() && plain.validation_history().is_empty());
        spec.early_stopping = Some(EarlyStopping {
            validation_share: 1.0,
            patience: 5,
        });
        assert!(spec.fit(&d, &y).is_err());
    }

    #[test]
    fn zero_epochs_is_the_glm() {
        let (d, y, _) = data(50, 3);
        let d = d.with_offset(vec![0.3; 50]).unwrap();
        let mut spec = Cann::new(Family::Poisson, Link::Log);
        spec.epochs = 0;
        let fit = spec.fit(&d, &y).unwrap();
        assert!(
            fit.fitted()
                .iter()
                .all(|m| (m - 0.3f64.exp()).abs() < 1e-15)
        );
        assert!(
            Cann::new(Family::Binomial, Link::Logit)
                .fit(&d, &y)
                .is_err()
        );
    }
}
