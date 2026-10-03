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

use act_core::{Error, Result, StreamRng};
use act_models::{Design, Family, Fitted, Link, Model};
use act_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Initializer, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

type Cpu = NdArray<f64>;
type Train = Autodiff<Cpu>;

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
        }
    }

    fn check(&self) -> Result<()> {
        self.family.validate()?;
        let ok = matches!(
            (self.family, self.link),
            (
                Family::Poisson | Family::Gamma | Family::Tweedie { .. },
                Link::Log
            ) | (Family::Gaussian, Link::Identity)
        );
        if !ok {
            return Err(Error::Data(format!(
                "CANN supports Poisson, gamma and Tweedie with the log link and Gaussian \
                 with the identity link, not {} with {:?}",
                self.family.name(),
                self.link
            )));
        }
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

/// Per-column centring and scaling learned on the training design.
#[derive(Debug, Clone, PartialEq)]
struct Scaling {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl Scaling {
    fn fit(design: &Design) -> Self {
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
    fn apply(&self, design: &Design) -> Vec<f64> {
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
fn loss<B: Backend>(
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
        Train::seed(&device, self.seed);
        let scaling = Scaling::fit(design);
        let x = scaling.apply(design);
        let mut net: Net<Train> = Net::new(p, &self.hidden, &device);
        let mut optimizer = AdamConfig::new().init();
        let (offset, w) = (design.offset(), design.weights());
        let mut order: Vec<usize> = (0..n).collect();
        for epoch in 0..self.epochs {
            // Fisher–Yates on stream `epoch` of the seed.
            let mut rng = StreamRng::new(self.seed, epoch as u64);
            for i in (1..n).rev() {
                let j = ((rng.next_open01() * (i + 1) as f64) as usize).min(i);
                order.swap(i, j);
            }
            for batch in order.chunks(self.batch_size) {
                let m = batch.len();
                let xb: Vec<f64> = batch
                    .iter()
                    .flat_map(|&i| x[i * p..(i + 1) * p].to_vec())
                    .collect();
                let pick = |v: &[f64]| batch.iter().map(|&i| v[i]).collect::<Vec<f64>>();
                let xb = Tensor::<Train, 2>::from_data(TensorData::new(xb, [m, p]), &device);
                let ob = Tensor::<Train, 1>::from_data(TensorData::new(pick(offset), [m]), &device);
                let yb = Tensor::<Train, 1>::from_data(TensorData::new(pick(y), [m]), &device);
                let wb = Tensor::<Train, 1>::from_data(TensorData::new(pick(w), [m]), &device);
                let eta = ob + net.forward(xb);
                let l = loss(self.family, eta, yb, wb);
                let grads = GradientsParams::from_grads(l.backward(), &net);
                net = optimizer.step(self.learning_rate, net, grads);
            }
        }
        let net = net.valid();
        let mut fit = CannFit {
            spec: self.clone(),
            names: design.names().to_vec(),
            scaling,
            net,
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
        let n = design.n_rows();
        let components: Vec<ComponentKey> =
            (0..n).map(|i| vec![KeyValue::from(i as i64)]).collect();
        let (family, phi) = (self.spec.family, self.dispersion);
        let weights = design.weights().to_vec();
        let provenance = Provenance::new("cann")
            .version("act-nn", env!("CARGO_PKG_VERSION"))
            .param("family", family.name())
            .param("hidden", format!("{:?}", self.spec.hidden))
            .param("epochs", self.spec.epochs)
            .param("training_seed", self.spec.seed);
        PredictiveDistribution::simulate(
            vec!["row".into()],
            components,
            n_sims,
            seed,
            provenance,
            |rng, row| {
                for (i, out) in row.iter_mut().enumerate() {
                    *out = family
                        .draw(mu[i], phi, weights[i], rng.next_open01())
                        .unwrap_or(f64::NAN);
                }
            },
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
