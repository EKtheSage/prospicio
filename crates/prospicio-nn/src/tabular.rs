//! Tabular networks with learned embeddings of rating factors and one or
//! more claim heads on a shared body (`docs/design/models.md`, "Tabular
//! networks with embeddings and several heads").
//!
//! Each factor named in [`TabularNet::embedding`] is read from its
//! indicator columns in the design (`region[B]`, `region[C]`, …; a row with
//! none set is the reference level) and enters the network as a learned
//! vector of a few dimensions, not as its indicator columns: a factor with
//! dozens of levels then costs a few parameters a level, and its
//! embeddings can be read off and used as numeric columns in a GLM
//! ([`TabularNetFit::embedding_design`]). Wüthrich and Merz, *Statistical
//! Foundations of Actuarial Learning* (2023), section 7.4, and Schelldorfer
//! and Wüthrich's freMTPL2 tutorial (2019) use the same construction.
//!
//! The other columns are standardized as in [`Cann`](crate::Cann). The
//! body is hidden ReLU layers on the standardized columns and the
//! embeddings side by side; each [`Head`] is a linear output layer that
//! starts at zero, added to its own offset (a GLM's linear predictor, or
//! the log exposure):
//!
//! ```text
//! η_h = offset_h + out_h(body(x, e(f₁), e(f₂), …)),    μ_h = g_h⁻¹(η_h)
//! ```
//!
//! Training minimizes `Σ_h λ_h D_h`, each head's weighted mean deviance
//! (on its own weights) times its loss weight, by Adam on mini-batches of
//! rows, on the CPU in `f64`, reproducible from the seed. A spec with one
//! head and no embeddings is [`Cann`](crate::Cann): the same network, the
//! same draws and the same steps, so the same fit bit for bit.

use burn::module::{AutodiffModule, Initializer, Module};
use burn::nn::{Embedding as EmbeddingLayer, EmbeddingConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use prospicio_core::{Error, Result, StreamRng};
use prospicio_models::{Design, Family, Fitted, Link, Model};
use prospicio_prob::{PredictiveDistribution, Provenance};

use crate::{
    Cpu, EarlyStopping, Scaling, Train, check_family, input_hash, loss, process_draws, seeded,
};

/// A factor entered as a learned embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The factor's name: its indicator columns are `factor[level]`.
    pub factor: String,
    /// Dimensions of the embedding, at least 1.
    pub dim: usize,
    /// The label of the reference level (the rows with no indicator set),
    /// for [`TabularNetFit::embeddings`]; `"(reference)"` if not given.
    pub reference: Option<String>,
}

/// One output of the network: a family and link, and the head's weight in
/// the training loss.
#[derive(Debug, Clone, PartialEq)]
pub struct Head {
    pub name: String,
    pub family: Family,
    pub link: Link,
    /// The head's weight `λ` in the training loss, positive; 1 by default.
    pub loss_weight: f64,
}

impl Head {
    /// A head with loss weight 1 and the family's network link: log for
    /// the Poisson, gamma and Tweedie, identity for the Gaussian.
    pub fn new(name: &str, family: Family) -> Self {
        let link = if family == Family::Gaussian {
            Link::Identity
        } else {
            Link::Log
        };
        Self {
            name: name.to_string(),
            family,
            link,
            loss_weight: 1.0,
        }
    }

    /// The same head with another link.
    pub fn link(mut self, link: Link) -> Self {
        self.link = link;
        self
    }

    /// The same head with another loss weight.
    pub fn loss_weight(mut self, weight: f64) -> Self {
        self.loss_weight = weight;
        self
    }
}

/// The training data of one head: its response and, where they differ from
/// the design's, its prior weights and offset. A row with weight 0 does not
/// enter the head's loss (a severity head on a policy without claims), and
/// its response is not read.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    pub y: &'a [f64],
    /// The head's prior weights (claim counts for a severity); the
    /// design's if `None`.
    pub weights: Option<&'a [f64]>,
    /// The head's offset (a GLM's linear predictor, the log exposure); the
    /// design's if `None`.
    pub offset: Option<&'a [f64]>,
}

impl<'a> Target<'a> {
    /// A response with the design's weights and offset.
    pub fn new(y: &'a [f64]) -> Self {
        Self {
            y,
            weights: None,
            offset: None,
        }
    }

    /// The same target with its own prior weights.
    pub fn weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// The same target with its own offset.
    pub fn offset(mut self, offset: &'a [f64]) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// A tabular network specification.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularNet {
    pub embeddings: Vec<Embedding>,
    /// Widths of the hidden layers of the shared body.
    pub hidden: Vec<usize>,
    pub heads: Vec<Head>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Seeds the initial weights and the mini-batch order.
    pub seed: u64,
    /// Hold out rows and keep the best epoch by the loss on them; off by
    /// default.
    pub early_stopping: Option<EarlyStopping>,
}

impl Default for TabularNet {
    fn default() -> Self {
        Self::new()
    }
}

impl TabularNet {
    /// No embeddings and no heads yet, two hidden layers of 16 and 8
    /// units, 200 epochs, batches of 64, learning rate `1e-3`, seed 0:
    /// [`Cann`](crate::Cann)'s defaults.
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            hidden: vec![16, 8],
            heads: Vec::new(),
            epochs: 200,
            batch_size: 64,
            learning_rate: 1e-3,
            seed: 0,
            early_stopping: None,
        }
    }

    /// Enters `factor` as an embedding of `dim` dimensions.
    pub fn embedding(mut self, factor: &str, dim: usize) -> Self {
        self.embeddings.push(Embedding {
            factor: factor.to_string(),
            dim,
            reference: None,
        });
        self
    }

    /// Names the reference level of an embedded factor.
    pub fn reference(mut self, factor: &str, level: &str) -> Self {
        if let Some(e) = self.embeddings.iter_mut().find(|e| e.factor == factor) {
            e.reference = Some(level.to_string());
        }
        self
    }

    /// The widths of the hidden layers.
    pub fn hidden(mut self, widths: &[usize]) -> Self {
        self.hidden = widths.to_vec();
        self
    }

    /// Adds a head.
    pub fn head(mut self, head: Head) -> Self {
        self.heads.push(head);
        self
    }

    fn check(&self) -> Result<()> {
        if self.heads.is_empty() {
            return Err(Error::Data(
                "a tabular network needs at least one head".into(),
            ));
        }
        for (k, h) in self.heads.iter().enumerate() {
            check_family(h.family, h.link)?;
            if !(h.loss_weight > 0.0 && h.loss_weight.is_finite()) {
                return Err(Error::InvalidParameter {
                    name: "loss_weight",
                    value: h.loss_weight,
                    reason: "a head's loss weight must be positive and finite",
                });
            }
            if self.heads[..k].iter().any(|o| o.name == h.name) {
                return Err(Error::Data(format!("head {:?} is named twice", h.name)));
            }
        }
        for (k, e) in self.embeddings.iter().enumerate() {
            if e.dim == 0 {
                return Err(Error::InvalidParameter {
                    name: "dim",
                    value: 0.0,
                    reason: "an embedding needs at least one dimension",
                });
            }
            if self.embeddings[..k].iter().any(|o| o.factor == e.factor) {
                return Err(Error::Data(format!(
                    "factor {:?} is embedded twice",
                    e.factor
                )));
            }
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

    /// Fits the network with one target per head, in the heads' order.
    pub fn fit_heads(&self, design: &Design, targets: &[Target<'_>]) -> Result<TabularNetFit> {
        self.check()?;
        let n = design.n_rows();
        if targets.len() != self.heads.len() {
            return Err(Error::Data(format!(
                "{} targets for {} heads",
                targets.len(),
                self.heads.len()
            )));
        }
        let layout = Layout::new(design, &self.embeddings)?;
        let mut data = Vec::with_capacity(targets.len());
        for (h, t) in self.heads.iter().zip(targets) {
            data.push(HeadData::new(h, t, design)?);
        }
        let scaling = Scaling::fit(design);
        let inputs = Inputs::new(&layout, &scaling, design)?;
        let device = Default::default();
        let net: TabNet<Train> = seeded::<Train, _>(&device, self.seed, || {
            TabNet::new(&layout, &self.hidden, self.heads.len(), &device)
        });
        let trained = train_heads(self, &data, net, &inputs)?;
        let mut fit = TabularNetFit {
            spec: self.clone(),
            names: design.names().to_vec(),
            layout,
            scaling,
            net: trained.net,
            best_epoch: trained.best_epoch,
            validation_history: trained.history,
            dispersion: vec![1.0; self.heads.len()],
            fitted: Vec::new(),
            input_hash: tabular_hash(design, &data),
        };
        let corrections = fit.corrections(design)?;
        for (k, (h, d)) in self.heads.iter().zip(&data).enumerate() {
            let mu: Vec<f64> = (0..n)
                .map(|i| h.link.inverse(d.offset[i] + corrections[k][i]))
                .collect();
            if !h.family.unit_dispersion() {
                let used = d.weights.iter().filter(|&&w| w > 0.0).count().max(1);
                fit.dispersion[k] = (0..n)
                    .filter(|&i| d.weights[i] > 0.0)
                    .map(|i| d.weights[i] * (d.y[i] - mu[i]).powi(2) / h.family.variance(mu[i]))
                    .sum::<f64>()
                    / used as f64;
            }
            fit.fitted.push(mu);
        }
        Ok(fit)
    }
}

/// A one-head network as a [`Model`]: the response with the design's
/// offset and weights.
impl Model for TabularNet {
    type Fitted = TabularNetFit;

    fn fit(&self, design: &Design, y: &[f64]) -> Result<TabularNetFit> {
        if self.heads.len() != 1 {
            return Err(Error::Data(format!(
                "Model::fit takes one response, and the network has {} heads: use fit_heads",
                self.heads.len()
            )));
        }
        self.fit_heads(design, &[Target::new(y)])
    }
}

/// One head's response, weights and offset over the design's rows. A row
/// with weight 0 has its response set to the weighted mean, which it never
/// reads, so a placeholder (0 for a policy without claims) cannot reach the
/// loss as a NaN.
struct HeadData {
    y: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
}

impl HeadData {
    fn new(head: &Head, t: &Target<'_>, design: &Design) -> Result<Self> {
        let n = design.n_rows();
        let weights = t.weights.unwrap_or(design.weights()).to_vec();
        let offset = t.offset.unwrap_or(design.offset()).to_vec();
        for (what, len) in [
            ("responses", t.y.len()),
            ("weights", weights.len()),
            ("offsets", offset.len()),
        ] {
            if len != n {
                return Err(Error::Data(format!(
                    "head {:?}: {len} {what} for {n} design rows",
                    head.name
                )));
            }
        }
        if let Some(w) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: *w,
                reason: "a head's weights must be finite and non-negative",
            });
        }
        if let Some(o) = offset.iter().find(|o| !o.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "offset",
                value: *o,
                reason: "a head's offset must be finite",
            });
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(Error::Data(format!(
                "head {:?} has no row with a positive weight",
                head.name
            )));
        }
        let mut y = t.y.to_vec();
        for (v, w) in y.iter().zip(&weights) {
            if *w > 0.0 && !head.family.valid_y(*v) {
                return Err(Error::InvalidParameter {
                    name: "y",
                    value: *v,
                    reason: "is outside the family's range",
                });
            }
        }
        if weights.contains(&0.0) {
            let mean = y
                .iter()
                .zip(&weights)
                .map(|(v, w)| if *w > 0.0 { v * w } else { 0.0 })
                .sum::<f64>()
                / total;
            for (v, w) in y.iter_mut().zip(&weights) {
                if *w == 0.0 {
                    *v = mean;
                }
            }
        }
        Ok(Self { y, weights, offset })
    }
}

/// Which design columns feed the network as numbers and which make up each
/// embedded factor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Layout {
    /// Columns entered as standardized numbers, in design order.
    pub(crate) numeric: Vec<usize>,
    pub(crate) factors: Vec<FactorLayout>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FactorLayout {
    pub(crate) name: String,
    pub(crate) dim: usize,
    /// The factor's indicator columns, in design order.
    pub(crate) columns: Vec<usize>,
    /// The reference level first, then one level per indicator column.
    pub(crate) levels: Vec<String>,
}

impl Layout {
    fn new(design: &Design, embeddings: &[Embedding]) -> Result<Self> {
        let names = design.names();
        let mut used = vec![false; names.len()];
        let mut factors = Vec::with_capacity(embeddings.len());
        for e in embeddings {
            let prefix = format!("{}[", e.factor);
            let mut columns = Vec::new();
            let mut levels = vec![e.reference.clone().unwrap_or_else(|| "(reference)".into())];
            for (j, name) in names.iter().enumerate() {
                if let Some(level) = name.strip_prefix(&prefix).and_then(|r| r.strip_suffix(']')) {
                    columns.push(j);
                    levels.push(level.to_string());
                    used[j] = true;
                }
            }
            if columns.is_empty() {
                return Err(Error::Data(format!(
                    "the design has no indicator columns {}[level] for the embedded factor {:?}",
                    e.factor, e.factor
                )));
            }
            factors.push(FactorLayout {
                name: e.factor.clone(),
                dim: e.dim,
                columns,
                levels,
            });
        }
        let numeric = (0..names.len()).filter(|&j| !used[j]).collect();
        Ok(Self { numeric, factors })
    }
}

/// The network's inputs for every row: the standardized numeric columns,
/// row-major, and each factor's level index.
struct Inputs {
    p: usize,
    x: Vec<f64>,
    levels: Vec<Vec<i64>>,
}

impl Inputs {
    fn new(layout: &Layout, scaling: &Scaling, design: &Design) -> Result<Self> {
        let n = design.n_rows();
        let all = scaling.apply(design);
        let width = design.n_cols();
        let p = layout.numeric.len();
        let x = if p == width {
            all
        } else {
            let mut x = Vec::with_capacity(n * p);
            for i in 0..n {
                x.extend(layout.numeric.iter().map(|&j| all[i * width + j]));
            }
            x
        };
        let mut levels = Vec::with_capacity(layout.factors.len());
        for f in &layout.factors {
            let mut idx = vec![0_i64; n];
            for (k, &j) in f.columns.iter().enumerate() {
                for (i, &v) in design.column(j).iter().enumerate() {
                    if v == 1.0 {
                        if idx[i] != 0 {
                            return Err(Error::Data(format!(
                                "row {i} has two levels of the embedded factor {:?} set",
                                f.name
                            )));
                        }
                        idx[i] = k as i64 + 1;
                    } else if v != 0.0 {
                        return Err(Error::Data(format!(
                            "column {:?} of the embedded factor {:?} holds {v}, not 0 or 1",
                            design.names()[j],
                            f.name
                        )));
                    }
                }
            }
            levels.push(idx);
        }
        Ok(Self { p, x, levels })
    }

    fn batch<B: Backend>(
        &self,
        rows: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 2, Int>>) {
        let m = rows.len();
        let p = self.p;
        let x: Vec<f64> = rows
            .iter()
            .flat_map(|&i| self.x[i * p..(i + 1) * p].iter().copied())
            .collect();
        let x = Tensor::from_data(TensorData::new(x, [m, p]), device);
        let levels = self
            .levels
            .iter()
            .map(|l| {
                let v: Vec<i64> = rows.iter().map(|&i| l[i]).collect();
                Tensor::from_data(TensorData::new(v, [m, 1]), device)
            })
            .collect();
        (x, levels)
    }
}

/// The embeddings, the shared body and one linear output per head, each
/// starting at zero.
#[derive(Module, Debug)]
pub(crate) struct TabNet<B: Backend> {
    embed: Vec<EmbeddingLayer<B>>,
    hidden: Vec<Linear<B>>,
    out: Vec<Linear<B>>,
}

impl<B: Backend> TabNet<B> {
    pub(crate) fn new(layout: &Layout, hidden: &[usize], heads: usize, device: &B::Device) -> Self {
        // Keras' default for an embedding, as Schelldorfer and Wüthrich's
        // tutorial uses.
        let embed: Vec<EmbeddingLayer<B>> = layout
            .factors
            .iter()
            .map(|f| {
                EmbeddingConfig::new(f.levels.len(), f.dim)
                    .with_initializer(Initializer::Uniform {
                        min: -0.05,
                        max: 0.05,
                    })
                    .init(device)
            })
            .collect();
        let mut width = layout.numeric.len() + layout.factors.iter().map(|f| f.dim).sum::<usize>();
        let mut layers = Vec::with_capacity(hidden.len());
        for &h in hidden {
            layers.push(LinearConfig::new(width, h).init(device));
            width = h;
        }
        let out = (0..heads)
            .map(|_| {
                LinearConfig::new(width, 1)
                    .with_initializer(Initializer::Zeros)
                    .init(device)
            })
            .collect();
        Self {
            embed,
            hidden: layers,
            out,
        }
    }

    /// The factors' embedding vectors for `levels`, `[rows, dim]` each.
    fn embedded(&self, levels: Vec<Tensor<B, 2, Int>>) -> Vec<Tensor<B, 2>> {
        self.embed
            .iter()
            .zip(levels)
            .map(|(e, l)| {
                let [m, _] = l.dims();
                let out = e.forward(l);
                let d = out.dims()[2];
                out.reshape([m, d])
            })
            .collect()
    }

    /// One output per head, a value per row.
    fn forward(&self, x: Tensor<B, 2>, levels: Vec<Tensor<B, 2, Int>>) -> Vec<Tensor<B, 1>> {
        let n = x.dims()[0];
        let mut h = if self.embed.is_empty() {
            x
        } else {
            let mut parts = vec![x];
            parts.extend(self.embedded(levels));
            Tensor::cat(parts, 1)
        };
        for layer in &self.hidden {
            h = relu(layer.forward(h));
        }
        self.out
            .iter()
            .map(|o| o.forward(h.clone()).reshape([n]))
            .collect()
    }
}

struct Trained {
    net: TabNet<Cpu>,
    best_epoch: Option<usize>,
    history: Vec<f64>,
}

/// Adam on mini-batches of rows, the loss `Σ_h λ_h D_h`. The steps are
/// [`train`](crate::train)'s, so one head with weight 1 is a CANN's run.
fn train_heads(
    spec: &TabularNet,
    data: &[HeadData],
    mut net: TabNet<Train>,
    inputs: &Inputs,
) -> Result<Trained> {
    let n = data[0].y.len();
    let (mut order, holdout) = match spec.early_stopping {
        None => ((0..n).collect::<Vec<usize>>(), Vec::new()),
        Some(e) => {
            e.check()?;
            let mut all: Vec<usize> = (0..n).collect();
            let mut rng = StreamRng::new(spec.seed, u64::MAX);
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
    let held_weight: Vec<f64> = data
        .iter()
        .map(|d| holdout.iter().map(|&i| d.weights[i]).sum())
        .collect();
    let device = Default::default();
    let mut optimizer = AdamConfig::new().init();
    let mut best: Option<(usize, f64, TabNet<Cpu>)> = None;
    let mut history = Vec::new();
    for epoch in 0..spec.epochs {
        let mut rng = StreamRng::new(spec.seed, epoch as u64);
        for i in (1..order.len()).rev() {
            let j = ((rng.next_open01() * (i + 1) as f64) as usize).min(i);
            order.swap(i, j);
        }
        for batch in order.chunks(spec.batch_size) {
            let m = batch.len();
            let (x, levels) = inputs.batch::<Train>(batch, &device);
            let outputs = net.forward(x, levels);
            let mut total: Option<Tensor<Train, 1>> = None;
            for ((h, d), out) in spec.heads.iter().zip(data).zip(outputs) {
                let pick = |v: &[f64]| batch.iter().map(|&i| v[i]).collect::<Vec<f64>>();
                let ob =
                    Tensor::<Train, 1>::from_data(TensorData::new(pick(&d.offset), [m]), &device);
                let yb = Tensor::<Train, 1>::from_data(TensorData::new(pick(&d.y), [m]), &device);
                let wb =
                    Tensor::<Train, 1>::from_data(TensorData::new(pick(&d.weights), [m]), &device);
                let l = loss(h.family, ob + out, yb, wb);
                // A loss weight of 1 is left out, so one head is a CANN's
                // loss exactly.
                let l = if h.loss_weight == 1.0 {
                    l
                } else {
                    l.mul_scalar(h.loss_weight)
                };
                total = Some(match total {
                    None => l,
                    Some(t) => t + l,
                });
            }
            let total = total.expect("at least one head");
            let grads = GradientsParams::from_grads(total.backward(), &net);
            net = optimizer.step(spec.learning_rate, net, grads);
        }
        let Some(e) = spec.early_stopping else {
            continue;
        };
        let current = net.valid();
        let (x, levels) = inputs.batch::<Cpu>(&holdout, &device);
        let outputs = current.forward(x, levels);
        let mut score = 0.0;
        for (k, ((h, d), out)) in spec.heads.iter().zip(data).zip(outputs).enumerate() {
            let out = to_vec(out)?;
            if held_weight[k] <= 0.0 {
                continue;
            }
            let dev = holdout
                .iter()
                .zip(&out)
                .map(|(&i, c)| {
                    let mu = h.link.inverse(d.offset[i] + c);
                    d.weights[i] * h.family.unit_deviance(d.y[i], mu)
                })
                .sum::<f64>()
                / held_weight[k];
            score += if h.loss_weight == 1.0 {
                dev
            } else {
                h.loss_weight * dev
            };
        }
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

fn to_vec(t: Tensor<Cpu, 1>) -> Result<Vec<f64>> {
    t.into_data()
        .to_vec::<f64>()
        .map_err(|e| Error::Data(format!("network output: {e:?}")))
}

/// Hash of the training data: the design, and each head's response, weights
/// and offset. One head with the design's weights and offset hashes as a
/// [`Cann`](crate::Cann)'s data under its own kind.
fn tabular_hash(design: &Design, data: &[HeadData]) -> String {
    if let [d] = data
        && d.weights == design.weights()
        && d.offset == design.offset()
    {
        return input_hash("tabular_net_fit", design, &d.y);
    }
    let mut h = prospicio_prob::InputHasher::new();
    h.str("tabular_net_fit");
    for (j, name) in design.names().iter().enumerate() {
        h.str(name).f64s(design.column(j));
    }
    for d in data {
        h.f64s(&d.offset).f64s(&d.weights).f64s(&d.y);
    }
    h.finish()
}

/// The embedding of one factor: a vector per level.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingTable {
    pub factor: String,
    /// The reference level first, then the levels in the design's column
    /// order.
    pub levels: Vec<String>,
    pub dim: usize,
    /// Row-major `levels × dim`.
    pub values: Vec<f64>,
}

impl EmbeddingTable {
    /// The vector of level `k`.
    pub fn level(&self, k: usize) -> &[f64] {
        &self.values[k * self.dim..(k + 1) * self.dim]
    }

    /// The column names an embedding takes in a design: `factor:e1`,
    /// `factor:e2`, ….
    pub fn column_names(&self) -> Vec<String> {
        (1..=self.dim)
            .map(|k| format!("{}:e{k}", self.factor))
            .collect()
    }
}

/// A fitted tabular network.
#[derive(Debug)]
pub struct TabularNetFit {
    pub(crate) spec: TabularNet,
    pub(crate) names: Vec<String>,
    pub(crate) layout: Layout,
    pub(crate) scaling: Scaling,
    pub(crate) net: TabNet<Cpu>,
    pub(crate) best_epoch: Option<usize>,
    pub(crate) validation_history: Vec<f64>,
    pub(crate) dispersion: Vec<f64>,
    /// The fitted means per head on the training data, with the heads'
    /// training offsets.
    pub(crate) fitted: Vec<Vec<f64>>,
    pub(crate) input_hash: String,
}

impl TabularNetFit {
    /// The specification that was fitted.
    pub fn spec(&self) -> &TabularNet {
        &self.spec
    }

    /// The heads' names, in order.
    pub fn head_names(&self) -> Vec<&str> {
        self.spec.heads.iter().map(|h| h.name.as_str()).collect()
    }

    /// The index of the head named `name`.
    pub fn head_index(&self, name: &str) -> Result<usize> {
        self.spec
            .heads
            .iter()
            .position(|h| h.name == name)
            .ok_or_else(|| Error::Data(format!("the network has no head named {name:?}")))
    }

    fn check_design(&self, design: &Design) -> Result<()> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(
                "design columns do not match the fitted network".into(),
            ));
        }
        Ok(())
    }

    /// Each head's correction to its linear predictor, per row.
    pub fn corrections(&self, design: &Design) -> Result<Vec<Vec<f64>>> {
        self.check_design(design)?;
        let inputs = Inputs::new(&self.layout, &self.scaling, design)?;
        let rows: Vec<usize> = (0..design.n_rows()).collect();
        let device = Default::default();
        let (x, levels) = inputs.batch::<Cpu>(&rows, &device);
        self.net
            .forward(x, levels)
            .into_iter()
            .map(to_vec)
            .collect()
    }

    /// The means of head `head` for the rows of `design`, on `offset` (the
    /// design's if `None`).
    pub fn predict_head(
        &self,
        head: usize,
        design: &Design,
        offset: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let h = self.spec.heads.get(head).ok_or_else(|| {
            Error::Data(format!("the network has {} heads", self.spec.heads.len()))
        })?;
        let offset = offset.unwrap_or(design.offset());
        if offset.len() != design.n_rows() {
            return Err(Error::Data(format!(
                "{} offsets for {} design rows",
                offset.len(),
                design.n_rows()
            )));
        }
        let c = self.corrections(design)?.swap_remove(head);
        Ok(offset
            .iter()
            .zip(&c)
            .map(|(o, c)| h.link.inverse(o + c))
            .collect())
    }

    /// The embedding of `factor`, a vector per level.
    pub fn embeddings(&self, factor: &str) -> Result<EmbeddingTable> {
        let k = self
            .layout
            .factors
            .iter()
            .position(|f| f.name == factor)
            .ok_or_else(|| Error::Data(format!("{factor:?} is not an embedded factor")))?;
        let f = &self.layout.factors[k];
        let values = self.net.embed[k]
            .weight
            .val()
            .into_data()
            .to_vec::<f64>()
            .map_err(|e| Error::Data(format!("embedding weights: {e:?}")))?;
        Ok(EmbeddingTable {
            factor: f.name.clone(),
            levels: f.levels.clone(),
            dim: f.dim,
            values,
        })
    }

    /// `design` with each embedded factor's indicator columns replaced by
    /// its embedding coordinates (`factor:e1`, …, in the place of its first
    /// indicator column), the offset and weights kept: numeric columns a
    /// GLM can take in place of the factor.
    pub fn embedding_design(&self, design: &Design) -> Result<Design> {
        self.check_design(design)?;
        let inputs = Inputs::new(&self.layout, &self.scaling, design)?;
        let tables: Vec<EmbeddingTable> = self
            .layout
            .factors
            .iter()
            .map(|f| self.embeddings(&f.name))
            .collect::<Result<_>>()?;
        let mut names = Vec::new();
        let mut columns = Vec::new();
        for (j, name) in design.names().iter().enumerate() {
            match self
                .layout
                .factors
                .iter()
                .position(|f| f.columns.contains(&j))
            {
                None => {
                    names.push(name.clone());
                    columns.push(design.column(j).to_vec());
                }
                Some(k) if self.layout.factors[k].columns[0] == j => {
                    let t = &tables[k];
                    names.extend(t.column_names());
                    for d in 0..t.dim {
                        columns.push(
                            inputs.levels[k]
                                .iter()
                                .map(|&l| t.level(l as usize)[d])
                                .collect(),
                        );
                    }
                }
                Some(_) => {}
            }
        }
        Design::new(names, columns)?
            .with_offset(design.offset().to_vec())?
            .with_weights(design.weights().to_vec())
    }

    /// Fitted means of head `head` on the training data.
    pub fn fitted_head(&self, head: usize) -> &[f64] {
        &self.fitted[head]
    }

    /// With early stopping, the epoch (from 1) whose network was kept.
    pub fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    /// With early stopping, the validation loss `Σ_h λ_h D_h` after each
    /// epoch.
    pub fn validation_history(&self) -> &[f64] {
        &self.validation_history
    }

    /// Each head's dispersion: 1 for the Poisson, otherwise Pearson's
    /// estimate on the rows with a positive weight
    /// (`Σ w (y - μ)² / V(μ) / n₊`).
    pub fn dispersions(&self) -> &[f64] {
        &self.dispersion
    }

    /// The hash of the training data.
    pub fn input_hash(&self) -> &str {
        &self.input_hash
    }
}

/// The first head, on the design's offset.
impl Fitted for TabularNetFit {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        self.predict_head(0, design, None)
    }

    /// Process uncertainty only, for the first head: each row's response
    /// drawn from its family at the fitted mean (see
    /// [`CannFit`](crate::CannFit)).
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let mu = self.predict(design)?;
        let h = &self.spec.heads[0];
        let provenance = Provenance::new("tabular_net")
            .version("prospicio-nn", env!("CARGO_PKG_VERSION"))
            .param("head", h.name.as_str())
            .param("family", h.family.name())
            .param("hidden", format!("{:?}", self.spec.hidden))
            .param("epochs", self.spec.epochs)
            .param("training_seed", self.spec.seed);
        process_draws(
            h.family,
            self.dispersion[0],
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
    use crate::Cann;
    use prospicio_models::metrics::mean_deviance;

    fn poisson(m: f64, u: f64) -> f64 {
        prospicio_prob::Counting::quantile(&prospicio_prob::Poisson::new(m).unwrap(), u).unwrap()
            as f64
    }

    /// Counts on a hump in x, as the CANN's tests use.
    fn hump(n: usize, seed: u64) -> (Design, Vec<f64>) {
        let mut rng = StreamRng::new(seed, 0);
        let x: Vec<f64> = (0..n).map(|_| rng.next_open01() * 4.0 - 2.0).collect();
        let y = x
            .iter()
            .map(|v| poisson((1.0 - v * v).exp(), rng.next_open01()))
            .collect();
        (Design::new(vec!["x".into()], vec![x]).unwrap(), y)
    }

    #[test]
    fn one_head_without_embeddings_is_a_cann_bit_for_bit() {
        let (d, y) = hump(300, 1);
        let base = (y.iter().sum::<f64>() / y.len() as f64).ln();
        let d = d.with_offset(vec![base; 300]).unwrap();
        for early in [
            None,
            Some(EarlyStopping {
                validation_share: 0.25,
                patience: 3,
            }),
        ] {
            let mut cann = Cann::new(Family::Poisson, Link::Log);
            cann.epochs = 25;
            cann.learning_rate = 1e-2;
            cann.batch_size = 32;
            cann.seed = 7;
            cann.early_stopping = early;
            let mut tab = TabularNet::new().head(Head::new("freq", Family::Poisson));
            tab.epochs = 25;
            tab.learning_rate = 1e-2;
            tab.batch_size = 32;
            tab.seed = 7;
            tab.early_stopping = early;
            let a = cann.fit(&d, &y).unwrap();
            let b = tab.fit(&d, &y).unwrap();
            assert_eq!(a.fitted(), b.fitted_head(0));
            assert_eq!(a.predict(&d).unwrap(), b.predict(&d).unwrap());
            assert_eq!(a.best_epoch(), b.best_epoch());
            assert_eq!(a.validation_history(), b.validation_history());
            assert_eq!(a.dispersion(), b.dispersions()[0]);
        }
    }

    /// A factor of 20 levels whose effect is a smooth function of a hidden
    /// position, and a numeric column; claim counts on an exposure and, for
    /// policies with claims, a mean gamma severity driven by the same
    /// position.
    struct Book {
        design: Design,
        log_exposure: Vec<f64>,
        counts: Vec<f64>,
        severity: Vec<f64>,
        true_freq: Vec<f64>,
        true_sev: Vec<f64>,
        position: Vec<f64>,
    }

    #[allow(clippy::needless_range_loop)]
    fn book(n: usize, seed: u64) -> Book {
        let levels = 20;
        let position: Vec<f64> = (0..levels)
            .map(|k| (k as f64 / (levels - 1) as f64) * 2.0 - 1.0)
            .collect();
        let mut rng = StreamRng::new(seed, 0);
        let (mut cols, mut x) = (vec![vec![0.0; n]; levels - 1], Vec::new());
        let (mut log_exposure, mut counts, mut severity, mut tf, mut ts) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let k = ((rng.next_open01() * levels as f64) as usize).min(levels - 1);
            if k > 0 {
                cols[k - 1][i] = 1.0;
            }
            let xi = rng.next_open01() * 2.0 - 1.0;
            x.push(xi);
            let e = 0.5 + rng.next_open01() * 0.5;
            log_exposure.push(e.ln());
            let p = position[k];
            let f = (-1.6 + 0.8 * p + 0.3 * xi).exp();
            let s = (1.0 + p * p - 0.4 * xi).exp();
            tf.push(f);
            ts.push(s);
            let c = poisson(f * e, rng.next_open01());
            counts.push(c);
            severity.push(if c > 0.0 {
                // The mean of c gamma claims with shape 2.
                let g = prospicio_prob::Gamma::new(2.0 * c, s / (2.0 * c)).unwrap();
                <prospicio_prob::Gamma as prospicio_prob::Distribution>::quantile(
                    &g,
                    rng.next_open01(),
                )
                .unwrap()
            } else {
                0.0
            });
        }
        let mut names = vec!["(Intercept)".to_string(), "x".to_string()];
        names.extend((1..levels).map(|k| format!("brand[B{k}]")));
        let mut columns = vec![vec![1.0; n], x];
        columns.extend(cols);
        let design = Design::new(names, columns).unwrap();
        Book {
            design,
            log_exposure,
            counts,
            severity,
            true_freq: tf,
            true_sev: ts,
            position,
        }
    }

    fn spec() -> TabularNet {
        let mut s = TabularNet::new()
            .embedding("brand", 1)
            .reference("brand", "B0")
            .hidden(&[16, 8])
            .head(Head::new("freq", Family::Poisson))
            .head(Head::new("sev", Family::Gamma));
        s.epochs = 30;
        s.learning_rate = 1e-2;
        s.batch_size = 128;
        s.seed = 3;
        s
    }

    #[test]
    fn two_heads_recover_frequency_and_severity() {
        let b = book(20000, 11);
        let s = spec();
        let fit = s
            .fit_heads(
                &b.design,
                &[
                    Target::new(&b.counts).offset(&b.log_exposure),
                    Target::new(&b.severity)
                        .weights(&b.counts)
                        .offset(&vec![0.0; 20000]),
                ],
            )
            .unwrap();
        assert_eq!(fit.head_names(), ["freq", "sev"]);
        // Each head's rate is near the truth: the mean absolute log error is
        // a fraction of the truth's spread.
        let zero = vec![0.0; 20000];
        let freq = fit.predict_head(0, &b.design, Some(&zero)).unwrap();
        let sev = fit.predict_head(1, &b.design, Some(&zero)).unwrap();
        let err = |p: &[f64], t: &[f64]| {
            p.iter()
                .zip(t)
                .map(|(a, b)| (a.ln() - b.ln()).abs())
                .sum::<f64>()
                / p.len() as f64
        };
        let spread = |t: &[f64]| {
            let m = t.iter().map(|v| v.ln()).sum::<f64>() / t.len() as f64;
            t.iter().map(|v| (v.ln() - m).abs()).sum::<f64>() / t.len() as f64
        };
        assert!(
            err(&freq, &b.true_freq) < spread(&b.true_freq) / 3.0,
            "freq {} vs {}",
            err(&freq, &b.true_freq),
            spread(&b.true_freq)
        );
        assert!(
            err(&sev, &b.true_sev) < spread(&b.true_sev) / 3.0,
            "sev {} vs {}",
            err(&sev, &b.true_sev),
            spread(&b.true_sev)
        );
        // A one-dimensional embedding orders the levels by their position
        // (up to its sign).
        let e = fit.embeddings("brand").unwrap();
        assert_eq!(e.levels.len(), 20);
        assert_eq!(e.levels[0], "B0");
        let v: Vec<f64> = (0..20).map(|k| e.level(k)[0]).collect();
        let r = pearson(&v, &b.position).abs();
        assert!(r > 0.9, "{r}");
        // Severity rows without claims do not enter: their placeholder 0
        // is not a valid gamma response and the fit succeeded.
        assert!(fit.dispersions()[1] > 0.0 && fit.dispersions()[0] == 1.0);
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
        let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
        cov / (va * vb).sqrt()
    }

    #[test]
    fn embeddings_beat_indicators_on_held_out_data() {
        let (train, test) = (book(4000, 21), book(4000, 22));
        let freq = |s: &TabularNet, b: &Book| {
            s.fit_heads(&b.design, &[Target::new(&b.counts).offset(&b.log_exposure)])
                .unwrap()
        };
        let mut emb = TabularNet::new()
            .embedding("brand", 2)
            .head(Head::new("freq", Family::Poisson));
        let mut one_hot = TabularNet::new().head(Head::new("freq", Family::Poisson));
        for s in [&mut emb, &mut one_hot] {
            s.epochs = 30;
            s.learning_rate = 1e-2;
            s.batch_size = 128;
            s.seed = 5;
        }
        let score = |s: &TabularNet| {
            let fit = freq(s, &train);
            let mu = fit
                .predict_head(0, &test.design, Some(&test.log_exposure))
                .unwrap();
            mean_deviance(Family::Poisson, &test.counts, &mu, None).unwrap()
        };
        let (a, b) = (score(&emb), score(&one_hot));
        assert!(a < b, "embedding {a} vs indicators {b}");
    }

    #[test]
    fn the_forward_pass_is_the_formula() {
        let b = book(200, 31);
        let mut s = spec();
        s.epochs = 3;
        let fit = s
            .fit_heads(
                &b.design,
                &[
                    Target::new(&b.counts).offset(&b.log_exposure),
                    Target::new(&b.severity).weights(&b.counts),
                ],
            )
            .unwrap();
        // Recompute both heads by hand from the weights.
        let params = crate::artifact::parameter_list(&fit.net);
        let get = |k: &str| params.iter().find(|(n, _, _)| n == k).unwrap().clone();
        let (_, _, table) = get("embed.0.weight");
        let layers: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)> = (0..2)
            .map(|l| {
                let (_, shape, w) = get(&format!("hidden.{l}.weight"));
                let (_, _, bias) = get(&format!("hidden.{l}.bias"));
                (shape, w, bias)
            })
            .collect();
        let c = fit.corrections(&b.design).unwrap();
        let all = fit.scaling.apply(&b.design);
        let lev = &Inputs::new(&fit.layout, &fit.scaling, &b.design)
            .unwrap()
            .levels[0];
        for i in [0, 17, 199] {
            let mut h = vec![all[i * 21], all[i * 21 + 1]];
            h.push(table[lev[i] as usize]);
            for (shape, w, bias) in &layers {
                let (inp, outp) = (shape[0], shape[1]);
                h = (0..outp)
                    .map(|o| {
                        (bias[o] + (0..inp).map(|k| h[k] * w[k * outp + o]).sum::<f64>()).max(0.0)
                    })
                    .collect();
            }
            for (head, ch) in c.iter().enumerate() {
                let (_, _, w) = get(&format!("out.{head}.weight"));
                let (_, _, bias) = get(&format!("out.{head}.bias"));
                let v = bias[0] + h.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
                assert!((v - ch[i]).abs() < 1e-12, "{v} vs {}", ch[i]);
            }
        }
    }

    #[test]
    fn the_embedding_design_feeds_a_glm() {
        let b = book(300, 41);
        let mut s = spec();
        s.epochs = 2;
        let fit = s
            .fit_heads(
                &b.design,
                &[
                    Target::new(&b.counts).offset(&b.log_exposure),
                    Target::new(&b.severity).weights(&b.counts),
                ],
            )
            .unwrap();
        let e = fit.embedding_design(&b.design).unwrap();
        assert_eq!(e.names(), ["(Intercept)", "x", "brand:e1"]);
        let table = fit.embeddings("brand").unwrap();
        for i in 0..300 {
            let k = (2..21)
                .find(|&j| b.design.column(j)[i] == 1.0)
                .map_or(0, |j| j - 1);
            assert_eq!(e.column(2)[i], table.level(k)[0]);
        }
        assert_eq!(e.offset(), b.design.offset());
    }

    #[test]
    fn bad_settings_are_refused() {
        let b = book(100, 51);
        let t = [Target::new(&b.counts)];
        assert!(TabularNet::new().fit_heads(&b.design, &t).is_err());
        let s = TabularNet::new()
            .embedding("colour", 2)
            .head(Head::new("f", Family::Poisson));
        assert!(
            s.fit_heads(&b.design, &t)
                .unwrap_err()
                .to_string()
                .contains("colour")
        );
        let s = TabularNet::new()
            .head(Head::new("f", Family::Poisson))
            .head(Head::new("f", Family::Gamma));
        assert!(s.fit_heads(&b.design, &[t[0], t[0]]).is_err());
        let s = TabularNet::new().head(Head::new("f", Family::Poisson).loss_weight(0.0));
        assert!(s.fit_heads(&b.design, &t).is_err());
        let s = TabularNet::new().head(Head::new("f", Family::Gamma));
        // A zero gamma response with a positive weight.
        assert!(s.fit_heads(&b.design, &t).is_err());
        let two = spec();
        assert!(
            two.fit(&b.design, &b.counts)
                .unwrap_err()
                .to_string()
                .contains("fit_heads")
        );
        let s = TabularNet::new()
            .embedding("brand", 0)
            .head(Head::new("f", Family::Poisson));
        assert!(s.fit_heads(&b.design, &t).is_err());
    }
}
