//! A CANN whose correction is a transformer over feature tokens, so
//! attention can be read off per row: which rating factors the model
//! looks at, for which risks.
//!
//! Each feature becomes a token: a numeric column, or all the indicator
//! columns of one factor (`region[B]`, `region[C]`, … become the token
//! `region`, so each level gets its own embedding). A learned `[CLS]`
//! token joins them, and pre-norm transformer blocks (multi-head
//! self-attention, then a ReLU feed-forward layer, each with a residual
//! connection) mix them. The `[CLS]` token's final state, through a linear
//! layer that starts at zero, is the correction to the GLM's linear
//! predictor:
//!
//! ```text
//! η = offset + head(CLS after L blocks),    μ = g⁻¹(η)
//! ```
//!
//! This is the FT-Transformer of Gorishniy et al. (*Revisiting Deep
//! Learning Models for Tabular Data*, 2021) in the CANN setting of
//! Wüthrich and Merz, the combination Richman, Scognamiglio and Wüthrich
//! develop in *The Credibility Transformer* (2024). Training is as
//! [`Cann`](crate::Cann): the family's weighted deviance by Adam on
//! mini-batches, on the CPU in `f64`, reproducible from the seed.

use act_core::{Error, Result};
use act_models::{Design, Family, Fitted, Link, Model};
use act_prob::{PredictiveDistribution, Provenance};
use burn::module::{Initializer, Module, Param};
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::{
    Cpu, EarlyStopping, Scaling, Train, Training, check_family, process_draws, seeded, train,
};

/// One pre-norm transformer block.
#[derive(Module, Debug)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> Block<B> {
    fn new(d: usize, heads: usize, ff: usize, device: &B::Device) -> Self {
        Self {
            norm1: LayerNormConfig::new(d).init(device),
            attention: MultiHeadAttentionConfig::new(d, heads)
                .with_dropout(0.0)
                .init(device),
            norm2: LayerNormConfig::new(d).init(device),
            ff1: LinearConfig::new(d, ff).init(device),
            ff2: LinearConfig::new(ff, d).init(device),
        }
    }

    /// The block's output and its attention weights
    /// `[batch, heads, tokens, tokens]`.
    fn forward(&self, h: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let out = self
            .attention
            .forward(MhaInput::self_attn(self.norm1.forward(h.clone())));
        let h = h + out.context;
        let f = self
            .ff2
            .forward(relu(self.ff1.forward(self.norm2.forward(h.clone()))));
        (h + f, out.weights)
    }
}

/// Token embeddings, the `[CLS]` token, the blocks and the output head.
#[derive(Module, Debug)]
struct TokenNet<B: Backend> {
    embed: Vec<Linear<B>>,
    cls: Param<Tensor<B, 2>>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    out: Linear<B>,
}

impl<B: Backend> TokenNet<B> {
    fn new(sizes: &[usize], spec: &AttentionCann, device: &B::Device) -> Self {
        let d = spec.d_model;
        Self {
            embed: sizes
                .iter()
                .map(|&k| LinearConfig::new(k, d).init(device))
                .collect(),
            cls: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([1, d], device),
            blocks: (0..spec.layers)
                .map(|_| Block::new(d, spec.heads, spec.feed_forward, device))
                .collect(),
            norm: LayerNormConfig::new(d).init(device),
            out: LinearConfig::new(d, 1)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    /// One correction per row, and the last block's attention weights
    /// (`None` without blocks). `tokens[g]` is `[batch, columns of g]`.
    fn forward(&self, tokens: Vec<Tensor<B, 2>>) -> (Tensor<B, 1>, Option<Tensor<B, 4>>) {
        let m = tokens[0].dims()[0];
        let d = self.cls.val().dims()[1];
        let mut all = vec![self.cls.val().repeat_dim(0, m)];
        all.extend(self.embed.iter().zip(tokens).map(|(e, x)| e.forward(x)));
        let mut h: Tensor<B, 3> = Tensor::stack(all, 1);
        let mut weights = None;
        for block in &self.blocks {
            let (next, w) = block.forward(h);
            h = next;
            weights = Some(w);
        }
        let cls = h.slice([0..m, 0..1, 0..d]).reshape([m, d]);
        (
            self.out.forward(self.norm.forward(cls)).reshape([m]),
            weights,
        )
    }
}

/// An attention CANN specification.
///
/// Supported families are [`Cann`](crate::Cann)'s. Dropout is off, so a
/// seed fixes the fit.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionCann {
    pub family: Family,
    pub link: Link,
    /// Token width; a multiple of `heads`.
    pub d_model: usize,
    pub heads: usize,
    /// Transformer blocks.
    pub layers: usize,
    /// Width of each block's feed-forward layer.
    pub feed_forward: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Seeds the initial weights and the mini-batch order.
    pub seed: u64,
    /// Hold out rows and keep the best epoch; off by default.
    pub early_stopping: Option<EarlyStopping>,
}

impl AttentionCann {
    /// Tokens of width 16, 2 heads, 2 blocks with feed-forward width 32,
    /// 100 epochs, batches of 64, learning rate `1e-3`, seed 0.
    pub fn new(family: Family, link: Link) -> Self {
        Self {
            family,
            link,
            d_model: 16,
            heads: 2,
            layers: 2,
            feed_forward: 32,
            epochs: 100,
            batch_size: 64,
            learning_rate: 1e-3,
            seed: 0,
            early_stopping: None,
        }
    }

    fn check(&self) -> Result<()> {
        check_family(self.family, self.link)?;
        if self.heads == 0 || self.d_model == 0 || !self.d_model.is_multiple_of(self.heads) {
            return Err(Error::InvalidParameter {
                name: "d_model",
                value: self.d_model as f64,
                reason: "must be a positive multiple of the number of heads",
            });
        }
        if self.feed_forward == 0
            || self.batch_size == 0
            || self.learning_rate.is_nan()
            || self.learning_rate <= 0.0
        {
            return Err(Error::InvalidParameter {
                name: "batch_size",
                value: self.batch_size as f64,
                reason: "feed-forward width, batch size and learning rate must be positive",
            });
        }
        Ok(())
    }
}

/// Design columns grouped into tokens: a factor's indicator columns
/// (`name[level]`) share one token named `name`; every other column is its
/// own token. Constant columns (the intercept) carry nothing and are left
/// out.
fn tokens(design: &Design) -> (Vec<String>, Vec<Vec<usize>>) {
    let mut names: Vec<String> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (j, name) in design.names().iter().enumerate() {
        let c = design.column(j);
        if c.iter().all(|&v| v == c[0]) {
            continue;
        }
        let token = match name.find('[') {
            Some(at) if name.ends_with(']') && at > 0 => name[..at].to_string(),
            _ => name.clone(),
        };
        match names.iter().position(|t| *t == token) {
            Some(g) => groups[g].push(j),
            None => {
                names.push(token);
                groups.push(vec![j]);
            }
        }
    }
    (names, groups)
}

/// The token inputs for `rows`: `[rows, columns of g]` per group, from the
/// row-major standardized features `x` with `p` columns.
fn token_inputs<B: Backend>(
    x: &[f64],
    p: usize,
    rows: &[usize],
    groups: &[Vec<usize>],
    device: &B::Device,
) -> Vec<Tensor<B, 2>> {
    groups
        .iter()
        .map(|g| {
            let data: Vec<f64> = rows
                .iter()
                .flat_map(|&i| g.iter().map(move |&j| x[i * p + j]))
                .collect();
            Tensor::from_data(TensorData::new(data, [rows.len(), g.len()]), device)
        })
        .collect()
}

impl Model for AttentionCann {
    type Fitted = AttentionCannFit;

    fn fit(&self, design: &Design, y: &[f64]) -> Result<AttentionCannFit> {
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
        let (names, groups) = tokens(design);
        if groups.is_empty() {
            return Err(Error::Data(
                "the design has no non-constant columns to attend to".into(),
            ));
        }
        let device = Default::default();
        let scaling = Scaling::fit(design);
        let x = scaling.apply(design);
        let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
        let net: TokenNet<Train> =
            seeded::<Train, _>(&device, self.seed, || TokenNet::new(&sizes, self, &device));
        let (offset, w) = (design.offset(), design.weights());
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
            |net: &TokenNet<Train>, rows| {
                net.forward(token_inputs::<Train>(&x, p, rows, &groups, &device))
                    .0
            },
            |net: &TokenNet<Cpu>, rows| {
                net.forward(token_inputs::<Cpu>(&x, p, rows, &groups, &device))
                    .0
                    .into_data()
                    .to_vec::<f64>()
                    .map_err(|e| Error::Data(format!("network output: {e:?}")))
            },
        )?;
        let mut fit = AttentionCannFit {
            spec: self.clone(),
            names: design.names().to_vec(),
            tokens: names,
            groups,
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

/// Attention from the `[CLS]` token to each feature token in the last
/// block, per row and head: where the model looked for each risk.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenAttention {
    /// Feature tokens, in column order.
    pub tokens: Vec<String>,
    pub rows: usize,
    pub heads: usize,
    /// Row-major `[row][head][token]`. For each row and head the weights
    /// over the feature tokens plus `[CLS]` itself sum to 1; the share on
    /// `[CLS]` is the remainder.
    pub weights: Vec<f64>,
}

impl TokenAttention {
    /// Weight of `token` for `row` and `head`.
    pub fn get(&self, row: usize, head: usize, token: usize) -> f64 {
        let t = self.tokens.len();
        self.weights[(row * self.heads + head) * t + token]
    }

    /// Mean weight per token over rows and heads (or over the given rows),
    /// a global view of what the model attends to.
    pub fn mean(&self, rows: Option<&[usize]>) -> Vec<f64> {
        let all: Vec<usize> = (0..self.rows).collect();
        let rows = rows.unwrap_or(&all);
        let t = self.tokens.len();
        let mut out = vec![0.0; t];
        for &r in rows {
            for h in 0..self.heads {
                for (k, o) in out.iter_mut().enumerate() {
                    *o += self.get(r, h, k);
                }
            }
        }
        let count = (rows.len() * self.heads).max(1) as f64;
        out.iter_mut().for_each(|o| *o /= count);
        out
    }
}

/// A fitted attention CANN.
#[derive(Debug)]
pub struct AttentionCannFit {
    spec: AttentionCann,
    names: Vec<String>,
    tokens: Vec<String>,
    groups: Vec<Vec<usize>>,
    scaling: Scaling,
    net: TokenNet<Cpu>,
    best_epoch: Option<usize>,
    validation_history: Vec<f64>,
    dispersion: f64,
    fitted: Vec<f64>,
}

impl AttentionCannFit {
    /// The specification that was fitted.
    pub fn spec(&self) -> &AttentionCann {
        &self.spec
    }

    /// Feature tokens, in column order.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    fn run(&self, design: &Design) -> Result<(Tensor<Cpu, 1>, Option<Tensor<Cpu, 4>>)> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(
                "design columns do not match the fitted model".into(),
            ));
        }
        let (n, p) = (design.n_rows(), design.n_cols());
        let rows: Vec<usize> = (0..n).collect();
        let x = self.scaling.apply(design);
        let device = Default::default();
        let inputs = token_inputs::<Cpu>(&x, p, &rows, &self.groups, &device);
        Ok(self.net.forward(inputs))
    }

    /// The network's correction to the linear predictor, per row.
    pub fn correction(&self, design: &Design) -> Result<Vec<f64>> {
        self.run(design)?
            .0
            .into_data()
            .to_vec::<f64>()
            .map_err(|e| Error::Data(format!("network output: {e:?}")))
    }

    /// The last block's attention from `[CLS]` to each feature token, per
    /// row of `design` and head. Fails for a model with no blocks.
    pub fn attention(&self, design: &Design) -> Result<TokenAttention> {
        let weights = self
            .run(design)?
            .1
            .ok_or_else(|| Error::Data("the model has no attention blocks".into()))?;
        let [rows, heads, seq, _] = weights.dims();
        let t = seq - 1;
        // Row 0 of each attention matrix is the [CLS] query; drop its
        // weight on itself (column 0).
        let cls = weights.slice([0..rows, 0..heads, 0..1, 1..seq]);
        let weights = cls
            .reshape([rows * heads * t])
            .into_data()
            .to_vec::<f64>()
            .map_err(|e| Error::Data(format!("attention weights: {e:?}")))?;
        Ok(TokenAttention {
            tokens: self.tokens.clone(),
            rows,
            heads,
            weights,
        })
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

impl Fitted for AttentionCannFit {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        let correction = self.correction(design)?;
        Ok(design
            .offset()
            .iter()
            .zip(&correction)
            .map(|(o, c)| self.spec.link.inverse(o + c))
            .collect())
    }

    /// Process uncertainty only, as [`Cann`](crate::Cann).
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let mu = self.predict(design)?;
        let provenance = Provenance::new("attention_cann")
            .version("act-nn", env!("CARGO_PKG_VERSION"))
            .param("family", self.spec.family.name())
            .param("d_model", self.spec.d_model)
            .param("heads", self.spec.heads)
            .param("layers", self.spec.layers)
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
    use act_core::StreamRng;
    use act_models::metrics::mean_deviance;
    use act_prob::{Counting, Poisson};

    /// Poisson claims driven by an interaction: the rate doubles and more
    /// for young drivers in region C only, which no main-effects model
    /// captures. Columns: intercept, age, region[B], region[C].
    fn data(n: usize, seed: u64) -> (Design, Vec<f64>) {
        let mut rng = StreamRng::new(seed, 0);
        let (mut age, mut b, mut c, mut y) = (vec![], vec![], vec![], vec![]);
        for _ in 0..n {
            let a = 18.0 + 60.0 * rng.next_open01();
            let r = (rng.next_open01() * 3.0) as usize;
            let young_c = a < 35.0 && r == 2;
            let mu = 0.3 * if young_c { 3.0 } else { 1.0 };
            let k = Poisson::new(mu)
                .unwrap()
                .quantile(rng.next_open01())
                .unwrap();
            age.push(a);
            b.push(f64::from(u8::from(r == 1)));
            c.push(f64::from(u8::from(r == 2)));
            y.push(k as f64);
        }
        let d = Design::new(
            vec![
                "(Intercept)".into(),
                "age".into(),
                "region[B]".into(),
                "region[C]".into(),
            ],
            vec![vec![1.0; n], age, b, c],
        )
        .unwrap();
        (d, y)
    }

    fn small() -> AttentionCann {
        let mut spec = AttentionCann::new(Family::Poisson, Link::Log);
        spec.d_model = 8;
        spec.layers = 1;
        spec.feed_forward = 16;
        spec.epochs = 15;
        spec.learning_rate = 1e-2;
        spec
    }

    #[test]
    fn tokens_group_factor_levels() {
        let (d, _) = data(10, 1);
        let (names, groups) = tokens(&d);
        assert_eq!(names, vec!["age", "region"]);
        assert_eq!(groups, vec![vec![1], vec![2, 3]]);
    }

    #[test]
    fn zero_epochs_is_the_glm() {
        let (d, y) = data(40, 2);
        let d = d.with_offset(vec![-1.0; 40]).unwrap();
        let mut spec = small();
        spec.epochs = 0;
        let fit = spec.fit(&d, &y).unwrap();
        assert!(
            fit.fitted()
                .iter()
                .all(|m| (m - (-1f64).exp()).abs() < 1e-15)
        );
        let mut bad = small();
        bad.heads = 3;
        assert!(bad.fit(&d, &y).is_err());
    }

    #[test]
    fn learns_the_interaction_and_shows_its_attention() {
        let n = 800;
        let (d, y) = data(n, 3);
        let base = (y.iter().sum::<f64>() / n as f64).ln();
        let d = d.with_offset(vec![base; n]).unwrap();
        let spec = small();
        let fit = spec.fit(&d, &y).unwrap();
        let (h, hy) = data(800, 4);
        let h = h.with_offset(vec![base; 800]).unwrap();
        let hold = mean_deviance(Family::Poisson, &hy, &fit.predict(&h).unwrap(), None).unwrap();
        let flat = mean_deviance(Family::Poisson, &hy, &vec![base.exp(); 800], None).unwrap();
        assert!(hold < 0.97 * flat, "{hold} vs {flat}");
        // Young drivers in region C get the higher rate.
        let pred = fit.predict(&h).unwrap();
        let (age, c) = (h.column(1), h.column(3));
        let mean_of = |keep: &dyn Fn(usize) -> bool| {
            let v: Vec<f64> = (0..800).filter(|&i| keep(i)).map(|i| pred[i]).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let hot = mean_of(&|i| age[i] < 35.0 && c[i] == 1.0);
        let cold = mean_of(&|i| !(age[i] < 35.0 && c[i] == 1.0));
        assert!(hot > 1.8 * cold, "{hot} vs {cold}");
        // Attention: one weight per row, head and token, each row's
        // feature weights at most 1.
        let att = fit.attention(&h).unwrap();
        assert_eq!((att.rows, att.heads, att.tokens.len()), (800, 2, 2));
        for r in 0..800 {
            for head in 0..2 {
                let s = att.get(r, head, 0) + att.get(r, head, 1);
                assert!((0.0..=1.0 + 1e-12).contains(&s));
            }
        }
        assert_eq!(att.mean(None).len(), 2);
    }

    #[test]
    fn same_seed_same_fit() {
        let (d, y) = data(200, 5);
        let mut spec = small();
        spec.epochs = 2;
        let a = spec.fit(&d, &y).unwrap();
        let b = spec.fit(&d, &y).unwrap();
        assert_eq!(a.fitted(), b.fitted());
        spec.seed = 1;
        assert_ne!(spec.fit(&d, &y).unwrap().fitted(), a.fitted());
    }
}
