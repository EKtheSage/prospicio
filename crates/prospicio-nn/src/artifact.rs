//! Saving and loading fitted networks: versioned JSON artifacts in the
//! envelope every model shares (`docs/design/models.md`, "Model
//! artifacts").
//!
//! An artifact holds the spec, the training columns, the feature scaling,
//! every weight by its path in the network (`hidden.0.weight`, with its
//! shape), the early-stopping record, the dispersion and the fitted
//! values, with provenance and a hash of the training data. Weights are
//! `f64` and round-trip bit for bit, so a loaded fit predicts exactly as
//! the one saved. As for the GLM, the coding of the design (factor levels)
//! stays with the bindings.

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{Tensor, TensorData};
use prospicio_core::{Error, Result};
use prospicio_models::artifact::{
    bad, family_from, family_json, field, float, floats, input_hash, link_from, link_json, num,
    nums, object, parse, provenance, strings, uint,
};
use serde_json::{Map, Value, json};

use crate::attention::{AttentionCann, AttentionCannFit, TokenNet};
use crate::{Cann, CannFit, Cpu, EarlyStopping, Net, Scaling, seeded};

const CANN_FORMAT: &str = "risk_rs.cann_fit";
const ATTENTION_FORMAT: &str = "risk_rs.attention_cann_fit";
/// The newest version this build reads, for both formats.
const FORMAT_VERSION: u64 = 1;

impl CannFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_models::{Design, Family, Fitted, Link, Model};
    /// use prospicio_nn::{Cann, CannFit};
    ///
    /// let d = Design::new(vec!["x".into()], vec![vec![0.0, 1.0, 2.0, 3.0]]).unwrap();
    /// let mut spec = Cann::new(Family::Poisson, Link::Log);
    /// spec.hidden = vec![4];
    /// spec.epochs = 5;
    /// let fit = spec.fit(&d, &[1.0, 2.0, 2.0, 5.0]).unwrap();
    /// let back = CannFit::from_json(&fit.to_json()).unwrap();
    /// assert_eq!(back.predict(&d).unwrap(), fit.predict(&d).unwrap());
    /// assert_eq!(back.to_json(), fit.to_json());
    /// ```
    pub fn to_json(&self) -> String {
        let s = &self.spec;
        let doc = json!({
            "format": CANN_FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": nn_provenance("cann", &self.input_hash),
            "spec": {
                "family": family_json(s.family),
                "link": link_json(s.link),
                "hidden": s.hidden,
                "epochs": s.epochs,
                "batch_size": s.batch_size,
                "learning_rate": num(s.learning_rate),
                "seed": s.seed,
                "early_stopping": early_stopping_json(s.early_stopping),
            },
            "names": self.names,
            "scaling": scaling_json(&self.scaling),
            "parameters": parameters_json(&self.net),
            "best_epoch": self.best_epoch,
            "validation_history": nums(&self.validation_history),
            "dispersion": num(self.dispersion),
            "fitted": nums(&self.fitted),
        });
        serde_json::to_string(&doc).expect("a JSON value always serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json), rebuilding
    /// the network from the spec and loading every weight. Fails on another
    /// format, a newer version, or weights that do not fit the spec.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, CANN_FORMAT, FORMAT_VERSION)?;
        let top = object(&doc, "artifact")?;
        let spec = object(field(top, "spec")?, "spec")?;
        let spec = Cann {
            family: family_from(object(field(spec, "family")?, "family")?)?,
            link: link_from(object(field(spec, "link")?, "link")?)?,
            hidden: usizes(spec, "hidden")?,
            epochs: uint(spec, "epochs")? as usize,
            batch_size: uint(spec, "batch_size")? as usize,
            learning_rate: float(field(spec, "learning_rate")?, "learning_rate")?,
            seed: uint(spec, "seed")?,
            early_stopping: early_stopping_from(field(spec, "early_stopping")?)?,
        };
        let names = strings(top, "names")?;
        let device = Default::default();
        let net = seeded::<Cpu, _>(&device, 0, || Net::new(names.len(), &spec.hidden, &device));
        Ok(Self {
            net: load(net, top)?,
            scaling: scaling_from(top, names.len())?,
            names,
            spec,
            best_epoch: optional_uint(top, "best_epoch")?,
            validation_history: floats(top, "validation_history")?,
            dispersion: float(field(top, "dispersion")?, "dispersion")?,
            fitted: floats(top, "fitted")?,
            input_hash: input_hash(top)?,
        })
    }

    /// The hash of the training data (names, values, offset, weights and
    /// response), as the GLM's.
    pub fn input_hash(&self) -> &str {
        &self.input_hash
    }
}

impl AttentionCannFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    pub fn to_json(&self) -> String {
        let s = &self.spec;
        let doc = json!({
            "format": ATTENTION_FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": nn_provenance("attention_cann", &self.input_hash),
            "spec": {
                "family": family_json(s.family),
                "link": link_json(s.link),
                "d_model": s.d_model,
                "heads": s.heads,
                "layers": s.layers,
                "feed_forward": s.feed_forward,
                "epochs": s.epochs,
                "batch_size": s.batch_size,
                "learning_rate": num(s.learning_rate),
                "seed": s.seed,
                "early_stopping": early_stopping_json(s.early_stopping),
            },
            "names": self.names,
            "tokens": self.tokens,
            "groups": self.groups,
            "scaling": scaling_json(&self.scaling),
            "parameters": parameters_json(&self.net),
            "best_epoch": self.best_epoch,
            "validation_history": nums(&self.validation_history),
            "dispersion": num(self.dispersion),
            "fitted": nums(&self.fitted),
        });
        serde_json::to_string(&doc).expect("a JSON value always serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json).
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, ATTENTION_FORMAT, FORMAT_VERSION)?;
        let top = object(&doc, "artifact")?;
        let spec = object(field(top, "spec")?, "spec")?;
        let spec = AttentionCann {
            family: family_from(object(field(spec, "family")?, "family")?)?,
            link: link_from(object(field(spec, "link")?, "link")?)?,
            d_model: uint(spec, "d_model")? as usize,
            heads: uint(spec, "heads")? as usize,
            layers: uint(spec, "layers")? as usize,
            feed_forward: uint(spec, "feed_forward")? as usize,
            epochs: uint(spec, "epochs")? as usize,
            batch_size: uint(spec, "batch_size")? as usize,
            learning_rate: float(field(spec, "learning_rate")?, "learning_rate")?,
            seed: uint(spec, "seed")?,
            early_stopping: early_stopping_from(field(spec, "early_stopping")?)?,
        };
        if spec.heads == 0 || !spec.d_model.is_multiple_of(spec.heads) {
            return Err(bad("spec.heads"));
        }
        let names = strings(top, "names")?;
        let groups = field(top, "groups")?
            .as_array()
            .ok_or_else(|| bad("groups"))?
            .iter()
            .map(|g| {
                g.as_array()
                    .ok_or_else(|| bad("groups"))?
                    .iter()
                    .map(|j| match j.as_u64() {
                        Some(j) if (j as usize) < names.len() => Ok(j as usize),
                        _ => Err(bad("groups")),
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let tokens = strings(top, "tokens")?;
        if tokens.len() != groups.len() {
            return Err(bad("tokens"));
        }
        let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
        let device = Default::default();
        let net = seeded::<Cpu, _>(&device, 0, || TokenNet::new(&sizes, &spec, &device));
        Ok(Self {
            net: load(net, top)?,
            scaling: scaling_from(top, names.len())?,
            names,
            tokens,
            groups,
            spec,
            best_epoch: optional_uint(top, "best_epoch")?,
            validation_history: floats(top, "validation_history")?,
            dispersion: float(field(top, "dispersion")?, "dispersion")?,
            fitted: floats(top, "fitted")?,
            input_hash: input_hash(top)?,
        })
    }

    /// The hash of the training data, as [`CannFit::input_hash`].
    pub fn input_hash(&self) -> &str {
        &self.input_hash
    }
}

fn nn_provenance(model: &str, hash: &str) -> Value {
    provenance(model, "prospicio-nn", env!("CARGO_PKG_VERSION"), hash)
}

fn early_stopping_json(e: Option<EarlyStopping>) -> Value {
    match e {
        None => Value::Null,
        Some(e) => json!({"validation_share": num(e.validation_share), "patience": e.patience}),
    }
}

fn early_stopping_from(v: &Value) -> Result<Option<EarlyStopping>> {
    if v.is_null() {
        return Ok(None);
    }
    let o = object(v, "early_stopping")?;
    Ok(Some(EarlyStopping {
        validation_share: float(field(o, "validation_share")?, "validation_share")?,
        patience: uint(o, "patience")? as usize,
    }))
}

fn scaling_json(s: &Scaling) -> Value {
    json!({"mean": nums(&s.mean), "scale": nums(&s.scale)})
}

fn scaling_from(top: &Map<String, Value>, columns: usize) -> Result<Scaling> {
    let o = object(field(top, "scaling")?, "scaling")?;
    let (mean, scale) = (floats(o, "mean")?, floats(o, "scale")?);
    if mean.len() != columns || scale.len() != columns {
        return Err(bad("scaling"));
    }
    Ok(Scaling { mean, scale })
}

fn usizes(o: &Map<String, Value>, key: &str) -> Result<Vec<usize>> {
    field(o, key)?
        .as_array()
        .ok_or_else(|| bad(key))?
        .iter()
        .map(|v| v.as_u64().map(|k| k as usize).ok_or_else(|| bad(key)))
        .collect()
}

fn optional_uint(o: &Map<String, Value>, key: &str) -> Result<Option<usize>> {
    match field(o, key)? {
        Value::Null => Ok(None),
        v => v.as_u64().map(|k| Some(k as usize)).ok_or_else(|| bad(key)),
    }
}

/// One weight tensor: its path in the network, shape and values.
struct Weight {
    path: String,
    shape: Vec<usize>,
    values: Vec<f64>,
}

/// Reads every float parameter in visiting order, with its path.
#[derive(Default)]
struct Collect {
    path: Vec<String>,
    out: Vec<Weight>,
}

impl ModuleVisitor<Cpu> for Collect {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Cpu, D>>) {
        let t = param.val();
        self.out.push(Weight {
            path: self.path.join("."),
            shape: t.dims().to_vec(),
            values: t
                .into_data()
                .to_vec::<f64>()
                .expect("the CPU backend holds f64"),
        });
    }
}

fn parameters_json<M: Module<Cpu>>(net: &M) -> Value {
    let mut c = Collect::default();
    net.visit(&mut c);
    Value::Array(
        c.out
            .into_iter()
            .map(|w| json!({"path": w.path, "shape": w.shape, "values": nums(&w.values)}))
            .collect(),
    )
}

/// Replaces every float parameter, in visiting order, by the artifact's,
/// checking each path and shape.
struct Load {
    path: Vec<String>,
    weights: std::vec::IntoIter<Weight>,
    error: Option<Error>,
}

impl ModuleMapper<Cpu> for Load {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<Cpu, D>>) -> Param<Tensor<Cpu, D>> {
        let (id, tensor, mapper) = param.consume();
        let path = self.path.join(".");
        let tensor = match self.weights.next() {
            Some(w) if w.path == path && w.shape == tensor.dims() => {
                Tensor::from_data(TensorData::new(w.values, w.shape), &tensor.device())
            }
            other => {
                self.error.get_or_insert_with(|| {
                    Error::Data(match other {
                        Some(w) => format!(
                            "artifact weight {} {:?} does not fit the network's {path} {:?}",
                            w.path,
                            w.shape,
                            tensor.dims()
                        ),
                        None => format!("artifact has no weight for {path}"),
                    })
                });
                tensor
            }
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

fn load<M: Module<Cpu>>(net: M, top: &Map<String, Value>) -> Result<M> {
    let weights = field(top, "parameters")?
        .as_array()
        .ok_or_else(|| bad("parameters"))?
        .iter()
        .map(|w| {
            let w = object(w, "parameters")?;
            let shape = usizes(w, "shape")?;
            let values = floats(w, "values")?;
            if values.len() != shape.iter().product::<usize>() {
                return Err(bad("parameters.values"));
            }
            Ok(Weight {
                path: field(w, "path")?
                    .as_str()
                    .ok_or_else(|| bad("parameters.path"))?
                    .to_string(),
                shape,
                values,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut mapper = Load {
        path: Vec::new(),
        weights: weights.into_iter(),
        error: None,
    };
    let net = net.map(&mut mapper);
    if let Some(e) = mapper.error {
        return Err(e);
    }
    if mapper.weights.next().is_some() {
        return Err(Error::Data(
            "artifact has more weights than the network".into(),
        ));
    }
    Ok(net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenAttention;
    use prospicio_models::{Design, Family, Fitted, Link, Model};

    fn data() -> (Design, Vec<f64>) {
        let n = 40;
        let age: Vec<f64> = (0..n).map(|i| 18.0 + i as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 3 == 1))).collect();
        let c: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 3 == 2))).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 * 0.5).collect();
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

    #[test]
    fn a_cann_round_trips_with_early_stopping() {
        let (d, y) = data();
        let mut spec = Cann::new(Family::Gamma, Link::Log);
        spec.hidden = vec![5, 3];
        spec.epochs = 8;
        spec.batch_size = 16;
        spec.early_stopping = Some(EarlyStopping {
            validation_share: 0.25,
            patience: 3,
        });
        let y: Vec<f64> = y.iter().map(|v| v + 0.5).collect();
        let fit = spec.fit(&d, &y).unwrap();
        let text = fit.to_json();
        let back = CannFit::from_json(&text).unwrap();
        assert_eq!(back.to_json(), text);
        assert_eq!(back.correction(&d).unwrap(), fit.correction(&d).unwrap());
        assert_eq!(back.best_epoch(), fit.best_epoch());
        assert_eq!(back.input_hash(), fit.input_hash());
        assert!(!back.input_hash().is_empty());
        // The weights are named by their path in the network.
        let doc: Value = serde_json::from_str(&text).unwrap();
        let paths: Vec<&str> = doc["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w["path"].as_str().unwrap())
            .collect();
        assert_eq!(
            paths,
            [
                "hidden.0.weight",
                "hidden.0.bias",
                "hidden.1.weight",
                "hidden.1.bias",
                "out.weight",
                "out.bias"
            ]
        );
        assert_eq!(doc["parameters"][0]["shape"], json!([4, 5]));
    }

    #[test]
    fn an_attention_cann_round_trips() {
        let (d, y) = data();
        let mut spec = AttentionCann::new(Family::Poisson, Link::Log);
        spec.d_model = 4;
        spec.heads = 2;
        spec.layers = 1;
        spec.feed_forward = 6;
        spec.epochs = 3;
        let fit = spec.fit(&d, &y).unwrap();
        let text = fit.to_json();
        let back = AttentionCannFit::from_json(&text).unwrap();
        assert_eq!(back.to_json(), text);
        assert_eq!(back.predict(&d).unwrap(), fit.predict(&d).unwrap());
        assert_eq!(back.tokens(), fit.tokens());
        let (a, b): (TokenAttention, TokenAttention) =
            (back.attention(&d).unwrap(), fit.attention(&d).unwrap());
        assert_eq!(a.weights, b.weights);
    }

    #[test]
    fn a_foreign_newer_or_mismatched_artifact_is_refused() {
        let (d, y) = data();
        let mut spec = Cann::new(Family::Poisson, Link::Log);
        spec.hidden = vec![3];
        spec.epochs = 2;
        let text = spec.fit(&d, &y).unwrap().to_json();
        assert!(AttentionCannFit::from_json(&text).is_err());
        let newer = text.replace("\"format_version\":1", "\"format_version\":2");
        assert!(CannFit::from_json(&newer).is_err());
        let mut doc: Value = serde_json::from_str(&text).unwrap();
        doc["spec"]["hidden"] = json!([4]);
        let err = CannFit::from_json(&doc.to_string()).unwrap_err();
        assert!(err.to_string().contains("does not fit"), "{err}");
        let mut doc: Value = serde_json::from_str(&text).unwrap();
        doc["parameters"].as_array_mut().unwrap().pop();
        assert!(CannFit::from_json(&doc.to_string()).is_err());
    }
}
