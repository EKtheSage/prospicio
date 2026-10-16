//! Networks trained elsewhere (PyTorch, Keras, scikit-learn's exporters),
//! loaded from ONNX files for inference behind the `prospicio-models`
//! interface (`docs/design/models.md`, "ONNX import").
//!
//! The file is parsed at run time by Burn's ONNX front end, `onnx-ir`, the
//! parser behind `burn-import`; its graph (already simplified: `MatMul`
//! then `Add` fused to `Linear`, `Identity` and no-op `Flatten`s removed)
//! is converted to an owned plan of steps and evaluated in `f64` by a small
//! interpreter here. A float32 file therefore evaluates within float32
//! rounding of onnxruntime. The operators covered are those a tabular
//! network exports:
//!
//! - `Gemm` (`alpha`, `beta`, `transA`, `transB`), `Linear`, `MatMul`;
//! - `Add`, `Sub`, `Mul`, `Div`, `Pow`, with NumPy broadcasting;
//! - `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `Softplus`, `Exp`, `Log`,
//!   `Neg`, `Abs`, `Sqrt`, `Clip` (constant bounds), `Softmax`;
//! - `Concat`, `Gather` (embeddings), `Flatten`, `Reshape`, `Squeeze`,
//!   `Unsqueeze` (constant shapes and axes), `ReduceSum`, `ReduceMean`,
//!   `Cast`, `Identity`, `Constant`.
//!
//! Anything else (convolutions, recurrent cells, control flow, shapes
//! computed at run time) is refused when the file is loaded, with the
//! operators the file uses. [`OnnxModel`] runs the graph on tensors;
//! [`OnnxPredictor`] binds its inputs to a design's columns, reads its
//! output as a mean or on a link's scale (`μ = g⁻¹(offset + output)`,
//! CANN-style), and draws predictive distributions from a family.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use onnx_ir::{ArgType, Argument, Node, OnnxGraphBuilder, TensorDataExt, ValueSource};
use prospicio_core::{Error, Result};
use prospicio_models::{Design, Family, Fitted, Link};
use prospicio_prob::{PredictiveDistribution, Provenance};

/// A dense tensor of `f64`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

impl OnnxTensor {
    /// Fails unless `values` has `Π shape` entries.
    pub fn new(shape: Vec<usize>, values: Vec<f64>) -> Result<Self> {
        if shape.iter().product::<usize>() != values.len() {
            return Err(Error::Data(format!(
                "{} values for a tensor of shape {shape:?}",
                values.len()
            )));
        }
        Ok(Self { shape, values })
    }

    fn scalar(v: f64) -> Self {
        Self {
            shape: Vec::new(),
            values: vec![v],
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// A graph input: its name, rank, the sizes the file fixes (`None` where
/// it leaves a size free, such as the batch), and whether it holds
/// integers (category codes for an embedding).
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxInput {
    pub name: String,
    pub rank: usize,
    pub shape: Option<Vec<Option<usize>>>,
    pub integer: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Unary {
    Relu,
    Sigmoid,
    Tanh,
    Softplus,
    Exp,
    Log,
    Neg,
    Abs,
    Sqrt,
}

impl Unary {
    fn apply(self, x: f64) -> f64 {
        match self {
            Unary::Relu => x.max(0.0),
            Unary::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Unary::Tanh => x.tanh(),
            // log(1 + e^x), without overflow.
            Unary::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
            Unary::Exp => x.exp(),
            Unary::Log => x.ln(),
            Unary::Neg => -x,
            Unary::Abs => x.abs(),
            Unary::Sqrt => x.sqrt(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl Binary {
    fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Binary::Add => a + b,
            Binary::Sub => a - b,
            Binary::Mul => a * b,
            Binary::Div => a / b,
            Binary::Pow => a.powf(b),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Op {
    Gemm {
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    },
    Linear {
        transpose_weight: bool,
    },
    MatMul,
    Binary(Binary),
    Unary(Unary),
    LeakyRelu(f64),
    Clip(Option<f64>, Option<f64>),
    Softmax(usize),
    Concat(usize),
    Gather(usize),
    Flatten(usize),
    Reshape(Vec<i64>),
    Squeeze(Option<Vec<i64>>),
    Unsqueeze(Vec<i64>),
    Reduce {
        dims: Vec<usize>,
        keepdims: bool,
        mean: bool,
    },
    Cast {
        integer: bool,
    },
    Identity,
}

#[derive(Debug, Clone, PartialEq)]
enum Operand {
    Value(String),
    Const(Arc<OnnxTensor>),
    Missing,
}

#[derive(Debug, Clone, PartialEq)]
struct Step {
    op: Op,
    inputs: Vec<Operand>,
    output: String,
}

/// An ONNX graph, loaded and ready to run.
///
/// ```
/// use prospicio_nn::onnx::{OnnxModel, OnnxTensor};
///
/// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../validation/data/onnx/mlp.onnx");
/// let model = OnnxModel::load(path).unwrap();
/// assert_eq!(model.inputs()[0].name, "x");
/// assert_eq!(model.ops(), ["Gemm", "Relu", "Gemm", "Exp"]);
/// let x = OnnxTensor::new(vec![2, 4], vec![0.0; 8]).unwrap();
/// let out = model.run(&[x]).unwrap();
/// assert_eq!(out[0].shape, [2, 1]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxModel {
    inputs: Vec<OnnxInput>,
    outputs: Vec<String>,
    steps: Vec<Step>,
    ops: Vec<String>,
}

impl OnnxModel {
    /// Loads an ONNX file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Data(format!("cannot read {}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    /// Loads an ONNX model from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let graph = OnnxGraphBuilder::new()
            .parse_bytes(bytes)
            .map_err(|e| Error::Data(format!("not a readable ONNX model: {e}")))?;
        let inputs = graph
            .inputs
            .iter()
            .map(|a| match &a.ty {
                ArgType::Tensor(t) => Ok(OnnxInput {
                    name: a.name.clone(),
                    rank: t.rank,
                    shape: t.static_shape.clone(),
                    integer: t.dtype.is_int(),
                }),
                other => Err(Error::Data(format!(
                    "input {} is not a tensor ({other:?})",
                    a.name
                ))),
            })
            .collect::<Result<Vec<_>>>()?;
        let mut ops = Vec::new();
        let mut steps = Vec::new();
        let mut unsupported = Vec::new();
        for node in &graph.nodes {
            let name = op_name(node);
            if name == "Constant" {
                continue;
            }
            ops.push(name.clone());
            match convert(node) {
                Ok(Some(step)) => steps.push(step),
                Ok(None) => {}
                Err(reason) => unsupported.push(format!("{name} ({reason})")),
            }
        }
        if !unsupported.is_empty() {
            return Err(Error::Data(format!(
                "the model uses operators prospicio_nn::onnx does not run: {}; its operators: {}",
                unsupported.join(", "),
                ops.join(", ")
            )));
        }
        let outputs = graph.outputs.iter().map(|a| a.name.clone()).collect();
        Ok(Self {
            inputs,
            outputs,
            steps,
            ops,
        })
    }

    /// The graph's inputs, in order.
    pub fn inputs(&self) -> &[OnnxInput] {
        &self.inputs
    }

    /// The names of the graph's outputs, in order.
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// The operators the graph runs, in order (after `onnx-ir`'s
    /// simplification, constants left out).
    pub fn ops(&self) -> &[String] {
        &self.ops
    }

    /// Runs the graph on one tensor per input (in [`inputs`](Self::inputs)
    /// order; integers as whole `f64`), returning one tensor per output.
    pub fn run(&self, inputs: &[OnnxTensor]) -> Result<Vec<OnnxTensor>> {
        if inputs.len() != self.inputs.len() {
            return Err(Error::Data(format!(
                "{} tensors for {} inputs",
                inputs.len(),
                self.inputs.len()
            )));
        }
        let mut values: HashMap<&str, Arc<OnnxTensor>> = HashMap::new();
        for (spec, t) in self.inputs.iter().zip(inputs) {
            if t.shape.len() != spec.rank {
                return Err(Error::Data(format!(
                    "input {} has rank {}, not {}",
                    spec.name,
                    spec.rank,
                    t.shape.len()
                )));
            }
            if let Some(fixed) = &spec.shape {
                for (k, (want, got)) in fixed.iter().zip(&t.shape).enumerate() {
                    if want.is_some_and(|w| w != *got) {
                        return Err(Error::Data(format!(
                            "input {} needs size {} on axis {k}, got {got}",
                            spec.name,
                            want.unwrap_or(0)
                        )));
                    }
                }
            }
            values.insert(spec.name.as_str(), Arc::new(t.clone()));
        }
        for step in &self.steps {
            let args: Vec<Option<Arc<OnnxTensor>>> = step
                .inputs
                .iter()
                .map(|o| match o {
                    Operand::Value(name) => values
                        .get(name.as_str())
                        .cloned()
                        .ok_or_else(|| {
                            Error::Data(format!("value {name} is used before it is computed"))
                        })
                        .map(Some),
                    Operand::Const(t) => Ok(Some(t.clone())),
                    Operand::Missing => Ok(None),
                })
                .collect::<Result<_>>()?;
            let out = eval(&step.op, &args)?;
            values.insert(step.output.as_str(), Arc::new(out));
        }
        self.outputs
            .iter()
            .map(|name| {
                values
                    .get(name.as_str())
                    .map(|t| (**t).clone())
                    .ok_or_else(|| Error::Data(format!("output {name} is never computed")))
            })
            .collect()
    }
}

/// The operator's ONNX name.
fn op_name(node: &Node) -> String {
    let debug = format!("{node:?}");
    debug.split('(').next().unwrap_or("?").to_string()
}

/// An argument's value, as an operand.
fn operand(a: &Argument) -> std::result::Result<Operand, String> {
    match a.value_source {
        ValueSource::Dynamic => Ok(Operand::Value(a.name.clone())),
        ValueSource::Optional => Ok(Operand::Missing),
        ValueSource::Static(_) | ValueSource::Constant => {
            let data = a
                .value()
                .ok_or_else(|| format!("the constant {} has no value", a.name))?;
            let values = data
                .to_f64_vec()
                .map_err(|e| format!("the constant {} cannot be read: {e:?}", a.name))?;
            let shape = data.shape.to_vec();
            Ok(Operand::Const(Arc::new(OnnxTensor { shape, values })))
        }
    }
}

/// A node's step, `None` for a node that needs none.
fn convert(node: &Node) -> std::result::Result<Option<Step>, String> {
    let static_axes = |name: &str| format!("{name} computed at run time");
    let op = match node {
        Node::Gemm(n) => Op::Gemm {
            alpha: f64::from(n.config.alpha),
            beta: f64::from(n.config.beta),
            trans_a: n.config.trans_a != 0,
            trans_b: n.config.trans_b != 0,
        },
        Node::Linear(n) => Op::Linear {
            transpose_weight: n.config.transpose_weight,
        },
        Node::MatMul(_) => Op::MatMul,
        Node::Add(_) => Op::Binary(Binary::Add),
        Node::Sub(_) => Op::Binary(Binary::Sub),
        Node::Mul(_) => Op::Binary(Binary::Mul),
        Node::Div(_) => Op::Binary(Binary::Div),
        Node::Pow(_) => Op::Binary(Binary::Pow),
        Node::Relu(_) => Op::Unary(Unary::Relu),
        Node::Sigmoid(_) => Op::Unary(Unary::Sigmoid),
        Node::Tanh(_) => Op::Unary(Unary::Tanh),
        Node::Softplus(_) => Op::Unary(Unary::Softplus),
        Node::Exp(_) => Op::Unary(Unary::Exp),
        Node::Log(_) => Op::Unary(Unary::Log),
        Node::Neg(_) => Op::Unary(Unary::Neg),
        Node::Abs(_) => Op::Unary(Unary::Abs),
        Node::Sqrt(_) => Op::Unary(Unary::Sqrt),
        Node::LeakyRelu(n) => Op::LeakyRelu(n.config.alpha),
        Node::Clip(n) => {
            let bound = |b: &Option<onnx_ir::clip::ClipInput>| match b {
                None => Ok(None),
                Some(onnx_ir::clip::ClipInput::Static(v)) => Ok(Some(*v)),
                Some(_) => Err(static_axes("bounds")),
            };
            Op::Clip(bound(&n.config.min)?, bound(&n.config.max)?)
        }
        Node::Softmax(n) => Op::Softmax(n.config.axis),
        Node::Concat(n) => Op::Concat(n.config.axis),
        Node::Gather(n) => Op::Gather(n.config.axis),
        Node::Flatten(n) => Op::Flatten(n.config.axis),
        Node::Reshape(n) => match &n.config.shape {
            onnx_ir::reshape::ReshapeInput::Static(s) => Op::Reshape(s.clone()),
            _ => return Err(static_axes("a shape")),
        },
        Node::Squeeze(n) => match &n.config.axes {
            None => Op::Squeeze(None),
            Some(onnx_ir::squeeze::SqueezeInput::Static(a)) => Op::Squeeze(Some(a.clone())),
            Some(_) => return Err(static_axes("axes")),
        },
        Node::Unsqueeze(n) => match &n.config {
            onnx_ir::unsqueeze::UnsqueezeConfig::Static(a) => Op::Unsqueeze(a.clone()),
            _ => return Err(static_axes("axes")),
        },
        Node::ReduceSum(n) => Op::Reduce {
            dims: n.config.dims.clone(),
            keepdims: n.config.keepdims,
            mean: false,
        },
        Node::ReduceMean(n) => Op::Reduce {
            dims: n.config.dims.clone(),
            keepdims: n.config.keepdims,
            mean: true,
        },
        Node::Cast(n) => Op::Cast {
            integer: n.config.to.is_int(),
        },
        Node::Identity(_) => Op::Identity,
        _ => return Err("not supported".into()),
    };
    let mut inputs = node
        .inputs()
        .iter()
        .map(operand)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    // Inputs the configuration already holds (axes, shapes, bounds) are
    // not data.
    match op {
        Op::Reshape(_) | Op::Squeeze(_) | Op::Unsqueeze(_) | Op::Reduce { .. } | Op::Clip(..) => {
            inputs.truncate(1);
        }
        _ => {}
    }
    let output = node.outputs().first().ok_or("no output")?.name.clone();
    Ok(Some(Step { op, inputs, output }))
}

fn arg(args: &[Option<Arc<OnnxTensor>>], i: usize) -> Result<&OnnxTensor> {
    args.get(i)
        .and_then(|a| a.as_deref())
        .ok_or_else(|| Error::Data(format!("an operator is missing input {i}")))
}

fn eval(op: &Op, args: &[Option<Arc<OnnxTensor>>]) -> Result<OnnxTensor> {
    match op {
        Op::Gemm {
            alpha,
            beta,
            trans_a,
            trans_b,
        } => {
            let a = arg(args, 0)?;
            let b = arg(args, 1)?;
            let a = if *trans_a { transpose(a)? } else { a.clone() };
            let b = if *trans_b { transpose(b)? } else { b.clone() };
            let mut y = matmul(&a, &b)?;
            for v in &mut y.values {
                *v *= alpha;
            }
            match args.get(2).and_then(|c| c.as_deref()) {
                Some(c) if *beta != 0.0 => {
                    let scaled = OnnxTensor {
                        shape: c.shape.clone(),
                        values: c.values.iter().map(|v| v * beta).collect(),
                    };
                    broadcast(&y, &scaled, Binary::Add)
                }
                _ => Ok(y),
            }
        }
        Op::Linear { transpose_weight } => {
            let x = arg(args, 0)?;
            let w = arg(args, 1)?;
            let w = if *transpose_weight {
                transpose(w)?
            } else {
                w.clone()
            };
            let y = matmul(x, &w)?;
            match args.get(2).and_then(|b| b.as_deref()) {
                Some(b) => broadcast(&y, b, Binary::Add),
                None => Ok(y),
            }
        }
        Op::MatMul => matmul(arg(args, 0)?, arg(args, 1)?),
        Op::Binary(f) => broadcast(arg(args, 0)?, arg(args, 1)?, *f),
        Op::Unary(f) => Ok(map(arg(args, 0)?, |x| f.apply(x))),
        Op::LeakyRelu(alpha) => Ok(map(arg(args, 0)?, |x| if x >= 0.0 { x } else { alpha * x })),
        Op::Clip(lo, hi) => Ok(map(arg(args, 0)?, |x| {
            x.max(lo.unwrap_or(f64::NEG_INFINITY))
                .min(hi.unwrap_or(f64::INFINITY))
        })),
        Op::Softmax(axis) => softmax(arg(args, 0)?, *axis),
        Op::Concat(axis) => {
            let parts: Vec<&OnnxTensor> = args
                .iter()
                .map(|a| {
                    a.as_deref()
                        .ok_or_else(|| Error::Data("Concat input missing".into()))
                })
                .collect::<Result<_>>()?;
            concat(&parts, *axis)
        }
        Op::Gather(axis) => gather(arg(args, 0)?, arg(args, 1)?, *axis),
        Op::Flatten(axis) => {
            let x = arg(args, 0)?;
            let outer: usize = x.shape[..(*axis).min(x.shape.len())].iter().product();
            Ok(OnnxTensor {
                shape: vec![outer, x.len() / outer.max(1)],
                values: x.values.clone(),
            })
        }
        Op::Reshape(shape) => reshape(arg(args, 0)?, shape),
        Op::Squeeze(axes) => {
            let x = arg(args, 0)?;
            let r = x.shape.len() as i64;
            let shape = match axes {
                None => x.shape.iter().copied().filter(|&d| d != 1).collect(),
                Some(axes) => {
                    let drop: Vec<usize> = axes
                        .iter()
                        .map(|&a| a.rem_euclid(r.max(1)) as usize)
                        .collect();
                    x.shape
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| !drop.contains(k))
                        .map(|(_, &d)| d)
                        .collect()
                }
            };
            Ok(OnnxTensor {
                shape,
                values: x.values.clone(),
            })
        }
        Op::Unsqueeze(axes) => {
            let x = arg(args, 0)?;
            let rank = (x.shape.len() + axes.len()) as i64;
            let mut at: Vec<usize> = axes.iter().map(|&a| a.rem_euclid(rank) as usize).collect();
            at.sort_unstable();
            let mut shape = x.shape.clone();
            for k in at {
                shape.insert(k.min(shape.len()), 1);
            }
            Ok(OnnxTensor {
                shape,
                values: x.values.clone(),
            })
        }
        Op::Reduce {
            dims,
            keepdims,
            mean,
        } => reduce(arg(args, 0)?, dims, *keepdims, *mean),
        Op::Cast { integer } => {
            let x = arg(args, 0)?;
            Ok(if *integer {
                map(x, f64::trunc)
            } else {
                x.clone()
            })
        }
        Op::Identity => Ok(arg(args, 0)?.clone()),
    }
}

fn map(x: &OnnxTensor, f: impl Fn(f64) -> f64) -> OnnxTensor {
    OnnxTensor {
        shape: x.shape.clone(),
        values: x.values.iter().map(|&v| f(v)).collect(),
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * shape[k + 1];
    }
    s
}

/// NumPy broadcasting.
fn broadcast(a: &OnnxTensor, b: &OnnxTensor, f: Binary) -> Result<OnnxTensor> {
    let rank = a.shape.len().max(b.shape.len());
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut p = vec![1; rank - s.len()];
        p.extend_from_slice(s);
        p
    };
    let (sa, sb) = (pad(&a.shape), pad(&b.shape));
    let mut shape = Vec::with_capacity(rank);
    for (&x, &y) in sa.iter().zip(&sb) {
        shape.push(match (x, y) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(Error::Data(format!(
                    "shapes {:?} and {:?} do not broadcast",
                    a.shape, b.shape
                )));
            }
        });
    }
    let (ta, tb, to) = (strides(&sa), strides(&sb), strides(&shape));
    let n: usize = shape.iter().product();
    let values = (0..n)
        .map(|i| {
            let (mut ia, mut ib) = (0, 0);
            for k in 0..rank {
                let c = (i / to[k]) % shape[k];
                if sa[k] != 1 {
                    ia += c * ta[k];
                }
                if sb[k] != 1 {
                    ib += c * tb[k];
                }
            }
            f.apply(a.values[ia], b.values[ib])
        })
        .collect();
    Ok(OnnxTensor { shape, values })
}

fn transpose(x: &OnnxTensor) -> Result<OnnxTensor> {
    if x.shape.len() != 2 {
        return Err(Error::Data(format!(
            "a transpose needs a matrix, got shape {:?}",
            x.shape
        )));
    }
    let (r, c) = (x.shape[0], x.shape[1]);
    let mut values = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            values[j * r + i] = x.values[i * c + j];
        }
    }
    Ok(OnnxTensor {
        shape: vec![c, r],
        values,
    })
}

/// `a @ b` for `b` a matrix (or a vector) and `a` of any rank: the last
/// axis of `a` against the first of `b`.
fn matmul(a: &OnnxTensor, b: &OnnxTensor) -> Result<OnnxTensor> {
    let (k, n, b_vec) = match b.shape.as_slice() {
        [k] => (*k, 1, true),
        [k, n] => (*k, *n, false),
        _ => {
            return Err(Error::Data(format!(
                "MatMul supports a matrix or vector on the right, got shape {:?}",
                b.shape
            )));
        }
    };
    let ka = *a.shape.last().unwrap_or(&0);
    if ka != k {
        return Err(Error::Data(format!(
            "shapes {:?} and {:?} do not multiply",
            a.shape, b.shape
        )));
    }
    let m = a.len() / k.max(1);
    let mut values = vec![0.0; m * n];
    for i in 0..m {
        let row = &a.values[i * k..(i + 1) * k];
        let out = &mut values[i * n..(i + 1) * n];
        for (p, &x) in row.iter().enumerate() {
            let w = &b.values[p * n..(p + 1) * n];
            for (o, &wv) in out.iter_mut().zip(w) {
                *o += x * wv;
            }
        }
    }
    let mut shape = a.shape[..a.shape.len().saturating_sub(1)].to_vec();
    if !b_vec {
        shape.push(n);
    }
    Ok(OnnxTensor { shape, values })
}

fn softmax(x: &OnnxTensor, axis: usize) -> Result<OnnxTensor> {
    let shape = &x.shape;
    if axis >= shape.len() {
        return Err(Error::Data(format!(
            "Softmax axis {axis} for shape {shape:?}"
        )));
    }
    let inner: usize = shape[axis + 1..].iter().product();
    let size = shape[axis];
    let outer = x.len() / (size * inner).max(1);
    let mut values = x.values.clone();
    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| o * size * inner + j * inner + i;
            let max = (0..size)
                .map(|j| x.values[at(j)])
                .fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = (0..size).map(|j| (x.values[at(j)] - max).exp()).sum();
            for j in 0..size {
                values[at(j)] = (x.values[at(j)] - max).exp() / total;
            }
        }
    }
    Ok(OnnxTensor {
        shape: shape.clone(),
        values,
    })
}

fn concat(parts: &[&OnnxTensor], axis: usize) -> Result<OnnxTensor> {
    let first = parts
        .first()
        .ok_or_else(|| Error::Data("Concat of nothing".into()))?;
    let rank = first.shape.len();
    if axis >= rank {
        return Err(Error::Data(format!("Concat axis {axis} for rank {rank}")));
    }
    for p in parts {
        if p.shape.len() != rank
            || p.shape
                .iter()
                .zip(&first.shape)
                .enumerate()
                .any(|(k, (a, b))| k != axis && a != b)
        {
            return Err(Error::Data(format!(
                "shapes {:?} and {:?} do not concatenate on axis {axis}",
                first.shape, p.shape
            )));
        }
    }
    let outer: usize = first.shape[..axis].iter().product();
    let mut shape = first.shape.clone();
    shape[axis] = parts.iter().map(|p| p.shape[axis]).sum();
    let mut values = Vec::with_capacity(shape.iter().product());
    for o in 0..outer {
        for p in parts {
            let chunk = p.len() / outer.max(1);
            values.extend_from_slice(&p.values[o * chunk..(o + 1) * chunk]);
        }
    }
    Ok(OnnxTensor { shape, values })
}

fn gather(data: &OnnxTensor, idx: &OnnxTensor, axis: usize) -> Result<OnnxTensor> {
    if axis >= data.shape.len() {
        return Err(Error::Data(format!(
            "Gather axis {axis} for shape {:?}",
            data.shape
        )));
    }
    let size = data.shape[axis];
    let inner: usize = data.shape[axis + 1..].iter().product();
    let outer: usize = data.shape[..axis].iter().product();
    let indices = idx
        .values
        .iter()
        .map(|&v| {
            let k = v as i64;
            let k = if k < 0 { k + size as i64 } else { k };
            if v.fract() != 0.0 || k < 0 || k >= size as i64 {
                Err(Error::Data(format!(
                    "index {v} is not a whole number in 0..{size}"
                )))
            } else {
                Ok(k as usize)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let mut values = Vec::with_capacity(outer * indices.len() * inner);
    for o in 0..outer {
        for &k in &indices {
            let start = o * size * inner + k * inner;
            values.extend_from_slice(&data.values[start..start + inner]);
        }
    }
    let mut shape = data.shape[..axis].to_vec();
    shape.extend_from_slice(&idx.shape);
    shape.extend_from_slice(&data.shape[axis + 1..]);
    Ok(OnnxTensor { shape, values })
}

fn reshape(x: &OnnxTensor, target: &[i64]) -> Result<OnnxTensor> {
    let mut shape: Vec<usize> = Vec::with_capacity(target.len());
    let mut free = None;
    for (k, &d) in target.iter().enumerate() {
        match d {
            -1 => {
                free = Some(k);
                shape.push(1);
            }
            0 => shape.push(*x.shape.get(k).ok_or_else(|| {
                Error::Data(format!("Reshape copies axis {k} of shape {:?}", x.shape))
            })?),
            d if d > 0 => shape.push(d as usize),
            d => return Err(Error::Data(format!("Reshape to size {d}"))),
        }
    }
    let known: usize = shape.iter().product();
    if let Some(k) = free {
        shape[k] = x.len() / known.max(1);
    }
    OnnxTensor::new(shape, x.values.clone())
}

fn reduce(x: &OnnxTensor, dims: &[usize], keepdims: bool, mean: bool) -> Result<OnnxTensor> {
    let rank = x.shape.len();
    let dims: Vec<usize> = if dims.is_empty() {
        (0..rank).collect()
    } else {
        dims.to_vec()
    };
    if let Some(&bad) = dims.iter().find(|&&d| d >= rank) {
        return Err(Error::Data(format!(
            "reduce over axis {bad} of rank {rank}"
        )));
    }
    let out_full: Vec<usize> = (0..rank)
        .map(|k| if dims.contains(&k) { 1 } else { x.shape[k] })
        .collect();
    let (ti, to) = (strides(&x.shape), strides(&out_full));
    let mut values = vec![0.0; out_full.iter().product()];
    for (i, &v) in x.values.iter().enumerate() {
        let mut o = 0;
        for k in 0..rank {
            if !dims.contains(&k) {
                o += ((i / ti[k]) % x.shape[k]) * to[k];
            }
        }
        values[o] += v;
    }
    if mean {
        let count: usize = dims.iter().map(|&d| x.shape[d]).product();
        for v in &mut values {
            *v /= count as f64;
        }
    }
    let shape = if keepdims {
        out_full
    } else {
        (0..rank)
            .filter(|k| !dims.contains(k))
            .map(|k| x.shape[k])
            .collect()
    };
    if shape.is_empty() {
        return Ok(OnnxTensor::scalar(values[0]));
    }
    Ok(OnnxTensor { shape, values })
}

/// How the network's output is read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputScale {
    /// The output is the mean.
    Mean,
    /// The output is on the link's scale, added to the design's offset:
    /// `μ = g⁻¹(offset + output)` (a CANN's correction, or a network
    /// trained with an exposure offset).
    Link(Link),
}

/// An [`OnnxModel`] behind the model interface: each graph input bound to
/// design columns, one output read as each row's mean, and a family for
/// predictive draws.
///
/// A rank-2 input takes its columns in order (`[rows, columns]`); a
/// rank-1 input takes one column. An integer input (an embedding's
/// category codes) must hold whole numbers.
///
/// ```
/// use prospicio_models::{Design, Family, Fitted};
/// use prospicio_nn::onnx::{OnnxModel, OnnxPredictor, OutputScale};
///
/// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../validation/data/onnx/mlp.onnx");
/// let net = OnnxPredictor::new(OnnxModel::load(path).unwrap(), Family::Poisson)
///     .input("x", &["a", "b", "c", "d"])
///     .scale(OutputScale::Mean);
/// let cols = vec![vec![0.1, 0.4]; 4];
/// let design = Design::new(vec!["a".into(), "b".into(), "c".into(), "d".into()], cols).unwrap();
/// let mu = net.predict(&design).unwrap();
/// assert!(mu.len() == 2 && mu.iter().all(|&m| m > 0.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxPredictor {
    pub model: OnnxModel,
    /// Each graph input's design columns, in the graph's input order once
    /// all are bound.
    pub bindings: Vec<(String, Vec<String>)>,
    /// Which output is the mean (0 for the first).
    pub output: usize,
    pub scale: OutputScale,
    pub family: Family,
    /// The family's dispersion for predictive draws.
    pub dispersion: f64,
}

impl OnnxPredictor {
    /// The model with no inputs bound yet, its first output read as the
    /// mean, and dispersion 1.
    pub fn new(model: OnnxModel, family: Family) -> Self {
        Self {
            model,
            bindings: Vec::new(),
            output: 0,
            scale: OutputScale::Mean,
            family,
            dispersion: 1.0,
        }
    }

    /// Binds graph input `name` to design columns `columns`.
    pub fn input(mut self, name: &str, columns: &[&str]) -> Self {
        self.bindings.retain(|(n, _)| n != name);
        self.bindings.push((
            name.to_string(),
            columns.iter().map(|c| c.to_string()).collect(),
        ));
        self
    }

    pub fn output(mut self, index: usize) -> Self {
        self.output = index;
        self
    }

    pub fn scale(mut self, scale: OutputScale) -> Self {
        self.scale = scale;
        self
    }

    pub fn dispersion(mut self, dispersion: f64) -> Self {
        self.dispersion = dispersion;
        self
    }

    /// The tensors for `design`'s rows, one per graph input.
    fn tensors(&self, design: &Design) -> Result<Vec<OnnxTensor>> {
        let n = design.n_rows();
        let names = design.names();
        self.model
            .inputs()
            .iter()
            .map(|spec| {
                let cols = self
                    .bindings
                    .iter()
                    .find(|(name, _)| *name == spec.name)
                    .map(|(_, c)| c)
                    .ok_or_else(|| {
                        Error::Data(format!("graph input {} is not bound to columns", spec.name))
                    })?;
                let columns = cols
                    .iter()
                    .map(|c| {
                        let j = names.iter().position(|m| m == c).ok_or_else(|| {
                            Error::Data(format!("the design has no column {c}"))
                        })?;
                        Ok(design.column(j))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if spec.integer
                    && let Some(bad) = columns.iter().flat_map(|c| c.iter()).find(|v| v.fract() != 0.0)
                {
                    return Err(Error::Data(format!(
                        "graph input {} takes whole numbers (category codes), got {bad}",
                        spec.name
                    )));
                }
                match spec.rank {
                    1 if columns.len() == 1 => OnnxTensor::new(vec![n], columns[0].to_vec()),
                    2 => {
                        let k = columns.len();
                        let mut values = Vec::with_capacity(n * k);
                        for i in 0..n {
                            values.extend(columns.iter().map(|c| c[i]));
                        }
                        OnnxTensor::new(vec![n, k], values)
                    }
                    r => Err(Error::Data(format!(
                        "graph input {} has rank {r}; bind one column to a rank-1 input or several to a rank-2 one",
                        spec.name
                    ))),
                }
            })
            .collect()
    }
}

impl Fitted for OnnxPredictor {
    /// Runs the graph on the design's rows and reads the bound output: one
    /// value per row (`[rows]` or `[rows, 1]`), as the mean or through the
    /// link's inverse with the design's offset.
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        let n = design.n_rows();
        let outputs = self.model.run(&self.tensors(design)?)?;
        let out = outputs.get(self.output).ok_or_else(|| {
            Error::Data(format!(
                "output {} of a model with {}",
                self.output,
                outputs.len()
            ))
        })?;
        if out.len() != n {
            return Err(Error::Data(format!(
                "the output has shape {:?}, not one value per row of {n}",
                out.shape
            )));
        }
        Ok(match self.scale {
            OutputScale::Mean => out.values.clone(),
            OutputScale::Link(link) => out
                .values
                .iter()
                .zip(design.offset())
                .map(|(v, o)| link.inverse(o + v))
                .collect(),
        })
    }

    /// Process draws from the family at the predicted means
    /// ([`prospicio_models::simulate::from_means`]), with the design's
    /// weights: an imported network carries no parameter uncertainty.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let mu = self.predict(design)?;
        let provenance = Provenance::new("onnx")
            .version("prospicio-nn", env!("CARGO_PKG_VERSION"))
            .param("family", self.family.name())
            .param("ops", self.model.ops().join(" "));
        prospicio_models::simulate::from_means(
            self.family,
            &[mu],
            &[self.dispersion],
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

    fn t(shape: &[usize], v: &[f64]) -> OnnxTensor {
        OnnxTensor::new(shape.to_vec(), v.to_vec()).unwrap()
    }

    #[test]
    fn kernels_by_hand() {
        let a = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(&[3], &[10.0, 20.0, 30.0]);
        assert_eq!(
            broadcast(&a, &b, Binary::Add).unwrap().values,
            [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
        let col = t(&[2, 1], &[1.0, -1.0]);
        assert_eq!(
            broadcast(&a, &col, Binary::Mul).unwrap().values,
            [1.0, 2.0, 3.0, -4.0, -5.0, -6.0]
        );
        assert!(broadcast(&a, &t(&[2], &[1.0, 2.0]), Binary::Add).is_err());
        let w = t(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let m = matmul(&a, &w).unwrap();
        assert_eq!(
            (m.shape.clone(), m.values),
            (vec![2, 2], vec![4.0, 5.0, 10.0, 11.0])
        );
        assert_eq!(
            transpose(&a).unwrap().values,
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        let s = softmax(&t(&[1, 2], &[0.0, 0.0]), 1).unwrap();
        assert_eq!(s.values, [0.5, 0.5]);
        let c = concat(&[&a, &col], 1).unwrap();
        assert_eq!(c.shape, [2, 4]);
        assert_eq!(c.values, [1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, -1.0]);
        let e = t(&[3, 2], &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]);
        let g = gather(&e, &t(&[3], &[2.0, 0.0, -1.0]), 0).unwrap();
        assert_eq!(g.shape, [3, 2]);
        assert_eq!(g.values, [2.0, 2.1, 0.0, 0.1, 2.0, 2.1]);
        assert!(gather(&e, &t(&[1], &[3.0]), 0).is_err());
        assert!(gather(&e, &t(&[1], &[0.5]), 0).is_err());
        let r = reduce(&a, &[1], true, false).unwrap();
        assert_eq!((r.shape, r.values), (vec![2, 1], vec![6.0, 15.0]));
        let r = reduce(&a, &[0], false, true).unwrap();
        assert_eq!((r.shape, r.values), (vec![3], vec![2.5, 3.5, 4.5]));
        assert_eq!(reshape(&a, &[-1]).unwrap().shape, [6]);
        assert_eq!(reshape(&a, &[0, 3, 1]).unwrap().shape, [2, 3, 1]);
        assert!((Unary::Softplus.apply(800.0) - 800.0).abs() < 1e-12);
        assert!((Unary::Softplus.apply(0.0) - 2.0_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn refuses_what_it_cannot_run() {
        assert!(OnnxModel::from_bytes(b"not a model").is_err());
        assert!(OnnxModel::load("/nonexistent/model.onnx").is_err());
    }
}
