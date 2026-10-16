"""Regenerate validation/data/onnx/*.onnx and validation/reference/onnx_outputs.csv.

    uv run --no-project --with onnx==1.23.2 --with onnxruntime==1.31.0 --with numpy==2.5.4 \
        python validation/scripts/onnx_models.py

The reference for prospicio_nn::onnx: three small networks of the kinds a
PyTorch tabular model exports, built with onnx.helper (no PyTorch needed),
in float32 as PyTorch exports them, and run by onnxruntime, whose results
are the semantics to match:

- mlp.onnx: x [N, 4] -> Gemm (transB, as torch.nn.Linear) -> Relu ->
  Gemm -> Exp, output [N, 1]: a mean, through a log link.
- embedding.onnx: x_num [N, 3] float and x_cat [N] int64 -> Gather
  (an embedding of 5 levels in 2 dimensions) -> Concat with x_num ->
  MatMul + Add -> Tanh -> MatMul + Add -> Softplus -> Squeeze, output [N].
- gemm.onnx: x [N, 3] -> Gemm (alpha 0.5, beta 2, a bias of shape [3])
  -> Sigmoid -> Mul by a constant -> Sub -> Div -> Flatten -> Identity ->
  ReduceSum over axis 1 (keepdims), output [N, 1]: every elementwise op
  and a reduction.

Inputs are fixed (a seeded generator), so the outputs replay; the CSV has
one row per (model, row) with the expected output. Tolerance: 1e-5
relative, the float32 rounding onnxruntime carries against a float64
evaluation of the same graph.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation/data/onnx"
OUT = ROOT / "validation/reference/onnx_outputs.csv"
OPSET = 17
N = 12


def init(name, array):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def save(graph, name):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, DATA / name)
    return ort.InferenceSession(str(DATA / name), providers=["CPUExecutionProvider"])


def main():
    rng = np.random.default_rng(20261016)
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []

    # mlp.onnx
    w1, b1 = rng.normal(0, 0.5, (8, 4)), rng.normal(0, 0.1, 8)
    w2, b2 = rng.normal(0, 0.3, (1, 8)), np.array([-1.0])
    g = helper.make_graph(
        [
            helper.make_node("Gemm", ["x", "w1", "b1"], ["h1"], transB=1),
            helper.make_node("Relu", ["h1"], ["a1"]),
            helper.make_node("Gemm", ["a1", "w2", "b2"], ["eta"], transB=1),
            helper.make_node("Exp", ["eta"], ["mu"]),
        ],
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 4])],
        [helper.make_tensor_value_info("mu", TensorProto.FLOAT, ["N", 1])],
        [init("w1", w1), init("b1", b1), init("w2", w2), init("b2", b2)],
    )
    x = rng.normal(0, 1, (N, 4)).astype(np.float32)
    out = save(g, "mlp.onnx").run(None, {"x": x})[0]
    for i in range(N):
        rows.append(("mlp", i, list(x[i]), [], float(out[i, 0])))

    # embedding.onnx
    emb = rng.normal(0, 0.7, (5, 2))
    wa, ba = rng.normal(0, 0.5, (5, 6)), rng.normal(0, 0.1, 6)
    wb, bb = rng.normal(0, 0.5, (6, 1)), np.array([0.2])
    g = helper.make_graph(
        [
            helper.make_node("Gather", ["emb", "x_cat"], ["e"], axis=0),
            helper.make_node("Concat", ["x_num", "e"], ["z"], axis=1),
            helper.make_node("MatMul", ["z", "wa"], ["za"]),
            helper.make_node("Add", ["za", "ba"], ["zb"]),
            helper.make_node("Tanh", ["zb"], ["h"]),
            helper.make_node("MatMul", ["h", "wb"], ["ha"]),
            helper.make_node("Add", ["ha", "bb"], ["hb"]),
            helper.make_node("Softplus", ["hb"], ["s"]),
            helper.make_node("Squeeze", ["s", "axes"], ["y"]),
        ],
        "embedding",
        [
            helper.make_tensor_value_info("x_num", TensorProto.FLOAT, ["N", 3]),
            helper.make_tensor_value_info("x_cat", TensorProto.INT64, ["N"]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N"])],
        [
            init("emb", emb),
            init("wa", wa),
            init("ba", ba),
            init("wb", wb),
            init("bb", bb),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
        ],
    )
    x_num = rng.normal(0, 1, (N, 3)).astype(np.float32)
    x_cat = (np.arange(N) % 5).astype(np.int64)
    out = save(g, "embedding.onnx").run(None, {"x_num": x_num, "x_cat": x_cat})[0]
    for i in range(N):
        rows.append(("embedding", i, list(x_num[i]), [int(x_cat[i])], float(out[i])))

    # gemm.onnx
    w, c = rng.normal(0, 0.5, (3, 3)), rng.normal(0, 0.5, 3)
    g = helper.make_graph(
        [
            helper.make_node("Gemm", ["x", "w", "c"], ["g"], alpha=0.5, beta=2.0),
            helper.make_node("Sigmoid", ["g"], ["s"]),
            helper.make_node("Mul", ["s", "k"], ["m"]),
            helper.make_node("Sub", ["m", "one"], ["d"]),
            helper.make_node("Div", ["d", "two"], ["q"]),
            helper.make_node("Flatten", ["q"], ["f"], axis=1),
            helper.make_node("Identity", ["f"], ["i"]),
            helper.make_node("ReduceSum", ["i", "raxes"], ["y"], keepdims=1),
        ],
        "gemm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])],
        [
            init("w", w),
            init("c", c),
            init("k", [3.0, 1.5, 0.5]),
            init("one", [1.0]),
            init("two", [2.0]),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "raxes"),
        ],
    )
    x = rng.normal(0, 1, (N, 3)).astype(np.float32)
    out = save(g, "gemm.onnx").run(None, {"x": x})[0]
    for i in range(N):
        rows.append(("gemm", i, list(x[i]), [], float(out[i, 0])))

    with open(OUT, "w") as f:
        f.write("model,row,inputs,codes,expected,abs_tol,rel_tol,source\n")
        for model, i, xs, codes, y in rows:
            xs = "|".join(repr(float(v)) for v in xs)
            cs = "|".join(str(v) for v in codes)
            f.write(f"{model},{i},{xs},{cs},{y!r},1e-6,1e-5,onnxruntime {ort.__version__}\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
