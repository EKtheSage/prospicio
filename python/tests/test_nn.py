"""Neural networks: the CANN, the attention CANN and ONNX import (prospicio-nn)."""

import csv
import math
import pathlib
import random

import pytest

from prospicio.models import (
    AttentionCann,
    AttentionCannFit,
    Cann,
    CannFit,
    Design,
    OnnxModel,
    OnnxPredictor,
    TokenAttention,
    deviance,
)

VALIDATION = pathlib.Path(__file__).resolve().parents[2] / "validation"


def poisson_counts(n, seed):
    """Counts with mean exp(1 - x^2): a curve a GLM with a constant misses."""
    rng = random.Random(seed)
    x = [rng.uniform(-2, 2) for _ in range(n)]
    y = []
    for v in x:
        lam, k, p = math.exp(1 - v * v), 0, 1.0
        u = rng.random()
        while True:  # inverse transform
            q = math.exp(-lam) * lam**k / math.factorial(k)
            if u < q or p - q <= 0:
                break
            u -= q
            p -= q
            k += 1
        y.append(float(k))
    return x, y


def test_cann_learns_what_the_glm_misses_and_replays():
    x, y = poisson_counts(600, 1)
    base = math.log(sum(y) / len(y))
    d = Design([x], ["x"], offset=[base] * len(x))
    spec = Cann("poisson", hidden=[16, 8], epochs=60, learning_rate=1e-2, seed=3)
    fit = spec.fit(d, y)
    assert isinstance(fit, CannFit)
    glm = deviance("poisson", y, [math.exp(base)] * len(y))
    assert deviance("poisson", y, fit.fitted) < 0.8 * glm
    assert fit.predict(d) == pytest.approx(fit.fitted, rel=1e-12)
    # The correction is on the link scale: mu = exp(offset + NN(x)).
    c = fit.correction(d)
    assert fit.predict(d)[0] == pytest.approx(math.exp(base + c[0]), rel=1e-12)
    assert fit.dispersion == 1.0
    assert spec.fit(d, y).fitted == fit.fitted  # reproducible from the seed
    pd = fit.predict_distribution(d, 4_000, 9)
    assert pd.n_components == len(y)
    assert pd.total().mean() == pytest.approx(sum(fit.fitted), rel=0.02)


def test_cann_early_stopping_and_refusals():
    x, y = poisson_counts(300, 4)
    base = math.log(sum(y) / len(y))
    d = Design([x], ["x"], offset=[base] * len(x))
    fit = Cann(
        "poisson", hidden=[32, 32], epochs=400, batch_size=16, learning_rate=3e-2,
        validation_share=0.3, patience=5,
    ).fit(d, y)
    assert 1 <= fit.best_epoch <= len(fit.validation_history) < 400
    assert fit.validation_history[fit.best_epoch - 1] == min(fit.validation_history)
    with pytest.raises(ValueError):
        Cann("poisson", link="identity").fit(d, y)
    with pytest.raises(ValueError):
        Cann("poisson", batch_size=0).fit(d, y)
    with pytest.raises(ValueError):
        fit.predict(Design([x], ["z"]))


def test_attention_cann_reads_factor_tokens():
    rng = random.Random(5)
    n = 300
    region = [rng.choice("ABC") for _ in range(n)]
    age = [rng.uniform(-1, 1) for _ in range(n)]
    mean = [math.exp(0.5 * a + (0.4 if r == "B" else 0.0)) for a, r in zip(age, region)]
    y = [m * rng.uniform(0.5, 1.5) for m in mean]
    cols = [age, [float(r == "B") for r in region], [float(r == "C") for r in region]]
    d = Design(cols, ["age", "region[B]", "region[C]"], offset=[0.0] * n)
    spec = AttentionCann("gamma", d_model=8, heads=2, layers=1, feed_forward=16, epochs=8, seed=2)
    fit = spec.fit(d, y)
    assert isinstance(fit, AttentionCannFit)
    assert fit.tokens == ["age", "region"]
    assert len(fit.predict(d)) == n
    assert fit.dispersion > 0
    a = fit.attention(d)
    assert isinstance(a, TokenAttention)
    assert (a.rows, a.heads, a.tokens) == (n, 2, ["age", "region"])
    for r in range(3):
        for h in range(2):
            w = [a.get(r, h, t) for t in range(2)]
            assert all(0 <= v <= 1 for v in w) and sum(w) <= 1 + 1e-12
    m = a.mean()
    assert len(m) == 2 and sum(m) <= 1 + 1e-12
    assert a.mean([0, 1]) == pytest.approx(
        [sum(a.get(r, h, t) for r in (0, 1) for h in (0, 1)) / 4 for t in range(2)], rel=1e-12
    )
    with pytest.raises(ValueError):
        a.get(n, 0, 0)
    with pytest.raises(ValueError):
        a.mean([n])
    with pytest.raises(ValueError):
        AttentionCann("gamma", d_model=7, heads=2).fit(d, y)
    with pytest.raises(ValueError):
        AttentionCann("gamma", layers=0, epochs=1).fit(d, y).attention(d)


def reference():
    with open(VALIDATION / "reference" / "onnx_outputs.csv") as f:
        rows = list(csv.DictReader(f))
    nums = lambda s: [float(v) for v in s.split("|") if v]
    return [(r["model"], nums(r["inputs"]), nums(r["codes"]), float(r["expected"])) for r in rows]


def test_onnx_models_match_onnxruntime():
    cases = reference()
    assert len(cases) == 36
    for name in ["mlp", "embedding", "gemm"]:
        m = OnnxModel(str(VALIDATION / "data" / "onnx" / f"{name}.onnx"))
        rows = [c for c in cases if c[0] == name]
        n, width = len(rows), len(rows[0][1])
        inputs = [([n, width], [v for r in rows for v in r[1]])]
        if name == "embedding":
            inputs.append(([n], [r[2][0] for r in rows]))
        (shape, values), *_ = m.run(inputs)
        assert len(values) == n
        for r, got in zip(rows, values):
            assert got == pytest.approx(r[3], rel=1e-5, abs=1e-6), name
    mlp = OnnxModel(str(VALIDATION / "data" / "onnx" / "mlp.onnx"))
    assert mlp.ops == ["Gemm", "Relu", "Gemm", "Exp"]
    e = OnnxModel.from_bytes((VALIDATION / "data" / "onnx" / "embedding.onnx").read_bytes())
    assert [i["name"] for i in e.inputs] == ["x_num", "x_cat"]
    assert [i["integer"] for i in e.inputs] == [False, True]
    assert e.inputs[0]["shape"] == [None, 3]
    with pytest.raises(ValueError):
        mlp.run([([1, 5], [0.0] * 5)])
    with pytest.raises(ValueError):
        mlp.run([([2, 4], [0.0] * 5)])
    with pytest.raises(ValueError):
        OnnxModel.from_bytes(b"not an onnx file")


def test_onnx_predictor_binds_design_columns_and_draws():
    rows = [c for c in reference() if c[0] == "embedding"]
    n = len(rows)
    cols = [[r[1][j] for r in rows] for j in range(3)] + [[r[2][0] for r in rows]]
    names = ["a", "b", "c", "region"]
    model = OnnxModel(str(VALIDATION / "data" / "onnx" / "embedding.onnx"))
    inputs = {"x_num": ["a", "b", "c"], "x_cat": ["region"]}
    net = OnnxPredictor(model, "gamma", inputs, dispersion=0.5)
    mu = net.predict(Design(cols, names))
    assert mu == pytest.approx([r[3] for r in rows], rel=1e-5)
    pd = net.predict_distribution(Design(cols, names), 20_000, 3)
    assert pd.n_components == n
    assert pd.total().mean() == pytest.approx(sum(mu), rel=0.02)
    log_net = OnnxPredictor(model, "gamma", inputs, scale="link", link="log")
    lm = log_net.predict(Design(cols, names, offset=[0.3] * n))
    assert lm == pytest.approx([math.exp(0.3 + m) for m in mu], rel=1e-12)
    with pytest.raises(ValueError):
        OnnxPredictor(model, "gamma", {"x_num": ["a", "b", "c"]}).predict(Design(cols, names))
    with pytest.raises(ValueError):
        OnnxPredictor(model, "gamma", {**inputs, "x_cat": ["zone"]}).predict(Design(cols, names))
    bad = [c[:] for c in cols]
    bad[3][0] = 0.5
    with pytest.raises(ValueError):
        net.predict(Design(bad, names))
    with pytest.raises(ValueError):
        OnnxPredictor(model, "gamma", inputs, scale="mean", link="log")
    with pytest.raises(ValueError):
        OnnxPredictor(model, "gamma", inputs, scale="probability")
