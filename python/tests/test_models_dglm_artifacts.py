"""Double GLM fits round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import Design, DoubleGlm, DoubleGlmFit


def data():
    urban = [float(i % 2) for i in range(24)]
    x = [i / 8 for i in range(24)]
    y = [5 + (i % 2) * 3 + ((i * 7) % 5) * (1 + i % 2) for i in range(24)]
    return Design([[1.0] * 24, urban, x], ["(Intercept)", "urban", "x"]), [float(v) for v in y]


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_round_trip(method):
    d, y = data()
    fit = DoubleGlm("gamma", link="log", method=method).fit(d, d, y)
    for back in (DoubleGlmFit.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert isinstance(back, DoubleGlmFit)
        assert back.coefficients == fit.coefficients
        assert back.dispersion_coefficients == fit.dispersion_coefficients
        assert back.predict(d) == fit.predict(d)
        assert back.to_json() == fit.to_json()


def test_refuses_other_documents():
    d, y = data()
    text = DoubleGlm("gamma", link="log").fit(d, d, y).to_json()
    with pytest.raises(ValueError):
        DoubleGlmFit.from_json(text.replace("risk_rs.double_glm_fit", "risk_rs.glm_fit"))
    with pytest.raises(ValueError):
        DoubleGlmFit.from_json(text.replace('"format_version":1', '"format_version":2'))
