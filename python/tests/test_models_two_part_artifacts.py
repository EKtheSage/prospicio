"""Zero-inflated and hurdle fits round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import Design, Hurdle, TwoPartFit, ZeroInflated


def data():
    x = [i % 4 / 3 for i in range(120)]
    y = [0.0 if i % 3 == 0 else float((i * 13 % 11) * (i % 5 + 1) // 2) for i in range(120)]
    return Design([[1.0] * 120, x], ["(Intercept)", "x"]), y


@pytest.mark.parametrize(
    "spec",
    [ZeroInflated(), ZeroInflated("negative_binomial"), Hurdle(), Hurdle("negative_binomial", "probit")],
)
def test_round_trip(spec):
    d, y = data()
    fit = spec.fit(d, y)
    for back in (TwoPartFit.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert isinstance(back, TwoPartFit)
        assert back.kind == fit.kind
        assert back.count_coefficients == fit.count_coefficients
        assert back.theta == fit.theta
        assert back.predict(d) == fit.predict(d)
        assert back.to_json() == fit.to_json()


def test_refuses_other_documents():
    d, y = data()
    text = ZeroInflated().fit(d, y).to_json()
    with pytest.raises(ValueError):
        TwoPartFit.from_json(text.replace("risk_rs.two_part_fit", "risk_rs.glm_fit"))
    with pytest.raises(ValueError):
        TwoPartFit.from_json(text.replace('"format_version":1', '"format_version":2'))
