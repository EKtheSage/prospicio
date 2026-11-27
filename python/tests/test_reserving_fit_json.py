"""JSON artifacts and pickling of reserving fits (#978)."""

import json
import math
import pickle

import pytest

from prospicio.reserving import ChainLadder, ChainLadderFit, Triangle


def two_segments():
    return Triangle.from_long(
        [2020, 2020, 2020, 2021, 2021, 2022] * 2,
        [12, 24, 36, 12, 24, 12] * 2,
        {"paid": [100.0, 150.0, 160.0, 110.0, 170.0, 120.0,
                  10.0, 20.0, 25.0, 12.0, 22.0, 14.0]},
        keys={"lob": ["Auto"] * 6 + ["Home"] * 6},
    )


def test_a_chain_ladder_fit_round_trips_through_json():
    fit = ChainLadder().fit(two_segments(), "paid")
    text = fit.to_json()
    back = ChainLadderFit.from_json(text)
    assert back.to_json() == text
    assert back.index == fit.index
    assert back.ultimate == fit.ultimate
    assert back.segment(lob="Home").ldf == fit.segment(lob="Home").ldf
    document = json.loads(text)
    assert document["format"] == "risk_rs.chain_ladder_fits"
    assert document["format_version"] == 1


def test_a_chain_ladder_fit_pickles():
    fit = ChainLadder().fit(two_segments(), "paid")
    back = pickle.loads(pickle.dumps(fit))
    assert isinstance(back, ChainLadderFit)
    assert back.to_json() == fit.to_json()
    assert all(math.isclose(a, b) for a, b in zip(back.reserve, fit.reserve))


def test_another_document_is_refused():
    with pytest.raises(ValueError):
        ChainLadderFit.from_json('{"format": "risk_rs.triangle", "format_version": 1}')
    with pytest.raises(ValueError):
        ChainLadderFit.from_json("not json")


def square():
    return Triangle.from_long(
        [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
        [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
        {"paid": [100.0, 150.0, 165.0, 170.0, 110.0, 172.0, 185.0, 120.0, 175.0, 130.0]},
    )


def test_a_mack_fit_round_trips_and_pickles():
    from prospicio.reserving import Mack, MackFit

    fit = Mack().fit(square(), "paid")
    text = fit.to_json()
    back = MackFit.from_json(text)
    assert back.to_json() == text
    assert back.standard_error == fit.standard_error
    assert back.total_standard_error == fit.total_standard_error
    assert json.loads(text)["format"] == "risk_rs.mack_fits"
    assert pickle.loads(pickle.dumps(fit)).to_json() == text
    with pytest.raises(ValueError):
        MackFit.from_json(ChainLadder().fit(square(), "paid").to_json())


def with_premium():
    return Triangle.from_long(
        [2020, 2020, 2020, 2021, 2021, 2022],
        [12, 24, 36, 12, 24, 12],
        {"paid": [100.0, 150.0, 160.0, 110.0, 170.0, 120.0],
         "premium": [250.0] * 3 + [260.0] * 2 + [280.0]},
    )


@pytest.mark.parametrize("method", ["bf", "benktander", "expected_loss"])
def test_an_expected_loss_fit_round_trips_and_pickles(method):
    from prospicio.reserving import (
        Benktander, BornhuetterFerguson, ExpectedLoss, ExpectedLossFit)

    model = {"bf": BornhuetterFerguson(apriori=0.6),
             "benktander": Benktander(apriori=0.6, n_iters=2),
             "expected_loss": ExpectedLoss(apriori=0.6)}[method]
    fit = model.fit(with_premium(), "paid", "premium")
    text = fit.to_json()
    assert ExpectedLossFit.from_json(text).ultimate == fit.ultimate
    assert pickle.loads(pickle.dumps(fit)).to_json() == text


def test_a_cape_cod_fit_round_trips_and_pickles():
    from prospicio.reserving import CapeCod, CapeCodFit, ExpectedLossFit

    fit = CapeCod(trend=0.05).fit(with_premium(), "paid", "premium")
    text = fit.to_json()
    back = CapeCodFit.from_json(text)
    assert back.trended_apriori == fit.trended_apriori
    assert pickle.loads(pickle.dumps(fit)).to_json() == text
    with pytest.raises(ValueError):
        ExpectedLossFit.from_json(text)


@pytest.mark.parametrize("name", ["OdpBootstrap", "MackBootstrap"])
def test_a_bootstrap_fit_round_trips_and_pickles(name):
    import prospicio.reserving as r

    model = getattr(r, name)(n_sims=100, seed=1)
    fit = model.fit(square(), "paid")
    text = fit.to_json()
    back = getattr(r, name + "Fit").from_json(text)
    assert back.to_json() == text
    assert back.reserves.mean() == fit.reserves.mean()
    assert pickle.loads(pickle.dumps(fit)).to_json() == text
