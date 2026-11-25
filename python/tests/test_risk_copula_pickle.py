"""Copulas pickle, the Gaussian and t ones bit for bit through their factor (#978)."""

import pickle

from prospicio.risk import (
    ArchimedeanCopula,
    GaussianCopula,
    NestedArchimedeanCopula,
    PairCopula,
    RVineCopula,
    StudentTCopula,
    VineCopula,
)

R = [[1.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]]


def _copulas():
    g = PairCopula("gaussian", 0.5)
    return [
        GaussianCopula(R),
        StudentTCopula(R, 4.5),
        ArchimedeanCopula("clayton", 2.0, 3),
        NestedArchimedeanCopula("gumbel", (1.5, [0, (3.0, [1, 2])])),
        VineCopula(
            "d",
            [0, 1, 2],
            [[PairCopula("gumbel", 2.0), PairCopula("clayton", 1.5, 90)], [PairCopula("frank", 3.0)]],
        ),
        RVineCopula([[2, 3, 3], [3, 2, 0], [1, 0, 0]], [[g, g], [g]]),
    ]


def test_copulas_pickle_and_sample_the_same():
    for c in _copulas():
        back = pickle.loads(pickle.dumps(c))
        assert type(back) is type(c)
        assert back.dim == c.dim
        assert back.sample(50, 11) == c.sample(50, 11), type(c).__name__


def test_pair_copulas_pickle():
    for p in (PairCopula("clayton", 2.0, 270), PairCopula("student", 0.4, nu=5.0), PairCopula("gaussian", -0.2)):
        back = pickle.loads(pickle.dumps(p))
        assert (back.family, back.parameter, back.rotation, back.nu) == (p.family, p.parameter, p.rotation, p.nu)
        assert back.cdf(0.3, 0.7) == p.cdf(0.3, 0.7)
