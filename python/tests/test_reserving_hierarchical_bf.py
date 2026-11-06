"""HierarchicalBf, the hierarchical Bayesian Bornhuetter-Ferguson over programs
(docs/design/reserving-bayes.md, decision 30).

The model's limits and its density are checked in the Rust crate and in
validation/tests; these tests check the binding: settings, the fit's fields
and frames on a small panel of three programs (one of them thin, with two
origins), that program reserves add up to the joint total, replay by seed,
the pattern prior and the errors.
"""

import math
import warnings

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.reserving import (
    HierarchicalBf,
    HierarchicalBfFit,
    HierarchicalBfOneYearFit,
    HierarchicalBfProgram,
    PatternPrior,
    Triangle,
)

QUICK = dict(chains=2, tune=300, draws=300, n_sims=600)

# Programs a (four origins), b (three) and c (two, a thin one), on ages 12 to 48.
PROGRAMS = {
    "a": {
        2020: ([60.0, 100.0, 125.0, 135.0], 200.0),
        2021: ([66.0, 108.0, 133.0], 210.0),
        2022: ([70.0, 115.0], 215.0),
        2023: ([72.0], 220.0),
    },
    "b": {
        2021: ([35.0, 60.0, 75.0], 120.0),
        2022: ([38.0, 64.0], 125.0),
        2023: ([40.0], 130.0),
    },
    "c": {
        2022: ([22.0, 38.0], 80.0),
        2023: ([25.0], 85.0),
    },
}


def panel(programs=PROGRAMS):
    program, origin, age, paid, premium = [], [], [], [], []
    for name, origins in programs.items():
        for year, (cells, prem) in origins.items():
            for k, value in enumerate(cells):
                program.append(name)
                origin.append(year)
                age.append(12 * (k + 1))
                paid.append(value)
                premium.append(prem)
    return Triangle.from_long(
        origin, age, {"paid": paid, "premium": premium}, keys={"program": program}
    )


def programs_by_name(programs=PROGRAMS):
    """Each program as a triangle of its own, as the dict form takes them."""
    return {name: panel({name: origins}) for name, origins in programs.items()}


@pytest.fixture(scope="module")
def tri():
    return panel()


@pytest.fixture(scope="module")
def fit(tri):
    model = HierarchicalBf(apriori=0.7, level_cv=0.3, seed=3, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(tri, "paid", "premium")


def test_settings_and_defaults():
    m = HierarchicalBf()
    assert (m.apriori, m.level_cv, m.cv, m.development_sd) == (1.0, None, 0.1, 10.0)
    assert (m.tau_scale, m.omega_scale, m.process) == (0.3, 0.5, "gamma")
    assert (m.chains, m.tune, m.draws, m.n_sims, m.seed) == (4, 1000, 1000, 10_000, 0)
    assert (m.target_accept, m.max_depth, m.pattern_prior) == (0.99, 10, None)
    m = HierarchicalBf(apriori=0.7, level_cv=0.2, tau_scale=0.1, process="none", seed=9)
    assert (m.apriori, m.level_cv, m.tau_scale, m.process, m.seed) == (0.7, 0.2, 0.1, "none", 9)
    assert repr(m).startswith("HierarchicalBf(apriori=0.7, level_cv=0.2, cv=0.1")


def test_fit_fields(fit):
    assert isinstance(fit, HierarchicalBfFit)
    assert fit.ages == [12, 24, 36, 48]
    assert fit.scale_per_premium > 0.0
    assert len(fit.programs) == 3
    assert all(isinstance(p, HierarchicalBfProgram) for p in fit.programs)
    assert [p.label for p in fit.programs] == ["a", "b", "c"]
    assert [p.key for p in fit.programs] == [["a"], ["b"], ["c"]]
    assert [len(p.origins) for p in fit.programs] == [4, 3, 2]

    for p in fit.programs:
        n = len(p.origins)
        for values in (p.exposure, p.latest, p.prior_ultimate, p.expected_ultimate):
            assert len(values) == n
        assert len(p.pattern) == 4 and sum(p.pattern) == pytest.approx(1.0)
        assert all(x > 0.0 for x in p.pattern)
        assert p.scale > 0.0 and p.loss_ratio > 0.0 and p.own_loss_ratio > 0.0
        assert math.isfinite(p.shrinkage)
        assert isinstance(p.reserves, PredictiveDistribution)
        assert p.reserves.n_sims == 600
        assert repr(p).startswith(f"HierarchicalBfProgram(label='{p.label}'")
    thin = fit.programs[2]
    assert thin.origins == ["2022", "2023"]
    assert thin.exposure == [80.0, 85.0] and thin.latest == [38.0, 25.0]
    assert thin.prior_ultimate == pytest.approx([0.7 * 80.0, 0.7 * 85.0])
    # A program's scale is the pooled scale per premium times its mean premium.
    assert thin.scale == pytest.approx(fit.scale_per_premium * (80.0 + 85.0) / 2)

    assert repr(fit).startswith("HierarchicalBfFit(programs=3, ages=4, draws=600")


def test_reserves_add_up(fit):
    r = fit.reserves
    assert isinstance(r, PredictiveDistribution)
    assert r.dims == ["program", "origin"]
    assert len(r.components()) == 9
    assert r.n_sims == 600
    assert sum(p.reserves.mean() for p in fit.programs) == pytest.approx(
        r.mean(), rel=1e-9
    )
    # Paired by posterior draw: each program's draws are the joint draws'
    # columns for that program, simulation by simulation.
    joint = r.draw_matrix()
    keys = [tuple(k) for k in r.components()]
    for p in fit.programs:
        columns = [j for j, k in enumerate(keys) if k[0] == p.label]
        assert len(columns) == len(p.origins)
        own = p.reserves.draw_matrix()
        assert len(own) == len(joint) == 600
        for row_own, row_joint in zip(own, joint):
            assert row_own == [row_joint[j] for j in columns]
    assert r.mean() > 0.0
    # The thin program has a reserve past its own last age.
    assert fit.programs[2].reserves.mean() > 0.0


def test_posterior_and_frames(fit):
    names = fit.parameter_names
    assert names[:2] == ["log_level", "log_tau"]
    assert "log_omega" in names
    draws = fit.draws
    assert fit.chains == 2
    assert len(draws) == 600 and all(len(row) == len(names) for row in draws)

    s = fit.summary()
    assert list(s.columns) == [
        "parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk", "ess_tail",
    ]
    assert list(s["parameter"]) == names

    h = fit.hyperparameters()
    assert list(h["parameter"]) == ["level", "tau", "omega"]
    assert (h["mean"] > 0).all() and (h["sd"] > 0).all()
    assert (h["q05"] > 0).all()

    f = fit.to_frame()
    assert list(f["program"]) == ["a", "b", "c"]
    assert list(f["origins"]) == [4, 3, 2]
    assert list(f.columns) == [
        "program", "origins", "exposure", "latest", "prior_ultimate",
        "expected_ultimate", "scale", "loss_ratio", "own_loss_ratio", "shrinkage",
        "mean", "std_dev",
    ]
    assert f["exposure"][2] == pytest.approx(165.0)
    assert (f["std_dev"] > 0).all()
    assert f["mean"].sum() == pytest.approx(fit.reserves.mean(), rel=1e-9)

    d = fit.diagnostics
    assert set(d) == {"max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings"}


def test_replays_by_seed(tri):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HierarchicalBf(seed=5, **QUICK)
        a = model.fit(tri, "paid", "premium")
        b = model.fit(tri, "paid", "premium")
        c = HierarchicalBf(seed=6, **QUICK).fit(tri, "paid", "premium")
    assert a.draws == b.draws
    assert a.reserves.mean() == b.reserves.mean()
    assert a.draws != c.draws


def test_one_program_is_a_triangle_with_one_segment():
    one = panel({"a": PROGRAMS["a"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = HierarchicalBf(seed=1, **QUICK).fit(one, "paid", "premium")
    assert [p.label for p in fit.programs] == ["a"]
    assert fit.reserves.n_sims == 600


def test_pattern_prior(tri):
    ages = [12, 24, 36, 48]
    prior = PatternPrior(ages, [0.4, 0.6, 0.7], [0.3, 0.3, 0.3])
    model = HierarchicalBf(seed=2, pattern_prior=prior, **QUICK)
    assert model.pattern_prior is not None
    assert "pattern_prior=PatternPrior(...)" in repr(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(tri, "paid", "premium")
    assert len(fit.programs) == 3
    wrong = PatternPrior([12, 24], [0.4], [0.3])
    with pytest.raises(ValueError):
        HierarchicalBf(pattern_prior=wrong, **QUICK).fit(tri, "paid", "premium")


def test_diagnostics_warning(tri):
    # A run too short to converge fails a threshold: one UserWarning, and the
    # fit still returns with the warnings listed.
    short = HierarchicalBf(chains=2, tune=20, draws=20, n_sims=100, seed=1)
    with pytest.warns(UserWarning, match="HierarchicalBf: the MCMC diagnostics failed"):
        fit = short.fit(tri, "paid", "premium")
    assert fit.diagnostics["warnings"]


def test_errors(tri):
    with pytest.raises(ValueError, match="n_sims must be positive"):
        HierarchicalBf(n_sims=0)
    with pytest.raises(ValueError, match="process"):
        HierarchicalBf(process="lognormal")
    with pytest.raises(OverflowError):
        HierarchicalBf(seed=-1)
    for name, value in [("apriori", 0.0), ("cv", 0.0), ("tau_scale", 0.0), ("omega_scale", -1.0)]:
        with pytest.raises(ValueError):
            HierarchicalBf(**{name: value}, **QUICK).fit(tri, "paid", "premium")
    with pytest.raises(ValueError):
        HierarchicalBf(**QUICK).fit(tri, "paid", "missing")
    with pytest.raises(ValueError):
        HierarchicalBf(**QUICK).fit(tri, "missing", "premium")

    # An origin with a gap before its latest cell is refused.
    gap = Triangle.from_long(
        [2022, 2022, 2023, 2023, 2023],
        [12, 36, 12, 24, 36],
        {"paid": [10.0, 20.0, 11.0, 15.0, 21.0], "premium": [50.0] * 2 + [55.0] * 3},
        keys={"program": ["a"] * 5},
    )
    with pytest.raises(ValueError):
        HierarchicalBf(**QUICK).fit(gap, "paid", "premium")


def test_falling_column_names_the_program_and_omega_scale():
    # Program b's second column falls by many of its scales (the chain
    # ladders of the other program fit almost exactly), which no posterior
    # supports at omega_scale 0.5.
    healthy = PROGRAMS["a"]
    falling = {2021: ([100.0, 20.0], 100.0), 2022: ([100.0], 100.0)}
    tri = panel({"a": healthy, "b": falling})
    with pytest.raises(ValueError) as e:
        HierarchicalBf(**QUICK).fit(tri, "paid", "premium")
    assert "program b at 24 months" in str(e.value)
    assert "omega_scale" in str(e.value)


def test_named_triangles_equal_the_panel_with_a_program_key(tri):
    model = HierarchicalBf(seed=4, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = model.fit(programs_by_name(), "paid", "premium")
        whole = model.fit(tri, "paid", "premium")
    assert stacked.draws == whole.draws
    assert stacked.reserves.draw_matrix() == whole.reserves.draw_matrix()
    assert [p.label for p in stacked.programs] == ["a", "b", "c"]
    # The programs run in sorted order of their names, not the dict's.
    backwards = dict(reversed(list(programs_by_name().items())))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flipped = model.fit(backwards, "paid", "premium")
    assert [p.label for p in flipped.programs] == ["a", "b", "c"]
    assert flipped.draws == whole.draws


def test_named_triangle_errors():
    fit = lambda programs, **kw: HierarchicalBf(**QUICK).fit(  # noqa: E731
        programs, kw.get("column", "paid"), "premium"
    )
    one = programs_by_name()
    with pytest.raises(ValueError, match="no programs"):
        fit({})
    with pytest.raises(ValueError, match="empty name"):
        fit({"": one["a"], "b": one["b"]})
    with pytest.raises(TypeError, match="program names"):
        fit({1: one["a"], "b": one["b"]})
    with pytest.raises(TypeError, match="not a Triangle"):
        fit({"a": one["a"], "b": 3})
    with pytest.raises(TypeError, match="Triangle or a dict"):
        fit([one["a"], one["b"]])
    with pytest.raises(ValueError, match='program "x" has 3 segments'):
        fit({"x": panel(), "b": one["b"]})
    with pytest.raises(ValueError, match='program "a" has no column "missing"'):
        fit(one, column="missing")
    quarterly = Triangle.from_long(
        [2022, 2022], [3, 6], {"paid": [1.0, 2.0], "premium": [9.0, 9.0]},
        origin_grain="Q", development_grain="Q",
    )
    with pytest.raises(ValueError, match='program "q" differs'):
        fit({"a": one["a"], "q": quarterly})
    incremental = Triangle.from_long(
        [2022, 2022], [12, 24], {"paid": [1.0, 2.0], "premium": [9.0, 9.0]},
        cumulative=False,
    )
    with pytest.raises(ValueError, match='program "i" differs'):
        fit({"a": one["a"], "i": incremental})


# --- the one-year view (decision 30, PR 17) -----------------------------------


@pytest.fixture(scope="module")
def view(tri):
    model = HierarchicalBf(apriori=0.7, level_cv=0.3, seed=3, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.one_year(tri, "paid", "premium", n_sims=500)


def test_one_year_fields(view, fit):
    assert isinstance(view, HierarchicalBfOneYearFit)
    assert isinstance(view.fit, HierarchicalBfFit)
    # The view's fit is the model's own fit, draw for draw.
    assert view.fit.draws == fit.draws
    assert view.program == ["a"] * 4 + ["b"] * 3 + ["c"] * 2
    assert view.origins == ["2020", "2021", "2022", "2023", "2021", "2022", "2023", "2022", "2023"]
    assert view.latest == [135.0, 133.0, 115.0, 72.0, 75.0, 64.0, 40.0, 38.0, 25.0]
    assert len(view.opening_reserve) == len(view.opening_ultimate) == 9
    for latest, ultimate, reserve in zip(view.latest, view.opening_ultimate, view.opening_reserve):
        assert ultimate == pytest.approx(latest + reserve)
    assert view.cdr.dims == ["program", "origin"]
    assert view.cdr.n_sims == 500 and len(view.pareto_k) == 500
    assert len(view.cdr.components()) == 9
    assert 0.0 <= view.flagged_share <= 1.0
    assert view.k_threshold == 0.7
    assert set(view.refitted) <= set(range(500))
    assert set(view.moment_matched) <= set(range(500))
    assert view.unrefitted == [] and view.refit_warnings >= 0
    assert repr(view).startswith("HierarchicalBfOneYearFit(programs=3, origins=9, n_sims=500")


def test_one_year_is_narrower_than_the_lifetime_view(view):
    # The oldest origin of program a is at its last age: no cell, no change.
    assert view.cdr.marginal(("a", "2020")).variance() == 0.0
    assert view.opening_reserve[0] == 0.0
    assert view.cdr.variance() < view.fit.reserves.variance()
    # The opening reserve is the posterior mean of the lifetime reserve.
    assert sum(view.opening_reserve) == pytest.approx(view.fit.reserves.mean(), rel=0.05)


def test_one_year_frames(view):
    f = view.to_frame()
    assert list(f.columns) == [
        "program", "origin", "latest", "opening_ultimate", "opening_reserve",
        "cdr_mean", "cdr_std_dev",
    ]  # fmt: skip
    assert len(f) == 9 and list(f["program"]) == view.program
    t = view.totals_frame()
    assert list(t["program"]) == ["a", "b", "c"]
    assert list(t.columns[:3]) == ["program", "latest", "opening_ultimate"]
    # A program's total is the sum of its origins.
    assert list(t["opening_reserve"]) == pytest.approx(
        [sum(view.opening_reserve[:4]), sum(view.opening_reserve[4:7]), sum(view.opening_reserve[7:])]
    )
    assert list(t["cdr_mean"]) == pytest.approx(
        [f["cdr_mean"][:4].sum(), f["cdr_mean"][4:7].sum(), f["cdr_mean"][7:].sum()]
    )
    by_program = view.cdr.aggregate(["program"])
    assert by_program.components() == [("a",), ("b",), ("c",)]
    assert list(t["cdr_std_dev"]) == pytest.approx(
        [by_program.marginal((k,)).variance() ** 0.5 for k in ("a", "b", "c")]
    )  # fmt: skip


def test_one_year_replays_by_seed_and_takes_dicts(tri, view):
    model = HierarchicalBf(apriori=0.7, level_cv=0.3, seed=3, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = model.one_year(tri, "paid", "premium", n_sims=500)
        named = model.one_year(programs_by_name(), "paid", "premium", n_sims=500)
        smaller = model.one_year(tri, "paid", "premium", n_sims=100)
        # The default number of simulations is the model's own; the cap of 2,000 is tested below.
        default = model.one_year(tri, "paid", "premium", refit="reweight_only")
    assert again.pareto_k == view.pareto_k
    assert again.cdr.draw_matrix() == view.cdr.draw_matrix()
    assert named.cdr.draw_matrix() == view.cdr.draw_matrix()
    # A smaller count is the first simulations of a larger one.
    assert smaller.pareto_k == view.pareto_k[:100]
    assert default.cdr.n_sims == 600 and default.refitted == [] and default.moment_matched == []


def test_one_year_default_count_is_capped_at_two_thousand(tri):
    # The model's n_sims is above the cap: the view runs 2,000, the lifetime
    # reserves keep 2,500, and an n_sims argument is used as it is.
    model = HierarchicalBf(
        apriori=0.7, level_cv=0.3, seed=3, chains=2, tune=100, draws=100, n_sims=2500
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        capped = model.one_year(tri, "paid", "premium", refit="reweight_only")
        more = model.one_year(tri, "paid", "premium", refit="reweight_only", n_sims=2100)
    assert capped.cdr.n_sims == 2000 and len(capped.pareto_k) == 2000
    assert capped.fit.reserves.n_sims == 2500
    assert more.cdr.n_sims == 2100


def test_one_year_settings_and_errors(tri):
    model = HierarchicalBf(apriori=0.7, level_cv=0.3, seed=3, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reweighted = model.one_year(tri, "paid", "premium", n_sims=200, refit="reweight_only")
        budget = model.one_year(
            tri, "paid", "premium", n_sims=200, refit="refit_flagged", max_refits=0
        )
    assert reweighted.refitted == [] and reweighted.moment_matched == []
    # No refit within a budget of 0: every flagged simulation is left unrefitted.
    assert budget.refitted == []
    assert len(budget.unrefitted) == round(budget.flagged_share * 200)
    with pytest.raises(ValueError, match="refit"):
        model.one_year(tri, "paid", "premium", refit="nonsense")
    with pytest.raises(ValueError, match="k_threshold"):
        model.one_year(tri, "paid", "premium", k_threshold=float("nan"))
    with pytest.raises(ValueError, match="n_sims"):
        model.one_year(tri, "paid", "premium", n_sims=0)
    with pytest.raises(ValueError, match="missing"):
        model.one_year(tri, "paid", "missing", n_sims=10)
    with pytest.raises(TypeError, match="Triangle"):
        model.one_year(5, "paid", "premium")


def test_a_refit_samples_the_whole_panel(tri):
    model = HierarchicalBf(
        apriori=0.7, level_cv=0.3, seed=3, chains=2, tune=150, draws=150, n_sims=100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        every = model.one_year(tri, "paid", "premium", n_sims=3, refit="refit_all")
    assert every.refitted == [0, 1, 2]
    assert every.cdr.n_sims == 3
