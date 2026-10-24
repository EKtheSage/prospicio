"""The latest diagonal on its own projected with a pattern from elsewhere
(``docs/design/reserving-v02.md``, decisions 16 to 19, issue #544): the
``LatestDiagonal`` type, ``fit_diagonal`` on the chain ladder and the
expected-loss methods, ``ChainLadderFit.to_chain_ladder``, ``ldf_origin``
and the rule past a pattern's end. The Rust core is checked in
``validation/tests/reserving_latest_diagonal.rs``,
``reserving_fit_apply.rs`` and ``reserving_diagonal_segments.rs``; these
tests reproduce their numbers through the bindings."""

import csv
import math
from datetime import date
from pathlib import Path

import pytest

from prospicio.reserving import (
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    ExpectedLoss,
    LatestDiagonal,
    Mack,
    MackBootstrap,
    OdpBootstrap,
    TailConstant,
    TailCurve,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def close(found, expected, rel=1e-12, abs_=0.0):
    assert len(found) == len(expected), (found, expected)
    for f, e in zip(found, expected):
        assert f == e or abs(f - e) <= max(abs_, rel * abs(e)), (found, expected)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def long_rows(name):
    """(origin, age, values) rows of a validation dataset."""
    rows = read_rows(VALIDATION / "data" / f"{name}.csv")
    columns = [c for c in rows[0] if c not in ("origin", "development")]
    return [
        (int(r["origin"]), int(r["development"]), [float(r[c]) for c in columns])
        for r in rows
    ], columns


def book(rows, columns, keep=lambda year, age: True, value=lambda c, year, v: v, shift=0):
    origin, age, values = [], [], {c: [] for c in columns}
    for year, a, cells in rows:
        if not keep(year, a):
            continue
        origin.append(year + shift)
        age.append(a)
        for k, c in enumerate(columns):
            values[c].append(value(k, year, cells[k]))
    return Triangle.from_long(origin, age, values)


GENINS, GENINS_COLUMNS = long_rows("genins")
WKCOMP, WKCOMP_COLUMNS = long_rows("clrd_wkcomp")


def genins():
    return book(GENINS, GENINS_COLUMNS)


def wkcomp():
    return book(WKCOMP, WKCOMP_COLUMNS)


def target(dataset, name):
    """The books of ``target()`` in ``validation/tests/reserving_fit_apply.rs``."""
    if dataset == "genins":
        rows, columns = GENINS, GENINS_COLUMNS
        return {
            "diagonal": lambda: book(rows, columns),
            "perturbed": lambda: book(
                rows, columns, value=lambda c, year, v: v * (1.0 + 0.01 * (year - 2000))
            ),
            "double": lambda: book(rows, columns, value=lambda c, year, v: v * 2.0),
            "from_2006": lambda: book(rows, columns, keep=lambda year, age: year >= 2006),
            "to_2004": lambda: book(rows, columns, keep=lambda year, age: year <= 2004),
            "shifted": lambda: book(rows, columns, shift=10),
            "early": lambda: book(
                rows, columns, keep=lambda year, age: year + age // 12 - 1 < 2008
            ),
        }[name]()
    rows, columns = WKCOMP, WKCOMP_COLUMNS

    def perturbed(c, year, v):
        v = v * (1.0 + 0.01 * (year - 1987))
        return v * 1.1 if c == 1 else v

    return {
        "diagonal": lambda: book(rows, columns),
        "diagonal_raw": lambda: book(rows, columns),
        "perturbed": lambda: book(rows, columns, value=perturbed),
        "shifted": lambda: book(rows, columns, shift=10),
        "from_1993": lambda: book(rows, columns, keep=lambda year, age: year >= 1993),
    }[name]()


def params(method):
    out = {}
    for part in method.split(";")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def check_row(row, found):
    expected = float(row["expected"])
    abs_tol, rel_tol = float(row["abs_tol"]), float(row["rel_tol"])
    assert found is not None, row
    assert abs(found - expected) <= max(abs_tol, rel_tol * abs(expected)), (row, found)


def years(*ys):
    return [date(y, 1, 1) for y in ys]


DEC_2024 = date(2024, 12, 1)


# 1. Three accident years at December 2024.
def test_three_accident_years_by_hand():
    d = LatestDiagonal.at_valuation(years(2022, 2023, 2024), DEC_2024, [900.0, 600.0, 150.0],
                                    exposure=[1000.0] * 3)
    assert d.ages == [36, 24, 12]
    cdf = {12: 2.0, 24: 1.25, 36: 1.0}
    cl = ChainLadder(ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    assert cl.ultimate == [900.0, 750.0, 300.0]
    bf = BornhuetterFerguson(apriori=0.7, ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    assert bf.ultimate[2] == pytest.approx(500.0, abs=1e-9)


# 2. The six-origin hand diagonal.
HAND_LATEST = [980.0, 860.0, 700.0, 520.0, 400.0, 150.0]
HAND_CDF = [1.0, 1.02, 1.07, 1.15, 1.3, 12.0 / 7.0]
HAND_AGES = [72, 60, 48, 36, 24, 12]
PREMIUM = 200.0


def hand():
    return LatestDiagonal(list(range(2019, 2025)), HAND_AGES, HAND_LATEST,
                          exposure=[PREMIUM] * 6)


def hand_pattern():
    return {a: c for a, c in zip(HAND_AGES, HAND_CDF)}


def test_the_hand_diagonal_by_each_method():
    d, cdf, elr = hand(), hand_pattern(), 0.7
    reported = [1.0 / c for c in HAND_CDF]
    cl = ChainLadder(ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    close(cl.ultimate, [l * c for l, c in zip(HAND_LATEST, HAND_CDF)])
    assert cl.ultimate[5] == pytest.approx(257.142857142857, abs=1e-9)

    bf = BornhuetterFerguson(apriori=elr, ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    close(bf.ultimate, [l + elr * PREMIUM * (1 - p) for l, p in zip(HAND_LATEST, reported)])
    assert bf.ultimate[5] == pytest.approx(208.333333333333, abs=1e-9)

    bk = Benktander(apriori=elr, n_iters=2, ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    close(bk.ultimate, [
        l + (l + elr * PREMIUM * (1 - p)) * (1 - p) for l, p in zip(HAND_LATEST, reported)
    ])

    cc = CapeCod(trend=0.0, decay=1.0, ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    cc_elr = sum(HAND_LATEST) / sum(PREMIUM * p for p in reported)
    close(cc.ultimate, [l + cc_elr * PREMIUM * (1 - p) for l, p in zip(HAND_LATEST, reported)])

    percent = {a: 1.0 / c for a, c in cdf.items()}
    by_percent = ChainLadder(ldf=percent, ldf_kind="percent_developed").fit_diagonal(d)
    close(by_percent.ultimate, cl.ultimate)
    el = ExpectedLoss(apriori=0.7, ldf=percent, ldf_kind="percent_developed").fit_diagonal(d)
    close(el.ultimate, [0.7 * PREMIUM] * 6)


# 3. Half-year origins with the annual pattern.
def test_half_year_origins_read_the_annual_pattern():
    origins = [date(2023, 1, 1), date(2023, 7, 1), date(2024, 1, 1)]
    d = LatestDiagonal.at_valuation(origins, DEC_2024, [500.0, 300.0, 100.0],
                                    exposure=[800.0] * 3, origin_grain="S")
    assert d.ages == [24, 18, 12]
    assert d.origin_grain == "S"
    cdf = {12: 2.0, 24: 1.25, 36: 1.0}
    cdf18 = math.sqrt(2.0 * 1.25)
    cl = ChainLadder(ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    assert cl.development == [12, 18, 24, 30, 36]
    close(cl.ultimate, [625.0, 300.0 * cdf18, 200.0])
    bf = BornhuetterFerguson(apriori=0.5, ldf=cdf, ldf_kind="cdf").fit_diagonal(d)
    close(bf.ultimate, [
        500 + 400 * (1 - 1 / 1.25), 300 + 400 * (1 - 1 / cdf18), 100 + 400 * 0.5
    ])
    ldf = {12: 1.6, 24: 1.25}
    close(ChainLadder(ldf=ldf, ldf_kind="ldf").fit_diagonal(d).ultimate, cl.ultimate)
    with_tail = ChainLadder(ldf=ldf, ldf_kind="ldf", tail=1.05).fit_diagonal(d)
    close(with_tail.ultimate, [u * 1.05 for u in cl.ultimate])


# 4. Past the pattern's end.
def test_past_the_patterns_end():
    ldf = {12: 1.5, 24: 1.2}
    origin = list(range(2021, 2025))

    def diagonal(rule):
        return LatestDiagonal(origin, [48, 36, 24, 12], [100.0] * 4, past_pattern_end=rule)

    with pytest.raises(ValueError, match="outside the pattern's ages from 12 to 36 months"):
        ChainLadder(ldf=ldf, ldf_kind="ldf").fit_diagonal(diagonal("reject"))
    held = ChainLadder(ldf=ldf, ldf_kind="ldf").fit_diagonal(diagonal("no_development"))
    assert held.ultimate[0] == 100.0
    tailed = ChainLadder(ldf=ldf, ldf_kind="ldf", tail=1.05).fit_diagonal(diagonal("tail"))
    close(tailed.ultimate[:2], [102.53798721500054, 105.0])

    ten = {12 + 12 * k: 1.1 for k in range(9)}
    d = LatestDiagonal(list(range(2000, 2005)), [144, 132, 120, 108, 12],
                       [100.0] * 5, past_pattern_end="tail")
    fit = ChainLadder(ldf=ten, ldf_kind="ldf", tail=1.05).fit_diagonal(d)
    close([u / 100.0 for u in fit.ultimate[:4]],
          [1.0132158418990078, 1.0253798721500054, 1.05, 1.155], rel=1e-14)

    percents = {12: 0.4, 60: 0.8, 120: 0.95}
    d = LatestDiagonal([2000, 2001, 2002], [120, 108, 12], [100.0] * 3,
                       past_pattern_end="no_development")
    fit = ChainLadder(ldf=percents, ldf_kind="percent_developed").fit_diagonal(d)
    assert fit.pattern_rescale == pytest.approx(1.0526315789473684, rel=1e-15)
    assert fit.ultimate[2] == pytest.approx(237.5, rel=1e-14)
    rejected = ChainLadder(ldf=percents, ldf_kind="percent_developed").fit_diagonal(
        d.with_past_pattern_end("reject"))
    assert rejected.ultimate[2] == pytest.approx(250.0, rel=1e-14)
    assert rejected.pattern_tail == pytest.approx(1 / 0.95, rel=1e-14)
    assert rejected.pattern_rescale == pytest.approx(1.0, rel=1e-15)

    with pytest.raises(ValueError):
        ChainLadder(ldf=ldf, ldf_kind="ldf", tail=1.05).fit_diagonal(diagonal("no_development"))
    with pytest.raises(ValueError):
        ChainLadder(ldf=ldf, ldf_kind="ldf").fit_diagonal(diagonal("tail"))
    with pytest.raises(ValueError):
        ChainLadder(ldf=ldf, ldf_kind="ldf",
                    tail=TailConstant(1.05, attachment_age=36)).fit_diagonal(diagonal("tail"))
    with pytest.raises(ValueError, match="reject.*no_development.*tail"):
        diagonal("reject").with_past_pattern_end("hold")


# 5. ldf_origin.
MONTHLY = {t: t / 24.0 for t in range(1, 25)}


def test_ldf_origin_averages_a_monthly_pattern_over_the_year():
    d = LatestDiagonal([2023, 2024], [24, 12], [37.0, 13.0])
    cl = ChainLadder(ldf=MONTHLY, ldf_kind="percent_developed", ldf_origin="M")
    assert cl.ldf_origin == "M"
    assert 'ldf_origin="M"' in repr(cl)
    assert cl.ldf == MONTHLY
    close(cl.fit_diagonal(d).ultimate, [48.0, 48.0], rel=1e-13)

    tri = Triangle.from_long([2023, 2023, 2024], [12, 24, 12], {"paid": [13.0, 37.0, 13.0]})
    fit = cl.fit(tri, "paid")
    assert fit.selected_ldf[0] == pytest.approx(37.0 / 13.0, rel=1e-14)

    own = {12: 1.5}
    same = ChainLadder(ldf=own, ldf_kind="ldf", ldf_origin="Y").fit(tri, "paid")
    plain = ChainLadder(ldf=own, ldf_kind="ldf").fit(tri, "paid")
    assert same.ultimate == plain.ultimate
    assert same.ldf == plain.ldf


def test_ldf_origin_errors():
    tri = Triangle.from_long([2023, 2023, 2024], [12, 24, 12], {"paid": [13.0, 37.0, 13.0]})
    with pytest.raises(ValueError):
        ChainLadder(ldf=MONTHLY, ldf_kind="percent_developed", ldf_weight={12: 0.5},
                    ldf_origin="M").fit(tri, "paid")
    quarters = LatestDiagonal([date(2024, 1, 1), date(2024, 4, 1)], [6, 3], [10.0, 5.0],
                              origin_grain="Q")
    with pytest.raises(ValueError):
        ChainLadder(ldf={12: 2.0, 24: 1.0}, ldf_kind="cdf", ldf_origin="Y").fit_diagonal(quarters)
    with pytest.raises(ValueError, match="ldf_origin is given without ldf"):
        ChainLadder(ldf_origin="M")
    for method in (Mack, ExpectedLoss, BornhuetterFerguson, Benktander, CapeCod,
                   OdpBootstrap, MackBootstrap):
        assert method(ldf={12: 1.5}, ldf_kind="ldf", ldf_origin="Q").ldf_origin == "Q"
        with pytest.raises(ValueError, match="ldf_origin is given without ldf"):
            method(ldf_origin="M")


# MONTHLY averaged over accident years is 13/48 developed at 12 months and
# 37/48 at 24, where it ends: factors to ultimate 48/13 and 48/37, the
# development past 24 months the triangle's own.
AVERAGED = dict(ldf=MONTHLY, ldf_kind="percent_developed", ldf_origin="M")
DIRECT = dict(ldf={12: 48.0 / 13.0, 24: 48.0 / 37.0}, ldf_kind="cdf")


def four_years():
    return Triangle.from_long(
        [2021] * 4 + [2022] * 3 + [2023] * 2 + [2024],
        [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
        {
            "paid": [100.0, 280.0, 380.0, 400.0, 110.0, 300.0, 400.0, 120.0, 330.0, 130.0],
            "premium": [500.0] * 4 + [520.0] * 3 + [540.0] * 2 + [560.0],
        },
    )


def close_draws(found, expected, rel=1e-10):
    assert len(found) == len(expected)
    for f, e in zip(found, expected):
        close(f, e, rel=rel, abs_=1e-9)


def test_ldf_origin_reaches_every_method():
    # Each method's own path applies ldf_origin: its result equals the same
    # method given the averaged pattern directly, and differs from the
    # monthly pattern read as if it were annual.
    tri = four_years()
    cl = ChainLadder(**AVERAGED).fit(tri, "paid")
    direct = ChainLadder(**DIRECT).fit(tri, "paid")
    assert cl.selected_ldf[0] == pytest.approx(37.0 / 13.0, rel=1e-14)
    close(cl.ldf, direct.ldf)
    close(cl.ultimate, direct.ultimate)
    unaveraged = ChainLadder(ldf=MONTHLY, ldf_kind="percent_developed").fit(tri, "paid")
    assert unaveraged.selected_ldf[0] != pytest.approx(cl.selected_ldf[0], rel=1e-3)

    mack, mack_direct = Mack(**AVERAGED).fit(tri, "paid"), Mack(**DIRECT).fit(tri, "paid")
    close(mack.selected_ldf[:1], mack_direct.selected_ldf[:1])
    close(mack.ultimate, mack_direct.ultimate)
    close(mack.standard_error, mack_direct.standard_error)
    close(mack.cdr.one_year_standard_error, mack_direct.cdr.one_year_standard_error)

    # The averaged pattern runs to 35 months, so the diagonal keeps to the
    # origins within 24 months, where the direct pattern ends.
    d = LatestDiagonal([2023, 2024], [24, 12], [330.0, 130.0], exposure=[540.0, 560.0])
    for method in (
        lambda p: ExpectedLoss(apriori=0.8, **p),
        lambda p: BornhuetterFerguson(apriori=0.8, **p),
        lambda p: Benktander(apriori=0.8, n_iters=2, **p),
        lambda p: CapeCod(trend=0.02, **p),
    ):
        close(method(AVERAGED).fit(tri, "paid", "premium").ultimate,
              method(DIRECT).fit(tri, "paid", "premium").ultimate)
        close(method(AVERAGED).fit_diagonal(d).ultimate, method(DIRECT).fit_diagonal(d).ultimate)

    for boot in (OdpBootstrap, MackBootstrap):
        found = boot(n_sims=200, seed=11, **AVERAGED).fit(tri, "paid").reserves.draw_matrix()
        want = boot(n_sims=200, seed=11, **DIRECT).fit(tri, "paid").reserves.draw_matrix()
        close_draws(found, want)
        # The re-reserving method's ldf_origin in a one-year view, and the
        # bootstrap's own.
        for view, method in (
            (boot(n_sims=200, seed=11), lambda p: ChainLadder(**p)),
            (boot(n_sims=200, seed=11), lambda p: BornhuetterFerguson(apriori=0.8, **p)),
        ):
            exposure = None if isinstance(method({}), ChainLadder) else "premium"
            found = view.one_year(tri, "paid", method(AVERAGED), exposure=exposure)
            want = view.one_year(tri, "paid", method(DIRECT), exposure=exposure)
            close_draws(found.cdr.draw_matrix(), want.cdr.draw_matrix())
        own = boot(n_sims=200, seed=11, **AVERAGED).one_year(tri, "paid", ChainLadder(**AVERAGED))
        own_direct = boot(n_sims=200, seed=11, **DIRECT).one_year(tri, "paid", ChainLadder(**DIRECT))
        close_draws(own.cdr.draw_matrix(), own_direct.cdr.draw_matrix())


# 6. Off-grid reading on RAA.
def test_an_off_grid_age_changes_how_the_other_factors_read():
    rows, columns = long_rows("raa")
    raa = book(rows, columns)
    fit = ChainLadder(ldf={12: 1.5, 18: 1.1, 36: 1.2}, ldf_kind="ldf").fit(raa, "value")
    sel = fit.selected_ldf
    close(sel[:3], [1.5484201731845508, 1.0656022367666107, 1.129243234657234], rel=1e-15)
    assert all(s is None for s in sel[3:])
    sparse = ChainLadder(ldf={12: 1.5, 36: 1.2}, ldf_kind="ldf").fit(raa, "value").selected_ldf
    assert sparse[:3] == [1.5, None, 1.2]
    assert all(s is None for s in sparse[3:])


# 7. chainladder-python's DevelopmentConstant on the latest diagonal.
def test_the_diagonal_matches_chainladder_python():
    rows = read_rows(VALIDATION / "reference" / "reserving_latest_diagonal_python.csv")
    patterns = {}
    for r in rows:
        if r["quantity"] == "pattern":
            patterns.setdefault((r["dataset"], r["method"]), {})[int(r["arg"])] = float(r["expected"])
    fits = {}
    checked = 0
    for r in rows:
        if r["quantity"] == "pattern":
            continue
        key = (r["dataset"], r["method"])
        if key not in fits:
            kind = "ldf" if "style=ldf" in r["method"] else "cdf"
            p = params(r["method"])
            pattern = dict(ldf=patterns[key], ldf_kind=kind)
            name = r["method"].split(";")[0]
            if r["dataset"] == "genins":
                fit = ChainLadder(**pattern).fit_diagonal(
                    LatestDiagonal.from_triangle(genins(), "value"))
            else:
                d = LatestDiagonal.from_triangle(wkcomp(), "paid", exposure="premium")
                if name == "bornhuetter_ferguson":
                    fit = BornhuetterFerguson(apriori=float(p["apriori"]), **pattern)
                elif name == "benktander":
                    fit = Benktander(apriori=float(p["apriori"]), n_iters=int(p["n_iters"]),
                                     **pattern)
                else:
                    fit = CapeCod(trend=float(p["trend"]), decay=float(p["decay"]), **pattern)
                fit = fit.fit_diagonal(d)
            fits[key] = fit
        fit = fits[key]
        if r["quantity"] == "total_ultimate":
            check_row(r, fit.total_ultimate)
        else:
            check_row(r, fit.ultimate[fit.origins.index(r["arg"])])
        checked += 1
    assert checked == sum(r["quantity"] != "pattern" for r in rows) == 55


# 8. A fit applied to another book against chainladder-python.
def genins_tail(method):
    p = params(method)
    if p["tail"] == "none":
        return None
    if p["tail"] == "constant":
        return float(p["factor"])
    return TailCurve(p.get("curve", "exponential"))


def source_fit(dataset, base):
    if dataset == "genins":
        tail = genins_tail(base)
        cl = ChainLadder() if tail is None else ChainLadder(tail=tail)
        return cl.fit(genins(), "value")
    return ChainLadder().fit(wkcomp(), "paid")


def target_diagonal(dataset, name, source):
    tri = target(dataset, name)
    if dataset == "genins":
        return LatestDiagonal.from_triangle(tri, "value")
    d = LatestDiagonal.from_triangle(tri, "paid", exposure="premium")
    if name == "diagonal_raw":
        first = [source.development[0]] * len(d.ages)
        d = LatestDiagonal([int(o) for o in d.origins], first, d.latest, exposure=d.exposure)
    return d


def apply(base, development, d, route):
    """``development`` is the frozen chain ladder's arguments, or a source fit
    for ``route == "development"``."""
    name, p = base.split(";")[0], params(base)
    if route == "development":
        args, extra = {}, dict(development=development)
    else:
        args, extra = development, {}
    if name == "chain_ladder":
        return ChainLadder(**args).fit_diagonal(d, **extra)
    if name == "bornhuetter_ferguson":
        method = BornhuetterFerguson(apriori=float(p["apriori"]), **args)
    elif name == "benktander":
        method = Benktander(apriori=float(p["apriori"]), n_iters=int(p["n_iters"]), **args)
    else:
        method = CapeCod(trend=float(p["trend"]), decay=float(p["decay"]), **args)
    return method.fit_diagonal(d, **extra)


def frozen_args(cl):
    return dict(ldf=cl.ldf, ldf_kind=cl.ldf_kind, tail=cl.tail)


@pytest.mark.parametrize("arm", ["own", "reference"])
def test_a_fit_applied_to_another_book_matches_chainladder_python(arm):
    rows = read_rows(VALIDATION / "reference" / "reserving_fit_apply_python.csv")
    patterns = {}
    for r in rows:
        if r["quantity"] == "pattern":
            patterns.setdefault((r["dataset"], r["method"]), {})[int(r["arg"])] = float(r["expected"])
    if arm == "reference":
        rows = [r for r in rows if ";target=" in r["method"]]
    sources, applied = {}, {}
    checked = 0
    for r in rows:
        dataset, method = r["dataset"], r["method"]
        base, _, target_name = method.partition(";target=")
        key = (dataset, base)
        if key not in sources:
            sources[key] = source_fit(dataset, base)
        source = sources[key]
        if not target_name:
            q = r["quantity"]
            if q == "pattern":
                found = source.cdf[source.development.index(int(r["arg"]))]
            elif q == "tail":
                found = source.tail
            elif q == "refitted_tail_perturbed":
                refit = ChainLadder(tail=genins_tail(base)).fit(target("genins", "perturbed"),
                                                                "value")
                carried, found = source.tail, refit.tail
                assert abs(found - carried) > float(r["rel_tol"]) * carried
            else:
                raise AssertionError(r)
            check_row(r, found)
            checked += 1
            continue
        if (dataset, method) not in applied:
            d = target_diagonal(dataset, target_name, source)
            if arm == "own":
                frozen = source.to_chain_ladder()
                fit = apply(base, frozen_args(frozen), d, "args")
                # Passing the fit as development is the same fit, bit for bit.
                by_development = apply(base, source, d, "development")
                assert by_development.ultimate == fit.ultimate
                assert by_development.cdf == fit.cdf
            else:
                fit = apply(base, dict(ldf=patterns[key], ldf_kind="cdf"), d, "args")
            applied[(dataset, method)] = fit
        fit = applied[(dataset, method)]
        q = r["quantity"]
        if q == "total_ultimate":
            found = fit.total_ultimate
        elif q == "ultimate":
            found = fit.ultimate[fit.origins.index(r["arg"])]
        elif q == "apriori":
            found = fit.trended_apriori[fit.origins.index(r["arg"])]
        else:
            raise AssertionError(r)
        check_row(r, found)
        checked += 1
    assert checked > 100


def test_to_chain_ladder_round_trips_through_its_arguments():
    source = ChainLadder(tail=TailCurve()).fit(genins(), "value")
    frozen = source.to_chain_ladder()
    assert frozen.ldf_kind == "ldf"
    assert isinstance(frozen.tail, float)
    assert list(frozen.ldf) == source.development[:-1]
    d = LatestDiagonal.from_triangle(genins(), "value")
    a = frozen.fit_diagonal(d)
    b = ChainLadder(**frozen_args(frozen)).fit_diagonal(d)
    assert a.ultimate == b.ultimate
    assert a.ldf == source.ldf
    close(a.ultimate, source.ultimate, rel=1e-14)


# 9. Segments.
def keyed_genins():
    lob, origin, age, values = [], [], [], []
    perturbed = [(y, a, [v[0] * (1.0 + 0.01 * (y - 2000))]) for y, a, v in GENINS]
    for label, rows in (("a", GENINS), ("b", perturbed)):
        for y, a, v in rows:
            lob.append(label)
            origin.append(y)
            age.append(a)
            values.append(v[0])
    return Triangle.from_long(origin, age, {"value": values}, keys={"lob": lob})


def test_each_segment_takes_its_source_segment():
    tri = keyed_genins()
    fits = ChainLadder(tail=TailCurve()).fit(tri, "value")
    d = LatestDiagonal.from_triangle(tri, "value")
    assert d.keys == ["lob"]
    assert d.index == ["a", "b"]
    applied = ChainLadder().fit_diagonal(d, development=fits)
    assert applied.index == ["a", "b"]
    for lob in ("a", "b"):
        s, f = fits.segment(lob=lob), applied.segment(lob=lob)
        assert f.ldf == s.ldf
        close(f.ultimate, s.ultimate, rel=1e-14)

    mapped = ChainLadder().fit_diagonal(d, development=fits, segment_map={"b": "a", "a": "a"})
    by_hand = fits.segment(lob="a").to_chain_ladder().fit_diagonal(d.segment(lob="b"))
    assert mapped.segment(lob="b").ultimate == by_hand.ultimate

    one = ChainLadder(tail=TailCurve()).fit(genins(), "value")
    broadcast = ChainLadder().fit_diagonal(d, development=one)
    for lob in ("a", "b"):
        alone = one.to_chain_ladder().fit_diagonal(d.segment(lob=lob))
        assert broadcast.segment(lob=lob).ultimate == alone.ultimate

    # Another method's fit gives its development through .chain_ladder.
    bf = BornhuetterFerguson(apriori=0.6)
    with pytest.raises(TypeError, match=".chain_ladder"):
        bf.fit_diagonal(d, development=bf)


def keyed_wkcomp():
    lob, origin, age, paid, premium = [], [], [], [], []
    for label, scale in (("a", 1.0), ("b", 1.1)):
        for y, a, v in WKCOMP:
            lob.append(label)
            origin.append(y)
            age.append(a)
            paid.append(v[0] * scale * (1.0 + 0.01 * (y - 1988) * (label == "b")))
            premium.append(v[1])
    return Triangle.from_long(origin, age, {"paid": paid, "premium": premium}, keys={"lob": lob})


def test_another_methods_fit_passes_its_chain_ladder():
    tri = keyed_wkcomp()
    d = LatestDiagonal.from_triangle(tri, "paid", exposure="premium")
    plain = ChainLadder(tail=TailCurve()).fit(tri, "paid")
    bf_fits = BornhuetterFerguson(apriori=0.6, tail=TailCurve()).fit(tri, "paid", "premium")
    via_bf = bf_fits.chain_ladder
    assert via_bf.index == plain.index
    for method in (ChainLadder(), BornhuetterFerguson(apriori=0.6)):
        from_bf = method.fit_diagonal(d, development=via_bf)
        from_plain = method.fit_diagonal(d, development=plain)
        for lob in ("a", "b"):
            assert from_bf.segment(lob=lob).ultimate == from_plain.segment(lob=lob).ultimate
    # A single-segment fit's .chain_ladder serves every segment.
    one = BornhuetterFerguson(apriori=0.6).fit(wkcomp(), "paid", "premium").chain_ladder
    broadcast = ChainLadder().fit_diagonal(d, development=one)
    for lob in ("a", "b"):
        alone = one.to_chain_ladder().fit_diagonal(d.segment(lob=lob))
        assert broadcast.segment(lob=lob).ultimate == alone.ultimate


def test_segment_errors():
    tri = keyed_genins()
    fits = ChainLadder().fit(tri, "value")
    d = LatestDiagonal.from_triangle(tri, "value")
    with pytest.raises(ValueError, match="no source"):
        ChainLadder().fit_diagonal(d, development=fits, segment_map={"a": "a"})
    # Two segments under another key name.
    two = Triangle.from_long([2001, 2001, 2002] * 2, [12, 24, 12] * 2, {"value": [1.0, 2.0, 1.0] * 2},
                             keys={"line": ["x"] * 3 + ["y"] * 3})
    # The match leaves out the message's hint, whose wording is decision 19
    # item 11's to settle.
    with pytest.raises(ValueError, match="the source has keys"):
        ChainLadder().fit_diagonal(d, development=ChainLadder().fit(two, "value"))
    with pytest.raises(ValueError, match="the development comes from the source fit"):
        ChainLadder(ldf={12: 1.5}, ldf_kind="ldf").fit_diagonal(d, development=fits)
    with pytest.raises(ValueError, match="the development comes from the source fit"):
        ChainLadder(tail=1.05).fit_diagonal(d, development=fits)
    # Link ratio exclusions are refused too, rather than ignored.
    for method in (ChainLadder(n_periods=3), ChainLadder(drop_high=1),
                   BornhuetterFerguson(apriori=0.6, drop_low=1)):
        with pytest.raises(ValueError, match="no link ratio exclusions"):
            method.fit_diagonal(d, development=fits)
    with pytest.raises(ValueError, match="segment"):
        fits.to_chain_ladder()
    with pytest.raises(ValueError, match="segment_map is given without development"):
        ChainLadder().fit_diagonal(d, segment_map={"a": "a"})


def test_cape_cod_trends_every_segment_to_one_valuation():
    lob, origin, age, paid, premium = [], [], [], [], []
    for label, shift in (("a", 0), ("b", -1)):
        for y, a, v in WKCOMP:
            lob.append(label)
            origin.append(y + shift)
            age.append(a)
            paid.append(v[0])
            premium.append(v[1])
    tri = Triangle.from_long(origin, age, {"paid": paid, "premium": premium}, keys={"lob": lob})
    d = LatestDiagonal.from_triangle(tri, "paid", exposure="premium")
    cc = CapeCod(trend=0.05, ldf=dict(zip(*_wkcomp_cdf())), ldf_kind="cdf")
    together = cc.fit_diagonal(d)
    alone = cc.fit_diagonal(d.segment(lob="b"))
    close(together.segment(lob="b").trended_apriori, [t * 1.05 for t in alone.trended_apriori])
    close(together.segment(lob="b").apriori, alone.apriori)
    close(together.segment(lob="b").ultimate, alone.ultimate)


def _wkcomp_cdf():
    fit = ChainLadder().fit(wkcomp(), "paid")
    return fit.development, fit.cdf


# 10. Constructors.
def test_from_frame_with_keys_and_valuations_is_the_triangle_diagonal():
    pd = pytest.importorskip("pandas")
    tri = keyed_genins()
    d = LatestDiagonal.from_triangle(tri, "value")
    frame = d.to_frame()
    assert list(frame.columns) == ["lob", "origin", "age", "latest"]
    frame["valuation"] = [date(int(o) + a // 12 - 1, 12, 1) for o, a in
                          zip(frame["origin"], frame["age"])]
    frame["origin_date"] = [date(int(o), 1, 1) for o in frame["origin"]]
    again = LatestDiagonal.from_frame(frame, "origin_date", "latest", valuation="valuation",
                                      keys=["lob"])
    assert again.index == d.index
    assert again.ages == d.ages
    assert again.latest == d.latest
    assert again.origins == d.origins

    exposure = [float(k + 1) for k in range(len(frame))]
    with_exposure = d.with_exposure(exposure)
    assert with_exposure.exposure == exposure
    assert with_exposure.to_frame()["exposure"].tolist() == exposure
    b = with_exposure.segment(lob="b")
    assert b.exposure == exposure[len(d.segment(lob="a").ages):]
    by_age = LatestDiagonal.from_frame(frame, "origin_date", "latest", age="age", keys=["lob"])
    assert by_age.ages == d.ages
    with pytest.raises(ValueError, match="exactly one of age"):
        LatestDiagonal.from_frame(frame, "origin_date", "latest")


def test_constructor_errors():
    with pytest.raises(TypeError, match="give the valuation month as a date"):
        LatestDiagonal.at_valuation([2023, 2024], 2024, [1.0, 2.0])
    with pytest.raises(ValueError, match="reject.*no_development.*tail"):
        LatestDiagonal([2023], [12], [1.0], past_pattern_end="hold")
    d = LatestDiagonal([2023, 2024], [24, 12], [1.0, 2.0], past_pattern_end="tail")
    assert d.past_pattern_end == "tail"
    assert d.with_past_pattern_end("no_development").past_pattern_end == "no_development"
    assert d.keys == []
    assert d.index == ["Total"]
    assert d.exposure is None
