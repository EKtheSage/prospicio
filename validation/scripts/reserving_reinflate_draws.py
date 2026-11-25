# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.5"]
# [tool.uv]
# exclude-newer = "2026-11-24T00:00:00Z"
# ///
"""Regenerate validation/reference/reserving_reinflate_draws.csv.

Stochastic re-inflation of simulated future cells (`reserving-inflation.md`,
PR 8, decision 11, #979): `Indexation::reinflate_draws`. The inputs are
stated here and seeded (numpy `default_rng(20261124)`): simulated future
increments at the base price level, a calendar index, and one index path
per simulation for the periods after the index. The expected values are
computed here from those inputs alone, with this script's own calendar
arithmetic:

* a cell of annual origin `o` at age `a` months (the age at the end of its
  development period) is paid in month `(a - 1) % 12 + 1` of year
  `o + (a - 1) // 12`;
* its payment period is the index period that contains that month (the
  year, the quarter `(month - 1) // 3 + 1`, or the month itself);
* the re-inflated cell is the draw times `I(payment period) / I(base
  period)`, where `I` is the given index inside it, simulation `i`'s path
  after it (`*_paths`), or the last given value grown at the future rate,
  `(1 + rate)^(years past the last given period)` (`annual_rate`).

Every origin is valued at the end of 2023, its future cells at the ages
after its latest one up to 60 months: origin 2020 at age 60, 2021 at 48
and 60, 2022 at 36 to 60 and 2023 at 24 to 60, paid 2024 to 2027.

Methods:

* `annual_paths`: an annual index 2017 to 2024 (so the 2024 payments take a
  given value), base July 2020, paths 2025 to 2027;
* `annual_rate`: the same index with a future rate of 3.5% a year, no
  paths;
* `quarterly_paths`: a quarterly index 2018Q1 to 2024Q4, base May 2020
  (2020Q2), quarterly paths 2025Q1 to 2027Q4: the annual cells are paid in
  December, inside the fourth quarters, paths' positions 3, 7 and 11;
* `monthly_paths`: a monthly index 2019-01 to 2024-12, base July 2020,
  monthly paths 2025-01 to 2027-12: the payment month itself is priced,
  December 2024 from the given index and paths' positions 11, 23 and 35,
  so a payment month off by one or two within the year would take a
  different level.

Rows. Inputs, which the test builds its objects from and reads back:
`draw` (arg `sim=<i>;origin=<year>;age=<months>`, the base-level
increment), `index` (arg `period=<label>`), `path` (arg
`sim=<i>;period=<label>`), `future_rate` and `base` (arg `month=<yyyy-mm>`,
expected the base level `I(base)`). Outputs: `reinflated` (arg as `draw`),
`reserve` (arg `sim=<i>;origin=<year>`) and `total` (arg `sim=<i>`), to
1e-12 relative. Run from the repository root:

    uv run validation/scripts/reserving_reinflate_draws.py
"""

import csv

import numpy as np

OUT = "validation/reference/reserving_reinflate_draws.csv"
SEED = 20261124
N_SIMS = 4
VALUED = 2023
LAST_AGE = 60
RATE = 0.035
SOURCE = f"numpy {np.__version__} (closed form)"


def cells():
    """(origin, age) of every future cell, origin by origin, ages increasing."""
    out = []
    for o in range(2020, VALUED + 1):
        latest = 12 * (VALUED - o + 1)
        out.extend((o, a) for a in range(latest + 12, LAST_AGE + 1, 12))
    return out


def payment(origin, age):
    """(year, month) of the month a cell is paid in."""
    return origin + (age - 1) // 12, (age - 1) % 12 + 1


def label(year, month, grain):
    if grain == "quarter":
        return f"{year}Q{(month - 1) // 3 + 1}"
    if grain == "month":
        return f"{year}-{month:02d}"
    return str(year)


def periods(first_year, n, grain):
    if grain == "quarter":
        return [f"{first_year + k // 4}Q{k % 4 + 1}" for k in range(n)]
    if grain == "month":
        return [f"{first_year + k // 12}-{k % 12 + 1:02d}" for k in range(n)]
    return [str(first_year + k) for k in range(n)]


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    def emit(method, quantity, arg, value, rel_tol=0.0):
        rows.append(["synthetic", method, quantity, arg, repr(float(value)), "0", repr(rel_tol), SOURCE])

    future = cells()

    def draws():
        # Base-level increments: gamma with mean 1000 * 0.6^(age / 12 - 1)
        # and shape 4.
        means = np.array([1000.0 * 0.6 ** (a / 12 - 1) for _, a in future])
        return rng.gamma(4.0, means / 4.0, size=(N_SIMS, len(future)))

    top = {"year": 0.06, "quarter": 0.015, "month": 0.005}
    steps = {"year": 0.03, "quarter": 0.008, "month": 0.003}

    def index(first_year, n, grain):
        growth = rng.uniform(0.002 if grain != "month" else 0.0005, top[grain], n - 1)
        values = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + growth)])
        return dict(zip(periods(first_year, n, grain), values)), values

    def paths(last, first_year, n, grain):
        step = steps[grain]
        growth = 1.0 + rng.normal(step, step / 2, size=(N_SIMS, n))
        return last * np.cumprod(growth, axis=1), periods(first_year, n, grain)

    for method, grain, has_paths in [
        ("annual_paths", "year", True),
        ("annual_rate", "year", False),
        ("quarterly_paths", "quarter", True),
        ("monthly_paths", "month", True),
    ]:
        x = draws()
        if grain == "quarter":
            given, values = index(2018, 28, grain)
            base_month = (2020, 5)
            levels, path_periods = paths(values[-1], 2025, 12, grain)
        elif grain == "month":
            given, values = index(2019, 72, grain)
            base_month = (2020, 7)
            levels, path_periods = paths(values[-1], 2025, 36, grain)
        else:
            given, values = index(2017, 8, grain)
            base_month = (2020, 7)
            if has_paths:
                levels, path_periods = paths(values[-1], 2025, 3, grain)
        base = given[label(*base_month, grain)]
        last_year = 2024

        def level(sim, year, month):
            key = label(year, month, grain)
            if key in given:
                return given[key]
            if has_paths:
                return levels[sim][path_periods.index(key)]
            return values[-1] * (1.0 + RATE) ** (year - last_year)

        for p, v in given.items():
            emit(method, "index", f"period={p}", v)
        emit(method, "base", f"month={base_month[0]}-{base_month[1]:02d}", base)
        if has_paths:
            for i in range(N_SIMS):
                for p, v in zip(path_periods, levels[i]):
                    emit(method, "path", f"sim={i};period={p}", v)
        else:
            emit(method, "future_rate", "", RATE)
        for i in range(N_SIMS):
            reserves = {}
            for (o, a), d in zip(future, x[i]):
                arg = f"sim={i};origin={o};age={a}"
                emit(method, "draw", arg, d)
                out = d * level(i, *payment(o, a)) / base
                emit(method, "reinflated", arg, out, 1e-12)
                reserves[o] = reserves.get(o, 0.0) + out
            for o, r in reserves.items():
                emit(method, "reserve", f"sim={i};origin={o}", r, 1e-12)
            emit(method, "total", f"sim={i}", sum(reserves.values()), 1e-12)

    with open(OUT, "w", newline="") as f:
        f.write(
            f"# Generated by validation/scripts/reserving_reinflate_draws.py with numpy {np.__version__}.\n"
            "# Simulated future increments at a base price level re-inflated with one index path per\n"
            f"# simulation, or at a {RATE:.1%} future rate (reserving-inflation.md, PR 8, decision 11, #979).\n"
        )
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
