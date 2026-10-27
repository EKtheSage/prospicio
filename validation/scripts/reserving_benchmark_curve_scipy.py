"""Print the reference values of validation/tests/reserving_benchmark_curve.rs.

A growth curve fitted to a benchmark pattern by weighted least squares on the
percents developed (docs/design/reserving-v02.md, decision 23), with
scipy.optimize.least_squares. Run from the repository root:

    uv run --no-project --with scipy==1.18.1 --with numpy==2.5.4 python validation/scripts/reserving_benchmark_curve_scipy.py

The residuals are sqrt(w) * (p - G(c(a)) / G(c(m))), the parameters
(ln omega, ln theta), the method "lm" (MINPACK's Levenberg-Marquardt) with
ftol = xtol = gtol = 1e-15, started from (ln 1.5, ln 24). c is the age, or
with an origin width w Clark's age from the average date of loss
(a - w / 2 at or past w, a / 2 before); m is max_age (G(c(m)) = 1 without
one). The printed digits are pasted into the test.
"""

import numpy as np
from scipy.optimize import least_squares

# A benchmark paid pattern: percents developed at 12 to 120 months.
AGES = np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0])
PERCENT = np.array([0.18, 0.47, 0.66, 0.78, 0.855, 0.90, 0.93, 0.951, 0.965, 0.975])
WEIGHTS = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

CASES = [
    # name, curve, max_age, origin_width, weighted
    ("loglogistic", "loglogistic", None, None, False),
    ("weibull_shift_max", "weibull", 240.0, 12.0, False),
    ("loglogistic_weighted_shift", "loglogistic", None, 12.0, True),
]


def curve_age(a, width):
    if width is None:
        return a
    return np.where(a < width, a / 2.0, a - width / 2.0)


def growth(curve, x, omega, theta):
    if curve == "loglogistic":
        return 1.0 / (1.0 + (theta / x) ** omega)
    return -np.expm1(-((x / theta) ** omega))


def percent(curve, a, omega, theta, max_age, width):
    g = growth(curve, curve_age(a, width), omega, theta)
    if max_age is None:
        return g
    return g / growth(curve, curve_age(np.array(max_age), width), omega, theta)


for name, curve, max_age, width, weighted in CASES:
    w = WEIGHTS if weighted else np.ones_like(AGES)

    def residuals(x):
        omega, theta = np.exp(x)
        return np.sqrt(w) * (PERCENT - percent(curve, AGES, omega, theta, max_age, width))

    r = least_squares(
        residuals, [np.log(1.5), np.log(24.0)], method="lm", ftol=1e-15, xtol=1e-15, gtol=1e-15
    )
    omega, theta = np.exp(r.x)
    print(f"{name}: omega = {omega:.15g}, theta = {theta:.15g}, sse = {2 * r.cost:.15g}")
