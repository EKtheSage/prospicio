"""Regenerate the parametric-bootstrap KS p-value reference from SciPy.

Run from the repository root:

    python validation/scripts/scipy_ks_bootstrap.py

Writes `validation/data/ks_bootstrap.csv`, 40 synthetic losses drawn from a
gamma with shape 1.5 and scale 20,000, none truncated or censored, and
`validation/reference/ks_bootstrap_scipy.csv`: for each body family SciPy's
`goodness_of_fit` fits (loc 0 known), the Kolmogorov-Smirnov distance of
the maximum likelihood fit and the p-value of `scipy.stats.goodness_of_fit(
..., statistic="ks", n_mc_samples=9999)`, SciPy's own parametric bootstrap:
draw from the fit, refit, recompute the distance, `(k + 1) / (B + 1)`.

The p-value is a Monte Carlo estimate on both sides, so its tolerance is four
standard errors of the difference of two estimates with B = 9999 each,
`4 sqrt(2 p (1 - p) / 9999)`; the Rust test draws its own 9999 replicates.
"""

import csv
import math

import numpy as np
import scipy
from scipy import stats

DATA = "validation/data/ks_bootstrap.csv"
OUT = "validation/reference/ks_bootstrap_scipy.csv"
SOURCE = f"scipy {scipy.__version__}"
B = 9999

FAMILIES = {
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "loglogistic": stats.fisk,
    "exponential": stats.expon,
}


def main():
    rng = np.random.default_rng(812)
    y = np.sort(rng.gamma(1.5, 20_000.0, 40))
    with open(DATA, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["loss"])
        for v in y:
            w.writerow([repr(float(v))])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family", "quantity", "expected", "abs_tol", "rel_tol", "source"])
        for i, (name, dist) in enumerate(FAMILIES.items()):
            res = stats.goodness_of_fit(
                dist, y, known_params={"loc": 0.0}, statistic="ks",
                n_mc_samples=B, rng=np.random.default_rng(1000 + i),
            )
            p = float(res.pvalue)
            tol = 4 * math.sqrt(2 * max(p * (1 - p), 1 / B) / B)
            w.writerow([name, "ks_distance", repr(float(res.statistic)), 1e-6, 0.0, SOURCE])
            w.writerow([name, "ks_bootstrap_p_value", repr(p), round(tol, 4), 0.0, SOURCE])
            print(name, res.statistic, p, res.fit_result.params)


if __name__ == "__main__":
    main()
