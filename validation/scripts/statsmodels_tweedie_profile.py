"""Regenerate validation/reference/tweedie_profile_statsmodels.csv.

    python validation/scripts/statsmodels_tweedie_profile.py

Profile likelihood of a Tweedie GLM's power on
validation/data/glm_policies.csv: pure premium on intercept, age and region
(treatment coding, reference A), log link, exposure as variance weights.
For each power p, statsmodels fits the GLM; the dispersion phi then
maximizes the exact Tweedie log-likelihood, with row i at dispersion
phi / exposure_i. The density is the compound Poisson-gamma series summed
here in log space with NumPy, independently of the Rust code:

    y = 0:  log f = -lam
    y > 0:  log f = logsumexp_n [log Pois(n; lam) + log Gamma(y; n alpha, theta)]

with lam = mu^(2-p) / (phi (2-p)), alpha = (2-p)/(p-1) and
theta = phi (p-1) mu^(p-1). The best power is refined between the grid
points around the best one. Records the profile log-likelihood and phi at
each grid power, the refined power, its phi and log-likelihood, and the
95% profile-likelihood interval for the power.
"""

import csv
import sys

import numpy as np
import statsmodels
import statsmodels.api as sm
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, logsumexp

OUT = "validation/reference/tweedie_profile_statsmodels.csv"
SOURCE = f"statsmodels {statsmodels.__version__} + NumPy series"
POWERS = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]


def data():
    rows = list(csv.DictReader(open("validation/data/glm_policies.csv")))
    age = np.array([float(r["age"]) for r in rows])
    region = [r["region"] for r in rows]
    X = np.column_stack(
        [np.ones(len(rows)), age] + [np.array([float(g == lv) for g in region]) for lv in "BCD"]
    )
    y = np.array([float(r["pure"]) for r in rows])
    w = np.array([float(r["exposure"]) for r in rows])
    return X, y, w


def log_density(y, mu, phi, p):
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    theta = phi * (p - 1) * mu ** (p - 1)
    out = -lam.copy()
    pos = y > 0
    yp, lp, tp = y[pos], lam[pos], theta[pos]
    n_max = int(max(200, 20 * (yp / tp).max() / alpha))
    n = np.arange(1, n_max + 1)[:, None]
    terms = (
        n * np.log(lp) - lp - gammaln(n + 1)
        + (n * alpha - 1) * np.log(yp) - yp / tp - gammaln(n * alpha) - n * alpha * np.log(tp)
    )
    out[pos] = logsumexp(terms, axis=0)
    return out


def profile(X, y, w, p):
    fam = sm.families.Tweedie(var_power=p, link=sm.families.links.Log())
    res = sm.GLM(y, X, family=fam, var_weights=w).fit(tol=1e-14, maxiter=200)
    mu = res.fittedvalues
    ll = lambda t: -log_density(y, mu, np.exp(t) / w, p).sum()
    start = np.log(res.scale)
    opt = minimize_scalar(ll, bounds=(start - 5, start + 5), method="bounded",
                          options={"xatol": 1e-12})
    return -opt.fun, float(np.exp(opt.x))


def main():
    X, y, w = data()
    grid = [profile(X, y, w, p) for p in POWERS]
    best = max(range(len(POWERS)), key=lambda i: grid[i][0])
    lo, hi = POWERS[max(best - 1, 0)], POWERS[min(best + 1, len(POWERS) - 1)]
    opt = minimize_scalar(lambda p: -profile(X, y, w, p)[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-7})
    p_hat = float(opt.x)
    ll_hat, phi_hat = profile(X, y, w, p_hat)
    # 95% profile-likelihood interval: where the profile is 1.92 below its
    # maximum (half the chi-squared(1) 95% point), bracketed by the grid.
    drop = lambda p: profile(X, y, w, p)[0] - (ll_hat - 3.841458820694124 / 2)
    lower = brentq(drop, POWERS[best - 1], p_hat, xtol=1e-12)
    upper = brentq(drop, p_hat, POWERS[best + 1], xtol=1e-12)
    with open(OUT, "w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for p, (ll, phi) in zip(POWERS, grid):
            out.writerow(["log_likelihood", p, repr(float(ll)), 0.0, 1e-9, SOURCE])
            out.writerow(["dispersion", p, repr(phi), 0.0, 1e-6, SOURCE])
        # The profile is flat at its peak: the power agrees to about 1e-3.
        out.writerow(["power", "", repr(p_hat), 2e-3, 0.0, SOURCE])
        out.writerow(["max_log_likelihood", "", repr(float(ll_hat)), 0.0, 1e-9, SOURCE])
        out.writerow(["dispersion_at_power", "", repr(phi_hat), 0.0, 1e-3, SOURCE])
        out.writerow(["interval_lower", "", repr(lower), 1e-6, 0.0, SOURCE])
        out.writerow(["interval_upper", "", repr(upper), 1e-6, 0.0, SOURCE])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
