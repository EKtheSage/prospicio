"""Write validation/reference/hawkes_scipy.csv: Hawkes fits by SciPy.

Independent of prospicio: paths simulated by Ogata's thinning with numpy,
and the exponential-kernel log-likelihood maximised by scipy.optimize
(L-BFGS-B on the log parameters, from several starts, then Nelder-Mead to
polish). Each row is one data set: periods of length `horizon` years
(";"-joined, the times within a period "|"-joined) and the maximum
likelihood estimates. Rust's prospicio_aggregate::Hawkes::fit must find the
same optimum.

python3 validation/scripts/hawkes_scipy.py   (numpy and SciPy 1.14.1)
"""

from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parents[1] / "reference/hawkes_scipy.csv"


def ogata(rng, mu, alpha, beta, horizon):
    t, times, excite = 0.0, [], 0.0
    while True:
        lam_bar = mu + excite
        w = rng.exponential(1.0 / lam_bar)
        excite *= np.exp(-beta * w)
        t += w
        if t >= horizon:
            return times
        if rng.uniform() * lam_bar <= mu + excite:
            times.append(t)
            excite += alpha


def loglik(params, periods, horizon):
    mu, alpha, beta = params
    ll = -mu * horizon * len(periods)
    for p in periods:
        p = np.asarray(p)
        a = np.zeros(len(p))
        for i in range(1, len(p)):
            a[i] = np.exp(-beta * (p[i] - p[i - 1])) * (1.0 + a[i - 1])
        ll += np.sum(np.log(mu + alpha * a)) - alpha / beta * np.sum(1.0 - np.exp(-beta * (horizon - p)))
    return ll


def fit(periods, horizon):
    f = lambda x: -loglik(np.exp(x), periods, horizon)
    best = None
    for start in ([0.0, 0.0, 1.0], [0.5, 1.5, 2.5], [-0.5, 1.0, 1.5], [1.0, 2.0, 3.0]):
        r = minimize(f, start, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    r = minimize(f, best.x, method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 40000})
    return np.exp(r.x), -r.fun


def main():
    rng = np.random.default_rng(567)
    lines = ["case,horizon,periods,baseline,alpha,beta,log_likelihood,source"]
    cases = [
        ("one long path", (1.2, 3.0, 6.0), 1, 60.0),
        ("years", (2.0, 12.0, 24.0), 150, 1.0),
        ("weak excitation", (5.0, 1.0, 8.0), 80, 1.0),
        ("strong clustering", (0.8, 16.0, 20.0), 120, 1.0),
    ]
    for name, (mu, a, b), n, horizon in cases:
        periods = [ogata(rng, mu, a, b, horizon) for _ in range(n)]
        (m, al, be), ll = fit(periods, horizon)
        enc = ";".join("|".join(f"{t:.17g}" for t in p) for p in periods)
        lines.append(f"{name},{horizon},{enc},{m:.17g},{al:.17g},{be:.17g},{ll:.17g},SciPy {scipy.__version__}")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
