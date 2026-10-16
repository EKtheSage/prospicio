"""Regenerate validation/reference/credibility_bayes_grid.csv.

    pip install numpy
    python validation/scripts/credibility_bayes_grid.py

Exact posterior moments of prospicio_bayes::credibility's one-level model
(Buhlmann-Straub's, Bayesian) on Hachemeister's data
(validation/reference/hachemeister.csv), by integration on a grid, for the
NUTS fit to be checked against:

    X_ij | theta_i ~ N(theta_i, sigma^2 / w_ij)
    theta_i | mu   ~ N(mu, tau^2),   mu flat
    sigma ~ half-normal(60000),  tau ~ half-normal(1500)

The group means and mu are integrated out by numerical quadrature on a
grid of mu at each (sigma, tau), written independently of the Rust code
(which integrates them in closed form); the posterior of (log sigma,
log tau), with the Jacobians, is then integrated on a 2-D grid. Each
group's premium is E[theta_i | data], the posterior mean of
Z_i Xbar_i + (1 - Z_i) E[mu | sigma, tau, data], whose own expectation over
mu's conditional is found by the same quadrature.

Tolerances: a tenth of the posterior standard deviation for means, 8% for
standard deviations, as validation/scripts/bayes_glm_grid.py, except 20%
for the standard deviation of tau^2: its posterior has a long right tail
(sd 1.7 times the mean), and at the test's 10,000 draws the sample sd was
14% low on one seed; 100,000 draws on three seeds gave 360,000 against the
grid's 357,000.
"""

import csv

import numpy as np

DATA = "validation/reference/hachemeister.csv"
OUT = "validation/reference/credibility_bayes_grid.csv"
SIGMA_SCALE = 60000.0
TAU_SCALE = 1500.0


def main():
    ratios = np.zeros((5, 12))
    weights = np.zeros((5, 12))
    with open(DATA) as f:
        for row in csv.DictReader(f):
            i, j = int(row["state"]) - 1, int(row["quarter"]) - 1
            ratios[i, j] = float(row["ratio"])
            weights[i, j] = float(row["weight"])
    w = weights.sum(axis=1)
    xbar = (weights * ratios).sum(axis=1) / w
    ss = (weights * (ratios - xbar[:, None]) ** 2).sum()
    n_obs, n_groups = ratios.size, 5

    u = np.linspace(np.log(6000.0), np.log(25000.0), 401)  # log sigma
    v = np.linspace(np.log(1.0), np.log(12000.0), 1201)  # log tau
    mu = np.linspace(xbar.min() - 3000.0, xbar.max() + 3000.0, 3001)
    dmu = mu[1] - mu[0]

    U, V = np.meshgrid(u, v, indexing="ij")
    sigma, tau = np.exp(U), np.exp(V)
    log_post = np.empty_like(U)
    e_mu = np.empty_like(U)
    for a in range(len(u)):
        s = sigma[a, :, None]
        t = tau[a, :, None]
        # log of prod_i N(xbar_i; mu, s^2/w_i + t^2), on the mu grid.
        lp_mu = np.zeros((len(v), len(mu)))
        for i in range(n_groups):
            var = s**2 / w[i] + t**2
            lp_mu += -0.5 * np.log(2 * np.pi * var) - (xbar[i] - mu[None, :]) ** 2 / (2 * var)
        top = lp_mu.max(axis=1, keepdims=True)
        dens = np.exp(lp_mu - top)
        mass = dens.sum(axis=1) * dmu
        log_int = np.log(mass) + top[:, 0]
        e_mu[a] = (dens * mu[None, :]).sum(axis=1) * dmu / mass
        sg = sigma[a]
        within = -(n_obs - n_groups) * np.log(sg) - ss / (2 * sg**2)
        prior = -(sg**2) / (2 * SIGMA_SCALE**2) - tau[a] ** 2 / (2 * TAU_SCALE**2)
        log_post[a] = log_int + within + prior + U[a] + V[a]
    p = np.exp(log_post - log_post.max())
    p /= p.sum()

    def moments(x):
        m = (p * x).sum()
        return m, np.sqrt((p * (x - m) ** 2).sum())

    rows = []

    def add(param, quantity, value, abs_tol, rel_tol):
        rows.append(
            {
                "case": "hachemeister",
                "param": param,
                "quantity": quantity,
                "expected": repr(float(value)),
                "abs_tol": repr(float(abs_tol)),
                "rel_tol": repr(float(rel_tol)),
                "source": "NumPy grid integration",
            }
        )

    for name, x in [("within_variance", sigma**2), ("between_variance[0]", tau**2)]:
        m, sd = moments(x)
        add(name, "mean", m, sd / 10, 0.0)
        add(name, "sd", sd, 0.0, 0.2 if name.startswith("between") else 0.08)
    m, sd = moments(e_mu)
    add("collective_mean", "mean", m, sd / 10, 0.0)
    for i in range(n_groups):
        z = w[i] * tau**2 / (w[i] * tau**2 + sigma**2)
        prem = z * xbar[i] + (1 - z) * e_mu
        m, sd = moments(prem)
        add(f"premium[{i}]", "mean", m, sd / 10, 0.0)
        mz, sdz = moments(z)
        add(f"credibility[{i}]", "mean", mz, sdz / 10, 0.0)

    with open(OUT, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} rows")
    # Edges of the grid hold no mass.
    print("mass at the edges:", p[0].sum(), p[-1].sum(), p[:, 0].sum(), p[:, -1].sum())


if __name__ == "__main__":
    main()
