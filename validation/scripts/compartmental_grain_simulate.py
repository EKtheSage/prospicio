"""Write validation/data/compartmental_sim_quarterly.csv.

A triangle with annual origins and quarterly development, simulated from
Gesmann and Morris' (2020) model 1 (docs/design/reserving-compartmental.md,
decision 13, PR I) with known parameters, for the unequal-grain parity test
(validation/tests/reserving_compartmental_grain.rs) and its Stan reference
(validation/scripts/reserving_compartmental_grain_stan.py). No public
dataset has unequal grain.

The model is the one of validation/scripts/stan/compartmental_gm1.stan, on
its own parameter scale: k_er = 3 e^(0.1 b_ker), k_p = e^(0.1 b_kp),
RLR_i = 0.7 e^(0.2 (b_RLR + r_i1)), RRF_i = 0.8 e^(0.1 (b_RRF + r_i2)),
r_i = diag(sd) L z_i with z_i standard normal and L the Cholesky factor of
the correlation rho, each cell normal about premium times the closed-form
outstanding or cumulative paid at its development time t in years (age in
months over 12), with sigma_os or sigma_pd. Ten accident years 2011 to 2020
valued at the end of 2020: origin 2011 has 40 quarterly ages (3 to 120
months), 2020 has four (3 to 12), 220 cells of each type.

Run from the repository root (numpy only):

    uv run --no-project --python 3.14 --with numpy==2.5.4 python validation/scripts/compartmental_grain_simulate.py
"""

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/data/compartmental_sim_quarterly.csv"
SEED = 20261022
ORIGINS = list(range(2011, 2021))
VALUATION = 2020
DEV_MONTHS = 3

# The true parameters, on the Stan program's scale.
TRUTH = {
    "b_ker": 1.0,
    "b_kp": -2.0,
    "b_RLR": 1.0,
    "b_RRF": -1.0,
    "tau_RLR": 0.4,
    "tau_RRF": 0.2,
    "rho": 0.3,
    "sigma_os": 1500.0,
    "sigma_pd": 500.0,
}


def outstanding(t, ker, kp, rlr):
    return rlr * ker / (ker - kp) * (np.exp(-kp * t) - np.exp(-ker * t))


def paid(t, ker, kp, rlr, rrf):
    return rlr * rrf / (ker - kp) * (ker * (1 - np.exp(-kp * t)) - kp * (1 - np.exp(-ker * t)))


def main():
    rng = np.random.default_rng(SEED)
    k_er = 3 * np.exp(0.1 * TRUTH["b_ker"])
    k_p = np.exp(0.1 * TRUTH["b_kp"])
    rho = TRUTH["rho"]
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho * rho)]])
    sd = np.array([TRUTH["tau_RLR"], TRUTH["tau_RRF"]])
    rows = []
    origin_truth = []
    for j, o in enumerate(ORIGINS):
        premium = round(100_000 * 1.02**j)
        r = sd * (chol @ rng.standard_normal(2))
        rlr = 0.7 * np.exp(0.2 * (TRUTH["b_RLR"] + r[0]))
        rrf = 0.8 * np.exp(0.1 * (TRUTH["b_RRF"] + r[1]))
        origin_truth.append((o, premium, rlr, rrf))
        n_ages = (VALUATION - o + 1) * 12 // DEV_MONTHS
        for k in range(1, n_ages + 1):
            age = k * DEV_MONTHS
            t = age / 12
            os_ = premium * outstanding(t, k_er, k_p, rlr) + TRUTH["sigma_os"] * rng.standard_normal()
            pd_ = premium * paid(t, k_er, k_p, rlr, rrf) + TRUTH["sigma_pd"] * rng.standard_normal()
            rows.append(f"{o},{age},{pd_:.2f},{os_:.2f},{premium}")
    truth = ", ".join(f"{k} {v!r}" for k, v in TRUTH.items())
    header = [
        "# Simulated by validation/scripts/compartmental_grain_simulate.py (seed "
        f"{SEED}): Gesmann and Morris' model 1, annual origins, quarterly development,",
        f"# accident years {ORIGINS[0]}-{ORIGINS[-1]} valued at the end of {VALUATION}; "
        "development in months, t = development / 12 years.",
        f"# Truth: {truth}; k_er {float(k_er)!r}, k_p {float(k_p)!r}.",
        "# Each origin's truth (origin premium RLR RRF): "
        + "; ".join(f"{o} {p} {float(a)!r} {float(b)!r}" for o, p, a, b in origin_truth),
        "origin,development,paid,outstanding,premium",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}; k_er {k_er:.4f} k_p {k_p:.4f}")
    for o, p, a, b in origin_truth:
        print(o, p, round(a, 4), round(b, 4), round(a * b, 4))


if __name__ == "__main__":
    main()
