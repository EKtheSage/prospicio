"""Regenerate validation/reference/reserving_compartmental_incurred_stan.csv.

The reference for prospicio_reserving::bayes::Compartmental with
`fitted: Fitted::PaidAndIncurred` (docs/design/reserving-compartmental.md,
decision 12): Gesmann and Morris' (2020) model 1 with incurred amounts
in place of outstanding as the second response (the mean of an incurred cell
is the premium times RLR (OS(t) + RRF PD(t))), the Stan program
validation/scripts/stan/compartmental_gm1_incurred.stan, which is
compartmental_gm1.stan with that one change. Same data as
reserving_compartmental_stan.py (company 337, calendar years up to 1996, 90
observations over the 9 accident years with data; the second response is
the incurred of the CAS Loss Reserving Database as the paper's `incurred`
column), same sampler settings and tolerances. There is no printed
posterior of this variant to check against.

Run from the repository root, in PowerShell or cmd:

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 --with arviz==0.23.4 python validation/scripts/reserving_compartmental_incurred_stan.py

Needs CmdStan 2.39.0 and RTools 4.5 (knowledge/environment/local-cmdstan.md).
CI reads the committed CSV and never runs Stan. CmdStan's output goes to
target/stan/compartmental_gm1_incurred_output/; --reuse reads the draws from
there (when they came from the same program, data and seed).
"""

import hashlib
import json
import sys

import arviz as az
import cmdstanpy
import numpy as np
import pandas as pd

from reserving_meyers_stan import CMDSTAN, ROOT, compile_model, summarise, toolchain

DATA = ROOT / "validation/data/clrd_wkcomp_337.csv"
OUT = ROOT / "validation/reference/reserving_compartmental_incurred_stan.csv"
STAN = ROOT / "validation/scripts/stan/compartmental_gm1_incurred.stan"
SEED = 20261014
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 15
LAST_TRAIN_YEAR = 1996
DATASET = "wkcomp_337"

# Tolerances, with s the reference posterior sd: those of
# reserving_compartmental_stan.py (decision 7), where each is at least 4
# times the seed-to-seed sd of our defaults over seeds 0 to 199. For this
# variant the same multiples were checked on a sweep of fewer seeds (the
# ignored test seed_sweep in validation/tests/reserving_compartmental_incurred.rs,
# COMPARTMENTAL_SEEDS seeds), recorded in
# knowledge/findings/compartmental-paid-incurred.md.
MEAN_TOL = 0.15  # x s: parameter means, reserve means and test means
SD_REL = 0.15  # parameter sds
RESERVE_SD_REL = 0.15  # reserve sds
QUANTILE_TOL = 0.3  # x s: reserve quantiles
ELPD_TOL = 1.5
EXACT = 1e-9

def cells():
    """The square: origin, age in years, incurred, paid, premium."""
    df = pd.read_csv(DATA, comment="#")
    df["age"] = df["development"] // 12
    df["os"] = df["incurred"]  # the second response is incurred
    return df


def stan_data(df):
    """The training cells (calendar years up to LAST_TRAIN_YEAR), incurred
    first, then paid, each by origin and age; and the test diagonal."""
    origins = sorted(df["origin"].unique())
    n = len(origins)
    premium = df.groupby("origin")["premium"].first().loc[origins].to_numpy(float)
    cal = df["origin"] + df["age"] - 1
    train = df[cal <= LAST_TRAIN_YEAR].sort_values(["origin", "age"])
    test = df[cal == LAST_TRAIN_YEAR + 1].sort_values(["origin", "age"])
    with_data = sorted(train["origin"].unique())
    level = [with_data.index(o) + 1 if o in with_data else 0 for o in origins]
    latest = train.groupby("origin")["paid"].last()
    latest_paid = [float(latest.get(o, 0.0)) for o in origins]
    y, t, delta, origin = [], [], [], []
    for d, col in [(0, "os"), (1, "paid")]:
        y += train[col].astype(float).tolist()
        t += train["age"].astype(float).tolist()
        delta += [d] * len(train)
        origin += [origins.index(o) + 1 for o in train["origin"]]
    test_origin, test_t, test_delta, test_args = [], [], [], []
    for d, name in [(0, "incurred"), (1, "paid")]:
        for o, a in zip(test["origin"], test["age"]):
            test_origin.append(origins.index(o) + 1)
            test_t.append(float(a))
            test_delta.append(d)
            test_args.append(f"{name}_{o}")
    data = {
        "N": len(y),
        "Y": y,
        "t": t,
        "delta": delta,
        "n_origin": n,
        "premium": premium.tolist(),
        "N_1": len(with_data),
        "level": level,
        "origin": origin,
        "t_last": float(n),
        "latest_paid": latest_paid,
        "N_test": len(test_origin),
        "test_origin": test_origin,
        "test_t": test_t,
        "test_delta": test_delta,
    }
    assert data["N"] == 90 and data["N_1"] == 9 and n == 10, (data["N"], data["N_1"], n)
    assert data["N_test"] == 20
    return data, origins, with_data, test_args


def main():
    toolchain()
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    sha = hashlib.sha256(STAN.read_bytes()).hexdigest()
    data, origins, with_data, test_args = stan_data(cells())
    output = ROOT / "target/stan/compartmental_gm1_incurred_output"
    stamp = output / "stamp.json"
    key = {"sha": sha, "data": data, "seed": SEED, "chains": CHAINS, "warmup": WARMUP,
           "draws": DRAWS, "adapt_delta": ADAPT_DELTA, "max_treedepth": MAX_TREEDEPTH}
    if "--reuse" in sys.argv and stamp.exists() and json.loads(stamp.read_text()) == key:
        fit = cmdstanpy.from_csv(str(output))
        print(f"reusing the draws in {output}")
    else:
        model = compile_model(STAN)
        output.mkdir(parents=True, exist_ok=True)
        for old in output.glob("*"):
            old.unlink()
        fit = model.sample(
            data=data,
            chains=CHAINS,
            iter_warmup=WARMUP,
            iter_sampling=DRAWS,
            seed=SEED,
            adapt_delta=ADAPT_DELTA,
            max_treedepth=MAX_TREEDEPTH,
            output_dir=str(output),
            show_progress=False,
        )
        stamp.write_text(json.dumps(key))
    draws = {
        "b_ker": fit.stan_variable("b_oker"),
        "b_kp": fit.stan_variable("b_okp"),
        "b_RLR": fit.stan_variable("b_oRLR"),
        "b_RRF": fit.stan_variable("b_oRRF"),
        "tau_RLR": fit.stan_variable("sd_1")[:, 0],
        "tau_RRF": fit.stan_variable("sd_1")[:, 1],
        "rho": fit.stan_variable("rho"),
        "log_sigma_os": fit.stan_variable("b_sigma")[:, 0],
        "log_sigma_pd": fit.stan_variable("b_sigma")[:, 1],
        "k_er": fit.stan_variable("k_er"),
        "k_p": fit.stan_variable("k_p"),
        "RLR": fit.stan_variable("RLR_pop"),
        "RRF": fit.stan_variable("RRF_pop"),
        "sigma_os": fit.stan_variable("sigma_os"),
        "sigma_pd": fit.stan_variable("sigma_pd"),
    }
    for name in ["RLR", "RRF"]:
        values = fit.stan_variable(name)
        for o in with_data:
            draws[f"{name}[{o}]"] = values[:, origins.index(o)]
    summary = fit.summary()
    rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
    sampled = ["b_oRLR", "b_oRRF", "b_oker", "b_okp", "sd_1[1]", "sd_1[2]", "rho",
               "b_sigma[1]", "b_sigma[2]"]
    sampled += [f"r_1[{j},{k}]" for j in range(1, 10) for k in (1, 2)]
    worst = summary.loc[sampled, rhat_col].max()
    method_vars = fit.method_variables()
    divergences = int(method_vars["divergent__"].sum())
    depth = int((method_vars["treedepth__"] >= MAX_TREEDEPTH).sum())
    swapped = int((draws["k_er"] <= draws["k_p"]).sum())
    loo = az.loo(az.from_cmdstanpy(fit, log_likelihood="log_lik"))
    print(
        f"max R-hat {worst:.5f}, {divergences} divergences, {depth} draws at the maximum depth,"
        f" {swapped} draws with k_er <= k_p, elpd_loo {loo.elpd_loo:.3f}"
        f" (p_loo {loo.p_loo:.3f}, max k {float(np.max(loo.pareto_k)):.3f})"
    )
    assert worst <= 1.005, f"R-hat {worst} above 1.005"
    assert divergences == 0, f"{divergences} divergent transitions"

    rows = []
    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: compartmental_gm1_incurred.stan sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) adapt_delta {ADAPT_DELTA}"
        f" max_treedepth {MAX_TREEDEPTH}"
    )

    def row(method, quantity, arg, expected, abs_tol, rel_tol, src):
        rows.append(
            f"{DATASET},{method},{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},{float(rel_tol)!r},{src}"
        )

    method = "stan_gm1_incurred"
    for name, values in draws.items():
        s = summarise(values)
        row(method, "mean", name, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
        row(method, "sd", name, s["sd"], 0.0, SD_REL, source)
    ulr = fit.stan_variable("ulr")
    for k, o in enumerate(origins):
        s = summarise(ulr[:, k])
        row(method, "ulr_mean", str(o), s["mean"], MEAN_TOL * s["sd"], 0.0, source)
    reserve = fit.stan_variable("reserve")
    targets = [(str(o), reserve[:, k]) for k, o in enumerate(origins)]
    targets.append(("total", fit.stan_variable("total_reserve")))
    for arg, values in targets:
        s = summarise(values)
        row(method, "reserve_mean", arg, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
        row(method, "reserve_sd", arg, s["sd"], 0.0, RESERVE_SD_REL, source)
        for q in ["q05", "q50", "q95"]:
            row(method, f"reserve_{q}", arg, s[q], QUANTILE_TOL * s["sd"], 0.0, source)
    test_mean = fit.stan_variable("test_mean")
    for k, arg in enumerate(test_args):
        s = summarise(test_mean[:, k])
        row(method, "test_mean", arg, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
    row(method, "elpd_loo", "", loo.elpd_loo, ELPD_TOL, 0.0, source)

    header = [
        f"# Generated by validation/scripts/reserving_compartmental_incurred_stan.py with {cmdstan}, cmdstanpy {cmdstanpy.__version__},"
        f" pandas {pd.__version__}, numpy {np.__version__} and arviz {az.__version__} (elpd_loo).",
        f"# Model validation/scripts/stan/compartmental_gm1_incurred.stan (sha256 {sha}), Gesmann and Morris' (2020) model 1",
        "# fitted to paid and incurred (decision 12 of the compartmental design note). Data",
        f"# validation/data/clrd_wkcomp_337.csv, company 337, calendar years up to {LAST_TRAIN_YEAR} (90 observations,",
        "# incurred and paid, direct premium); the 1997 diagonal is the test set. Seed"
        f" {SEED}; {CHAINS} chains of {WARMUP} warm-up",
        f"# and {DRAWS} kept draws; adapt_delta {ADAPT_DELTA}; max_treedepth {MAX_TREEDEPTH}. With s the reference posterior sd:",
        "# means, test means and reserve quantiles abs_tol a multiple of s, sds rel_tol, elpd_loo abs_tol, each at least 4",
        "# times the seed-to-seed sd of our defaults over seeds 0 to 199 (every seed passes). Reserves are paid to age 10 less the latest paid;",
        "# 1997 has no training cell and takes a new effect from the population in each draw.",
        "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
