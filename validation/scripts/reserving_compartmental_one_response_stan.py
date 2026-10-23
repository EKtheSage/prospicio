"""Regenerate validation/reference/reserving_compartmental_one_response_stan.csv.

The references for prospicio_reserving::bayes::Compartmental with one
response and ordered rates (docs/design/reserving-compartmental.md,
decision 12, PR H): Gesmann and Morris' (2020) model 1 fitted to paid alone
(`fitted: Fitted::PaidOnly`, the Stan program
validation/scripts/stan/compartmental_gm1_paid_only.stan) and to incurred
alone (`fitted: Fitted::IncurredOnly`,
validation/scripts/stan/compartmental_gm1_incurred_only.stan), both with
`rate_order: RateOrder::Ordered` (k_er = k_p + e^delta, the paper's rate
priors restricted to k_er > k_p). Same data as
reserving_compartmental_incurred_stan.py (company 337, calendar years up to
1996, 45 cells of the one response over the 9 accident years with data; the
paid column supplies each origin's latest paid in both), same sampler
settings and tolerances. There is no printed posterior of either variant to
check against.

Run from the repository root, in PowerShell or cmd:

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 --with arviz==0.23.4 python validation/scripts/reserving_compartmental_one_response_stan.py

Needs CmdStan 2.39.0 and RTools 4.5 (knowledge/environment/local-cmdstan.md).
CI reads the committed CSV and never runs Stan. CmdStan's output goes to
target/stan/compartmental_gm1_<mode>_output/; --reuse reads the draws from
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
OUT = ROOT / "validation/reference/reserving_compartmental_one_response_stan.csv"
SEED = 20261022
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 15
LAST_TRAIN_YEAR = 1996
DATASET = "wkcomp_337"

# The two variants: the Stan program, the column of the one response, the
# name of its sigma in the posterior and the method name of the rows.
MODES = {
    "paid_only": {
        "stan": ROOT / "validation/scripts/stan/compartmental_gm1_paid_only.stan",
        "column": "paid",
        "sigma": "pd",
        "method": "stan_gm1_paid_only",
    },
    "incurred_only": {
        "stan": ROOT / "validation/scripts/stan/compartmental_gm1_incurred_only.stan",
        "column": "incurred",
        "sigma": "os",
        "method": "stan_gm1_incurred_only",
    },
}

# Tolerances, with s the reference posterior sd: those of
# reserving_compartmental_stan.py (decision 7), except ELPD_TOL, checked on
# a sweep of 20 seeds per mode (the ignored test seed_sweep in
# validation/tests/reserving_compartmental_one_response.rs, recorded in
# knowledge/findings/compartmental-one-response.md): the incurred-only
# elpd_loo, with Pareto k above 1, was up to 2.1 from Stan's.
MEAN_TOL = 0.15  # x s: parameter means, reserve means and test means
SD_REL = 0.15  # parameter sds
RESERVE_SD_REL = 0.15  # reserve sds
QUANTILE_TOL = 0.3  # x s: reserve quantiles
ELPD_TOL = 2.5


def stan_data(column):
    """The training cells of `column` (calendar years up to LAST_TRAIN_YEAR)
    by origin and age, each origin's latest training paid, and the test
    diagonal's cells of `column`."""
    df = pd.read_csv(DATA, comment="#")
    df["age"] = df["development"] // 12
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
    data = {
        "N": len(train),
        "Y": train[column].astype(float).tolist(),
        "t": train["age"].astype(float).tolist(),
        "n_origin": n,
        "premium": premium.tolist(),
        "N_1": len(with_data),
        "level": level,
        "origin": [origins.index(o) + 1 for o in train["origin"]],
        "t_last": float(n),
        "latest_paid": latest_paid,
        "N_test": len(test),
        "test_origin": [origins.index(o) + 1 for o in test["origin"]],
        "test_t": test["age"].astype(float).tolist(),
    }
    test_args = [f"{column}_{o}" for o in test["origin"]]
    assert data["N"] == 45 and data["N_1"] == 9 and n == 10, (data["N"], data["N_1"], n)
    assert data["N_test"] == 10
    return data, origins, with_data, test_args


def run(mode, spec, cmdstan, rows):
    stan = spec["stan"]
    sha = hashlib.sha256(stan.read_bytes()).hexdigest()
    data, origins, with_data, test_args = stan_data(spec["column"])
    output = ROOT / f"target/stan/compartmental_gm1_{mode}_output"
    stamp = output / "stamp.json"
    key = {"sha": sha, "data": data, "seed": SEED, "chains": CHAINS, "warmup": WARMUP,
           "draws": DRAWS, "adapt_delta": ADAPT_DELTA, "max_treedepth": MAX_TREEDEPTH}
    if "--reuse" in sys.argv and stamp.exists() and json.loads(stamp.read_text()) == key:
        fit = cmdstanpy.from_csv(str(output))
        print(f"{mode}: reusing the draws in {output}")
    else:
        model = compile_model(stan)
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
    sigma = spec["sigma"]
    draws = {
        "b_ker": fit.stan_variable("b_ker"),
        "b_kp": fit.stan_variable("b_okp"),
        "b_RLR": fit.stan_variable("b_oRLR"),
        "b_RRF": fit.stan_variable("b_oRRF"),
        "tau_RLR": fit.stan_variable("sd_1")[:, 0],
        "tau_RRF": fit.stan_variable("sd_1")[:, 1],
        "rho": fit.stan_variable("rho"),
        f"log_sigma_{sigma}": fit.stan_variable("b_sigma"),
        "k_er": fit.stan_variable("k_er"),
        "k_p": fit.stan_variable("k_p"),
        "RLR": fit.stan_variable("RLR_pop"),
        "RRF": fit.stan_variable("RRF_pop"),
        f"sigma_{sigma}": fit.stan_variable(f"sigma_{sigma}"),
    }
    for name in ["RLR", "RRF"]:
        values = fit.stan_variable(name)
        for o in with_data:
            draws[f"{name}[{o}]"] = values[:, origins.index(o)]
    summary = fit.summary()
    rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
    sampled = ["b_oRLR", "b_oRRF", "delta", "b_okp", "sd_1[1]", "sd_1[2]", "rho", "b_sigma"]
    sampled += [f"r_1[{j},{k}]" for j in range(1, 10) for k in (1, 2)]
    worst = summary.loc[sampled, rhat_col].max()
    method_vars = fit.method_variables()
    divergences = int(method_vars["divergent__"].sum())
    depth = int((method_vars["treedepth__"] >= MAX_TREEDEPTH).sum())
    loo = az.loo(az.from_cmdstanpy(fit, log_likelihood="log_lik"))
    total = summarise(fit.stan_variable("total_reserve"))
    print(
        f"{mode}: max R-hat {worst:.5f}, {divergences} divergences, {depth} draws at the maximum"
        f" depth, elpd_loo {loo.elpd_loo:.3f} (p_loo {loo.p_loo:.3f}, max k"
        f" {float(np.max(loo.pareto_k)):.3f}), total reserve {total['mean']:.0f}"
        f" (sd {total['sd']:.0f}), k_er {draws['k_er'].mean():.4f}, k_p {draws['k_p'].mean():.4f}"
    )
    assert worst <= 1.005, f"{mode}: R-hat {worst} above 1.005"
    assert divergences == 0, f"{mode}: {divergences} divergent transitions"

    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: {stan.name} sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) adapt_delta {ADAPT_DELTA}"
        f" max_treedepth {MAX_TREEDEPTH}"
    )

    def row(quantity, arg, expected, abs_tol, rel_tol):
        rows.append(
            f"{DATASET},{spec['method']},{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},"
            f"{float(rel_tol)!r},{source}"
        )

    for name, values in draws.items():
        s = summarise(values)
        row("mean", name, s["mean"], MEAN_TOL * s["sd"], 0.0)
        row("sd", name, s["sd"], 0.0, SD_REL)
    ulr = fit.stan_variable("ulr")
    for k, o in enumerate(origins):
        s = summarise(ulr[:, k])
        row("ulr_mean", str(o), s["mean"], MEAN_TOL * s["sd"], 0.0)
    reserve = fit.stan_variable("reserve")
    targets = [(str(o), reserve[:, k]) for k, o in enumerate(origins)]
    targets.append(("total", fit.stan_variable("total_reserve")))
    for arg, values in targets:
        s = summarise(values)
        row("reserve_mean", arg, s["mean"], MEAN_TOL * s["sd"], 0.0)
        row("reserve_sd", arg, s["sd"], 0.0, RESERVE_SD_REL)
        for q in ["q05", "q50", "q95"]:
            row(f"reserve_{q}", arg, s[q], QUANTILE_TOL * s["sd"], 0.0)
    test_mean = fit.stan_variable("test_mean")
    for k, arg in enumerate(test_args):
        s = summarise(test_mean[:, k])
        row("test_mean", arg, s["mean"], MEAN_TOL * s["sd"], 0.0)
    row("elpd_loo", "", loo.elpd_loo, ELPD_TOL, 0.0)
    return sha


def main():
    toolchain()
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    rows = []
    shas = {mode: run(mode, spec, cmdstan, rows) for mode, spec in MODES.items()}
    header = [
        f"# Generated by validation/scripts/reserving_compartmental_one_response_stan.py with {cmdstan}, cmdstanpy"
        f" {cmdstanpy.__version__}, pandas {pd.__version__}, numpy {np.__version__} and arviz {az.__version__} (elpd_loo).",
        f"# Models validation/scripts/stan/compartmental_gm1_paid_only.stan (sha256 {shas['paid_only']}) and",
        f"# compartmental_gm1_incurred_only.stan (sha256 {shas['incurred_only']}), Gesmann and Morris' (2020) model 1",
        "# fitted to paid alone and to incurred alone with ordered rates (decision 12 of the compartmental design note).",
        f"# Data validation/data/clrd_wkcomp_337.csv, company 337, calendar years up to {LAST_TRAIN_YEAR} (45 observations of",
        "# the one response, direct premium; the paid column gives each origin's latest paid); the 1997 diagonal is the test",
        f"# set. Seed {SEED}; {CHAINS} chains of {WARMUP} warm-up and {DRAWS} kept draws; adapt_delta {ADAPT_DELTA};",
        f"# max_treedepth {MAX_TREEDEPTH}. With s the reference posterior sd: means, test means and reserve quantiles",
        "# abs_tol a multiple of s, sds rel_tol, elpd_loo abs_tol. Reserves are paid to age 10 less the latest paid, drawn",
        "# with the paid sigma under paid only and the mean paid with no noise under incurred only; 1997 has no training",
        "# cell and takes a new effect from the population in each draw.",
        "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
