"""Regenerate validation/reference/reserving_compartmental_grain_stan.csv.

The reference for prospicio_reserving::bayes::Compartmental on a triangle
whose origin and development grains differ (docs/design/reserving-
compartmental.md, decision 13, PR I): Gesmann and Morris' (2020) model 1 with
its defaults, the Stan program validation/scripts/stan/compartmental_gm1.stan
unchanged (it already takes each cell's development time t as data), on the
triangle simulated from that model with known parameters by
validation/scripts/compartmental_grain_simulate.py
(validation/data/compartmental_sim_quarterly.csv: accident years 2011 to 2020,
quarterly development, 220 outstanding and 220 paid cells). Each cell's t is
its age in months over 12, in years, as on annual triangles; the square ends
at ten years (t_last = 10, 40 quarterly ages), so each origin's reserve is the
paid at 120 months less its latest paid, and 0 for 2011, observed there
(decision 5; the Stan program draws one, which the script replaces with 0).
Every origin has data, so no origin
is drawn from the population, and there is no test diagonal.

Run from the repository root, in PowerShell or cmd:

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 --with arviz==0.23.4 python validation/scripts/reserving_compartmental_grain_stan.py

Needs CmdStan 2.39.0 and RTools 4.5 (knowledge/environment/local-cmdstan.md).
CI reads the committed CSV and never runs Stan. CmdStan's output goes to
target/stan/compartmental_gm1_grain_output/; --reuse reads the draws from
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

DATA = ROOT / "validation/data/compartmental_sim_quarterly.csv"
OUT = ROOT / "validation/reference/reserving_compartmental_grain_stan.csv"
STAN = ROOT / "validation/scripts/stan/compartmental_gm1.stan"
SEED = 20261022
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 15
DATASET = "sim_quarterly"

# Tolerances, with s the reference posterior sd: those of
# reserving_compartmental_stan.py (decision 7), checked on a sweep of 20
# seeds (the ignored test seed_sweep in
# validation/tests/reserving_compartmental_grain.rs, recorded in
# knowledge/findings/compartmental-grain.md).
MEAN_TOL = 0.15  # x s: parameter means and reserve means
SD_REL = 0.15  # parameter sds
RESERVE_SD_REL = 0.15  # reserve sds
QUANTILE_TOL = 0.3  # x s: reserve quantiles
ELPD_TOL = 1.5


def stan_data():
    df = pd.read_csv(DATA, comment="#").sort_values(["origin", "development"])
    origins = sorted(df["origin"].unique())
    n = len(origins)
    premium = df.groupby("origin")["premium"].first().loc[origins].to_numpy(float)
    latest = df.groupby("origin")["paid"].last()
    latest_paid = [float(latest.loc[o]) for o in origins]
    y, t, delta, origin = [], [], [], []
    for d, col in [(0, "outstanding"), (1, "paid")]:
        y += df[col].astype(float).tolist()
        t += (df["development"] / 12.0).astype(float).tolist()
        delta += [d] * len(df)
        origin += [origins.index(o) + 1 for o in df["origin"]]
    data = {
        "N": len(y),
        "Y": y,
        "t": t,
        "delta": delta,
        "n_origin": n,
        "premium": premium.tolist(),
        "N_1": n,
        "level": list(range(1, n + 1)),
        "origin": origin,
        "t_last": float(n),
        "latest_paid": latest_paid,
        "N_test": 0,
        "test_origin": [],
        "test_t": [],
        "test_delta": [],
    }
    assert data["N"] == 440 and n == 10, (data["N"], n)
    # Origins observed at the square's last age (2011 at 120 months): their
    # reserve is 0 (decision 5), where the Stan program draws one.
    last = df.groupby("origin")["development"].max()
    closed = [o for o in origins if last.loc[o] / 12.0 >= n]
    return data, origins, closed


def main():
    toolchain()
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    sha = hashlib.sha256(STAN.read_bytes()).hexdigest()
    data_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    data, origins, closed = stan_data()
    output = ROOT / "target/stan/compartmental_gm1_grain_output"
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
        for k, o in enumerate(origins):
            draws[f"{name}[{o}]"] = values[:, k]
    summary = fit.summary()
    rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
    sampled = ["b_oRLR", "b_oRRF", "b_oker", "b_okp", "sd_1[1]", "sd_1[2]", "rho",
               "b_sigma[1]", "b_sigma[2]"]
    sampled += [f"r_1[{j},{k}]" for j in range(1, len(origins) + 1) for k in (1, 2)]
    worst = summary.loc[sampled, rhat_col].max()
    method_vars = fit.method_variables()
    divergences = int(method_vars["divergent__"].sum())
    depth = int((method_vars["treedepth__"] >= MAX_TREEDEPTH).sum())
    swapped = int((draws["k_er"] <= draws["k_p"]).sum())
    loo = az.loo(az.from_cmdstanpy(fit, log_likelihood="log_lik"))
    reserve = fit.stan_variable("reserve").copy()
    for o in closed:
        reserve[:, origins.index(o)] = 0.0
    total_reserve = reserve.sum(axis=1)
    total = summarise(total_reserve)
    print(
        f"max R-hat {worst:.5f}, {divergences} divergences, {depth} draws at the maximum depth,"
        f" {swapped} draws with k_er <= k_p, elpd_loo {loo.elpd_loo:.3f}"
        f" (p_loo {loo.p_loo:.3f}, max k {float(np.max(loo.pareto_k)):.3f}),"
        f" total reserve {total['mean']:.0f} (sd {total['sd']:.0f})"
    )
    for name in ["b_ker", "b_kp", "b_RLR", "b_RRF", "tau_RLR", "tau_RRF", "rho", "sigma_os",
                 "sigma_pd", "k_er", "k_p"]:
        s = summarise(draws[name])
        q = np.quantile(draws[name], [0.025, 0.975])
        print(f"{name}: mean {s['mean']:.4f} sd {s['sd']:.4f} 95% [{q[0]:.4f}, {q[1]:.4f}]")
    assert worst <= 1.005, f"R-hat {worst} above 1.005"
    assert divergences == 0, f"{divergences} divergent transitions"

    rows = []
    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: compartmental_gm1.stan sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) adapt_delta {ADAPT_DELTA}"
        f" max_treedepth {MAX_TREEDEPTH}"
    )

    def row(quantity, arg, expected, abs_tol, rel_tol):
        rows.append(
            f"{DATASET},stan_gm1_grain,{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},"
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
    targets = [(str(o), reserve[:, k]) for k, o in enumerate(origins) if o not in closed]
    targets.append(("total", total_reserve))
    for arg, values in targets:
        s = summarise(values)
        row("reserve_mean", arg, s["mean"], MEAN_TOL * s["sd"], 0.0)
        row("reserve_sd", arg, s["sd"], 0.0, RESERVE_SD_REL)
        for q in ["q05", "q50", "q95"]:
            row(f"reserve_{q}", arg, s[q], QUANTILE_TOL * s["sd"], 0.0)
    row("elpd_loo", "", loo.elpd_loo, ELPD_TOL, 0.0)

    header = [
        f"# Generated by validation/scripts/reserving_compartmental_grain_stan.py with {cmdstan}, cmdstanpy"
        f" {cmdstanpy.__version__}, pandas {pd.__version__}, numpy {np.__version__} and arviz {az.__version__} (elpd_loo).",
        f"# Model validation/scripts/stan/compartmental_gm1.stan (sha256 {sha}), Gesmann and Morris' (2020) model 1,",
        f"# on validation/data/compartmental_sim_quarterly.csv (sha256 {data_sha}), simulated from that model with known",
        "# parameters: annual origins 2011-2020, quarterly development, 440 observations, t = development / 12 years",
        "# (decision 13 of the compartmental design note). Reserves are paid to 120 months less the latest paid,",
        "# 0 for 2011 (observed at 120 months, decision 5); no test set.",
        f"# Seed {SEED}; {CHAINS} chains of {WARMUP} warm-up and {DRAWS} kept draws; adapt_delta {ADAPT_DELTA};",
        f"# max_treedepth {MAX_TREEDEPTH}. With s the reference posterior sd: means and reserve quantiles abs_tol a multiple",
        "# of s, sds rel_tol, elpd_loo abs_tol.",
        "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
