"""Regenerate validation/reference/reserving_meyers_cay_stan.csv.

The reference for prospicio_reserving::bayes::Meyers with MeyersModel::Cay
(docs/design/reserving-bayes.md, decisions 7 and 10): Meyers' own Stan
program, transcribed into current syntax in
validation/scripts/stan/meyers_cay.stan (provenance in its header), sampled
long enough that the reference's own Monte Carlo error is negligible next to
the tolerances, on incurred data of Meyers' illustrative insurer (group 353
Commercial Auto, validation/data/clrd_comauto_353.csv, the cells at calendar
years up to 1997), as Meyers fits CAY (2019, sec. 8). Run from the repository
root, in PowerShell or cmd (CmdStan's make does not detect the OS from Git
Bash):

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 --with arviz==0.23.4 python validation/scripts/reserving_meyers_cay_stan.py

Needs CmdStan 2.39.0 built at ~/.cmdstan/cmdstan-2.39.0 and RTools 4.5 at
C:/rtools45 (knowledge/environment/local-cmdstan.md); the data handling and
the toolchain are those of reserving_meyers_stan.py. CI reads the committed
CSV and never runs Stan.

Besides the Stan summaries, the CSV holds Meyers' printed numbers for the same
data (method meyers_2019_cay), a looser second check.
"""

import hashlib

import arviz as az
import cmdstanpy
import numpy as np
import pandas as pd

from reserving_meyers_stan import CMDSTAN, ROOT, compile_model, summarise, toolchain, training

OUT = ROOT / "validation/reference/reserving_meyers_cay_stan.csv"
COLUMN = "incurred"
SEED = 20261013
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 12
LAST_YEAR = 1997

# Tolerances, with s the reference posterior sd (decision 10). Each is at
# least 4 times the largest Monte Carlo error of Meyers' defaults (4 x 2,000
# draws, 10,000 simulations) in its family, measured as the seed-to-seed sd
# of every row over seeds 0 to 199 (Stan's own error is negligible): means
# 0.025 s (sigma[7]), parameter sds 3.3% (logelr), the reserve sds of origins
# 1989 to 1996 2.4% (1990), the sd of the sum of the reserves of 1988 to 1996
# 1.9%, quantiles 0.063 s (95% of 1997), the correlation of neighbouring
# origins' reserves 0.019 (1995 and 1996), elpd_loo 0.43. The total
# reserve's and total ultimate's sds have a seed-to-seed sd of 5.9% but a
# heavy tail (+66% at seed 38, from the youngest origin's log-normal), so
# they get 75%, and the sum without the youngest origin checks the joint
# spread instead. The youngest origin (1997) has no reserve sd row, as for
# CRC and CSR (its reserve rides on the one-cell alpha[10] through a
# log-normal); its 5%, 50% and 95% quantiles check its spread instead.
# Seeds 0 to 199 pass every row, using at most 0.88 of a tolerance.
MEAN_TOL = 0.15  # x s
SD_REL = 0.15  # parameter sds
RESERVE_SD_REL = 0.2  # an origin's reserve sd
TOTAL_SD_REL = 0.75  # the total reserve's and total ultimate's sds
SUM_SD_REL = 0.1  # the sd of the sum of all but the youngest origin's reserves
QUANTILE_TOL = 0.35  # x s
CORR_TOL = 0.08  # the correlation of neighbouring origins' reserves
ELPD_TOL = 2.0
EXACT = 1e-9

# Meyers (2019), CAS Monograph 8, 2nd ed., sec. 8, on the same data, 4 chains
# of 2,500 kept draws (CAY.R: iter 5000, half warm-up). "Estimate" is the mean
# total ultimate (the oldest origin's observed value plus the others' draws,
# each ceiling()ed, about +4.5 in all). (quantity, arg, value, abs_tol,
# rel_tol, table and page). Each tolerance is at least 4 times the combined
# Monte Carlo error of our defaults (the seed sweep) and of Meyers' single run
# (his error taken as ours for the totals and elpd_loo, and as
# 1 / sqrt(3,000 effective draws) of an sd for logelr and rho): the total
# ultimate's mean within 1% (combined error 0.11%); the means of logelr and
# rho within 0.15 printed sd (combined 0.024 sd); their sds within 20%
# (combined 3.8%); elpd_loo within 2.5 (combined 0.61). The total ultimate's
# sd within 75%, as for the Stan rows (its heavy tail).
MEYERS = [
    ("total_ultimate_mean", "", 39193, 0.0, 0.01, "Table 8.2 p. 39"),
    ("total_ultimate_sd", "", 1859, 0.0, 0.75, "Table 8.2 p. 39"),
    ("mean", "logelr", -0.3945, 0.15 * 0.0150, 0.0, "Table 8.1 p. 39"),
    ("sd", "logelr", 0.0150, 0.0, 0.2, "Table 8.1 p. 39"),
    ("mean", "rho", 0.1709, 0.15 * 0.2071, 0.0, "Table 8.1 p. 39"),
    ("sd", "rho", 0.2071, 0.0, 0.2, "Table 8.1 p. 39"),
    ("elpd_loo", "", 68.65, 2.5, 0.0, "Table 8.3 p. 41"),
]


def inits(rng):
    """Meyers' init.CAY, drawn per chain."""
    return [
        {
            "r_alpha": rng.normal(0, 0.2, 9).tolist(),
            "r_beta": rng.uniform(size=9).tolist(),
            "a_ig": rng.uniform(size=10).tolist(),
            "logelr": float(rng.uniform(-0.75, -0.25)),
            "r_rho": float(rng.uniform()),
        }
        for _ in range(CHAINS)
    ]


def main():
    toolchain()
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    rng = np.random.default_rng(SEED)
    stan = ROOT / "validation/scripts/stan/meyers_cay.stan"
    sha = hashlib.sha256(stan.read_bytes()).hexdigest()
    model = compile_model(stan)
    rows = []

    def row(method, quantity, arg, expected, abs_tol, rel_tol, source):
        rows.append(
            f"{COLUMN},{method},{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},{float(rel_tol)!r},{source}"
        )

    data, origins = training(COLUMN)
    # Meyers' (age, origin) order makes cell i - 1 the previous origin.
    data["wne1"] = [int(w > 1) for w in data["w"]]
    for i in range(1, data["len_data"]):
        if data["wne1"][i]:
            assert (data["w"][i - 1], data["d"][i - 1]) == (data["w"][i] - 1, data["d"][i])
    fit = model.sample(
        data=data,
        chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=DRAWS,
        seed=SEED,
        inits=inits(rng),
        metric="dense_e",
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        show_progress=False,
    )
    n = data["n"]
    params = (
        ["logelr"]
        + [f"alpha[{k}]" for k in range(2, n + 1)]
        + [f"beta[{k}]" for k in range(1, n)]
        + ["rho"]
        + [f"sigma[{k}]" for k in range(1, n + 1)]
    )
    summary = fit.summary()
    rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
    worst = summary.loc[params, rhat_col].max()
    assert worst <= 1.005, f"cay: R-hat {worst} above 1.005"
    divergences = int(fit.method_variables()["divergent__"].sum())
    assert divergences == 0, f"cay: {divergences} divergent transitions"
    loo = az.loo(az.from_cmdstanpy(fit, log_likelihood="log_lik"))
    print(
        f"cay {COLUMN}: max R-hat {worst:.5f}, divergences 0, elpd_loo {loo.elpd_loo:.3f}"
        f" (p_loo {loo.p_loo:.3f}, max k {float(np.max(loo.pareto_k)):.3f})"
    )
    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: meyers_cay.stan sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) metric dense_e adapt_delta {ADAPT_DELTA}"
    )
    method = "stan_cay"
    for p in params:
        name, _, index = p.partition("[")
        draws = fit.stan_variable(name)
        if index:
            draws = draws[:, int(index[:-1]) - 1]
        s = summarise(draws)
        row(method, "mean", p, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
        row(method, "sd", p, s["sd"], 0.0, SD_REL, source)
    targets = [(str(o), fit.stan_variable("reserve")[:, k]) for k, o in enumerate(origins)]
    targets.append(("total", fit.stan_variable("total_reserve")))
    for arg, draws in targets:
        s = summarise(draws)
        if s["sd"] == 0.0:
            # The oldest origin: observed at age n, an exact 0.
            for q in ["mean", "sd", "q05", "q50", "q95"]:
                row(method, f"reserve_{q}", arg, 0.0, EXACT, 0.0, source)
            continue
        row(method, "reserve_mean", arg, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
        if arg == "total":
            row(method, "reserve_sd", arg, s["sd"], 0.0, TOTAL_SD_REL, source)
        elif arg != str(origins[-1]):
            # The youngest origin's sd is heavy-tailed (the tolerances above).
            row(method, "reserve_sd", arg, s["sd"], 0.0, RESERVE_SD_REL, source)
        for q in ["q05", "q50", "q95"]:
            row(method, f"reserve_{q}", arg, s[q], QUANTILE_TOL * s["sd"], 0.0, source)
    # The joint prediction: the sum of every origin's reserve but the
    # youngest's (whose heavy tail dominates the total's sd), and the
    # correlation of each pair of neighbouring origins' reserves from the
    # second to the second-youngest. On these data most of that correlation
    # comes from the parameters (logelr and the alphas); CAY's recursion
    # down age n adds at most 0.03 (sigma[n] is near 0.013), under 2 of our
    # Monte Carlo errors, so the recursion itself is checked by a unit test
    # in meyers.rs, not here.
    reserves = fit.stan_variable("reserve")
    older = f"{origins[0]}-{origins[-2]}"
    s = summarise(reserves[:, :-1].sum(axis=1))
    row(method, "reserve_mean", older, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
    row(method, "reserve_sd", older, s["sd"], 0.0, SUM_SD_REL, source)
    for q in ["q05", "q50", "q95"]:
        row(method, f"reserve_{q}", older, s[q], QUANTILE_TOL * s["sd"], 0.0, source)
    for k in range(1, n - 2):
        corr = np.corrcoef(reserves[:, k], reserves[:, k + 1])[0, 1]
        row(method, "reserve_corr", f"{origins[k]}/{origins[k + 1]}", corr, CORR_TOL, 0.0, source)
    s = summarise(fit.stan_variable("total_ultimate"))
    row(method, "total_ultimate_mean", "", s["mean"], MEAN_TOL * s["sd"], 0.0, source)
    row(method, "total_ultimate_sd", "", s["sd"], 0.0, TOTAL_SD_REL, source)
    row(method, "elpd_loo", "", loo.elpd_loo, ELPD_TOL, 0.0, source)

    for quantity, arg, value, abs_tol, rel_tol, where in MEYERS:
        source = f"Meyers (2019) CAS Monograph 8 2nd ed. {where}"
        row("meyers_2019_cay", quantity, arg, value, abs_tol, rel_tol, source)

    header = [
        f"# Generated by validation/scripts/reserving_meyers_cay_stan.py with {cmdstan}, cmdstanpy {cmdstanpy.__version__},"
        f" pandas {pd.__version__}, numpy {np.__version__} and arviz {az.__version__} (elpd_loo).",
        f"# Model validation/scripts/stan/meyers_cay.stan (sha256 {sha}), a transcription of Meyers' CAY.R",
        "# (meyers-appendix.zip, CAS). Data validation/data/clrd_comauto_353.csv, column incurred,",
        f"# calendar years up to {LAST_YEAR} (55 cells). Seed {SEED}; {CHAINS} chains of {WARMUP} warm-up and {DRAWS} kept draws;",
        f"# metric dense_e; adapt_delta {ADAPT_DELTA}; max_treedepth {MAX_TREEDEPTH}. dataset is the column. With s the reference",
        "# posterior sd: means and reserve quantiles abs_tol a multiple of s, sds rel_tol, elpd_loo abs_tol, each at least 4",
        "# times the Monte Carlo error of Meyers' defaults (4 x 2,000 draws, 10,000 simulations) over seeds 0 to 199;",
        "# total sds 75% for their heavy tail; no reserve sd row for the youngest origin (its 5/50/95% rows instead);",
        f"# arg {origins[0]}-{origins[-2]} is the sum of those origins' reserves, a/b the correlation of two origins' reserves.",
        "# meyers_2019_cay rows: Meyers' printed numbers, at least 4 times the combined error of our defaults and his run.",
        "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
