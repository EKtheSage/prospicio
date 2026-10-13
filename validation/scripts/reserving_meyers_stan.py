"""Regenerate validation/reference/reserving_meyers_stan.csv.

The reference for prospicio_reserving::bayes::Meyers (CRC and CSR,
docs/design/reserving-bayes.md, decisions 7 and 10): Meyers' own Stan
programs, transcribed into current syntax in validation/scripts/stan/
meyers_crc.stan and meyers_csr.stan (provenance in their headers), sampled
long enough that the reference's own Monte Carlo error is negligible next to
the tolerances, on Meyers' illustrative insurer (group 353 Commercial Auto,
validation/data/clrd_comauto_353.csv, the cells at calendar years up to
1997). Run from the repository root, in PowerShell or cmd (CmdStan's make
does not detect the OS from Git Bash):

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 --with arviz==0.23.4 python validation/scripts/reserving_meyers_stan.py

Needs CmdStan 2.39.0 built at ~/.cmdstan/cmdstan-2.39.0 and RTools 4.5 at
C:/rtools45 (knowledge/environment/local-cmdstan.md). CI reads the committed
CSV and never runs Stan.

Besides the Stan summaries, the CSV holds Meyers' printed numbers for the same
data (method meyers_2019_crc and meyers_2019_csr), a looser second check.
"""

import hashlib
import os
from pathlib import Path

import arviz as az
import cmdstanpy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation/data/clrd_comauto_353.csv"
OUT = ROOT / "validation/reference/reserving_meyers_stan.csv"
CMDSTAN = Path.home() / ".cmdstan/cmdstan-2.39.0"
RTOOLS = Path("C:/rtools45")
CASES = [("crc", "paid"), ("crc", "incurred"), ("csr", "paid"), ("csr", "incurred")]
SEED = 20261012
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 12
LAST_YEAR = 1997

# Tolerances, the same for each model and column, with s the reference
# posterior sd (decision 10). Each is at least 4 times the largest Monte
# Carlo error of Meyers' defaults (4 x 2,000 draws, 10,000 simulations) in
# its family, measured as the seed-to-seed sd of every row over seeds 0 to
# 199 (Stan's own error is negligible): means 0.026 s (CSR incurred
# sigma[8]), parameter sds 3.6% (CRC incurred logelr), the reserve sds of
# origins 1989 to 1996 2.6%, the total reserve's and total ultimate's sds
# 4.3% (CRC incurred), quantiles 0.063 s (95% of CSR incurred 1997),
# elpd_loo 0.43 (CSR incurred; PSIS with Pareto k up to 1). Total sds get
# 50%, not 20%: their error is heavy-tailed (CRC incurred's was +37% at seed
# 170). The youngest origin (1997) has no reserve sd row: its reserve rides
# on the one-cell alpha[10] through a log-normal, so its sd's error is
# heavy-tailed (seed-to-seed sd up to 7.5%, but +63% at seed 170 and outside
# 20% at 12 of 200 seeds); its 5%, 50% and 95% quantiles check its spread
# instead. Seeds 0 to 199 pass every row.
MEAN_TOL = 0.15  # x s
SD_REL = 0.15  # parameter sds
RESERVE_SD_REL = 0.2  # an origin's reserve sd
TOTAL_SD_REL = 0.5  # the total reserve's and total ultimate's sds
QUANTILE_TOL = 0.35  # x s
ELPD_TOL = 2.0
EXACT = 1e-9

# Meyers (2019), CAS Monograph 8, 2nd ed., on the same data, 4 chains of 2,500
# kept draws (CRC.R and CSR.R: iter 5000, half warm-up). "Estimate" is the mean
# total ultimate (the oldest origin's observed value plus the others' draws).
# (method, column, quantity, arg, value, abs_tol, rel_tol, table and page). Each
# tolerance is at least 4 times the combined Monte Carlo error of our defaults
# (the seed sweep) and of Meyers' single run (his error taken as ours for the
# totals, which use as many simulations, and as 1 / sqrt(3,000 effective
# draws) of an sd for logelr and gamma): the total ultimate's mean within 1%
# (combined error 0.13%); the means of logelr and gamma within 0.15 printed sd (combined 0.027 sd); their
# sds within 15% (combined 3.4%); elpd_loo within 2 (each run's about 0.36).
# The total ultimate's sd within 50%, as for the Stan rows: our error is
# heavy-tailed (seed-to-seed sd up to 4.3%, but 35% off at seed 170).
MEYERS = [
    ("meyers_2019_crc", "paid", "total_ultimate_mean", "", 40121, 0.0, 0.01, "Table 5.2 p. 18"),
    ("meyers_2019_crc", "paid", "total_ultimate_sd", "", 2487, 0.0, 0.5, "Table 5.2 p. 18"),
    ("meyers_2019_crc", "paid", "mean", "logelr", -0.3965, 0.15 * 0.0233, 0.0, "Table 5.1 p. 17"),
    ("meyers_2019_crc", "paid", "sd", "logelr", 0.0233, 0.0, 0.15, "Table 5.1 p. 17"),
    ("meyers_2019_crc", "paid", "elpd_loo", "", 47.80, 2.0, 0.0, "Tables 6.4 p. 29 and 7.3 p. 34"),
    ("meyers_2019_crc", "incurred", "total_ultimate_mean", "", 39147, 0.0, 0.01, "Table 5.3 p. 18"),
    ("meyers_2019_crc", "incurred", "total_ultimate_sd", "", 1642, 0.0, 0.5, "Table 5.3 p. 18"),
    ("meyers_2019_crc", "incurred", "elpd_loo", "", 70.97, 2.0, 0.0, "Table 6.4 p. 29"),
    ("meyers_2019_csr", "paid", "total_ultimate_mean", "", 37597, 0.0, 0.01, "Table 7.2 p. 33"),
    ("meyers_2019_csr", "paid", "total_ultimate_sd", "", 2401, 0.0, 0.5, "Table 7.2 p. 33"),
    ("meyers_2019_csr", "paid", "mean", "logelr", -0.3956, 0.15 * 0.0246, 0.0, "Table 7.1 p. 32"),
    ("meyers_2019_csr", "paid", "sd", "logelr", 0.0246, 0.0, 0.15, "Table 7.1 p. 32"),
    ("meyers_2019_csr", "paid", "mean", "gamma", 0.0446, 0.15 * 0.0282, 0.0, "Table 7.1 p. 32"),
    ("meyers_2019_csr", "paid", "sd", "gamma", 0.0282, 0.0, 0.15, "Table 7.1 p. 32"),
    ("meyers_2019_csr", "paid", "elpd_loo", "", 49.76, 2.0, 0.0, "Table 7.3 p. 34"),
]


def toolchain():
    """RTools' make and g++ and CmdStan's TBB on PATH; CmdStan located."""
    extra = [
        RTOOLS / "usr/bin",
        RTOOLS / "x86_64-w64-mingw32.static.posix/bin",
        CMDSTAN / "stan/lib/stan_math/lib/tbb",
    ]
    os.environ["PATH"] = os.pathsep.join([str(p) for p in extra] + [os.environ["PATH"]])
    # cmdstanpy calls mingw32-make on Windows; RTools 4.5 names it make.
    os.environ.setdefault("MAKE", str(RTOOLS / "usr/bin/make.exe"))
    cmdstanpy.set_cmdstan_path(str(CMDSTAN))


def compile_model(stan):
    """Compiles a copy under target/, so the executable stays out of the tree."""
    build = ROOT / "target/stan"
    build.mkdir(parents=True, exist_ok=True)
    copy = build / stan.name
    if not copy.exists() or copy.read_bytes() != stan.read_bytes():
        copy.write_bytes(stan.read_bytes())
    return cmdstanpy.CmdStanModel(stan_file=str(copy))


def training(column):
    """Stan data for `column`: the cells at calendar years up to LAST_YEAR,
    sorted by (age, origin) as Meyers sorts them, and per origin the
    premium, the latest value and whether age n is observed."""
    df = pd.read_csv(DATA, comment="#")
    df["age"] = df["development"] // 12
    df = df[df["origin"] + df["age"] - 1 <= LAST_YEAR].sort_values(["age", "origin"])
    origins = sorted(df["origin"].unique())
    n = len(origins)
    w = [origins.index(o) + 1 for o in df["origin"]]
    premium = df.groupby("origin")["premium"].first().loc[origins].to_numpy(float)
    latest_rows = df.sort_values("age").groupby("origin").last().loc[origins]
    data = {
        "n": n,
        "len_data": len(df),
        "logprem": np.log(df["premium"].to_numpy(float)).tolist(),
        "logloss": np.log(df[column].to_numpy(float)).tolist(),
        "w": w,
        "d": df["age"].tolist(),
        "premium": premium.tolist(),
        "latest": latest_rows[column].to_numpy(float).tolist(),
        "closed": [int(a == n) for a in latest_rows["age"]],
    }
    assert data["len_data"] == 55 and n == 10, (data["len_data"], n)
    return data, origins


def inits(model, rng):
    """Meyers' init.CRC and init.CSR, drawn per chain."""
    out = []
    for _ in range(CHAINS):
        init = {
            "r_alpha": rng.normal(0, 0.2, 9).tolist(),
            "r_beta": rng.uniform(size=9).tolist(),
            "a_ig": rng.uniform(size=10).tolist(),
            "logelr": float(rng.uniform(-0.75, -0.5)),
        }
        if model == "csr":
            init["gamma"] = float(rng.normal(0, 0.1))
        out.append(init)
    return out


def summarise(values):
    v = np.asarray(values)
    return {
        "mean": v.mean(),
        "sd": v.std(ddof=1),
        "q05": np.quantile(v, 0.05),
        "q50": np.quantile(v, 0.5),
        "q95": np.quantile(v, 0.95),
    }


def main():
    toolchain()
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    rng = np.random.default_rng(SEED)
    models, shas = {}, {}
    for model in ["crc", "csr"]:
        stan = ROOT / f"validation/scripts/stan/meyers_{model}.stan"
        shas[model] = hashlib.sha256(stan.read_bytes()).hexdigest()
        models[model] = compile_model(stan)
    rows = []

    def row(column, method, quantity, arg, expected, abs_tol, rel_tol, source):
        rows.append(
            f"{column},{method},{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},{float(rel_tol)!r},{source}"
        )

    for model, column in CASES:
        data, origins = training(column)
        fit = models[model].sample(
            data=data,
            chains=CHAINS,
            iter_warmup=WARMUP,
            iter_sampling=DRAWS,
            seed=SEED,
            inits=inits(model, rng),
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
            + (["gamma"] if model == "csr" else [])
            + [f"sigma[{k}]" for k in range(1, n + 1)]
        )
        summary = fit.summary()
        rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
        worst = summary.loc[params, rhat_col].max()
        assert worst <= 1.005, f"{model} {column}: R-hat {worst} above 1.005"
        divergences = int(fit.method_variables()["divergent__"].sum())
        assert divergences == 0, f"{model} {column}: {divergences} divergent transitions"
        loo = az.loo(az.from_cmdstanpy(fit, log_likelihood="log_lik"))
        print(
            f"{model} {column}: max R-hat {worst:.5f}, divergences 0, elpd_loo {loo.elpd_loo:.3f}"
            f" (p_loo {loo.p_loo:.3f}, max k {float(np.max(loo.pareto_k)):.3f})"
        )
        source = (
            f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: meyers_{model}.stan sha256 {shas[model][:16]}"
            f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) metric dense_e adapt_delta {ADAPT_DELTA}"
        )
        method = f"stan_{model}"
        for p in params:
            name, _, index = p.partition("[")
            draws = fit.stan_variable(name)
            if index:
                draws = draws[:, int(index[:-1]) - 1]
            s = summarise(draws)
            row(column, method, "mean", p, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
            row(column, method, "sd", p, s["sd"], 0.0, SD_REL, source)
        targets = [(str(o), fit.stan_variable("reserve")[:, k]) for k, o in enumerate(origins)]
        targets.append(("total", fit.stan_variable("total_reserve")))
        for arg, draws in targets:
            s = summarise(draws)
            if s["sd"] == 0.0:
                # The oldest origin: observed at age n, an exact 0.
                for q in ["mean", "sd", "q05", "q50", "q95"]:
                    row(column, method, f"reserve_{q}", arg, 0.0, EXACT, 0.0, source)
                continue
            row(column, method, "reserve_mean", arg, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
            if arg == "total":
                row(column, method, "reserve_sd", arg, s["sd"], 0.0, TOTAL_SD_REL, source)
            elif arg != str(origins[-1]):
                # The youngest origin's sd is heavy-tailed (the tolerances above).
                row(column, method, "reserve_sd", arg, s["sd"], 0.0, RESERVE_SD_REL, source)
            for q in ["q05", "q50", "q95"]:
                row(column, method, f"reserve_{q}", arg, s[q], QUANTILE_TOL * s["sd"], 0.0, source)
        s = summarise(fit.stan_variable("total_ultimate"))
        row(column, method, "total_ultimate_mean", "", s["mean"], MEAN_TOL * s["sd"], 0.0, source)
        row(column, method, "total_ultimate_sd", "", s["sd"], 0.0, TOTAL_SD_REL, source)
        row(column, method, "elpd_loo", "", loo.elpd_loo, ELPD_TOL, 0.0, source)

    for method, column, quantity, arg, value, abs_tol, rel_tol, where in MEYERS:
        source = f"Meyers (2019) CAS Monograph 8 2nd ed. {where}"
        row(column, method, quantity, arg, value, abs_tol, rel_tol, source)

    header = [
        f"# Generated by validation/scripts/reserving_meyers_stan.py with {cmdstan}, cmdstanpy {cmdstanpy.__version__},"
        f" pandas {pd.__version__}, numpy {np.__version__} and arviz {az.__version__} (elpd_loo).",
        f"# Models validation/scripts/stan/meyers_crc.stan (sha256 {shas['crc']}) and meyers_csr.stan (sha256 {shas['csr']}),",
        "# transcriptions of Meyers' CRC.R and CSR.R (meyers-appendix.zip, CAS). Data validation/data/clrd_comauto_353.csv,",
        f"# calendar years up to {LAST_YEAR} (55 cells). Seed {SEED}; {CHAINS} chains of {WARMUP} warm-up and {DRAWS} kept draws;",
        f"# metric dense_e; adapt_delta {ADAPT_DELTA}; max_treedepth {MAX_TREEDEPTH}. dataset is the column. With s the reference",
        "# posterior sd: means and reserve quantiles abs_tol a multiple of s, sds rel_tol, elpd_loo abs_tol, each at least 4",
        "# times the Monte Carlo error of Meyers' defaults (4 x 2,000 draws, 10,000 simulations) over seeds 0 to 199;",
        "# total sds 50% for their heavy tail; no reserve sd row for the youngest origin (its 5/50/95% rows instead).",
        "# meyers_2019_* rows: Meyers' printed numbers, at least 4 times the combined error of our defaults and his run.",
        "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
