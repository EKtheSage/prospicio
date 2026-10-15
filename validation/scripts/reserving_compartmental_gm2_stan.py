"""Regenerate validation/reference/reserving_compartmental_gm2_stan.csv.

The reference for prospicio_reserving::bayes::Compartmental with Gesmann and
Morris' model 2 (docs/design/reserving-compartmental.md, decisions 2 and 7):
the preset CompartmentalPriors::gesmann_morris_2(), the log-normal
likelihood on loss ratios and incremental paid, with random effects by
accident year and by development year on all four parameters, reconstructed
as the Stan program brms generates from the paper's code in
validation/scripts/stan/compartmental_gm2.stan (provenance in its header),
on the same data and training triangle as model 1's reference
(reserving_compartmental_stan.py): company 337, calendar years up to 1996,
outstanding = incurred - paid (with bulk and IBNR) and direct premium, 90
observations over 9 accident years and 9 development years. Run from the
repository root, in PowerShell or cmd:

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 --with arviz==0.23.4 python validation/scripts/reserving_compartmental_gm2_stan.py

with the toolchain of reserving_compartmental_stan.py
(knowledge/environment/local-cmdstan.md). CI reads the committed CSV and
never runs Stan.

Besides the Stan summaries, the CSV holds the paper's printed posterior of
model 2 (method gesmann_morris_2020_m2), a looser second check.

CmdStan's output goes to target/stan/compartmental_gm2_output/; with --reuse
the script rereads the draws from there (same program, data and seed).
"""

import hashlib
import json
import sys

import arviz as az
import cmdstanpy
import numpy as np
import pandas as pd

from reserving_compartmental_stan import DATASET, LAST_TRAIN_YEAR, cells
from reserving_meyers_stan import CMDSTAN, ROOT, compile_model, summarise, toolchain

OUT = ROOT / "validation/reference/reserving_compartmental_gm2_stan.csv"
STAN = ROOT / "validation/scripts/stan/compartmental_gm2.stan"
SEED = 20261015
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
# The paper's 0.99 left 5 divergent transitions in 100,000 kept draws (seed
# 20261015, max R-hat 1.0004, every mean within 0.08 printed sd of m2fit);
# decision 7 then tightens the step, so the reference runs at 0.999.
ADAPT_DELTA = 0.999
MAX_TREEDEPTH = 15

# Tolerances, with s the reference posterior sd (decision 7), each at least
# 4 times the largest seed-to-seed sd of our defaults in its family (the
# ignored test seed_sweep in validation/tests/reserving_compartmental_gm2.rs;
# Stan's own error is negligible). The measured spreads are in the design
# note and knowledge/findings/compartmental-gm2-parity.md.
#
# Spreads over seeds 0 to 199 (Compartmental::model_2's sampler: 4 x 8,000
# draws after 2,000 warm-up, target acceptance 0.999; 10,000 simulations):
# parameter means 0.036 s (tau_dev_kp; 0.017 s for the parameters not of
# k_p), parameter sds 1.4% (k_p, b_kp and tau_kp about 1% over seeds 0 to 39),
# ulr means 0.011 s, reserve means 0.017 s, reserve sds 1.6%, reserve
# quantiles 0.037 s (95% of 1997), test means 0.014 s, elpd_loo 0.45. The
# sds of k_p's development effects and origin values (tau_dev_kp,
# dev_kp[a], k_p[o]) are heavy-tailed from seed to seed: 4% to 14% over
# seeds 0 to 39 (with 4 x 4,000 draws, up to 18% over 200 seeds), from the
# long tail of tau_dev_kp's posterior; those 19 rows are dropped (decision
# 7), and their means kept. Every tolerance is at least 4.4 times its
# family's spread, and seeds 0 to 199 pass every row (at most 0.86 of a
# tolerance: 0.172 s on tau_dev_kp's mean; 1.41 on elpd_loo).
MEAN_TOL = 0.2  # x s: parameter means (5.6 times)
OUTPUT_MEAN_TOL = 0.1  # x s: ulr, reserve and test means (5.8 times)
SD_REL = 0.1  # parameter sds (7.1 times)
RESERVE_SD_REL = 0.1  # reserve sds (6.2 times)
QUANTILE_TOL = 0.2  # x s: reserve quantiles (5.4 times)
ELPD_TOL = 2.0  # 4.4 times


def heavy_tailed(name):
    """The parameters whose posterior sd is heavy-tailed from seed to seed."""
    return name == "tau_dev_kp" or name.startswith("dev_kp[") or name.startswith("k_p[")


# Rows dropped as heavy-tailed (decision 7): (quantity, predicate on arg).
DROPPED = [("sd", heavy_tailed)]

# Gesmann and Morris (2020), sec. 7.2.3, the printed summary of m2fit (brms
# 2.10.0 on rstan 2.19.2, 4 chains of 1,000 kept draws; bulk ESS 470 to
# 4,150): (parameter, mean, sd).
PRINTED = [
    ("b_RLR", 1.54, 0.44),
    ("b_RRF", -1.45, 0.69),
    ("b_ker", -1.31, 1.08),
    ("b_kp", -5.07, 1.75),
    ("tau_RLR", 0.69, 0.31),
    ("tau_RRF", 0.73, 0.47),
    ("tau_ker", 0.26, 0.22),
    ("tau_kp", 2.23, 1.35),
    ("rho", 0.37, 0.47),
    ("tau_dev_RLR", 0.35, 0.26),
    ("tau_dev_RRF", 1.11, 0.49),
    ("tau_dev_ker", 0.27, 0.23),
    ("tau_dev_kp", 0.45, 0.43),
    ("log_sigma_os", -1.78, 0.16),
    ("log_sigma_pd", -1.89, 0.16),
]
# Their error combines ours (at most 0.036 s on a mean, 1.4% on an sd), their
# run's (about 1 / sqrt(ESS): up to 0.046 sd on a mean at a bulk ESS of 470,
# and about 3.3% on an sd) and the rounding to two decimals (up to 0.031 sd
# on a mean and 3.1% on an sd, log sigma): at most about 0.062 printed sd on
# a mean (tau_dev_kp) and 4.7% on an sd. Means get 0.3 printed sd (4.9
# times), sds 20% (4.3 times). tau_dev_kp's sd is dropped, as for the Stan
# rows: heavy-tailed on our side, and the printed 0.43 is itself 17% below
# the reference's 0.518.
PRINTED_MEAN_TOL = 0.3  # x printed sd
PRINTED_SD_REL = 0.2


def stan_data(df):
    """The training cells (calendar years up to LAST_TRAIN_YEAR) as loss
    ratios to direct premium, outstanding first, then incremental paid, each
    by origin and age; and the test diagonal."""
    df = df.sort_values(["origin", "age"]).copy()
    df["incr"] = df.groupby("origin")["paid"].diff().fillna(df["paid"])
    origins = sorted(df["origin"].unique())
    n = len(origins)
    premium = df.groupby("origin")["premium"].first().loc[origins].to_numpy(float)
    cal = df["origin"] + df["age"] - 1
    train = df[cal <= LAST_TRAIN_YEAR]
    test = df[cal == LAST_TRAIN_YEAR + 1]
    with_data = sorted(train["origin"].unique())
    ages_with_data = sorted(train["age"].unique())
    level = [with_data.index(o) + 1 if o in with_data else 0 for o in origins]
    dev_level = [ages_with_data.index(a) + 1 if a in ages_with_data else 0 for a in range(1, n + 1)]
    latest = train.groupby("origin")["age"].max()
    latest_age = [int(latest.get(o, 0)) for o in origins]
    y, t, delta, origin, age = [], [], [], [], []
    for d, col in [(0, "os"), (1, "incr")]:
        y += (train[col] / train["premium"]).astype(float).tolist()
        t += train["age"].astype(float).tolist()
        delta += [d] * len(train)
        origin += [origins.index(o) + 1 for o in train["origin"]]
        age += train["age"].astype(int).tolist()
    assert min(y) > 0
    test_origin, test_age, test_delta, test_args = [], [], [], []
    for d, name in [(0, "os"), (1, "paid")]:
        for o, a in zip(test["origin"], test["age"]):
            test_origin.append(origins.index(o) + 1)
            test_age.append(int(a))
            test_delta.append(d)
            test_args.append(f"{name}_{o}")
    data = {
        "N": len(y),
        "Y": y,
        "t": t,
        "delta": delta,
        "n_origin": n,
        "n_age": n,
        "premium": premium.tolist(),
        "N_1": len(with_data),
        "level": level,
        "N_2": len(ages_with_data),
        "dev_level": dev_level,
        "origin": origin,
        "age": age,
        "latest_age": latest_age,
        "N_test": len(test_origin),
        "test_origin": test_origin,
        "test_age": test_age,
        "test_delta": test_delta,
    }
    assert data["N"] == 90 and data["N_1"] == 9 and data["N_2"] == 9 and n == 10
    assert data["N_test"] == 20
    return data, origins, with_data, ages_with_data, test_args


def main():
    toolchain()
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    sha = hashlib.sha256(STAN.read_bytes()).hexdigest()
    data, origins, with_data, ages, test_args = stan_data(cells())
    output = ROOT / "target/stan/compartmental_gm2_output"
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
    v = fit.stan_variable
    draws = {
        "b_ker": v("b_oker"),
        "b_kp": v("b_okp"),
        "b_RLR": v("b_oRLR"),
        "b_RRF": v("b_oRRF"),
        "tau_RLR": v("sd_1")[:, 0],
        "tau_RRF": v("sd_1")[:, 1],
        "tau_ker": v("sd_4"),
        "tau_kp": v("sd_6"),
        "rho": v("rho"),
        "tau_dev_ker": v("sd_5"),
        "tau_dev_kp": v("sd_7"),
        "tau_dev_RLR": v("sd_2"),
        "tau_dev_RRF": v("sd_3"),
        "log_sigma_os": v("b_sigma")[:, 0],
        "log_sigma_pd": v("b_sigma")[:, 1],
        "k_er": v("k_er"),
        "k_p": v("k_p"),
        "RLR": v("RLR_pop"),
        "RRF": v("RRF_pop"),
        "sigma_os": v("sigma_os"),
        "sigma_pd": v("sigma_pd"),
    }
    for name, stan_name in [("RLR", "RLR"), ("RRF", "RRF"), ("k_er", "ker_o"), ("k_p", "kp_o")]:
        values = v(stan_name)
        for o in with_data:
            draws[f"{name}[{o}]"] = values[:, origins.index(o)]
    for name, r in [("dev_RLR", "r_2"), ("dev_RRF", "r_3"), ("dev_ker", "r_5"), ("dev_kp", "r_7")]:
        values = v(r)
        for k, a in enumerate(ages):
            draws[f"{name}[{12 * a}]"] = values[:, k]
    summary = fit.summary()
    rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
    sampled = ["b_oRLR", "b_oRRF", "b_oker", "b_okp", "sd_1[1]", "sd_1[2]", "rho",
               "b_sigma[1]", "b_sigma[2]"] + [f"sd_{k}" for k in range(2, 8)]
    sampled += [f"r_1[{j},{k}]" for j in range(1, 10) for k in (1, 2)]
    sampled += [f"r_{g}[{j}]" for g in range(2, 8) for j in range(1, 10)]
    worst = summary.loc[sampled, rhat_col].max()
    method_vars = fit.method_variables()
    divergences = int(method_vars["divergent__"].sum())
    depth = int((method_vars["treedepth__"] >= MAX_TREEDEPTH).sum())
    swapped = int((draws["k_er"] <= draws["k_p"]).sum())
    loo = az.loo(az.from_cmdstanpy(fit, log_likelihood="log_lik"))
    print(
        f"max R-hat {worst:.5f}, {divergences} divergences, {depth} draws at the maximum depth,"
        f" {swapped} draws with population k_er <= k_p, elpd_loo {loo.elpd_loo:.3f}"
        f" (p_loo {loo.p_loo:.3f}, max k {float(np.max(loo.pareto_k)):.3f})"
    )
    for name, mean, sd in PRINTED:
        s = summarise(draws[name])
        print(f"{name}: Stan {s['mean']:.3f} ({s['sd']:.3f}), printed {mean} ({sd})")
    assert worst <= 1.005, f"R-hat {worst} above 1.005"
    assert divergences == 0, f"{divergences} divergent transitions"

    rows = []
    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: compartmental_gm2.stan sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) adapt_delta {ADAPT_DELTA}"
        f" max_treedepth {MAX_TREEDEPTH}"
    )

    def row(method, quantity, arg, expected, abs_tol, rel_tol, src):
        if method == "stan_gm2" and any(q == quantity and p(arg) for q, p in DROPPED):
            return
        rows.append(
            f"{DATASET},{method},{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},{float(rel_tol)!r},{src}"
        )

    method = "stan_gm2"
    for name, values in draws.items():
        s = summarise(values)
        row(method, "mean", name, s["mean"], MEAN_TOL * s["sd"], 0.0, source)
        row(method, "sd", name, s["sd"], 0.0, SD_REL, source)
    ulr = v("ulr")
    for k, o in enumerate(origins):
        s = summarise(ulr[:, k])
        row(method, "ulr_mean", str(o), s["mean"], OUTPUT_MEAN_TOL * s["sd"], 0.0, source)
    reserve = v("reserve")
    targets = [(str(o), reserve[:, k]) for k, o in enumerate(origins)]
    targets.append(("total", v("total_reserve")))
    for arg, values in targets:
        s = summarise(values)
        row(method, "reserve_mean", arg, s["mean"], OUTPUT_MEAN_TOL * s["sd"], 0.0, source)
        row(method, "reserve_sd", arg, s["sd"], 0.0, RESERVE_SD_REL, source)
        for q in ["q05", "q50", "q95"]:
            row(method, f"reserve_{q}", arg, s[q], QUANTILE_TOL * s["sd"], 0.0, source)
    test_mean = v("test_mean")
    for k, arg in enumerate(test_args):
        s = summarise(test_mean[:, k])
        row(method, "test_mean", arg, s["mean"], OUTPUT_MEAN_TOL * s["sd"], 0.0, source)
    row(method, "elpd_loo", "", loo.elpd_loo, ELPD_TOL, 0.0, source)

    printed = "Gesmann and Morris (2020) CAS Research Paper sec. 7.2.3 m2fit (brms 2.10.0 rstan 2.19.2)"
    for name, mean, sd in PRINTED:
        row("gesmann_morris_2020_m2", "mean", name, mean, PRINTED_MEAN_TOL * sd, 0.0, printed)
        if not heavy_tailed(name):
            row("gesmann_morris_2020_m2", "sd", name, sd, 0.0, PRINTED_SD_REL, printed)

    header = [
        f"# Generated by validation/scripts/reserving_compartmental_gm2_stan.py with {cmdstan}, cmdstanpy {cmdstanpy.__version__},"
        f" pandas {pd.__version__}, numpy {np.__version__} and arviz {az.__version__} (elpd_loo).",
        f"# Model validation/scripts/stan/compartmental_gm2.stan (sha256 {sha}), Gesmann and Morris' (2020) model 2",
        "# reconstructed as the Stan program brms generates from the paper's code (sec. 7.2.3). Data",
        f"# validation/data/clrd_wkcomp_337.csv, company 337, calendar years up to {LAST_TRAIN_YEAR} (90 observations:",
        "# outstanding and incremental paid loss ratios to direct premium); the 1997 diagonal is the test set. Seed"
        f" {SEED}; {CHAINS} chains of {WARMUP} warm-up",
        f"# and {DRAWS} kept draws; adapt_delta {ADAPT_DELTA}; max_treedepth {MAX_TREEDEPTH}. With s the reference posterior sd:",
        "# means, test means and reserve quantiles abs_tol a multiple of s, sds rel_tol, elpd_loo abs_tol, each at least 4",
        "# times the seed-to-seed sd of our defaults over seeds 0 to 199 (every seed passes); the sds of tau_dev_kp, dev_kp[a] and",
        "# k_p[o] are dropped as heavy-tailed from seed to seed. Reserves are the incremental paid to age 10 summed;",
        "# 1997 and development year 10 have no training cell and take new effects from the population in each draw.",
        "# Test means are the expected outstanding and incremental paid amounts of the 1997 diagonal.",
        "# gesmann_morris_2020_m2 rows: the paper's printed posterior, looser (0.3 printed sd, 20%).",
        "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
