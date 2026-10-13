"""Regenerate validation/reference/reserving_bayes_bf_stan.csv.

The reference for prospicio_reserving::bayes::BayesBf
(docs/design/reserving-bayes.md, decisions 9 and 10): the same model in Stan
(validation/scripts/stan/bayes_bf.stan), sampled long enough that the
reference's own Monte Carlo error is negligible next to the tolerances.
Run from the repository root, in PowerShell or cmd (CmdStan's make does not
detect the OS from Git Bash):

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 python validation/scripts/reserving_bayes_bf_stan.py

Needs CmdStan 2.39.0 and RTools 4.5 as reserving_bayes_stan.py does, whose
toolchain, plug-ins and summaries this script reuses. CI reads the committed
CSV and never runs Stan.

Cases (dataset, apriori, cv, development_sd), each with the exposure column
of its triangle:

* genins_premium (GenIns with R ChainLadder's ClarkCapeCod premium),
  apriori 0.5, cv 0.2, sd 10: a stated prior a little above the chain
  ladder's loss ratio (about 0.45);
* evw2012 (England, Verrall and Wuthrich 2012, Table 1; the prior column is
  their m_i), apriori 1, cv 0.1 (their a = 100), sd 10 (BayesBf's default);
* evw2012, cv 0.1, sd 100 and cv inf (their a = 0), sd 100: close to their
  flat prior on log gamma, for the comparison with their printed Tables 2 and
  3 (made in validation/tests/reserving_bayes_bf.rs). Not sd 1,000: with a
  flat prior on alpha too, the ridge (alpha + k, beta - k) that only the
  beta prior holds is then about 300 long, and BayesBf's sampler gives
  divergent transitions (200 or more in 4,000), where sd 100 gives none.

phi is the chain ladder's Pearson scale (reserving_bayes_stan.plug_ins), the
same as BayesBf's, and recorded so the Rust test checks that they agree.
Parameter rows are written only when cv is finite: under a flat prior on
alpha the sum alpha + beta is identified and alpha and beta apart are not
(their posterior is the prior's along that ridge).
"""

import hashlib
import math
import sys
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reserving_bayes_stan import CMDSTAN, plug_ins, summarise, toolchain  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STAN = ROOT / "validation/scripts/stan/bayes_bf.stan"
OUT = ROOT / "validation/reference/reserving_bayes_bf_stan.csv"
SEED = 20261012
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
ADAPT_DELTA = 0.99
EXACT = 1e-9

# (dataset, loss column, exposure column, apriori, cv, development_sd)
CASES = [
    ("genins_premium", "paid", "premium", 0.5, 0.2, 10.0),
    ("evw2012", "paid", "prior", 1.0, 0.1, 10.0),
    ("evw2012", "paid", "prior", 1.0, 0.1, 100.0),
    ("evw2012", "paid", "prior", 1.0, math.inf, 100.0),
]

# Tolerances, with s the reference posterior sd: (means abs x s, sds
# relative, reserve quantiles abs x s), each at least 4 times the Monte Carlo
# error of BayesBf's defaults (4 x 1,000 draws, 10,000 simulations) measured
# as the seed-to-seed sd of every row over seeds 0 to 199 (the ignored test
# seed_sweep in validation/tests/reserving_bayes_bf.rs): at most 0.021 s on a
# mean (beta[10] of evw2012), 2.75% on an sd (beta[10] again) and 0.063 s on
# a percentile (the 2002 reserve's 95%), the same on both datasets and all
# four cases. A first 10% on the sds was only 3.6 times the largest spread,
# so they take 15% (5.5 times). Seeds 0 to 199 pass every row, using at most
# 0.69 of a tolerance; seed 0 at most 0.38.
TOLERANCES = {
    "genins_premium": (0.1, 0.15, 0.3),
    "evw2012": (0.1, 0.15, 0.3),
}


def triangle(name, column, exposure):
    """Cumulative losses as an origins x ages array (NaN unobserved), the
    exposure per origin (its latest value), the origin labels."""
    df = pd.read_csv(ROOT / f"validation/data/{name}.csv", comment="#")
    origins = sorted(df["origin"].unique())
    ages = sorted(df["development"].unique())
    cum = np.full((len(origins), len(ages)), np.nan)
    expo = np.zeros(len(origins))
    latest_age = np.full(len(origins), -1)
    for row in df.itertuples():
        w, d = origins.index(row.origin), ages.index(row.development)
        cum[w, d] = getattr(row, column)
        if d > latest_age[w]:
            latest_age[w], expo[w] = d, getattr(row, exposure)
    return cum, expo, origins


def params(apriori, cv, sd):
    return f"apriori={apriori:g};cv={cv:g};development_sd={sd:g}"


def main():
    toolchain()
    sha = hashlib.sha256(STAN.read_bytes()).hexdigest()
    build = ROOT / "target/stan"
    build.mkdir(parents=True, exist_ok=True)
    copy = build / STAN.name
    if not copy.exists() or copy.read_bytes() != STAN.read_bytes():
        copy.write_bytes(STAN.read_bytes())
    model = cmdstanpy.CmdStanModel(stan_file=str(copy))
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: bayes_bf.stan sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) metric dense_e adapt_delta {ADAPT_DELTA}"
    )
    rows = []
    for name, column, exposure, apriori, cv, sd in CASES:
        cum, expo, origins = triangle(name, column, exposure)
        cells, phi, _, future = plug_ins(cum)
        no, nd = cum.shape
        m = apriori * expo
        shape = 0.0 if math.isinf(cv) else cv**-2
        data = {
            "N": len(cells),
            "n_origin": no,
            "n_dev": nd,
            "w": [c[0] + 1 for c in cells],
            "d": [c[1] + 1 for c in cells],
            "x": [c[2] for c in cells],
            "phi": phi,
            "m": list(m),
            "shape": shape,
            "development_sd": sd,
            "N_fut": len(future),
            "fw": [f[0] + 1 for f in future],
            "fd": [f[1] + 1 for f in future],
            "gamma_process": 1,
        }
        # Start at the strong-prior mode (BayesBf's Newton start): mu = m and
        # the raw pattern, which keeps the flat-prior ridge's warm-up short.
        s_d = np.zeros(nd)
        m_d = np.zeros(nd)
        for w, d, x in cells:
            s_d[d] += x
            m_d[d] += m[w]
        inits = {"alpha": list(np.log(m)), "beta": list(np.log(s_d / m_d))}
        fit = model.sample(
            data=data,
            chains=CHAINS,
            iter_warmup=WARMUP,
            iter_sampling=DRAWS,
            seed=SEED,
            metric="dense_e",
            adapt_delta=ADAPT_DELTA,
            inits=inits,
            show_progress=False,
        )
        summary = fit.summary()
        names = [f"alpha[{k}]" for k in range(1, no + 1)] + [f"beta[{k}]" for k in range(1, nd + 1)]
        rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
        checked = names if shape > 0 else ["reserve[%d]" % k for k in range(2, no + 1)] + ["total"]
        worst = summary.loc[checked, rhat_col].max()
        assert worst <= 1.005, f"{name} cv {cv}: R-hat {worst} above 1.005"
        divergences = int(fit.method_variables()["divergent__"].sum())
        assert divergences == 0, f"{name} cv {cv}: {divergences} divergent transitions"
        print(f"{name} cv {cv} sd {sd}: phi {phi:.6f} max R-hat {worst:.5f} divergences 0")

        p = params(apriori, cv, sd)

        def row(quantity, arg, expected, abs_tol, rel_tol):
            rows.append(
                f"{name},bayes_bf,{p},{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},{float(rel_tol)!r},{source}"
            )

        mean_tol, sd_rel, quantile_tol = TOLERANCES[name]
        row("scale", "", phi, 0.0, EXACT)
        if shape > 0:
            for k, n in enumerate(names):
                draws = fit.stan_variable("alpha" if k < no else "beta")[:, k if k < no else k - no]
                s = summarise(draws)
                row("mean", n, s["mean"], mean_tol * s["sd"], 0.0)
                row("sd", n, s["sd"], 0.0, sd_rel)
        reserve = fit.stan_variable("reserve")
        targets = [(str(o), reserve[:, k]) for k, o in enumerate(origins)]
        targets.append(("total", fit.stan_variable("total")))
        for arg, draws in targets:
            s = summarise(draws)
            if s["sd"] == 0.0:
                for q in ["mean", "sd", "q05", "q50", "q95"]:
                    row(f"reserve_{q}", arg, 0.0, EXACT, 0.0)
                continue
            row("reserve_mean", arg, s["mean"], mean_tol * s["sd"], 0.0)
            row("reserve_sd", arg, s["sd"], 0.0, sd_rel)
            for q in ["q05", "q50", "q95"]:
                row(f"reserve_{q}", arg, s[q], quantile_tol * s["sd"], 0.0)

    header = [
        f"# Generated by validation/scripts/reserving_bayes_bf_stan.py with {cmdstan}, cmdstanpy {cmdstanpy.__version__},"
        f" pandas {pd.__version__} and numpy {np.__version__}.",
        f"# Model validation/scripts/stan/bayes_bf.stan (sha256 {sha}).",
        f"# Seed {SEED}; {CHAINS} chains of {WARMUP} warm-up and {DRAWS} kept draws; metric dense_e; adapt_delta {ADAPT_DELTA}; Gamma process.",
        "# phi is the chain ladder's (rel_tol 1e-9). With s the reference posterior sd: means abs_tol 0.1 s, sds rel_tol 0.15,",
        "# reserve quantiles abs_tol 0.3 s (TOLERANCES in the script): each at least 4.7 times the Monte Carlo error of",
        "# BayesBf's defaults over seeds 0 to 199. The oldest origin's reserve is an exact 0. arg: parameter, origin label or total.",
        "dataset,method,params,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
