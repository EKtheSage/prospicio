"""Regenerate validation/reference/reserving_bayes_stan.csv.

The reference for prospicio_reserving::bayes::BayesOdp
(docs/design/reserving-bayes.md, decision 10): the same model in Stan
(validation/scripts/stan/bayes_odp.stan), sampled long enough that the
reference's own Monte Carlo error is negligible next to the tolerances.
Run from the repository root, in PowerShell or cmd (CmdStan's make does not
detect the OS from Git Bash):

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 python validation/scripts/reserving_bayes_stan.py

Needs CmdStan 2.39.0 built at ~/.cmdstan/cmdstan-2.39.0 and RTools 4.5 at
C:/rtools45 (its make and g++), which this script puts on PATH. CI reads the
committed CSV and never runs Stan.

phi (the chain ladder's Pearson scale, degrees of freedom N - p) and c0 (the
log of the mean observed increment) are computed here as BayesOdp computes
them and passed to Stan as data, so both sides use the same plug-ins; the CSV
records them so the Rust test checks that they agree.
"""

import hashlib
import math
import os
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAN = ROOT / "validation/scripts/stan/bayes_odp.stan"
OUT = ROOT / "validation/reference/reserving_bayes_stan.csv"
CMDSTAN = Path.home() / ".cmdstan/cmdstan-2.39.0"
RTOOLS = Path("C:/rtools45")
DATASETS = ["genins", "raa"]
SEED = 20261012
CHAINS = 4
WARMUP = 2000
DRAWS = 25000
PRIOR_SD = 10.0
ADAPT_DELTA = 0.99
RAA_PHI = 983.635027030873  # reserving_bootstrap_r.csv, R ChainLadder 0.2.21

# Tolerances per dataset, with s the reference posterior sd (decision 10):
# (means abs x s, sds relative, reserve quantiles abs x s). Each is at least
# 4 times the Monte Carlo error of BayesOdp's defaults, measured as the
# seed-to-seed sd over seeds 0 to 19 for every row (the reference's own error
# is negligible next to it). RAA's young origins and its last development
# effects are skewed (b[9] and b[10] rest on one or two small cells), so its
# errors are about twice GenIns': at most 0.03 s on a mean, 4.6% on an sd
# (the 1982 reserve's) and 0.083 s on a percentile (the 1982 reserve's 95%).
# GenIns' are at most 0.018 s, 2.4% (sd of a[8]) and 0.066 s (the 2002
# reserve's 95%).
TOLERANCES = {
    "genins": (0.1, 0.10, 0.3),
    "raa": (0.15, 0.20, 0.4),
}
EXACT = 1e-9


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


def triangle(name):
    """Cumulative values as an origins x ages array (NaN unobserved), the
    origin labels and the ages."""
    df = pd.read_csv(ROOT / f"validation/data/{name}.csv", comment="#")
    origins = sorted(df["origin"].unique())
    ages = sorted(df["development"].unique())
    cum = np.full((len(origins), len(ages)), np.nan)
    for row in df.itertuples():
        cum[origins.index(row.origin), ages.index(row.development)] = row.value
    return cum, origins, ages


def plug_ins(cum):
    """Observed increments as (w, d, x), 0-based, phi and c0, as BayesOdp
    computes them: the volume-weighted chain ladder, its fitted increments
    backed out from each origin's latest value, and the Pearson scale."""
    no, nd = cum.shape
    latest = [int(np.max(np.nonzero(~np.isnan(cum[w]))[0])) for w in range(no)]
    factors = []
    for k in range(nd - 1):
        rows = [w for w in range(no) if latest[w] > k]
        factors.append(sum(cum[w, k + 1] for w in rows) / sum(cum[w, k] for w in rows))
    cells, chi2 = [], 0.0
    for w in range(no):
        fitted = np.zeros(latest[w] + 1)
        fitted[latest[w]] = cum[w, latest[w]]
        for d in range(latest[w] - 1, -1, -1):
            fitted[d] = fitted[d + 1] / factors[d]
        prev_fit = prev_obs = 0.0
        for d in range(latest[w] + 1):
            m = fitted[d] - prev_fit
            x = cum[w, d] - prev_obs
            cells.append((w, d, x))
            if m != 0:
                chi2 += (x - m) ** 2 / abs(m)
            prev_fit, prev_obs = fitted[d], cum[w, d]
    n_params = no + nd - 1
    phi = chi2 / (len(cells) - n_params)
    c0 = math.log(sum(c[2] for c in cells) / len(cells))
    future = [(w, d) for w in range(no) for d in range(latest[w] + 1, nd)]
    return cells, phi, c0, future


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
    sha = hashlib.sha256(STAN.read_bytes()).hexdigest()
    # Compile a copy under target/, so the executable stays out of the tree.
    build = ROOT / "target/stan"
    build.mkdir(parents=True, exist_ok=True)
    copy = build / STAN.name
    if not copy.exists() or copy.read_bytes() != STAN.read_bytes():
        copy.write_bytes(STAN.read_bytes())
    model = cmdstanpy.CmdStanModel(stan_file=str(copy))
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: bayes_odp.stan sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) metric dense_e adapt_delta {ADAPT_DELTA}"
    )
    rows = []
    for name in DATASETS:
        cum, origins, ages = triangle(name)
        cells, phi, c0, future = plug_ins(cum)
        if name == "raa":
            assert abs(phi / RAA_PHI - 1) < 1e-9, f"RAA phi {phi} vs R's {RAA_PHI}"
        no, nd = cum.shape
        data = {
            "N": len(cells),
            "n_origin": no,
            "n_dev": nd,
            "w": [c[0] + 1 for c in cells],
            "d": [c[1] + 1 for c in cells],
            "x": [c[2] for c in cells],
            "phi": phi,
            "c0": c0,
            "intercept_sd": PRIOR_SD,
            "origin_sd": PRIOR_SD,
            "development_sd": PRIOR_SD,
            "N_fut": len(future),
            "fw": [f[0] + 1 for f in future],
            "fd": [f[1] + 1 for f in future],
            "gamma_process": 1,
        }
        fit = model.sample(
            data=data,
            chains=CHAINS,
            iter_warmup=WARMUP,
            iter_sampling=DRAWS,
            seed=SEED,
            metric="dense_e",
            # RAA's last development effect rests on one small cell (shape
            # x / phi of 0.17), so its posterior is strongly skewed; a small
            # step size keeps the reference free of divergences.
            adapt_delta=ADAPT_DELTA,
            show_progress=False,
        )
        summary = fit.summary()
        params = ["c"] + [f"a[{k}]" for k in range(2, no + 1)] + [f"b[{k}]" for k in range(2, nd + 1)]
        rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
        worst = summary.loc[params, rhat_col].max()
        assert worst <= 1.005, f"{name}: R-hat {worst} above 1.005"
        divergences = int(fit.method_variables()["divergent__"].sum())
        assert divergences == 0, f"{name}: {divergences} divergent transitions"
        print(f"{name}: phi {phi:.6f} c0 {c0:.9f} max R-hat {worst:.5f} divergences 0")

        def row(quantity, arg, expected, abs_tol, rel_tol):
            rows.append(
                f"{name},bayes_odp,{quantity},{arg},{float(expected)!r},{float(abs_tol)!r},{float(rel_tol)!r},{source}"
            )

        mean_tol, sd_rel, quantile_tol = TOLERANCES[name]
        row("scale", "", phi, 0.0, EXACT)
        row("c0", "", c0, 0.0, EXACT)
        for p in params:
            if "[" in p:
                # a[k] or b[k]: Stan's a and b hold a[1] = b[1] = 0 too.
                draws = fit.stan_variable(p[0])[:, int(p[2:-1]) - 1]
            else:
                draws = fit.stan_variable(p)
            s = summarise(draws)
            row("mean", p, s["mean"], mean_tol * s["sd"], 0.0)
            row("sd", p, s["sd"], 0.0, sd_rel)
        reserve = fit.stan_variable("reserve")
        targets = [(str(o), reserve[:, k]) for k, o in enumerate(origins)]
        targets.append(("total", fit.stan_variable("total")))
        for arg, draws in targets:
            s = summarise(draws)
            if s["sd"] == 0.0:
                # The oldest origin: fully developed, an exact 0.
                for q in ["mean", "sd", "q05", "q50", "q95"]:
                    row(f"reserve_{q}", arg, 0.0, EXACT, 0.0)
                continue
            row("reserve_mean", arg, s["mean"], mean_tol * s["sd"], 0.0)
            row("reserve_sd", arg, s["sd"], 0.0, sd_rel)
            for q in ["q05", "q50", "q95"]:
                row(f"reserve_{q}", arg, s[q], quantile_tol * s["sd"], 0.0)

    header = [
        f"# Generated by validation/scripts/reserving_bayes_stan.py with {cmdstan}, cmdstanpy {cmdstanpy.__version__},"
        f" pandas {pd.__version__} and numpy {np.__version__}.",
        f"# Model validation/scripts/stan/bayes_odp.stan (sha256 {sha}).",
        f"# Seed {SEED}; {CHAINS} chains of {WARMUP} warm-up and {DRAWS} kept draws; metric dense_e; adapt_delta {ADAPT_DELTA}; priors sd {PRIOR_SD:g}; Gamma process.",
        "# phi and c0 are the chain ladder's (rel_tol 1e-9). With s the reference posterior sd: means abs_tol 0.1 s (RAA 0.15 s);",
        "# sds rel_tol 0.10 (RAA 0.20); reserve quantiles abs_tol 0.3 s (RAA 0.4 s): at least 4 times the Monte Carlo error of",
        "# BayesOdp's defaults over seeds 0 to 19. The oldest origin's reserve is an exact 0. arg: parameter, origin label or total.",
        "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
