"""Regenerate validation/reference/reserving_hierarchical_bf_stan.csv.

The reference for prospicio_reserving::bayes::HierarchicalBf
(docs/design/reserving-bayes.md, decisions 30 and 10): the same model in Stan
(validation/scripts/stan/hierarchical_bf.stan), sampled long enough that the
reference's own Monte Carlo error is negligible next to the tolerances.
Run from the repository root, in PowerShell or cmd (CmdStan's make does not
detect the OS from Git Bash):

    uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with pandas==3.0.6 --with numpy==2.5.4 python validation/scripts/reserving_hierarchical_bf_stan.py

Needs CmdStan 2.39.0 and RTools 4.5 as reserving_bayes_stan.py does, whose
toolchain, plug-ins and summaries this script reuses. CI reads the committed
CSV and never runs Stan.

One case, three programs on a grid of ten ages (labels in alphabetical order,
which is the triangle's segment order):

* evw: England, Verrall and Wuthrich (2012, Table 1) with its prior ultimates
  m_i as the premium, cut to its first six development ages (12 to 72 months)
  and all ten origins, so that it never observes ages 7 to 10 and its shares
  there are the shared pattern widened by omega;
* genins: GenIns with R ChainLadder's ClarkCapeCod premium;
* raa: RAA with a synthetic premium of 40,000 in 1981 rising by 2,000 a year.

Settings: apriori 0.7, level_cv 0.5, cv 0.1, development_sd 10, tau_scale 0.5,
omega_scale 0.5.

phi_c is plugged in as HierarchicalBf does (decision 30, D2): phi_c = kappa
Pbar_c with kappa = sum_c X2_c / Pbar_c / sum_c df_c, X2_c the Pearson
statistic of program c's chain ladder (reserving_bayes_stan.plug_ins, the same
formula as BayesOdp's) and df_c its cells less its origins plus ages less 1,
Pbar_c its mean premium per origin. Each phi_c and kappa are recorded, and
the Rust test asserts that the fit's scale agrees (rel_tol 1e-9).
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
STAN = ROOT / "validation/scripts/stan/hierarchical_bf.stan"
OUT = ROOT / "validation/reference/reserving_hierarchical_bf_stan.csv"
SEED = 20261105
CHAINS = 4
WARMUP = 2000
DRAWS = 10000
ADAPT_DELTA = 0.99
EXACT = 1e-9

APRIORI = 0.7
LEVEL_CV = 0.5
CV = 0.1
DEVELOPMENT_SD = 10.0
TAU_SCALE = 0.5
OMEGA_SCALE = 0.5

# Tolerances, with s the reference posterior sd: (means abs x s, sds relative,
# reserve quantiles abs x s), and the relative tolerance of the sds that
# involve EVW's unobserved ages (its loss ratio, its reserves and the totals:
# a heavy tail). Set from the seed-to-seed spread of HierarchicalBf's defaults
# (4 x 1,000 draws) against this reference over seeds 0 to 23 (the ignored test
# seed_sweep in validation/tests/reserving_bayes_hierarchical_bf.rs), each at
# least 1.5 times the largest error measured in its family:
#   means 0.116 s (omega), sds 0.139 (EVW's share at age 10; 0.113 for a
#   hyperparameter), quantiles 0.40 s (the 95% of EVW's 2006 reserve, whose
#   draws are heavy-tailed; 0.09 s elsewhere), heavy-tailed sds 0.455 (EVW's
#   loss ratio; biased low by 0.10 on average, as a sd of a heavy tail is
#   from 4,000 correlated draws).
TOLERANCES = (0.25, 0.25, 0.8)
HEAVY_SD = 0.7


def program(name):
    """(label, cumulative origins x ages with NaN, premium per origin, origin years)."""
    if name == "evw":
        df = pd.read_csv(ROOT / "validation/data/evw2012.csv", comment="#")
        df = df[df["development"] <= 72]
        value, expo = "paid", "prior"
    elif name == "genins":
        df = pd.read_csv(ROOT / "validation/data/genins_premium.csv", comment="#")
        value, expo = "paid", "premium"
    else:
        df = pd.read_csv(ROOT / "validation/data/raa.csv", comment="#")
        df = df.assign(premium=40000.0 + 2000.0 * (df["origin"] - 1981))
        value, expo = "value", "premium"
    origins = sorted(df["origin"].unique())
    ages = sorted(df["development"].unique())
    cum = np.full((len(origins), len(ages)), np.nan)
    premium = np.zeros(len(origins))
    for row in df.itertuples():
        w, d = origins.index(row.origin), ages.index(row.development)
        cum[w, d] = getattr(row, value)
        premium[w] = getattr(row, expo)
    return name, cum, premium, origins


def main():
    toolchain()
    programs = [program(n) for n in ("evw", "genins", "raa")]
    m = max(p[1].shape[1] for p in programs)
    sha = hashlib.sha256(STAN.read_bytes()).hexdigest()

    # Plug-in scales (decision 30, D2).
    pearson, dfs, mean_premium, cells, futures = [], [], [], [], []
    for _, cum, premium, _ in programs:
        c, phi, _, _ = plug_ins(cum)
        no, nd = cum.shape
        df = len(c) - (no + nd - 1)
        pearson.append(phi * df)
        dfs.append(df)
        mean_premium.append(premium.mean())
        cells.append(c)
        latest = [int(np.max(np.nonzero(~np.isnan(cum[w]))[0])) for w in range(no)]
        futures.append([(w, d) for w in range(no) for d in range(latest[w] + 1, m)])
    kappa = sum(x / p for x, p in zip(pearson, mean_premium)) / sum(dfs)
    phi = [kappa * p for p in mean_premium]

    # Stacked data.
    prog, prem, o_idx, d_idx, x_val, fo, fd = [], [], [], [], [], [], []
    offset = 0
    for c, (_, cum, premium, _) in enumerate(programs):
        no = cum.shape[0]
        prog += [c + 1] * no
        prem += list(premium)
        for w, d, x in cells[c]:
            o_idx.append(offset + w + 1)
            d_idx.append(d + 1)
            x_val.append(x)
        for w, d in futures[c]:
            fo.append(offset + w + 1)
            fd.append(d + 1)
        offset += no
    data = {
        "J": len(programs),
        "m": m,
        "N_o": offset,
        "prog": prog,
        "P": prem,
        "N": len(x_val),
        "o": o_idx,
        "d": d_idx,
        "x": x_val,
        "phi": phi,
        "apriori": APRIORI,
        "level_cv": LEVEL_CV,
        "shape": CV**-2,
        "development_sd": DEVELOPMENT_SD,
        "tau_scale": TAU_SCALE,
        "omega_scale": OMEGA_SCALE,
        "N_fut": len(fo),
        "fo": fo,
        "fd": fd,
    }

    # Inits from GenIns' chain ladder: ultimates and a normalised pattern.
    gen = programs[1][1]
    factors = [
        np.nansum(gen[: gen.shape[0] - k - 1, k + 1]) / np.nansum(gen[: gen.shape[0] - k - 1, k])
        for k in range(m - 1)
    ]
    cdf = np.cumprod(factors[::-1])[::-1]  # to ultimate, from age k
    share = np.diff(np.concatenate([[0.0], 1.0 / np.concatenate([cdf, [1.0]])]))
    beta0 = list(np.log(share[1:] / share[0]))
    alpha0 = []
    for _, cum, _, _ in programs:
        for w in range(cum.shape[0]):
            last = int(np.max(np.nonzero(~np.isnan(cum[w]))[0]))
            alpha0.append(math.log(cum[w, last] * (cdf[last] if last < m - 1 else 1.0)))
    inits = {
        "log_level": math.log(APRIORI),
        "tau": 0.2,
        "z": [0.0] * len(programs),
        "alpha": alpha0,
        "beta": beta0,
        "omega": 0.2,
        "delta": [[0.0] * (m - 1) for _ in programs],
    }

    build = ROOT / "target/stan"
    build.mkdir(parents=True, exist_ok=True)
    copy = build / STAN.name
    if not copy.exists() or copy.read_bytes() != STAN.read_bytes():
        copy.write_bytes(STAN.read_bytes())
    model = cmdstanpy.CmdStanModel(stan_file=str(copy))
    cmdstan = "CmdStan " + CMDSTAN.name.removeprefix("cmdstan-")
    source = (
        f"{cmdstan} via cmdstanpy {cmdstanpy.__version__}: hierarchical_bf.stan sha256 {sha[:16]}"
        f" seed {SEED} {CHAINS}x{DRAWS} draws ({WARMUP} warm-up) metric dense_e adapt_delta {ADAPT_DELTA}"
    )
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
    rhat_col = [c for c in summary.columns if c.lower().startswith("r_hat")][0]
    checked = ["level", "tau", "omega"] + [f"ratio[{k}]" for k in range(1, len(programs) + 1)]
    checked += [f"reserve[{k}]" for k in range(1, offset + 1)]
    worst = summary.loc[checked, rhat_col].max()
    divergences = int(fit.method_variables()["divergent__"].sum())
    print(f"kappa {kappa:.6f} phi {phi} max R-hat {worst:.5f} divergences {divergences}")
    assert worst <= 1.005, f"R-hat {worst} above 1.005"
    assert divergences == 0, f"{divergences} divergent transitions"

    params = (
        f"apriori={APRIORI:g};level_cv={LEVEL_CV:g};cv={CV:g};development_sd={DEVELOPMENT_SD:g};"
        f"tau_scale={TAU_SCALE:g};omega_scale={OMEGA_SCALE:g}"
    )
    rows = []
    mean_tol, sd_rel, quantile_tol = TOLERANCES

    def row(quantity, arg, expected, abs_tol, rel_tol):
        rows.append(
            f"hierarchical,hierarchical_bf,{params},{quantity},{arg},{float(expected)!r},"
            f"{float(abs_tol)!r},{float(rel_tol)!r},{source}"
        )

    def moments(prefix, arg, draws, heavy=False):
        s = summarise(draws)
        row(f"{prefix}_mean", arg, s["mean"], mean_tol * s["sd"], 0.0)
        row(f"{prefix}_sd", arg, s["sd"], 0.0, HEAVY_SD if heavy else sd_rel)
        return s

    row("kappa", "", kappa, 0.0, EXACT)
    for c, (label, cum, _, origins) in enumerate(programs):
        row("scale", label, phi[c], 0.0, EXACT)
    for name in ("level", "tau", "omega"):
        moments("hyper", name, fit.stan_variable(name))
    ratio = fit.stan_variable("ratio")
    log_share = fit.stan_variable("log_share")
    reserve = fit.stan_variable("reserve")
    program_reserve = fit.stan_variable("program_reserve")
    offset = 0
    for c, (label, cum, _, origins) in enumerate(programs):
        moments("ratio", label, ratio[:, c], heavy=label == "evw")
        for d in range(m):
            moments("share", f"{label}:{d + 1}", np.exp(log_share[:, c, d]))
        targets = [(f"{label}:{y}", reserve[:, offset + w]) for w, y in enumerate(origins)]
        targets.append((f"{label}:total", program_reserve[:, c]))
        offset += len(origins)
        for arg, draws in targets:
            s = summarise(draws)
            if s["sd"] == 0.0:
                for q in ["mean", "sd", "q05", "q50", "q95"]:
                    row(f"reserve_{q}", arg, 0.0, EXACT, 0.0)
                continue
            row("reserve_mean", arg, s["mean"], mean_tol * s["sd"], 0.0)
            row("reserve_sd", arg, s["sd"], 0.0, HEAVY_SD if label == "evw" else sd_rel)
            for q in ["q05", "q50", "q95"]:
                row(f"reserve_{q}", arg, s[q], quantile_tol * s["sd"], 0.0)
    s = summarise(fit.stan_variable("total"))
    row("reserve_mean", "total", s["mean"], mean_tol * s["sd"], 0.0)
    row("reserve_sd", "total", s["sd"], 0.0, HEAVY_SD)
    for q in ["q05", "q50", "q95"]:
        row(f"reserve_{q}", "total", s[q], quantile_tol * s["sd"], 0.0)

    header = [
        f"# Generated by validation/scripts/reserving_hierarchical_bf_stan.py with {cmdstan}, cmdstanpy {cmdstanpy.__version__},",
        f"# pandas {pd.__version__} and numpy {np.__version__}. Model validation/scripts/stan/hierarchical_bf.stan (sha256 {sha}).",
        f"# Seed {SEED}; {CHAINS} chains of {WARMUP} warm-up and {DRAWS} kept draws; metric dense_e; adapt_delta {ADAPT_DELTA}; Gamma process.",
        "# Programs evw (EVW cut to six ages), genins and raa; scale and kappa are the plug-ins (rel_tol 1e-9). With s the reference",
        f"# posterior sd: means abs_tol {mean_tol} s, sds rel_tol {sd_rel} ({HEAVY_SD} for EVW's loss ratio and reserves and the totals), reserve quantiles abs_tol {quantile_tol} s (TOLERANCES in the script).",
        "# arg: a hyperparameter, a program, program:age, program:origin year or program:total; an exact 0 reserve has no spread.",
        "dataset,method,params,quantity,arg,expected,abs_tol,rel_tol,source",
    ]
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
