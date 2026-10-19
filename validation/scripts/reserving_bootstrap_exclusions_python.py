"""Regenerate validation/reference/reserving_bootstrap_exclusions_python.csv.

The ODP bootstrap under link ratio exclusions (decision 14 of
docs/design/reserving-v02.md) against chainladder-python's
BootstrapODPSample(n_periods, drop_high, drop_low). Run from the repository
root:

    uv run --no-project --with chainladder==0.10.1 python validation/scripts/reserving_bootstrap_exclusions_python.py

raa and genins (column value) are bootstrapped under n_periods=3,
drop_high=1 with drop_low=1, and both together, with hat_adj=False. Rows
give, per setting:

* fitted: the fitted incremental value of each cell (arg `origin;age`), from
  Chainladder().fit(Development(...).fit_transform(tri)).full_expectation_,
  the back-cast from each origin's latest value with the factors under the
  exclusions.
* has_residual: 1 where the cell has a residual under prospicio's rule (the
  cell after a kept link ratio), 0 where it does not, for every cell after
  the first age. The first age is left out: chainladder-python gives it the
  weight of link 0, so dropping link (o, 0) also drops the first age's
  residual of origin o, which prospicio does not copy (its first age always
  has a residual). chainladder-python's weights w_ are read off a fit of the
  bootstrap itself, so this is what its pool uses.
* residual: the unscaled Pearson residual (x - m) / sqrt(|m|) of each cell
  with a residual (the first age included), where the fitted value m is not
  zero.
* phi: sum of the kept r^2 over (N - p), N all observed cells and p = 19
  (raa and genins: origins + ages - 1), with prospicio's rule for which
  cells are kept, so the first age's residuals count. chainladder-python's
  own scale_ is phi_chainladder; it differs from phi by the first-age
  residuals its quirk drops, and the script checks that identity.
* mean_reserve and mean_reserve_se: the mean total reserve over N_SIMS
  samples (random_state=42) and its standard error, each sample refitted by
  Development with the observed exclusions as fixed positions (its `drop`
  argument), because prospicio keeps the observed triangle's link ratios in
  every simulation (decision 14 item 2); then Chainladder(). The samples
  carry parameter error alone (no process error), as prospicio with
  ProcessDistribution::None. The samples come from OursODP, chainladder's
  BootstrapODPSample with one method, _get_simulation, copied and changed
  to prospicio's pool: the first age's residuals always in (chainladder's
  weight quirk drops them with link 0), no centring, zeros kept, residuals
  times sqrt(N / (N - p)) (chainladder's hat_adj=False applies no such
  factor), and the cells without a residual out. Everything else (the
  fitted values, the resampling, the refit, the projection) is
  chainladder's. The first-age residuals matter: under n_periods=3 on RAA the
  oldest origins' first-age fitted values are poor and their residuals large
  (up to 135 against a pool of 20), and with them the newest origin's pseudo
  value, so the native pool's mean reserve is far from this one's.
  mean_reserve_reranked is the mean when each sample ranks its own link
  ratios again (Development with the same n_periods, drop_high and
  drop_low), and mean_reserve_native the mean with chainladder's own pool
  (the quirk, centring, no zeros), both for the record and not compared.
  median_reserve and median_halfwidth (half the spread between the
  0.5 +- 2.5 / sqrt(N_SIMS) quantiles, a five-sigma interval) give a
  comparison that survives heavy tails: RAA under n_periods=3 with both
  drops leaves ten link ratios and a pool of twenty residuals, a single
  link at almost every age and a newest origin that is one cell, so a
  pseudo factor can divide by a near-zero value and the reserve has outliers
  of 1e8 on a median of 4e4; its mean has no usable standard error.
  The default setting (no exclusions) has the same means and checks that
  OursODP's pool convention agrees with prospicio's where no exclusions
  hide a difference.

`arg` is empty for totals. The comparisons are exact (abs_tol 0, rel_tol
1e-9) for the deterministic quantities.
"""

import csv
import math
import warnings

import chainladder as cl
import numpy as np
import pandas as pd

OUT = "validation/reference/reserving_bootstrap_exclusions_python.csv"
VER = cl.__version__
REL = 1e-9
N_SIMS = 20000
SEED = 42

SETTINGS = [
    {},
    {"n_periods": 3},
    {"drop_high": 1, "drop_low": 1},
    {"n_periods": 3, "drop_high": 1, "drop_low": 1},
]


def read_triangle(name):
    df = pd.read_csv(f"validation/data/{name}.csv", comment="#")
    df["origin"] = df["origin"].astype(str)
    df["valuation"] = (df["origin"].astype(int) + df["development"] // 12 - 1).astype(str)
    return cl.Triangle(
        df, origin="origin", development="valuation", columns=["value"], cumulative=True
    )


def label(setting):
    return ";".join(f"{k}={v}" for k, v in setting.items()) or "default"


class OursODP(cl.BootstrapODPSample):
    """BootstrapODPSample with prospicio's residual pool; see the docstring."""

    def _get_simulation(self, X, exp_incr_triangle):
        # Copied from chainladder 0.10.1 BootstrapODPSample._get_simulation
        # (hat_adj=False), with the pool changed.
        xp = X.get_array_module()
        unscaled = (
            (X.cum_to_incr().values - exp_incr_triangle) / xp.sqrt(xp.abs(exp_incr_triangle))
        )[0, 0, ...]
        w = np.nan_to_num(self.w_[0, 0])
        kept = np.concatenate((np.ones((w.shape[0], 1), dtype=bool), w > 0), axis=1)
        unscaled = np.where(kept, unscaled, np.nan)
        n_params = self.design_matrix_.shape[1]
        n_obs = xp.nansum(X.nan_triangle)
        degree_freedom = n_obs - n_params
        scale_phi = np.nansum(np.where(np.isfinite(unscaled), unscaled, 0.0) ** 2) / degree_freedom
        adjusted = unscaled * np.sqrt(n_obs / degree_freedom)
        pool = adjusted[np.isfinite(adjusted)]
        random_state = xp.random.RandomState(self.random_state)
        k, v, o, d = X.shape
        draws = np.concatenate(
            [
                (random_state.choice(pool, size=exp_incr_triangle.shape, replace=True)
                 * (exp_incr_triangle * 0 + 1))[None, ...]
                for _ in range(self.n_sims)
            ],
            0,
        ).reshape(self.n_sims, o, d)
        b = np.repeat(exp_incr_triangle[None, ...], self.n_sims, 0)
        triangles = (draws * np.sqrt(np.abs(b)) + b).cumsum(2)[None, ...].swapaxes(0, 1)
        obj = X.copy()
        obj.kdims = np.arange(self.n_sims)
        obj.key_labels = ["Simulation_#"]
        obj.values = triangles
        obj._set_slicers()
        return obj, scale_phi


def call(setting):
    args = ", ".join(f"{k}={v}" for k, v in setting.items())
    return f"BootstrapODPSample({args}, hat_adj=False)"


def main():
    rows = []

    def emit(dataset, method, quantity, arg, value, source, abs_tol=0, rel_tol=REL):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"not finite: {dataset} {method} {quantity} {arg}")
        rows.append([dataset, method, quantity, arg, repr(value), abs_tol, rel_tol, source])

    for name in ["raa", "genins"]:
        tri = read_triangle(name)
        no = tri.shape[2]
        origins = [str(d.year) for d in tri.origin.to_timestamp()]
        ages = [int(a) for a in tri.development.values]
        n_cells = int(np.isfinite(tri.values[0, 0]).sum())
        n_params = no + len(ages) - 1
        dof = n_cells - n_params
        for setting in SETTINGS:
            method = label(setting)
            src = f"chainladder-python {VER}: {call(setting)}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot = cl.BootstrapODPSample(
                    n_sims=10, random_state=SEED, hat_adj=False, **setting
                ).fit(tri)
                dev = cl.Development(**setting).fit(tri)
                model = cl.Chainladder().fit(dev.transform(tri))
            full = (
                f"chainladder-python {VER}: Chainladder().fit(Development("
                f"{', '.join(f'{k}={v}' for k, v in setting.items())}).fit_transform(tri))"
            )

            expected = np.nan_to_num(
                model.full_expectation_.cum_to_incr().values[0, 0, :, : len(ages)]
            )
            observed = tri.cum_to_incr().values[0, 0]
            finite = np.isfinite(observed)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (observed - expected) / np.sqrt(np.abs(expected))

            # Weights of the links of the bootstrap itself, padded to the
            # incremental grid: the first age takes link 0's weight.
            w = np.nan_to_num(np.asarray(boot.w_)[0, 0])
            kept = (w > 0) & np.isfinite(observed[:, 1:]) & np.isfinite(observed[:, :-1])
            has = np.zeros_like(finite)
            has[:, 0] = True
            has[:, 1:] = kept
            has &= finite

            # chainladder-python's quirk: the first age's residual of an
            # origin follows link 0's weight.
            quirk_dropped = np.zeros_like(has)
            quirk_dropped[:, 0] = finite[:, 0] & ~((w[:, 0] > 0) & np.isfinite(observed[:, 1]))
            quirk_dropped[no - 1, 0] = False  # the newest origin has no link 0

            for o in range(no):
                for d in range(len(ages)):
                    if not finite[o, d]:
                        continue
                    arg = f"{origins[o]};{ages[d]}"
                    emit(name, method, "fitted", arg, expected[o, d], full + ".full_expectation_")
                    if d > 0:
                        emit(name, method, "has_residual", arg, has[o, d], src + ".w_")
                    if has[o, d] and expected[o, d] != 0:
                        # The corner cells fit exactly, so their residuals are
                        # rounding noise: an absolute tolerance.
                        emit(name, method, "residual", arg, r[o, d],
                             full + " (x - m) / sqrt(|m|)", abs_tol=1e-8)

            sq = np.where(has & (expected != 0), r**2, 0.0)
            phi = sq.sum() / dof
            emit(name, method, "phi", "", phi, src + " scale_, first age kept")
            # The identity that ties prospicio's phi to scale_.
            lost = np.where(quirk_dropped & (expected != 0), r**2, 0.0).sum()
            scale = float(np.asarray(boot.scale_))
            assert abs(scale - (phi - lost / dof)) < 1e-8 * scale, (name, method, scale, phi, lost)
            emit(name, method, "phi_chainladder", "", scale, src + ".scale_")

            # The simulated mean total reserve, samples refitted with the
            # observed exclusions as fixed positions.
            excluded = [
                (origins[o], ages[k])
                for o in range(no)
                for k in range(len(ages) - 1)
                if np.isfinite(observed[o, k + 1]) and not (w[o, k] > 0)
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fixed = cl.Development(drop=excluded) if excluded else cl.Development()

                def reserve(sims, dev):
                    ibnr = cl.Chainladder().fit(dev.fit_transform(sims)).ibnr_.sum("origin")
                    return np.asarray(ibnr.values).reshape(-1)

                sims = OursODP(
                    n_sims=N_SIMS, random_state=SEED, hat_adj=False, **setting
                ).fit_transform(tri)
                total = reserve(sims, fixed)
                reranked = reserve(sims, cl.Development(**setting))
                native = reserve(
                    cl.BootstrapODPSample(
                        n_sims=N_SIMS, random_state=SEED, hat_adj=False, **setting
                    ).fit_transform(tri),
                    fixed,
                )
            sim_src = f"chainladder-python {VER}: {call(setting)} n_sims={N_SIMS} random_state={SEED}"
            emit(name, method, "mean_reserve", "", total.mean(), sim_src + " fixed drop positions",
                 abs_tol=0, rel_tol=0)
            emit(name, method, "mean_reserve_se", "", total.std(ddof=1) / math.sqrt(N_SIMS),
                 sim_src, abs_tol=0, rel_tol=0)
            # The median and a half-width of its five-sigma interval from the
            # order statistics, for settings whose mean has no usable
            # standard error (heavy tails).
            band = 2.5 / math.sqrt(N_SIMS)
            lo, mid, hi = np.quantile(total, [0.5 - band, 0.5, 0.5 + band])
            emit(name, method, "median_reserve", "", mid, sim_src + " fixed drop positions",
                 abs_tol=0, rel_tol=0)
            emit(name, method, "median_halfwidth", "", (hi - lo) / 2, sim_src,
                 abs_tol=0, rel_tol=0)
            emit(name, method, "mean_reserve_reranked", "", reranked.mean(),
                 sim_src + " re-ranked per sample", abs_tol=0, rel_tol=0)
            emit(name, method, "mean_reserve_native", "", native.mean(),
                 sim_src + " native pool", abs_tol=0, rel_tol=0)
            emit(name, method, "n_excluded_links", "", len(excluded), src + ".w_")

    with open(OUT, "w", newline="") as f:
        f.write(
            f"# Generated by validation/scripts/reserving_bootstrap_exclusions_python.py "
            f"with chainladder-python {VER}.\n"
            "# BootstrapODPSample(n_periods, drop_high, drop_low, hat_adj=False) on raa "
            "and genins (value).\n"
            "# See the script's docstring.\n"
        )
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
