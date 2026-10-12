"""Reinsurance pricing reference numbers from Mildenhall's `aggregate`.

Writes validation/reference/re_pricing_aggregate.csv (long format:
example,quantity,value,note).  Six worked examples built from the reinsurance pricing
user guide of the `aggregate` package (docs 2_x_re_pricing.rst, "Reinsurance
Pricing"), plus designed extras (the reinstatement schedules, the swing terms,
the half placements and the risk pricing), marked by comments in this script.

aggregate version: 1.0.1 (the version used for the repo's other references).
Doc source (the guide was removed from master after 0.22, so read the tag):
  https://github.com/mynl/aggregate/blob/v0.22.0/docs/2_user_guides/2_x_re_pricing.rst
  https://github.com/mynl/aggregate/blob/v0.22.0/docs/2_user_guides/DecL/080_reinsurance.rst
  https://github.com/mynl/aggregate/blob/master/docs/2_aggregate_overview/pipeline-reinsurance.rst
  (the last describes 1.0.1's reinsurance reporting surface)

Run (from any directory; the output path does not depend on it):
  uv venv .venv --python 3.13
  uv pip install --python .venv aggregate==1.0.1
  set PYTHONIOENCODING=utf-8
  .venv\\Scripts\\python validation/scripts/aggregate_re_pricing.py

Every build uses normalize=False (so quantiles compare with prospicio, which
lumps the cut tail on the last grid point).  The grid (bs, log2) that aggregate
chose or was given is recorded as `bs` and `log2` rows for each example.
"""

import csv
import math
import os
import sys
import warnings

import numpy as np
import scipy.stats as ss

warnings.filterwarnings("ignore")

import aggregate
from aggregate import Distortion, build, lognorm_lev, xsden_to_meancvskew
from aggregate._grid_distribution import GridDistribution
from aggregate.contract_terms import (
    ProfitCommissionTerms,
    ReinstatementTerms,
    RetroTerms,
    SlideTerms,
    SwingTerms,
)
from aggregate.underwriter import Underwriter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "reference", "re_pricing_aggregate.csv"))
PS = [0.5, 0.9, 0.99, 0.999]
ROWS = []


def add(example, quantity, value, note=""):
    ROWS.append((example, quantity, repr(float(value)), note))


def stats(ser):
    """mean, sd, cv, skew of a density Series indexed by loss."""
    m, cv, sk = xsden_to_meancvskew(ser.index.to_numpy(float), ser.to_numpy(float))
    return float(m), float(m * cv), float(cv), float(sk)


def quant(ser, bs, p):
    """Lower quantile on the grid, exactly as Aggregate.q does."""
    return float(GridDistribution.from_series(ser, bs=bs, name="x", is_loss_value=True).q(p))


def summ(ex, prefix, ser, bs, note="", quantiles=True):
    m, sd, cv, sk = stats(ser)
    add(ex, f"{prefix}_mean", m, note)
    add(ex, f"{prefix}_sd", sd, note)
    add(ex, f"{prefix}_cv", cv, note)
    add(ex, f"{prefix}_skew", sk, note)
    # xsden_to_meancvskew puts the density's deficit 1 - sum(p) (FFT rounding, 1e-15) at the
    # implied maximum loss xs[-1] + bs, which moves the skew of a light-tailed density in the
    # fifth digit.  skew_central is the skewness of the density as given, without that.
    x, p = ser.index.to_numpy(float), ser.to_numpy(float)
    mu = float(x @ p)
    var = float(((x - mu) ** 2) @ p)
    add(ex, f"{prefix}_skew_central", float(((x - mu) ** 3) @ p) / var**1.5,
        "central-moment skewness of the density, without xsden_to_meancvskew's deficit convention")
    add(ex, f"{prefix}_density_deficit", 1.0 - float(p.sum()), "1 - sum(density), placed at xs[-1] + bs by xsden_to_meancvskew")
    if quantiles:
        for p in PS:
            add(ex, f"{prefix}_q{p}", quant(ser, bs, p), note)
    return m, sd, cv, sk


def grid(ex, a):
    add(ex, "bs", a.bs, "bucket size")
    add(ex, "log2", a.log2, "grid has 2**log2 buckets, first at 0")


def comp_counts(ex, a, label):
    """Expected claim count of each exposure class (component) of a built aggregate."""
    for i, v in enumerate(np.atleast_1d(a.en), 1):
        add(ex, f"class{i}_expected_count", v, f"{label}: expected claims of class {i}")


def lev_ln(mu, sigma, limit):
    return float(lognorm_lev(mu, sigma, 1, limit))


def agg_density(a):
    """Aggregate density of the object's own (requested) view."""
    return a.density_df.set_index("loss").p_total


def reins_df(a):
    df = a.reins_density_df
    return df.set_index("loss") if "loss" in df.columns else df


# ---------------------------------------------------------------------------
# Book shared by examples 1, 2, 4 and 6: docs "Reinsurance Functions" ReTester
# ---------------------------------------------------------------------------
RT_SEV = "5000 xs 0 sev lognorm 100 cv 5"
RT_N = 10
RT_SIGMA = math.sqrt(math.log(1 + 25))
RT_MU = math.log(100) - RT_SIGMA**2 / 2
TOWER = [(250, 0), (250, 250), (500, 500), (1000, 1000), (3000, 2000)]


def retester(clause_occ="", clause_agg="", name="Re.T"):
    return build(
        f"agg {name} {RT_N} claims {RT_SEV} {clause_occ} poisson {clause_agg}",
        normalize=False,
    )


def layer_decl(layers):
    return " and ".join(f"{lim} xs {att}" for lim, att in layers)


# ---------------------------------------------------------------------------
# Example 1: tower_gross
# ---------------------------------------------------------------------------
def ex_tower_gross():
    ex = "tower_gross"
    a = retester("occurrence ceded to " + layer_decl(TOWER))
    grid(ex, a)
    add(ex, "claim_count_mean", RT_N, "Poisson, ground-up claims per year")
    add(ex, "sev_mean_exact", lev_ln(RT_MU, RT_SIGMA, 5000), "exact E[min(X,5000)] per claim")
    add(ex, "gross_mean_exact", RT_N * lev_ln(RT_MU, RT_SIGMA, 5000), "n * exact limited severity mean")
    rd = reins_df(a)
    g = rd["p_agg_gross"]
    m, sd, cv, sk = summ(ex, "gross", g, a.bs, "annual aggregate, grid moments")
    add(ex, "gross_mean_grid_est_m", a.est_m, "Aggregate.est_m, the grid mean")
    for p in PS:
        add(ex, f"gross_q{p}_aggregate_q", float(a.q(p)), "Aggregate.q on its own density (ceded=gross here)")
    # severity distribution per claim
    sev = rd["p_sev_gross"]
    sm = float((sev.index.to_numpy(float) * sev.to_numpy(float)).sum())
    add(ex, "sev_mean_grid", sm, "mean of the discretized severity")
    for lim, att in TOWER:
        tag = f"layer_{lim}x{att}"
        al = retester(f"occurrence ceded to {lim} xs {att}")
        cd = reins_df(al)["p_agg_ceded_occ"]
        summ(ex, f"{tag}_ceded", cd, al.bs, "annual ceded aggregate of this layer alone")
        lev = n_lev = RT_N * (lev_ln(RT_MU, RT_SIGMA, att + lim) - lev_ln(RT_MU, RT_SIGMA, att))
        add(ex, f"{tag}_ceded_mean_exact", n_lev, "n * (LEV(att+lim) - LEV(att)), exact")
        add(ex, f"{tag}_pr_attach_exact", 1 - float(ss.lognorm(RT_SIGMA, scale=math.exp(RT_MU)).cdf(att)),
            "P(X > attachment), per claim, exact")
        add(ex, f"{tag}_expected_count", RT_N * (1 - float(ss.lognorm(RT_SIGMA, scale=math.exp(RT_MU)).cdf(att))),
            "exact expected claims exceeding the attachment")
        add(ex, f"{tag}_loss_on_line", float(cd.index.to_numpy(float) @ cd.to_numpy(float)) / lim,
            "grid mean ceded / limit")
        st = a.reins_stats_df
        k = TOWER.index((lim, att)) + 1
        col = ("occ", f"layer.{k}")
        add(ex, f"{tag}_tower_agg_mean", st.loc[("agg", "mean"), col],
            "reins_stats_df occ layer.k agg mean (tower build)")
        add(ex, f"{tag}_tower_agg_cv", st.loc[("agg", "cv"), col], "reins_stats_df occ layer.k agg cv")
        add(ex, f"{tag}_pr_attach_grid", st.loc[("meta", "pr_attach"), col], "reins_stats_df, grid probability")
        add(ex, f"{tag}_pr_detach_grid", st.loc[("meta", "pr_detach"), col], "reins_stats_df, P(X >= att+lim)")
        add(ex, f"{tag}_cond_sev_mean", st.loc[("sev", "mean"), col],
            "mean layer loss given the claim reaches the layer (conditional)")
    tot = reins_df(a)["p_agg_ceded_occ"]
    summ(ex, "tower_ceded_total", tot, a.bs, "whole tower ceded (equals gross: tower covers 0 to 5000)")


# ---------------------------------------------------------------------------
# Example 2: tower_program (ceded / net of a two-layer program, plus a share)
# ---------------------------------------------------------------------------
PROG = [(500, 500), (1000, 1000)]


def ex_tower_program():
    ex = "tower_program"
    a = retester("occurrence net of " + layer_decl(PROG))
    grid(ex, a)
    rd = reins_df(a)
    summ(ex, "gross", rd["p_agg_gross"], a.bs, "annual aggregate before the program")
    summ(ex, "ceded", rd["p_agg_ceded_occ"], a.bs, "annual aggregate ceded to 500x500 + 1000x1000")
    summ(ex, "net", rd["p_agg_net_occ"], a.bs, "annual aggregate retained")
    add(ex, "gross_mean_exact", RT_N * lev_ln(RT_MU, RT_SIGMA, 5000))
    ced_ex = RT_N * sum(lev_ln(RT_MU, RT_SIGMA, att + lim) - lev_ln(RT_MU, RT_SIGMA, att) for lim, att in PROG)
    add(ex, "ceded_mean_exact", ced_ex, "n * sum of layer exact LEV differences")
    add(ex, "net_mean_exact", RT_N * lev_ln(RT_MU, RT_SIGMA, 5000) - ced_ex)
    add(ex, "net_sev_mean", a.reins_stats_df.loc[("sev", "mean"), ("occ", "Net")], "net severity per claim, grid")
    add(ex, "ceded_sev_mean_unconditional", ced_ex / RT_N, "exact ceded severity per ground-up claim")
    add(ex, "net_est_m", a.est_m, "Aggregate.est_m of the net view (the object's own density)")
    for lim, att in PROG:
        al = retester(f"occurrence ceded to {lim} xs {att}")
        summ(ex, f"layer_{lim}x{att}_ceded", reins_df(al)["p_agg_ceded_occ"], al.bs)
    # a partial placement: 80% part of 1000 xs 1000
    s = retester("occurrence ceded to 80% po 1000 xs 1000")
    summ(ex, "share80_1000x1000_ceded", reins_df(s)["p_agg_ceded_occ"], s.bs,
         "80% share of 1000 xs 1000; share ceded losses land off-grid and are re-bucketed linearly")
    add(ex, "share80_1000x1000_ceded_mean_exact",
        0.8 * RT_N * (lev_ln(RT_MU, RT_SIGMA, 2000) - lev_ln(RT_MU, RT_SIGMA, 1000)))
    # per-layer annual stats in the program build
    st = a.reins_stats_df
    for k, (lim, att) in enumerate(PROG, 1):
        col = ("occ", f"layer.{k}")
        add(ex, f"layer_{lim}x{att}_program_agg_mean", st.loc[("agg", "mean"), col])
        add(ex, f"layer_{lim}x{att}_program_agg_cv", st.loc[("agg", "cv"), col])


# ---------------------------------------------------------------------------
# Example 3: agg_limit_reinst (Bear and Nemlick Treaty 2)
# ---------------------------------------------------------------------------
BN2 = (
    "[2000 2000 2000] exposure at [.1 .14 .21] rate 700 xs 0 "
    "sev 300 * pareto [1.5 1.3 1.1] - 300 mixed gamma 0.07"
)


def ex_agg_limit_reinst():
    ex = "agg_limit_reinst"
    a = build(f"agg Re.BN2 {BN2}", bs=1 / 8, normalize=False)
    grid(ex, a)
    add(ex, "claim_count_mean", a.n, "expected claims per year, all three classes")
    comp_counts(ex, a, "BN2")
    add(ex, "gross_mean_exact", a.stats_df.loc[("agg", "mean"), "mixed"], "stats_df mixed (theoretical) mean")
    add(ex, "gross_cv_exact", a.stats_df.loc[("agg", "cv"), "mixed"], "stats_df mixed (theoretical) cv")
    dens = agg_density(a)
    summ(ex, "gross", dens, a.bs, "layer 700 xs 300 of the single-parameter Pareto, all three classes")
    add(ex, "gross_mean_grid_est_m", a.est_m)
    lim = build(f"agg Re.BN2a {BN2} aggregate ceded to 2800 xs 0", bs=1 / 8, normalize=False)
    ld = agg_density(lim)
    summ(ex, "agglimit2800_ceded", ld, lim.bs, "recovery under a 2800 annual aggregate limit")
    add(ex, "agglimit2800_ceded_est_m", lim.est_m)
    add(ex, "agglimit2800_pr_hit",
        float(dens[dens.index >= 2800].sum()), "P(aggregate loss >= 2800), exhausts the limit")
    add(ex, "agglimit2800_mean_loss_beyond", float(((dens.index.to_numpy(float) - 2800).clip(0) * dens.to_numpy(float)).sum()),
        "E[(X - 2800)+], the loss above the aggregate limit")
    # reinstatement premium: layer 700 xs 0 of the shifted severity, 3 reinstatements
    R = dens.index.to_numpy(float)
    p = dens.to_numpy(float)
    for tag, rates, dep in [("A", (1.0, 1.0, 1.0), 1200.0), ("B", (0.0, 0.5, 1.0), 1200.0)]:
        t = ReinstatementTerms(limit=700.0, rates=rates, deposit=dep)
        h = t.reinstatement_premium(R)
        rec = t.recovery(R)
        mh = float(h @ p)
        add(ex, f"reinst{tag}_deposit", dep, f"rates {rates}, limit 700, rate on line deposit/limit")
        add(ex, f"reinst{tag}_rates_sum", sum(rates))
        add(ex, f"reinst{tag}_premium_mean", mh, "E[h(R)], expected reinstatement premium (currency)")
        add(ex, f"reinst{tag}_premium_sd", float(math.sqrt(max(float((h**2) @ p) - mh**2, 0))))
        add(ex, f"reinst{tag}_total_premium_mean", dep + mh, "deposit + E[h(R)]")
        add(ex, f"reinst{tag}_recovery_mean", float(rec @ p), "E[min(R, 2800)] (equals the aggregate limit cession)")
        add(ex, f"reinst{tag}_max_premium", t.maximum_reinstatement_premium)
        add(ex, f"reinst{tag}_pr_any_reinstatement", float(p[R > 0].sum()), "P(R > 0)")
    # per-class gross check: independent classes would differ; shared mixing induces correlation


# ---------------------------------------------------------------------------
# Example 4: stop_loss (aggregate stop loss and aggregate deductible)
# ---------------------------------------------------------------------------
BN1 = (
    "[9000 3000] exposure at [0.04 0.03] rate 160 xs 0 "
    "sev 40 * pareto [0.9 0.95] - 40"
)


def ex_stop_loss():
    ex = "stop_loss"
    # (a) ReTester gross with aggregate stop-loss layers (docs Re:03 / Re:04 style)
    g = retester()
    grid(ex, g)
    gd = agg_density(g)
    summ(ex, "gross", gd, g.bs)
    for lim, att in [(250, 750), (1500, 1000)]:
        a = retester(clause_agg=f"aggregate ceded to {lim} xs {att}")
        d = agg_density(a)
        summ(ex, f"asl_{lim}x{att}_ceded", d, a.bs, "ceded annual aggregate stop loss layer")
        add(ex, f"asl_{lim}x{att}_pr_attach", float(gd[gd.index > att].sum()), "P(gross > attachment), grid")
        add(ex, f"asl_{lim}x{att}_pr_exhaust", float(gd[gd.index >= att + lim].sum()), "P(gross >= att+limit), grid")
        add(ex, f"asl_{lim}x{att}_loss_on_line", float(d.index.to_numpy(float) @ d.to_numpy(float)) / lim)
    both = retester(clause_agg="aggregate ceded to 250 xs 750 and 1500 xs 1000")
    summ(ex, "asl_both_ceded", agg_density(both), both.bs, "250 xs 750 plus 1500 xs 1000 together")
    net = retester(clause_agg="aggregate net of 250 xs 750 and 1500 xs 1000")
    summ(ex, "asl_both_net", agg_density(net), net.bs, "gross net of both stop-loss layers")
    # a 1000 xs 1000 aggregate stop loss with a half placement
    h = retester(clause_agg="aggregate ceded to 50% po 1500 xs 1000")
    summ(ex, "asl_50pct_1500x1000_ceded", agg_density(h), h.bs, "50% part of 1500 xs 1000")
    # (b) Bear and Nemlick Treaty 1: aggregate deductible 360 on 160 xs 40
    mix_cv = ((1.036 - 1) / 5.154) ** 0.5
    add(ex, "aad_mix_cv", mix_cv, "gamma mixing cv: ((1.036-1)/5.154)**.5")
    b = build(f"agg Re.BN1 {BN1} mixed gamma {mix_cv!r}", normalize=False)
    grid(ex + "_aad", b)
    summ(ex + "_aad", "gross", agg_density(b), b.bs, "subject 160 xs 40 layer, two classes, shared mixing")
    add(ex + "_aad", "gross_mean_exact", b.stats_df.loc[("agg", "mean"), "mixed"])
    add(ex + "_aad", "gross_cv_exact", b.stats_df.loc[("agg", "cv"), "mixed"])
    add(ex + "_aad", "claim_count_mean", b.n)
    comp_counts(ex + "_aad", b, "BN1")
    n = build(f"agg Re.BN1a {BN1} mixed gamma {mix_cv!r} aggregate net of 360 xs 0", normalize=False)
    nd = agg_density(n)
    summ(ex + "_aad", "after_aad", nd, n.bs, "loss to the reinsurer after a 360 aggregate deductible")
    add(ex + "_aad", "aad_insurance_charge", n.est_m / b.est_m, "ratio of means after/before the deductible")
    add(ex + "_aad", "aad_value", b.est_m - n.est_m, "expected loss absorbed by the deductible: E[min(X,360)]")
    gdens = agg_density(b)
    x = gdens.index.to_numpy(float)
    add(ex + "_aad", "aad_value_by_hand", float(np.minimum(x, 360.0) @ gdens.to_numpy(float)), "E[min(X,360)] from the density")
    pn = build(f"agg Re.BN1p {BN1} poisson aggregate net of 360 xs 0", normalize=False)
    pg = build(f"agg Re.BN1pg {BN1} poisson", normalize=False)
    add(ex + "_aad", "poisson_gross_mean", pg.est_m)
    summ(ex + "_aad", "poisson_after_aad", agg_density(pn), pn.bs, quantiles=True)


# ---------------------------------------------------------------------------
# Example 5: variable_features
# ---------------------------------------------------------------------------
def ex_variable_features():
    # ---- retro rating (Treaty 4) -------------------------------------------
    ex = "variable_retro"
    a = build(f"agg Re.BN4 {BN1} poisson", normalize=False)
    grid(ex, a)
    d = agg_density(a)
    x = d.index.to_numpy(float)
    p = d.to_numpy(float)
    summ(ex, "ceded_loss", d, a.bs, "annual loss to 160 xs 40, Poisson")
    comp_counts(ex, a, "BN4 (same classes as BN1, Poisson)")
    subject, min_rate, max_rate, lcf = 12000.0, 0.03, 0.10, 100 / 75
    prem = np.minimum(max_rate * subject, np.maximum(min_rate * subject, lcf * x))
    add(ex, "subject_premium", subject)
    add(ex, "lcf", lcf, "loss conversion factor 100/75")
    add(ex, "min_premium", min_rate * subject)
    add(ex, "max_premium", max_rate * subject)
    add(ex, "retro_premium_mean", float(prem @ p), "E[min(max, max(min, lcf * L))]")
    add(ex, "retro_rate_mean", float(prem @ p) / subject, "docs: erp / subject premium")
    rt = RetroTerms(basic=0.0, lcm=lcf, minimum=min_rate * subject, maximum=max_rate * subject)
    add(ex, "retro_premium_mean_via_terms", float(rt.phi(x) @ p), "RetroTerms(basic=0, lcm=lcf, minimum, maximum)")
    add(ex, "pr_min_binds", float(p[lcf * x <= min_rate * subject].sum()))
    add(ex, "pr_max_binds", float(p[lcf * x >= max_rate * subject].sum()))
    # ---- swing rating, designed on the same layer loss -----------------------
    ex = "variable_swing"
    sw = dict(basic=300.0, lcm=1.25, minimum=360.0, maximum=1200.0)
    t1 = Underwriter._make_feature_terms("swing", dict(sw), share=1.0)
    add(ex, "basic", sw["basic"]); add(ex, "lcm", sw["lcm"])
    add(ex, "minimum", sw["minimum"]); add(ex, "maximum", sw["maximum"])
    add(ex, "share100_premium_mean", float(t1.phi(x) @ p), "swing premium clip(300+1.25 L, 360, 1200), share 1")
    t4 = Underwriter._make_feature_terms("swing", dict(sw), share=0.4)
    add(ex, "share40_premium_mean", float(t4.phi(0.4 * x) @ p),
        "ceded loss at the 40% placed share is 0.4 L; basic, min and max scale by 0.4, lcm does not")
    add(ex, "share40_basic_scaled", 0.4 * sw["basic"])
    add(ex, "share40_minimum_scaled", 0.4 * sw["minimum"])
    add(ex, "share40_maximum_scaled", 0.4 * sw["maximum"])
    add(ex, "share100_pr_min_binds", float(p[sw["basic"] + sw["lcm"] * x <= sw["minimum"]].sum()))
    add(ex, "share100_pr_max_binds", float(p[sw["basic"] + sw["lcm"] * x >= sw["maximum"]].sum()))
    # ---- profit commission (Treaty 5) ----------------------------------------
    ex = "variable_pc"
    a17 = build(
        "agg Re:BN2p [2000 2000 2000] exposure at [.1 .14 .21] rate 700 xs 0 "
        "sev 300 * pareto [1.5 1.3 1.1] - 300 poisson",
        normalize=False,
    )
    port = build(
        "port Treaty.5 agg Year.1 agg.Re:BN2p agg Year.2 agg.Re:BN2p agg Year.3 agg.Re:BN2p",
        bs=1 / 4, normalize=False,
    )
    pd_ = port.density_df.set_index("loss").p_total
    add(ex, "bs", port.bs); add(ex, "log2", port.log2)
    summ(ex, "three_year_loss", pd_, port.bs, "three independent years of the 700 xs 300 layer, Poisson, 100% share")
    comp_counts(ex, a17, "BN2 Poisson, one year")
    add(ex, "one_year_mean", a17.est_m, "single year, bs chosen by aggregate for Re:BN2p")
    x3 = pd_.index.to_numpy(float)
    p3 = pd_.to_numpy(float)
    subj, coins, re_rate, share, allow = 18000.0, 0.2, 0.25, 0.25, 0.2
    ced_prem = re_rate * subj
    add(ex, "subject_premium", subj)
    add(ex, "ceded_premium", ced_prem, "treaty rate 25% of subject premium")
    add(ex, "coinsurance", coins, "reinsurer takes 80% of the layer")
    lr = x3 * (1 - coins) / ced_prem
    pcr = np.maximum(0.0, share * (1 - allow - lr))
    add(ex, "ceded_lr_mean", float(lr @ p3))
    add(ex, "pc_rate_mean", float(pcr @ p3), "docs: E[0.25 * max(0, 1 - 0.2 - LR)], per unit of ceded premium")
    add(ex, "pc_amount_mean", float(pcr @ p3) * ced_prem)
    pct = ProfitCommissionTerms(share=share, allowance=allow)
    add(ex, "pc_rate_mean_via_terms", float(pct.phi(lr) @ p3), "ProfitCommissionTerms(share=.25, allowance=.2).phi(LR)")
    add(ex, "pc_at_expected_lr", float(pct.phi(float(lr @ p3))), "Jensen: pc at the expected LR (0.08)")
    add(ex, "pr_pc_positive", float(p3[lr < 1 - allow].sum()))
    # ---- sliding scale commission (Treaty 6) ---------------------------------
    ex = "variable_slide"
    a18 = build(
        "agg Re.BN6 25000 exposure at 0.1 rate 900 xs 0 sev 100 * pareto 1.05 - 100 mixed gamma 0.095",
        normalize=False,
    )
    grid(ex, a18)
    d6 = agg_density(a18)
    x6 = d6.index.to_numpy(float)
    p6 = d6.to_numpy(float)
    summ(ex, "ceded_loss", d6, a18.bs, "900 xs 100 layer on 25000 subject premium, gamma-mixed Poisson cv 0.095")
    add(ex, "claim_count_mean", a18.n)
    add(ex, "gross_mean_exact", a18.stats_df.loc[("agg", "mean"), "mixed"])
    add(ex, "gross_cv_exact", a18.stats_df.loc[("agg", "cv"), "mixed"])
    re_prem = 25000.0 * 0.2
    add(ex, "ceded_premium", re_prem, "25000 subject premium * 20% rate")
    anchors = ((0.40, 0.35), (0.25, 0.55), (0.20, 0.65))
    st = SlideTerms(anchors=anchors)
    lr6 = x6 / re_prem
    add(ex, "slide_commission_rate_mean", float(st.phi(lr6) @ p6), "E[phi(LR)], anchors 40% at 35, 25% at 55, 20% at 65")
    add(ex, "slide_commission_amount_mean", float(st.phi(lr6) @ p6) * re_prem)
    add(ex, "ceded_lr_mean", float(lr6 @ p6))
    add(ex, "slide_commission_at_expected_lr", float(st.phi(float(lr6 @ p6))), "phi at E[LR] (Jensen)")
    add(ex, "pr_lr_below_35", float(p6[lr6 <= 0.35].sum()))
    add(ex, "pr_lr_above_65", float(p6[lr6 >= 0.65].sum()))
    f = lambda y: 0.4 - 0.75 * np.minimum(0.2, np.maximum(0, y - 0.35)) - 0.5 * np.minimum(0.1, np.maximum(0, y - 0.55))
    add(ex, "slide_commission_rate_mean_by_hand", float(f(lr6) @ p6),
        "0.4 - 0.75 (.2 xs .35) - 0.5 (.1 xs .55): the slopes the docs' text states (0.75:1 and 0.5:1)")
    # the docs' code uses shares 0.25 and 0.5 in make_ceder_netter (slope 0.25 on the first segment): record it
    fd = lambda y: 0.4 - 0.25 * np.minimum(0.2, np.maximum(0, y - 0.35)) - 0.5 * np.minimum(0.1, np.maximum(0, y - 0.55))
    add(ex, "slide_commission_rate_mean_docs_code", float(fd(lr6) @ p6),
        "what the docs' make_ceder_netter([(0.25,.2,.35),(0.5,.1,.55)]) computes; slope 0.25 not 0.75, a docs inconsistency, NOT the stated terms")


# ---------------------------------------------------------------------------
# Example 6: risk_pricing (distortions at a capital standard)
# ---------------------------------------------------------------------------
def ex_risk_pricing():
    a = retester("occurrence net of " + layer_decl(PROG))
    COC, P = 0.10, 0.99
    for ex, view in [("risk_pricing_gross", "gross"), ("risk_pricing_ceded", "ceded")]:
        r = a.calibrate_distortions(COC, p=P, reins_view=view)
        grid(ex, a)
        add(ex, "coc", COC, "target cost of capital (P - L) / Q")
        add(ex, "capital_p", P, "VaR probability fixing assets, a = q(p) (lower)")
        c = r.calibration_df.iloc[0]
        for k in ["F(a)", "L", "M", "P", "Q", "a", "LR", "PQ", "ROE"]:
            add(ex, f"cal_{k}", c[k], "calibration_df: L=E[min(X,a)], M=P-L, Q=a-P")
        for name in ["ccoc", "ph", "wang", "dual", "tvar"]:
            col = r.distortion_df.loc[name]
            add(ex, f"param_{name}", col["param"], f"{col['param_name']}; calibrated so the premium equals P")
            add(ex, f"error_{name}", col["error"], "aggregate's Newton premium error")
            add(ex, f"gini_p_{name}", col["gini_p"])
        # premium by each distortion at assets a (independent of calibrating)
        dens = reins_df(a)[{"gross": "p_agg_gross", "ceded": "p_agg_ceded_occ"}[view]]
        aa = float(c["a"])
        for name, d in r.distortions.items():
            pr = d.price(dens, a=aa, kind="both")
            add(ex, f"price_ask_{name}", pr.ask, f"Distortion.price(ser, a={aa}); equals calibration P")
            add(ex, f"price_bid_{name}", pr.bid, "bid price, dual distortion")
            add(ex, f"price_el_{name}", pr.el, "E[min(X,a)]")
        # fixed-parameter (uncalibrated) distortions, same assets
        for name, d in [
            ("ph0.7", Distortion("ph", a=0.7)),
            ("wang0.4", Distortion("wang", lam=0.4)),
            ("dual1.8", Distortion("dual", b=1.8)),
            ("tvar0.3", Distortion("tvar", p=0.3)),
            ("ccoc0.1", Distortion("ccoc", r=0.1)),
        ]:
            pr = d.price(dens, a=aa, kind="ask")
            add(ex, f"fixed_price_ask_{name}", pr.ask, f"fixed parameter, assets {aa}")
        # unlimited-assets premium of the calibrated distortions
        for name, d in r.distortions.items():
            if name == "ccoc":
                continue
            pr = d.price(dens, a=np.inf, kind="ask")
            add(ex, f"unlimited_price_ask_{name}", pr.ask, "assets = infinity (all outcomes)")




def main():
    ex_tower_gross()
    ex_tower_program()
    ex_agg_limit_reinst()
    ex_stop_loss()
    ex_variable_features()
    ex_risk_pricing()
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["example", "quantity", "value", "note"])
        w.writerows(ROWS)
    print(f"aggregate {aggregate.__version__}: wrote {OUT}: {len(ROWS)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
