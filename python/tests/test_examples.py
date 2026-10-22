"""The worked notebooks in examples/ run, and what their prose says holds.

Each notebook's code cells run in order in one namespace, without Jupyter. Chart cells (tag ``plot``) run
too, under matplotlib's Agg backend, so a broken chart fails here and not only in the docs site's build.
The numbers and claims a notebook's markdown states are checked against what its cells compute (#269).
"""

import csv
import json
import math
import os
import subprocess
import warnings
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def cells_of(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def run_notebook(path, after_cell=None):
    """Run every code cell of a notebook in one namespace and return it.

    Chart cells draw with the Agg backend and must leave at least one figure; the figures are closed after
    each. ``after_cell(ns)``, if given, runs after each code cell.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ns = {}
    with warnings.catch_warnings():
        # plt.show() on a non-interactive backend warns that it cannot show anything.
        warnings.filterwarnings("ignore", message=".*non-interactive.*", category=UserWarning)
        for cell in cells_of(path):
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), f"{path.name}", "exec"), ns)
            if "plot" in cell["metadata"].get("tags", []):
                assert plt.get_fignums(), f"{path.name}: a plot cell drew no figure"
                plt.close("all")
            if after_cell is not None:
                after_cell(ns)
    return ns


def prose(path):
    """A notebook's markdown cells, joined."""
    return "\n".join("".join(c["source"]) for c in cells_of(path) if c["cell_type"] == "markdown")


def sci(x, digits=1):
    """A number as the prose writes it in scientific notation: 1.5e-6, 6e-10."""
    mantissa, exponent = f"{abs(x):.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def assert_says(text, *quotes):
    # Each quote is a passage of the prose with the computed values put in at the prose's own precision,
    # so a value that drifts, or prose edited away from the value, fails here.
    for quote in quotes:
        assert quote in text, f"the notebook no longer says: {quote!r}"


@pytest.mark.parametrize("name, lookup", [
    ("one_year_view.ipynb", "DATA = next("),
    ("reinsurance_pricing.ipynb", "REFERENCE = next("),
    ("submission_evaluation.ipynb", "DATA = next("),
])
def test_data_folder_missing_is_a_clear_error(name, lookup, monkeypatch, tmp_path):
    # A reader who copies a notebook out of the repository gets an error that says what is missing and
    # where to run it, not a bare StopIteration.
    monkeypatch.chdir(tmp_path)
    ns = {}
    with pytest.raises(FileNotFoundError, match="inside a clone of the repository"):
        for cell in cells_of(EXAMPLES / name):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, name, "exec"), ns)
                assert lookup not in source, f"{name}: the lookup cell ran without an error"


def test_one_year_view(monkeypatch, capsys):
    # At the published count of simulations (20,000, a few seconds) and the notebook's seeds, so the claims
    # are checked on the numbers the docs site shows. Every inequality below also holds, with wide margins,
    # at 4,000 simulations, so none rests on simulation noise.
    monkeypatch.delenv("PROSPICIO_EXAMPLE_SIMS", raising=False)
    monkeypatch.chdir(EXAMPLES)
    ns = run_notebook(EXAMPLES / "one_year_view.ipynb")
    assert "reinsurance cuts the company's TVaR" in capsys.readouterr().out
    assert ns["N_SIMS"] == 20000
    text = prose(EXAMPLES / "one_year_view.ipynb")
    # Simulation agrees with exposure rating; the allocation adds up.
    assert_says(text, "The simulated cessions agree with exposure rating")
    surplus = ns["surplus_loss"].mean()
    assert surplus == pytest.approx(ns["profile"].expected_surplus_loss(2e6, 4.0), rel=0.05)
    assert ns["xl_loss"].mean() == pytest.approx(ns["xl_cost"], rel=0.05)
    for a in ns["allocations"].values():
        assert sum(a.allocated) == pytest.approx(a.total, rel=1e-9)
    assert ns["saved"] > 0

    # "More lines cut the net tail."
    assert_says(text, "More lines cut the net tail")
    Tower, Layer = ns["Tower"], ns["Layer"]
    tails = [Tower.inuring([[Layer.surplus("surplus", ns["retention"], more)], [ns["xl"]]])
             .apply(ns["events"]).marginal(("net", "retained")).quantile(0.995) for more in [4.0, 8.0, 12.0]]
    assert tails == sorted(tails, reverse=True) and tails[0] > 1.5 * tails[-1], tails

    # "Reading the result". The allocations are by line: reserve, motor, property.
    gross, net = ns["allocations"]["gross"], ns["allocations"]["net"]
    assert [k[0] for k in ns["companies"]["gross"].components()] == ["reserve", "motor", "property"]
    (g_reserve, g_motor, g_property), (n_reserve, n_motor, n_property) = gross.allocated, net.allocated
    # Reserve and motor dominate the tail: together they take most of TVaR gross and net, and they are the
    # larger risks, correlated with each other (0.4, against 0.1 with property). Gross, property's CoTVaR is
    # above reserve's alone, so the claim holds for the two together.
    assert_says(text, "Reserve and motor dominate the tail. They are larger and correlated with each other.")
    assert g_reserve + g_motor > 0.6 * gross.total and n_reserve + n_motor > 0.8 * net.total
    assert min(ns["reserve"].mean(), ns["motor"].mean()) > 2 * ns["gross"].mean()
    assert ns["corr"][0][1] > max(ns["corr"][0][2], ns["corr"][1][2])
    # Reinsurance cuts property's share sharply, and the company's TVaR falls by less than property's.
    assert_says(text, "Reinsurance cuts property's share of TVaR sharply. The company's TVaR falls by less than that")
    assert n_property / net.total < 0.5 * g_property / gross.total
    assert ns["saved"] == pytest.approx(gross.total - net.total, rel=1e-12)
    assert ns["saved"] < g_property - n_property
    # Reserve and motor take more of the net TVaR than of the gross, as shares and as amounts. This rests on
    # how the notebook models reserve risk and the reinsurance, which #266 to #268 may change.
    assert_says(text, "Reserve and motor take *more* of the net TVaR than of the gross.")
    assert n_reserve / net.total > g_reserve / gross.total and n_motor / net.total > g_motor / gross.total
    assert n_reserve > g_reserve and n_motor > g_motor


def test_reinsurance_pricing(monkeypatch, capsys):
    monkeypatch.chdir(EXAMPLES)
    # Every distribution compare_distribution sees, to measure the quantiles' distance from a tie below.
    compared = []

    def record_grids(ns):
        compare = ns.get("compare_distribution")
        if compare is not None and not getattr(compare, "recording", False):
            def recording(section, name, example, prefix, grid, bs, share=False, **kw):
                compared.append((f"{section}: {name}", grid, share))
                return compare(section, name, example, prefix, grid, bs, share=share, **kw)

            recording.recording = True
            ns["compare_distribution"] = recording

    ns = run_notebook(EXAMPLES / "reinsurance_pricing.ipynb", after_cell=record_grids)
    assert "comparisons are inside their tolerance" in capsys.readouterr().out
    # Every comparison with aggregate's numbers is inside its stated tolerance.
    comparisons = ns["COMPARISONS"]
    assert len(comparisons) > 300
    assert [c["name"] for c in comparisons if not c["ok"]] == []
    # A few values, read from the reference file here and not through the notebook's helpers.
    root = EXAMPLES.parent / "validation" / "reference" / "re_pricing_aggregate.csv"
    with open(root) as f:
        ref = {(r["example"], r["quantity"]): float(r["value"]) for r in csv.DictReader(f)}
    assert ns["gross"].mean() == pytest.approx(ref["tower_gross", "gross_mean"], rel=1e-9)
    assert ns["gross"].quantile(0.99) == ref["tower_gross", "gross_q0.99"]
    assert ns["net"].mean() == pytest.approx(ref["tower_program", "net_mean"], rel=1e-9)
    assert ns["limited"].mean() == pytest.approx(ref["agg_limit_reinst", "agglimit2800_ceded_mean"], rel=1e-7)
    assert ns["after_aad"].mean() == pytest.approx(ref["stop_loss_aad", "after_aad_mean"], rel=1e-9)
    assert ns["retro_grids"].expected_premium[0] == pytest.approx(
        ref["variable_retro", "retro_premium_mean"], rel=1e-9)
    assert ns["commission"] == pytest.approx(ref["variable_pc", "pc_amount_mean"], rel=1e-9)
    assert ns["slide_commission"] == pytest.approx(ref["variable_slide", "slide_commission_amount_mean"], rel=1e-9)
    assert ns["priced"]["gross", "ph"].measure(ns["gross"]) == pytest.approx(
        ref["risk_pricing_gross", "unlimited_price_ask_ph"], rel=1e-6)
    # The exact closed forms the grid means approach.
    assert ns["severity"].lev(5000.0) * 10 == pytest.approx(ref["tower_gross", "gross_mean_exact"], rel=1e-12)

    # The numbers the prose quotes. Nothing in this notebook is simulated: every number is a deterministic
    # FFT on the same grid as aggregate's, so each is checked to the digits the prose prints it with, and a
    # change in the library that moves one fails here instead of leaving the prose stale. The tolerances in
    # the notebook were set after a first run; checking the worst difference observed in each class against
    # the prose (below) means a drift towards a tolerance shows up here, not only when it crosses it.
    text = prose(EXAMPLES / "reinsurance_pricing.ipynb")
    agg = ns["AGG"]
    diff = {(r["section"], r["name"]): r["diff"] for r in comparisons}
    severity, sev_grid, bs1 = ns["severity"], ns["sev_grid"], ns["BS1"]
    gross = ns["gross"]
    exact = 10 * severity.lev(5000.0)
    gap = 1.0 - gross.mean() / exact
    step = ns["size"].step

    # Section 1.
    reaches = 1.0 - sev_grid.cdf(0.0)
    conditional = sev_grid.layer(250.0, 0.0) / reaches
    assert_says(
        text,
        f"The mean on the grid ({gross.mean():.2f}) is {gap:.3%} below the exact mean ({exact:.2f})",
        f"pick {step:g} where `aggregate`, which sees the 5,000 limit, picks {agg['tower_gross', 'bs']:g}",
        f"the lognormal puts {severity.cdf(bs1 / 2):.1%} of claims below {bs1 / 2:g}, half a bucket",
        f"The {conditional:.1f} in the last column is therefore the mean over the {reaches:.1%} of claims that "
        f"survive rounding; over all claims the exact mean claim in the layer is {severity.layer(250.0, 0.0):.1f}",
        f"`aggregate` reports the same {agg['tower_gross', 'layer_250x0_pr_attach_grid']:.1%} and "
        f"{agg['tower_gross', 'layer_250x0_cond_sev_mean']:.1f}",
        "`Tower.on_grid` says `on_points` is `False`",
    )
    assert ns["tower_grids"].on_points is False

    # Section 2: ceded and net add up in the mean, not in the standard deviation.
    ceded, net = ns["program_ceded"], ns["net"]
    sd_gross, sd_ceded, sd_net = (math.sqrt(g.variance()) for g in (gross, ceded, net))
    assert ceded.mean() + net.mean() == pytest.approx(gross.mean(), rel=1e-9)
    assert sd_ceded + sd_net > sd_gross
    share_q90 = diff["2 program", "80% of 1000 xs 1000 q0.9"]
    assert share_q90 == pytest.approx(0.2)  # "a fifth of a bucket"
    assert_says(
        text,
        f"({gross.mean():.2f} = {ceded.mean():.2f} + {net.mean():.2f})",
        f"{sd_ceded:.1f} + {sd_net:.1f} = {sd_ceded + sd_net:,.1f} is more than the gross {sd_gross:,.1f}",
        f"The standard deviation differs by {sci(diff['2 program', '80% of 1000 xs 1000 sd'])}, the skewness by "
        f"{sci(diff['2 program', '80% of 1000 xs 1000 skew'])}, and the 90% quantile by {share_q90 * bs1:.1f}, "
        "a fifth of a bucket",
    )

    # Section 3: the same recovery under both schedules; the totals at the designed deposit and at 65%.
    section3 = [v for (s, name), v in diff.items() if s == "3 limit and reinstatements"
                and (name.endswith("expected reinstatement premium") or name.endswith("recovery"))]
    assert len(section3) == 4 and max(section3) <= 1e-8  # "agree to 1e-8 or better"
    (k_a, recovery), (k_b, recovery_b) = ns["per_deposit"]["A"], ns["per_deposit"]["B"]
    assert recovery == recovery_b == pytest.approx(ns["limited"].mean(), rel=1e-12)
    deposit, target = ns["DEPOSIT"], ns["TARGET_LR"]
    total_a, total_b = deposit * (1 + k_a), deposit * (1 + k_b)
    deposit_a, deposit_b = (recovery / target / (1 + k) for k in (k_a, k_b))
    assert deposit_a < deposit_b  # "A needs the smaller deposit"
    assert deposit_a * (1 + k_a) == pytest.approx(deposit_b * (1 + k_b), rel=1e-12)
    assert_says(
        text,
        f"The recovery under schedule A or B is the same {recovery:.2f}",
        f"schedule A collects {total_a:,.2f} against an expected recovery of {recovery:.2f} "
        f"(a {recovery / total_a:.0%} loss ratio) and B collects {total_b:,.2f} ({recovery / total_b:.0%})",
        f"Priced to the same {target:.0%} loss ratio, both collect {deposit_a * (1 + k_a):,.2f} in expectation",
        f"keeps the {sci(ns['report3'].tail_mass, 0)} of probability beyond the grid's top",
        f"shifts the result by {sci(diff['3 limit and reinstatements', 'loss above the limit'], 0)} relative",
    )

    # Section 4.
    gross4, absorbed = ns["gross4"].mean(), ns["absorbed"]
    assert_says(
        text,
        f"(an expected loss ratio of {gross.mean() / 1000:.0%})",
        f"`Grid.lev(360)`: {absorbed:.2f}, a loss elimination ratio of {absorbed / gross4:.1%} of the "
        f"{gross4:.0f} expected loss. The other {ns['after_aad'].mean() / gross4:.1%}",
        f"entry ratio of 360 / {gross4:.0f} = {360 / gross4:.1f}",
    )

    # Section 5: Jensen's gap for the profit commission and the sliding scale.
    premium_b, premium_c, lr_probs = ns["PREMIUM5B"], ns["PREMIUM5C"], ns["lr_probs"]
    scale, middle = ns["scale"], ns["middle"]
    below = sum(q * (scale(x) - middle(x)) for x, q in lr_probs if x < 0.35)
    above = sum(q * (scale(x) - middle(x)) for x, q in lr_probs if x > 0.55)
    assert above > -below > 0  # the years above 55% add more than the years below 35% take off
    assert_says(
        text,
        f"so its average ({ns['commission'] / premium_b:.2%} of premium) is above its value at the expected loss "
        f"ratio ({ns['pc_layer'].profit_commission_for(ns['pc_grids'].ceded[0].mean()) / premium_b:.2%})",
        f"Its average ({ns['slide_commission'] / premium_c:.1%}) is above its value at the expected loss ratio "
        f"({ns['at_expected']:.1%})",
        f"the {sum(q for x, q in lr_probs if x < 0.35):.1%} of years below 35% take off",
        f"a commission of {agg['variable_slide', 'slide_commission_rate_mean_docs_code']:.1%}, not "
        f"{ns['slide_commission'] / premium_c:.1%}",
    )

    # Section 6: aggregate's Newton iteration stops at a premium error, which shows at the 1e-9 level.
    def g_rows(view, family, points=(0.05, 0.5)):
        return max(diff[f"6 {view} view", f"{family}: g({s}) at aggregate's parameter"] for s in points)

    assert_says(
        text,
        f"({sci(agg['risk_pricing_gross', 'error_tvar'])} for the gross view's TVaR)",
        f"The TVaR parameter differs by {sci(diff['6 gross view', 'tvar: parameter p'])} on the gross view and "
        f"{sci(diff['6 ceded view', 'tvar: parameter p'])} on the ceded view, and its `g` rows by "
        f"{sci(g_rows('gross', 'tvar'))} and {sci(g_rows('ceded', 'tvar'))}",
        f"its parameter differs by {sci(diff['6 ceded view', 'ph: parameter rho'])} and its `g(0.05)` by "
        f"{sci(g_rows('ceded', 'ph', (0.05,)))}",
    )

    # Where and why the numbers differ, and the worst difference in each tolerance class.
    independent = [r for r in comparisons if not r["check"]]

    def worst(cls):
        return max((r for r in independent if r["cls"] == cls), key=lambda r: r["diff"])

    assert worst("mean")["diff"] < 1e-7 and worst("sd")["diff"] < 1e-7  # "far inside" 1e-6
    assert worst("quantile")["diff"] == 0
    assert worst("share quantile")["diff"] == 0.5  # "at most half a bucket"
    assert worst("share sd")["name"] == "50% of 1500 xs 1000 sd"
    assert worst("skew")["name"] == "50% of 1500 xs 1000 skew"
    share_skews = [r["diff"] for r in independent if r["cls"] == "skew" and "%" in r["name"]]
    # The quantiles of the shared grid: how far the cdf is from a tie at each one compared.
    tie = min(min(grid.cdf(grid.quantile(p)) - p, p - grid.cdf(grid.quantile(p) - grid.step))
              for _, grid, share in compared if not share for p in (0.5, 0.9, 0.99, 0.999))
    assert len(compared) >= 25 and tie > 0
    layer = ns["tower_grids"].ceded[1]  # 250 xs 250
    central = agg["tower_gross", "layer_250x250_ceded_skew_central"]
    kept = [(i * layer.step, q) for i, q in enumerate(layer.probs) if i * layer.step < 10000.0]
    mean = sum(x * q for x, q in kept)
    var = sum((x - mean) ** 2 * q for x, q in kept)
    skew_below = sum((x - mean) ** 3 * q for x, q in kept) / var**1.5
    assert abs(skew_below / central - 1) < 1e-13  # "brings it under 1e-13"
    quoted = ns["quoted"]  # aggregate's quoted skew against its density's, largest difference first
    for (example, quantity), _ in quoted[:4]:
        # "the density's rounding deficit (`1 - sum(p)`, of order 1e-15 to 1e-14)"
        assert 1e-16 < agg[example, quantity.replace("_skew", "_density_deficit")] < 1e-13, (example, quantity)
    assert_says(
        text,
        f"at none of those compared is the cdf within {sci(tie)} of a tie",
        f"recommend a step of {step:g}, where `aggregate`, which sees the policy limit, uses "
        f"{agg['tower_gross', 'bs']:g}",
        f"the cdf is at least {sci(tie)} away from the probability",
        f"{gross.mean():.2f} against an exact {exact:.2f} in section 1, {gap:.3%} low, and "
        f"{ns['tower_grids'].ceded[0].mean():.2f} against {10 * severity.layer(250.0, 0.0):.2f} for the layer 250 xs 0",
        f"the standard deviation differs by {sci(diff['2 program', '80% of 1000 xs 1000 sd'])} for the 80% share "
        f"and {sci(worst('share sd')['diff'])} for the 50% stop-loss share, the skewness by up to "
        f"{sci(max(share_skews))}, and a quantile by at most half a bucket",
        f"({sci(worst('tail')['diff'])} observed)",
        f"({ns['POINTS'] * bs1:,.0f} for a bucket of {bs1:g})",
        f"the layer 250 xs 250 differs by {sci(ns['whole'] / central - 1)};",
        f"differs again, by up to {sci(quoted[0][1])}",
        f"The worst difference observed, {sci(worst('skew')['diff'])}, is the 50% stop-loss share",
    )

    # A quantile matches "the same grid point" with a tolerance of 0 buckets: the two packages' grid points
    # i * bs are equal floats only because every bucket size is a power of two, so i * bs is exact in both.
    # A bucket that is not (0.1, say) would need a tolerance of a few ulps instead.
    buckets = {v for (_, quantity), v in agg.items() if quantity == "bs"}
    buckets |= {ns[name] for name in ("BS1", "BS3", "BS4", "BS5", "BS5B", "BS5C")}
    assert all(math.frexp(b)[0] == 0.5 for b in buckets), sorted(buckets)


def test_submission_evaluation(monkeypatch, capsys):
    monkeypatch.setitem(os.environ, "PROSPICIO_EXAMPLE_SIMS", "4000")
    monkeypatch.chdir(EXAMPLES)
    ns = run_notebook(EXAMPLES / "submission_evaluation.ipynb")
    capsys.readouterr()
    with open(EXAMPLES / "data" / "answer_key.json") as f:
        key = json.load(f)

    # The files reconcile, and what is read from them agrees with how they were generated.
    assert [name for name, ok in ns["checks"] if not ok] == []
    assert ns["onlevel"][2025] == pytest.approx(key["onlevel_premium"]["2025"], rel=0.005)
    assert ns["exposure_trend"].annual_change == pytest.approx(key["exposure_growth"], abs=0.002)
    assert ns["SEVERITY_TREND"] == pytest.approx(key["severity_trend"], abs=0.005)

    # The Pareto fit allows for truncation and censoring: it lands within two standard errors of the truth,
    # and the fit that ignores the per-claim reporting thresholds is far from it.
    from prospicio.distributions import Pareto

    se = ns["ALPHA"] / math.sqrt(ns["n_uncensored"])
    assert abs(ns["ALPHA"] - key["alpha"]) < 2 * se
    ignoring = Pareto.fit(ns["losses"], ns["T"], censored=ns["censored"]).alpha
    assert ns["ALPHA"] - ignoring > 0.2

    # Layer losses: the blend lies between its two inputs, with credibility in (0, 1), and the grid reproduces it.
    for z, e, x, b, grid in zip(ns["weights"], ns["experience_loss"], ns["exposure_loss"], ns["blended"],
                                ns["grids"].ceded):
        assert 0.0 < z < 1.0
        assert min(e, x) <= b <= max(e, x)
        assert grid.mean() == pytest.approx(b, rel=5e-3)
    # The simulated program agrees with the grid for the working layer.
    assert ns["simulated"].marginal(["ceded", ns["NAMES"][0]]).mean() == pytest.approx(
        ns["grids"].ceded[0].mean(), rel=0.15)

    # Technical premiums: loss plus a capital charge, so above the expected loss; the one-function chain
    # reproduces the step-by-step result.
    for row, base_row in zip(ns["priced"], ns["base"]["rows"]):
        assert row["technical"] > row["loss"]
        assert base_row["technical"] == pytest.approx(row["technical"], rel=1e-7)
    # Prices fall with the attachment as a rate on line. The case is built so that the decision differs by
    # layer: quote the working layer, counter the middle one, refer the top one to the portfolio view.
    rols = [row["rol"] for row in ns["priced"]]
    assert rols == sorted(rols, reverse=True)
    assert [verdict.split()[0] for verdict in ns["closing"]] == ["quote", "counter", "refer"]

    # Against the generator's truth: the blended layer losses, and the technical premium they lead to.
    for name, blended in zip(ns["NAMES"], ns["blended"]):
        truth = key["layer_expected_loss_2026_occurrence"][name]
        assert blended == pytest.approx(truth, rel=0.10 if name == "5m xs 5m" else 0.25), name
    for row, true_row in zip(ns["priced"], ns["true_rows"]):
        assert row["technical"] == pytest.approx(true_row["technical"], rel=0.10), row["name"]


def test_reading_path(monkeypatch, capsys):
    monkeypatch.chdir(EXAMPLES)
    ns = run_notebook(EXAMPLES / "reading_path.ipynb")
    assert "indicated change" in capsys.readouterr().out
    # Station 1: the fits land near the model that drew the claims.
    assert ns["severity"].mean() == pytest.approx(ns["TRUE_SEVERITY"].mean(), rel=0.05)
    assert ns["yearly_mean"] == pytest.approx(400.0, rel=0.1)
    # Station 2: the FFT, the compound moments and the simulation agree.
    assert ns["annual"].mean() == pytest.approx(ns["moments"].mean, rel=1e-6)
    assert ns["simulated"].mean() == pytest.approx(ns["annual"].mean(), rel=0.01)
    # Station 3: the grid reproduces the exposure rate; the blend lies between experience and exposure.
    assert ns["layer_annual"].mean() == pytest.approx(ns["exposure_loss"], rel=0.01)
    assert 0.0 < ns["z"] < 1.0
    assert min(ns["experience_loss"], ns["exposure_loss"]) <= ns["selected"] <= max(
        ns["experience_loss"], ns["exposure_loss"])
    assert ns["pentagon"].premium > ns["pentagon"].loss
    # Station 4: the indication is the permissible loss ratio's ratio.
    assert 1.0 + ns["change"] == pytest.approx(
        (ns["experience"].loss_ratio + ns["FIXED"]) / (1.0 - 0.20 - 0.05), rel=1e-12)
    # Station 5: Mack and the bootstrap agree on the reserve and roughly on its spread.
    assert ns["mack"].total_reserve == pytest.approx(ns["cl"].total_reserve, rel=1e-12)
    assert ns["boot"].mean() == pytest.approx(ns["mack"].total_reserve, rel=0.03)
    assert math.sqrt(ns["boot"].variance()) == pytest.approx(ns["mack"].total_standard_error, rel=0.25)
    # Station 6: diversification, and every distortion's allocation adds up to the same total margin.
    book = ns["book"]
    assert book.assets(0.99) < ns["alone"]
    margins = []
    for family in ns["FAMILIES"]:
        priced = book.price(book.calibrate(family, p=0.99, return_on_capital=0.10), p=0.99)
        assert sum(u.margin for u in priced.allocated) == pytest.approx(priced.total.margin, rel=1e-9)
        margins.append(priced.total.margin)
    assert max(margins) == pytest.approx(min(margins), rel=1e-6)


def test_submission_files_match_generator(tmp_path):
    # The committed files are what examples/data/make_submission.py writes: the CSV files byte for byte (a
    # checkout under core.autocrlf may turn their LF into CRLF), the answer key to rounding, since its
    # full-precision floats come through libm and could differ in the last digit between platforms.
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_submission", EXAMPLES / "data" / "make_submission.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.OUT = tmp_path / "submission"
    generator.KEY = tmp_path / "answer_key.json"
    generator.main()
    committed = EXAMPLES / "data" / "submission"
    names = sorted(p.name for p in generator.OUT.iterdir())
    assert names == sorted(p.name for p in committed.iterdir())
    for name in names:
        assert (generator.OUT / name).read_bytes() == (committed / name).read_bytes().replace(b"\r\n", b"\n"), name

    def close(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
        if isinstance(a, float):
            return a == pytest.approx(b, rel=1e-9)
        return a == b

    with open(generator.KEY) as f, open(EXAMPLES / "data" / "answer_key.json") as g:
        assert close(json.load(f), json.load(g))


def staged(path):
    """The notebook as git stores it: the index copy, after the nbstripout
    filter, so a local run that left outputs in the working tree does not fail
    this check. Falls back to the file outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "show", f":{path.relative_to(EXAMPLES.parent).as_posix()}"],
            cwd=EXAMPLES.parent, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        out = path.read_bytes()
    return json.loads(out.decode("utf-8"))


def test_notebooks_are_committed_without_outputs():
    # Results are published on the docs site, which executes the notebooks
    # (`cargo xtask docs`). The repository keeps code only, so re-running a
    # notebook changes nothing git tracks (the nbstripout filter, .gitattributes).
    paths = sorted(EXAMPLES.glob("*.ipynb"))
    assert paths
    for path in paths:
        nb = staged(path)
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code, path.name
        assert all(c["outputs"] == [] for c in code), f"{path.name} is committed with outputs"
        assert all(c["execution_count"] is None for c in code), f"{path.name} has execution counts"
        assert "version" not in nb["metadata"].get("language_info", {}), path.name
