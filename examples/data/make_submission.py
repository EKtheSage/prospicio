"""Generate the synthetic reinsurance submission that examples/submission_evaluation.ipynb reads.

Run from anywhere:  python examples/data/make_submission.py
It needs numpy only, and writes small CSV files to examples/data/submission/
and the answer key to examples/data/answer_key.json. The output is the same every run (seed below), so
the files are committed and the notebook does not need this script, except
that it is how you change the case.

The story. A commercial property insurer (the cedant) asks for a quote on its
2026 per-risk excess of loss program: three layers, 5m xs 5m, 10m xs 10m and
20m xs 20m, losses occurring during 2026, each with paid reinstatements. The
submission has the files a broker sends:

  subject_premium.csv   earned subject premium by calendar year, 2016 to 2025
  rate_changes.csv      the cedant's average rate changes, by effective date
  cost_index.csv        a published building cost index, quarterly (the trend source)
  large_losses.csv      every claim with incurred over 500,000, valued at each
                        year end since it was reported (paid and incurred)
  triangle.csv          the whole book's paid and incurred triangle, by accident year
  risk_profile.csv      the 2026 portfolio by band of sum insured
  treaty_terms.csv      the proposed layers and the broker's quoted rates on line
  treaty_program.csv    the treaty period, basis, estimated premium, brokerage

answer_key.json is NOT part of the submission, so it sits outside the
submission's folder. It holds the parameters that generated the data, so the
notebook can compare its estimates with the truth at the end.

The TRUE dictionary below is the whole model of the world. Money is in US
dollars. Losses are generated at 2026 prices and scaled back to the price
level of their accident date with the cost index trend.
"""

import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "submission"
KEY = Path(__file__).resolve().parent / "answer_key.json"

TRUE = {
    "seed": 20261013,
    "first_year": 2016,
    "last_year": 2025,  # the last accident year; valuation is 31 December 2025
    "treaty_year": 2026,
    "price_reference": 2026.5,  # the middle of the treaty year
    # Exposure grows 3% a year; the cedant's rate changes come on top.
    "exposure_growth": 0.03,
    "onlevel_premium_2026": 180_000_000.0,  # premium at the current rate level, 2026
    # Claims (and sums insured) inflate at the cost index trend.
    "severity_trend": 0.045,
    # Losses above 1,000,000 at 2026 prices: a Poisson count mixed with a gamma,
    # per year at the 2026 exposure, each Pareto(1,000,000, alpha) before the policy limit.
    "pareto_reference": 1_000_000.0,
    "alpha": 1.6,
    "lambda_reference": 10.0,
    "frequency_mixing_cv": 0.25,
    # A claim's policy limit is drawn from the risk profile's bands of sum insured,
    # band k with probability proportional to premium_k * (mid_k / 1m) ** gamma.
    "limit_gamma": 1.25,
    # Everything else (losses below the listing) is attritional: a share of the
    # premium at 2026 rate level and 2026 prices.
    "attritional_elr": 0.40,
    "attritional_cv": 0.04,
    # Generation and listing thresholds (nominal, at the price of the accident date).
    "generation_threshold": 400_000.0,
    "listing_threshold": 500_000.0,
}

# The cedant's rate changes: first of the month, overall average change, written policies.
RATE_CHANGES = [
    (2017, 4, 0.03), (2018, 7, 0.05), (2019, 7, 0.08), (2020, 7, 0.09), (2021, 10, 0.06),
    (2022, 7, -0.02), (2023, 1, 0.03), (2024, 7, 0.02), (2025, 4, 0.01),
]

# Sum insured bands of the 2026 portfolio: lower, upper, number of risks, mean sum insured,
# and a rate on sum insured before scaling to the premium above.
BANDS = [
    (100_000, 500_000, 55_000, 250_000, 0.0036),
    (500_000, 1_000_000, 14_000, 700_000, 0.0034),
    (1_000_000, 2_500_000, 6_500, 1_600_000, 0.0032),
    (2_500_000, 5_000_000, 2_000, 3_500_000, 0.0029),
    (5_000_000, 10_000_000, 500, 7_000_000, 0.0026),
    (10_000_000, 25_000_000, 120, 15_000_000, 0.0023),
    (25_000_000, 50_000_000, 20, 35_000_000, 0.0020),
    (50_000_000, 100_000_000, 3, 70_000_000, 0.0017),
]

# The proposed layers: limit, attachment, paid reinstatements (a count at a rate, pro rata as to
# amount), and the broker's quoted rate on line on the deposit. The quotes are what the
# broker asks; they are not derived from the model below.
LAYERS = [
    ("5m xs 5m", 5_000_000, 5_000_000, 2, 1.0, 0.43),
    ("10m xs 10m", 10_000_000, 10_000_000, 1, 1.0, 0.15),
    ("20m xs 20m", 20_000_000, 20_000_000, 1, 1.0, 0.06),
]
BROKERAGE = 0.10

# Development patterns of the attritional losses, by development year (age 12, 24, ... months).
# Paid never runs ahead of incurred.
ATTRITIONAL_INCURRED = [0.72, 0.89, 0.955, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0, 1.0]
ATTRITIONAL_PAID = [0.45, 0.75, 0.90, 0.96, 0.985, 0.993, 0.997, 0.999, 1.0, 1.0]


def rate_level(s):
    level = 1.0
    for y, m, c in RATE_CHANGES:
        if s >= y + (m - 1) / 12:
            level *= 1 + c
    return level


def earned(year, f, step=1 / 120):
    """Earned value in a calendar year of f(s) written uniformly at date s, policies of one year."""
    total, n = 0.0, int(2 / step)
    for i in range(n):
        s = year - 1 + (i + 0.5) * step
        total += f(s) * (1 - abs(s - year)) * step
    return total


def to_date(frac):
    year = int(math.floor(frac))
    return dt.date(year, 1, 1) + dt.timedelta(days=int((frac - year) * 365))


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(TRUE["seed"])
    years = list(range(TRUE["first_year"], TRUE["last_year"] + 1))
    g = math.log(1 + TRUE["exposure_growth"])
    s_tr = TRUE["severity_trend"]
    alpha, ref = TRUE["alpha"], TRUE["pareto_reference"]

    def price(a):  # price level at accident date a, relative to the treaty year
        return (1 + s_tr) ** (a - TRUE["price_reference"])

    # Exposure and premium. Exposure is written uniformly at intensity exp(g (s - 2026)).
    exposure = {y: earned(y, lambda s: math.exp(g * (s - 2026))) for y in years + [2026]}
    current = rate_level(2100.0)
    k = TRUE["onlevel_premium_2026"] / (exposure[2026] * current)
    premium = {y: k * earned(y, lambda s: math.exp(g * (s - 2026)) * rate_level(s)) for y in years + [2026]}
    onlevel = {y: k * exposure[y] * current for y in years + [2026]}

    write("subject_premium.csv", ["calendar_year", "subject_premium"],
          [(y, round(premium[y], -3)) for y in years])
    write("rate_changes.csv", ["effective_date", "average_rate_change", "applies_to"],
          [(f"{y}-{m:02d}-01", c, "policies written on or after the date") for y, m, c in RATE_CHANGES])

    # Cost index, quarterly, with noise around the true trend.
    qrows = []
    for i in range(40):
        t = TRUE["first_year"] + i / 4
        val = 100 * (1 + s_tr) ** (t - TRUE["first_year"] + 0.125) * math.exp(rng.normal(0, 0.012))
        qrows.append((to_date(t).isoformat(), round(val, 2)))
    write("cost_index.csv", ["quarter_start", "index"], qrows)

    # The risk profile: premium scaled to the on-level premium of 2026.
    tsi = [n * m for _, _, n, m, _ in BANDS]
    raw = [t * r for t, (_, _, _, _, r) in zip(tsi, BANDS)]
    scale = TRUE["onlevel_premium_2026"] / sum(raw)
    prof_prem = [round(x * scale, -3) for x in raw]
    write("risk_profile.csv", ["band", "lower_sum_insured", "upper_sum_insured", "risks", "total_sum_insured", "premium"],
          [(i + 1, lo, up, n, n * m, p) for i, ((lo, up, n, m, _), p) in enumerate(zip(BANDS, prof_prem))])
    mids = [(lo + up) / 2 for lo, up, *_ in BANDS]
    weights = [p * (mid / 1e6) ** TRUE["limit_gamma"] if lo >= 500_000 else 0.0
               for p, mid, (lo, *_) in zip(prof_prem, mids, BANDS)]
    weights = np.array(weights) / sum(weights)

    # Large losses.
    gen_t = TRUE["generation_threshold"]
    claims = []
    for y in years:
        x_lo = gen_t / price(y + 1.0)
        mix = rng.gamma(1 / TRUE["frequency_mixing_cv"] ** 2, TRUE["frequency_mixing_cv"] ** 2)
        mean = TRUE["lambda_reference"] * (exposure[y] / exposure[2026]) * (x_lo / ref) ** (-alpha) * mix
        for _ in range(rng.poisson(mean)):
            a = y + rng.uniform()
            x = x_lo * rng.uniform() ** (-1 / alpha)  # at 2026 prices, before the limit
            p = price(a)
            if x * p < gen_t:
                continue  # thinning: this claim is below the generation threshold at its own prices
            band = rng.choice(len(BANDS), p=weights)
            si26 = rng.uniform(BANDS[band][0], BANDS[band][1])
            limit = round(si26 * p, -3)
            ult = min(x * p, limit)
            lag = rng.exponential(0.08) if rng.uniform() < 0.85 else rng.exponential(0.9)
            claims.append({"acc": a, "report": a + lag, "limit": limit, "ult": ult})

    claims.sort(key=lambda c: c["acc"])
    listing, large_by_val = [], {}  # large_by_val[(origin, valuation_year)] = (paid, incurred)
    n_listed = 0
    for c in claims:
        if c["report"] >= TRUE["last_year"] + 1:
            continue  # not reported by the valuation date
        rows = []
        for v in range(int(math.floor(c["report"])), TRUE["last_year"] + 1):
            age = (v + 1) - c["acc"]
            f = 1 - 0.35 * math.exp(-age / 0.9)
            sig = 0.2 * math.exp(-age / 2)
            eps = math.exp(rng.normal(0, sig) - sig**2 / 2)
            inc = min(c["limit"], c["ult"] * f * eps)
            paid = min(inc, inc * (1 - math.exp(-age / 1.1)))
            rows.append((v, paid, inc))
        if rows[-1][2] < TRUE["listing_threshold"]:
            continue
        n_listed += 1
        cid = f"C{int(c['acc'])}-{n_listed:03d}"
        for v, paid, inc in rows:
            listing.append((cid, to_date(c["acc"]).isoformat(), to_date(c["report"]).isoformat(),
                            c["limit"], f"{v}-12-31", round(paid, -2), round(inc, -2)))
            key = (int(math.floor(c["acc"])), v)
            p0, i0 = large_by_val.get(key, (0.0, 0.0))
            large_by_val[key] = (p0 + round(paid, -2), i0 + round(inc, -2))
    write("large_losses.csv",
          ["claim_id", "accident_date", "report_date", "policy_limit", "valuation_date", "paid", "incurred"], listing)

    # The whole book's triangle: attritional losses plus the listed claims. The attritional part's
    # noise is drawn once per accident year (its ultimate), so its paid stays below its incurred and
    # its incurred never falls; the listed claims' incurred can fall as case reserves are revised.
    tri = []
    for y in years:
        ult = (TRUE["attritional_elr"] * onlevel[y] * (1 + s_tr) ** (y + 0.5 - TRUE["price_reference"])
               * math.exp(rng.normal(0, TRUE["attritional_cv"]) - TRUE["attritional_cv"] ** 2 / 2))
        for d in range(TRUE["last_year"] - y + 1):
            v = y + d
            lp, li = large_by_val.get((y, v), (0.0, 0.0))
            ai = ult * ATTRITIONAL_INCURRED[d]
            ap = ult * ATTRITIONAL_PAID[d]
            tri.append((y, 12 * (d + 1), round(ap + lp, -3), round(ai + li, -3)))
    write("triangle.csv", ["accident_year", "development_months", "paid", "incurred"], tri)

    # Treaty terms.
    write("treaty_terms.csv",
          ["layer", "limit", "attachment", "reinstatements", "reinstatement_rate", "quoted_rate_on_line"], LAYERS)
    gnpi = round(premium[2026], -5)
    write("treaty_program.csv", ["item", "value"], [
        ("cedant", "Northgate Mutual (synthetic)"),
        ("line_of_business", "commercial property, per risk excess of loss"),
        ("currency", "USD"),
        ("period_start", "2026-01-01"),
        ("period_end", "2026-12-31"),
        ("basis", "losses occurring during the period"),
        ("estimated_subject_premium_income", gnpi),
        ("brokerage", BROKERAGE),
        ("listing_threshold", TRUE["listing_threshold"]),
        ("valuation_date", "2025-12-31"),
    ])

    # The answer key: expected loss of each layer to occurrence terms at 2026 prices.
    def occ_layer(limit, attachment):
        tot = 0.0
        for w, (lo, up, *_) in zip(weights, BANDS):
            if w == 0:
                continue
            si = np.linspace(lo, up, 401)[:-1] + (up - lo) / 800
            top = np.minimum(si, attachment + limit)
            ok = top > attachment
            val = ref**alpha * (attachment ** (1 - alpha) - top ** (1 - alpha)) / (alpha - 1)
            tot += w * float(np.mean(np.where(ok, val, 0.0)))
        return TRUE["lambda_reference"] * tot

    key = {
        **TRUE,
        "rate_changes": RATE_CHANGES,
        "limit_band_weights": [round(float(w), 6) for w in weights],
        "layer_expected_loss_2026_occurrence": {n: occ_layer(lim, att) for n, lim, att, *_ in LAYERS},
        "onlevel_premium": {str(y): onlevel[y] for y in years + [2026]},
        "claims_generated": len(claims),
        "claims_listed": n_listed,
    }
    with open(KEY, "w", newline="\n") as f:  # LF on every platform, like the CSV files
        json.dump(key, f, indent=2)
        f.write("\n")
    print(f"{len(claims)} claims generated, {n_listed} listed, {len(listing)} listing rows")
    for n, v in key["layer_expected_loss_2026_occurrence"].items():
        print(f"  true expected loss {n}: {v:,.0f}")


if __name__ == "__main__":
    main()
