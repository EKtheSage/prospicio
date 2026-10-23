"""Generate the synthetic property-catastrophe book that the cat-* guides read.

Run from anywhere:  python examples/data/make_cat_book.py
It needs numpy only and writes four small CSV files next to itself. The output
is the same every run (SEED below), so the files are committed and the guides
do not need this script, except that it is how you change the case.

The story. Larkspur Re, a fictional reinsurer, writes property catastrophe
cover for three fictional cedents in the fictional country of Varland, whose
three regions are the coast, the plains and the range. Four perils hit them:
windstorm (WS), earthquake (EQ), flood (FL) and convective storm (CS). A
vendor catastrophe model gives an event catalog (one row per event, with an
annual rate and each cedent's mean loss and secondary uncertainty) and, from
it, 2,000 simulated years of dated events.

  cat_book_elt_alder.csv   Alder Mutual's event loss table: one row per event
                           of the catalog that can hit it (rate, mean, sd, exposure)
  cat_book_yelt.csv        the year event loss table: one row per loss to one
                           cedent from one event piece in one simulated year,
                           with its day, hour, peril, region and cedent
  cat_book_cedents.csv     each cedent's subject premium and home region
  cat_book_contracts.csv   Larkspur Re's contracts for the coming year

A convective storm outbreak and some windstorms reach the YELT as two or three
timed pieces within 60 hours of each other whose losses sum to the event's
loss, so an hours clause decides whether they are one occurrence. Money is in
millions of Varland crowns (a synthetic currency).

Two simplifications are worth knowing. Each cedent's loss from an event is a
beta draw with standard deviation sd_independent + sd_correlated (the two
parts add as standard deviations), drawn independently for each cedent: the
event is shared, but the correlated part of secondary uncertainty is not
correlated across cedents. And a YELT row is written only for a loss of at
least 0.05, as many vendor extracts drop tiny losses, so the YELT's mean
annual loss is slightly lower than an unfloored simulation's.
"""

import csv
import math
from pathlib import Path

import numpy as np

SEED = 20261023
N_YEARS = 2000
N_EVENTS = 300
CEDENTS = ["alder", "bracken", "corran"]
# Each cedent's relative exposure by region (coast, plains, range).
EXPOSURE = {
    "alder": {"coast": 1.0, "plains": 0.4, "range": 0.2},
    "bracken": {"coast": 0.3, "plains": 1.0, "range": 0.5},
    "corran": {"coast": 0.5, "plains": 0.3, "range": 1.0},
}
# Peril: share of events, regions it hits with weights, log-mean and log-sd of
# the industry-scale mean loss, season (start day, end day) or None. Ten per cent
# of a seasonal peril's events fall on any day of the year (day_and_hour).
PERILS = {
    "WS": dict(share=0.40, regions={"coast": 0.8, "plains": 0.2}, mu=math.log(60.0), sigma=1.3, season=(213, 305)),
    "EQ": dict(share=0.15, regions={"range": 0.7, "coast": 0.3}, mu=math.log(120.0), sigma=1.5, season=None),
    "FL": dict(share=0.20, regions={"plains": 0.6, "coast": 0.4}, mu=math.log(30.0), sigma=1.0, season=(60, 180)),
    "CS": dict(share=0.25, regions={"plains": 0.9, "range": 0.1}, mu=math.log(15.0), sigma=0.9, season=(100, 220)),
}
TOTAL_RATE = 2.4  # catalog events a year


def catalog(rng):
    """The event catalog: one dict per event with each cedent's loss parameters."""
    perils = list(PERILS)
    shares = np.array([PERILS[p]["share"] for p in perils])
    rows = []
    for event_id in range(1, N_EVENTS + 1):
        peril = perils[rng.choice(len(perils), p=shares)]
        spec = PERILS[peril]
        regions = list(spec["regions"])
        region = regions[rng.choice(len(regions), p=np.array(list(spec["regions"].values())))]
        size = float(np.exp(rng.normal(spec["mu"], spec["sigma"])))
        # Bigger events are rarer: rate falls with size, then is scaled to TOTAL_RATE below.
        weight = float(rng.uniform(0.5, 1.5)) / size**0.55
        cedents = {}
        for c in CEDENTS:
            mean = size * EXPOSURE[c][region] * float(rng.uniform(0.15, 0.45))
            if mean < 0.1:
                continue
            cv = float(rng.uniform(0.6, 1.4)) / (1.0 + mean / 50.0)
            sd = mean * cv
            exposure = mean * float(rng.uniform(6.0, 12.0))
            # Beta on [0, exposure] needs sd^2 < mean (exposure - mean).
            sd = min(sd, 0.95 * math.sqrt(mean * (exposure - mean)))
            share_corr = float(rng.uniform(0.2, 0.5))
            cedents[c] = dict(
                mean=round(mean, 4),
                sd_independent=round(sd * (1 - share_corr), 4),
                sd_correlated=round(sd * share_corr, 4),
                exposure=round(exposure, 4),
            )
        if cedents:
            rows.append(dict(event_id=event_id, peril=peril, region=region, weight=weight, cedents=cedents))
    total = sum(r["weight"] for r in rows)
    for r in rows:
        r["rate"] = round(TOTAL_RATE * r.pop("weight") / total, 8)
    return rows


def beta_loss(rng, mean, sd, exposure):
    """A beta draw on [0, exposure] with the given mean and sd (prospicio's ELT convention)."""
    m = mean / exposure
    v = (sd / exposure) ** 2
    k = m * (1 - m) / v - 1
    return exposure * float(rng.beta(m * k, (1 - m) * k))


def day_and_hour(rng, season):
    if season is None or rng.uniform() < 0.1:
        day = int(rng.integers(1, 366))
    else:
        day = int(rng.integers(season[0], season[1] + 1))
    return day, int(rng.integers(0, 24))


def pieces(rng, peril):
    """How many timed pieces an event reaches the YELT as, and their hour offsets."""
    if peril == "CS":
        k = 1 + int(rng.binomial(2, 0.35))
    elif peril == "WS":
        k = 1 + int(rng.uniform() < 0.25)
    else:
        k = 1
    offsets = [0] + sorted(int(h) for h in rng.integers(6, 61, size=k - 1))
    weights = rng.dirichlet(np.ones(k)) if k > 1 else np.array([1.0])
    return offsets, weights


def yelt(rng, events):
    rates = np.array([e["rate"] for e in events])
    probs = rates / rates.sum()
    rows = []
    for year in range(1, N_YEARS + 1):
        n = int(rng.poisson(rates.sum()))
        draws = []
        for _ in range(n):
            e = events[rng.choice(len(events), p=probs)]
            day, hour = day_and_hour(rng, PERILS[e["peril"]]["season"])
            offsets, weights = pieces(rng, e["peril"])
            losses = {c: beta_loss(rng, p["mean"], p["sd_independent"] + p["sd_correlated"], p["exposure"]) for c, p in e["cedents"].items()}
            for off, w in zip(offsets, weights):
                t = (day - 1) * 24 + hour + off
                if t >= 8760:  # a piece past year end stays in the year's last hour
                    t = 8759
                draws.append((t, e, {c: x * float(w) for c, x in losses.items()}))
        draws.sort(key=lambda d: d[0])
        for slot, (t, e, losses) in enumerate(draws, start=1):
            for c in CEDENTS:
                if c in losses and losses[c] >= 0.05:
                    rows.append(
                        dict(
                            year=year,
                            slot=slot,
                            event_id=e["event_id"],
                            day=t // 24 + 1,
                            hour=t % 24,
                            peril=e["peril"],
                            region=e["region"],
                            cedent=c,
                            loss=f"{losses[c]:.4f}",
                        )
                    )
    return rows


CEDENT_INFO = [
    dict(cedent="alder", name="Alder Mutual", home_region="coast", subject_premium=420.0, cat_premium=40.0),
    dict(cedent="bracken", name="Bracken Insurance", home_region="plains", subject_premium=310.0, cat_premium=36.0),
    dict(cedent="corran", name="Corran General", home_region="range", subject_premium=260.0, cat_premium=30.0),
]

# Larkspur Re's contracts for the coming year. Amounts are on a 100% basis; share is Larkspur's
# signed line (the cession for a quota share). perils is "all" or a list joined by "+". An
# occurrence layer's premium is rate_on_line x limit x share; a quota share's is share x the
# cedent's cat_premium, less a provisional ceding commission. hours is the hours clause.
CONTRACT_FIELDS = ["contract", "cedent", "kind", "perils", "attachment", "limit", "aggregate_deductible",
                   "aggregate_limit", "share", "reinstatements", "reinstatement_rate", "rate_on_line",
                   "ceding_commission", "hours"]
CONTRACTS = [
    dict(contract="alder_cat_1", cedent="alder", kind="occurrence_xl", perils="all", attachment=30.0, limit=40.0,
         aggregate_deductible=0.0, aggregate_limit="", share=0.30, reinstatements=1, reinstatement_rate=1.0,
         rate_on_line=0.15, ceding_commission="", hours=72),
    dict(contract="alder_cat_2", cedent="alder", kind="occurrence_xl", perils="all", attachment=70.0, limit=100.0,
         aggregate_deductible=0.0, aggregate_limit="", share=0.20, reinstatements=1, reinstatement_rate=1.0,
         rate_on_line=0.05, ceding_commission="", hours=72),
    dict(contract="bracken_cat", cedent="bracken", kind="occurrence_xl", perils="all", attachment=25.0, limit=50.0,
         aggregate_deductible=0.0, aggregate_limit="", share=0.25, reinstatements=2, reinstatement_rate=1.0,
         rate_on_line=0.14, ceding_commission="", hours=72),
    dict(contract="bracken_agg", cedent="bracken", kind="aggregate_xl", perils="CS+FL", attachment="", limit="",
         aggregate_deductible=40.0, aggregate_limit=30.0, share=0.25, reinstatements="", reinstatement_rate="",
         rate_on_line=0.06, ceding_commission="", hours=""),
    dict(contract="corran_qs", cedent="corran", kind="quota_share", perils="all", attachment=0.0, limit=150.0,
         aggregate_deductible=0.0, aggregate_limit=300.0, share=0.10, reinstatements="", reinstatement_rate="",
         rate_on_line="", ceding_commission=0.25, hours=72),
]


def write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main(out=None):
    out = Path(out) if out is not None else Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)
    events = catalog(rng)
    elt = [
        dict(event_id=e["event_id"], peril=e["peril"], region=e["region"], rate=e["rate"], **e["cedents"]["alder"])
        for e in events
        if "alder" in e["cedents"]
    ]
    write(out / "cat_book_elt_alder.csv", elt,
          ["event_id", "peril", "region", "rate", "mean", "sd_independent", "sd_correlated", "exposure"])
    write(out / "cat_book_yelt.csv", yelt(rng, events),
          ["year", "slot", "event_id", "day", "hour", "peril", "region", "cedent", "loss"])
    write(out / "cat_book_cedents.csv", CEDENT_INFO, ["cedent", "name", "home_region", "subject_premium", "cat_premium"])
    write(out / "cat_book_contracts.csv", CONTRACTS, CONTRACT_FIELDS)


if __name__ == "__main__":
    main()
