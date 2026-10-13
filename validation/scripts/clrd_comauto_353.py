"""Regenerate validation/data/clrd_comauto_353.csv.

Meyers' illustrative insurer (CAS Monograph 8, 2nd ed., 2019, sec. 2):
group 353, Commercial Auto, accident years 1988-1997 at 12 to 120 months,
the full 10 x 10 square, from the CAS loss reserve database (NAIC Schedule P,
assembled by Meyers and Shi) as rebuilt by the public repository
EKtheSage/cas-schedule-p-data-model (docs/design/reserving-bayes.md, Data).
Run from the repository root:

    uv run --no-project --python 3.14 --with duckdb==1.5.6 python -I -X utf8 validation/scripts/clrd_comauto_353.py

Needs the GitHub CLI (`gh`). The script downloads the release's manifest and
the table into a fresh temporary directory, checks the publish id and the
Parquet's sha256 against the manifest, selects the 100 cells, and asserts
before writing: the age-10 sums Meyers prints (paid 40,000, incurred 40,061),
one premium per origin, and every cell against Tables 2.1 to 2.3 as printed
(monograph p. 4-5), with the one misprint listed in MISPRINTS.

Columns: paid is cum_paid_loss; incurred is case incurred,
incurred_loss - bulk_loss (as Meyers' ins.line.data computes it); premium is
earned_prem_net (Meyers' net_premium, column 11 of the CAS files, which
Table 2.1 labels "Net Written Premium"). USD thousands.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/data/clrd_comauto_353.csv"
REPO = "EKtheSage/cas-schedule-p-data-model"
RELEASE = "20260613_041006"
TABLE = "mart_reserving_model_training"

# Meyers (2019), Table 2.1 (p. 4): premium by accident year 1988-1997.
PREMIUM = [5812, 4908, 5454, 5165, 5214, 5230, 4992, 5466, 5226, 4962]
# Table 2.2 (p. 4): paid, accident years 1988-1997 by lag 1-10, as printed.
PAID = [
    [952, 1529, 2813, 3647, 3724, 3832, 3899, 3907, 3911, 3912],
    [849, 1564, 2202, 2432, 2468, 2487, 2513, 2526, 2531, 2527],
    [983, 2211, 2830, 3832, 4039, 4065, 4102, 4155, 4268, 4274],
    [1657, 2685, 3169, 3600, 3900, 4320, 4332, 4338, 4341, 4341],
    [932, 1940, 2626, 3332, 3368, 3491, 3531, 3540, 3540, 3583],
    [1162, 2402, 2799, 2996, 3034, 3042, 3230, 3238, 3241, 3268],
    [1478, 2980, 3945, 4714, 5462, 5680, 5682, 5683, 5684, 5684],
    [1240, 2080, 2607, 3080, 3678, 2004, 4117, 4125, 4128, 4128],
    [1326, 2412, 3367, 3843, 3965, 4127, 4133, 4141, 4142, 4144],
    [1413, 2683, 3173, 3674, 3805, 4005, 4020, 4095, 4132, 4139],
]
# Table 2.3 (p. 5): case incurred, as printed.
INCURRED = [
    [1722, 3830, 3603, 3835, 3873, 3895, 3918, 3918, 3917, 3917],
    [1581, 2192, 2528, 2533, 2528, 2530, 2534, 2541, 2538, 2532],
    [1834, 3009, 3488, 4000, 4105, 4087, 4112, 4170, 4271, 4279],
    [2305, 3473, 3713, 4018, 4295, 4334, 4343, 4340, 4342, 4341],
    [1832, 2625, 3086, 3493, 3521, 3563, 3542, 3541, 3541, 3587],
    [2289, 3160, 3154, 3204, 3190, 3206, 3351, 3289, 3267, 3268],
    [2881, 4254, 4841, 5176, 5551, 5689, 5683, 5688, 5684, 5684],
    [2489, 2956, 3382, 3755, 4148, 4123, 4126, 4127, 4128, 4128],
    [2541, 3307, 3789, 3973, 4031, 4157, 4143, 4142, 4144, 4144],
    [2203, 2934, 3608, 3977, 4040, 4121, 4147, 4155, 4183, 4181],
]
# (table, accident year, lag): (printed, data). Table 2.2 prints the 1995 row's
# "Source" year 2004 in its lag-6 cell (and 1997 as its source); the database
# holds 4116, between the printed 3678 and 4117.
MISPRINTS = {("paid", 1995, 6): (2004, 4116)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(
            ["gh", "release", "download", RELEASE, "-R", REPO, "-D", str(tmp),
             "-p", "manifest.json", "-p", f"{TABLE}.parquet"],
            check=True,
        )
        manifest = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["publish_id"] == RELEASE, manifest["publish_id"]
        parquet = tmp / f"{TABLE}.parquet"
        digest = sha256(parquet)
        expected = manifest["tables"][TABLE]["sha256"]
        assert digest == expected, f"{TABLE}.parquet sha256 {digest}, manifest {expected}"
        rows = duckdb.sql(
            f"""
            select accident_year, development_age, cum_paid_loss,
                   incurred_loss - bulk_loss as incurred, earned_prem_net
            from read_parquet('{parquet.as_posix()}')
            where company_code = '353' and line_of_business = 'commercial_auto'
              and accident_year between 1988 and 1997
              and development_age between 1 and 10
            order by accident_year, development_age
            """
        ).fetchall()

    assert len(rows) == 100, f"{len(rows)} rows, want 100"
    cells = {(ay, age): (paid, inc, prem) for ay, age, paid, inc, prem in rows}
    assert len(cells) == 100, "repeated (accident year, age)"
    for (ay, age), (paid, inc, prem) in cells.items():
        w = ay - 1988
        assert prem == PREMIUM[w], f"{ay} age {age}: premium {prem}, Table 2.1 {PREMIUM[w]}"
        for name, value, table in [("paid", paid, PAID), ("incurred", inc, INCURRED)]:
            printed = table[w][age - 1]
            misprint = MISPRINTS.get((name, ay, age))
            if misprint is not None:
                assert (printed, value) == misprint, f"{name} {ay} age {age}: {printed} vs {value}"
            else:
                assert value == printed, f"{name} {ay} age {age}: data {value}, printed {printed}"
    paid10 = sum(cells[(ay, 10)][0] for ay in range(1988, 1998))
    inc10 = sum(cells[(ay, 10)][1] for ay in range(1988, 1998))
    assert (paid10, inc10) == (40000, 40061), (paid10, inc10)
    assert all(v > 0 for c in cells.values() for v in c), "a non-positive cell"

    header = [
        "# CAS loss reserve database (NAIC Schedule P, assembled for research by Glenn Meyers and Peng Shi, CAS):",
        "# group 353, Commercial Auto, Meyers' illustrative insurer (CAS Monograph 8, 2nd ed., 2019, Tables 2.1-2.3),",
        "# accident years 1988-1997 at 12 to 120 months, the full 10 x 10 square, USD thousands.",
        f"# Via github.com/{REPO} release {RELEASE}, table {TABLE} (sha256 {digest}),",
        "# company_code '353', line_of_business 'commercial_auto': paid = cum_paid_loss, incurred = incurred_loss -",
        "# bulk_loss (case incurred), premium = earned_prem_net. Generated by validation/scripts/clrd_comauto_353.py,",
        "# which checks the age-10 sums (paid 40,000, incurred 40,061) and every cell against Tables 2.1-2.3.",
        "origin,development,paid,incurred,premium",
    ]
    lines = [
        f"{ay},{12 * age},{paid:.0f},{inc:.0f},{prem:.0f}"
        for (ay, age), (paid, inc, prem) in sorted(cells.items())
    ]
    OUT.write_text("\n".join(header + lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} rows to {OUT}; paid {paid10:.0f}, incurred {inc10:.0f} at age 10")


if __name__ == "__main__":
    main()
