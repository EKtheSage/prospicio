"""Regenerate validation/data/clrd_wkcomp_337.csv.

The illustrative company of Gesmann and Morris (2020, "Hierarchical
Compartmental Reserving Models", CAS Research Paper) and Morris (2016):
NAIC company 337, workers' compensation, accident years 1988-1997 at ages 1
to 10, the full 10 x 10 square, from the CAS loss reserve database (NAIC
Schedule P, assembled by Meyers and Shi) as rebuilt by the public repository
EKtheSage/cas-schedule-p-data-model (docs/design/reserving-compartmental.md,
"Data"). Run from the repository root:

    uv run --no-project --python 3.14 --with duckdb==1.5.6 python -I -X utf8 validation/scripts/clrd_wkcomp_337.py

Needs the GitHub CLI (`gh`). The script downloads the release's manifest and
the table into a fresh temporary directory, checks the publish id and the
Parquet's sha256 against the manifest, selects the 100 cells, and asserts
before writing: one premium of each kind per origin, every value finite and
non-negative, the 1988 cell at age 1 (paid 9,558, incurred 62,679, bulk
24,619, case reserve 28,502, direct premium 104,437), and Morris' (2016,
sec. 3) printed upper triangles cell by cell, premium, outstanding and
cumulative paid (calendar years up to 1997, rounded to USD millions though
headed $'000s), which the data match only with the papers' definitions:
outstanding = incurred - paid, bulk and IBNR included, and direct premium.

Columns, USD thousands: paid is cum_paid_loss; incurred is incurred_loss
(with bulk and IBNR); case_incurred is incurred_loss - bulk_loss (Meyers'
convention, as in clrd_comauto_353.csv); premium is earned_prem_direct (the
papers' premium); premium_net is earned_prem_net (Meyers' convention). The
papers' outstanding is incurred - paid.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/data/clrd_wkcomp_337.csv"
REPO = "EKtheSage/cas-schedule-p-data-model"
RELEASE = "20260613_041006"
TABLE = "mart_reserving_model_training"

# The 1988 cell at age 1 (USD thousands), queried from the mart on
# 2026-10-13: paid, incurred, bulk, direct premium.
FIRST = (9558, 62679, 24619, 104437)
# Morris (2016), "Hierarchical Compartmental Models for Loss Reserving", CAS
# E-Forum Summer 2016, sec. 3, the upper triangles as printed (headed $'000s,
# in USD millions): premium, outstanding and cumulative paid, accident years
# 1988-1997 by development year.
MORRIS_PREMIUM = [104, 89, 86, 99, 105, 119, 111, 78, 64, 48]
MORRIS_OS = [
    [53, 41, 32, 25, 17, 13, 10, 7, 2, 1],
    [54, 37, 27, 20, 14, 10, 7, 3, 2],
    [55, 37, 28, 18, 11, 7, 4, 3],
    [61, 42, 26, 15, 9, 6, 4],
    [66, 46, 31, 22, 12, 8],
    [68, 51, 40, 22, 17],
    [62, 47, 32, 24],
    [57, 49, 35],
    [57, 42],
    [41],
]
MORRIS_PAID = [
    [10, 23, 33, 40, 45, 48, 50, 51, 52, 52],
    [8, 19, 30, 37, 41, 43, 45, 46, 46],
    [9, 24, 35, 43, 48, 51, 53, 54],
    [13, 33, 47, 56, 62, 65, 67],
    [11, 29, 42, 51, 56, 59],
    [12, 27, 38, 47, 51],
    [11, 27, 38, 46],
    [13, 32, 44],
    [13, 31],
    [9],
]


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
            select accident_year, development_age, cum_paid_loss, incurred_loss,
                   bulk_loss, earned_prem_direct, earned_prem_net
            from read_parquet('{parquet.as_posix()}')
            where company_code = '337' and line_of_business = 'workers_compensation'
              and accident_year between 1988 and 1997
              and development_age between 1 and 10
            order by accident_year, development_age
            """
        ).fetchall()

    assert len(rows) == 100, f"{len(rows)} rows, want 100"
    cells = {(r[0], r[1]): r[2:] for r in rows}
    assert len(cells) == 100, "repeated (accident year, age)"
    for ay in range(1988, 1998):
        direct = {cells[(ay, age)][3] for age in range(1, 11)}
        net = {cells[(ay, age)][4] for age in range(1, 11)}
        assert len(direct) == 1 and len(net) == 1, f"{ay}: premiums {direct} {net}"
    for key, values in cells.items():
        assert all(v is not None and v >= 0 for v in values), f"{key}: {values}"
        paid, incurred, bulk, _, _ = values
        assert incurred >= paid, f"{key}: incurred {incurred} below paid {paid}"
        assert paid > 0 and incurred - bulk > 0, f"{key}: {values}"
    paid, incurred, bulk, direct, _ = cells[(1988, 1)]
    assert (paid, incurred, bulk, direct) == FIRST, cells[(1988, 1)]
    assert incurred - bulk - paid == 28502, "case reserve"

    def rounds_to(value, printed):
        return abs(value / 1000 - printed) <= 0.5

    for w, ay in enumerate(range(1988, 1998)):
        assert rounds_to(cells[(ay, 1)][3], MORRIS_PREMIUM[w]), f"{ay} premium"
        for age in range(1, 11 - w):
            p, i = cells[(ay, age)][0], cells[(ay, age)][1]
            assert rounds_to(i - p, MORRIS_OS[w][age - 1]), f"{ay} age {age}: outstanding {i - p}"
            assert rounds_to(p, MORRIS_PAID[w][age - 1]), f"{ay} age {age}: paid {p}"

    header = [
        "# CAS loss reserve database (NAIC Schedule P, assembled for research by Glenn Meyers and Peng Shi, CAS):",
        "# company 337, workers' compensation, the illustrative company of Gesmann and Morris (2020) and Morris (2016),",
        "# accident years 1988-1997 at 12 to 120 months, the full 10 x 10 square, USD thousands.",
        f"# Via github.com/{REPO} release {RELEASE}, table {TABLE} (sha256 {digest}),",
        "# company_code '337', line_of_business 'workers_compensation': paid = cum_paid_loss, incurred = incurred_loss",
        "# (with bulk and IBNR), case_incurred = incurred_loss - bulk_loss (Meyers' convention), premium =",
        "# earned_prem_direct (the papers' premium), premium_net = earned_prem_net (Meyers' convention). The papers'",
        "# outstanding is incurred - paid. Generated by validation/scripts/clrd_wkcomp_337.py, which checks the 1988",
        "# cell at age 1 against the mart and every cell of Morris' (2016) printed upper triangles (USD millions).",
        "origin,development,paid,incurred,case_incurred,premium,premium_net",
    ]
    lines = [
        f"{ay},{12 * age},{p:.0f},{i:.0f},{i - b:.0f},{d:.0f},{n:.0f}"
        for (ay, age), (p, i, b, d, n) in sorted(cells.items())
    ]
    OUT.write_text("\n".join(header + lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} rows to {OUT}")


if __name__ == "__main__":
    main()
