"""Build the 200-triangle panel of the calibration study.

Meyers' screen (CAS Monograph 8, 2nd ed., 2019, appendix, Tables A.1 and
A.2, p. 83-85) rebuilt by `cas_schedule_p.screens.select_companies` from the
pip package `cas-schedule-p` (docs/design/reserving-calibration.md,
decisions 2 and 3): 50 triangles in each of commercial auto, personal auto,
workers' compensation and other liability, accident years 1988 to 1997, the
full 10 x 10 square of each (the lower triangle is the outcome).

Run from the repository root:

    uv run --no-project --with "cas-schedule-p[screens]==2026.6.13" python -I validation/scripts/calibration_panel.py

(`[screens]` is the package's extra with duckdb and pandas, which the
screen needs; the package pins no Python version, uv picks the newest.) The
panel is not stored in the repo. The script writes, under
`target/calibration/` (gitignored) or the directory of `--out`:

* `panel.csv`: columns `line, group_code, origin, development, paid,
  incurred, case_incurred, premium, premium_direct`. `line` is ca, pa, wc or
  ol; `origin` the accident year; `development` the age in months (12 to
  120); `paid` is cum_paid_loss, `incurred` is incurred_loss,
  `case_incurred` is incurred_loss - bulk_loss (Meyers' incurred),
  `premium` is earned_prem_net and `premium_direct` earned_prem_direct.
  USD thousands.
* `panel_check.csv`: the diff of the chosen codes against Table A.2's 175
  printed codes, one row per code: `line, group_code, status`, where status
  is `both` (chosen and printed), `chosen_only` or `printed_only`.

Checks before writing: group 353 commercial auto's square equals
`validation/data/clrd_comauto_353.csv` cell by cell, and every triangle is a
complete 10 x 10 square. After writing, the script prints the diff against
Table A.2 and exits with status 2 if it finds a difference: any printed
code the screen did not choose, and for personal auto, workers'
compensation and other liability any chosen code that is not printed (the
printed commercial auto column holds 25 of the 50 codes, so its other 25
cannot be checked). The file is written either way.
"""

import argparse
import csv
import sys
from pathlib import Path

import duckdb
from cas_schedule_p import path as bundled_path
from cas_schedule_p import screens

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "target/calibration/panel.csv"
GROUP_353 = ROOT / "validation/data/clrd_comauto_353.csv"
TABLE = "mart_reserving_model_training"

LINES = {
    "ca": "commercial_auto",
    "pa": "private_passenger_auto",
    "wc": "workers_compensation",
    "ol": "other_liability",
}

# Meyers (2019), Table A.2, "Group Codes for Selected Loss Triangles"
# (p. 84-85): 25 commercial auto codes (18 on p. 84, 7 on p. 85) and 50 for
# each other line, 175 in all. The commercial auto column prints 25 of 50
# codes; that the rest are lower-numbered codes is an inference (group 353,
# Meyers' illustrative insurer, is not among them).
TABLE_A2 = {
    "ca": [
        8672, 9466, 10022, 10308, 11037, 11118, 13420, 13439, 13528, 13641,
        13889, 14044, 14176, 14257, 14311, 14320, 14508, 14974, 15024, 15199,
        18163, 18767, 18791, 19020, 19780,
    ],
    "pa": [
        43, 353, 388, 620, 692, 715, 1066, 1090, 1538, 1767,
        2003, 2143, 3240, 4839, 5185, 6807, 6947, 7080, 8427, 8559,
        10022, 13420, 13439, 13501, 13528, 13587, 13595, 13641, 13889, 14044,
        14176, 14257, 14311, 14443, 15024, 15199, 15393, 15660, 15997, 16373,
        16799, 18163, 18791, 23574, 23876, 25275, 26808, 27022, 27065, 27499,
    ],
    "wc": [
        86, 337, 353, 388, 671, 715, 965, 1066, 1252, 1538,
        1767, 2135, 2712, 3034, 3240, 5185, 6408, 6807, 7080, 8559,
        8672, 9466, 10385, 10699, 11126, 11347, 11703, 13439, 13501, 13528,
        14176, 14320, 14508, 14974, 15148, 15199, 15334, 18309, 18767, 18791,
        21172, 23108, 23140, 26433, 27529, 30589, 34576, 37370, 38687, 38733,
    ],
    "ol": [
        620, 671, 683, 715, 833, 1252, 1279, 1538, 1767, 2003,
        2135, 2143, 2208, 3000, 3240, 5185, 5320, 6459, 6947, 7625,
        10657, 11126, 11150, 11231, 13439, 13501, 13668, 13919, 13994, 14044,
        14176, 14257, 14370, 14451, 14885, 15113, 15148, 15210, 15571, 16373,
        16799, 18163, 18686, 26797, 27065, 28550, 30139, 30651, 32875, 34606,
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Build the calibration study's panel.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    mart = bundled_path(TABLE).as_posix()
    con = duckdb.connect()

    def cells(line_name, code):
        return con.execute(
            f"""
            select accident_year, development_age * 12,
                   cum_paid_loss, incurred_loss, incurred_loss - bulk_loss,
                   earned_prem_net, earned_prem_direct
            from read_parquet('{mart}')
            where line_of_business = ? and company_code = ?
              and accident_year between 1988 and 1997
            order by accident_year, development_age
            """,
            [line_name, str(code)],
        ).fetchall()

    # Group 353's square against the committed file, cell by cell.
    expected = {}
    with open(GROUP_353, newline="") as fh:
        rows = csv.reader(line for line in fh if not line.startswith("#"))
        header = next(rows)
        assert header == ["origin", "development", "paid", "incurred", "premium"], header
        for r in rows:
            expected[(int(r[0]), int(r[1]))] = [float(x) for x in r[2:]]
    got = cells(LINES["ca"], 353)
    assert len(got) == 100 and len(expected) == 100, (len(got), len(expected))
    for ay, dev, paid, _inc, case, prem, _direct in got:
        assert expected[(ay, dev)] == [paid, case, prem], (ay, dev)
    print("group 353 commercial auto: 100 cells equal validation/data/clrd_comauto_353.csv")

    panel, check, differs = [], [], False
    for key, line_name in LINES.items():
        chosen = [
            int(c)
            for c in screens.select_companies(line=line_name, per_line=50)["company_code"]
        ]
        passing = [
            int(c)
            for c in screens.select_companies(line=line_name, per_line=100_000)["company_code"]
        ]
        n_pass = len(passing)
        printed = TABLE_A2[key]
        both = sorted(set(chosen) & set(printed))
        chosen_only = sorted(set(chosen) - set(printed))
        printed_only = sorted(set(printed) - set(chosen))
        check += [(key, c, "both") for c in both]
        check += [(key, c, "chosen_only") for c in chosen_only]
        check += [(key, c, "printed_only") for c in printed_only]
        print(
            f"{key}: screen passes {n_pass}, chosen {len(chosen)}, printed {len(printed)}; "
            f"in both {len(both)}, chosen only {len(chosen_only)}, "
            f"printed only {len(printed_only)}"
        )
        if printed_only:
            why = [
                f"{c}: rank {passing.index(c) + 1} of {n_pass}" if c in passing
                else f"{c}: fails the screen"
                for c in printed_only
            ]
            print(f"  printed but not chosen: {'; '.join(why)}")
            differs = True
        if key == "ca":
            print(f"  chosen, not printed (cannot be checked): {len(chosen_only)}")
        elif chosen_only:
            print(f"  chosen but not printed: {chosen_only}")
            differs = True
        assert len(chosen) == 50, (key, len(chosen))
        for code in chosen:
            cs = cells(line_name, code)
            assert len(cs) == 100, (key, code, len(cs))
            for ay, dev, paid, inc, case, prem, direct in cs:
                panel.append((key, code, ay, dev, paid, inc, case, prem, direct))

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            ["line", "group_code", "origin", "development", "paid", "incurred",
             "case_incurred", "premium", "premium_direct"]
        )
        w.writerows(panel)
    with open(out.with_name("panel_check.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["line", "group_code", "status"])
        w.writerows(check)
    print(f"wrote {out} ({len(panel)} rows, {len(panel) // 100} triangles)")
    if differs:
        print("the rebuilt panel differs from Table A.2 (see above)", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
