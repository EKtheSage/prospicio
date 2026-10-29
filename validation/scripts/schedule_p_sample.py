"""Write validation/data/schedule_p_comauto_sample.csv: two commercial auto
companies of the CAS Schedule P data model in the CAS loss reserve
database's own file layout, for R's `schedule_p_triangle()` tests.

The CAS files (`comauto_pos.csv`, ...) name the loss and premium columns
with a suffix per line (`_C` for commercial auto). The sample holds the
columns the reader uses: `GRCODE, GRNAME, AccidentYear, DevelopmentYear,
DevelopmentLag, IncurLoss_C, CumPaidLoss_C, BulkLoss_C, EarnedPremDIR_C,
EarnedPremCeded_C, EarnedPremNet_C`, from the package's
`mart_reserving_model_training` (`EarnedPremCeded_C` is direct less net),
accident years 1988 to 1997, the full 10 x 10 squares of groups 353 and
8672. Run from the repository root:

    uv run --no-project --with "cas-schedule-p==2026.6.13" --with pyarrow python -I validation/scripts/schedule_p_sample.py
"""

import csv
from pathlib import Path

import cas_schedule_p
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/data/schedule_p_comauto_sample.csv"
CODES = ["353", "8672"]

rows = pq.read_table(
    cas_schedule_p.path("mart_reserving_model_training"),
    filters=[
        ("line_of_business", "=", "commercial_auto"),
        ("company_code", "in", CODES),
        ("accident_year", ">=", 1988),
        ("accident_year", "<=", 1997),
    ],
).to_pylist()
rows.sort(key=lambda r: (int(r["company_code"]), r["accident_year"], r["development_age"]))
assert len(rows) == 100 * len(CODES), len(rows)


def number(x):
    return int(x) if float(x).is_integer() else x


with open(OUT, "w", newline="") as fh:
    fh.write(
        f"# CAS loss reserve database layout (comauto_pos.csv), groups {' and '.join(CODES)}, "
        f"from cas-schedule-p {cas_schedule_p.__version__} (validation/scripts/schedule_p_sample.py).\n"
    )
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(
        ["GRCODE", "GRNAME", "AccidentYear", "DevelopmentYear", "DevelopmentLag",
         "IncurLoss_C", "CumPaidLoss_C", "BulkLoss_C", "EarnedPremDIR_C",
         "EarnedPremCeded_C", "EarnedPremNet_C"]
    )
    for r in rows:
        ay, lag = r["accident_year"], r["development_age"]
        w.writerow(
            [r["company_code"], r["company_name"], ay, ay + lag - 1, lag,
             number(r["incurred_loss"]), number(r["cum_paid_loss"]), number(r["bulk_loss"]),
             number(r["earned_prem_direct"]),
             number(r["earned_prem_direct"] - r["earned_prem_net"]),
             number(r["earned_prem_net"])]
        )
print(f"wrote {OUT} ({len(rows)} rows)")
