"""Write validation/data/raa.xlsx, RAA in a workbook for the triangle readers' tests.

Sheets: `wide` (origin years and ages as numbers, from A1), `long` (origins
as date cells, the table starting at B3 with nothing above or to its left) and `bad` (a
text value in the paid column). Run from the repository root:

    uv run --no-project --with openpyxl python -I validation/scripts/raa_xlsx.py validation/data/raa.csv validation/data/raa.xlsx

The workbook's timestamps are fixed, but openpyxl's output can still change
between versions, so regenerate it only when the sheets change
(docs/design/triangle.md, decision 6).
"""
import csv, datetime, sys
from openpyxl import Workbook

src, out = sys.argv[1], sys.argv[2]
rows = []
with open(src, newline="") as fh:
    r = csv.reader(l for l in fh if not l.startswith("#"))
    next(r)
    for o, d, v in r:
        rows.append((int(o), int(d), float(v)))
wb = Workbook()
# Wide, at A1: integer years and ages as header numbers.
ws = wb.active
ws.title = "wide"
ages = sorted({d for _, d, _ in rows})
ws.append(["origin"] + ages)
for o in sorted({o for o, _, _ in rows}):
    vals = {d: v for oo, d, v in rows if oo == o}
    ws.append([o] + [vals.get(a) for a in ages])
# Long, starting at B3, origins as date cells.
ws = wb.create_sheet("long")
ws.cell(row=3, column=2, value="origin")
ws.cell(row=3, column=3, value="development")
ws.cell(row=3, column=4, value="paid")
for k, (o, d, v) in enumerate(rows):
    ws.cell(row=4 + k, column=2, value=datetime.date(o, 1, 1))
    ws.cell(row=4 + k, column=3, value=d)
    ws.cell(row=4 + k, column=4, value=v)
# Errors: a text value in a number column.
ws = wb.create_sheet("bad")
ws.append(["origin", "development", "paid"])
ws.append([1981, 12, 5012])
ws.append([1981, 24, "n/a"])
wb.properties.created = datetime.datetime(2026, 10, 28)
wb.properties.modified = datetime.datetime(2026, 10, 28)
wb.save(out)
