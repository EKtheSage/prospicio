"""Writes a test workbook for the XLL: RAA with every function in place
and R ChainLadder's totals beside them.

    python3 make_workbook.py prospicio-xll-test.xlsx

Each table is saved as an Excel 365 dynamic-array formula that spills from
its first cell, as if typed there. openpyxl writes only legacy (Ctrl+Shift+
Enter) array formulas, so the saved file is patched: each array formula's
cell gets `cm="1"`, pointing at a dynamic-array entry in `xl/metadata.xml`,
which is how Excel itself saves a spilling formula.
"""

import re
import sys
import zipfile

from openpyxl import Workbook
from openpyxl.worksheet.formula import ArrayFormula
from openpyxl.styles import Font, PatternFill
out = sys.argv[1]
RAA = [
 [5012,8269,10907,11805,13539,16181,18009,18608,18662,18834],
 [106,4285,5396,10666,13782,15599,15496,16169,16704],
 [3410,8992,13873,16141,18735,22214,22863,23466],
 [5655,11555,15766,21266,23425,26083,27067],
 [1092,9565,15836,22169,25955,26180],
 [1513,6445,11702,12935,15852],
 [557,4020,10946,12314],
 [1351,6947,13112],
 [3133,5395],
 [2063]]
wb = Workbook(); ws = wb.active; ws.title = "RAA"
bold = Font(bold=True); note = Font(italic=True, color="666666")
inp = PatternFill("solid", fgColor="FFF2CC")
ws["A1"] = "prospicio XLL test: RAA (Mack 1993), paid, cumulative"; ws["A1"].font = Font(bold=True, size=13)
ws["A2"] = "Load prospicio.xll first (File > Options > Add-ins > Excel Add-ins > Browse). If cells show #NAME?, press Ctrl+Alt+F9."; ws["A2"].font = note
ws["M1"] = "Add-in:"; ws["N1"] = ArrayFormula("N1", "=PRO.VERSION()")
ws["A3"] = "AY"; ws["A3"].font = bold
for c in range(10):
    cell = ws.cell(row=3, column=2+c, value=12*(c+1)); cell.font = bold
for r, row in enumerate(RAA):
    ws.cell(row=4+r, column=1, value=1981+r).font = bold
    for c, x in enumerate(row):
        ws.cell(row=4+r, column=2+c, value=x).fill = inp
T = "B4:K13"
ws["A14"] = "LDF"; ws["A14"].font = bold
ws["B14"] = ArrayFormula("B14:K14", f"=PRO.CL.LDF({T})")
ws["A15"] = "CDF"; ws["A15"].font = bold
ws["B15"] = ArrayFormula("B15:K15", f"=PRO.CL.CDF({T})")
ws["A16"] = "LDF, simple avg, tail 1.05"; ws["A16"].font = note
ws["B16"] = ArrayFormula("B16:K16", f'=PRO.CL.LDF({T},"simple",1.05)')

ws["M3"] = ArrayFormula("M3:Q14", f"=PRO.CL({T},,,A4:A13)")
ws["M16"] = "=PRO.CL.ULTIMATE / RESERVE as plain columns:"; ws["M16"].font = note
ws["M17"] = ArrayFormula("M17:M26", f"=PRO.CL.ULTIMATE({T})")
ws["N17"] = ArrayFormula("N17:N26", f"=PRO.CL.RESERVE({T})")

ws["A18"] = "Mack (volume-weighted, no tail)"; ws["A18"].font = bold
ws["A19"] = ArrayFormula("A19:H30", f"=PRO.MACK({T},,,A4:A13)")
ws["A32"] = "Development factors, exponential tail"; ws["A32"].font = bold
ws["A33"] = ArrayFormula("A33:E43", f'=PRO.CL.FACTORS({T},,"exponential")')

ws["J32"] = "Check against R ChainLadder 0.2.21"; ws["J32"].font = bold
checks = [
 ("Total CL reserve", "=Q14", 52135.2282612101),
 ("Total Mack SE", "=G30", 26880.7403298914),
 ("Total process SE", "=E30", 24917.2657465242),
 ("Total parameter SE", "=F30", 10084.8434990404),
]
for h, col in zip(["Quantity","prospicio","R","Match"], "JKLM"):
    ws[f"{col}33"] = h; ws[f"{col}33"].font = bold
for i, (name, f, r) in enumerate(checks):
    row = 34+i
    ws[f"J{row}"] = name; ws[f"K{row}"] = f; ws[f"L{row}"] = r
    ws[f"M{row}"] = f"=ABS(K{row}-L{row})<1E-6"
ws["J39"] = "Try an error: average \"median\""; ws["J39"].font = note
ws["J40"] = ArrayFormula("J40", f'=PRO.CL.LDF({T},"median")')
for col, w in {"A":12,"J":22,"M":12,"N":12}.items():
    ws.column_dimensions[col].width = w
for col in "BCDEFGHIKLOPQ":
    ws.column_dimensions[col].width = 12
for row in ws.iter_rows(min_row=4, max_row=43):
    for c in row:
        if c.column > 1 and c.number_format == "General":
            c.number_format = "#,##0.0000" if c.row in (14,15,16) or (33 <= c.row <= 43 and c.column <= 5) else "#,##0"
# Layer pricing: parameters in cells, a column of attachments against a
# row of limits, so one formula fills the table.
ly = wb.create_sheet("Layers")
ly["A1"] = "prospicio XLL test: layer pricing"; ly["A1"].font = Font(bold=True, size=13)
ly["A2"] = "Change the yellow cells; every table recalculates."; ly["A2"].font = note
for row, (label, value) in enumerate([
    ("Pareto t", 500), ("Pareto alpha", 2),
    ("Lognormal mean", 50000), ("Lognormal CV", 3), ("Basic limit", 100000)], start=4):
    ly[f"A{row}"] = label; ly[f"A{row}"].font = bold
    ly[f"B{row}"] = value; ly[f"B{row}"].fill = inp
PAR = '"pareto",$B$4,$B$5'
LOG = '"lognormal_cv",$B$6,$B$7'
ly["A10"] = "Pareto: expected loss per ground-up loss, PRO.LAYER"; ly["A10"].font = bold
ly["A11"] = "Attachment \\ limit"; ly["A11"].font = note
for c, lim in enumerate([1000, 4000, 10000, "unlimited"]):
    cell = ly.cell(row=11, column=2+c, value=lim); cell.fill = inp; cell.font = bold
for r, a in enumerate([1000, 5000, 10000]):
    cell = ly.cell(row=12+r, column=1, value=a); cell.fill = inp; cell.font = bold
ly["B12"] = ArrayFormula("B12:E14", f"=PRO.LAYER(B11:E11,A12:A14,{PAR})")
ly["A16"] = "Layer CV given a loss above the attachment, PRO.LAYER.CV"; ly["A16"].font = note
ly["B17"] = ArrayFormula("B17:E19", f"=PRO.LAYER.CV(B11:E11,A12:A14,{PAR})")

ly["G10"] = "Lognormal: ILF and LEV by limit"; ly["G10"].font = bold
for col, h in zip("GHIJ", ["Limit", "ILF", "LEV", "LEV, Pareto"]):
    ly[f"{col}11"] = h; ly[f"{col}11"].font = bold
for r, lim in enumerate([100000, 250000, 500000, 1000000, 2000000, 5000000, "unlimited"]):
    ly.cell(row=12+r, column=7, value=lim).fill = inp
ly["H12"] = ArrayFormula("H12:H18", f"=PRO.ILF(G12:G18,$B$8,{LOG})")
ly["I12"] = ArrayFormula("I12:I18", f"=PRO.LEV(G12:G18,{LOG})")
ly["J12"] = ArrayFormula("J12:J18", f"=PRO.LEV(G12:G18,{PAR})")

ly["A22"] = "Check against the R package Pareto (vignette, t = 500, alpha = 2)"; ly["A22"].font = bold
for h, col in zip(["Quantity", "prospicio", "Expected", "Match"], "ABCD"):
    ly[f"{col}23"] = h; ly[f"{col}23"].font = bold
for i, (name, f, x) in enumerate([
        ("4000 xs 1000", "=C12", 200),
        ("unlimited xs 1000, t^a d^(1-a)/(a-1)", "=E12", 250)]):
    row = 24 + i
    ly[f"A{row}"] = name; ly[f"B{row}"] = f; ly[f"C{row}"] = x
    ly[f"D{row}"] = f"=ABS(B{row}-C{row})<1E-9"
ly["A27"] = "Valid only with the inputs above unchanged."; ly["A27"].font = note
ly.column_dimensions["A"].width = 34
for col in "BCDEGHIJ":
    ly.column_dimensions[col].width = 14
for row in ly.iter_rows(min_row=12, max_row=19, min_col=2, max_col=10):
    for c in row:
        c.number_format = "0.0000" if (c.row >= 17 and c.column <= 5) or c.column == 8 else "#,##0.00"
for row in ly.iter_rows(min_row=11, max_row=18, min_col=7, max_col=7):
    for c in row:
        c.number_format = "#,##0"

wb.calculation.fullCalcOnLoad = True
wb.save(out)

METADATA = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<metadata xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" \
xmlns:xda="http://schemas.microsoft.com/office/spreadsheetml/2017/dynamicarray">\
<metadataTypes count="1"><metadataType name="XLDAPR" minSupportedVersion="120000" copy="1" \
pasteAll="1" pasteValues="1" merge="1" splitFirst="1" rowColShift="1" clearFormats="1" \
clearComments="1" assign="1" coerce="1" cellMeta="1"/></metadataTypes>\
<futureMetadata name="XLDAPR" count="1"><bk><extLst>\
<ext uri="{bdbb8cdc-fa1e-496e-a857-3c3f30c029c3}">\
<xda:dynamicArrayProperties fDynamic="1" fCollapsed="0"/></ext></extLst></bk></futureMetadata>\
<cellMetadata count="1"><bk><rc t="1" v="0"/></bk></cellMetadata></metadata>"""
REL = ('<Relationship Id="rIdMeta" Target="metadata.xml" Type="http://schemas.openxmlformats.org/'
       'officeDocument/2006/relationships/sheetMetadata"/>')
OVERRIDE = ('<Override PartName="/xl/metadata.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.sheetMetadata+xml"/>')


def make_dynamic(path):
    """Turns every legacy array formula in `path` into a spilling one."""
    with zipfile.ZipFile(path) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    n = 0
    for name in [p for p in parts if p.startswith("xl/worksheets/sheet")]:
        sheet, k = re.subn(
            r'<c (r="[A-Z]+[0-9]+")([^>]*)><f t="array"', r'<c \1\2 cm="1"><f t="array"', parts[name].decode()
        )
        parts[name] = sheet.encode()
        n += k
    rels = parts["xl/_rels/workbook.xml.rels"].decode()
    parts["xl/_rels/workbook.xml.rels"] = rels.replace("</Relationships>", REL + "</Relationships>").encode()
    types = parts["[Content_Types].xml"].decode()
    parts["[Content_Types].xml"] = types.replace("</Types>", OVERRIDE + "</Types>").encode()
    parts["xl/metadata.xml"] = METADATA.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return n


print("wrote", out, "with", make_dynamic(out), "spilling formulas")
