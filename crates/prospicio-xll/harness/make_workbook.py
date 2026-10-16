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

# ODP bootstrap on the RAA sheet's triangle.
od = wb.create_sheet("ODP")
od["A1"] = "prospicio XLL test: ODP bootstrap on RAA"; od["A1"].font = Font(bold=True, size=13)
od["A2"] = "The triangle is on the RAA sheet. Change the yellow cells; the same seed gives the same table."
od["A2"].font = note
for row, (label, value) in enumerate([("Simulations", 20000), ("Seed", 1)], start=4):
    od[f"A{row}"] = label; od[f"A{row}"].font = bold
    od[f"B{row}"] = value; od[f"B{row}"].fill = inp
od["D4"] = "Percentiles"; od["D4"].font = bold
for c, p in enumerate([0.75, 0.9, 0.95, 0.995]):
    cell = od.cell(row=4, column=5+c, value=p); cell.fill = inp
TRI, ORI = "RAA!$B$4:$K$13", "RAA!$A$4:$A$13"
od["A7"] = "Gamma process error (R's process.distr = 'gamma')"; od["A7"].font = bold
od["A8"] = ArrayFormula("A8:I19", f"=PRO.ODP({TRI},$B$4,$B$5,$E$4:$H$4,,,{ORI})")
od["A21"] = "Parameter error only (process \"none\")"; od["A21"].font = bold
od["A22"] = ArrayFormula("A22:I33", f'=PRO.ODP({TRI},$B$4,$B$5,$E$4:$H$4,,"none",{ORI})')

od["K7"] = "Check: total against R ChainLadder 0.2.21 BootChainLadder (100,000 sims)"; od["K7"].font = bold
for h, col in zip(["Quantity", "prospicio", "R", "Tolerance", "Within"], "KLMNO"):
    od[f"{col}8"] = h; od[f"{col}8"].font = bold
checks = [
    ("Gamma: mean", "=C19", 53842.2718407722, 589.413),
    ("Gamma: SD", "=D19", 18965.3468418162, 483.93),
    ("Gamma: P95", "=G19", 87856.1780690262, 1854.44),
    ("None: mean", "=C33", 53928.2509498396, 506.674),
    ("None: SD", "=D33", 17453.9114703321, 513.428),
    ("None: P95", "=G33", 85236.8809859552, 1701.59),
]
for i, (name, f, r, tol) in enumerate(checks):
    row = 9 + i
    od[f"K{row}"] = name; od[f"L{row}"] = f; od[f"M{row}"] = r; od[f"N{row}"] = tol
    od[f"O{row}"] = f"=ABS(L{row}-M{row})<N{row}"
od["K16"] = "Tolerance: four standard errors at 20,000 simulations; use 20000 or more."
od["K16"].font = note
od.column_dimensions["A"].width = 12
od.column_dimensions["K"].width = 16
for col in "BCDEFGHILMNO":
    od.column_dimensions[col].width = 12
for row in od.iter_rows(min_row=9, max_row=33, min_col=2, max_col=14):
    for c in row:
        if not (c.column >= 11 and c.row > 14):
            c.number_format = "0.000" if c.column == 5 and c.row != 4 else "#,##0"
for c in od[4][4:8]:
    c.number_format = "0.0%"

# The draws behind the ODP sheet's gamma table: same simulations and seed.
dr = wb.create_sheet("ODP draws")
dr["A1"] = "prospicio XLL test: the ODP bootstrap's draws (PRO.ODP.SIMS)"; dr["A1"].font = Font(bold=True, size=13)
dr["A2"] = "Simulations and seed come from the ODP sheet; the average of each column is that sheet's Mean."
dr["A2"].font = note
dr["N3"] = "Check"; dr["N3"].font = bold
dr["N4"] = "Mean total, here"; dr["O4"] = "=AVERAGE(K5:K100004)"
dr["N5"] = "Mean total, ODP sheet"; dr["O5"] = "=ODP!C19"
dr["N6"] = "Match"; dr["O6"] = "=ABS(O4-O5)<1E-6*O5"
dr["N8"] = "Total reserve histogram"; dr["N8"].font = bold
dr["N9"] = "Up to"; dr["O9"] = "Count"
for h in ("N3", "N9", "O9"):
    dr[h].font = bold
for i, b in enumerate(range(20000, 160001, 20000)):
    dr.cell(row=10+i, column=14, value=b).fill = inp
dr["N18"] = "more"
# FREQUENCY returns one count per bin and one for everything above.
dr["O10"] = ArrayFormula("O10:O18", "=FREQUENCY(K5:K100004,N10:N17)")
dr["A4"] = ArrayFormula("A4:K20004", f"=PRO.ODP.SIMS({TRI},ODP!$B$4,ODP!$B$5,,,{ORI})")
dr.column_dimensions["N"].width = 22
for col in "ABCDEFGHIJKO":
    dr.column_dimensions[col].width = 11
for row in dr.iter_rows(min_row=5, max_row=200, min_col=1, max_col=11):
    for c in row:
        c.number_format = "#,##0"
dr["O4"].number_format = dr["O5"].number_format = "#,##0.00"

# Mack's bootstrap on the RAA sheet's triangle, beside Mack's closed form.
mk = wb.create_sheet("Mack bootstrap")
mk["A1"] = "prospicio XLL test: Mack's bootstrap on RAA (England, Verrall and Wuthrich)"
mk["A1"].font = Font(bold=True, size=13)
mk["A2"] = "Centred residuals: the simulated SD is Mack's standard error (the RAA sheet's PRO.MACK)."
mk["A2"].font = note
for row, (label, value) in enumerate([("Simulations", 20000), ("Seed", 1)], start=4):
    mk[f"A{row}"] = label; mk[f"A{row}"].font = bold
    mk[f"B{row}"] = value; mk[f"B{row}"].fill = inp
mk["A7"] = "Gamma process error"; mk["A7"].font = bold
mk["A8"] = ArrayFormula("A8:I19", f"=PRO.MACK.BOOT({TRI},$B$4,$B$5,,,,,{ORI})")
mk["K7"] = "Check: total against Mack's closed form"; mk["K7"].font = bold
for h, col in zip(["Quantity", "Bootstrap", "Mack", "Tolerance", "Within"], "KLMNO"):
    mk[f"{col}8"] = h; mk[f"{col}8"].font = bold
# Mack's SD with the parameter risk scaled by the residual pool's variance
# 0.981 on RAA (validation/tests/reserving_mack_bootstrap.rs).
mk["K9"] = "Mean"; mk["L9"] = "=C19"; mk["M9"] = "=RAA!D30"; mk["N9"] = "=5*D19/SQRT($B$4)"
mk["K10"] = "SD"; mk["L10"] = "=D19"; mk["M10"] = "=SQRT(RAA!E30^2+0.981*RAA!F30^2)"
mk["N10"] = "=5*D19*SQRT(5/(4*$B$4))"
for r in (9, 10):
    mk[f"O{r}"] = f"=ABS(L{r}-M{r})<N{r}"
mk.column_dimensions["K"].width = 14
for col in "ABCDEFGHILMNO":
    mk.column_dimensions[col].width = 12
for row in mk.iter_rows(min_row=9, max_row=19, min_col=2, max_col=14):
    for c in row:
        c.number_format = "0.000" if c.column == 5 else "#,##0"

# The expected-loss methods on GenIns with R ChainLadder's ClarkCapeCod
# premium, against chainladder-python 0.10.1's totals.
GENINS = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
    [396132, 1333217, 2180715, 2985752, 3691712],
    [440832, 1288463, 2419861, 3483130],
    [359480, 1421128, 2864498],
    [376686, 1363294],
    [344014],
]
el = wb.create_sheet("Expected loss")
el["A1"] = "prospicio XLL test: expected-loss methods on GenIns (premium as in R's ?ClarkCapeCod)"
el["A1"].font = Font(bold=True, size=13)
el["A3"] = "AY"; el["L3"] = "Premium"; el["A3"].font = el["L3"].font = bold
for c in range(10):
    el.cell(row=3, column=2+c, value=12*(c+1)).font = bold
for r, row in enumerate(GENINS):
    el.cell(row=4+r, column=1, value=2001+r).font = bold
    for c, x in enumerate(row):
        el.cell(row=4+r, column=2+c, value=x).fill = inp
    el.cell(row=4+r, column=12, value=10_000_000 + 400_000*r).fill = inp
el["N3"] = "A priori LR"; el["N3"].font = bold; el["O3"] = 0.75; el["O3"].fill = inp
G, P, O = "$B$4:$K$13", "$L$4:$L$13", "$A$4:$A$13"
el["A16"] = "Bornhuetter-Ferguson"; el["A16"].font = bold
el["A17"] = ArrayFormula("A17:G28", f"=PRO.BF({G},{P},$O$3,,,{O})")
el["I16"] = "Benktander, 2 iterations"; el["I16"].font = bold
el["I17"] = ArrayFormula("I17:O28", f"=PRO.BENKTANDER({G},{P},$O$3,2,,,{O})")
el["A31"] = "Cape Cod, trend 5%, decay 0.75"; el["A31"].font = bold
el["A32"] = ArrayFormula("A32:G43", f"=PRO.CAPECOD({G},{P},0.05,0.75,,,{O})")
el["I31"] = "Expected loss"; el["I31"].font = bold
el["I32"] = ArrayFormula("I32:O43", f"=PRO.EL({G},{P},$O$3,,,{O})")
el["A46"] = "Check: total reserve against chainladder-python 0.10.1 (with a priori 0.75)"; el["A46"].font = bold
for h, col in zip(["Method", "prospicio", "chainladder-python", "Match"], "ABCD"):
    el[f"{col}47"] = h; el[f"{col}47"].font = bold
for i, (name, f, x) in enumerate([
        ("Bornhuetter-Ferguson", "=G28", 32357669.898322247),
        ("Benktander, 2", "=O28", 27674175.15049441),
        ("Cape Cod 5%, 0.75", "=G43", 22446795.86898648),
        ("Expected loss", "=O43", 54141910.0)]):
    row = 48 + i
    el[f"A{row}"] = name; el[f"B{row}"] = f; el[f"C{row}"] = x
    el[f"D{row}"] = f"=ABS(B{row}-C{row})<1E-6*C{row}"
el.column_dimensions["A"].width = 22
for col in "BCDEFGHIJKLMNO":
    el.column_dimensions[col].width = 12
for row in el.iter_rows(min_row=17, max_row=50, min_col=2, max_col=15):
    for c in row:
        c.number_format = "#,##0"

# The one-year view on the RAA sheet's triangle.
oy = wb.create_sheet("One-year")
oy["A1"] = "prospicio XLL test: the one-year view on RAA"; oy["A1"].font = Font(bold=True, size=13)
oy["A2"] = "CDR = opening ultimate less the re-estimated one: a negative CDR is adverse."
oy["A2"].font = note
oy["A4"] = "Merz-Wuthrich (closed form)"; oy["A4"].font = bold
oy["A5"] = ArrayFormula("A5:D16", f"=PRO.MACK.CDR({TRI},,,{ORI})")
oy["F4"] = "CDR SE by future calendar year"; oy["F4"].font = bold
oy["F5"] = ArrayFormula("F5:O16", f"=PRO.MACK.CDR.YEARS({TRI},,,{ORI})")
oy["A19"] = "ODP bootstrap, 20000 simulations, seed 1"; oy["A19"].font = bold
oy["A20"] = ArrayFormula("A20:H31", f"=PRO.ODP.CDR({TRI},20000,1,,,,{ORI})")
oy["J19"] = "Mack bootstrap, 20000 simulations, seed 1"; oy["J19"].font = bold
oy["J20"] = ArrayFormula("J20:Q31", f"=PRO.MACK.BOOT.CDR({TRI},20000,1,,,,,{ORI})")
oy["A34"] = "Check against R ChainLadder 0.2.21 CDR(MackChainLadder(RAA))"; oy["A34"].font = bold
for h, col in zip(["Quantity", "prospicio", "R", "Match"], "ABCD"):
    oy[f"{col}35"] = h; oy[f"{col}35"].font = bold
for i, (name, f, x) in enumerate([
        ("Total CDR(1) SE", "=C16", 25166.3025264382),
        ("Total Mack SE", "=D16", 26880.7403298914),
        ("Total CDR SE, year 2", "=H16", 7306.52121591382)]):
    row = 36 + i
    oy[f"A{row}"] = name; oy[f"B{row}"] = f; oy[f"C{row}"] = x
    oy[f"D{row}"] = f"=ABS(B{row}-C{row})<1E-6"
oy.column_dimensions["A"].width = 22
for col in "BCDEFGHIJKLMNOPQ":
    oy.column_dimensions[col].width = 12
for row in oy.iter_rows(min_row=6, max_row=38, min_col=2, max_col=17):
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
