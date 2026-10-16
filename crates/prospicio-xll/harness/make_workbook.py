"""Writes a test workbook for the XLL: RAA with every function in place
and R ChainLadder's totals beside them.

    python3 make_workbook.py prospicio-xll-test.xlsx

Each table is saved as an Excel 365 dynamic-array formula that spills from
its first cell, as if typed there. openpyxl writes only legacy (Ctrl+Shift+
Enter) array formulas, so the saved file is patched: each array formula's
cell gets `cm="1"`, pointing at a dynamic-array entry in `xl/metadata.xml`,
which is how Excel itself saves a spilling formula.
"""

import csv
import datetime
import re
import sys
import zipfile
from pathlib import Path

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

ly["A30"] = "Every severity family: mean, 99.5% quantile and LEV at 1,000,000"; ly["A30"].font = bold
for h, col in zip(["Family", "p1", "p2", "p3", "Mean", "Q 99.5%", "LEV 1m", "LER at 10,000"], "ABCDEFGH"):
    ly[f"{col}31"] = h; ly[f"{col}31"].font = bold
for i, (fam, ps) in enumerate([
        ("pareto", [500, 2]), ("lognormal_cv", [50000, 3]), ("gamma_cv", [50000, 1]),
        ("weibull", [0.7, 30000]), ("loglogistic", [2.5, 30000]), ("inverse_gamma", [3, 100000]),
        ("inverse_gaussian", [50000, 20000]), ("burr", [3, 1.5, 60000]), ("gpd", [0.3, 30000])]):
    r = 32 + i
    ly[f"A{r}"] = fam
    for c, x in zip("BCD", ps):
        ly[f"{c}{r}"] = x; ly[f"{c}{r}"].fill = inp
    a = f'A{r},B{r},C{r},D{r}'
    ly[f"E{r}"] = ArrayFormula(f"E{r}", f"=INDEX(PRO.SEV.STATS({a}),2,2)")
    ly[f"F{r}"] = ArrayFormula(f"F{r}", f"=PRO.SEV.QUANTILE(0.995,{a})")
    ly[f"G{r}"] = ArrayFormula(f"G{r}", f"=PRO.LEV(1000000,{a})")
    ly[f"H{r}"] = ArrayFormula(f"H{r}", f"=PRO.LER(10000,{a})")
    for c in "EFG":
        ly[f"{c}{r}"].number_format = "#,##0"
    ly[f"H{r}"].number_format = "0.0000"

ly["A43"] = "Pareto helpers (R package Pareto)"; ly["A43"].font = bold
ly["A44"] = "Alpha from two layers: 1m xs 1m expects 1, 2m xs 3m expects"; ly["F44"] = 0.6; ly["F44"].fill = inp
ly["G44"] = ArrayFormula("G44", "=PRO.PARETO.ALPHA(1000000,1000000,1,2000000,3000000,F44)")
ly["A45"] = "Alpha from 4 losses above 1m and 1 above 2m"
ly["G45"] = ArrayFormula("G45", "=PRO.PARETO.ALPHA.FREQS(1000000,4,2000000,1)")
ly["A46"] = "1m xs 1m at alpha G44, extrapolated to each layer below (per unit)"
for c, lim in enumerate([1000000, 2000000, 5000000]):
    ly.cell(row=47, column=2+c, value=lim).fill = inp
for r, att in enumerate([2000000, 3000000]):
    ly.cell(row=48+r, column=1, value=att).fill = inp
ly["B48"] = ArrayFormula("B48:D49", "=PRO.PARETO.EXTRAPOLATE(1000000,1000000,B47:D47,A48:A49,G44)")

ly["A52"] = "Piecewise Pareto: thresholds and alphas as rows, truncated at 50m (last piece)"; ly["A52"].font = bold
for c, x in zip("BCD", [1000000, 2000000, 5000000]):
    ly[f"{c}53"] = x; ly[f"{c}53"].fill = inp
for c, x in zip("BCD", [1.2, 1.8, 2.5]):
    ly[f"{c}54"] = x; ly[f"{c}54"].fill = inp
ly["A53"] = "Thresholds"; ly["A54"] = "Alphas"; ly["A55"] = "Truncation"
ly["B55"] = 50000000; ly["B55"].fill = inp; ly["C55"] = "lp"; ly["C55"].fill = inp
ly["A56"] = "Layer"; ly["B56"] = "Expected loss"; ly["C56"] = "SD"
for h in ("A56", "B56", "C56"):
    ly[h].font = bold
for i, (lim, att) in enumerate([(1000000, 1000000), (3000000, 2000000), (5000000, 5000000)]):
    r = 57 + i
    ly[f"A{r}"] = f"{lim // 1000000}m xs {att // 1000000}m"
    a = f'{lim},{att},"piecewise_pareto",$B$53:$D$53,$B$54:$D$54,$B$55:$C$55'
    ly[f"B{r}"] = ArrayFormula(f"B{r}", f"=PRO.LAYER({a})")
    ly[f"C{r}"] = ArrayFormula(f"C{r}", f"=SQRT(PRO.LAYER.VAR({a}))")
    for c in "BC":
        ly[f"{c}{r}"].number_format = "#,##0"
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
el["A54"] = "Clark's Cape Cod, log-logistic, max age 240 (R's ClarkCapeCod)"; el["A54"].font = bold
el["A55"] = ArrayFormula("A55:G66", f"=PRO.CLARK({G},{P},,240,{O})")
el["I54"] = "Its parameters"; el["I54"].font = bold
el["I55"] = ArrayFormula("I55:J60", f"=PRO.CLARK.PARAMS({G},{P},,240)")
el["A69"] = "Clark's LDF method on GenIns, Weibull"; el["A69"].font = bold
el["A70"] = ArrayFormula("A70:G81", f'=PRO.CLARK({G},,"weibull",,{O})')
el.column_dimensions["A"].width = 22
for col in "BCDEFGHIJKLMNO":
    el.column_dimensions[col].width = 12
for row in el.iter_rows(min_row=17, max_row=81, min_col=2, max_col=15):
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

# Triangle helpers on the RAA sheet's triangle.
th = wb.create_sheet("Triangle helpers")
th["A1"] = "prospicio XLL test: triangle helpers on RAA"; th["A1"].font = Font(bold=True, size=13)
th["A3"] = "Incremental (PRO.TRI.INCREMENTAL)"; th["A3"].font = bold
th["A4"] = ArrayFormula("A4:J13", f"=PRO.TRI.INCREMENTAL({TRI})")
th["L3"] = "Latest diagonal"; th["L3"].font = bold
th["L4"] = ArrayFormula("L4:L13", f"=PRO.TRI.LATEST({TRI})")
th["A16"] = "Link ratios (PRO.TRI.LINKRATIOS)"; th["A16"].font = bold
th["A17"] = ArrayFormula("A17:I26", f"=PRO.TRI.LINKRATIOS({TRI})")
th["A29"] = "Cumulative again (PRO.TRI.CUMULATIVE of the incremental)"; th["A29"].font = bold
th["A30"] = ArrayFormula("A30:J39", "=PRO.TRI.CUMULATIVE(A4:J13)")
th["L29"] = "Same as RAA"; th["L29"].font = bold
th["L30"] = f"=SUMPRODUCT((A30:J39={TRI})*1)=100"
for row in th.iter_rows(min_row=4, max_row=39, min_col=1, max_col=12):
    for c in row:
        c.number_format = "0.0000" if 17 <= c.row <= 26 else "#,##0"

wb.calculation.fullCalcOnLoad = True

# Exposure curves: a risk profile rated with a Swiss Re curve.
ex = wb.create_sheet("Exposure")
ex["A1"] = "Exposure rating with a Swiss Re curve (MBBEFD)"; ex["A1"].font = bold
ex["A2"] = "Swiss Re c"; ex["B2"] = 3; ex["B2"].fill = inp
ex["A3"] = "Layer limit"; ex["B3"] = 4000000; ex["B3"].fill = inp
ex["A4"] = "Layer attachment"; ex["B4"] = 1000000; ex["B4"].fill = inp
ex["D2"] = ArrayFormula("D2:E6", '=PRO.EXPOSURE.STATS("swiss_re",B2)')
for h, col in zip(["Sum insured", "Premium", "Loss ratio", "Expected loss", "Layer share", "Layer loss"], "ABCDEF"):
    ex[f"{col}8"] = h; ex[f"{col}8"].font = bold
for k, (si, prem) in enumerate([(500000, 2000000), (2000000, 3000000), (5000000, 2500000), (10000000, 1500000), (25000000, 1000000)]):
    r = 9 + k
    ex[f"A{r}"] = si; ex[f"B{r}"] = prem; ex[f"C{r}"] = 0.6
    for c in "ABC":
        ex[f"{c}{r}"].fill = inp
    ex[f"D{r}"] = f"=B{r}*C{r}"
for c in "ABDF":
    for r in range(9, 14):
        ex[f"{c}{r}"].number_format = "#,##0"
ex["E9"] = ArrayFormula("E9:E13", '=PRO.EXPOSURE.SHARE($B$3,$B$4,A9:A13,"swiss_re",$B$2)')
for r in range(9, 14):
    ex[f"E{r}"].number_format = "0.0000"
    ex[f"F{r}"] = f"=D{r}*E{r}"
ex["A14"] = "Total"; ex["A14"].font = bold
ex["D14"] = "=SUM(D9:D13)"; ex["F14"] = "=SUM(F9:F13)"
ex["D14"].number_format = ex["F14"].number_format = "#,##0"
ex["A16"] = "G(x) for c = 1.5, 2, 3, 4, 5"; ex["A16"].font = bold
ex["A17"] = "x"
for k, c in enumerate([1.5, 2, 3, 4, 5]):
    ex.cell(row=17, column=2 + k, value=c).fill = inp
for k, x in enumerate([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]):
    ex.cell(row=18 + k, column=1, value=x).fill = inp
ex["B18"] = ArrayFormula("B18:F26", '=PRO.EXPOSURE.CURVE(A18:A26,"swiss_re",B17:F17)')
ex["A28"] = "The c whose mean rate is"; ex["B28"] = 0.05; ex["B28"].fill = inp
ex["C28"] = ArrayFormula("C28", "=PRO.SWISSRE.C(B28)")
ex["A30"] = "The same profile by PRO.PROFILE, net of a surplus of 4 lines on 1,000,000"; ex["A30"].font = bold
ex["A31"] = "Retention"; ex["B31"] = 1000000; ex["B31"].fill = inp
ex["A32"] = "Lines"; ex["B32"] = 4; ex["B32"].fill = inp
ex["A33"] = ArrayFormula("A33:G39", '=PRO.PROFILE(A9:A13,D9:D13,B3,B4,"swiss_re",B2,,,,,,B31,B32)')
for row in ex.iter_rows(min_row=34, max_row=39, min_col=1, max_col=6):
    for c in row:
        c.number_format = "#,##0.00" if c.column == 4 else "#,##0"
for r in range(34, 40):
    ex[f"G{r}"].number_format = "0.0000"
ex.column_dimensions["A"].width = 26
for col in "BCDEFG":
    ex.column_dimensions[col].width = 15


# Credibility: Hachemeister's data (R actuar), and the limited fluctuation table.
cr = wb.create_sheet("Credibility")
cr["A1"] = "Buhlmann-Straub on Hachemeister's data (R actuar): ratios, then weights"; cr["A1"].font = bold
HACHEMEISTER = Path(__file__).resolve().parents[3] / "validation" / "reference" / "hachemeister.csv"
with open(HACHEMEISTER, newline="") as f:
    hach = list(csv.DictReader(f))
for q in range(12):
    cr.cell(row=2, column=2 + q, value=f"Q{q + 1}").font = bold
    cr.cell(row=9, column=2 + q, value=f"Q{q + 1}").font = bold
for st in range(5):
    cr.cell(row=3 + st, column=1, value=f"State {st + 1}")
    cr.cell(row=10 + st, column=1, value=f"State {st + 1}")
for h in hach:
    i, j = int(h["state"]) - 1, int(h["quarter"]) - 1
    cr.cell(row=3 + i, column=2 + j, value=float(h["ratio"])).fill = inp
    cr.cell(row=10 + i, column=2 + j, value=float(h["weight"])).fill = inp
cr["A17"] = ArrayFormula("A17:E22", "=PRO.BUHLMANN(B3:M7,B10:M14,,A3:A7)")
cr["G17"] = ArrayFormula("G17:H21", "=PRO.BUHLMANN.PARAMS(B3:M7,B10:M14)")
cr["J16"] = "Iterative (Bichsel-Straub)"; cr["J16"].font = bold
cr["J17"] = ArrayFormula("J17:K21", '=PRO.BUHLMANN.PARAMS(B3:M7,B10:M14,"iterative")')
cr["A23"] = "Matches R: premium of state 1 is 2055.17"
cr["A24"] = "=ROUND(E18,2)=2055.17"
cr["A26"] = "Limited fluctuation: full credibility standards in claims"; cr["A26"].font = bold
cr["A27"] = "p by k"
for c, k in enumerate([0.025, 0.05, 0.1]):
    cr.cell(row=27, column=2 + c, value=k).fill = inp
for r, p in enumerate([0.9, 0.95, 0.99]):
    cr.cell(row=28 + r, column=1, value=p).fill = inp
cr["B28"] = ArrayFormula("B28:D30", "=PRO.CRED.STANDARD(A28:A30,B27:D27)")
cr["A32"] = "Credibility of 500 claims against each standard"
cr["B32"] = ArrayFormula("B32:D34", "=PRO.CRED.Z(500,B28:D30)")
for row in cr.iter_rows(min_row=28, max_row=30, min_col=2, max_col=4):
    for c in row:
        c.number_format = "#,##0"
for row in cr.iter_rows(min_row=32, max_row=34, min_col=2, max_col=4):
    for c in row:
        c.number_format = "0.000"
cr.column_dimensions["A"].width = 18

# Ratemaking: Werner and Modlin's chapter 5 parallelogram and Table 6.14's trends.
rm = wb.create_sheet("Ratemaking")
rm["A1"] = "Rate level by the parallelogram method (Werner and Modlin, ch. 5)"; rm["A1"].font = bold
rm["A2"] = "Effective"; rm["B2"] = "Change"; rm["A2"].font = rm["B2"].font = bold
for k, (d, c) in enumerate([(datetime.date(2010, 7, 1), 0.05), (datetime.date(2011, 1, 1), 0.10), (datetime.date(2012, 4, 1), -0.01)]):
    rm.cell(row=3 + k, column=1, value=d).number_format = "yyyy-mm-dd"
    rm.cell(row=3 + k, column=2, value=c).number_format = "0.0%"
    rm.cell(row=3 + k, column=1).fill = rm.cell(row=3 + k, column=2).fill = inp
rm["D2"] = "Calendar year"; rm["E2"] = "On-level factor"; rm["F2"] = "Six-month policies"
for h in ("D2", "E2", "F2"):
    rm[h].font = bold
for k, y in enumerate(range(2010, 2014)):
    rm.cell(row=3 + k, column=4, value=y).fill = inp
rm["E3"] = ArrayFormula("E3:E6", "=PRO.ONLEVEL(A3:A5,B3:B5,D3:D6,D3:D6+1)")
rm["F3"] = ArrayFormula("F3:F6", '=PRO.ONLEVEL(A3:A5,B3:B5,D3:D6,D3:D6+1,"calendar",0.5)')
for r in range(3, 7):
    rm[f"E{r}"].number_format = rm[f"F{r}"].number_format = "0.0000"
rm["A9"] = "Calendar year 2011's cells (Figure 5.17)"; rm["A9"].font = bold
rm["A10"] = ArrayFormula("A10:D16", "=PRO.RATELEVEL(A3:A5,B3:B5,2011,2012)")
rm["A18"] = "Trend length: CY2011 losses to rates effective 1/1/2015 for a year"
rm["A19"] = ArrayFormula("A19", '=PRO.AVGDATE(2015,2016,"policy")-PRO.AVGDATE(2011,2012)')
rm["A21"] = "Exponential trend of Table 6.14's severities, quarterly"; rm["A21"].font = bold
sev = [1061.45, 1076.56, 1085.56, 1098.02, 1102.46, 1107.52, 1117.71, 1123.70, 1127.63, 1124.01,
       1116.42, 1109.48, 1105.38, 1102.61, 1096.67, 1093.77, 1100.67, 1108.79, 1122.13, 1126.46]
rm["A22"] = "Time"; rm["B22"] = "Severity"; rm["A22"].font = rm["B22"].font = bold
for k, v in enumerate(sev):
    rm.cell(row=23 + k, column=1, value=2010 + k / 4)
    rm.cell(row=23 + k, column=2, value=v).fill = inp
rm["D22"] = "20 points"; rm["G22"] = "Last 8"; rm["D22"].font = rm["G22"].font = bold
rm["D23"] = ArrayFormula("D23:E28", "=PRO.TREND.FIT(A23:A42,B23:B42)")
rm["G23"] = ArrayFormula("G23:H28", "=PRO.TREND.FIT(A35:A42,B35:B42)")
rm["E24"].number_format = rm["H24"].number_format = "0.0%"
rm["A45"] = "Loss ratio indication (Werner and Modlin, Appendix A)"; rm["A45"].font = bold
heads = ["Earned premium", "On-level", "Premium trend", "Losses", "Development", "Loss trend"]
appendix_a = [
    (1122372, 1.2161, 1.1342, 856495, 1.0000, 0.9912), (1154508, 1.2176, 1.1116, 867184, 0.9799, 0.9962),
    (1280545, 1.1311, 1.0879, 835120, 1.0003, 1.0012), (1369976, 1.0892, 1.0663, 821509, 1.0282, 1.0062),
    (1397750, 1.0991, 1.0452, 797866, 1.0966, 1.0113)]
for c, h in enumerate(heads):
    rm.cell(row=46, column=1 + c, value=h).font = bold
for r, row in enumerate(appendix_a):
    for c, x in enumerate(row):
        rm.cell(row=47 + r, column=1 + c, value=x).fill = inp
rm["H46"] = "ULAE factor"; rm["H46"].font = bold; rm["H47"] = 1.143; rm["H47"].fill = inp
rm["A53"] = ArrayFormula("A53:C59", "=PRO.LR.EXPERIENCE(A47:A51,D47:D51,B47:B51,C47:C51,E47:E51,F47:F51,H47)")
for row in rm.iter_rows(min_row=54, max_row=59, min_col=1, max_col=2):
    for c in row:
        c.number_format = "#,##0"
for r in range(54, 60):
    rm[f"C{r}"].number_format = "0.0%"
rm["A61"] = "Expenses (premium-based): ratio, fixed share"; rm["A61"].font = bold
for r, (x, sh) in enumerate([(0.063, 0.75), (0.085, 0.75), (0.002, 1.0), (0.112, 0.0), (0.021, 0.0)]):
    rm.cell(row=62 + r, column=1, value=x).fill = inp
    rm.cell(row=62 + r, column=2, value=sh).fill = inp
rm["D62"] = ArrayFormula("D62:E63", "=PRO.EXPENSES(A62:A66,B62:B66)")
rm["D65"] = "Profit"; rm["E65"] = 0.05; rm["E65"].fill = inp
rm["D66"] = "Indicated change"
rm["E66"] = ArrayFormula("E66", "=PRO.INDICATED.CHANGE(C59,E63,E65,E62)")
rm["D67"] = "Permissible loss ratio"
rm["E67"] = ArrayFormula("E67", "=PRO.PERMISSIBLE.LR(E63,E65,E62)")
for c in ("E62", "E63", "E66", "E67"):
    rm[c].number_format = "0.0%"
rm.column_dimensions["A"].width = 16
for col in "BCDEFGH":
    rm.column_dimensions[col].width = 15

# Burning cost: Clark's per-risk example (Basics of Reinsurance Pricing, A.4.a).
bc = wb.create_sheet("Burning cost")
bc["A1"] = "Burning cost of 400,000 xs 100,000, trended at 4% to 7/1/1997 (Clark, A.4.a)"; bc["A1"].font = bold
bc["A2"] = "Accident date"; bc["B2"] = "Amount"; bc["A2"].font = bc["B2"].font = bold
clark = [((1988, 9, 20), 240946), ((1988, 10, 11), 821499), ((1989, 3, 15), 158129), ((1990, 6, 21), 114051),
         ((1990, 10, 24), 78043), ((1991, 1, 10), 162533), ((1992, 2, 23), 324298), ((1992, 4, 30), 100549),
         ((1992, 9, 22), 75476), ((1993, 1, 1), 171885), ((1993, 5, 18), 94218), ((1993, 8, 1), 170297),
         ((1994, 8, 15), 87133), ((1995, 7, 12), 771249)]
for r, ((y, m, d), x) in enumerate(clark):
    c = bc.cell(row=3 + r, column=1, value=datetime.date(y, m, d)); c.number_format = "yyyy-mm-dd"; c.fill = inp
    c = bc.cell(row=3 + r, column=2, value=x); c.number_format = "#,##0"; c.fill = inp
bc["D2"] = "Year"; bc["E2"] = "Premium"; bc["F2"] = "Excess LDF"
for h in ("D2", "E2", "F2"):
    bc[h].font = bold
years = [(1988, 1422554, 1.0), (1989, 1823103, 1.0), (1990, 2054034, 1.0), (1991, 2147147, 1.0),
         (1992, 2151541, 1.01), (1993, 2159198, 1.05), (1994, 2167158, 1.15), (1995, 2187654, 1.3)]
for r, (y, p_, f) in enumerate(years):
    for c, x in enumerate((y, p_, f)):
        bc.cell(row=3 + r, column=4 + c, value=x).fill = inp
    bc.cell(row=3 + r, column=5).number_format = "#,##0"
bc["H2"] = "Limit"; bc["H3"] = 400000; bc["I2"] = "Attachment"; bc["I3"] = 100000
bc["H4"] = "Trend"; bc["H5"] = 0.04; bc["I4"] = "Trended to"; bc["I5"] = datetime.date(1997, 7, 1)
bc["I5"].number_format = "yyyy-mm-dd"
for c in ("H3", "I3", "H5", "I5"):
    bc[c].fill = inp
bc["A19"] = ArrayFormula("A19:G29", "=PRO.BURNINGCOST(A3:A16,B3:B16,H3,I3,D3:D10,E3:E10,F3:F10,H5,I5)")
for row in bc.iter_rows(min_row=20, max_row=29, min_col=2, max_col=6):
    for c in row:
        c.number_format = "0.000" if c.column == 4 else "#,##0"
for r in range(20, 30):
    bc[f"G{r}"].number_format = "0.0%"
bc.column_dimensions["A"].width = 14
for col in "BCDEFGHI":
    bc.column_dimensions[col].width = 16

# Aggregate distributions: a compound total and a layer table.
ag = wb.create_sheet("Aggregate")
ag["A1"] = "Annual aggregate loss: 3 claims a year (dispersion 2), gamma severity mean 50,000, CV 1.5"; ag["A1"].font = bold
for r, (label, x) in enumerate([("Claims a year", 3), ("Dispersion", 2), ("Family", "gamma_cv"), ("Mean", 50000), ("CV", 1.5)]):
    ag.cell(row=2 + r, column=1, value=label)
    ag.cell(row=2 + r, column=2, value=x).fill = inp
ag["D1"] = "Ground up"; ag["G1"] = "100,000 xs 50,000 per claim"; ag["D1"].font = ag["G1"].font = bold
ag["D2"] = ArrayFormula("D2:E18", "=PRO.AGG.STATS(B2,B3,B4,B5,B6)")
ag["G2"] = ArrayFormula("G2:H18", "=PRO.AGG.STATS(B2,B3,B4,B5,B6,,100000,50000)")
for c in ("E", "H"):
    for r in range(2, 19):
        ag[f"{c}{r}"].number_format = "#,##0.0000" if r in (4, 5) or r >= 16 else "#,##0"
ag["A21"] = "Per-occurrence layers (aggregate's reins_stats_df)"; ag["A21"].font = bold
ag["A22"] = "Limits"; ag["A23"] = "Attachments"
for c, (l, a) in enumerate([(50000, 50000), (100000, 100000), (300000, 200000)]):
    ag.cell(row=22, column=2 + c, value=l).fill = inp
    ag.cell(row=23, column=2 + c, value=a).fill = inp
ag["A25"] = ArrayFormula("A25:D39", "=PRO.AGG.LAYERS(B2,B3,B4,B5,B6,,B22:D22,B23:D23)")
for row in ag.iter_rows(min_row=26, max_row=39, min_col=2, max_col=4):
    for c in row:
        c.number_format = "0.0000"
ag.column_dimensions["A"].width = 18
ag.column_dimensions["D"].width = 16
ag.column_dimensions["G"].width = 16
for col in "BCEH":
    ag.column_dimensions[col].width = 14

# Bayesian MCMC: the ODP chain ladder sampled with NUTS on RAA (England and Verrall).
by = wb.create_sheet("Bayesian")
by["A1"] = "Bayesian ODP chain ladder on RAA, sampled with NUTS (4 chains, 1,000 warm-up and 1,000 kept draws each)"; by["A1"].font = bold
by["A2"] = "The triangle is the RAA sheet's; change the seed or draws below and the fit reruns."
by["A3"] = "Draws per chain"; by["B3"] = 1000; by["B3"].fill = inp
by["A4"] = "Chains"; by["B4"] = 4; by["B4"].fill = inp
by["A5"] = "Seed"; by["B5"] = 0; by["B5"].fill = inp
by["A6"] = "Prior SD"; by["B6"] = 10; by["B6"].fill = inp
args = "RAA!B4:K13,{out},B3,B4,B5,,,B6"
by["A8"] = "Diagnostics (check these first: R-hat under 1.01, ESS above 400, no divergences)"; by["A8"].font = bold
by["A9"] = ArrayFormula("A9:B16", "=PRO.BAYES.ODP(" + args.format(out='"diagnostics"') + ")")
by["D8"] = "Reserves: chain ladder, posterior mean, SD, CV and percentiles"; by["D8"].font = bold
by["D9"] = ArrayFormula("D9:L20", "=PRO.BAYES.ODP(" + args.format(out='"reserves"') + ")")
by["A23"] = "Posterior of the log-scale parameters: c, the origin effects a[w] and development effects b[d]"; by["A23"].font = bold
by["A24"] = ArrayFormula("A24:I43", "=PRO.BAYES.ODP(" + args.format(out='"params"') + ")")
by["D22"] = "Compare: the ODP bootstrap's total SD"
by["H22"] = ArrayFormula("H22", "=INDEX(PRO.ODP(RAA!B4:K13,20000),12,4)")
for row in by.iter_rows(min_row=10, max_row=20, min_col=5, max_col=12):
    for c in row:
        c.number_format = "0.000" if c.column == 8 else "#,##0"
for row in by.iter_rows(min_row=25, max_row=43, min_col=2, max_col=9):
    for c in row:
        c.number_format = "0.0000" if c.column <= 7 else "#,##0"
by.column_dimensions["A"].width = 20
for col in "BCDEFGHIJKL":
    by.column_dimensions[col].width = 13
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
