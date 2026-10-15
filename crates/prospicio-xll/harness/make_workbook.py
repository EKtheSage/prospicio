import sys
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
ws["M1"] = "Add-in:"; ws["N1"] = "=PRO.VERSION()"
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
ws["J40"] = f'=PRO.CL.LDF({T},"median")'
for col, w in {"A":12,"J":22,"M":12,"N":12}.items():
    ws.column_dimensions[col].width = w
for col in "BCDEFGHIKLOPQ":
    ws.column_dimensions[col].width = 12
for row in ws.iter_rows(min_row=4, max_row=43):
    for c in row:
        if c.column > 1 and c.number_format == "General":
            c.number_format = "#,##0.0000" if c.row in (14,15,16) or (33 <= c.row <= 43 and c.column <= 5) else "#,##0"
wb.calculation.fullCalcOnLoad = True
wb.save(out)
print("wrote", out)
