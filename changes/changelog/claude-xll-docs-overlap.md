### Breaking changes

- Excel: `PRO.NPV`, `PRO.IRR` and `PRO.MOIC` are removed. They duplicated
  Excel's `NPV`, `XNPV`, `IRR` and `XIRR`, or a one-line formula
  (`1 + released / (funded + opening)`); `PRO.CASHFLOW` still reports each
  simulation's IRR, NPV and MOIC (`docs/design/xll.md` decision 46; #723).

### Excel

- The add-in's README is now a manual grouped by area (reserving,
  severity and layers, exposure rating, experience rating and ratemaking,
  aggregate and treaty terms, capital), with a "Start here" table of the
  functions most work needs and each area's arguments beside its table (#723).
