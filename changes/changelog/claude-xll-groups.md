### Excel

- Every function's name now carries its group: `PRO.RES.` for reserving,
  `PRO.PRC.` for pricing and `PRO.CAP.` for capital (`PRO.MACK` is
  `PRO.RES.MACK`, `PRO.ILF` is `PRO.PRC.ILF`, `PRO.AGG.STATS` is
  `PRO.CAP.AGG.STATS`); severities, EVT tails, triangle helpers,
  `PRO.VERSION` and `PRO.YEARFRACTION` keep their names. Each group is a
  category of its own in Insert Function. The 98 old names still calculate
  through 0.0.9 but are no longer listed, and 0.0.10 removes them
  (`docs/design/xll.md` decision 47 has the table of old and new names;
  #746).
- The add-in is built with link-time optimization: 14.0 MB in place of
  16.3 MB (#746).
