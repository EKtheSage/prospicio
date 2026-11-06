### Breaking changes

- Excel: every function's name now carries its group, `PRO.RES.` for
  reserving, `PRO.PRC.` for pricing and `PRO.CAP.` for capital
  (`PRO.MACK` is `PRO.RES.MACK`, `PRO.ILF` is `PRO.PRC.ILF`,
  `PRO.AGG.STATS` is `PRO.CAP.AGG.STATS`), and the old names are gone: a
  formula with one shows `#NAME?`. Severities, EVT tails, triangle
  helpers, `PRO.VERSION` and `PRO.YEARFRACTION` keep their names.
  `docs/design/xll.md` decision 47 lists every old and new name (#746,
  #747).

### Excel

- Each group of functions is a category of its own in Insert Function
  ("Prospicio - Reserving", "- Pricing", "- Capital", "- Common") (#746).
- The add-in is built with link-time optimization: 14.0 MB in place of
  16.3 MB (#746).
