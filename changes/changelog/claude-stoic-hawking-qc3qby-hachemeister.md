### Pricing

- Hachemeister's regression credibility: `credibility::Hachemeister` fits
  each risk's weighted least squares line, or any design shared by the
  risks, and shrinks it towards the collective by a credibility matrix,
  at the origin or with the intercept at the barycentre (`actuar::cm`'s
  `regformula` and `adj.intercept`), with `predict` at a design row.
  Python `prospicio.credibility.hachemeister`, R `hachemeister()` with
  `predict()` (#561).
