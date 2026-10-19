### Guides

- The R twins of the guides (`docs/guides/*.qmd`) and of the notebooks
  (`examples/*.qmd`) are pages of the R docs site, under "Articles", grouped
  as in `docs/guides/reading-list.md`. `cargo xtask r` stages them as pkgdown
  articles (the `.qmd` files stay the only source) and points their links to
  the Python guides at the Python site's pages. A guide's chunks are shown
  with the output that `test-guides.R` checks, not re-run; the two examples
  run in full, and the build fails if `reinsurance_pricing.qmd` reports a
  comparison outside its tolerance. The site index links to them (#388).
