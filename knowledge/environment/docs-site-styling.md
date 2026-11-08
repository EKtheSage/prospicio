---
type: Environment Fact
title: Styling the great-docs and pkgdown sites
description: Where each docs site takes a stylesheet and fonts, why great-docs' tables scrolled sideways, which selectors and variables carry each site's colours and dark mode, and how to check a stylesheet without the hour-long docs build.
tags: [environment, docs, python, r, css]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-08T12:00:00Z }
stale_after: 2027-05-08T00:00:00Z
sources:
  - id: great-docs
    resource: ../python/great-docs.yml
    title: The Python site's configuration (site.css, include_in_header)
  - id: pkgdown
    resource: ../R/prospicio/_pkgdown.yml
    title: The R site's configuration (template, bslib, light-switch)
  - id: python-css
    resource: ../python/docs-site.css
    title: The Python site's stylesheet
  - id: r-css
    resource: ../R/prospicio/pkgdown/extra.css
    title: The R site's stylesheet
---

# Styling the great-docs and pkgdown sites

Found on 2026-11-07 and 2026-11-08 with great-docs as installed in
`python/.venv` and pkgdown 2.2.1.

* **Wide tables.** great-docs wraps every table in `div.gd-table-scroll`
  (its `responsive-tables.js`) and sets `.gd-table-scroll > table { width:
  max-content; min-width: 100% }`. `max-content` lays each table out as
  wide as its longest row on one line, so the cells never wrap: the prose
  tables of the severity guide were 1,820, 1,623 and 1,558 px wide against a
  text column of at most about 850 px. Collapsing the sidebar would not have
  helped. `width: 100%` on the same selector lets them wrap; a table whose
  cells cannot wrap stays wider than the column and keeps the scroll box.
* **great-docs takes a stylesheet, not a theme file.** `site.css` in
  `great-docs.yml` (a path relative to `python/`) is copied into the build
  and listed after great-docs' own theme, so its rules win at equal
  specificity. A custom `.scss` added to `site.theme` is not copied, so the
  build cannot find it. Fonts go in `include_in_header`.
* **great-docs' colours are its own variables.** It sets `--gd-bg-primary`,
  `--gd-text-primary`, `--gd-code-bg`, `--gd-sidebar-bg`, `--gd-navbar-bg`
  and others on `html.quarto-dark, body.quarto-dark` (and the light set on
  `:root`). Reassigning those, in both modes, restyles most components at
  once. A few rules need a stronger selector: the floating sidebar
  (`body.quarto-dark #quarto-sidebar.gd-sidebar-float`) and inline code in
  dark mode (`body.quarto-dark :not(pre) > code`, a pink `#f0a0a0`). It also
  names Open Sans and Inter on `p`, headings, `.nav-link`, `.table` and the
  sidebar items, so a new face must be set on those selectors too.
* **great-docs' dark mode** follows `prefers-color-scheme` unless the
  visitor chose a mode, which it keeps in `localStorage` under
  `great-docs-theme`; `theme-init.js` sets `quarto-dark` or `quarto-light`
  on `<html>` and `data-bs-theme` before the page shows.
* **pkgdown** loads `R/prospicio/pkgdown/extra.css` after Bootstrap, uses
  `R/prospicio/pkgdown/index.md` as the home page before the DESCRIPTION
  text (`^pkgdown$` is already in `.Rbuildignore`), and takes
  `template: light-switch: true` for a light and dark switch. Bootstrap 5.3
  puts `data-bs-theme` on `<html>` and on the navbar too, so token blocks
  must select `html[data-bs-theme=...]`; selecting any `[data-bs-theme]`
  element gives the navbar the other mode's colours.
* **Checking a stylesheet without the build.** `cargo xtask docs` executes
  the notebooks and takes about an hour. Instead, download a few live pages,
  add `<base href>` pointing at the live page and the stylesheet inline at
  the end of `<head>`, set the theme key in `localStorage` before the
  theme script runs, and screenshot them in a browser. Stylesheets loaded
  through `<base href>` are cross-origin, so `document.styleSheets` cannot
  list their rules; read the selectors from great-docs' `great-docs.scss`
  in the venv instead. The real build is checked from the `docs-site`
  artifact of CI's `docs` job.
