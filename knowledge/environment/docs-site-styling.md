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
* **Fences of several words.** The guides mark code that CI does not run
  with the info string `python no-run`. Pandoc's Markdown reader takes a
  bare info string of one word only, so on the Python site such a block
  was a paragraph (curly quotes and all) until `cargo xtask python` began
  staging it as `{.python .no-run}` (`fence_info` in `xtask/src/main.rs`).
  The source keeps the bare form, which `python/tests/test_guides.py` reads.
* **No navbar or sidebar settings in great-docs.** great-docs writes the
  Quarto navbar itself and strips the sidebar's section toggles; nothing in
  `great-docs.yml` adds a navbar link or hides the sidebar. The script in
  `include_in_header` adds the link back to the landing page (resolved
  from `.navbar-brand`'s href, so it works at any depth) and a button that
  hides `#quarto-sidebar` and widens `main.content` to `page-start`, kept
  in `localStorage` under `prospicio-sidebar-hidden`. On the R site,
  `navbar: components:` in `_pkgdown.yml` takes a relative `href`.
* **Guides that run on the Python site.** A guide with the line
  `<!-- site: execute -->` is staged as a Quarto document (`executed` in
  `xtask/src/main.rs`) and run by great-docs' Quarto build with the
  `python3` Jupyter kernel of `python/.venv`. Quarto's last cell runs
  `%reset` and then prints a JSON line that Quarto parses; an IPython
  `post_run_cell` or `post_execute` hook that reads the cleared namespace
  fails there and prints its traceback into that cell, and the build stops
  with "Extra data: line 2 column 1". The hook in
  `docs/guides/site/figures.py` binds what it uses as default arguments.
  The run leaves Quarto's cache in `python/_freeze/` (gitignored); while
  that directory exists great-docs turns `freeze` on, so a later local
  build shows the cached output of an executed guide until it is deleted.
* **Figures that follow the mode.** One inline SVG per figure, with
  matplotlib's colours replaced by the site's CSS variables
  (`var(--p-ink)`, `var(--p-series-1)` and so on), follows the light and
  dark modes and the switch between them without a second image; Quarto's
  `renderings: [light, dark]` would need two drawings in each cell. Greys
  become `color-mix()` of the ink, so `"0.75"` stays a light grey in either
  mode.
* **Screenshots while the Browser pane is hidden.** A hidden pane paints
  nothing. Headless Edge takes one:
  `msedge --headless=new --user-data-dir=<new dir> --window-size=1200,900
  --virtual-time-budget=6000 --screenshot=<file> <url>`; great-docs reads its
  mode from `localStorage["great-docs-theme"]`, which a page on the same
  origin can set before it loads the guide in an iframe.
* **ggplot2 in the R twins.** `library(ggplot2)` after `library(prospicio)`
  masks prospicio's `layer()` (ggplot2 exports a `layer()` too; no other
  export of ggplot2 or gt clashes), and a later chunk then fails with
  "`data` must be a <data.frame>"; attach it with
  `library(ggplot2, exclude = "layer")`. Under `scale_x_log10()`,
  `geom_histogram(breaks = )` takes the edges in data units, not log10
  units. svglite quotes the face inside a single-quoted style attribute
  (`font-family: "Liberation Sans";`), so a rewrite of the face must take
  the quotes with it.
* **pkgdown's mode in a screenshot.** The R site's mode is the
  `data-bs-theme` attribute on `<html>`; a copy of the page with a script
  that sets it (and moves the element to check into `main`, since headless
  Edge does not scroll to it) shows either mode.
* **An executed R twin runs alone.** The R test harness attaches
  prospicio before it runs any twin, but the site renders each twin as its
  own article, with nothing attached. Four twins never called
  `library(prospicio)` and so failed only in the site build ("could not find
  function"). `R/prospicio/tests/test-guides.R` now fails on an executed
  twin without that line.
* **A guides-only PR never builds the site.** CI's `docs` job runs only when
  a PR touches the bindings, `python/`, `R/`, `xtask/` or `ci.yml`, so the
  first site build of a changed guide is the Pages build on `main`. Before
  merging guide changes, run `cargo xtask docs --no-tests` locally (about an
  hour on the local Windows machine) or `cargo xtask r --no-tests` for the R
  site alone.
* **matplotlib's backend in the Python harness.** On the local Windows
  machine, matplotlib in uv's CPython chooses TkAgg, whose Tcl files are not
  found, so a guide that drew figures failed now and then. The harness sets
  `MPLBACKEND=Agg` and closes each guide's figures.
* **Self-timed R twins.** `pir-portfolio-pricing.qmd` prints whether it ran
  under 15 s. Locally the unchanged twin took 15.8 s on one run and its new
  figures add about 4.7 s (ggplot2 renders each printed plot). CI's Linux
  runner passes it.
