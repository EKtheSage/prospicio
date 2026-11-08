# Figures and tables on the R docs site that follow its light and dark modes.
#
# `cargo xtask r` runs this file in a hidden first chunk of each R twin it
# executes (one that holds `<!-- site: execute -->`). A twin's code is plain
# ggplot2 and gt: nothing here is needed to run it elsewhere. The R twin of
# figures.py, which does the same for the Python site.
#
# Figures. ggplot2 draws with placeholder colours (`ink`, `muted`, `rule` and
# the three series), each plot that a chunk prints is written as SVG by
# svglite, and the placeholders are replaced by the site's CSS variables
# (`var(--p-ink)` and so on, defined for both modes in
# R/prospicio/pkgdown/extra.css). A grey (`"grey75"`) is an amount of ink: it
# becomes the page's ink mixed into the background in that proportion, and
# white becomes the background. The SVG is written into the page, so it takes
# the colours of the mode the reader has chosen, and changes with it.
#
# Tables. gt's own colours are cleared by the site's CSS (`.gt_table` in
# extra.css), so a table takes the page's colours too.

local({
  ink <- "#010203"
  muted <- "#020304"
  rule <- "#030405"
  # Three series at most: the palette (extra.css) passes the colour-vision
  # checks for three; a fourth series belongs in another panel.
  series <- c("#0a0b01", "#0a0b02", "#0a0b03")
  css <- c(
    "#010203" = "var(--p-ink)", "#020304" = "var(--p-muted)", "#030405" = "var(--p-rule)",
    "#0a0b01" = "var(--p-series-1)", "#0a0b02" = "var(--p-series-2)",
    "#0a0b03" = "var(--p-series-3)"
  )

  ggplot2::theme_set(
    ggplot2::theme_minimal(base_size = 11) +
      ggplot2::theme(
        text = ggplot2::element_text(colour = ink),
        axis.text = ggplot2::element_text(colour = muted),
        axis.title = ggplot2::element_text(colour = muted),
        legend.text = ggplot2::element_text(colour = ink),
        plot.title = ggplot2::element_text(colour = ink, hjust = 0),
        plot.title.position = "plot",
        panel.grid.major = ggplot2::element_line(colour = rule, linewidth = 0.4),
        panel.grid.minor = ggplot2::element_blank(),
        axis.line = ggplot2::element_line(colour = rule),
        plot.background = ggplot2::element_rect(fill = "transparent", colour = NA),
        panel.background = ggplot2::element_rect(fill = "transparent", colour = NA),
        legend.background = ggplot2::element_rect(fill = "transparent", colour = NA),
        legend.key = ggplot2::element_rect(fill = "transparent", colour = NA)
      )
  )
  options(ggplot2.discrete.colour = series, ggplot2.discrete.fill = series)

  grey <- function(hex) {
    level <- strtoi(substr(hex, 2, 3), 16L) / 255
    if (level == 1) return("var(--p-bg)")
    sprintf("color-mix(in srgb, var(--p-ink) %d%%, transparent)", round(100 * (1 - level)))
  }

  svg_of <- function(plot, width = 7, height = 3.6) {
    out <- svglite::svgstring(width = width, height = height, bg = "transparent",
                              standalone = FALSE)
    print(plot)
    grDevices::dev.off()
    text <- paste(out(), collapse = "\n")
    for (hex in names(css)) {
      text <- gsub(hex, css[[hex]], text, ignore.case = TRUE, fixed = FALSE)
    }
    greys <- unique(regmatches(text, gregexpr("#([0-9A-Fa-f]{2})\\1\\1\\b", text, perl = TRUE))[[1]])
    for (hex in greys) text <- gsub(hex, grey(hex), text, fixed = TRUE)
    # svglite quotes the face (`font-family: "Liberation Sans";`), inside a
    # single-quoted style attribute.
    text <- gsub("font-family: ?(?:\"[^\"]*\"|[^;\"']*)", "font-family: var(--p-font)", text,
                 perl = TRUE)
    # One line, so that pandoc keeps it as one block of raw HTML.
    gsub("\n\\s*", " ", text)
  }

  knit_print_ggplot <- function(x, options = NULL, ...) {
    width <- if (is.null(options$fig.width)) 7 else options$fig.width
    height <- if (is.null(options$fig.height)) 3.6 else options$fig.height
    knitr::asis_output(htmltools::htmlPreserve(
      paste0('<figure class="p-figure">', svg_of(x, width, height), "</figure>")
    ))
  }
  registerS3method("knit_print", "ggplot", knit_print_ggplot, envir = asNamespace("knitr"))
})
