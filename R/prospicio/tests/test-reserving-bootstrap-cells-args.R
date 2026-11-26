suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The bootstraps' `cells` argument and property (#979, the follow-up to the
# review of #1124): `cells` is a single TRUE or FALSE, as Python's
# `cells: bool`, and the fit's `cells` is built once with the fit, as
# `reserves` is, not again on every access.

raa_long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
raa <- triangle(raa_long, "origin", "development", "value")
boots <- list(odp = odp_bootstrap, mack = mack_bootstrap)

for (name in names(boots)) {
  boot <- boots[[name]]

  # Anything but a single TRUE or FALSE is refused, not read as FALSE.
  for (bad in list(1, 0L, "TRUE", NA, c(TRUE, TRUE), logical(0), NULL)) {
    expect_error_like(boot(raa, n_sims = 10, cells = bad), "cells must be TRUE or FALSE")
  }

  # FALSE keeps no cells; TRUE keeps them.
  stopifnot(is.null(boot(raa, n_sims = 10, cells = FALSE)@cells))
  fit <- boot(raa, n_sims = 10, cells = TRUE)
  stopifnot(S7::S7_inherits(fit@cells, predictive_distribution))

  # The cells are stored with the fit: two accesses give the same Rust
  # object, where a getter would build a new copy of the draws each time.
  stopifnot(identical(fit@cells@ptr, fit@cells@ptr))
  stopifnot(identical(fit@reserves@ptr, fit@reserves@ptr))
}

cat("bootstrap cells arguments: ok\n")
