library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# invert_chf on stable laws with the tilt (#779, second PR), against
# aggregate 2.1.0's AggregateCHF: the rows of
# validation/reference/chf_tilt_aggregate.csv the Rust parity test reads.
ref <- read.csv(validation("reference", "chf_tilt_aggregate.csv"), stringsAsFactors = FALSE)
params <- function(text) {
  kv <- strsplit(strsplit(text, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.numeric(vapply(kv, `[`, "", 2)), vapply(kv, `[`, "", 1))
}
fits <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  if (is.null(fits[[r$params]])) {
    v <- params(r$params)
    s <- stable_distribution(v[["alpha"]], v[["beta"]], v[["scale"]], v[["loc"]],
                             if (v[["parameterization"]] == 1) "S0" else "S1")
    tilt <- if (v[["tilt"]] == -1) TRUE else if (v[["tilt"]] == -2) FALSE else v[["tilt"]]
    fits[[r$params]] <- invert_chf(s, log2 = v[["log2"]],
                                   step = if (v[["bs"]] > 0) v[["bs"]] else NULL, tilt = tilt)
  }
  d <- fits[[r$params]]
  got <- switch(r$quantity,
    step = d@step, x_min = d@x_min,
    tilt_left = d@report$tilt_left, tilt_right = d@report$tilt_right,
    edge_error = d@report$edge_error,
    truncation_error = d@report$truncation_error,
    outside_below = d@report$outside_below, outside_above = d@report$outside_above,
    sum = d@report$total_mass,
    quantile = quantile(d, as.numeric(r$arg)),
    mass = d@masses[as.integer(r$arg) + 1])
  near(got, r$expected, rel = r$rel_tol, abs = r$abs_tol)
}

# The tilt options and the stable law's parameters.
s <- stable_distribution(1.5, 1, 2)
near(s@tail_constant, 1.128379, abs = 1e-6)
stopifnot(s@left_tail_constant == 0, s@mean == 0)
near_abs(stable_distribution(1.5, 1, 2, s@loc0, "S0")@loc, s@loc, 1e-12)
auto <- invert_chf(s, log2 = 12)
off <- invert_chf(s, log2 = 12, tilt = FALSE)
stopifnot(auto@report$tilt_right > 0, off@report$tilt_right == 0,
          auto@report$edge_error < 1e-4, off@report$edge_error > 1e-4)
stopifnot(invert_chf(s, log2 = 12, tilt = 8)@report$tilt_right == 8)
stopifnot(is.nan(invert_chf(gamma_distribution(2, 1), log2 = 10)@report$edge_error))
expect_error_like(invert_chf(gamma_distribution(2, 1), log2 = 10, tilt = 4), "power tail")
split <- invert_chf(stable_distribution(1.5, 0.5), log2 = 10)
stopifnot(split@report$tilt_left > 0, split@report$tilt_right > 0)
expect_error_like(invert_chf(s, tilt = "yes"), "tilt must be")
expect_error_like(stable_distribution(2.5, 0), "alpha")
