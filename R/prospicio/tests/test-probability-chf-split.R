library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# invert_chf on stable laws with two power tails, split and tilted (#780), against
# aggregate 2.1.0's AggregateCHF: the rows of
# validation/reference/chf_split_aggregate.csv the Rust parity test reads.
ref <- read.csv(validation("reference", "chf_split_aggregate.csv"), stringsAsFactors = FALSE)
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
