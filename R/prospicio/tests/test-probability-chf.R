library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# invert_chf against aggregate 2.1.0's AggregateCHF (#779): the rows of
# validation/reference/chf_aggregate.csv the Rust parity test reads.
ref <- read.csv(validation("reference", "chf_aggregate.csv"),
                stringsAsFactors = FALSE)
params <- function(text) {
  kv <- strsplit(strsplit(text, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.numeric(vapply(kv, `[`, "", 2)), vapply(kv, `[`, "", 1))
}
law <- function(kind, v) {
  switch(kind,
    gamma = gamma_distribution(v[["shape"]], v[["scale"]]),
    normal = normal_distribution(v[["mu"]], v[["sigma"]]),
    inverse_gaussian = inverse_gaussian_distribution(v[["mean"]], v[["shape"]]),
    tweedie = tweedie(v[["mean"]], v[["dispersion"]], v[["power"]]))
}
fits <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  key <- paste(r$distribution, r$params)
  if (is.null(fits[[key]])) {
    v <- params(r$params)
    l <- if (startsWith(r$distribution, "affine_")) {
      base <- v[startsWith(names(v), "base_")]
      names(base) <- sub("^base_", "", names(base))
      affine_distribution(law(sub("^affine_", "", r$distribution), base), v[["scale"]], v[["shift"]])
    } else {
      law(r$distribution, v)
    }
    fits[[key]] <- invert_chf(l, log2 = v[["log2"]], step = if (v[["bs"]] > 0) v[["bs"]] else NULL)
  }
  d <- fits[[key]]
  got <- switch(r$quantity,
    step = d@step, x_min = d@x_min,
    truncation_error = d@report$truncation_error,
    outside_below = d@report$outside_below, outside_above = d@report$outside_above,
    sum = d@report$total_mass, mean = mean(d), sd = sqrt(variance(d)),
    quantile = quantile(d, as.numeric(r$arg)),
    mass = d@masses[as.integer(r$arg) + 1])
  near(got, r$expected, rel = r$rel_tol, abs = r$abs_tol)
}

# Masses are bucket probabilities; a user's function on a given grid.
d <- invert_chf(gamma_distribution(4, 25), log2 = 12)
g <- gamma_distribution(4, 25)
k <- c(11, 401, 1601)
near_abs(d@masses[k], cdf(g, d@xs[k] + 0.125) - cdf(g, d@xs[k] - 0.125), 1e-12)
stopifnot(!d@report$is_truncated, d@report$law_mean == 100)
own <- invert_chf(function(t) (1 - 25i * t)^-4, log2 = 12, step = 0.25, x_min = 0)
near_abs(own@masses, d@masses, 1e-14)
stopifnot(is.nan(own@report$law_mean))
expect_error_like(invert_chf(function(t) (1 - 25i * t)^-4, log2 = 12), "step and x_min")
expect_error_like(invert_chf(function(t) stop("no chf here"), log2 = 4, step = 1, x_min = 0),
                  "no chf here")
expect_error_like(invert_chf(lognormal(0, 1)), "characteristic-function engine")
expect_error_like(invert_chf(gamma_distribution(1, 1), x_min = 0), "give step too")

# A normal reaches below 0, which a grid cannot; a shifted gamma becomes one.
n <- invert_chf(normal_distribution(0, 1), log2 = 12)
stopifnot(n@x_min < -7, abs(quantile(n, 0.5)) < 1e-12)
expect_error_like(chf_to_grid(n), "start")
gr <- chf_to_grid(invert_chf(affine_distribution(g, 1, 1000), log2 = 12))
near_abs(mean(gr), 1100, 1e-6)
