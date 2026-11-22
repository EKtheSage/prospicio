library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# invert_chf on the normal inverse Gaussian, variance gamma and CGMY
# tempered stable laws (#780), against aggregate 2.1.0's AggregateCHF: the
# rows of validation/reference/chf_levy_aggregate.csv the Rust parity test
# reads.
ref <- read.csv(validation("reference", "chf_levy_aggregate.csv"), stringsAsFactors = FALSE)
params <- function(text) {
  kv <- strsplit(strsplit(text, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.numeric(vapply(kv, `[`, "", 2)), vapply(kv, `[`, "", 1))
}
law <- function(kind, v) {
  switch(kind,
    normal_inverse_gaussian = normal_inverse_gaussian_distribution(v[["alpha"]], v[["beta"]],
                                                                   v[["delta"]], v[["mu"]]),
    variance_gamma = variance_gamma_distribution(v[["sigma"]], v[["nu"]], v[["theta"]], v[["mu"]]),
    tempered_stable = tempered_stable_distribution(v[["C"]], v[["G"]], v[["M"]], v[["Y"]],
                                                   v[["mu"]]))
}
fits <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  key <- paste(r$distribution, r$params)
  if (is.null(fits[[key]])) {
    v <- params(r$params)
    fits[[key]] <- invert_chf(law(r$distribution, v), log2 = v[["log2"]])
  }
  d <- fits[[key]]
  # aggregate does not take out the atom of a CGMY law with Y < 0 (#944): its
  # masses ring near mu and its truncation includes the atom. Those rows are
  # left out (the truncation row, the masses within 400 buckets of mu at
  # bucket 2110) and the others compare at 1e-8.
  atom_law <- r$distribution == "tempered_stable" && params(r$params)[["Y"]] < 0
  if (atom_law && (r$quantity == "truncation_error" ||
                   (r$quantity == "mass" && abs(as.integer(r$arg) - 2110L) < 400L))) {
    next
  }
  got <- switch(r$quantity,
    step = d@step, x_min = d@x_min,
    truncation_error = d@report$truncation_error,
    outside_below = d@report$outside_below, outside_above = d@report$outside_above,
    sum = d@report$total_mass, mean = mean(d), sd = sqrt(variance(d)),
    quantile = quantile(d, as.numeric(r$arg)),
    mass = d@masses[as.integer(r$arg) + 1])
  tol <- if (atom_law && r$quantity %in% c("mass", "mean", "sd")) 1e-8 else r$abs_tol
  near(got, r$expected, rel = r$rel_tol, abs = tol)
}

# Parameters and moments.
nig <- normal_inverse_gaussian_distribution(2, 0.5, 1.5, 0.3)
g <- sqrt(4 - 0.25)
near(nig@mean, 0.3 + 1.5 * 0.5 / g)
near(nig@variance, 1.5 * 4 / g^3)
stopifnot(nig@alpha == 2, nig@mu == 0.3)
vg <- variance_gamma_distribution(0.2, 0.5, -0.1)
near(vg@variance, 0.045)
ts <- tempered_stable_distribution(1, 5, 10, 0.5)
stopifnot(ts@C == 1, ts@G == 5, ts@M == 10, ts@Y == 0.5, ts@mu == 0)
expect_error_like(normal_inverse_gaussian_distribution(1, 1, 1), "beta")
expect_error_like(variance_gamma_distribution(1, 0), "nu")
expect_error_like(tempered_stable_distribution(1, 1, 1, 1), "Y")
expect_error_like(invert_chf(vg, tilt = 3), "power tail")
