library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# TVaR of a chf_distribution (#779, #780) against aggregate 2.1.0's
# AggregateCHF.tvar: the rows of validation/reference/chf_tvar_aggregate.csv
# the Rust parity test reads.
ref <- read.csv(validation("reference", "chf_tvar_aggregate.csv"), stringsAsFactors = FALSE)
params <- function(text) {
  kv <- strsplit(strsplit(text, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.numeric(vapply(kv, `[`, "", 2)), vapply(kv, `[`, "", 1))
}
law <- function(kind, v) {
  switch(kind,
    gamma = gamma_distribution(v[["shape"]], v[["scale"]]),
    normal = normal_distribution(v[["mu"]], v[["sigma"]]),
    inverse_gaussian = inverse_gaussian_distribution(v[["mean"]], v[["shape"]]),
    tweedie_power = tweedie_power_distribution(v[["power"]], v[["mean"]], v[["dispersion"]]),
    normal_inverse_gaussian = normal_inverse_gaussian_distribution(v[["alpha"]], v[["beta"]],
                                                                   v[["delta"]], v[["mu"]]),
    variance_gamma = variance_gamma_distribution(v[["sigma"]], v[["nu"]], v[["theta"]], v[["mu"]]),
    tempered_stable = tempered_stable_distribution(v[["C"]], v[["G"]], v[["M"]], v[["Y"]],
                                                   v[["mu"]]),
    stable = stable_distribution(v[["alpha"]], v[["beta"]], v[["scale"]], v[["loc"]],
                                 if (v[["parameterization"]] == 1) "S0" else "S1"))
}
stopifnot(nrow(ref) == 84)
fits <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  key <- paste(r$distribution, r$params)
  if (is.null(fits[[key]])) {
    v <- params(r$params)
    fits[[key]] <- invert_chf(law(r$distribution, v), log2 = v[["log2"]])
  }
  near(TVaR(fits[[key]], as.numeric(r$arg)), r$expected, rel = r$rel_tol, abs = r$abs_tol)
}

# At least the VaR, vectorised, the largest point of positive mass at 1.
d <- invert_chf(gamma_distribution(4, 25), log2 = 12)
p <- c(0, 0.5, 0.99)
stopifnot(all(TVaR(d, p) >= VaR(d, p)))
stopifnot(TVaR(d, 1) == max(d@xs[d@masses > 0]))
expect_error_like(TVaR(d, 1.5), "probab")
