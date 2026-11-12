suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Merz and Wuthrich's paid-incurred chain (docs/design/reserving-v02.md,
# decision 32). The Rust core is checked against R ChainLadder's
# PaidIncurredChain in validation/tests/reserving_paid_incurred.rs; these
# tests check the binding's arguments, results, frames, print and errors
# against the same R ChainLadder reference file.



# Tests run from the repository root (CI and `cargo xtask r`).
load_triangle <- function(name) {
  long <- utils::read.csv(validation("data", paste0(name, ".csv")), comment.char = "#")
  triangle(long, "origin", "development", c("paid", "incurred"))
}
lines <- grep("^#", readLines(validation("reference", "reserving_paid_incurred_r.csv")),
              value = TRUE, invert = TRUE)
# The last column, `source`, holds commas: keep the first six fields.
fields <- lapply(strsplit(lines[-1], ",", fixed = TRUE), function(f) c(f, "")[1:6])
ref <- stats::setNames(as.data.frame(do.call(rbind, fields), stringsAsFactors = FALSE),
                       strsplit(lines[1], ",", fixed = TRUE)[[1]][1:6])
ref$expected <- as.numeric(ref$expected)
want <- function(dataset, quantity, arg = "") {
  rows <- ref[ref$dataset == dataset & ref$quantity == quantity & ref$arg %in% arg, ]
  if (nrow(rows) != length(arg)) stop(sprintf("no reference rows %s %s", dataset, quantity))
  rows$expected[match(arg, rows$arg)]
}

# R ChainLadder's convention against R ChainLadder.
for (name in c("usaa", "dahms")) {
  tri <- load_triangle(name)
  fit <- paid_incurred_chain(tri, "paid", "incurred", convention = "r_chainladder")
  stopifnot(S7::S7_inherits(fit, paid_incurred_fit))
  later <- fit@origins[-1]
  near(fit@ultimate[-1], want(name, "ultimate", later), 1e-9, abs = 1e-12)
  near(fit@reserve[-1], want(name, "reserve", later), 1e-9, abs = 1e-12)
  near(fit@std_err[-1], want(name, "std_err", later), 1e-9, abs = 1e-12)
  near(fit@total_reserve, want(name, "total_reserve"), 1e-9, abs = 1e-12)
  near(fit@total_std_err, want(name, "total_std_err"), 1e-9, abs = 1e-12)
  near(fit@total_ultimate - fit@ultimate[1], want(name, "ultimate_after_first"), 1e-9, abs = 1e-12)
  n <- length(fit@development)
  near(fit@sigma, want(name, "sigma", as.character(0:(n - 1))), 1e-9, abs = 1e-12)
  near(fit@tau, want(name, "tau", as.character(0:(n - 2))), 1e-9, abs = 1e-12)
  near(fit@phi, want(name, "phi", as.character(0:(n - 1))), 1e-9, abs = 1e-12)
  near(fit@psi, want(name, "psi", as.character(0:(n - 2))), 1e-9, abs = 1e-12)
  near(fit@phi_sd, want(name, "phi_sd", as.character(0:(n - 1))), 1e-9, abs = 1e-12)
  near(fit@psi_sd, want(name, "psi_sd", as.character(0:(n - 2))), 1e-9, abs = 1e-12)
  # R's beta_j weighs the origin whose latest age is j.
  near(rev(fit@incurred_weight[-1]), want(name, "beta", as.character(0:(n - 2))), 1e-9, abs = 1e-12)
}

# The default, the model's posterior: about 0.5% above R's on USAA.
tri <- load_triangle("usaa")
fit <- paid_incurred_chain(tri, "paid", "incurred")
r <- paid_incurred_chain(tri, "paid", "incurred", convention = "r_chainladder")
stopifnot(abs(fit@total_reserve / r@total_reserve - 1 - 0.0050) < 1e-4,
          abs(fit@total_std_err / r@total_std_err - 1 - 0.0049) < 1e-4,
          fit@reserve[[1]] == 0, fit@std_err[[1]] == 0,
          identical(fit@sigma, r@sigma),
          identical(names(fit@reserve), fit@origins),
          identical(names(fit@tau), as.character(fit@development[-10])),
          all(dim(fit@posterior_covariance) == c(19, 19)))
near(sqrt(fit@posterior_covariance["phi_12", "phi_12"]), fit@phi_sd[["12"]], 1e-9, abs = 1e-12)

# Settings reach the fit.
given <- paid_incurred_chain(tri, "paid", "incurred", sigma = fit@sigma, tau = fit@tau)
stopifnot(identical(given@total_reserve, fit@total_reserve))
mack <- paid_incurred_chain(load_triangle("dahms"), "paid", "incurred", extrapolation = "mack")
stopifnot(mack@sigma[[10]] == mack@sigma[[8]])
vague <- paid_incurred_chain(tri, "paid", "incurred", phi_mean = rep(0, 10),
                             phi_sd = rep(Inf, 10), psi_mean = rep(0, 9), psi_sd = rep(Inf, 9))
stopifnot(identical(vague@total_reserve, fit@total_reserve))
tight <- paid_incurred_chain(tri, "paid", "incurred", phi_mean = unname(fit@phi),
                             phi_sd = rep(1e-9, 10))
near(tight@phi, fit@phi, rel = 0, abs = 1e-12)
stopifnot(tight@total_std_err < fit@total_std_err)

# Frames and print.
long <- as.data.frame(fit)
stopifnot(identical(names(long), c("origin", "latest", "latest_incurred", "ultimate", "reserve",
                                   "incurred_weight", "std_err")),
          nrow(long) == 10)
near(long$reserve, fit@reserve, 1e-9, abs = 1e-12)
near(totals_frame(fit)$std_err, fit@total_std_err, 1e-9, abs = 1e-12)
dev <- development_frame(fit)
stopifnot(identical(dev$development, fit@development), is.na(dev$tau[10]))
out <- utils::capture.output(print(fit))
stopifnot(grepl("<paid_incurred_fit> total reserve", out[1], fixed = TRUE))

# Errors.
expect_error_like(paid_incurred_chain(tri, "paid", "incurred", extrapolation = "bondy"),
                  "'arg' should be one of")
expect_error_like(paid_incurred_chain(tri, "paid", "incurred", phi_mean = rep(0, 10)),
                  "give both phi_mean and phi_sd")
expect_error_like(paid_incurred_chain(tri, "paid", "incurred", sigma = rep(0.1, 9)),
                  "sigma needs 10 values")
expect_error_like(paid_incurred_chain(tri, "paid", "missing"), "missing")
expect_error_like(paid_incurred_chain(tri, 1, "incurred"), "paid must be the name of one column")

cat("test-reserving-paid-incurred.R: ok\n")
