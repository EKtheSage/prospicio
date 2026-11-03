# Regenerate validation/reference/reserving_curves_r.csv: development curves
# fitted over every age (prospicio_reserving::DevelopmentCurve,
# docs/design/reserving-v02.md decision 31) by R's lm and nls, on the
# volume-weighted factors of RAA and GenIns.
#
# Run from the repository root:
#
#     Rscript validation/scripts/reserving_curves_r.R
#
# `k` is the 1-based development index; a growth curve reads the ages in
# months. Every factor above 1.00001 enters the fit (all of them on RAA and
# GenIns). `arg` is the 0-based development position for per-age
# quantities and empty otherwise. Residuals, the sum of squares and
# R-squared are on the factor scale, over the points of the fit. `tail` is
# the product of the curve's factors over the 100 periods past the oldest
# age.

suppressPackageStartupMessages(library(ChainLadder))

OUT <- "validation/reference/reserving_curves_r.csv"
DATASETS <- c("raa", "genins")

read_triangle <- function(name) {
  d <- read.csv(file.path("validation/data", paste0(name, ".csv")), comment.char = "#")
  origins <- sort(unique(d$origin))
  devs <- sort(unique(d$development))
  tri <- matrix(NA_real_, length(origins), length(devs),
                dimnames = list(origin = origins, dev = devs))
  tri[cbind(match(d$origin, origins), match(d$development, devs))] <- d$value
  as.triangle(tri)
}

# Volume-weighted age-to-age factors, as chainladder(tri)'s.
volume_factors <- function(tri) {
  n <- ncol(tri)
  sapply(seq_len(n - 1), function(j) {
    ok <- !is.na(tri[, j]) & !is.na(tri[, j + 1])
    sum(tri[ok, j + 1]) / sum(tri[ok, j])
  })
}

rows <- character(0)
emit <- function(dataset, method, quantity, arg, value, rel) {
  if (!is.finite(value)) stop(sprintf("%s %s %s %s is not finite", dataset, method, quantity, arg))
  rows <<- c(rows, sprintf("%s,%s,%s,%s,%s,0,%s,R %s.%s", dataset, method, quantity, arg,
                           sprintf("%.15g", value), format(rel), R.version$major, R.version$minor))
}

growth <- function(curve) {
  if (curve == "loglogistic") function(x, omega, theta) 1 / (1 + (theta / x)^omega)
  else function(x, omega, theta) 1 - exp(-(x / theta)^omega)
}

tol <- nls.control(maxiter = 1000)

for (name in DATASETS) {
  tri <- read_triangle(name)
  ages <- as.numeric(colnames(tri))
  f <- volume_factors(tri)
  n <- length(f)
  k <- seq_len(n)
  d <- data.frame(k = k, f = f, y = f - 1, t0 = ages[k], t1 = ages[k + 1])
  step <- ages[n + 1] - ages[n]
  t_at <- function(p) ifelse(p <= n, ages[pmin(p + 1, n + 1)], ages[n + 1] + step * (p - n))
  kk <- n + seq_len(100)   # 1-based indices of the 100 factors past the oldest age
  fits <- list()

  e <- lm(log(y) ~ k, d)
  fits$exponential <- list(par = c(a = unname(coef(e)[1]), b = unname(coef(e)[2])),
                           rel = 1e-10,
                           f = function(k, p) 1 + exp(p[1] + p[2] * k))
  ip <- lm(log(y) ~ log(k), d)
  fits$inverse_power <- list(par = c(a = unname(coef(ip)[1]), b = unname(coef(ip)[2])),
                             rel = 1e-10,
                             f = function(k, p) 1 + exp(p[1] + p[2] * log(k)))
  h <- lm(log(y) ~ k + log(k), d)
  fits$hoerl <- list(par = c(a = unname(coef(h)[1]), b = unname(coef(h)[2]),
                             c = unname(coef(h)[3])),
                     rel = 1e-9,
                     f = function(k, p) 1 + exp(p[1] + p[2] * k + p[3] * log(k)))
  # Sherman: nonlinear least squares on f - 1, from the inverse power (c = 0).
  s <- nls(y ~ a * (k + c)^(-b), d,
           start = list(a = exp(unname(coef(ip)[1])), b = -unname(coef(ip)[2]), c = 0),
           algorithm = "port", lower = c(0, -Inf, -0.999999), control = tol)
  fits$sherman <- list(par = coef(s), rel = 1e-6,
                       f = function(k, p) 1 + p[1] * (k + p[3])^(-p[2]))
  for (curve in c("loglogistic", "weibull")) {
    G <- growth(curve)
    # A start from optim on the log parameters, then nls on the factors.
    sse <- function(q) sum((d$f - G(d$t1, exp(q[1]), exp(q[2])) / G(d$t0, exp(q[1]), exp(q[2])))^2)
    o <- optim(c(log(1.5), log(median(ages))), sse, control = list(reltol = 1e-14, maxit = 5000))
    g <- nls(f ~ G(t1, omega, theta) / G(t0, omega, theta), d,
             start = list(omega = exp(o$par[1]), theta = exp(o$par[2])),
             algorithm = "port", lower = c(1e-6, 1e-6), control = tol)
    fits[[curve]] <- local({
      GG <- G
      list(par = coef(g), rel = 1e-6,
           f = function(k, p) GG(t_at(k), p[1], p[2]) / GG(t_at(k - 1), p[1], p[2]))
    })
  }

  for (method in names(fits)) {
    ft <- fits[[method]]
    p <- unname(ft$par)
    for (i in seq_along(p)) emit(name, method, paste0("param_", names(ft$par)[i]), "", p[i], ft$rel)
    fitted <- ft$f(k, p)
    for (j in k) emit(name, method, "fitted", j - 1, fitted[j], ft$rel)
    res <- f - fitted
    sse <- sum(res^2)
    emit(name, method, "sum_of_squares", "", sse, ft$rel * 10)
    emit(name, method, "r_squared", "", 1 - sse / sum((f - mean(f))^2), ft$rel)
    emit(name, method, "tail", "", prod(ft$f(kk, p)), ft$rel)
  }
}

header <- c(
  sprintf("# Generated by validation/scripts/reserving_curves_r.R with R %s.%s, ChainLadder %s.",
          R.version$major, R.version$minor, packageVersion("ChainLadder")),
  "# Volume-weighted factors of validation/data/{raa,genins}.csv. exponential, inverse_power and",
  "# hoerl: lm on log(f - 1); sherman: nls on f - 1 = a (k + c)^(-b); loglogistic and weibull:",
  "# nls on f = G(t1) / G(t0), ages in months. nls by the port algorithm, 1000 iterations.",
  "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source")
writeLines(c(header, rows), OUT)
cat(sprintf("wrote %d rows to %s\n", length(rows), OUT))
