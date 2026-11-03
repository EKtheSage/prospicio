# Quantile and expectile regression (#550): references for
# validation/tests/models.rs from R quantreg (rq, summary.rq) and expectreg
# (expectreg.ls).
#
# Data: quantreg's engel (Engel's 235 Belgian households, food expenditure
# against income), written to validation/data/engel.csv, and a seeded book
# of 400 losses with a heteroscedastic, skewed response and prior weights,
# written to validation/data/quantile_book.csv.
#
# Quantile regression: rq(method = "br") (Barrodale and Roberts' simplex,
# the exact vertex) for the coefficients and the objective, rq(method =
# "fn") (the Frisch-Newton interior point, which stops at a duality gap of
# 1e-6) at a loose tolerance, and summary.rq(se = "ker") for Powell's
# kernel standard errors with Hall and Sheather's bandwidth (hs = TRUE)
# and Bofinger's (hs = FALSE).
#
# Expectile regression: expectreg.ls(estimate = "laws") on a parametric
# term (expectreg centres the covariate, so its intercept is the expectile
# at the mean income: the uncentred intercept is recorded), and on
# expectreg's own P-spline basis at a fixed lambda = 5, whose design and
# penalty are written to validation/data/expectreg_engel_basis.csv and
# expectreg_engel_penalty.csv. Asymmetric least squares written out here
# reproduces both and gives the sandwich standard errors
# A^-1 B A^-1, A = X' W X (+ lambda P), B = X' (W e)^2 X, which expectreg
# does not report; at tau = 0.5 they are lm's HC0.
#
# Run from the repository root (expectreg from CRAN):
#   Rscript validation/scripts/r_quantile_expectile.R
suppressPackageStartupMessages({
  library(quantreg)
  library(expectreg)
})

data(engel, package = "quantreg")
write.csv(engel, "validation/data/engel.csv", row.names = FALSE, quote = FALSE)

set.seed(550)
n <- 400
book <- data.frame(x = round(runif(n, 0, 10), 3), z = rbinom(n, 1, 0.3),
                   w = sample(1:3, n, replace = TRUE))
book$y <- round(exp(1 + 0.15 * book$x + 0.4 * book$z + (0.4 + 0.05 * book$x) * rnorm(n)), 4)
write.csv(book, "validation/data/quantile_book.csv", row.names = FALSE, quote = FALSE)

rows <- list()
add <- function(case, quantity, arg, expected, rel_tol, source, abs_tol = 0) {
  rows[[length(rows) + 1]] <<- data.frame(
    case = case, quantity = quantity, arg = arg,
    expected = sprintf("%.15g", expected), abs_tol = abs_tol, rel_tol = rel_tol,
    source = gsub(",", ";", source)
  )
}
qv <- paste0("quantreg ", packageVersion("quantreg"))
ev <- paste0("expectreg ", packageVersion("expectreg"))

rq_case <- function(case, formula, data, tau, weights = NULL) {
  data$.w <- if (is.null(weights)) 1 else weights
  br <- rq(formula, tau = tau, data = data, weights = .w, method = "br")
  fn <- rq(formula, tau = tau, data = data, weights = .w, method = "fn")
  ker <- summary(br, se = "ker", hs = TRUE)$coefficients
  bof <- summary(br, se = "ker", hs = FALSE)$coefficients
  for (nm in names(coef(br))) {
    add(case, "coef", nm, coef(br)[[nm]], 1e-10, paste(qv, "rq br"), abs_tol = 1e-12)
    add(case, "coef_fn", nm, coef(fn)[[nm]], 1e-5, paste(qv, "rq fn (duality gap 1e-6)"),
        abs_tol = 1e-6)
    add(case, "se", nm, ker[nm, "Std. Error"], 1e-9, paste(qv, "summary.rq ker, hs = TRUE"))
    add(case, "se_bofinger", nm, bof[nm, "Std. Error"], 1e-9,
        paste(qv, "summary.rq ker, hs = FALSE"))
  }
  # rq's rho is the check loss without the weights; the weighted loss is
  # the one minimised.
  rho <- function(r) r * (tau - (r < 0))
  r <- drop(data[[all.vars(formula)[1]]] - model.matrix(formula, data) %*% coef(br))
  stopifnot(abs(br$rho - sum(rho(r))) < 1e-9 * br$rho)
  add(case, "objective", "", sum(data$.w * rho(r)), 1e-10,
      paste(qv, "rq br, the weighted check loss at its estimates"))
}
for (tau in c(0.1, 0.25, 0.5, 0.75, 0.9)) {
  rq_case(paste0("engel_", tau), foodexp ~ income, engel, tau)
}
for (tau in c(0.5, 0.95)) {
  rq_case(paste0("book_", tau), y ~ x + z, book, tau, weights = book$w)
}

# Asymmetric least squares and its sandwich, written out.
als <- function(X, y, tau, pen = 0) {
  w <- rep(0.5, length(y))
  for (k in 1:200) {
    b <- solve(crossprod(X, w * X) + pen, crossprod(X, w * y))
    nw <- ifelse(drop(y - X %*% b) < 0, 1 - tau, tau)
    if (all(nw == w)) break
    w <- nw
  }
  e <- drop(y - X %*% b)
  a_inv <- solve(crossprod(X, w * X) + pen)
  list(coef = drop(b), cov = a_inv %*% crossprod(X, (w * e)^2 * X) %*% a_inv)
}

X <- cbind(1, engel$income)
for (tau in c(0.1, 0.5, 0.9)) {
  case <- paste0("engel_", tau)
  m <- expectreg.ls(foodexp ~ rb(income, "parametric"), data = engel, estimate = "laws",
                    expectiles = tau, smooth = "fixed")
  slope <- m$coefficients$income[1, 1]
  intercept <- m$intercepts[1] - slope * mean(engel$income)
  ours <- als(X, engel$foodexp, tau)
  stopifnot(abs(ours$coef - c(intercept, slope)) < 1e-8 * abs(c(intercept, slope)))
  add(case, "ecoef", "(Intercept)", intercept, 1e-9, paste(ev, "expectreg.ls laws, uncentred"))
  add(case, "ecoef", "income", slope, 1e-9, paste(ev, "expectreg.ls laws"))
  se <- sqrt(diag(ours$cov))
  add(case, "ese", "(Intercept)", se[1], 1e-9, "asymmetric least squares sandwich (R)")
  add(case, "ese", "income", se[2], 1e-9, "asymmetric least squares sandwich (R)")
}
if (requireNamespace("sandwich", quietly = TRUE)) {
  hc0 <- sqrt(diag(sandwich::vcovHC(lm(foodexp ~ income, engel), type = "HC0")))
  stopifnot(abs(hc0 - sqrt(diag(als(X, engel$foodexp, 0.5)$cov))) < 1e-9 * hc0)
}

m <- expectreg.ls(foodexp ~ rb(income, "pspline"), data = engel, estimate = "laws",
                  expectiles = c(0.2, 0.8), smooth = "fixed", lambda = 5)
B <- m$design
P <- m$bases$income$P
colnames(B) <- paste0("b", seq_len(ncol(B)) - 1)
write.csv(B, "validation/data/expectreg_engel_basis.csv", row.names = FALSE, quote = FALSE)
write.csv(P, "validation/data/expectreg_engel_penalty.csv", row.names = FALSE, quote = FALSE)
k <- ncol(B)
pen <- matrix(0, k, k)
pen[2:k, 2:k] <- 5 * P
for (j in 1:2) {
  tau <- c(0.2, 0.8)[j]
  case <- paste0("engel_pspline_", tau)
  co <- c(m$intercepts[j], m$coefficients$income[, j])
  ours <- als(B, engel$foodexp, tau, pen)
  stopifnot(max(abs(ours$coef - co)) < 1e-8 * max(abs(co)))
  for (i in seq_len(k)) {
    add(case, "ecoef", colnames(B)[i], co[i], 1e-8, paste(ev, "expectreg.ls laws, lambda 5"),
        abs_tol = 1e-8 * max(abs(co)))
  }
  fit <- drop(B %*% co)
  for (i in c(1, 50, 100, 200)) {
    add(case, "efitted", as.character(i), m$fitted[i, j], 1e-9, paste(ev, "expectreg.ls fitted"))
  }
}

out <- do.call(rbind, rows)
con <- file("validation/reference/quantile_expectile_r.csv", "w")
writeLines(sprintf("# %s, quantreg %s, expectreg %s", R.version.string,
                   packageVersion("quantreg"), packageVersion("expectreg")), con)
write.csv(out, con, row.names = FALSE, quote = FALSE)
close(con)
cat(nrow(out), "cases\n")
