# Double GLMs (a GLM for the mean and a gamma GLM for the dispersion, Smyth
# 1989, Smyth and Verbyla 1999): references for validation/tests/models.rs.
#
# dglm 1.8.6 is the reference where it is right. Its mean step uses the
# working weights w V(mu) / phi and evaluates mu.eta at eta without the
# offset, and its dispersion step evaluates the log link's derivative
# without the dispersion's offset; all are right only for a canonical
# link without offsets
# (knowledge/findings/dglm-mean-step-weights.md). `dglm_fixed` below is
# dglm's loop with those two lines corrected: it reproduces dglm where
# dglm is right, glm() when the dispersion is constant, and, by ML, the
# maximum of the same -2 log-likelihood found directly by nlminb. It is
# the reference for the other cases (a log link on the gamma or Tweedie,
# an offset).
#
# Writes validation/reference/dglm_r.csv on validation/data/glmm_simulated.csv
# (written by r_glmm.R). Run from the repository root:
#   Rscript validation/scripts/r_dglm.R
suppressPackageStartupMessages({
  library(dglm)
  library(statmod)
})
sim <- read.csv("validation/data/glmm_simulated.csv")

# dglm's -2 log-likelihood (the saddlepoint approximation). Under REML,
# dglm means to add log det(X' W X) as 2 log|prod(diag(mfit$R))|, but
# lm.wfit returns no `R`, so the term is 0 and dglm reports this -2
# log-likelihood at the REML estimates; `reml` below adds the term. dglm.constant's
# switch names "Gaussian" and "Poisson", which R's families are not
# ("gaussian", "poisson"), so for every family but the gamma the constant
# is its default, sum(log(2 pi V(y))) with V(0) taken as V(1/6).
m2ll <- function(y, mu, phi, w, family, digamma) {
  d <- family$dev.resids(y, mu, w)
  const <- if (family$family == "Gamma") 2 * sum(log(y)) else {
    v <- family$variance(y)
    v[v == 0] <- family$variance(y[v == 0] + 1 / 6)
    sum(log(2 * pi * v))
  }
  h <- if (digamma) 2 * (lgamma(w / phi) + (1 + log(phi / w)) * w / phi) else log(phi / w)
  const + sum(h + d / phi)
}

# The loop of dglm() with the mean step's working weights
# w mu.eta^2 / (V phi) and mu.eta at the full linear predictor; the
# dispersion's link is the log. Standard errors at the converged values:
# the mean's from (X' W X)^-1, the dispersion's from 2 (Z' Wd Z)^-1, as
# summary.dglm (dispersion 1 and 2).
dglm_fixed <- function(X, Z, y, family, offset = rep(0, length(y)), method = "ml",
                       eps = 1e-14, maxit = 1000, weights = rep(1, length(y)),
                       doffset = rep(0, length(y))) {
  n <- length(y)
  w <- weights
  reml <- method == "reml"
  digamma <- family$family == "Gamma"
  dfam <- Gamma(link = "log")
  # The GLM's fit starts the mean (dglm starts from one least-squares
  # step, which diverges on the Tweedie's pure premiums); the solution
  # does not depend on the start.
  start <- glm.fit(X, y, weights = w, family = family, offset = offset)
  eta <- start$linear.predictors - offset
  mu <- start$fitted.values
  d <- family$dev.resids(y, mu, w)
  deta <- lm.fit(Z, dfam$linkfun(d + (d == 0) / 6) - doffset)$fitted.values + 1.27036 + doffset
  phi <- dfam$linkinv(deta)
  mean_step <- function(eta, mu, phi) {
    me <- family$mu.eta(eta + offset)
    zm <- eta + (y - mu) / me
    wm <- w * me^2 / family$variance(mu) / phi
    lm.wfit(X, zm, wm, method = "qr")
  }
  logdet <- function(mfit) 2 * log(abs(prod(diag(qr.R(mfit$qr)))))
  objective <- function(mu, phi, mfit) {
    m2ll(y, mu, phi, w, family, digamma) + if (reml) logdet(mfit) else 0
  }
  dispersion_step <- function(phi, deta, mfit, d) {
    hdot <- 1 / dfam$mu.eta(deta)
    if (digamma) {
      delta <- 2 * w * (log(w / phi) - base::digamma(w / phi))
      u <- 2 * w^2 * (trigamma(w / phi) - phi / w)
      fdot <- phi^2 / u * hdot
    } else {
      delta <- phi
      u <- phi^2
      fdot <- hdot
    }
    wd <- 1 / (fdot^2 * u)
    if (reml) {
      h <- hat(mfit$qr)
      delta <- delta - phi * h
      wd <- wd - 2 * (h / hdot^2 / phi^2) + h^2
    }
    stopifnot(all(wd > 0))
    list(fit = lm.wfit(Z, deta - doffset + (d - delta) * fdot, wd, method = "qr"), wd = wd)
  }
  mfit <- mean_step(eta, mu, phi)
  eta <- mfit$fitted.values
  mu <- family$linkinv(eta + offset)
  d <- family$dev.resids(y, mu, w)
  m2 <- objective(mu, phi, mfit)
  old <- m2 + 1
  iter <- 0
  # dglm stops on the -2 log-likelihood alone, which pins the coefficients
  # only to about the square root of its tolerance; this loop also waits
  # for them to settle.
  theta <- c(mfit$coefficients, rep(Inf, ncol(Z)))
  moved <- Inf
  while ((abs(old - m2) / (abs(old) + 1) > eps || moved > 1e-12) && iter < maxit) {
    dstep <- dispersion_step(phi, deta, mfit, d)$fit
    deta <- dstep$fitted.values + doffset
    phi <- dfam$linkinv(deta)
    mfit <- mean_step(eta, mu, phi)
    eta <- mfit$fitted.values
    mu <- family$linkinv(eta + offset)
    d <- family$dev.resids(y, mu, w)
    old <- m2
    m2 <- objective(mu, phi, mfit)
    iter <- iter + 1
    now <- c(mfit$coefficients, dstep$coefficients)
    moved <- max(abs(now - theta) / pmax(abs(now), 1))
    theta <- now
  }
  stopifnot(iter < maxit)
  # One more pass of each step at the solution, so that the weights and
  # the coefficients are those of the converged values.
  mfit <- mean_step(eta, mu, phi)
  ds <- dispersion_step(phi, deta, mfit, d)
  W <- w * family$mu.eta(eta + offset)^2 / family$variance(mu) / phi
  list(coef = mfit$coefficients, dcoef = ds$fit$coefficients,
       se = sqrt(diag(solve(crossprod(X * sqrt(W))))),
       dse = sqrt(diag(2 * solve(crossprod(Z * sqrt(ds$wd))))),
       m2loglik = m2ll(y, mu, phi, w, family, digamma), reml = m2ll(y, mu, phi, w, family, digamma) + logdet(mfit))
}

ctl <- dglm.control(epsilon = 1e-14, maxit = 1000)
X <- model.matrix(~ x + urban, sim)
Z <- model.matrix(~ x + urban, sim)
Zu <- model.matrix(~ urban, sim)
Z1 <- model.matrix(~ 1, sim)
near <- function(a, b, tol) {
  err <- max(abs(a - b) / pmax(abs(b), 1))
  if (!(err < tol)) stop(sprintf("off by %g", err))
}

rows <- list()
emit <- function(case, method, fit, src, tol) {
  q <- function(quantity, arg, value, t = tol) {
    rows[[length(rows) + 1]] <<- data.frame(
      case = case, method = method, quantity = quantity, arg = arg,
      expected = sprintf("%.15g", value), abs_tol = 0, rel_tol = t, source = src)
  }
  for (j in seq_along(fit$coef)) {
    q("coef", names(fit$coef)[j], fit$coef[j])
    q("se", names(fit$coef)[j], fit$se[j])
  }
  for (j in seq_along(fit$dcoef)) {
    q("dcoef", names(fit$dcoef)[j], fit$dcoef[j])
    q("dse", names(fit$dcoef)[j], fit$dse[j])
  }
  q("m2loglik", "", fit$m2loglik, 1e-10)
  q("reml_criterion", "", fit$reml, 1e-10)
}
dv <- sprintf("dglm %s corrected (equal to dglm)", packageVersion("dglm"))
fx <- sprintf("dglm %s corrected", packageVersion("dglm"))

# 1. Where dglm is right (canonical links, no offset), the corrected loop
#    is dglm, by ML and REML, to dglm's own precision (about 1e-7: see
#    above); the stored values are the corrected loop's.
for (method in c("ml", "reml")) {
  g <- dglm(severity ~ x + urban, ~ x + urban, family = gaussian, data = sim,
            method = method, control = ctl)
  ours <- dglm_fixed(X, Z, sim$severity, gaussian(), method = method)
  near(ours$coef, coef(g), 1e-6)
  near(ours$dcoef, coef(g$dispersion.fit), 1e-6)
  near(ours$m2loglik, g$m2loglik, 1e-10)
  near(ours$se, summary(g, dispersion = 1)$coefficients[, 2], 1e-6)
  emit("gaussian", method, ours, dv, 1e-8)

  p <- dglm(count ~ x + urban, ~ urban, family = poisson, data = sim, method = method,
            control = ctl)
  ours <- dglm_fixed(X, Zu, sim$count, poisson(), method = method)
  near(ours$coef, coef(p), 1e-6)
  near(ours$dcoef, coef(p$dispersion.fit), 1e-6)
  near(ours$m2loglik, p$m2loglik, 1e-10)
  emit("poisson", method, ours, dv, 1e-8)
}

# Prior weights (the exposure), where dglm is right: the Gaussian.
for (method in c("ml", "reml")) {
  g <- dglm(severity ~ x + urban, ~ x + urban, family = gaussian, data = sim,
            weights = exposure, method = method, control = ctl)
  ours <- dglm_fixed(X, Z, sim$severity, gaussian(), method = method, weights = sim$exposure)
  near(ours$coef, coef(g), 1e-6)
  near(ours$dcoef, coef(g$dispersion.fit), 1e-6)
  near(ours$m2loglik, g$m2loglik, 1e-10)
  emit("gaussian_weighted", method, ours, dv, 1e-8)
}

# 2. With a constant dispersion the corrected loop is glm(), for the
#    gamma's log link and the Poisson with an offset; dglm is not.
tight <- glm.control(epsilon = 1e-15, maxit = 200)
off <- log(sim$exposure)
near(dglm_fixed(X, Z1, sim$severity, Gamma(link = "log"))$coef,
      coef(glm(severity ~ x + urban, Gamma(link = "log"), sim, control = tight)), 1e-8)
near(dglm_fixed(X, Z1, sim$count, poisson(), offset = off)$coef,
      coef(glm(count ~ x + urban + offset(off), poisson, sim, control = tight)), 1e-8)

# 3. By ML the corrected loop is the maximum of the -2 log-likelihood,
#    found directly by nlminb over the mean's and the dispersion's
#    coefficients together.
direct <- function(X, Z, y, family, offset, start, weights = rep(1, length(y)),
                   doffset = rep(0, length(y))) {
  p <- ncol(X)
  digamma <- family$family == "Gamma"
  obj <- function(th) {
    mu <- family$linkinv(drop(X %*% th[1:p]) + offset)
    phi <- exp(drop(Z %*% th[-(1:p)]) + doffset)
    m2ll(y, mu, phi, weights, family, digamma)
  }
  nlminb(start, obj, control = list(rel.tol = 1e-15, x.tol = 1e-12, eval.max = 1e4,
                                    iter.max = 1e4))
}
none <- rep(0, nrow(sim))
cases <- list(
  poisson_offset = list(y = sim$count, family = poisson(), Z = Zu, offset = off),
  gamma = list(y = sim$severity, family = Gamma(link = "log"), Z = Z, offset = none),
  tweedie = list(y = sim$pure, family = tweedie(var.power = 1.5, link.power = 0), Z = Zu,
                 offset = none),
  # An offset on the log dispersion, which dglm mishandles as the mean's.
  gamma_doffset = list(y = sim$severity, family = Gamma(link = "log"), Z = Zu, offset = none,
                       doffset = log(sim$exposure)))
for (case in names(cases)) {
  k <- cases[[case]]
  for (method in c("ml", "reml")) {
    dof <- if (is.null(k$doffset)) none else k$doffset
    fit <- dglm_fixed(X, k$Z, k$y, k$family, offset = k$offset, method = method, doffset = dof)
    if (method == "ml") {
      o <- direct(X, k$Z, k$y, k$family, k$offset, start = c(fit$coef, fit$dcoef) + 0.05,
                  doffset = dof)
      cat(case, "nlminb", o$objective, "ours", fit$m2loglik, "max diff",
          max(abs(o$par - c(fit$coef, fit$dcoef))), "\n")
      stopifnot(o$objective >= fit$m2loglik - 1e-9)
      near(o$par, c(fit$coef, fit$dcoef), 1e-4)
    }
    emit(case, method, fit, fx, 1e-8)
  }
}

out <- do.call(rbind, rows)
con <- file("validation/reference/dglm_r.csv", "w")
writeLines(sprintf("# %s, dglm %s, statmod %s", R.version.string, packageVersion("dglm"),
                   packageVersion("statmod")), con)
write.csv(out, con, row.names = FALSE, quote = FALSE)
close(con)
