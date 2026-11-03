# Zero-inflated and hurdle count models (#548): references for
# validation/tests/models.rs from pscl's zeroinfl() and hurdle().
#
# Two data sets: pscl's bioChemists (Long 1990, 915 PhD biochemists'
# article counts), written to validation/data/biochemists.csv, and a seeded
# book of 2,000 policies with exposure, a structural zero share that
# depends on a covariate and prior weights, written to
# validation/data/zero_book.csv.
#
# pscl maximises the likelihood by BFGS and takes the standard errors from
# optim's finite-difference Hessian (step 1e-3). Even at a relative
# tolerance of 1e-15 from the EM start its BFGS stops short of the maximum
# (up to 2e-6 in a coefficient), so each fit is refined by Newton's method on pscl's
# likelihood (an analytic gradient checked against numDeriv's) and the
# reference is that maximum, the inverse of the Hessian there (numDeriv's
# Jacobian of the gradient) and pscl's predictions at it; pscl's own
# estimates and standard errors are kept at loose tolerances. The Vuong
# statistics are pscl::vuong's (raw, AIC- and BIC-corrected), whose
# parameter counts are the coefficients only (no theta).
#
# Run from the repository root:
#   Rscript validation/scripts/r_zero_counts.R
suppressPackageStartupMessages({
  library(pscl)
  library(MASS)
  library(numDeriv)
})

data("bioChemists", package = "pscl")
bio <- bioChemists
bio$fem <- as.integer(bio$fem == "Women")
bio$mar <- as.integer(bio$mar == "Married")
write.csv(bio, "validation/data/biochemists.csv", row.names = FALSE, quote = FALSE)

set.seed(548)
n <- 2000
book <- data.frame(
  x = round(rnorm(n), 4),
  urban = rbinom(n, 1, 0.4),
  exposure = round(runif(n, 0.2, 1.5), 3),
  weight = sample(1:3, n, replace = TRUE)
)
mu <- book$exposure * exp(-0.3 + 0.4 * book$x + 0.5 * book$urban)
pi0 <- plogis(-0.8 + 0.6 * book$x)
size <- 1.8
count <- rnbinom(n, mu = mu, size = size)
book$y <- ifelse(runif(n) < pi0, 0L, count)
write.csv(book, "validation/data/zero_book.csv", row.names = FALSE, quote = FALSE)

rows <- list()
add <- function(case, quantity, arg, expected, rel_tol, source, abs_tol = 0) {
  rows[[length(rows) + 1]] <<- data.frame(
    case = case, quantity = quantity, arg = arg,
    expected = sprintf("%.15g", expected), abs_tol = abs_tol, rel_tol = rel_tol,
    source = gsub(",", ";", source)
  )
}

# Whether a zeroinfl (dist a string) or hurdle (dist a list) fit
# counts by the negative binomial.
is_negbin <- function(fit) {
  d <- if (is.list(fit$dist)) fit$dist$count else fit$dist
  d == "negbin"
}

# The log-likelihood of a zeroinfl or hurdle fit at a full parameter
# vector (count, zero, log theta), rebuilt from the fit's own model
# matrices, and its analytic gradient (the weighted score, theta's
# included). `refine` checks the gradient against numDeriv's of the
# likelihood.
model_funs <- function(fit, type) {
  X <- model.matrix(fit, model = "count")
  Z <- model.matrix(fit, model = "zero")
  y <- fit$y
  w <- fit$weights
  off_x <- if (is.null(fit$offset$count)) 0 else fit$offset$count
  off_z <- if (is.null(fit$offset$zero)) 0 else fit$offset$zero
  kx <- ncol(X); kz <- ncol(Z)
  link <- make.link(fit$link)
  negbin <- is_negbin(fit)
  unpack <- function(par) {
    mu <- drop(exp(X %*% par[1:kx] + off_x))
    g <- drop(Z %*% par[kx + 1:kz] + off_z)
    th <- if (negbin) exp(par[kx + kz + 1]) else NA
    list(mu = mu, h = link$linkinv(g), a = link$mu.eta(g), th = th)
  }
  lf <- function(k, mu, th) {
    if (negbin) dnbinom(k, size = th, mu = mu, log = TRUE) else dpois(k, mu, log = TRUE)
  }
  # d log f(k) / d eta and d log f(k) / d log theta.
  dlf <- function(k, mu, th) {
    if (!negbin) return(list(eta = k - mu, th = 0 * k))
    list(eta = th * (k - mu) / (th + mu),
         th = th * (digamma(k + th) - digamma(th) + log(th / (th + mu)) + (mu - k) / (th + mu)))
  }
  ll <- function(par) {
    u <- unpack(par)
    v <- if (type == "zeroinfl") {
      ifelse(y == 0, log(u$h + (1 - u$h) * exp(lf(0, u$mu, u$th))),
             log(1 - u$h) + lf(y, u$mu, u$th))
    } else {
      ifelse(y == 0, log(1 - u$h),
             log(u$h) + lf(y, u$mu, u$th) - log(1 - exp(lf(0, u$mu, u$th))))
    }
    sum(w * v)
  }
  gr <- function(par) {
    u <- unpack(par)
    f0 <- exp(lf(0, u$mu, u$th))
    d0 <- dlf(0, u$mu, u$th)
    dy <- dlf(y, u$mu, u$th)
    if (type == "zeroinfl") {
      p0 <- u$h + (1 - u$h) * f0
      share <- (1 - u$h) * f0 / p0
      de <- ifelse(y == 0, share * d0$eta, dy$eta)
      dz <- ifelse(y == 0, u$a * (1 - f0) / p0, -u$a / (1 - u$h))
      dt <- ifelse(y == 0, share * d0$th, dy$th)
    } else {
      odds <- f0 / (1 - f0)
      de <- ifelse(y == 0, 0, dy$eta + odds * d0$eta)
      dz <- ifelse(y == 0, -u$a / (1 - u$h), u$a / u$h)
      dt <- ifelse(y == 0, 0, dy$th + odds * d0$th)
    }
    out <- c(colSums(w * de * X), colSums(w * dz * Z))
    if (negbin) out <- c(out, sum(w * dt))
    out
  }
  list(ll = ll, gr = gr)
}

# Newton's method on pscl's likelihood from its estimates, on the analytic
# gradient with the Hessian its numDeriv Jacobian, until the step is below
# 1e-13: the maximum that pscl's BFGS stops short of. The
# gradient is first checked against numDeriv's gradient of the likelihood.
refine <- function(f, par) {
  num <- numDeriv::grad(f$ll, par)
  stopifnot(max(abs(num - f$gr(par)) / pmax(1, abs(num))) < 1e-5)
  for (k in 1:50) {
    step <- solve(-numDeriv::jacobian(f$gr, par), f$gr(par))
    par <- par + step
    if (max(abs(step)) < 1e-13) break
  }
  par
}

# `weighted`: pscl's negative binomial score for theta (gradNegBin's
# wres_theta) leaves out the prior weights, so with weights its estimates
# and its Hessian are not the weighted maximum's
# (knowledge/findings/pscl-weighted-theta-score.md); pscl's own standard
# errors are then left out.
record <- function(case, fit, type, data, weighted = FALSE) {
  src <- paste0("pscl ", packageVersion("pscl"), " ", type)
  max_src <- paste(src, "likelihood maximised by Newton")
  kx <- length(fit$coefficients$count)
  kz <- length(fit$coefficients$zero)
  co <- c(fit$coefficients$count, fit$coefficients$zero)
  nm <- c(paste0("count_", names(fit$coefficients$count)),
          paste0("zero_", names(fit$coefficients$zero)))
  negbin <- is_negbin(fit)
  par0 <- co
  if (negbin) par0 <- c(par0, log(fit$theta))
  f <- model_funs(fit, type)
  ll <- f$ll
  par <- refine(f, par0)
  se_pscl <- sqrt(diag(vcov(fit)))
  se_exact <- sqrt(diag(solve(-numDeriv::jacobian(f$gr, par))))
  for (j in seq_along(co)) {
    add(case, "coef", nm[j], par[j], 1e-9, max_src, abs_tol = 1e-11)
    add(case, "coef_pscl", nm[j], co[j], 1e-4, src, abs_tol = 1e-6)
    if (!weighted) {
      add(case, "se_pscl", nm[j], se_pscl[j], 2e-3, paste(src, "(optim Hessian, step 1e-3)"))
    }
    add(case, "se", nm[j], se_exact[j], 1e-7, paste(max_src, "and its Hessian"))
  }
  if (negbin) {
    add(case, "theta", "", exp(par[length(par)]), 1e-9, max_src)
    add(case, "theta_pscl", "", fit$theta, 1e-4, src)
    add(case, "se_log_theta", "", se_exact[length(par)], 1e-7, paste(max_src, "and its Hessian"))
  }
  add(case, "loglik", "", ll(par), 1e-12, max_src)
  add(case, "loglik_pscl", "", as.numeric(logLik(fit)), 1e-9, src)
  add(case, "aic", "", AIC(fit), 1e-9, src)
  add(case, "bic", "", BIC(fit), 1e-9, src)
  # pscl's predictions at the maximum.
  at <- fit
  at$coefficients$count[] <- par[1:kx]
  at$coefficients$zero[] <- par[kx + 1:kz]
  if (negbin) at$theta <- exp(par[length(par)])
  pr <- predict(at, newdata = data, type = "response")
  probs <- predict(at, newdata = data, type = "prob", at = 0:4)
  for (i in c(1, 2, 3, 10, 100)) {
    add(case, "mean", as.character(i), pr[i], 1e-8, paste(max_src, "and pscl's predict"))
    add(case, "prob_zero", as.character(i), probs[i, 1], 1e-8, paste(max_src, "and pscl's predict"))
  }
  for (k in 1:4) add(case, "prob", paste0("1:", k), probs[1, k + 1], 1e-8, paste(max_src, "and pscl's predict"))
  invisible(at)
}

tight <- function(...) zeroinfl.control(reltol = 1e-15, EM = TRUE, ...)
htight <- hurdle.control(reltol = 1e-15)

f_bio <- art ~ fem + mar + kid5 + phd + ment
zip <- record("bio_zip", zeroinfl(f_bio, data = bio, dist = "poisson", control = tight()), "zeroinfl", bio)
zinb <- record("bio_zinb", zeroinfl(f_bio, data = bio, dist = "negbin", control = tight()), "zeroinfl", bio)
record("bio_zip_probit", zeroinfl(art ~ fem + mar + kid5 + phd + ment | ment, data = bio,
                                  dist = "poisson", link = "probit", control = tight()), "zeroinfl", bio)
record("bio_hp", hurdle(f_bio, data = bio, dist = "poisson", control = htight), "hurdle", bio)
hnb <- record("bio_hnb", hurdle(f_bio, data = bio, dist = "negbin", control = htight), "hurdle", bio)
record("bio_hnb_cloglog", hurdle(art ~ fem + mar + kid5 + phd + ment | kid5 + ment, data = bio,
                                 dist = "negbin", link = "cloglog", control = htight), "hurdle", bio)

f_book <- y ~ x + urban + offset(log(exposure)) | x
record("book_zip", zeroinfl(f_book, data = book, dist = "poisson", control = tight()), "zeroinfl", book)
record("book_zinb", zeroinfl(f_book, data = book, dist = "negbin", control = tight()), "zeroinfl", book)
record("book_zinb_weighted", zeroinfl(f_book, data = book, dist = "negbin", weights = weight,
                                      control = tight()), "zeroinfl", book, weighted = TRUE)
record("book_hnb", hurdle(f_book, data = book, dist = "negbin", control = htight), "hurdle", book)

# Vuong's test of the zero-inflated models (at their maximum) against the
# plain counts, as pscl::vuong computes it.
vu <- function(case, m1, m2, data) {
  y <- data$art
  p1 <- predict(m1, newdata = data, type = "prob", at = 0:max(y))
  p2 <- if (inherits(m2, "negbin")) dnbinom(y, size = m2$theta, mu = fitted(m2)) else dpois(y, fitted(m2))
  m <- log(p1[cbind(seq_along(y), y + 1)]) - log(p2)
  k1 <- length(coef(m1)); k2 <- length(coef(m2)); nn <- length(m)
  s <- sd(m)
  stats <- c(raw = sum(m) / (s * sqrt(nn)),
             aic = (sum(m) - (k1 - k2)) / (s * sqrt(nn)),
             bic = (sum(m) - (k1 - k2) * log(nn) / 2) / (s * sqrt(nn)))
  src <- paste0("pscl ", packageVersion("pscl"), " vuong's formula at the maximum")
  for (a in names(stats)) {
    add(case, "vuong", a, stats[[a]], 1e-6, src)
    pv <- if (stats[[a]] > 0) 1 - pnorm(stats[[a]]) else pnorm(stats[[a]])
    add(case, "vuong_p", a, pv, 1e-6, src)
  }
}
tight_glm <- glm.control(epsilon = 1e-14, maxit = 100)
vu("bio_zip", zip, glm(f_bio, data = bio, family = poisson, control = tight_glm), bio)
vu("bio_zinb", zinb, glm.nb(f_bio, data = bio, control = tight_glm), bio)

out <- do.call(rbind, rows)
con <- file("validation/reference/zero_counts_pscl.csv", "w")
writeLines(sprintf("# %s, pscl %s, MASS %s, numDeriv %s", R.version.string,
                   packageVersion("pscl"), packageVersion("MASS"), packageVersion("numDeriv")), con)
write.csv(out, con, row.names = FALSE, quote = FALSE)
close(con)
cat(nrow(out), "cases\n")
