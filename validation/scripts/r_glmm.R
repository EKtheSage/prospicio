# Generalized linear mixed models with random intercepts and slopes: glmmTMB's
# Laplace fits of the Poisson, gamma and Tweedie (power fixed at 1.5), the
# Poisson cross-checked against lme4's glmer, and glmer's adaptive
# Gauss-Hermite quadrature (nAGQ = 9) for a single factor.
#
# Writes validation/data/glmm_salamanders.csv (glmmTMB's Salamanders),
# glmm_grouseticks.csv (lme4's grouseticks), glmm_simulated.csv (a seeded
# book with two crossed factors), glmm_slopes.csv (a seeded book with
# random slopes) and validation/reference/glmm_r.csv, which
# validation/tests/models.rs checks. Run from the repository root:
#   Rscript validation/scripts/r_glmm.R
suppressPackageStartupMessages({
  library(glmmTMB)
  library(lme4)
})
data(Salamanders, package = "glmmTMB")
data(grouseticks, package = "lme4")

sal <- data.frame(count = Salamanders$count, spp = as.character(Salamanders$spp),
                  mined = as.character(Salamanders$mined),
                  site = as.character(Salamanders$site))
write.csv(sal, "validation/data/glmm_salamanders.csv", row.names = FALSE)

gt <- data.frame(ticks = grouseticks$TICKS, year = as.character(grouseticks$YEAR),
                 height = (grouseticks$HEIGHT - mean(grouseticks$HEIGHT)) / 100,
                 location = as.character(grouseticks$LOCATION),
                 brood = as.character(grouseticks$BROOD))
write.csv(gt, "validation/data/glmm_grouseticks.csv", row.names = FALSE)

# A book of 600 policies in 25 territories crossed with 8 classes: claim
# counts on exposure, positive severities, and a pure premium with zeros.
set.seed(20261020)
n <- 600
territory <- sprintf("t%02d", sample(25, n, replace = TRUE))
class <- sprintf("c%d", sample(8, n, replace = TRUE))
x <- round(runif(n, -1, 1), 4)
urban <- sample(c("no", "yes"), n, replace = TRUE)
exposure <- round(runif(n, 0.2, 1), 3)
u_t <- rnorm(25, 0, 0.4)
u_c <- rnorm(8, 0, 0.25)
eta <- 0.3 + 0.5 * x + 0.4 * (urban == "yes") +
  u_t[as.integer(substr(territory, 2, 3))] + u_c[as.integer(substr(class, 2, 2))]
count <- rpois(n, exposure * exp(eta - 1))
severity <- round(rgamma(n, shape = 2, scale = exp(eta + 6) / 2), 2)
claims <- rpois(n, exposure * exp(eta - 1.5) * 2)
premium <- sapply(claims, function(k) if (k == 0) 0 else sum(rgamma(k, shape = 1.5, scale = 400)))
pure <- round(premium / exposure, 2)
sim <- data.frame(count, severity, pure, exposure, x, urban, territory, class)
write.csv(sim, "validation/data/glmm_simulated.csv", row.names = FALSE)

rows <- list()
# Each case passes within its relative tolerance or, for values near 0
# (random effects, small coefficients), an absolute one a tenth as large.
source_of <- function(model) {
  if (grepl("glmer|agq", model)) sprintf("lme4 %s", packageVersion("lme4"))
  else sprintf("glmmTMB %s", packageVersion("glmmTMB"))
}
emit <- function(dataset, model, quantity, arg, value, tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    dataset = dataset, model = model, quantity = quantity, arg = arg,
    expected = sprintf("%.15g", value), abs_tol = tol / 10, rel_tol = tol,
    source = source_of(model))
}

# A random term's argument: the factor for its intercept, `g|x` for a
# slope; a level's, `g|level` and `g|level|x`.
term_arg <- function(g, term) if (term == "(Intercept)") g else paste0(g, "|", term)
level_arg <- function(g, level, term) {
  if (term == "(Intercept)") paste0(g, "|", level) else paste(g, level, term, sep = "|")
}

# Term c's conditional variance at level j: lme4 gives the independent
# terms of `(1 + x || g)` one array each.
cond_var <- function(v, c, j) if (is.list(v)) v[[c]][1, 1, j] else v[c, c, j]

# glmmTMB: β, their SEs, each σ, log σ's SE, φ (the response's variance is
# φ μ^p), log φ's SE, each correlation and its parameter's SE (glmmTMB's
# theta: two terms have ρ = θ / √(1 + θ²)), the log-likelihood, and every
# level's modes and condVar.
emit_tmb <- function(dataset, model, fit, power) {
  b <- fixef(fit)$cond
  se <- sqrt(diag(vcov(fit)$cond))
  for (j in seq_along(b)) {
    emit(dataset, model, "coef", names(b)[j], b[j], 2e-5)
    emit(dataset, model, "se", names(b)[j], se[j], 2e-4)
  }
  vf <- vcov(fit, full = TRUE)
  vc <- VarCorr(fit)$cond
  th <- grep("^theta", colnames(vf))
  # Each factor's theta: its log standard deviations, then (an
  # unstructured covariance) its correlation parameters.
  at <- 0
  for (j in seq_along(vc)) {
    g <- names(vc)[j]
    s <- attr(vc[[j]], "stddev")
    terms <- names(s)
    for (c in seq_along(s)) {
      emit(dataset, model, "sd", term_arg(g, terms[c]), s[c], 2e-5)
      emit(dataset, model, "se_log_sd", term_arg(g, terms[c]),
           sqrt(vf[th[at + c], th[at + c]]), 2e-4)
    }
    n_theta <- length(s)
    cor <- attr(vc[[j]], "correlation")
    if (length(s) > 1 && !grepl("diag", model)) {
      for (r in 2:length(s)) for (c in 1:(r - 1)) {
        emit(dataset, model, "cor", paste(g, terms[r], terms[c], sep = "|"), cor[r, c], 2e-4)
      }
      n_theta <- length(s) * (length(s) + 1) / 2
      for (k in (length(s) + 1):n_theta) {
        emit(dataset, model, "se_theta", paste0(g, "|", k), sqrt(vf[th[at + k], th[at + k]]),
             2e-4)
      }
    }
    at <- at + n_theta
  }
  if (power > 1) {
    s <- sigma(fit)
    d <- grep("^d~|disp", colnames(vf))
    # sigma() is the gamma's coefficient of variation (φ = σ²) and the
    # Tweedie's φ. glmmTMB's dispersion parameter is the gamma's log
    # shape, -log φ, and the Tweedie's log φ: either way its standard error
    # is log φ's.
    emit(dataset, model, "dispersion", "", if (power == 2) s^2 else s, 2e-5)
    emit(dataset, model, "se_log_dispersion", "", sqrt(vf[d, d]), 2e-4)
  }
  emit(dataset, model, "loglik", "", as.numeric(logLik(fit)), 1e-8)
  re <- ranef(fit, condVar = TRUE)$cond
  for (g in names(re)) {
    v <- attr(re[[g]], "condVar")
    for (c in seq_len(ncol(re[[g]]))) {
      for (j in seq_len(nrow(re[[g]]))) {
        lab <- level_arg(g, rownames(re[[g]])[j], colnames(re[[g]])[c])
        emit(dataset, model, "ranef", lab, re[[g]][j, c], 1e-4)
        emit(dataset, model, "prediction_var", lab, cond_var(v, c, j), 1e-3)
      }
    }
  }
}

emit_lme4 <- function(dataset, model, fit) {
  b <- fixef(fit)
  se <- sqrt(diag(as.matrix(vcov(fit))))
  for (j in seq_along(b)) {
    emit(dataset, model, "coef", names(b)[j], b[j], 2e-5)
    emit(dataset, model, "se", names(b)[j], se[j], 2e-3)
  }
  # lme4 names the factors of `(1 + x || g)` g and g.1.
  factor_of <- function(g) sub("\\.[0-9]+$", "", g)
  vc <- VarCorr(fit)
  for (g in names(vc)) {
    s <- attr(vc[[g]], "stddev")
    for (c in seq_along(s)) {
      emit(dataset, model, "sd", term_arg(factor_of(g), names(s)[c]), s[c], 2e-5)
    }
    if (length(s) > 1) {
      cor <- attr(vc[[g]], "correlation")
      for (r in 2:length(s)) for (c in 1:(r - 1)) {
        emit(dataset, model, "cor", paste(factor_of(g), names(s)[r], names(s)[c], sep = "|"),
             cor[r, c], 2e-4)
      }
    }
  }
  # With quadrature (nAGQ > 1) glmer's log-likelihood leaves out the
  # saturated model's, sum(dpois(y, y, log = TRUE)); it is added back.
  ll <- as.numeric(logLik(fit))
  if (grepl("agq", model)) ll <- ll + sum(dpois(fit@resp$y, fit@resp$y, log = TRUE))
  emit(dataset, model, "loglik", "", ll, 1e-8)
  re <- ranef(fit, condVar = TRUE)
  for (g in names(re)) {
    v <- attr(re[[g]], "postVar")
    for (c in seq_len(ncol(re[[g]]))) {
      for (j in seq_len(nrow(re[[g]]))) {
        lab <- level_arg(factor_of(g), rownames(re[[g]])[j], colnames(re[[g]])[c])
        emit(dataset, model, "ranef", lab, re[[g]][j, c], 1e-4)
        emit(dataset, model, "conditional_var", lab, cond_var(v, c, j), 1e-4)
      }
    }
  }
}

# nlminb stops glmmTMB with gradients near 1e-3 (Salamanders), leaving
# the estimates about 3e-5 short. Each fit is polished by Newton steps on
# TMB's objective and gradient, then refitted from the polished values
# with an optimizer that returns its start, so that every reported
# quantity (vcov, sdreport, ranef) is at the polished maximum.
keep <- function(start, objective, gradient, control)
  list(par = start, objective = objective(start), convergence = 0, message = "polished")
tmb <- function(...) {
  fit <- glmmTMB(...)
  par <- fit$fit$par
  for (k in 1:8) {
    g <- fit$obj$gr(par)
    if (max(abs(g)) < 1e-9) break
    par <- par - solve(optimHess(par, fit$obj$fn, fit$obj$gr), as.numeric(g))
  }
  stopifnot(max(abs(fit$obj$gr(par))) < 1e-7)
  start <- lapply(split(par, factor(names(par), unique(names(par)))), unname)
  args <- list(...)
  args$start <- c(start, args$start[setdiff(names(args$start), names(start))])
  args$control <- glmmTMBControl(optimizer = keep)
  do.call(glmmTMB, args)
}
gctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5, rhoend = 1e-12),
                      tolPwrss = 1e-12)

f <- tmb(count ~ spp + mined + (1 | site), Salamanders, family = poisson)
emit_tmb("salamanders", "poisson", f, 1)
g <- glmer(count ~ spp + mined + (1 | site), Salamanders, family = poisson, control = gctrl)
emit_lme4("salamanders", "poisson_glmer", g)
g9 <- glmer(count ~ spp + mined + (1 | site), Salamanders, family = poisson, nAGQ = 9,
            control = gctrl)
emit_lme4("salamanders", "poisson_agq9", g9)

f <- tmb(ticks ~ year + height + (1 | location) + (1 | location:brood), gt,
         family = poisson)
emit_tmb("grouseticks", "poisson", f, 1)
g <- glmer(ticks ~ year + height + (1 | location) + (1 | location:brood), gt,
           family = poisson, control = gctrl)
emit_lme4("grouseticks", "poisson_glmer", g)

f <- tmb(count ~ x + urban + (1 | territory) + (1 | class) + offset(log(exposure)), sim,
         family = poisson)
emit_tmb("simulated", "poisson", f, 1)
f <- tmb(severity ~ x + urban + (1 | territory) + (1 | class), sim,
         family = Gamma(link = "log"))
emit_tmb("simulated", "gamma", f, 2)
# The Tweedie power fixed at 1.5: glmmTMB's psi is logit(p - 1). No prior
# weights: glmmTMB multiplies the log-likelihood by them rather than
# dividing the dispersion.
tw <- function(form) tmb(form, sim, family = tweedie(link = "log"),
                         start = list(psi = qlogis(0.5)), map = list(psi = factor(NA)))
emit_tmb("simulated", "tweedie", tw(pure ~ x + urban + (1 | territory) + (1 | class)), 1.5)
emit_tmb("simulated", "tweedie_one", tw(pure ~ x + urban + (1 | territory)), 1.5)

# Random slopes: a book of 800 policies in 30 territories, x and w varying
# within each, the territories' intercepts and slopes on x correlated.
set.seed(20261021)
n <- 800
territory <- sprintf("t%02d", sample(30, n, replace = TRUE))
x <- round(runif(n, -1, 1), 4)
w <- round(runif(n, -1, 1), 4)
exposure <- round(runif(n, 0.2, 1), 3)
z <- matrix(rnorm(90), 30)
b0 <- 0.4 * z[, 1]
b1 <- 0.3 * (0.5 * z[, 1] + sqrt(0.75) * z[, 2])
b2 <- 0.2 * z[, 3]
t <- as.integer(substr(territory, 2, 3))
eta <- 0.3 + 0.5 * x - 0.2 * w + b0[t] + b1[t] * x + b2[t] * w
count <- rpois(n, exposure * exp(eta + 0.5))
severity <- round(rgamma(n, shape = 2, scale = exp(eta + 6) / 2), 2)
claims <- rpois(n, exposure * exp(eta + 0.7) * 2)
premium <- sapply(claims, function(k) if (k == 0) 0 else sum(rgamma(k, shape = 1.5, scale = 400)))
pure <- round(premium / exposure, 2)
sl <- data.frame(count, severity, pure, exposure, x, w, territory)
write.csv(sl, "validation/data/glmm_slopes.csv", row.names = FALSE)

f <- tmb(count ~ x + w + (1 + x | territory) + offset(log(exposure)), sl, family = poisson)
emit_tmb("slopes", "poisson", f, 1)
g <- glmer(count ~ x + w + (1 + x | territory) + offset(log(exposure)), sl, family = poisson,
           control = gctrl)
emit_lme4("slopes", "poisson_glmer", g)
f <- tmb(count ~ x + w + diag(1 + x | territory) + offset(log(exposure)), sl, family = poisson)
emit_tmb("slopes", "poisson_diag", f, 1)
g <- glmer(count ~ x + w + (1 + x || territory) + offset(log(exposure)), sl, family = poisson,
           control = gctrl)
emit_lme4("slopes", "poisson_diag_glmer", g)
f <- tmb(count ~ x + w + (1 + x + w | territory) + offset(log(exposure)), sl, family = poisson)
emit_tmb("slopes", "poisson_three", f, 1)
f <- tmb(severity ~ x + w + (1 + x | territory), sl, family = Gamma(link = "log"))
emit_tmb("slopes", "gamma", f, 2)
f <- tmb(pure ~ x + w + (1 + x | territory), sl, family = tweedie(link = "log"),
         start = list(psi = qlogis(0.5)), map = list(psi = factor(NA)))
emit_tmb("slopes", "tweedie", f, 1.5)

out <- do.call(rbind, rows)
con <- file("validation/reference/glmm_r.csv", "w")
writeLines(sprintf("# %s, glmmTMB %s, lme4 %s, TMB %s", R.version.string,
                   packageVersion("glmmTMB"), packageVersion("lme4"),
                   packageVersion("TMB")), con)
write.csv(out, con, row.names = FALSE, quote = FALSE)
close(con)
