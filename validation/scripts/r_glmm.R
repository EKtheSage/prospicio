# Generalized linear mixed models with random intercepts: glmmTMB's
# Laplace fits of the Poisson, gamma and Tweedie (power fixed at 1.5), the
# Poisson cross-checked against lme4's glmer, and glmer's adaptive
# Gauss-Hermite quadrature (nAGQ = 9) for a single factor.
#
# Writes validation/data/glmm_salamanders.csv (glmmTMB's Salamanders),
# glmm_grouseticks.csv (lme4's grouseticks), glmm_simulated.csv (a seeded
# book with two crossed factors) and validation/reference/glmm_r.csv, which
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

# glmmTMB: β, their SEs, each σ, log σ's SE, φ (the response's variance is
# φ μ^p), log φ's SE, the log-likelihood, and every level's mode and
# condVar.
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
  for (j in seq_along(vc)) {
    emit(dataset, model, "sd", names(vc)[j], attr(vc[[j]], "stddev"), 2e-5)
    emit(dataset, model, "se_log_sd", names(vc)[j], sqrt(vf[th[j], th[j]]), 2e-4)
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
    for (j in seq_len(nrow(re[[g]]))) {
      lab <- paste0(g, "|", rownames(re[[g]])[j])
      emit(dataset, model, "ranef", lab, re[[g]][j, 1], 1e-4)
      emit(dataset, model, "prediction_var", lab, v[1, 1, j], 1e-3)
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
  vc <- VarCorr(fit)
  for (g in names(vc)) emit(dataset, model, "sd", g, attr(vc[[g]], "stddev"), 2e-5)
  # With quadrature (nAGQ > 1) glmer's log-likelihood leaves out the
  # saturated model's, sum(dpois(y, y, log = TRUE)); it is added back.
  ll <- as.numeric(logLik(fit))
  if (grepl("agq", model)) ll <- ll + sum(dpois(fit@resp$y, fit@resp$y, log = TRUE))
  emit(dataset, model, "loglik", "", ll, 1e-8)
  re <- ranef(fit, condVar = TRUE)
  for (g in names(re)) {
    v <- attr(re[[g]], "postVar")
    for (j in seq_len(nrow(re[[g]]))) {
      lab <- paste0(g, "|", rownames(re[[g]])[j])
      emit(dataset, model, "ranef", lab, re[[g]][j, 1], 1e-4)
      emit(dataset, model, "conditional_var", lab, v[1, 1, j], 1e-4)
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

out <- do.call(rbind, rows)
con <- file("validation/reference/glmm_r.csv", "w")
writeLines(sprintf("# %s, glmmTMB %s, lme4 %s, TMB %s", R.version.string,
                   packageVersion("glmmTMB"), packageVersion("lme4"),
                   packageVersion("TMB")), con)
write.csv(out, con, row.names = FALSE, quote = FALSE)
close(con)
