# Regenerate validation/reference/elastic_net_relax_glmnet.csv:
#
#   Rscript validation/scripts/r_glmnet_relax.R
#
# glmnet's coefficient bounds (lower.limits, upper.limits) and relaxed
# fits (relax = TRUE: each lambda's active columns refitted at lambda = 0,
# with the same bounds, and the coefficients gamma * penalized +
# (1 - gamma) * refitted), on the data and columns of r_glmnet.R: age,
# age2 = age^2 / 100, region in treatment coding. Each case records
# lambda_max and, at four lambdas below it, every coefficient; the
# relaxed cases at two gammas. The cross-validation of relaxed fits
# (cv.glmnet(relax = TRUE) on fixed folds, row i in fold (i - 1) %% 5 + 1)
# records the mean deviance and its standard error at each gamma and
# lambda, and the chosen lambda and gamma. Convergence is tightened as in
# r_glmnet.R so the values are exact solutions.
#
# glmnet's blend.relaxed raises gamma to at least 1e-5, so its gamma = 0
# is gamma = 1e-5: the gamma column records that effective value, which
# prospicio is given (its own gamma = 0 is exactly the refit).
#
# glmnet is GPL-2: this script only produces reference values.

suppressPackageStartupMessages(library(glmnet))

glmnet.control(epsnr = 1e-16, mxitnr = 1000)
d <- read.csv("validation/data/glm_policies.csv")
x <- cbind(
  age = d$age,
  age2 = d$age^2 / 100,
  "region[B]" = as.numeric(d$region == "B"),
  "region[C]" = as.numeric(d$region == "C"),
  "region[D]" = as.numeric(d$region == "D")
)
n <- nrow(x)
ratios <- c(0.5, 0.1, 0.02, 0.004)
off <- log(d$exposure)

cases <- list(
  # Every coefficient non-negative: age and age2, negative unbounded, stay 0.
  poisson_nonneg = list(y = d$claims, family = "poisson", offset = off,
                        lower = rep(0, 5), upper = rep(Inf, 5)),
  # Bounds that bind at the smaller lambdas: age2 <= 0.1, region[C] >= -2,
  # region[D] <= 5.
  gaussian_bounded = list(y = d$gauss, family = "gaussian",
                          lower = c(-Inf, -Inf, -Inf, -2, -Inf),
                          upper = c(Inf, 0.1, Inf, Inf, 5)),
  poisson_relaxed = list(y = d$claims, family = "poisson", offset = off, relax = TRUE),
  gaussian_relaxed = list(y = d$gauss, family = "gaussian", relax = TRUE),
  # Relaxed under bounds: the refit keeps them.
  poisson_nonneg_relaxed = list(y = d$claims, family = "poisson", offset = off,
                                lower = rep(0, 5), upper = rep(Inf, 5), relax = TRUE)
)
gammas <- c(0, 0.5)

source <- paste("glmnet", packageVersion("glmnet"))
effective <- function(g) sprintf("%.17g", max(g, 1e-5))
rows <- list()
add <- function(case, quantity, term, arg, value, rel, gamma = "") {
  rows[[length(rows) + 1]] <<- data.frame(
    case = case, quantity = quantity, term = term,
    arg = if (is.numeric(arg)) sprintf("%.17g", arg) else arg,
    gamma = gamma, expected = sprintf("%.17g", value), abs_tol = 1e-9, rel_tol = rel,
    source = source
  )
}
args_of <- function(cs) {
  args <- list(x = x, y = cs$y, family = cs$family, alpha = 1, thresh = 1e-20, maxit = 1e9)
  if (!is.null(cs$offset)) args$offset <- cs$offset
  if (!is.null(cs$lower)) {
    args$lower.limits <- cs$lower
    args$upper.limits <- cs$upper
  }
  args
}
for (name in names(cases)) {
  cs <- cases[[name]]
  args <- args_of(cs)
  lmax <- do.call(glmnet, args)$lambda[1]
  add(name, "lambda_max", "", "", lmax, 1e-9)
  lambdas <- lmax * ratios
  args$lambda <- lambdas
  if (isTRUE(cs$relax)) args$relax <- TRUE
  fit <- do.call(glmnet, args)
  for (g in if (isTRUE(cs$relax)) gammas else 1) {
    beta <- as.matrix(if (isTRUE(cs$relax)) coef(fit, s = lambdas, gamma = g) else coef(fit))
    for (k in seq_along(lambdas)) {
      for (j in seq_len(nrow(beta))) {
        add(name, "coef", rownames(beta)[j], lambdas[k], beta[j, k], 1e-7,
            if (isTRUE(cs$relax)) effective(g) else "")
      }
    }
  }
}
write.csv(do.call(rbind, rows), "validation/reference/elastic_net_relax_glmnet.csv",
          row.names = FALSE, quote = FALSE)
cat("wrote validation/reference/elastic_net_relax_glmnet.csv\n")

# Cross-validated relaxed lasso.
foldid <- (seq_len(n) - 1) %% 5 + 1
cv_gammas <- c(0, 0.25, 0.5, 0.75, 1)
cv_rows <- list()
for (name in c("poisson_relaxed", "gaussian_relaxed")) {
  cs <- cases[[name]]
  args <- args_of(cs)
  lmax <- do.call(glmnet, args)$lambda[1]
  args$lambda <- lmax * 10^(-seq(0, 3, length.out = 20))
  args$foldid <- foldid
  args$type.measure <- "deviance"
  args$relax <- TRUE
  args$gamma <- cv_gammas
  cv <- do.call(cv.glmnet, args)
  for (gi in seq_along(cv_gammas)) {
    st <- cv$relaxed$statlist[[gi]]
    for (k in seq_along(st$lambda)) {
      for (q in c("mean", "se")) {
        cv_rows[[length(cv_rows) + 1]] <- data.frame(
          case = name, quantity = q, arg = sprintf("%.17g", st$lambda[k]),
          gamma = effective(cv_gammas[gi]),
          expected = sprintf("%.17g", if (q == "mean") st$cvm[k] else st$cvsd[k]),
          abs_tol = 1e-12, rel_tol = if (q == "mean") 1e-7 else 1e-6, source = source)
      }
    }
  }
  for (q in c("lambda.min", "gamma.min", "lambda.1se", "gamma.1se")) {
    cv_rows[[length(cv_rows) + 1]] <- data.frame(
      case = name, quantity = sub(".", "_", q, fixed = TRUE), arg = "", gamma = "",
      expected = if (startsWith(q, "gamma")) effective(cv$relaxed[[q]])
                 else sprintf("%.17g", cv$relaxed[[q]]),
      abs_tol = 1e-12, rel_tol = 1e-12,
      source = source)
  }
}
write.csv(do.call(rbind, cv_rows), "validation/reference/elastic_net_relax_cv_glmnet.csv",
          row.names = FALSE, quote = FALSE)
cat("wrote validation/reference/elastic_net_relax_cv_glmnet.csv\n")
