# Regenerate validation/reference/elastic_net_glmnet.csv and
# validation/reference/elastic_net_cv_glmnet.csv.
#
#   Rscript validation/scripts/r_glmnet.R
#
# Elastic-net GLMs on validation/data/glm_policies.csv with glmnet: columns
# age, age2 = age^2 / 100 (strongly correlated with age, so the lasso has
# to choose) and region in treatment coding, plus an unpenalized
# intercept. glmnet minimizes
#
#   sum(w * deviance) / (2 sum(w)) + lambda * sum((1 - alpha) / 2 b^2 + alpha |b|)
#
# on standardized columns (when standardize = TRUE) and reports
# coefficients on the original scale. Each case records lambda_max (the
# first lambda of glmnet's default path) and, at four lambdas below it,
# every coefficient and the deviance. Convergence is tightened (thresh =
# 1e-20, and epsnr = 1e-16 for the family-object path) so the values are
# exact solutions: with age and age2 this correlated, coordinate descent
# creeps, and at thresh = 1e-14 glmnet's coefficients are still 5e-6 off
# (their KKT conditions miss by 4e-6; at 1e-20 by 4e-9).
#
# glmnet is GPL-2: this script only produces reference values.

library(glmnet)
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

cases <- list(
  gaussian_lasso = list(y = d$gauss, family = "gaussian", alpha = 1, standardize = TRUE,
                        fam = gaussian()),
  gaussian_enet_raw = list(y = d$gauss, family = "gaussian", alpha = 0.5, standardize = FALSE,
                           fam = gaussian()),
  gaussian_weighted = list(y = d$gauss, family = "gaussian", alpha = 0.7, standardize = TRUE,
                           weights = d$exposure, fam = gaussian()),
  poisson_lasso = list(y = d$claims, family = "poisson", alpha = 1, standardize = TRUE,
                       offset = log(d$exposure), fam = poisson()),
  poisson_ridge = list(y = d$claims, family = "poisson", alpha = 0, standardize = TRUE,
                       offset = log(d$exposure), fam = poisson()),
  binomial_enet = list(y = cbind(d$trials - d$successes, d$successes), family = "binomial",
                       alpha = 0.5, standardize = TRUE, fam = binomial(),
                       prop = d$successes / d$trials, dev_weights = d$trials),
  gamma_log_enet = list(y = d$severity, family = Gamma(link = "log"), alpha = 0.5,
                        standardize = TRUE, fam = Gamma(link = "log"))
)

source <- paste("glmnet", packageVersion("glmnet"))
rows <- list()
add <- function(case, quantity, term, arg, value, rel) {
  rows[[length(rows) + 1]] <<- data.frame(
    case = case, quantity = quantity, term = term,
    arg = if (is.numeric(arg)) sprintf("%.17g", arg) else arg,
    expected = sprintf("%.17g", value), abs_tol = 1e-9, rel_tol = rel, source = source
  )
}

for (name in names(cases)) {
  cs <- cases[[name]]
  w <- if (is.null(cs$weights)) rep(1, n) else cs$weights
  off <- if (is.null(cs$offset)) rep(0, n) else cs$offset
  args <- list(x = x, y = cs$y, family = cs$family, alpha = cs$alpha,
               standardize = cs$standardize, thresh = 1e-20, maxit = 1e9)
  if (!is.null(cs$weights)) args$weights <- w
  if (!is.null(cs$offset)) args$offset <- off
  path <- do.call(glmnet, args)
  lmax <- path$lambda[1]
  add(name, "lambda_max", "", "", lmax, 1e-9)
  lambdas <- lmax * ratios
  args$lambda <- lambdas
  fit <- do.call(glmnet, args)
  beta <- as.matrix(coef(fit))
  # Deviance on the original scale: sum of weighted unit deviances.
  y_dev <- if (is.null(cs$prop)) cs$y else cs$prop
  w_dev <- if (is.null(cs$dev_weights)) w else cs$dev_weights
  for (k in seq_along(lambdas)) {
    eta <- drop(cbind(1, x) %*% beta[, k]) + off
    mu <- cs$fam$linkinv(eta)
    add(name, "deviance", "", lambdas[k], sum(cs$fam$dev.resids(y_dev, mu, w_dev)), 1e-8)
    for (j in seq_len(nrow(beta))) {
      term <- rownames(beta)[j]
      add(name, "coef", term, lambdas[k], beta[j, k], 1e-7)
    }
  }
}

out <- do.call(rbind, rows)
write.csv(out, "validation/reference/elastic_net_glmnet.csv", row.names = FALSE, quote = FALSE)
cat("wrote validation/reference/elastic_net_glmnet.csv\n")

# Cross-validation: cv.glmnet on fixed folds (row i in fold (i - 1) %% 5 + 1)
# over 20 lambdas, type.measure = "deviance". Records the cross-validated
# mean and standard error per lambda, and lambda.min and lambda.1se, in
# validation/reference/elastic_net_cv_glmnet.csv. Lasso only, so glmnet's
# Gaussian rescaling of the ridge part does not enter.
foldid <- (seq_len(n) - 1) %% 5 + 1
cv_rows <- list()
cv_add <- function(case, quantity, arg, value, rel) {
  cv_rows[[length(cv_rows) + 1]] <<- data.frame(
    case = case, quantity = quantity,
    arg = if (is.numeric(arg)) sprintf("%.17g", arg) else arg,
    expected = sprintf("%.17g", value), abs_tol = 1e-12, rel_tol = rel, source = source
  )
}
for (name in c("gaussian_lasso", "poisson_lasso")) {
  cs <- cases[[name]]
  args <- list(x = x, y = cs$y, family = cs$family, alpha = cs$alpha,
               standardize = cs$standardize, thresh = 1e-20, maxit = 1e9)
  if (!is.null(cs$offset)) args$offset <- cs$offset
  lmax <- do.call(glmnet, args)$lambda[1]
  args$lambda <- lmax * 10^(-seq(0, 3, length.out = 20))
  args$foldid <- foldid
  args$type.measure <- "deviance"
  cv <- do.call(cv.glmnet, args)
  for (k in seq_along(cv$lambda)) {
    cv_add(name, "mean", cv$lambda[k], cv$cvm[k], 1e-7)
    cv_add(name, "se", cv$lambda[k], cv$cvsd[k], 1e-6)
  }
  cv_add(name, "lambda_min", "", cv$lambda.min, 1e-12)
  cv_add(name, "lambda_1se", "", cv$lambda.1se, 1e-12)
}
write.csv(do.call(rbind, cv_rows), "validation/reference/elastic_net_cv_glmnet.csv",
          row.names = FALSE, quote = FALSE)
cat("wrote validation/reference/elastic_net_cv_glmnet.csv\n")
