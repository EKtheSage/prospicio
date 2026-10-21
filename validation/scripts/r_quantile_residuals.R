# Targets for validation/tests/models.rs::quantile_residuals_match_statmod:
# statmod::qresiduals on the quine design (validation/data/quine.csv) at
# rows 1, 50 and 146. The continuous families' are exact, at the fit's
# summary dispersion; for the counts it prints each row's bounds
# F(y - 1) and F(y), between which the residual is drawn.
library(statmod)
library(MASS)
d <- read.csv("validation/data/quine.csv")
f <- Days ~ EthN + SexM + AgeF1 + AgeF2 + AgeF3 + LrnSL
rows <- c(1, 50, 146)
tight <- glm.control(epsilon = 1e-14, maxit = 100)
d$Days1 <- d$Days + 1
show <- function(label, m) {
  r <- qresiduals(m, dispersion = summary(m)$dispersion)
  cat(label, format(r[rows], digits = 15), "\n")
}
show("gaussian", glm(f, family = gaussian, data = d, control = tight))
g <- glm(update(f, Days1 ~ .), family = Gamma(link = "log"), data = d, control = tight)
show("gamma", g)
# Started at the gamma fit, which the inverse Gaussian needs to converge.
show("inverse_gaussian", glm(update(f, Days1 ~ .), family = inverse.gaussian(link = "log"),
                             data = d, start = coef(g), control = tight))
bounds <- function(label, a, b) {
  cat(label, "lo", format(a[rows], digits = 15), "\n")
  cat(label, "hi", format(b[rows], digits = 15), "\n")
}
m <- glm(f, family = poisson, data = d, control = tight)
bounds("poisson", ppois(d$Days - 1, fitted(m)), ppois(d$Days, fitted(m)))
nb <- glm.nb(f, data = d, control = tight)
cat("theta", format(nb$theta, digits = 15), "\n")
bounds("negative_binomial", pnbinom(d$Days - 1, size = nb$theta, mu = fitted(nb)),
       pnbinom(d$Days, size = nb$theta, mu = fitted(nb)))
d$Long <- as.numeric(d$Days > 10)
b <- glm(update(f, Long ~ .), family = binomial, data = d, control = tight)
bounds("binomial", pbinom(d$Long - 1, 1, fitted(b)), pbinom(d$Long, 1, fitted(b)))
