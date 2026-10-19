# Targets for validation/tests/models.rs::glm_diagnostics_match_r: residuals,
# hat values, standardized residuals, Cook's distances and prediction
# standard errors of R's glm on the quine design (validation/data/quine.csv),
# Poisson with log link and Gamma with log link on Days + 1.
d <- read.csv("validation/data/quine.csv")
f <- Days ~ EthN + SexM + AgeF1 + AgeF2 + AgeF3 + LrnSL
show <- function(m, label) {
  rows <- c(1, 50, 146)
  cat(label, "\n")
  for (type in c("response", "pearson", "deviance", "working")) {
    cat(type, format(residuals(m, type = type)[rows], digits = 15), "\n")
  }
  cat("weights", format(weights(m, type = "working")[rows], digits = 15), "\n")
  cat("hat", format(hatvalues(m)[rows], digits = 15), "\n")
  cat("rstandard", format(rstandard(m)[rows], digits = 15), "\n")
  cat("rstandard_pearson", format(rstandard(m, type = "pearson")[rows], digits = 15), "\n")
  cat("cooks", format(cooks.distance(m)[rows], digits = 15), "\n")
  cat("se", format(predict(m, type = "response", se.fit = TRUE)$se.fit[rows], digits = 15), "\n")
}
# R keeps the working weights (and so the hat values) of its last IRLS
# iteration, so it is refitted to a tight tolerance for a converged match.
tight <- glm.control(epsilon = 1e-14, maxit = 100)
show(glm(f, family = poisson, data = d, control = tight), "poisson")
d$Days1 <- d$Days + 1
show(glm(update(f, Days1 ~ .), family = Gamma(link = "log"), data = d, control = tight), "gamma")
