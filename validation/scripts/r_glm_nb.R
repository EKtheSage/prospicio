# Negative binomial GLM with theta estimated, R MASS::glm.nb on MASS's quine
# data (school absence days; Venables and Ripley, MASS 4th ed., 7.4).
# Writes validation/data/quine.csv (the response and the main-effects
# design) and prints the reference values that validation/tests/models.rs
# checks. Run from the repository root: Rscript validation/scripts/r_glm_nb.R
library(MASS)
m <- glm.nb(Days ~ Eth + Sex + Age + Lrn, data = quine)
x <- model.matrix(m)
out <- data.frame(Days = quine$Days, x[, -1], check.names = FALSE)
write.csv(out, "validation/data/quine.csv", row.names = FALSE)
cat(sprintf("MASS %s\n", packageVersion("MASS")))
cat(sprintf("theta %.15g\nse_theta %.15g\nloglik %.15g\n", m$theta, m$SE.theta,
            as.numeric(logLik(m))))
cat(sprintf("%s %.15g\n", names(coef(m)), coef(m)), sep = "")
