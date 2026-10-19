# Targets for validation/tests/models.rs::nested_glm_tests_match_r_anova:
# R's anova F and Chisq tests, BIC and variance inflation factors on the
# quine design of validation/data/quine.csv (written by r_glm_nb.R).
d <- read.csv("validation/data/quine.csv")
d$y <- log(d$Days + 1)
small <- glm(y ~ EthN + SexM, data = d)
big <- glm(y ~ EthN + SexM + AgeF1 + AgeF2 + AgeF3 + LrnSL, data = d)
print(anova(small, big, test = "F"), digits = 15)
print(anova(small, big, test = "Chisq"), digits = 15)
sp <- glm(Days ~ EthN + SexM, family = poisson, data = d)
bp <- glm(Days ~ EthN + SexM + AgeF1 + AgeF2 + AgeF3 + LrnSL, family = poisson, data = d)
print(anova(sp, bp, test = "Chisq"), digits = 15)
cat("poisson BIC", format(BIC(bp), digits = 15), "\n")
x <- d[, c("EthN", "SexM", "AgeF1", "AgeF2", "AgeF3", "LrnSL")]
cat("vif", format(sapply(names(x), function(v) 1 / (1 - summary(lm(x[[v]] ~ ., data = x[names(x) != v]))$r.squared)), digits = 15), "\n")
f <- anova(small, big, test = "F")
cat("F", format(f$F[2], digits = 15), "p", format(f$`Pr(>F)`[2], digits = 15), "\n")
cat("Chisq p", format(anova(small, big, test = "Chisq")$`Pr(>Chi)`[2], digits = 15), "\n")
cat("poisson Chisq p", format(anova(sp, bp, test = "Chisq")$`Pr(>Chi)`[2], digits = 15), "\n")
cat("poisson logLik", format(as.numeric(logLik(bp)), digits = 15), "\n")
