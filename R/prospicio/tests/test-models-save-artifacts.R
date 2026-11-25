library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# save_model() and load_model() for the GLMM, Bayesian, double GLM,
# zero-inflated, hurdle and expectile models: the estimates round-trip
# exactly and the loaded model predicts as the original (#978).
root <- if (dir.exists("validation")) "validation" else file.path("..", "..", "..", "validation")
sim <- utils::read.csv(file.path(root, "data", "glmm_simulated.csv"))
nd <- sim[1:5, ]
f <- tempfile(fileext = ".rds")
round_trip <- function(m) {
  save_model(m, f)
  load_model(f)
}

g <- glmm_fit(severity ~ x + urban + (1 | territory), sim, family = "gamma")
g2 <- round_trip(g)
stopifnot(identical(coef(g2), coef(g)), identical(g2@sd, g@sd), identical(g2@groups, g@groups))
stopifnot(identical(predict(g2, nd), predict(g, nd)))

dg <- dglm_fit(severity ~ x + urban, ~ urban, sim, family = "gamma", link = "log")
dg2 <- round_trip(dg)
stopifnot(identical(coef(dg2), coef(dg)))
stopifnot(identical(predict(dg2, nd), predict(dg, nd)),
          identical(predict(dg2, nd, type = "dispersion"), predict(dg, nd, type = "dispersion")))

counts <- data.frame(y = c(0, 0, 1, 0, 3, 0, 2, 0, 0, 5, 1, 0, 0, 4, 0, 2),
                     x = rep(c(-1, -0.3, 0.4, 1.2), 4))
zi <- zeroinfl_fit(y ~ x, counts)
zi2 <- round_trip(zi)
stopifnot(identical(zi2@count_coefficients, zi@count_coefficients),
          identical(zi2@zero_coefficients, zi@zero_coefficients), identical(zi2@n, zi@n))
stopifnot(identical(predict(zi2, counts[1:4, ]), predict(zi, counts[1:4, ])))
hu <- hurdle_fit(y ~ x, counts)
hu2 <- round_trip(hu)
stopifnot(identical(predict(hu2, counts[1:4, ], type = "zero"),
                    predict(hu, counts[1:4, ], type = "zero")))

e <- expectreg_fit(severity ~ x + urban, sim, tau = 0.8)
e2 <- round_trip(e)
stopifnot(identical(e2@tau, e@tau), identical(coef(e2), coef(e)),
          identical(predict(e2, nd), predict(e, nd)))

bd <- data.frame(claims = rep(c(1, 2, 3, 5), 10), x = rep(c(-1.5, -0.5, 0.5, 1.5), 10))
bm <- bayes_glm_fit(claims ~ x, bd, family = "poisson", chains = 2, tune = 200, draws = 200, seed = 4)
bm2 <- round_trip(bm)
stopifnot(identical(bm2@draws, bm@draws), identical(predict(bm2, bd[1:3, ]), predict(bm, bd[1:3, ])))
# A loaded Bayesian GLM has no training data for LOO.
stopifnot(inherits(try(bayes_loo(bm2), silent = TRUE), "try-error"))

sl <- utils::read.csv(file.path(root, "data", "glmm_slopes.csv"))
bp <- bayes_glmm_fit(count ~ x + (1 | territory) + offset(log(exposure)), sl,
                     chains = 2, tune = 200, draws = 200, seed = 7)
bp2 <- round_trip(bp)
stopifnot(identical(bp2@draws, bp@draws), identical(bp2@groups, bp@groups))
stopifnot(identical(predict(bp2, sl[1:3, ]), predict(bp, sl[1:3, ])))

# Files written before the other settings were saved still load.
m <- glm_fit(claims ~ x, bd, family = "poisson")
saveRDS(list(format = "prospicio.glm_model", artifact = m@ptr$to_json(),
             terms = m@terms, xlevels = m@xlevels), f)
stopifnot(identical(coef(load_model(f)), coef(m)))
saveRDS(list(format = "prospicio.nonsense", artifact = "{}"), f)
stopifnot(inherits(try(load_model(f), silent = TRUE), "try-error"))
