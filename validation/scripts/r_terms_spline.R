# Targets for validation/tests/models.rs::orthogonal_polys_and_natural_splines_match_r:
# R's poly(x, 3) columns, and the fitted values and predictions of
# lm(y ~ ns(x, knots = c(10, 25))), whose span a natural spline basis must
# reproduce.
x <- c(1, 2, 4, 7, 11, 16, 22, 29, 37, 46)
y <- sin(x / 7) * 10 + x / 3
p <- poly(x, 3)
for (j in 1:3) cat("poly", j, format(p[, j], digits = 17), "\n")
newx <- c(0, 20, 50)
pn <- predict(p, newx)
for (j in 1:3) cat("poly_new", j, format(pn[, j], digits = 17), "\n")
m <- lm(y ~ splines::ns(x, knots = c(10, 25)))
cat("ns_fitted", format(fitted(m), digits = 17), "\n")
cat("ns_new", format(predict(m, data.frame(x = newx)), digits = 17), "\n")
