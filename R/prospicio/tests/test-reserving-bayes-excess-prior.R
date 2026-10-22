suppressMessages(library(prospicio))

# layer_share_prior(), bayes_bf_layer() and bayes_bf_layer_one_year(): the
# Bayesian BF prior of an excess layer from the layer share and its
# uncertainty (docs/design/reserving-excess.md, PR 4). The prior's moments,
# the tight-prior limit and the widening of the interval are checked in
# validation/tests/reserving_bayes_excess_prior.rs; these tests check the
# binding and that R gives the numbers Python's LayerSharePrior gives
# (python/tests/test_reserving_bayes_excess_prior.py): the same cv, the same
# prior ultimates, the same layer exposure, and the same prior moments of the
# total with a share error common to all origins (decision 25 of
# docs/design/reserving-bayes.md, #518).

near <- function(a, b, rel) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop("expected an error", call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

share <- 0.25

# The cv of a product of independents.
p <- layer_share_prior(0.65, share, apriori_cv = 0.1, share_cv = 0.3)
stopifnot(
  S7::S7_inherits(p, layer_share_prior),
  p@apriori == 0.65, p@apriori_cv == 0.1, p@share_cv == 0.3, identical(p@shares, share)
)
near(p@cv^2, 1.01 * 1.09 - 1, 1e-12)
near(p@apriori * p@shares, 0.65 * share, 1e-15)
near(layer_share_prior(0.5, 0.2, apriori_cv = 0, share_cv = 0.3, share_dependence = "per_origin")@cv,
     0.3, 1e-12)
stopifnot(is.infinite(layer_share_prior(0.5, 0.2, share_cv = Inf)@cv))
stopifnot(p@share_dependence == "common", p@has_common_factor, p@common_cv == 0.3,
          p@model_cv == 0.1)

# The prior moments of the total, from first principles; the same numbers as
# Python's test_the_prior_moments_are_exact.
exposure <- c(100, 150, 220, 90)
shares4 <- c(0.2, 0.25, 0.3, 0.2)
ce <- 0.25
cs <- 0.7
m <- 0.6 * exposure * shares4
common <- layer_share_prior(0.6, shares4, apriori_cv = ce, share_cv = cs)
tot <- layer_prior_total(common, exposure)
near(tot[["mean"]], sum(m), 1e-12)
near(tot[["sd"]], sqrt(ce^2 * (1 + cs^2) * sum(m^2) + cs^2 * sum(m)^2), 1e-12)
near(c(tot[["mean"]], tot[["sd"]]), c(84.9, 61.2), 1e-3)
per <- layer_share_prior(0.6, shares4, apriori_cv = ce, share_cv = cs, share_dependence = "per_origin")
stopifnot(per@share_dependence == "per_origin", !per@has_common_factor, per@common_cv == 0)
near(layer_prior_total(per, exposure)[["sd"]], sqrt(((1 + ce^2) * (1 + cs^2) - 1) * sum(m^2)), 1e-12)
stopifnot(tot[["sd"]] > layer_prior_total(per, exposure)[["sd"]])
expect_error_like(layer_prior_total(common, c(1, 2)), "exposure")

# The standard error route, and a severity's share.
a <- layer_share_prior(0.6, 0.25, share_se = 0.05)
near(a@share_cv, 0.2, 1e-15)
near(layer_share(pareto(1, 2), excess_layer(4)), 0.125, 1e-12)
invisible(capture.output(print(p)))

expect_error_like(layer_share_prior(0, 0.2), "apriori")
expect_error_like(layer_share_prior(0.5, 1.5), "share")
expect_error_like(layer_share_prior(0.5, 0.2, share_cv = -1), "share_cv")
expect_error_like(layer_share_prior(0.5, "x"), "shares")
expect_error_like(bayes_bf_layer(NULL, "layer", "prior", 1), "prior")

# EVW's triangle: the layer is a quarter of the paid values.
evw <- utils::read.csv(file.path("validation", "data", "evw2012.csv"), comment.char = "#")
evw$layer <- share * evw$paid
tri <- triangle(evw, "origin", "development", c("layer", "prior"))

prior <- layer_share_prior(0.5, share, apriori_cv = 0.1, share_cv = 0.3)
fit <- suppressWarnings(bayes_bf_layer(tri, "layer", "prior", prior, n_sims = 500, seed = 3))
ground_up <- suppressWarnings(bayes_bf_normalised(tri, "layer", "prior", n_sims = 100))
stopifnot(S7::S7_inherits(fit, bayes_bf_normalised_fit))
near(fit@exposure, share * ground_up@exposure, 1e-12)
near(fit@prior_ultimate, 0.5 * share * ground_up@exposure, 1e-12)

# The same as fitting the model by hand on the layer exposure.
layered <- layer_exposure(tri, "prior", share, "layer_exposure")
by_hand <- suppressWarnings(bayes_bf_normalised(layered, "layer", "layer_exposure", apriori = 0.5,
                                                cv = prior@model_cv, common_cv = prior@common_cv,
                                                n_sims = 500, seed = 3))
near(mean(fit@reserves), mean(by_hand@reserves), 1e-12)
# A known loss ratio cannot be shared: an error when the model is fitted, as
# in Python.
expect_error_like(
  bayes_bf_layer(tri, "layer", "prior",
                 layer_share_prior(0.5, share, apriori_cv = 0, share_cv = 0.3), n_sims = 100),
  "cv")
# The multiplier is the last parameter of the fit.
stopifnot(tail(colnames(fit@draws), 1) == "log_share")

# The share's uncertainty widens the predictive interval over a known share;
# independent priors keep widening it, a common share is learned from all
# the origins.
for (dependence in c("common", "per_origin")) {
  sds <- vapply(c(1e-3, 0.5, 1.2), function(s) {
    pr <- layer_share_prior(0.35, share, apriori_cv = 0.1, share_cv = s,
                            share_dependence = dependence)
    f <- suppressWarnings(bayes_bf_layer(tri, "layer", "prior", pr, n_sims = 2000, seed = 3))
    sqrt(variance(f@reserves))
  }, numeric(1))
  stopifnot(sds[1] < sds[2])
  if (dependence == "per_origin") stopifnot(sds[2] < sds[3])
}

# The one-year view.
view <- suppressWarnings(bayes_bf_layer_one_year(tri, "layer", "prior", prior, n_sims = 300,
                                                 seed = 1, refit = "reweight_only"))
stopifnot(
  S7::S7_inherits(view, bayes_one_year_fit),
  variance(view@cdr) < variance(view@fit@reserves)
)
cat("ok\n")
