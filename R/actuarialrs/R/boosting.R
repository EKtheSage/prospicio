# Models lane: gradient boosting through the lightgbm or xgboost package,
# behind the same model protocol as glm_fit() (docs/design/models.md:
# "Gradient boosting stays an adapter"). The adapter turns a formula and data
# into the engine's inputs and wraps its output: means from predict(), joint
# draws from predict_distribution() through simulate_from_means(). It adds
# no evaluation logic; compare_models() and cross_validate() take it like any
# other fit.

#' @include models.R
NULL

# family -> lightgbm objective, xgboost objective, link
booster_objectives <- list(
  poisson = c("poisson", "count:poisson", "log"),
  gamma = c("gamma", "reg:gamma", "log"),
  tweedie = c("tweedie", "reg:tweedie", "log"),
  gaussian = c("regression", "reg:squarederror", "identity")
)

# Runs `code` with R's generator seeded, then puts the caller's generator
# state back.
with_seed <- function(seed, code) {
  had <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  old <- if (had) get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (had) assign(".Random.seed", old, envir = globalenv())
          else rm(".Random.seed", envir = globalenv()))
  set.seed(seed)
  code
}

booster_engine <- function(engine) {
  if (!requireNamespace(engine, quietly = TRUE)) {
    stop(sprintf("engine \"%s\" needs the %s package: install.packages(\"%s\")",
                 engine, engine, engine), call. = FALSE)
  }
}

booster_train <- function(spec, x, y, w, start, seed) {
  obj <- booster_objectives[[spec$family]]
  with_seed(seed, {
    if (spec$engine == "lightgbm") {
      params <- list(objective = obj[[1]], learning_rate = spec$learning_rate, seed = seed,
                     deterministic = TRUE, verbosity = -1)
      if (spec$family == "tweedie") params$tweedie_variance_power <- spec$power
      params[names(spec$params)] <- spec$params
      data <- lightgbm::lgb.Dataset(x, label = y, weight = w, init_score = start)
      lightgbm::lgb.train(params, data, nrounds = spec$n_rounds, verbose = -1)
    } else {
      params <- list(objective = obj[[2]], eta = spec$learning_rate, seed = seed)
      if (spec$family == "tweedie") params$tweedie_variance_power <- spec$power
      params[names(spec$params)] <- spec$params
      data <- xgboost::xgb.DMatrix(x, label = y, weight = w, base_margin = start)
      xgboost::xgb.train(params, data, nrounds = spec$n_rounds, verbose = 0)
    }
  })
}

# Means on the response scale; `start` is offset + base on the link scale.
booster_means <- function(spec, model, x, start) {
  eta <- if (spec$engine == "lightgbm") {
    predict(model, x, type = "raw") + start
  } else {
    predict(model, xgboost::xgb.DMatrix(x, base_margin = start), outputmargin = TRUE)
  }
  eta <- unname(as.double(eta))
  if (booster_objectives[[spec$family]][[3]] == "log") exp(eta) else eta
}

booster_variance <- function(family, mu, power) {
  switch(family, poisson = mu, gamma = mu^2, tweedie = mu^power, gaussian = rep(1, length(mu)))
}

# The design without its intercept column: trees need none.
booster_matrix <- function(x) {
  x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  if (!ncol(x)) stop("the formula needs at least one predictor")
  x
}

#' Fit gradient-boosted trees
#'
#' Gradient boosting through the lightgbm or xgboost package, behind the
#' same protocol as [glm_fit()]: [stats::predict()] gives means,
#' [predict_distribution()] joint draws, and [compare_models()] and
#' [cross_validate()] take it like any other fit.
#'
#' - The formula's `offset()` terms and `offset` are the engine's starting
#'   score (lightgbm `init_score`, xgboost `base_margin`) on the link scale:
#'   log exposure for a frequency. A constant `base` is added so the trees
#'   start from the weighted mean response, as the engines' own
#'   boost-from-average does without an offset: `log(sum(w y) / sum(w
#'   exp(offset)))` for the log link, `sum(w (y - offset)) / sum(w)` for the
#'   identity.
#' - The design is R's `model.matrix()` without the intercept, so factors
#'   enter as treatment-coded columns.
#' - xgboost computes margins in single precision, so its means agree with
#'   a double-precision calculation to about 1e-6 relative.
#' - `predict_distribution()` draws each row's response from the family
#'   around its mean (process noise); with `n_boot` bootstrap refits, each
#'   simulation also uses one refit's means (parameter uncertainty).
#'
#' The lightgbm and xgboost packages are optional: install the one you use.
#'
#' @inheritParams glm_fit
#' @param family `"poisson"`, `"gamma"`, `"tweedie"` (needs `power`) or
#'   `"gaussian"`; a log link for the first three.
#' @param engine `"lightgbm"` or `"xgboost"`.
#' @param power Tweedie variance power in `(1, 2)`.
#' @param n_rounds Boosting rounds.
#' @param learning_rate Shrinkage per round.
#' @param params A named list of further engine parameters (`num_leaves`,
#'   `max_depth`, `monotone_constraints`, ...), passed as they are.
#' @param n_boot Bootstrap refits for parameter uncertainty in
#'   [predict_distribution()]; 0 gives process noise only.
#' @param seed Seeds the engine and the bootstrap resamples; R's own
#'   generator state is left as it was.
#' @returns A `booster_model` with properties `model` (the engine's
#'   booster), `boots` (the refits), `base`, `dispersion` (1 for the
#'   Poisson, otherwise Pearson's estimate `sum(w (y - mu)^2 / V(mu)) / n`,
#'   with no degrees-of-freedom correction since trees have no fixed
#'   parameter count), `family`, `engine` and `power`.
#' @export
#' @examples
#' if (requireNamespace("lightgbm", quietly = TRUE)) {
#'   set.seed(1)
#'   d <- data.frame(x = rep(1:10 / 10, 20), exposure = rep(c(0.5, 1), 100))
#'   d$claims <- rpois(200, d$exposure * (0.1 + 0.3 * d$x))
#'   m <- booster_fit(claims ~ x + offset(log(exposure)), d, n_rounds = 20)
#'   head(predict(m, d))
#' }
booster_fit <- function(formula, data, family = "poisson", engine = c("lightgbm", "xgboost"),
                        power = NULL, n_rounds = 200, learning_rate = 0.05, params = list(),
                        n_boot = 0, seed = 0, offset = NULL, weights = NULL) {
  engine <- match.arg(engine)
  if (!family %in% names(booster_objectives)) {
    stop("family must be one of ", toString(names(booster_objectives)), ", got ", family)
  }
  if (family == "tweedie" && !(is.numeric(power) && power > 1 && power < 2)) {
    stop("tweedie needs power in (1, 2)")
  }
  if (n_rounds < 1 || n_boot < 0) stop("n_rounds must be positive and n_boot non-negative")
  if (length(params) && (is.null(names(params)) || any(names(params) == ""))) {
    stop("params must be a named list")
  }
  booster_engine(engine)
  des <- model_design(formula, data, offset, weights)
  x <- booster_matrix(des$x)
  y <- des$y
  n <- length(y)
  w <- if (length(des$weights)) des$weights else rep(1, n)
  off <- if (length(des$offset)) des$offset else rep(0, n)
  base <- if (booster_objectives[[family]][[3]] == "log") {
    if (sum(w * y) <= 0) stop("the weighted response total must be positive for a log link")
    log(sum(w * y) / sum(w * exp(off)))
  } else {
    sum(w * (y - off)) / sum(w)
  }
  spec <- list(family = family, engine = engine, power = power, n_rounds = n_rounds,
               learning_rate = learning_rate, params = params)
  model <- booster_train(spec, x, y, w, off + base, seed)
  rows <- with_seed(seed, lapply(seq_len(n_boot), function(b) sample.int(n, n, replace = TRUE)))
  boots <- lapply(seq_len(n_boot), function(b) {
    r <- rows[[b]]
    booster_train(spec, x[r, , drop = FALSE], y[r], w[r], off[r] + base, seed + b)
  })
  dispersion <- 1
  if (family != "poisson") {
    mu <- booster_means(spec, model, x, off + base)
    dispersion <- sum(w * (y - mu)^2 / booster_variance(family, mu, power)) / n
  }
  booster_model(spec = spec, model = model, boots = boots, base = base, dispersion = dispersion,
                columns = colnames(x), terms = des$terms, xlevels = des$xlevels)
}

#' Fitted gradient-boosted trees (class)
#'
#' Returned by [booster_fit()].
#'
#' @param spec,model,boots,base,dispersion,columns,terms,xlevels Internal.
#' @returns A `booster_model` object.
#' @export
booster_model <- S7::new_class(
  "booster_model",
  package = "actuarialrs",
  properties = list(
    spec = S7::class_list,
    model = S7::class_any,
    boots = S7::class_list,
    base = S7::class_double,
    dispersion = S7::class_double,
    columns = S7::class_character,
    terms = S7::class_any,
    xlevels = S7::class_any,
    family = S7::new_property(S7::class_character, getter = function(self) self@spec$family),
    engine = S7::new_property(S7::class_character, getter = function(self) self@spec$engine),
    power = S7::new_property(S7::class_any, getter = function(self) self@spec$power)
  )
)

booster_new_design <- function(object, newdata, offset) {
  nd <- new_design(object, newdata, offset)
  x <- nd$x[, object@columns, drop = FALSE]
  off <- if (length(nd$offset)) nd$offset else rep(0, nrow(x))
  list(x = x, start = off + object@base)
}

S7::method(predict, booster_model) <- function(object, newdata, offset = NULL, ...) {
  nd <- booster_new_design(object, newdata, offset)
  booster_means(object@spec, object@model, nd$x, nd$start)
}

S7::method(predict_distribution, booster_model) <- function(object, newdata, n_sims, seed,
                                                           offset = NULL, weights = NULL, ...) {
  nd <- booster_new_design(object, newdata, offset)
  models <- if (length(object@boots)) object@boots else list(object@model)
  means <- unlist(lapply(models, function(m) booster_means(object@spec, m, nd$x, nd$start)))
  simulate_from_means(object@spec$family, means, n_sims, seed, n_rows = nrow(nd$x),
                      dispersion = object@dispersion, weights = weights, power = object@spec$power)
}

S7::method(print, booster_model) <- function(x, ...) {
  cat(sprintf("<booster_model> %s %s, %d rounds, %d bootstrap refits\n", x@spec$engine,
              x@spec$family, as.integer(x@spec$n_rounds), length(x@boots)))
  invisible(x)
}

#' Predictive distribution from fitted means
#'
#' Joint draws of the responses of `n_rows` rows, for engines that give a
#' mean per row and nothing else (gradient boosting, an imported network):
#' the family adds the process noise, and several mean vectors (bootstrap
#' refits) add the parameter uncertainty. Simulation `i` picks one mean
#' vector uniformly, then draws each row's response from `family` with that
#' mean, `dispersion` and the row's weight. Components are keyed
#' `row = 0, 1, ...`, as for [glm_fit()]'s draws.
#'
#' @param family `"gaussian"`, `"poisson"`, `"gamma"`, `"inverse_gaussian"`,
#'   `"binomial"`, `"negative_binomial"` (needs `theta`) or `"tweedie"`
#'   (needs `power`).
#' @param means A matrix with one row per policy and one column per mean
#'   vector, or a vector of `n_rows * k` means stacked column by column.
#' @param n_sims Number of simulations.
#' @param seed Generator seed.
#' @param n_rows Rows per mean vector; defaults to `nrow(means)`, or the
#'   length of a vector.
#' @param dispersion The family's dispersion.
#' @param weights Optional prior weights, one per row.
#' @param theta Negative binomial `theta`.
#' @param power Tweedie power in `(1, 2)`.
#' @returns A [predictive_distribution].
#' @export
#' @examples
#' pd <- simulate_from_means("poisson", c(0.1, 0.4), n_sims = 20000, seed = 7)
#' mean(pd)
simulate_from_means <- function(family, means, n_sims, seed, n_rows = NULL, dispersion = 1,
                                weights = NULL, theta = NULL, power = NULL) {
  if (is.null(n_rows)) n_rows <- if (is.matrix(means)) nrow(means) else length(means)
  fa <- family_args(theta, power)
  ptr <- rust_result(simulate_from_means_rust(
    family, fa$theta, fa$power, as.double(means), as.double(n_rows), as.double(dispersion),
    if (is.null(weights)) double() else as.double(weights), as.double(n_sims), as.double(seed)
  ))
  predictive_distribution(ptr = ptr)
}
