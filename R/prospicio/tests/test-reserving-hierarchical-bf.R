suppressMessages(library(prospicio))

# bayes_hierarchical_bf(), the hierarchical Bayesian Bornhuetter-Ferguson over
# programs (docs/design/reserving-bayes.md, decision 30). The model's limits
# are checked in the Rust crate and in validation/tests; these tests check
# the binding on a small panel of three programs, one of them thin (two
# origins): the fit's fields, that the programs' reserves add up to the joint
# total, replay by seed, the pattern prior, the diagnostics warning and the
# errors.

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

# The warnings `expr` raises, and its value.
with_warnings <- function(expr) {
  found <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    found <<- c(found, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = found)
}

# Programs a (four origins), b (three) and c (two, a thin one), ages 12 to 48.
programs <- list(
  a = list(`2020` = list(c(60, 100, 125, 135), 200), `2021` = list(c(66, 108, 133), 210),
           `2022` = list(c(70, 115), 215), `2023` = list(72, 220)),
  b = list(`2021` = list(c(35, 60, 75), 120), `2022` = list(c(38, 64), 125),
           `2023` = list(40, 130)),
  c = list(`2022` = list(c(22, 38), 80), `2023` = list(25, 85))
)
panel <- function(programs) {
  rows <- do.call(rbind, lapply(names(programs), function(p) {
    do.call(rbind, lapply(names(programs[[p]]), function(y) {
      cells <- programs[[p]][[y]][[1]]
      data.frame(program = p, year = as.integer(y), age = 12 * seq_along(cells), paid = cells,
                 premium = programs[[p]][[y]][[2]])
    }))
  }))
  triangle(rows, "year", "age", c("paid", "premium"), keys = "program")
}
tri <- panel(programs)

quick <- list(chains = 2, tune = 300, draws = 300, n_sims = 600)
run <- with_warnings(do.call(bayes_hierarchical_bf,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 3), quick)))
fit <- run$value

stopifnot(
  S7::S7_inherits(fit, hierarchical_bf_fit),
  identical(fit@ages, c(12L, 24L, 36L, 48L)),
  fit@scale_per_premium > 0,
  identical(names(fit@programs), c("a", "b", "c")),
  all(vapply(fit@programs, function(p) S7::S7_inherits(p, hierarchical_bf_program), TRUE)),
  identical(vapply(fit@programs, function(p) length(p@origins), 0L, USE.NAMES = FALSE),
            c(4L, 3L, 2L))
)
for (p in fit@programs) {
  n <- length(p@origins)
  stopifnot(
    length(p@exposure) == n, length(p@latest) == n, length(p@prior_ultimate) == n,
    length(p@expected_ultimate) == n, length(p@pattern) == 4,
    all(p@pattern > 0), p@scale > 0, p@loss_ratio > 0, p@own_loss_ratio > 0,
    is.finite(p@shrinkage),
    S7::S7_inherits(p@reserves, predictive_distribution), p@reserves@n_sims == 600
  )
  near(sum(p@pattern), 1, 1e-9)
}
thin <- fit@programs$c
stopifnot(identical(thin@origins, c("2022", "2023")),
          identical(thin@exposure, c(80, 85)), identical(thin@latest, c(38, 25)),
          mean(thin@reserves) > 0)
near(thin@prior_ultimate, 0.7 * c(80, 85), 1e-12)
near(thin@scale, fit@scale_per_premium * (80 + 85) / 2, 1e-9)

# The programs' reserves add up to the joint total.
r <- fit@reserves
stopifnot(S7::S7_inherits(r, predictive_distribution), r@n_sims == 600,
          identical(r@dims, c("program", "origin")))
near(sum(vapply(fit@programs, function(p) mean(p@reserves), 0)), mean(r), 1e-9)
stopifnot(mean(r) > 0)

# The posterior and its frames.
nm <- fit@ptr$parameter_names()
stopifnot(identical(nm[1:2], c("log_level", "log_tau")), "log_omega" %in% nm,
          identical(dim(fit@draws), c(600L, length(nm))),
          identical(colnames(fit@draws), nm))
s <- fit@summary
stopifnot(identical(names(s), c("parameter", "mean", "sd", "q05", "q50", "q95", "rhat",
                                "ess_bulk", "ess_tail")),
          identical(s$parameter, nm))
h <- fit@hyperparameters
stopifnot(identical(h$parameter, c("level", "tau", "omega")),
          all(h$mean > 0), all(h$sd > 0), all(h$q05 > 0))
f <- as.data.frame(fit)
stopifnot(identical(f$program, c("a", "b", "c")), identical(f$origins, c(4L, 3L, 2L)),
          identical(names(f), c("program", "origins", "exposure", "latest", "prior_ultimate",
                                "expected_ultimate", "scale", "loss_ratio", "own_loss_ratio",
                                "shrinkage", "mean", "std_dev")),
          all(f$std_dev > 0))
near(f$exposure[3], 165, 1e-12)
near(sum(f$mean), mean(r), 1e-9)
stopifnot(identical(summary(fit), s), is.list(fit@diagnostics))
invisible(utils::capture.output(print(fit), print(fit@programs$a)))

# Replay by seed.
quiet <- function(...) with_warnings(do.call(bayes_hierarchical_bf, list(...)))$value
a <- quiet(tri, "paid", "premium", seed = 5, chains = 2, tune = 300, draws = 300, n_sims = 600)
b <- quiet(tri, "paid", "premium", seed = 5, chains = 2, tune = 300, draws = 300, n_sims = 600)
d <- quiet(tri, "paid", "premium", seed = 6, chains = 2, tune = 300, draws = 300, n_sims = 600)
stopifnot(identical(a@draws, b@draws), identical(mean(a@reserves), mean(b@reserves)),
          !identical(a@draws, d@draws))

# One segment is one program.
one <- quiet(panel(programs["a"]), "paid", "premium", seed = 1, chains = 2, tune = 300,
             draws = 300, n_sims = 600)
stopifnot(identical(names(one@programs), "a"), one@reserves@n_sims == 600)

# A pattern prior on the triangle's ages.
prior <- pattern_prior(c(12, 24, 36, 48), c(0.4, 0.6, 0.7), c(0.3, 0.3, 0.3))
withp <- quiet(tri, "paid", "premium", seed = 2, pattern_prior = prior, chains = 2, tune = 300,
               draws = 300, n_sims = 600)
stopifnot(length(withp@programs) == 3)
wrong <- pattern_prior(c(12, 24), 0.4, 0.3)
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", pattern_prior = wrong, chains = 2,
                                        tune = 100, draws = 100, n_sims = 100), "age")

# A run too short to converge: one warning, the fit still returned.
short <- with_warnings(bayes_hierarchical_bf(tri, "paid", "premium", chains = 2, tune = 20,
                                             draws = 20, n_sims = 100, seed = 1))
stopifnot(length(short$warnings) == 1,
          grepl("bayes_hierarchical_bf: the MCMC diagnostics failed", short$warnings),
          length(short$value@diagnostics$warnings) > 0)

# Errors.
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", n_sims = 0), "n_sims")
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", process = "lognormal"),
                  "gamma")
expect_error_like(bayes_hierarchical_bf(tri, "paid", "missing", chains = 2, tune = 100,
                                        draws = 100, n_sims = 100), "missing")
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", level_cv = "a"), "level_cv")
for (arg in c("apriori", "cv", "tau_scale", "omega_scale")) {
  args <- list(tri, "paid", "premium", chains = 2, tune = 100, draws = 100, n_sims = 100)
  args[[arg]] <- 0
  msg <- tryCatch({
    do.call(bayes_hierarchical_bf, args)
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop(sprintf("%s = 0 should fail", arg), call. = FALSE)
}
