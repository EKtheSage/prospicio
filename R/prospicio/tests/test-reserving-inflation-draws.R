suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Stochastic re-inflation of simulated future cells (#979,
# docs/design/reserving-inflation.md, PR 8, decision 11): the R bindings of
# prospicio_reserving::inflation_draws, against
# validation/reference/reserving_reinflate_draws.csv (from
# validation/scripts/reserving_reinflate_draws.py), with the cases of
# validation/tests/reserving_reinflate_draws.rs.

ref <- utils::read.csv(validation("reference", "reserving_reinflate_draws.csv"), comment.char = "#",
                       stringsAsFactors = FALSE)

# "name=value;name=value" as a named character vector.
args <- function(arg) {
  pairs <- strsplit(strsplit(arg, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(vapply(pairs, `[`, "", 2), vapply(pairs, `[`, "", 1))
}

# Every method's objects built from its input rows, its results checked
# against its output rows to their tolerance.
for (method in unique(ref$method)) {
  rows <- ref[ref$method == method, ]
  of <- function(q) rows[rows$quantity == q, ]
  idx <- of("index")
  rate <- of("future_rate")
  index <- calendar_index(args(idx$arg[1])[["period"]], idx$expected,
                          future_rate = if (nrow(rate)) rate$expected else NULL)
  ix <- indexation(index, args(of("base")$arg)[["month"]])
  near(calendar_index_value(index, ix@base), of("base")$expected, 1e-15)

  d <- of("draw")
  a <- do.call(rbind, lapply(d$arg, args))
  sim <- as.integer(a[, "sim"])
  n_sims <- max(sim) + 1L
  cell <- a[sim == 0, , drop = FALSE]
  keys <- data.frame(origin = cell[, "origin"], development = as.numeric(cell[, "age"]),
                     stringsAsFactors = FALSE)
  draws <- predictive_distribution(matrix(d$expected, nrow = n_sims, byrow = TRUE), keys)

  p <- of("path")
  paths <- NULL
  if (nrow(p)) {
    pa <- do.call(rbind, lapply(p$arg, args))
    paths <- index_paths(pa[1, "period"], matrix(p$expected, nrow = n_sims, byrow = TRUE))
    stopifnot(paths@n_sims == n_sims,
              identical(paths@periods, unname(pa[pa[, "sim"] == "0", "period"])))
  }
  out <- reinflate_draws(ix, draws, paths)
  stopifnot(identical(out@dims, c("origin", "development")), out@n_sims == n_sims,
            identical(out@keys, draws@keys),
            !is.null(provenance(out)$parameters$reinflated))

  r <- of("reinflated")
  near(as.vector(t(draw_matrix(out))), r$expected, max(r$rel_tol))

  by_origin <- aggregate(out, "origin")
  res <- of("reserve")
  ra <- do.call(rbind, lapply(res$arg, args))
  m <- draw_matrix(by_origin)
  at <- cbind(as.integer(ra[, "sim"]) + 1L, match(ra[, "origin"], by_origin@keys$origin))
  near(m[at], res$expected, max(res$rel_tol))

  tot <- of("total")
  near(total(out)@draws, tot$expected, max(tot$rel_tol))
}

# With no paths, each simulation is reinflate() of its own cells: an ODP
# GLM's draws, the cells matched by origin and age.
long <- utils::read.csv(validation("data", "genins.csv"), comment.char = "#")
genins <- triangle(long, "origin", "development", "value")
glm <- odp_glm(genins)
pd <- predict_distribution(glm, n_sims = 5, seed = 1)
cpi <- indexation(calendar_index("2001", c(100, 103, 101.5, 108, 112, 111, 118, 125, 131, 140),
                                 future_rate = 0.04), "2006-07")
flat <- reinflate_draws(cpi, pd)
stopifnot(identical(flat@keys, pd@keys), flat@n_sims == 5)
cells <- future_cells(glm)@cells
k <- pd@keys
pos <- match(paste(k$origin, k$development), paste(cells$origin, cells$age))
stopifnot(!anyNA(pos))
for (i in 1:5) {
  one <- future_cells(transform(cells[pos, ], value = draw_matrix(pd)[i, ]))
  want <- reinflate(cpi, one)
  near(draw_matrix(flat)[i, ], want@cells@cells$value, 1e-12)
}
# Every payment is after the base, so no cell shrinks (a draw can be zero).
stopifnot(all(draw_matrix(flat) >= draw_matrix(pd)), sum(draw_matrix(flat)) > sum(draw_matrix(pd)))

# A whole year names an origin as its label does.
by_year <- predictive_distribution(matrix(c(100, 50), nrow = 1),
                                   data.frame(origin = 2022, development = c(24, 36)))
at_2022 <- indexation(calendar_index("2021", c(1, 1.1), future_rate = 0.1), "2022-01")
near(draw_matrix(reinflate_draws(at_2022, by_year))[1, ], c(100 * 1.1, 50 * 1.21), 1e-12)

# An age dimension, origin not first and a line of business passed through:
# a quarterly index given to 2022Q4 at level 1, 10% a year after it.
mixed_keys <- data.frame(lob = c("A", "A", "B"), origin = c("2022", "2022", "2022Q2"),
                         age = c(24, 36, 9), stringsAsFactors = FALSE)
mixed <- predictive_distribution(matrix(c(100, 50, 10), nrow = 1), mixed_keys)
quarterly <- indexation(calendar_index("2021Q1", rep(1, 8), future_rate = 0.1), "2022-01")
out_mixed <- reinflate_draws(quarterly, mixed)
stopifnot(identical(out_mixed@dims, c("lob", "origin", "age")),
          identical(out_mixed@keys, mixed@keys))
# Paid in 2023Q4, 2024Q4 and the given 2022Q4.
near(draw_matrix(out_mixed)[1, ], c(100 * 1.1, 50 * 1.21, 10), 1e-12)

# The paths read back.
two <- index_paths(2023, rbind(c(1.10, 1.20), c(1.05, 1.08)))
stopifnot(identical(two@first, "2023"), two@n_sims == 2L, two@n_periods == 2L,
          identical(two@periods, c("2023", "2024")),
          identical(two@values, rbind(c(1.10, 1.20), c(1.05, 1.08))))
stopifnot(identical(index_paths("2025Q1", matrix(1, 1, 3))@periods, c("2025Q1", "2025Q2", "2025Q3")))

# Errors.
cell2 <- predictive_distribution(rbind(c(100, 50), c(120, 40)),
                                 data.frame(origin = "2022", development = c(24, 36)))
given <- indexation(calendar_index("2021", c(1, 1.1)), "2022-01")
expect_error_like(index_paths("2023", rbind(c(1, 0))), "positive")
expect_error_like(index_paths("2023", matrix(numeric(), 0, 2)), "at least one simulation")
expect_error_like(index_paths("2023", c(1, 2)), "numeric matrix")
expect_error_like(index_paths("twenty", matrix(1)), "period")
expect_error_like(index_paths(), "needs first and values")
bad <- index_paths("2023", matrix(1, 1, 2))
expect_error_like(bad@values <- c(1, 2), "numeric matrix")
expect_error_like(reinflate_draws(given, aggregate(cell2, "origin")), "no development or age")
expect_error_like(reinflate_draws(given, predictive_distribution(matrix(1), data.frame(development = 24))),
                  "no origin")
expect_error_like(reinflate_draws(given, predictive_distribution(matrix(1), data.frame(origin = "A", development = 24))),
                  "not a period")
expect_error_like(reinflate_draws(given, predictive_distribution(matrix(1), data.frame(origin = "2022", development = "x"))),
                  "not an age in months")
expect_error_like(reinflate_draws(given, predictive_distribution(matrix(1),
                                                                 data.frame(origin = "2022", development = 24, age = 24))),
                  "both a development and an age")
expect_error_like(reinflate_draws(given, cell2), "no future rate")
expect_error_like(reinflate_draws(given, cell2, index_paths("2024", matrix(1, 2, 2))), "not in 2023")
expect_error_like(reinflate_draws(given, cell2, index_paths("2023Q1", matrix(1, 2, 8))), "another grain")
expect_error_like(reinflate_draws(given, cell2, index_paths("2023", matrix(1, 3, 2))),
                  "3 index paths for 2 simulations")
expect_error_like(reinflate_draws(given, cell2, index_paths("2023", matrix(1, 2, 1))), "the paths end in 2023")
expect_error_like(reinflate_draws(indexation(calendar_index("2021", c(1, 1.1), future_rate = 0.1), "2025-01"),
                                  cell2, index_paths("2023", matrix(1, 2, 2))),
                  "outside the given index")
expect_error_like(reinflate_draws(given, cell2, paths = 1), "paths must be NULL or index_paths")
expect_error_like(reinflate_draws(given, 1), "draws must be a predictive_distribution")
expect_error_like(reinflate_draws(1, cell2), "indexation")
