suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Stochastic re-inflation of simulated future cells (#979,
# docs/design/reserving-inflation.md, PR 8): the R binding of
# Indexation::reinflate_draws, with the cases of
# python/tests/test_reserving_inflation_draws.py.

long <- utils::read.csv(validation("data", "genins.csv"), comment.char = "#")
genins <- triangle(long, "origin", "development", "value")
draws <- predict_distribution(odp_glm(genins), n_sims = 200, seed = 11)
history <- 1.04^(0:9)
at_2010 <- indexation(calendar_index("2001", history, future_rate = 0.04), "2010-01")

# Without paths every simulation takes the indexation's own index.
out <- reinflate_draws(at_2010, draws)
stopifnot(identical(out@dims, draws@dims), out@n_sims == draws@n_sims)
keys <- draws@keys
paid <- sprintf("%d-12", as.integer(keys[[1]]) + as.integer(keys[[2]]) %/% 12L - 1L)
factors <- indexation_factor(at_2010, paid)
near(draw_matrix(out), sweep(draw_matrix(draws), 2, factors, `*`), 1e-12)

# One path per simulation.
paths <- lapply(seq_len(draws@n_sims) - 1, function(i) {
  calendar_index("2001", history, future_rate = 0.1 * i / draws@n_sims)
})
own <- reinflate_draws(at_2010, draws, paths)
stopifnot(identical(draw_matrix(own)[1, ], draw_matrix(draws)[1, ]))

# The paths are one per simulation.
expect_error_like(reinflate_draws(at_2010, draws, paths[1:2]), "one per simulation")
