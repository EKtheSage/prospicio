suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The bootstraps' future cells (#979, docs/design/reserving-inflation.md,
# "The bootstraps' cells", decisions A to I): `cells = TRUE` of
# odp_bootstrap() and mack_bootstrap() and the fits' `cells`, with the
# cases of validation/tests/reserving_bootstrap_cells.rs.

raa_long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
genins_long <- utils::read.csv(validation("data", "genins.csv"), comment.char = "#")
raa <- triangle(raa_long, "origin", "development", "value")
genins <- triangle(genins_long, "origin", "development", "value")
boots <- list(odp = odp_bootstrap, mack = mack_bootstrap)

# `long` under lob A, and doubled under B.
twins <- function(long) {
  two <- long[rep(seq_len(nrow(long)), 2), ]
  two$lob <- rep(c("A", "B"), each = nrow(long))
  two$value <- two$value * rep(c(1, 2), each = nrow(long))
  triangle(two, "origin", "development", "value", keys = "lob")
}

# One string per component: its keys but the last `drop` pasted together.
key_of <- function(keys, drop = 0) {
  do.call(paste, c(unname(as.list(keys[seq_len(ncol(keys) - drop)])), sep = "|"))
}

# Each origin's cells sum to its reserve in every simulation, by hand and
# through aggregate(); an origin without a cell has a reserve of 0.
check_sums <- function(reserves, cells, rel = 1e-9) {
  stopifnot(identical(cells@dims, c(reserves@dims, "development")))
  owner <- key_of(cells@keys, 1)
  rkeys <- key_of(reserves@keys)
  stopifnot(all(owner %in% rkeys))
  cm <- draw_matrix(cells)
  rm <- draw_matrix(reserves)
  for (o in seq_along(rkeys)) {
    j <- which(owner == rkeys[o])
    if (length(j)) {
      near_unit(rowSums(cm[, j, drop = FALSE]), rm[, o], rel)
    } else {
      stopifnot(all(rm[, o] == 0))
    }
  }
  summed <- aggregate(cells, reserves@dims)
  sm <- draw_matrix(summed)
  skeys <- key_of(summed@keys)
  stopifnot(setequal(skeys, unique(owner)))
  for (k in seq_along(skeys)) near_unit(sm[, k], rm[, match(skeys[k], rkeys)], rel)
}

for (name in names(boots)) {
  boot <- boots[[name]]

  # Cells leave the reserves unchanged and sum to them.
  without <- boot(raa, n_sims = 200, seed = 979)
  fit <- boot(raa, n_sims = 200, seed = 979, cells = TRUE)
  stopifnot(is.null(without@cells), identical(draw_matrix(fit@reserves), draw_matrix(without@reserves)),
            identical(fit@reserves@keys, without@reserves@keys))
  cells <- fit@cells
  stopifnot(S7::S7_inherits(cells, predictive_distribution),
            identical(cells@dims, c("origin", "development")), cells@n_sims == 200,
            nrow(cells@keys) == 45,
            identical(as.character(cells@keys$origin[1]), "1982"),
            cells@keys$development[1] == 120,
            all(cells@keys$development[cells@keys$origin == "1990"] == seq(24, 120, 12)))
  check_sums(fit@reserves, cells)

  # A tail adds one cell per origin, at the oldest age plus one period.
  fit <- boot(raa, n_sims = 200, seed = 979, tail = 1.05, cells = TRUE)
  keys <- fit@cells@keys
  stopifnot(nrow(keys) == 55,
            identical(as.character(keys$origin[keys$development == 132]), as.character(1981:1990)),
            as.character(keys$origin[1]) == "1981", keys$development[1] == 132)
  near_unit(draw_matrix(fit@cells)[, 1], draw_matrix(fit@reserves)[, 1], 1e-12)
  check_sums(fit@reserves, fit@cells)

  # A fully developed triangle has no cells without a tail.
  sq_long <- data.frame(origin = rep(2020:2022, each = 3), development = rep(c(12, 24, 36), 3),
                        value = c(100, 150, 165, 110, 170, 182, 120, 175, 196))
  sq <- triangle(sq_long, "origin", "development", "value")
  stopifnot(is.null(boot(sq, n_sims = 50, seed = 979)@cells))
  expect_error_like(boot(sq, n_sims = 50, seed = 979, cells = TRUE), "no future cells")
  sq_tail <- boot(sq, n_sims = 50, seed = 979, tail = 1.05, cells = TRUE)
  stopifnot(all(sq_tail@cells@keys$development == 48), nrow(sq_tail@cells@keys) == 3)

  # Segments keep their own cells, under each dependence; segment() keeps
  # its part.
  for (dependence in c("independent", "synchronized", "rank_correlation")) {
    tri <- twins(raa_long)
    spearman <- if (dependence == "rank_correlation") matrix(c(1, 0.6, 0.6, 1), 2) else NULL
    without <- boot(tri, n_sims = 200, seed = 979, dependence = dependence, spearman = spearman)
    fit <- boot(tri, n_sims = 200, seed = 979, dependence = dependence, spearman = spearman,
                cells = TRUE)
    stopifnot(identical(draw_matrix(fit@reserves), draw_matrix(without@reserves)),
              identical(fit@cells@dims, c("lob", "origin", "development")),
              nrow(fit@cells@keys) == 90)
    check_sums(fit@reserves, fit@cells)
    a <- segment(fit, lob = "A")
    stopifnot(identical(a@cells@dims, c("lob", "origin", "development")),
              nrow(a@cells@keys) == 45, all(a@cells@keys$lob == "A"),
              is.null(segment(without, lob = "A")@cells))
    check_sums(a@reserves, a@cells)
  }

  # A segment with no future cell has NULL cells.
  sq3 <- data.frame(origin = rep(2018:2020, each = 3), development = rep(c(12, 24, 36), 3),
                    value = c(100, 150, 165, 110, 170, 182, 120, 175, 196))
  mixed <- rbind(cbind(raa_long, lob = "raa"), cbind(sq3, lob = "square"))
  fit <- boot(triangle(mixed, "origin", "development", "value", keys = "lob"),
              n_sims = 50, seed = 979, cells = TRUE)
  stopifnot(all(fit@cells@keys$lob == "raa"), is.null(segment(fit, lob = "square")@cells),
            !is.null(segment(fit, lob = "raa")@cells))
  check_sums(fit@reserves, fit@cells)

  # reinflate_draws() takes the cells: paths at the index's own future rate
  # give the deterministic re-inflation, and a unit index the reserves.
  given <- 100 * 1.04^(0:9)
  rate <- 0.05
  at <- indexation(calendar_index("2001", given, future_rate = rate), "2006-03")
  unit <- indexation(calendar_index("2001", rep(1, 10), future_rate = 0), "2006-03")
  for (tail in c(1, 1.03)) {
    fit <- boot(genins, n_sims = 300, seed = 979, tail = tail, cells = TRUE)
    cells <- fit@cells
    path <- given[10] * (1 + rate)^(1:11)
    paths <- index_paths("2011", matrix(path, nrow = 300, ncol = 11, byrow = TRUE))
    with_paths <- reinflate_draws(at, cells, paths)
    plain <- reinflate_draws(at, cells)
    stopifnot(identical(with_paths@keys, cells@keys))
    near_unit(as.vector(draw_matrix(with_paths)), as.vector(draw_matrix(plain)), 1e-12)
    check_sums(fit@reserves, reinflate_draws(unit, cells))
    # Each cell grows by its own payment year: a flat index to 2010 then 4%
    # a year, based at 2010-12, pays origin o at age d months in year
    # o + d / 12 - 1, the tail cell one year after the oldest age's.
    flat4 <- indexation(calendar_index("2001", rep(1, 10), future_rate = 0.04), "2010-12")
    years <- as.integer(as.character(cells@keys$origin)) + cells@keys$development %/% 12 - 1 - 2010
    stopifnot(min(years) == 1, max(years) == if (tail > 1) 10 else 9)
    want <- sweep(draw_matrix(cells), 2, 1.04^years, `*`)
    near_unit(as.vector(draw_matrix(reinflate_draws(flat4, cells))), as.vector(want), 1e-12)
  }
  fit <- boot(twins(genins_long), n_sims = 200, seed = 979, cells = TRUE)
  out <- reinflate_draws(unit, fit@cells)
  stopifnot(identical(out@dims, c("lob", "origin", "development")))
  check_sums(fit@reserves, out)
}
