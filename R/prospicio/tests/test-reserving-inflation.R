suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Indexation by a calendar index and re-inflation (#979,
# docs/design/reserving-inflation.md, PR 2): the R bindings of
# prospicio_reserving::inflation, with the cases of
# python/tests/test_reserving_inflation.py and
# validation/tests/reserving_inflation.rs.

long <- utils::read.csv(validation("data", "genins.csv"), comment.char = "#")
genins <- triangle(long, "origin", "development", "value")

# The index reads its periods and extends at the rate.
cpi <- calendar_index("2020", c(100, 104, 110), future_rate = 0.05)
near(calendar_index_value(cpi, c("2021-06", "2022-12")), c(104, 110))
near(calendar_index_value(cpi, "2024-01"), 110 * 1.05^2, 1e-14)
near(calendar_index_value(cpi, as.Date("2021-06-30")), 104)
stopifnot(identical(cpi@periods, c("2020", "2021", "2022")),
          identical(cpi@values, c(100, 104, 110)),
          cpi@future_rate == 0.05)
q <- calendar_index("2020Q2", c(100, 101, 103))
stopifnot(identical(q@periods, c("2020Q2", "2020Q3", "2020Q4")), is.null(q@future_rate))
near(calendar_index_value(q, "2020-06"), 100)
stopifnot(identical(calendar_index(2020, 1)@first, "2020"))

# A unit index restates to the same triangle and chain ladder.
unit <- indexation(calendar_index("2001", 1, future_rate = 0), "2010-12")
restated <- restate(unit, genins)
stopifnot(inherits(restated, "prospicio::triangle"))
near(restated@values[1, 1, , ], genins@values[1, 1, , ], 1e-14)
near(chain_ladder(restated)@reserve, chain_ladder(genins)@reserve, 1e-9)

# inflate undoes restate; the factor is the index ratio.
index <- calendar_index("2001", c(100, 103, 101.5, 108, 112, 111, 118, 125, 131, 140))
at <- indexation(index, "2006-07")
near(indexation_factor(at, "2008-03"), 118 / 111, 1e-14)
back <- inflate(at, restate(at, genins))
near(back@values[1, 1, , ], genins@values[1, 1, , ], 1e-12)
near(restate(at, genins)@values[1, 1, 1, 1], genins@values[1, 1, 1, 1] * 111 / 100, 1e-14)

# A chain ladder's future cells sum to its reserves; re-inflating at a
# constant rate grows each cell by its years past the base.
fit <- chain_ladder(genins)
cells <- future_cells(fit)
stopifnot(nrow(cells@cells) == 45)
near(unname(cells@reserves), unname(fit@reserve), 1e-9)
flat <- reinflate(unit, cells)
near(unname(flat@reserves), unname(fit@reserve), 1e-9)
near(flat@total_reserve, flat@total_base_reserve, 1e-12)
cpi4 <- indexation(calendar_index("2001", rep(1, 10), future_rate = 0.04), "2010-12")
grown <- reinflate(cpi4, cells)
stopifnot(grown@total_reserve > grown@total_base_reserve)
years <- (as.integer(substr(grown@cells@cells$valuation, 1, 4)) - 2010) +
  (as.integer(substr(grown@cells@cells$valuation, 6, 7)) - 12) / 12
near(grown@cells@cells$value, cells@cells$value * 1.04^years, 1e-12)

# An ODP GLM's cells.
glm <- odp_glm(genins)
glm_cells <- future_cells(glm)
near(unname(glm_cells@reserves), unname(glm@reserves), 1e-9)

# Several segments: choose one.
both <- rbind(transform(long, lob = "auto"), transform(long, value = long$value * 2, lob = "home"))
by_lob <- triangle(both, "origin", "development", "value", keys = "lob")
cl2 <- chain_ladder(by_lob)
expect_error_like(future_cells(cl2), "segment")
home <- future_cells(cl2, segment = c(lob = "home"))
near(sum(home@reserves), 2 * sum(future_cells(fit)@reserves), 1e-9)

# Taylor (1977), Example 1.
r <- c(0.5835, 0.2878, 0.0866, 0.0421)
lam <- c(86.4, 98.9, 102.0, 113.9)
rows <- do.call(rbind, lapply(1:4, function(i) {
  j <- if (i == 1) integer() else seq(6 - i, 4)
  data.frame(origin = as.character(2000 + i), age = 12 * j, valuation = sprintf("%d-12", 2000 + i + j - 1),
             value = r[j] * lam[4])
}))
taylor <- future_cells(rows, origins = as.character(2001:2004))
at_last <- indexation(calendar_index("2001", lam, future_rate = 0.10), "2004-01")
out <- reinflate(at_last, taylor)
printed <- c(5.3, 10.8, 5.8, 36.1, 11.9, 6.4)
stopifnot(length(out@cells@cells$value) == 6,
          all(abs(out@cells@cells$value - printed) <= 0.06),
          identical(out@cells@cells$origin, c("2002", "2003", "2003", "2004", "2004", "2004")))
near(out@total_reserve, sum(out@reserves), 1e-12)
near(out@total_base_reserve, sum(rows$value), 1e-12)

# Errors.
expect_error_like(calendar_index("2020", c(1, 0)), "positive")
expect_error_like(calendar_index("2020", numeric()), "at least one")
expect_error_like(calendar_index("twenty", 1), "period")
expect_error_like(calendar_index("2020", 1, future_rate = -1), "future_rate")
short <- calendar_index("2002", rep(1, 9))
expect_error_like(calendar_index_value(short, "2001-06"), "starts in 2002")
expect_error_like(calendar_index_value(short, "2020-01"), "no future rate")
expect_error_like(indexation(short, "2001-01"), "starts in 2002")
expect_error_like(restate(indexation(short, "2005-01"), genins), "starts in 2002")
expect_error_like(restate(indexation(calendar_index("2001", rep(1, 10)), "2005-01"), genins, "nope"),
                  "nope")
expect_error_like(reinflate(indexation(calendar_index("2001", rep(1, 10)), "2005-01"),
                            future_cells(data.frame(origin = "2009", age = 24,
                                                    valuation = "2011-12", value = 1))),
                  "no future rate")
expect_error_like(future_cells(data.frame(origin = "2010", age = 24, valuation = "2011-12", value = 1),
                               origins = "2009"), "not one of the origins")
expect_error_like(future_cells(1), "must be a chain_ladder_fit")
