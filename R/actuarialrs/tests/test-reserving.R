suppressMessages(library(actuarialrs))

# Reports the values on failure, so a CI log shows what diverged.
near <- function(a, b, rel = 1e-12) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

# The error message of `expr`, which must fail.
error_of <- function(expr) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop("expected an error", call. = FALSE)
  msg
}

expect_error_like <- function(expr, pattern) {
  msg <- error_of(expr)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

# Tests run from the repository root (CI and `cargo xtask r`).
validation <- function(...) file.path("validation", ...)
read_long <- function(name) utils::read.csv(validation("data", paste0(name, ".csv")), comment.char = "#")
load_triangle <- function(name) triangle(read_long(name), "origin", "development", "value")

tris <- list(raa = load_triangle("raa"), genins = load_triangle("genins"), abc = load_triangle("abc"))

# Shapes and accessors.
raa <- tris$raa
stopifnot(
  identical(unname(raa@shape), c(1L, 1L, 10L, 10L)),
  identical(raa@origins, as.character(1981:1990)),
  identical(raa@development, seq(12L, 120L, by = 12L)),
  identical(raa@columns, "value"),
  identical(raa@keys, character()),
  identical(dim(raa@index), c(1L, 0L)),
  identical(dimnames(raa@values)$index, "Total"),
  identical(raa@valuation, as.Date("1990-12-31")),
  raa@is_cumulative,
  raa@origin_grain == "Y", raa@development_grain == "Y",
  identical(dim(raa@values), c(1L, 1L, 10L, 10L)),
  is.na(raa@values[1, 1, "1990", "24"])
)
near(raa@values[1, "value", "1981", "12"], 5012)
near(sum(latest_diagonal(raa)), 160987)
near(sum(latest_diagonal(tris$genins)), 34358090)
near(sum(latest_diagonal(tris$abc)), 10221194)
near(link_ratios(raa)@values[1, 1, "1981", "12"], 8269 / 5012)
stopifnot(!link_ratios(raa)@is_cumulative, link_ratios(raa)@shape[["development"]] == 9L)

# Chain ladder and Mack against R ChainLadder: every reference row, with
# its own tolerance (a case passes if it meets either).
# The free-text `source` column holds unquoted commas, so keep the first
# seven fields of each line.
lines <- grep("^#", readLines(validation("reference", "reserving_chainladder_r.csv")),
              value = TRUE, invert = TRUE)
fields <- lapply(strsplit(lines, ",", fixed = TRUE), `[`, 1:7)
ref <- utils::read.csv(text = vapply(fields, paste, "", collapse = ","))
settings <- list(
  chain_ladder = c("volume", "log-linear"), mack = c("volume", "log-linear"),
  chain_ladder_simple = c("simple", "log-linear"), mack_alpha0 = c("simple", "log-linear"),
  mack_alpha2 = c("regression", "log-linear"), mack_sigma_mack = c("volume", "mack")
)
fits <- list()
fit_for <- function(dataset, method) {
  key <- paste(dataset, method)
  if (is.null(fits[[key]])) {
    s <- settings[[method]]
    if (is.null(s)) stop("no settings for reference method ", method, call. = FALSE)
    # Chain-ladder rows go through chain_ladder(), the rest through mack().
    fit <- if (startsWith(method, "chain_ladder")) chain_ladder else mack
    fits[[key]] <<- fit(tris[[dataset]], average = s[1], sigma_interpolation = s[2])
  }
  fits[[key]]
}
value_of <- function(fit, quantity, arg) {
  k <- as.integer(arg) + 1L
  o <- as.character(arg)
  switch(quantity,
    ata_factor = fit@ldf[[k]], cdf = fit@cdf[[k]], sigma = fit@sigma[[k]], f_se = fit@std_err[[k]],
    ultimate = fit@ultimate[[o]], reserve = fit@reserve[[o]], se = fit@standard_error[[o]],
    process_risk = fit@process_risk[[o]], parameter_risk = fit@parameter_risk[[o]],
    total_ultimate = fit@total_ultimate, total_reserve = fit@total_reserve,
    total_standard_error = fit@total_standard_error, total_process_risk = fit@total_process_risk,
    total_parameter_risk = fit@total_parameter_risk,
    stop("unknown quantity ", quantity)
  )
}
for (r in seq_len(nrow(ref))) {
  row <- ref[r, ]
  got <- value_of(fit_for(row$dataset, row$method), row$quantity, row$arg)
  err <- abs(got - row$expected)
  if (!(got == row$expected || err <= row$abs_tol || err <= row$rel_tol * abs(row$expected))) {
    stop(sprintf("%s %s %s[%s]: got %.17g, want %.17g", row$dataset, row$method, row$quantity,
                 row$arg, got, row$expected), call. = FALSE)
  }
}
stopifnot(nrow(ref) > 700)

# The deterministic chain ladder gives the projection Mack builds on.
for (name in names(tris)) {
  cl <- chain_ladder(tris[[name]])
  m <- mack(tris[[name]])
  stopifnot(identical(cl@ultimate, m@ultimate), identical(cl@ldf, m@chain_ladder@ldf))
  near(m@total_cv, m@total_standard_error / m@total_reserve)
  near(sum(cl@reserve), cl@total_reserve, 1e-12)
}
cl <- chain_ladder(raa, "value", tail = 1.05)
near(cl@cdf[["120-Ult"]], 1.05)
near(cl@ultimate[["1981"]], 18834 * 1.05)
stopifnot(identical(names(cl@ldf)[1], "12-24"), cl@alpha == 1, cl@tail == 1.05)
df <- as.data.frame(mack(raa))
stopifnot(identical(names(df), c("origin", "latest", "ultimate", "reserve", "process_risk",
                                 "parameter_risk", "standard_error")), nrow(df) == 10)
invisible(utils::capture.output(print(raa), print(cl), print(mack(raa))))

# Long-table round trip.
long <- as.data.frame(raa)
stopifnot(identical(names(long), c("origin", "development", "value")),
          inherits(long$origin, "Date"), nrow(long) == 55)
back <- triangle(long, "origin", "development", "value")
stopifnot(identical(back@values, raa@values), identical(back@origins, raa@origins))
stopifnot(identical(subset(raa)@values, raa@values), identical(aggregate(raa)@values, raa@values))

# Incremental and cumulative.
inc <- to_incremental(raa)
stopifnot(!inc@is_cumulative)
near(inc@values[1, 1, "1981", "24"], 8269 - 5012)
stopifnot(identical(to_cumulative(inc)@values, raa@values))
inc_long <- as.data.frame(inc)
from_inc <- triangle(inc_long, "origin", "development", "value", cumulative = FALSE)
stopifnot(identical(to_cumulative(from_inc)@values, raa@values))

# Development as valuation dates (and as valuation years).
raw <- read_long("raa")
by_val <- raw
by_val$development <- as.Date(sprintf("%d-12-31", raw$origin + raw$development / 12 - 1))
stopifnot(identical(triangle(by_val, "origin", "development", "value",
                             development_is_valuation = TRUE)@values,
                    raa@values))
by_year <- transform(raw, development = origin + development / 12 - 1)
stopifnot(identical(triangle(by_year, "origin", "development", "value",
                             development_is_valuation = TRUE)@values,
                    raa@values))

# Several key columns and value columns; subset and aggregate by name.
two <- rbind(transform(raw, lob = "auto", state = "CA", paid = value, incurred = 1.5 * value),
             transform(raw, lob = "home", state = "NY", paid = 2 * value, incurred = 3 * value))
multi <- triangle(two, "origin", "development", c("paid", "incurred"), keys = c("lob", "state"))
stopifnot(
  identical(multi@keys, c("lob", "state")),
  identical(unname(multi@shape), c(2L, 2L, 10L, 10L)),
  identical(multi@index, data.frame(lob = c("auto", "home"), state = c("CA", "NY"))),
  identical(dimnames(multi@values)$index, c("auto / CA", "home / NY"))
)
home <- subset(multi, lob = "home", state = "NY", columns = "incurred")
near(home@values[!is.na(home@values)], 3 * raa@values[!is.na(raa@values)])
both <- subset(multi, state = c("NY", "CA"))
stopifnot(identical(both@index$lob, c("auto", "home")), identical(both@keys, c("lob", "state")))
multi_long <- as.data.frame(multi)
stopifnot(identical(names(multi_long), c("lob", "state", "origin", "development", "paid", "incurred")))
again <- triangle(multi_long, "origin", "development", c("paid", "incurred"), keys = multi@keys)
stopifnot(identical(again@values, multi@values), identical(as.data.frame(again), multi_long))
# Key order follows the argument; factor and numeric keys become character.
flipped <- triangle(multi_long, "origin", "development", "paid", keys = c("state", "lob"))
stopifnot(identical(flipped@index, data.frame(state = c("CA", "NY"), lob = c("auto", "home"))),
          identical(names(as.data.frame(flipped))[1:2], c("state", "lob")))
coded <- transform(raw, company = factor(ifelse(origin < 1985, 7, 8)))
by_company <- triangle(coded, "origin", "development", "value", keys = "company")
stopifnot(identical(by_company@index$company, c("7", "8")),
          identical(as.data.frame(by_company)$company[1], "7"))
one <- triangle(two, "origin", "development", "paid", keys = "lob")
near(chain_ladder(subset(one, lob = "home"))@total_reserve, 2 * chain_ladder(raa)@total_reserve,
     1e-12)

# Two keys (lob x coverage) with paid and incurred, as in the Python and
# Rust tests: Auto from RAA and Home from GenIns (scaled), origins re-based
# to 2011, each coverage a share of the line with its own tilt by origin.
lob_coverage <- do.call(rbind, lapply(list(list("Auto", "raa", 1), list("Home", "genins", 1e-3)), function(l) {
  rows <- read_long(l[[2]])
  offset <- rows$origin - min(rows$origin)
  do.call(rbind, lapply(list(list("BI", 0.7), list("PD", 0.3)), function(cv) {
    paid <- rows$value * l[[3]] * cv[[2]] * (1 + cv[[2]] * offset / 20)
    data.frame(lob = l[[1]], coverage = cv[[1]], year = 2011 + offset, age = rows$development,
               paid = paid, incurred = paid * 1.2 + 10 * ((seq_along(paid) - 1) %% 3))
  }))
}))
measures <- c("paid", "incurred")
lc <- triangle(lob_coverage, "year", "age", measures, keys = c("lob", "coverage"))
stopifnot(identical(unname(lc@shape), c(4L, 2L, 10L, 10L)))
# Selection by name: one value or several per key, ANDed across keys.
stopifnot(
  identical(subset(lc, lob = "Auto", coverage = "BI")@index, data.frame(lob = "Auto", coverage = "BI")),
  identical(subset(lc, lob = "Auto", coverage = c("PD", "BI"))@index$coverage, c("BI", "PD")),
  identical(subset(lc, coverage = "BI", columns = c("incurred", "paid"))@columns, c("incurred", "paid")),
  identical(subset(lc, coverage = "BI")@index$lob, c("Auto", "Home")),
  identical(unname(subset(lc, columns = "paid")@shape), c(4L, 1L, 10L, 10L))
)
# Grouping equals building with fewer keys; totals are sums of segments.
by_lob <- aggregate(lc, keep = "lob")
stopifnot(identical(by_lob@keys, "lob"), identical(by_lob@index, data.frame(lob = c("Auto", "Home"))))
stopifnot(identical(by_lob@values, triangle(lob_coverage, "year", "age", measures, keys = "lob")@values))
total <- aggregate(lc)
stopifnot(identical(total@keys, character()),
          identical(total@values, triangle(lob_coverage, "year", "age", measures)@values))
for (m in measures) {
  segments <- sum(latest_diagonal(lc)[, m, ])
  near(sum(latest_diagonal(by_lob)[, m, ]), segments, 1e-12)
  near(sum(latest_diagonal(total)[, m, ]), segments, 1e-12)
}
stopifnot(identical(aggregate(lc, keep = c("coverage", "lob"))@index[1, ],
                    data.frame(coverage = "BI", lob = "Auto")))
# Chain ladder on a group equals chain ladder on the summed triangle.
summed <- triangle(lob_coverage[lob_coverage$lob == "Auto", ], "year", "age", measures)
for (m in measures) {
  grouped <- chain_ladder(subset(by_lob, lob = "Auto"), m)
  direct <- chain_ladder(summed, m)
  near(grouped@ultimate, direct@ultimate)
  near(grouped@ldf, direct@ldf)
}
# Incremental triangles are summed as cumulative values.
stopifnot(identical(aggregate(to_incremental(lc), keep = "lob")@values, to_incremental(by_lob)@values))
expect_error_like(subset(lc, line = "Auto"), "no key named line")
expect_error_like(subset(lc, coverage = "GL"), "no segment has coverage = \"GL\"")
expect_error_like(subset(lc, lob = c("Auto", "Auto")), "supplied twice")
expect_error_like(subset(lc, lob = character()), "at least one value")
expect_error_like(subset(lc, lob = "Home", columns = "reported"), "no column named reported")
expect_error_like(subset(lc, "Auto"), "must be named by key")
expect_error_like(subset(lc, lob = "Auto", lob = "Home"), "key lob is supplied twice")
expect_error_like(aggregate(lc, keep = "line"), "no key named line")
expect_error_like(aggregate(lc, keep = c("lob", "lob")), "key lob is supplied twice")
expect_error_like(chain_ladder(by_lob, "paid"), "select one or group_by")

# Grain: quarterly origins and ages to annual origins.
q <- data.frame(
  origin = as.Date(rep(c("2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"), 4:1)),
  age = c(3, 6, 9, 12, 3, 6, 9, 3, 6, 3),
  paid = 10
)
qt <- triangle(q, "origin", "age", "paid", origin_grain = "Q", development_grain = "Q")
stopifnot(identical(qt@origins, c("2020Q1", "2020Q2", "2020Q3", "2020Q4")),
          identical(qt@development, c(3L, 6L, 9L, 12L)))
yt <- grain(qt, "Y", "Q")
stopifnot(identical(yt@origins, "2020"), yt@origin_grain == "Y", yt@development_grain == "Q")
near(yt@values[1, 1, "2020", ], c(10, 20, 30, 40))
near(grain(qt, "Y", "Y")@values[1, 1, "2020", "12"], 40)
# development_grain defaults to origin_grain.
ya <- grain(qt, "Y")
stopifnot(ya@development_grain == "Y", identical(ya@values, grain(qt, "Y", "Y")@values),
          identical(ya@valuation, as.Date("2020-12-31")))
stopifnot(identical(grain(raa, "Y", "Y")@values, raa@values))

# Errors are ordinary R errors carrying the Rust message.
expect_error_like(chain_ladder(multi, "paid"), "select one or group_by")
expect_error_like(chain_ladder(multi), "several columns")
expect_error_like(chain_ladder(raa, "paid"), "no column named paid")
expect_error_like(chain_ladder(raa, tail = 0), "tail factor 0")
expect_error_like(chain_ladder(raa, average = "median"), "should be one of")
expect_error_like(chain_ladder(raa, tail = NA), "tail must be a single number")
expect_error_like(triangle(transform(raw, value = factor(value)), "origin", "development", "value"),
                  "value column value must be numeric")
expect_error_like(triangle(transform(raw, value = as.character(value)), "origin", "development", "value"),
                  "value column value must be numeric")
expect_error_like(triangle(transform(raw, development = factor(development)), "origin", "development",
                           "value"), "development column development must be numeric")
expect_error_like(grain(qt, "X"), "origin_grain must be")
expect_error_like(grain(qt), "origin_grain")
expect_error_like(grain(raa, "Q", "Q"), "invalid grain change")
expect_error_like(subset(multi, columns = c("paid", "paid")), "supplied twice")
expect_error_like(subset(multi, lob = "nope"), "no segment has lob = \"nope\"")
expect_error_like(mack(raa, sigma_interpolation = "x"), "should be one of")
short <- triangle(data.frame(y = c(2020, 2020, 2021), d = c(12, 24, 12), v = c(1, 2, 3)), "y", "d", "v")
expect_error_like(mack(short), "at least 3 development ages")
expect_error_like(triangle(transform(raw, value = Inf), "origin", "development", "value"), "infinite")
expect_error_like(triangle(raw, "origin", "development", "value", origin_grain = "Y",
                           development_grain = "Y", cumulative = TRUE, keys = "lob"), "no columns named lob")
expect_error_like(triangle(two, "origin", "development", "paid", keys = c("lob", "lob")),
                  "key lob is supplied twice")
expect_error_like(triangle(two, "origin", "development", "paid", keys = "paid"),
                  "paid is both a key and a value column")
expect_error_like(triangle(transform(two, lob = replace(lob, 3, NA)), "origin", "development", "paid",
                           keys = "lob"), "key column lob has missing values")
clash <- triangle(transform(raw, year = origin, origin = "x"), "year", "development", "value",
                  keys = "origin")
expect_error_like(as.data.frame(clash), "clashes with the")
expect_error_like(triangle(transform(raw, development = replace(development, 2, 25)), "origin", "development",
                           "value", development_grain = "Y"), "not on the development grid")
expect_error_like(triangle(transform(raw, origin = origin + 0.5), "origin", "development", "value"),
                  "whole-number years")

# ODP bootstrap against R ChainLadder's BootChainLadder
# (validation/reference/reserving_bootstrap_r.csv). The tolerances for
# simulated quantities assume 20000 simulations with this seed, as in
# validation/tests/reserving.rs.
boot_lines <- grep("^#", readLines(validation("reference", "reserving_bootstrap_r.csv")),
                   value = TRUE, invert = TRUE)
boot_fields <- lapply(strsplit(boot_lines, ",", fixed = TRUE), `[`, 1:7)
boot_ref <- utils::read.csv(text = vapply(boot_fields, paste, "", collapse = ","),
                            colClasses = c(arg = "character"))
boot_ref <- boot_ref[boot_ref$dataset == "raa", ]
boot <- odp_bootstrap(raa, n_sims = 20000, seed = 20261004)
stopifnot(S7::S7_inherits(boot, odp_bootstrap_fit),
          S7::S7_inherits(boot@reserves, predictive_distribution),
          S7::S7_inherits(boot@chain_ladder, chain_ladder_fit))
near(boot@scale, boot_ref$expected[boot_ref$quantity == "scale"], 1e-9)
res_ref <- boot_ref[boot_ref$quantity == "residual", ]
stopifnot(nrow(res_ref) == 55)
for (r in seq_len(nrow(res_ref))) {
  at <- strsplit(res_ref$arg[r], ":", fixed = TRUE)[[1]]
  got <- boot@residuals[at[1], as.integer(at[2]) + 1L]
  want <- res_ref$expected[r]
  if (!(abs(got - want) <= 1e-9 * max(abs(want), 1))) {
    stop(sprintf("residual %s: got %.17g, want %.17g", res_ref$arg[r], got, want), call. = FALSE)
  }
}
# Origin x development matrices, NA below the latest diagonal.
stopifnot(identical(dim(boot@residuals), c(10L, 10L)),
          identical(dimnames(boot@fitted)$origin, raa@origins),
          is.na(boot@residuals["1990", "24"]), is.na(boot@fitted["1990", "24"]))
near(rowSums(boot@fitted, na.rm = TRUE), unname(boot@chain_ladder@latest), 1e-12)
near(boot@chain_ladder@total_reserve, chain_ladder(raa)@total_reserve)
for (quantity in c("mean_total", "sd_total")) {
  row <- boot_ref[boot_ref$method == "odp_bootstrap_gamma" & boot_ref$quantity == quantity, ]
  got <- if (quantity == "mean_total") mean(boot@reserves) else sqrt(variance(boot@reserves))
  if (!(abs(got - row$expected) <= row$abs_tol)) {
    stop(sprintf("%s: got %.17g, want %.17g +- %g", quantity, got, row$expected, row$abs_tol),
         call. = FALSE)
  }
}
# One component per origin period; the oldest origin is fully developed.
stopifnot(identical(boot@reserves@dims, "origin"),
          identical(boot@reserves@keys$origin, raa@origins),
          identical(boot@origins, raa@origins),
          identical(boot@development, raa@development),
          boot@reserves@n_sims == 20000,
          identical(dim(draw_matrix(boot@reserves)), c(20000L, 10L)),
          all(draw_matrix(boot@reserves)[, 1] == 0),
          provenance(boot@reserves)$model == "odp_bootstrap")
# Same seed, same draws; process error adds variance.
a <- odp_bootstrap(raa, n_sims = 500, seed = 7)
stopifnot(identical(draw_matrix(a@reserves), draw_matrix(odp_bootstrap(raa, n_sims = 500, seed = 7)@reserves)),
          !identical(draw_matrix(a@reserves), draw_matrix(odp_bootstrap(raa, n_sims = 500, seed = 8)@reserves)))
gamma <- odp_bootstrap(raa, "value", n_sims = 5000, seed = 1, process = "gamma")
param <- odp_bootstrap(raa, "value", n_sims = 5000, seed = 1, process = "none")
stopifnot(variance(param@reserves) < variance(gamma@reserves), param@scale == gamma@scale)
invisible(utils::capture.output(print(boot)))
expect_error_like(odp_bootstrap(raa, process = "poisson"), "should be one of")
expect_error_like(odp_bootstrap(raa, n_sims = 0), "n_sims must be positive")
expect_error_like(odp_bootstrap(raa, n_sims = 1.5), "n_sims must be a non-negative whole number")
expect_error_like(odp_bootstrap(raa, seed = NA), "seed must be a single number")
expect_error_like(odp_bootstrap(multi, "paid", n_sims = 10), "select one or group_by")
expect_error_like(odp_bootstrap(raa, "paid", n_sims = 10), "no column named paid")
tiny <- triangle(data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(1, 2, 1)),
                 "year", "age", "paid")
expect_error_like(odp_bootstrap(tiny, n_sims = 10), "degrees of freedom")

cat("actuarialrs R reserving tests passed\n")
