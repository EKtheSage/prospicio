suppressMessages(library(prospicio))

# The Munich chain ladder (Quarg and Mack 2004; docs/design/reserving-v02.md,
# decision 30). The Rust core is checked against R ChainLadder and
# chainladder-python in validation/tests/reserving_munich.rs; these tests
# check the binding's arguments, results, frames, print and errors against
# the same R ChainLadder reference file, and against MunichChainLadder()
# itself when ChainLadder is installed.

near <- function(a, b, rel = 1e-12, abs = 1e-9) {
  a <- unname(a)
  b <- unname(b)
  if (length(a) != length(b) || !all(abs(a - b) <= pmax(rel * abs(b), abs))) {
    stop(sprintf("got %s, want %s", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " ")), call. = FALSE)
  }
}

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop(sprintf("expected an error mentioning %s", shQuote(pattern)), call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

# Tests run from the repository root (CI and `cargo xtask r`).
validation <- function(...) file.path("validation", ...)
long <- utils::read.csv(validation("data", "mcl.csv"), comment.char = "#")
mcl <- triangle(long, "origin", "development", c("paid", "incurred"))
lines <- grep("^#", readLines(validation("reference", "reserving_munich_r.csv")),
              value = TRUE, invert = TRUE)
# The last column, `source`, holds commas: keep the first five fields.
fields <- lapply(strsplit(lines[-1], ",", fixed = TRUE), function(f) c(f, "")[1:5])
ref <- stats::setNames(as.data.frame(do.call(rbind, fields), stringsAsFactors = FALSE),
                       c("dataset", "test", "quantity", "arg", "expected"))
ref$expected <- as.numeric(ref$expected)
want <- function(test, quantity) ref$expected[ref$test == test & ref$quantity == quantity]

# R's MunichChainLadder(MCLpaid, MCLincurred, est.sigmaP = "Mack",
# est.sigmaI = "Mack"), the reference's "mack" arm.
m <- munich_chain_ladder(mcl, sigma_interpolation_paid = "mack",
                         sigma_interpolation_incurred = "mack")
near(m@lambda_paid, want("mack", "lambda_paid"))
near(m@lambda_incurred, want("mack", "lambda_incurred"))
near(m@ultimate_paid, want("mack", "ultimate_paid"))
near(m@ultimate_incurred, want("mack", "ultimate_incurred"))
near(m@q, want("mack", "q"))
near(m@q_inverse, want("mack", "q_inverse"))
near(m@rho_paid, want("mack", "rho_paid"))
near(m@rho_incurred, want("mack", "rho_incurred"))
stopifnot(identical(m@n_residuals_paid, 20L), identical(m@n_residuals_incurred, 20L), identical(m@development, 12L * 1:7))
stopifnot(identical(names(m@ultimate_paid), as.character(2001:2007)))
near(m@total_ultimate_paid, 32121.4969949898, rel = 1e-12)

# The tails, R's tailP = 1.05 and tailI = 1.02.
t <- munich_chain_ladder(mcl, sigma_interpolation_paid = "mack",
                         sigma_interpolation_incurred = "mack",
                         tail_paid = 1.05, tail_incurred = 1.02)
near(t@ultimate_paid, want("mack;tail=1.05/1.02", "ultimate_paid"))
near(t@ultimate_incurred, want("mack;tail=1.05/1.02", "ultimate_incurred"))
lt <- munich_chain_ladder(mcl, sigma_interpolation_paid = "mack",
                          sigma_interpolation_incurred = "mack",
                          tail_paid = tail_log_linear(), tail_incurred = tail_log_linear())
near(lt@ultimate_paid, want("mack;tail=log-linear", "ultimate_paid"))

# Against R ChainLadder itself, when it is installed.
if (requireNamespace("ChainLadder", quietly = TRUE)) {
  r <- suppressWarnings(ChainLadder::MunichChainLadder(
    ChainLadder::MCLpaid, ChainLadder::MCLincurred, est.sigmaP = "Mack", est.sigmaI = "Mack"))
  near(m@ultimate_paid, r$MCLPaid[, 7])
  near(m@lambda_incurred, coef(r$lambdaI)[1])
}

# Incurred flat from 60 months in 2001 and 2002: incurred's sigma is 0 at 60
# to 72 months and its residuals there are 0/0, which R's lm drops from the
# incurred regression only (R ChainLadder 0.2.21, nobs 20 and 18).
flat <- long
at <- function(o, d) flat$origin == o & flat$development == d
flat$incurred[at(2001, 72) | at(2001, 84)] <- 2174
flat$incurred[at(2002, 72)] <- 2508
z <- munich_chain_ladder(triangle(flat, "origin", "development", c("paid", "incurred")),
                         sigma_interpolation_paid = "mack",
                         sigma_interpolation_incurred = "mack")
stopifnot(identical(z@n_residuals_paid, 20L), identical(z@n_residuals_incurred, 18L))
near(z@lambda_paid, 0.636021466425142)
near(z@lambda_incurred, 0.413992067837447)
if (requireNamespace("ChainLadder", quietly = TRUE)) {
  i <- ChainLadder::MCLincurred
  i[1, 6] <- i[1, 5]
  i[2, 6] <- i[2, 5]
  i[1, 7] <- i[1, 6]
  rz <- suppressWarnings(ChainLadder::MunichChainLadder(
    ChainLadder::MCLpaid, i, est.sigmaP = "Mack", est.sigmaI = "Mack"))
  near(z@lambda_paid, coef(rz$lambdaP)[1])
  near(z@lambda_incurred, coef(rz$lambdaI)[1])
  near(z@ultimate_paid, rz$MCLPaid[, 7], rel = 1e-10)
  near(z@ultimate_incurred, rz$MCLIncurred[, 7], rel = 1e-10)
}

# Frames, the separate fits and print.
df <- as.data.frame(m)
stopifnot(identical(names(df), c("origin", "latest_paid", "latest_incurred", "latest_ratio",
                                 "ultimate_paid", "ultimate_incurred", "ultimate_ratio",
                                 "reserve_paid", "ibnr_incurred", "chain_ladder_paid",
                                 "chain_ladder_incurred")))
near(df$ultimate_ratio[7], 0.980301200225115)
totals <- totals_frame(m)
near(totals$lambda_paid, m@lambda_paid)
dev <- development_frame(m)
stopifnot(identical(dev$development, 12L * 1:7), is.na(dev$ldf_paid[7]))
near(m@chain_ladder_paid, chain_ladder(mcl, "paid")@ultimate)
stopifnot(S7::S7_inherits(m@mack_paid, mack_fit))
out <- utils::capture.output(print(m))
stopifnot(grepl("<munich_fit> total ultimate paid 32121.49699, incurred 32719.5125", out[1], fixed = TRUE))

# The default (log-linear sigma) is chainladder-python's.
d <- munich_chain_ladder(mcl)
near(d@ultimate_paid[["2007"]], 7561.219696695722)

# Segments.
two <- rbind(cbind(lob = "A", long), cbind(lob = "B", long))
s <- munich_chain_ladder(triangle(two, "origin", "development", c("paid", "incurred"), keys = "lob"))
stopifnot(length(s@ultimate_paid) == 14)
expect_error_like(s@lambda_paid, "segment")
near(segment(s, lob = "B")@lambda_paid, d@lambda_paid)

# No information: incurred a fixed multiple of paid.
fixed <- transform(long, incurred = 1.25 * paid)
f <- munich_chain_ladder(triangle(fixed, "origin", "development", c("paid", "incurred")))
stopifnot(f@lambda_paid == 0, f@lambda_incurred == 0)
near(f@ultimate_paid, f@chain_ladder_paid)

# A missing incurred cell is refused.
holed <- long
holed$incurred[holed$origin == 2004 & holed$development == 48] <- NA
expect_error_like(
  munich_chain_ladder(triangle(holed, "origin", "development", c("paid", "incurred"))),
  "origin 2004 at age 48"
)
expect_error_like(munich_chain_ladder(mcl, sigma_interpolation_paid = "cubic"), "should be one of")
