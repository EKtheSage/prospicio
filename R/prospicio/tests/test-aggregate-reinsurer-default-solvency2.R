library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The Solvency II counterparty default module for type 1 exposures (#980)
# against validation/reference/reinsurer_default_solvency2.csv, which
# validation/scripts/aggregate_reinsurer_default.py computes from the text of
# Delegated Regulation (EU) 2015/35, Arts 192, 199, 200 and 201.
ref <- read.csv(validation("reference", "reinsurer_default_solvency2.csv"),
                colClasses = "character")
stopifnot(nrow(ref) == 78)
params <- function(s) {
  pairs <- strsplit(strsplit(s, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.double(vapply(pairs, `[`, "", 2)), vapply(pairs, `[`, "", 1))
}
evaluate <- function(row) {
  kind <- row[["distribution"]]
  quantity <- row[["quantity"]]
  arg <- row[["arg"]]
  if (kind == "probability_by_step" && quantity == "probability") {
    return(solvency2_probability_by_step(as.double(arg)))
  }
  if (kind == "probability_by_ratio" && quantity == "probability") {
    return(solvency2_probability_by_ratio(as.double(arg)))
  }
  p <- params(row[["params"]])
  if (kind == "reinsurance_lgd" && quantity == "lgd") {
    return(solvency2_reinsurance_lgd(p[["recoverables"]], p[["risk_mitigation"]],
                                     p[["collateral"]], lgd_share = p[["lgd_share"]],
                                     factor = p[["factor"]]))
  }
  if (kind == "type1" && quantity %in% c("v_inter", "v_intra", "sigma", "total_lgd", "scr")) {
    i <- seq_len(p[["n"]])
    # sprintf, not paste0, so that no exposures give no names.
    t <- solvency2_type1(pd = unname(p[sprintf("pd_%d", i)]), lgd = unname(p[sprintf("lgd_%d", i)]))
    return(t[[quantity]])
  }
  stop(sprintf("no evaluation for %s %s", kind, quantity))
}
for (k in seq_len(nrow(ref))) {
  row <- ref[k, ]
  near(evaluate(row), as.double(row[["expected"]]), rel = as.double(row[["rel_tol"]]),
       abs = as.double(row[["abs_tol"]]))
}

# One exposure: v_inter + v_intra = pd (1 - pd) lgd^2; 10.9 % of the total
# is the 5 sigma branch.
t <- solvency2_type1(0.012, 50)
near(t$v_inter + t$v_intra, 0.012 * 0.988 * 2500)
stopifnot(t$scr == 5 * t$sigma, t$total_lgd == 50)
stopifnot(solvency2_type1(numeric(0), numeric(0))$scr == 0)

expect_error_like(solvency2_probability_by_step(7), "step")
expect_error_like(solvency2_probability_by_step(1.5), "step")
expect_error_like(solvency2_probability_by_ratio(-0.5), "scr_ratio")
expect_error_like(solvency2_reinsurance_lgd(100, 0, -1), "collateral")
expect_error_like(solvency2_reinsurance_lgd(100, 0, 0, lgd_share = 1.5), "lgd_share")
expect_error_like(solvency2_type1(1.5, 10), "pd")
expect_error_like(solvency2_type1(c(0.01, 0.02), 10), "per exposure")
