library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The stable law's algebra (#780), against aggregate 2.1.0's Stable: the
# rows of validation/reference/stable_algebra_aggregate.csv the Rust
# parity test reads.
ref <- read.csv(validation("reference", "stable_algebra_aggregate.csv"), stringsAsFactors = FALSE)
params <- function(text) {
  kv <- strsplit(strsplit(text, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.numeric(vapply(kv, `[`, "", 2)), vapply(kv, `[`, "", 1))
}
law <- function(v, p) {
  stable_distribution(v[[paste0(p, "_alpha")]], v[[paste0(p, "_beta")]],
                      v[[paste0(p, "_scale")]], v[[paste0(p, "_loc")]])
}
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  v <- params(r$params)
  st <- switch(r$arg,
    identity = law(v, "x"),
    add = stable_add(law(v, "x"), law(v, "y")),
    shift = stable_add(law(v, "x"), v[["c"]]),
    neg = stable_scale(law(v, "x"), -1),
    scale = stable_scale(law(v, "x"), v[["c"]]),
    copies = stable_copies(law(v, "x"), v[["c"]]),
    s0 = stable_distribution(v[["x_alpha"]], v[["x_beta"]], v[["x_scale"]], v[["x_loc"]], "S0"),
    compound_pareto = stable_from_compound_pareto(v[["frequency"]], v[["alpha"]], v[["theta"]],
                                                  v[["sev_mean"]]))
  near(S7::prop(st, r$quantity), r$expected, rel = r$rel_tol, abs = r$abs_tol)
}

x <- stable_distribution(1.5, 0.4, 2, 1)
near(stable_copies(x, 2)@scale, stable_add(x, x)@scale)
expect_error_like(stable_add(x, stable_distribution(1.2, 0)), "alpha")
expect_error_like(stable_scale(x, 0), "point mass")
