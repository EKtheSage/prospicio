# R earth's MARS knots on the Chalk portfolio, for
# validation/tests/models.rs::mars_matches_earth_on_the_chalk_portfolio.
# install.packages("earth"); run from the repository root.
library(earth)
d <- read.csv("examples/data/chalk_portfolio.csv", comment.char = "#")
d$freq <- d$claims / d$exposure
f <- freq ~ age + veh_age + exper + noise
show <- function(label, m) {
  s <- m$selected.terms[-1]
  cat(label, "\n")
  print(data.frame(term = rownames(m$dirs)[s]))
  cat("rss", format(m$rss, digits = 15), "gcv", format(m$gcv, digits = 15), "\n\n")
}
show("default", earth(f, d, degree = 1, pmethod = "none"))
show("weighted", earth(f, d, weights = exposure, degree = 1, pmethod = "none"))
show("nk 15", earth(f, d, degree = 1, pmethod = "none", thresh = 0, nk = 15))
show("weighted nk 15", earth(f, d, weights = exposure, degree = 1, pmethod = "none", thresh = 0, nk = 15))
show("backward", earth(f, d, degree = 1, thresh = 0, nk = 15))
