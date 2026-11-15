library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The running premium-weighted loss ratio against the four values the AP
# pricing demo prints (#806), on its inputs in
# validation/data/ap_pricing_demo_inputs.json, the rows the Rust test reads.
# The two arrays are read with base R, so no JSON package is needed.
text <- paste(readLines(validation("data", "ap_pricing_demo_inputs.json")), collapse = "")
column <- function(name) {
  body <- regmatches(text, regexpr(sprintf('"%s": *\\[[^]]*\\]', name), text))
  as.numeric(strsplit(sub(".*\\[(.*)\\]", "\\1", body), ",")[[1]])
}
lr <- cumulative_loss_ratio(column("loss_ratios"), column("earned_premium"))
stopifnot(length(lr) == 14)
printed <- c(`1` = 0.610000000000, `2` = 0.594705882353, `3` = 0.638120662040,
             `14` = 0.637034303676)
for (row in names(printed)) near_abs(lr[as.integer(row)], printed[[row]], 5e-13)
