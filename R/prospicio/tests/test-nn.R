library(prospicio)

near <- function(a, b, rel = 1e-10) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " ")), call. = FALSE)
  }
}
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

# The CANN learns what a constant GLM misses, replays from its seed, and
# reads the offset on the log scale.
set.seed(1)
d <- data.frame(x = runif(600, -2, 2))
d$y <- rpois(600, exp(1 - d$x^2))
base <- log(mean(d$y))
m <- cann_fit(y ~ x, d, offset = rep(base, 600), hidden = c(16, 8), epochs = 60,
              learning_rate = 0.01, seed = 3)
stopifnot(S7::S7_inherits(m, cann_model))
glm_dev <- family_deviance("poisson", d$y, rep(exp(base), 600))
stopifnot(family_deviance("poisson", d$y, m@fitted) < 0.8 * glm_dev)
near(predict(m, d, offset = rep(base, 600)), m@fitted, 1e-12)
near(predict(m, d[1:3, , drop = FALSE], offset = rep(base, 3)),
     exp(base + network_correction(m, d[1:3, , drop = FALSE])), 1e-12)
stopifnot(m@dispersion == 1, is.na(m@best_epoch), length(m@validation_history) == 0)
m2 <- cann_fit(y ~ x, d, offset = rep(base, 600), hidden = c(16, 8), epochs = 60,
               learning_rate = 0.01, seed = 3)
stopifnot(identical(m@fitted, m2@fitted))
# Saved and loaded, the network predicts exactly as before.
mf <- tempfile(fileext = ".rds")
save_model(m, mf)
m3 <- load_model(mf)
stopifnot(S7::S7_inherits(m3, cann_model), identical(m3@ptr$input_hash(), m@ptr$input_hash()))
stopifnot(identical(predict(m3, d[1:5, , drop = FALSE], offset = rep(base, 5)),
                    predict(m, d[1:5, , drop = FALSE], offset = rep(base, 5))))
pd <- predict_distribution(m, d, n_sims = 4000, seed = 9, offset = rep(base, 600))
near(mean(rowSums(draw_matrix(pd))), sum(m@fitted), 0.02)

# Early stopping keeps the best epoch; unsupported settings are refused.
es <- cann_fit(y ~ x, d[1:300, ], offset = rep(base, 300), hidden = c(32, 32), epochs = 400,
               batch_size = 16, learning_rate = 0.03, validation_share = 0.3, patience = 5)
stopifnot(es@best_epoch >= 1, es@best_epoch <= length(es@validation_history),
          length(es@validation_history) < 400,
          es@validation_history[es@best_epoch] == min(es@validation_history))
stopifnot(fails(cann_fit(y ~ x, d, link = "identity", epochs = 1)))
stopifnot(fails(cann_fit(y ~ x, d, batch_size = 0)))

# The attention CANN reads a factor's indicators as one token.
set.seed(5)
a <- data.frame(age = runif(300, -1, 1), region = sample(c("A", "B", "C"), 300, TRUE))
a$y <- exp(0.5 * a$age + 0.4 * (a$region == "B")) * runif(300, 0.5, 1.5)
am <- attention_cann_fit(y ~ age + region, a, family = "gamma", d_model = 8, heads = 2,
                         layers = 1, feed_forward = 16, epochs = 8, seed = 2)
stopifnot(identical(am@tokens, c("age", "region")), am@dispersion > 0)
stopifnot(length(predict(am, a)) == 300)
af <- tempfile(fileext = ".rds")
save_model(am, af)
am2 <- load_model(af)
stopifnot(identical(predict(am2, a), predict(am, a)), identical(am2@tokens, am@tokens))
stopifnot(identical(attention_weights(am2, a[1:3, ]), attention_weights(am, a[1:3, ])))
w <- attention_weights(am, a)
stopifnot(identical(dim(w), c(300L, 2L, 2L)), identical(dimnames(w)[[3]], c("age", "region")),
          all(w >= 0), all(apply(w, c(1, 2), sum) <= 1 + 1e-12))
stopifnot(fails(attention_cann_fit(y ~ age + region, a, family = "gamma", d_model = 7)))
stopifnot(fails(attention_weights(
  attention_cann_fit(y ~ age + region, a, family = "gamma", layers = 0, epochs = 1), a)))

# ONNX import against onnxruntime (validation/reference/onnx_outputs.csv).
root <- normalizePath(file.path("..", "..", ".."), mustWork = FALSE)
if (!dir.exists(file.path(root, "validation"))) root <- normalizePath(".")
onnx_dir <- file.path(root, "validation", "data", "onnx")
ref <- utils::read.csv(file.path(root, "validation", "reference", "onnx_outputs.csv"),
                       colClasses = c(inputs = "character", codes = "character"))
nums <- function(s) as.double(strsplit(s, "|", fixed = TRUE)[[1]])
stopifnot(nrow(ref) == 36)
for (name in c("mlp", "embedding", "gemm")) {
  r <- ref[ref$model == name, ]
  x <- do.call(rbind, lapply(r$inputs, nums))
  inputs <- list(x)
  if (name == "embedding") inputs[[2]] <- vapply(r$codes, nums, 0)
  out <- onnx_run(onnx_model(file.path(onnx_dir, paste0(name, ".onnx"))), inputs)[[1]]
  stopifnot(length(out) == nrow(r))
  stopifnot(all(abs(out - r$expected) <= 1e-5 * pmax(abs(r$expected), 0.1)))
}
mlp <- onnx_model(file.path(onnx_dir, "mlp.onnx"))
stopifnot(identical(mlp@ops, c("Gemm", "Relu", "Gemm", "Exp")))
e <- onnx_model(readBin(file.path(onnx_dir, "embedding.onnx"), "raw", 1e6))
stopifnot(identical(e@inputs$name, c("x_num", "x_cat")), identical(e@inputs$integer, c(FALSE, TRUE)))
stopifnot(identical(attr(e@inputs, "shapes")$x_num, c(NA, 3)))
stopifnot(fails(onnx_run(mlp, list(matrix(0, 1, 5)))))
stopifnot(fails(onnx_model(as.raw(1:10))))

# The predictor binds data frame columns and draws.
r <- ref[ref$model == "embedding", ]
x <- do.call(rbind, lapply(r$inputs, nums))
nd <- data.frame(a = x[, 1], b = x[, 2], c = x[, 3], region = vapply(r$codes, nums, 0))
net <- onnx_predictor(e, "gamma", list(x_num = c("a", "b", "c"), x_cat = "region"), dispersion = 0.5)
mu <- predict(net, nd)
stopifnot(all(abs(mu - r$expected) <= 1e-5 * abs(r$expected)))
pd <- predict_distribution(net, nd, n_sims = 20000, seed = 3)
near(mean(rowSums(draw_matrix(pd))), sum(mu), 0.02)
log_net <- onnx_predictor(e, "gamma", list(x_num = c("a", "b", "c"), x_cat = "region"),
                          scale = "link", link = "log")
near(predict(log_net, nd, offset = rep(0.3, nrow(nd))), exp(0.3 + mu), 1e-12)
stopifnot(fails(predict(onnx_predictor(e, "gamma", list(x_num = c("a", "b", "c"))), nd)))
stopifnot(fails(predict(net, nd[, c("a", "b", "c")])))
bad <- nd
bad$region[1] <- 0.5
stopifnot(fails(predict(net, bad)))
stopifnot(fails(onnx_predictor(e, "gamma", list(x_num = "a"), scale = "mean", link = "log")))
cat("test-nn: ok\n")
