# Models lane: neural networks (prospicio-nn) over crates/prospicio-r/src/nn.rs:
# the CANN, the attention CANN and networks imported from ONNX. The design
# matrix is R's model.matrix(), as for glm_fit().

#' @include models.R
NULL

# A factor's indicator columns renamed `factor[level]`, so the attention
# CANN reads them as one token; other columns keep their names.
nn_columns <- function(x, terms, xlevels) {
  assign <- attr(x, "assign")
  labels <- attr(terms, "term.labels")
  nm <- colnames(x)
  for (j in seq_along(nm)) {
    a <- assign[j]
    if (a > 0 && labels[a] %in% names(xlevels) && startsWith(nm[j], labels[a])) {
      nm[j] <- paste0(labels[a], "[", substring(nm[j], nchar(labels[a]) + 1), "]")
    }
  }
  nm
}

nn_newdata <- function(object, newdata, offset) {
  nd <- new_design(object, newdata, offset)
  nd$names <- nn_columns(nd$x, object@terms, object@xlevels)
  nd
}

nn_training <- function(validation_share, patience) {
  list(share = if (is.null(validation_share)) NaN else as.double(validation_share),
       patience = as.double(patience))
}

#' Fit a CANN (combined actuarial neural network)
#'
#' Wüthrich and Merz's Combined Actuarial Neural Network: the offset (a
#' GLM's linear predictor, with `offset(...)` terms or `offset`) plus a
#' feed-forward network of the formula's design columns,
#' `eta = offset + NN(x)`. The network's output layer starts at zero, so
#' training starts exactly at the GLM and learns only what it misses.
#' Training minimizes the family's weighted deviance by Adam on
#' mini-batches, on the CPU in double precision, and is reproducible from
#' `seed`. Columns are standardized internally.
#'
#' @param formula A model formula; `offset(...)` terms are honoured.
#' @param data A data frame.
#' @param family `"poisson"`, `"gamma"`, `"tweedie"` (needs `power`) or
#'   `"gaussian"`.
#' @param link `NULL` for log (identity for the Gaussian), the only links the
#'   networks support.
#' @param hidden Widths of the hidden ReLU layers.
#' @param epochs,batch_size,learning_rate Adam's settings.
#' @param seed Seeds the initial weights and the mini-batch order.
#' @param validation_share `NULL`, or a share of rows held out for early
#'   stopping: the mean deviance on them is scored after each epoch and the
#'   best epoch kept.
#' @param patience With early stopping, epochs without improvement before
#'   stopping.
#' @param offset Optional offset added to any `offset()` terms.
#' @param weights Optional prior weights.
#' @param power Tweedie power in `(1, 2)`.
#' @returns A `cann_model` with properties `fitted`, `dispersion`,
#'   `best_epoch` (`NA` without early stopping) and `validation_history`.
#'   Use [stats::predict()], [predict_distribution()] and
#'   [network_correction()]. A fitted network has no artifact yet, so
#'   [save_model()] does not take it.
#' @export
#' @examples
#' set.seed(1)
#' d <- data.frame(x = seq(-2, 2, length.out = 200))
#' d$y <- rpois(200, exp(1 - d$x^2))
#' m <- cann_fit(y ~ x, d, offset = rep(log(mean(d$y)), 200), epochs = 20,
#'               learning_rate = 0.01)
#' head(predict(m, d))
cann_fit <- function(formula, data, family = "poisson", link = NULL, hidden = c(16, 8),
                     epochs = 200, batch_size = 64, learning_rate = 1e-3, seed = 0,
                     validation_share = NULL, patience = 10, offset = NULL, weights = NULL,
                     power = NULL) {
  des <- model_design(formula, data, offset, weights)
  tr <- nn_training(validation_share, patience)
  ptr <- rust_result(cann_fit_design(
    as.double(des$x), nn_columns(des$x, des$terms, des$xlevels), des$y, des$offset,
    des$weights, family, if (is.null(link)) "" else link,
    if (is.null(power)) NaN else as.double(power), as.double(hidden), as.double(epochs),
    as.double(batch_size), as.double(learning_rate), as.double(seed), tr$share, tr$patience
  ))
  cann_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels)
}

nn_properties <- function(ptr_class) {
  list(
    ptr = S7::new_S3_class(ptr_class),
    terms = S7::class_any,
    xlevels = S7::class_any,
    fitted = S7::new_property(S7::class_double, getter = function(self) self@ptr$fitted()),
    dispersion = S7::new_property(S7::class_double, getter = function(self) self@ptr$dispersion()),
    best_epoch = S7::new_property(S7::class_double, getter = function(self) self@ptr$best_epoch()),
    validation_history = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$validation_history()
    })
  )
}

#' Fitted CANN (class)
#'
#' Returned by [cann_fit()].
#'
#' @param ptr,terms,xlevels Internal.
#' @returns A `cann_model` object.
#' @export
cann_model <- S7::new_class(
  "cann_model",
  package = "prospicio",
  properties = nn_properties("CannModel"),
  constructor = function(ptr, terms, xlevels) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels)
  }
)

#' Fit an attention CANN
#'
#' A CANN whose correction is a transformer over feature tokens (an
#' FT-Transformer in the CANN setting). Each numeric column is a token and
#' each factor's indicator columns together form one token, so every level
#' gets a learned embedding. A learned `[CLS]` token joins them; pre-norm
#' blocks of multi-head self-attention and a feed-forward layer mix them;
#' the `[CLS]` state, through a zero-initialized head, corrects the linear
#' predictor, so training starts exactly at the GLM. [attention_weights()]
#' shows, per risk, which rating factors the model leans on.
#'
#' @inheritParams cann_fit
#' @param d_model Token width; a multiple of `heads`.
#' @param heads Attention heads.
#' @param layers Transformer blocks.
#' @param feed_forward Width of each block's feed-forward layer.
#' @returns An `attention_cann_model` with the properties of a [cann_model]
#'   and `tokens`.
#' @export
#' @examples
#' set.seed(1)
#' d <- data.frame(age = runif(200, -1, 1), region = sample(c("A", "B", "C"), 200, TRUE))
#' d$y <- rgamma(200, 4, 4 / exp(0.5 * d$age + 0.4 * (d$region == "B")))
#' m <- attention_cann_fit(y ~ age + region, d, family = "gamma", d_model = 8,
#'                         layers = 1, epochs = 5)
#' m@tokens
#' colMeans(attention_weights(m, d)[, 1, ])
attention_cann_fit <- function(formula, data, family = "poisson", link = NULL, d_model = 16,
                               heads = 2, layers = 2, feed_forward = 32, epochs = 100,
                               batch_size = 64, learning_rate = 1e-3, seed = 0,
                               validation_share = NULL, patience = 10, offset = NULL,
                               weights = NULL, power = NULL) {
  des <- model_design(formula, data, offset, weights)
  tr <- nn_training(validation_share, patience)
  ptr <- rust_result(attention_cann_fit_design(
    as.double(des$x), nn_columns(des$x, des$terms, des$xlevels), des$y, des$offset,
    des$weights, family, if (is.null(link)) "" else link,
    if (is.null(power)) NaN else as.double(power),
    as.double(c(d_model, heads, layers, feed_forward)), as.double(epochs),
    as.double(batch_size), as.double(learning_rate), as.double(seed), tr$share, tr$patience
  ))
  attention_cann_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels)
}

#' Fitted attention CANN (class)
#'
#' Returned by [attention_cann_fit()].
#'
#' @param ptr,terms,xlevels Internal.
#' @returns An `attention_cann_model` object.
#' @export
attention_cann_model <- S7::new_class(
  "attention_cann_model",
  package = "prospicio",
  properties = c(nn_properties("AttentionCannModel"), list(
    tokens = S7::new_property(S7::class_character, getter = function(self) self@ptr$tokens())
  )),
  constructor = function(ptr, terms, xlevels) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels)
  }
)

for (cls in list(cann_model, attention_cann_model)) {
  S7::method(predict, cls) <- function(object, newdata, offset = NULL, ...) {
    nd <- nn_newdata(object, newdata, offset)
    rust_result(object@ptr$predict(as.double(nd$x), nd$names, nd$offset), s7_call())
  }
  S7::method(predict_distribution, cls) <- function(object, newdata, n_sims, seed,
                                                     offset = NULL, weights = NULL, ...) {
    nd <- nn_newdata(object, newdata, offset)
    w <- if (is.null(weights)) double() else as.double(weights)
    ptr <- rust_result(object@ptr$predict_distribution(
      as.double(nd$x), nd$names, nd$offset, w, as.double(n_sims), as.double(seed)
    ), s7_call())
    predictive_distribution(ptr = ptr)
  }
  S7::method(print, cls) <- function(x, ...) {
    cat(sprintf("<%s> %d training rows, dispersion %s\n", class(x)[1], length(x@fitted),
                format(x@dispersion, digits = 6)))
    invisible(x)
  }
}

#' A network's correction to the linear predictor
#'
#' `NN(x)` per row of `newdata`, on the link scale: the prediction is
#' `g^-1(offset + NN(x))`.
#'
#' @param model A [cann_model] or [attention_cann_model].
#' @param newdata A data frame with the model's terms.
#' @returns A numeric vector.
#' @export
network_correction <- function(model, newdata) {
  nd <- nn_newdata(model, newdata, NULL)
  rust_result(model@ptr$correction(as.double(nd$x), nd$names))
}

#' Attention weights of an attention CANN
#'
#' The last block's attention from `[CLS]` to each feature token, per row of
#' `newdata` and head: where the model looked for each risk. Per row and
#' head the weights over the feature tokens and `[CLS]` itself sum to 1. A
#' diagnostic, not a decomposition of the prediction; averaged over a
#' segment's rows it shows which rating factors the model leans on there.
#'
#' @param model An [attention_cann_model] with at least one block.
#' @param newdata A data frame with the model's terms.
#' @returns An array `[row, head, token]` with the tokens as names of its
#'   third dimension.
#' @export
attention_weights <- function(model, newdata) {
  nd <- nn_newdata(model, newdata, NULL)
  a <- rust_result(model@ptr$attention(as.double(nd$x), nd$names))
  t <- length(a$tokens)
  # Row-major [row][head][token] read into R's column-major order.
  w <- aperm(array(a$weights, c(t, a$heads, a$rows)), 3:1)
  dimnames(w) <- list(NULL, NULL, a$tokens)
  w
}

# Row-major values of an array (or vector) and its shape.
row_major <- function(a) {
  d <- if (is.null(dim(a))) length(a) else dim(a)
  list(shape = as.double(d), values = as.double(aperm(array(a, d), rev(seq_along(d)))))
}

#' Load a network from an ONNX file
#'
#' A network trained elsewhere (PyTorch, Keras), run in double precision.
#' It supports the operators a tabular network exports: `Gemm`, `Linear`,
#' `MatMul`, broadcasting arithmetic, the common activations, `Softmax`,
#' `Concat`, `Gather` embeddings, reshapes and reductions; anything else is
#' refused at load, with the operators the file uses. Outputs agree with
#' onnxruntime within float32 rounding.
#'
#' @param file A path to an `.onnx` file, or its bytes as a raw vector.
#' @returns An `onnx_model` with properties `inputs` (a data frame of each
#'   graph input's `name`, `rank` and `integer`, with its `shape` in the
#'   `"shapes"` attribute, `NA` where the file leaves a size free), `outputs`
#'   and `ops`. Use [onnx_run()], or [onnx_predictor()] to predict from
#'   data frame columns.
#' @export
#' @examples
#' f <- system.file("extdata", "mlp.onnx", package = "prospicio")
#' if (nzchar(f)) onnx_model(f)@ops
onnx_model <- function(file) {
  ptr <- if (is.raw(file)) rust_result(onnx_from_bytes_rust(file))
         else rust_result(onnx_load_rust(normalizePath(file, mustWork = TRUE)))
  onnx_model_class(ptr = ptr)
}

#' ONNX model (class)
#'
#' Returned by [onnx_model()].
#'
#' @param ptr Internal.
#' @returns An `onnx_model` object.
#' @export
onnx_model_class <- S7::new_class(
  "onnx_model",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("OnnxNet"),
    inputs = S7::new_property(S7::class_data.frame, getter = function(self) {
      i <- self@ptr$inputs()
      out <- data.frame(name = i$name, rank = i$rank, integer = i$integer)
      shapes <- lapply(i$shape, function(s) replace(s, is.nan(s), NA))
      attr(out, "shapes") <- stats::setNames(shapes, i$name)
      out
    }),
    outputs = S7::new_property(S7::class_character, getter = function(self) self@ptr$outputs()),
    ops = S7::new_property(S7::class_character, getter = function(self) self@ptr$ops())
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

S7::method(print, onnx_model_class) <- function(x, ...) {
  cat(sprintf("<onnx_model> inputs %s; outputs %s; ops %s\n", toString(x@inputs$name),
              toString(x@outputs), toString(x@ops)))
  invisible(x)
}

#' Run an ONNX model
#'
#' @param model An [onnx_model].
#' @param inputs A list of arrays, matrices or vectors, one per graph input
#'   in input order (a matrix is `[rows, columns]`).
#' @returns A list of arrays, one per output, in R's usual layout.
#' @export
onnx_run <- function(model, inputs) {
  rm <- lapply(inputs, row_major)
  out <- rust_result(model@ptr$run(lapply(rm, `[[`, "shape"), lapply(rm, `[[`, "values")))
  lapply(out, function(o) {
    s <- o$shape
    if (length(s) <= 1) return(o$values)
    aperm(array(o$values, rev(s)), rev(seq_along(s)))
  })
}

#' An ONNX model behind the model interface
#'
#' Binds each graph input to data frame columns and reads one output as each
#' row's mean, with a family for [predict_distribution()] (an imported
#' network carries no parameter uncertainty and the file no family, so the
#' draws are process draws around the predicted means).
#'
#' @param model An [onnx_model].
#' @param family The family of the predictive draws (`"poisson"`, `"gamma"`,
#'   ...).
#' @param inputs A named list: each graph input's column names. A rank-2
#'   input takes its columns in order (`[rows, columns]`), a rank-1 input one
#'   column; an integer input (an embedding's category codes) must hold
#'   whole numbers.
#' @param output Which output is the mean (1 for the first).
#' @param scale `"mean"`: the output is the mean. `"link"`: the output is on
#'   the link scale and added to the offset, `mu = g^-1(offset + output)`.
#' @param link The link for `scale = "link"`; `NULL` for the family's
#'   canonical link.
#' @param dispersion The family's dispersion for predictive draws.
#' @param theta,power The negative binomial's `theta` or the Tweedie power.
#' @returns An `onnx_predictor`. Use [stats::predict()] and
#'   [predict_distribution()] with a data frame holding the bound columns
#'   (and `offset =` for `scale = "link"`).
#' @export
onnx_predictor <- function(model, family, inputs, output = 1, scale = c("mean", "link"),
                           link = NULL, dispersion = 1, theta = NULL, power = NULL) {
  scale <- match.arg(scale)
  if (!is.list(inputs) || is.null(names(inputs)) || any(!nzchar(names(inputs)))) {
    stop("inputs must be a named list of column names")
  }
  fa <- family_args(theta, power)
  ptr <- rust_result(onnx_predictor_rust(
    model@ptr, family, names(inputs), lapply(inputs, as.character), as.double(output - 1),
    scale, if (is.null(link)) "" else link, as.double(dispersion), fa$theta, fa$power
  ))
  onnx_predictor_class(ptr = ptr, columns = unique(unlist(inputs, use.names = FALSE)))
}

#' ONNX predictor (class)
#'
#' Returned by [onnx_predictor()].
#'
#' @param ptr,columns Internal.
#' @returns An `onnx_predictor` object.
#' @export
onnx_predictor_class <- S7::new_class(
  "onnx_predictor",
  package = "prospicio",
  properties = list(ptr = S7::new_S3_class("OnnxPredictorModel"), columns = S7::class_character),
  constructor = function(ptr, columns) {
    S7::new_object(S7::S7_object(), ptr = ptr, columns = columns)
  }
)

onnx_design <- function(object, newdata) {
  missing <- setdiff(object@columns, names(newdata))
  if (length(missing)) stop("newdata has no column ", toString(missing))
  as.double(unlist(lapply(object@columns, function(c) as.double(newdata[[c]]))))
}

S7::method(predict, onnx_predictor_class) <- function(object, newdata, offset = NULL, ...) {
  rust_result(object@ptr$predict(onnx_design(object, newdata), object@columns,
                                 if (is.null(offset)) double() else as.double(offset)), s7_call())
}

S7::method(predict_distribution, onnx_predictor_class) <- function(object, newdata, n_sims, seed,
                                                                  offset = NULL, weights = NULL,
                                                                  ...) {
  ptr <- rust_result(object@ptr$predict_distribution(
    onnx_design(object, newdata), object@columns,
    if (is.null(offset)) double() else as.double(offset),
    if (is.null(weights)) double() else as.double(weights), as.double(n_sims), as.double(seed)
  ), s7_call())
  predictive_distribution(ptr = ptr)
}

S7::method(print, onnx_predictor_class) <- function(x, ...) {
  cat(sprintf("<onnx_predictor> %s on columns %s\n", x@ptr$family(), toString(x@columns)))
  invisible(x)
}
