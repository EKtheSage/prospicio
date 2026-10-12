// The Bayesian over-dispersed Poisson chain ladder of
// docs/design/reserving-bayes.md (decision 3), the reference for
// prospicio_reserving::bayes::BayesOdp. Run by
// validation/scripts/reserving_bayes_stan.py.
//
// eta[w, d] = c + a[w] + b[d], a[1] = b[1] = 0, on the observed incremental
// cells, with the scaled Poisson quasi log likelihood (x eta - exp(eta)) / phi,
// which accepts negative and non-integer increments; phi (the chain ladder's
// Pearson scale) and c0 (the log of the mean observed increment) are data.
data {
  int<lower=1> N;                       // observed cells
  int<lower=2> n_origin;
  int<lower=2> n_dev;
  array[N] int<lower=1, upper=n_origin> w;
  array[N] int<lower=1, upper=n_dev> d;
  vector[N] x;                          // incremental values
  real<lower=0> phi;
  real c0;
  real<lower=0> intercept_sd;
  real<lower=0> origin_sd;
  real<lower=0> development_sd;
  int<lower=0> N_fut;                   // future cells up to the oldest age
  array[N_fut] int<lower=1, upper=n_origin> fw;
  array[N_fut] int<lower=1, upper=n_dev> fd;
  int<lower=0, upper=1> gamma_process;  // 1: Gamma process error; 0: none
}
parameters {
  real c;
  vector[n_origin - 1] a_raw;
  vector[n_dev - 1] b_raw;
}
transformed parameters {
  vector[n_origin] a = append_row(0, a_raw);
  vector[n_dev] b = append_row(0, b_raw);
}
model {
  vector[N] eta = c + a[w] + b[d];
  target += (dot_product(x, eta) - sum(exp(eta))) / phi;
  c ~ normal(c0, intercept_sd);
  a_raw ~ normal(0, origin_sd);
  b_raw ~ normal(0, development_sd);
}
generated quantities {
  vector[n_origin] reserve = rep_vector(0, n_origin);
  real total;
  for (k in 1:N_fut) {
    real mu = exp(c + a[fw[k]] + b[fd[k]]);
    // Gamma with mean mu and variance phi mu: shape mu / phi, rate 1 / phi.
    reserve[fw[k]] += gamma_process == 1 ? gamma_rng(mu / phi, 1 / phi) : mu;
  }
  total = sum(reserve);
}
