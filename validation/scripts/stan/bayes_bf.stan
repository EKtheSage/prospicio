// The Bayesian Bornhuetter-Ferguson of docs/design/reserving-bayes.md
// (decision 9), England, Verrall and Wuthrich (2012), Model 3.1 on the log
// scale: the reference for prospicio_reserving::bayes::BayesBf. Run by
// validation/scripts/reserving_bayes_bf_stan.py.
//
// eta[w, d] = alpha[w] + beta[d] on the observed incremental cells, with the
// scaled Poisson quasi log likelihood (x eta - exp(eta)) / phi; mu = exp(alpha)
// has a gamma prior with mean m[w] and shape 1 / cv^2 (none when shape is 0,
// a flat prior on alpha), and beta a normal prior. phi (the chain ladder's
// Pearson scale) and m (apriori times exposure) are data.
data {
  int<lower=1> N;                       // observed cells
  int<lower=2> n_origin;
  int<lower=2> n_dev;
  array[N] int<lower=1, upper=n_origin> w;
  array[N] int<lower=1, upper=n_dev> d;
  vector[N] x;                          // incremental values
  real<lower=0> phi;
  vector<lower=0>[n_origin] m;          // prior mean of mu
  real<lower=0> shape;                  // 1 / cv^2; 0 is flat on alpha
  real<lower=0> development_sd;
  int<lower=0> N_fut;                   // future cells up to the oldest age
  array[N_fut] int<lower=1, upper=n_origin> fw;
  array[N_fut] int<lower=1, upper=n_dev> fd;
  int<lower=0, upper=1> gamma_process;  // 1: Gamma process error; 0: none
}
parameters {
  vector[n_origin] alpha;
  vector[n_dev] beta;
}
model {
  vector[N] eta = alpha[w] + beta[d];
  target += (dot_product(x, eta) - sum(exp(eta))) / phi;
  if (shape > 0) {
    // mu = exp(alpha) ~ gamma(shape, shape / m), with the Jacobian of
    // the log.
    target += gamma_lpdf(exp(alpha) | shape, shape ./ m) + sum(alpha);
  }
  beta ~ normal(0, development_sd);
}
generated quantities {
  vector[n_origin] reserve = rep_vector(0, n_origin);
  real total;
  for (k in 1:N_fut) {
    real mu = exp(alpha[fw[k]] + beta[fd[k]]);
    // Gamma with mean mu and variance phi mu: shape mu / phi, rate 1 / phi.
    reserve[fw[k]] += gamma_process == 1 ? gamma_rng(mu / phi, 1 / phi) : mu;
  }
  total = sum(reserve);
}
