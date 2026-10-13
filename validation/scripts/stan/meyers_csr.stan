// Meyers' changing settlement rate model (CSR), the reference for
// prospicio_reserving::bayes::Meyers with MeyersModel::Csr
// (docs/design/reserving-bayes.md, decisions 7 and 10). Run by
// validation/scripts/reserving_meyers_stan.py.
//
// Provenance: a transcription of the Stan program held as the string
// CSRmodel_stan in Appendix/CSR.R (lines 157-204; file sha256
// 53d957ba1a0a851b...), in meyers-appendix.zip (sha256
// 480e99a63fd4041ee546058412cb398f40071aa27e52e51cf671d5aaafec0f9a),
// https://www.casact.org/sites/default/files/2021-04/meyers-appendix.zip,
// the code of Glenn Meyers, "Stochastic Loss Reserving Using Bayesian MCMC
// Models", 2nd ed., CAS Monograph 8 (2019). Neither the zip, its Read Me nor
// the monograph's page states a licence, so this file is a transcription
// with attribution, not a copy. The model, priors and parametrisation are
// Meyers'; the changes are those of meyers_crc.stan (array syntax,
// gamma_cdf's bar, data n for 10, sigma and the predictive ultimate in
// generated quantities, no ceiling()). CSR is CRC with beta[d] multiplied
// by speedup[w] = (1 - gamma)^(w - 1), built by repeated multiplication as
// Meyers builds it, and gamma ~ normal(0, 0.05). beta[n] = 0, so the
// ultimate at age n does not involve gamma (CSR.R lines 300-306 draw it as
// for CRC).
data {
  int<lower=2> n;                         // origins = development ages
  int<lower=1> len_data;
  array[len_data] real logprem;
  array[len_data] real logloss;
  array[len_data] int<lower=1, upper=n> w;
  array[len_data] int<lower=1, upper=n> d;
  vector[n] premium;                      // premium by origin
  vector[n] latest;                       // latest observed value by origin
  array[n] int<lower=0, upper=1> closed;  // 1: observed at age n
}
parameters {
  array[n - 1] real r_alpha;
  array[n - 1] real r_beta;
  real logelr;
  array[n] real<lower=0, upper=100000> a_ig;
  real gamma;
}
transformed parameters {
  array[n] real alpha;
  array[n] real beta;
  array[n] real speedup;
  array[n] real sig2;
  array[n] real sig;
  array[len_data] real mu;
  alpha[1] = 0;
  for (i in 2:n) alpha[i] = r_alpha[i - 1];
  for (i in 1:(n - 1)) beta[i] = r_beta[i];
  beta[n] = 0;
  speedup[1] = 1;
  for (i in 2:n) speedup[i] = speedup[i - 1] * (1 - gamma);
  sig2[n] = gamma_cdf(1 / a_ig[n] | 1, 1);
  for (i in 1:(n - 1)) sig2[n - i] = sig2[n + 1 - i] + gamma_cdf(1 / a_ig[i] | 1, 1);
  for (i in 1:n) sig[i] = sqrt(sig2[i]);
  for (i in 1:len_data) {
    mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]] * speedup[w[i]];
  }
}
model {
  r_alpha ~ normal(0, 3.162);
  r_beta ~ normal(0, 3.162);
  for (i in 1:n) a_ig[i] ~ inv_gamma(1, 1);
  logelr ~ normal(-.4, 3.162);
  gamma ~ normal(0, 0.05);
  for (i in 1:len_data) logloss[i] ~ normal(mu[i], sig[d[i]]);
}
generated quantities {
  vector[len_data] log_lik;
  vector[n] sigma = to_vector(sig);
  vector[n] ultimate;
  vector[n] reserve;
  real total_ultimate;
  real total_reserve;
  for (i in 1:len_data) log_lik[i] = normal_lpdf(logloss[i] | mu[i], sig[d[i]]);
  for (k in 1:n) {
    if (closed[k] == 1) {
      ultimate[k] = latest[k];
    } else {
      ultimate[k] = lognormal_rng(log(premium[k]) + logelr + alpha[k], sig[n]);
    }
  }
  reserve = ultimate - latest;
  total_ultimate = sum(ultimate);
  total_reserve = sum(reserve);
}
