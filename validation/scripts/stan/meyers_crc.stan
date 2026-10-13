// Meyers' correlated chain ladder on cumulative values (CRC), the reference
// for prospicio_reserving::bayes::Meyers with MeyersModel::Crc
// (docs/design/reserving-bayes.md, decisions 7 and 10). Run by
// validation/scripts/reserving_meyers_stan.py.
//
// Provenance: a transcription of the Stan program held as the string
// CRCmodel_stan in Appendix/CRC.R (lines 149-191; file sha256
// 8af844b639bc5944...), in meyers-appendix.zip (sha256
// 480e99a63fd4041ee546058412cb398f40071aa27e52e51cf671d5aaafec0f9a),
// https://www.casact.org/sites/default/files/2021-04/meyers-appendix.zip,
// the code of Glenn Meyers, "Stochastic Loss Reserving Using Bayesian MCMC
// Models", 2nd ed., CAS Monograph 8 (2019). Neither the zip, its Read Me nor
// the monograph's page states a licence, so this file is a transcription
// with attribution, not a copy. The model, priors and parametrisation are
// Meyers'; the changes are syntax and generality only:
//
// * array syntax: `real x[10]` becomes `array[10] real x` (the old syntax
//   was removed in Stan 2.33 and does not compile on CmdStan 2.39);
// * `gamma_cdf(y, 1, 1)` becomes `gamma_cdf(y | 1, 1)` (the same function);
// * the triangle size 10 becomes data n (origins and ages, a square
//   triangle), so 9 becomes n - 1 and the bound upper=10 becomes upper=n;
// * generated quantities add sigma (Meyers' sig, kept under a clearer
//   name) and the predictive ultimate by origin, which Meyers draws in R
//   (CRC.R lines 293-299): log-normal at age n with mean
//   log(premium) + logelr + alpha[w] (beta[n] = 0) and sd sig[n], without
//   his ceiling(); an origin observed at age n keeps its observed value
//   (Meyers' at.wd10[, 1] = loss[55]). The reserve is the ultimate less
//   the latest observed value.
//
// Note that a_ig[i] ~ inv_gamma(1, 1) maps to gamma_cdf(1 / a_ig[i] | 1, 1)
// = 1 - exp(-1 / a_ig[i]), which is exactly uniform on (0, 1); and that
// sig2[n] takes a_ig[n] while sig2[n - i] adds a_ig[i], a labelling that is
// the reverse of the monograph's (the increments are iid, so the posterior
// of sig is the same).
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
}
transformed parameters {
  array[n] real alpha;
  array[n] real beta;
  array[n] real sig2;
  array[n] real sig;
  array[len_data] real mu;
  alpha[1] = 0;
  for (i in 2:n) alpha[i] = r_alpha[i - 1];
  for (i in 1:(n - 1)) beta[i] = r_beta[i];
  beta[n] = 0;
  sig2[n] = gamma_cdf(1 / a_ig[n] | 1, 1);
  for (i in 1:(n - 1)) sig2[n - i] = sig2[n + 1 - i] + gamma_cdf(1 / a_ig[i] | 1, 1);
  for (i in 1:n) sig[i] = sqrt(sig2[i]);
  for (i in 1:len_data) {
    mu[i] = logprem[i] + logelr + alpha[w[i]] + beta[d[i]];
  }
}
model {
  r_alpha ~ normal(0, 3.162);
  r_beta ~ normal(0, 3.162);
  for (i in 1:n) a_ig[i] ~ inv_gamma(1, 1);
  logelr ~ normal(-.4, 3.162);
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
