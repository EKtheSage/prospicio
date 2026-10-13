// Meyers' correlated accident year model (CAY), the reference for
// prospicio_reserving::bayes::Meyers with MeyersModel::Cay
// (docs/design/reserving-bayes.md, decisions 7 and 10). Run by
// validation/scripts/reserving_meyers_cay_stan.py.
//
// Provenance: a transcription of the Stan program held as the string
// CAYModel_stan in Appendix/CAY.R (lines 160-211; file sha256
// ec04de319667a031...), in meyers-appendix.zip (sha256
// 480e99a63fd4041ee546058412cb398f40071aa27e52e51cf671d5aaafec0f9a),
// https://www.casact.org/sites/default/files/2021-04/meyers-appendix.zip,
// the code of Glenn Meyers, "Stochastic Loss Reserving Using Bayesian MCMC
// Models", 2nd ed., CAS Monograph 8 (2019), sec. 8. Neither the zip, its
// Read Me nor the monograph's page states a licence, so this file is a
// transcription with attribution, not a copy. The model, priors and
// parametrisation are Meyers'; the changes are those of meyers_crc.stan
// (array syntax, gamma_cdf's bar, data n for 10, sigma and the predictive
// ultimate in generated quantities, no ceiling()), and `real<lower=-1,
// upper=1> rho` in transformed parameters becomes a plain real (the bound
// only checks; r_rho in (0, 1) already keeps rho in (-1, 1)).
//
// The cells must be sorted by (age, origin), as Meyers sorts them
// (data.out[order(d, w)]): then cell i - 1 is the previous origin at the
// same age whenever wne1[i] = 1 (origin w > 1), and
// mu[i] = base[i] + rho (logloss[i - 1] - mu[i - 1]). The predictive
// ultimate follows CAY.R lines 317-328: down age n origin by origin, each
// mean takes rho times the previous origin's residual there, observed for
// a closed origin and simulated otherwise.
data {
  int<lower=2> n;                         // origins = development ages
  int<lower=1> len_data;
  array[len_data] int<lower=0, upper=1> wne1;
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
  real<lower=0, upper=1> r_rho;
}
transformed parameters {
  array[n] real alpha;
  array[n] real beta;
  array[n] real sig2;
  array[n] real sig;
  array[len_data] real mu;
  real rho;
  for (i in 1:(n - 1)) beta[i] = r_beta[i];
  beta[n] = 0;
  alpha[1] = 0;
  for (i in 2:n) alpha[i] = r_alpha[i - 1];
  sig2[n] = gamma_cdf(1 / a_ig[n] | 1, 1);
  for (i in 1:(n - 1)) sig2[n - i] = sig2[n + 1 - i] + gamma_cdf(1 / a_ig[i] | 1, 1);
  for (i in 1:n) sig[i] = sqrt(sig2[i]);
  mu[1] = logprem[1] + logelr + beta[d[1]];
  rho = -2 * r_rho + 1;
  for (i in 2:len_data) {
    mu[i] = logprem[i] + alpha[w[i]] + logelr + beta[d[i]]
            + rho * (logloss[i - 1] - mu[i - 1]) * wne1[i];
  }
}
model {
  logelr ~ normal(-0.4, 3.162);
  r_alpha ~ normal(0, 3.162);
  r_beta ~ normal(0, 3.162);
  a_ig ~ inv_gamma(1, 1);
  r_rho ~ beta(2, 2);
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
  {
    real resid = 0;
    for (k in 1:n) {
      real m = log(premium[k]) + logelr + alpha[k] + rho * resid;
      if (closed[k] == 1) {
        ultimate[k] = latest[k];
      } else {
        ultimate[k] = lognormal_rng(m, sig[n]);
      }
      resid = log(ultimate[k]) - m;
    }
  }
  reserve = ultimate - latest;
  total_ultimate = sum(ultimate);
  total_reserve = sum(reserve);
}
