// The eight schools model, centred, so CmdStan reports some divergent
// transitions; log_lik for PSIS-LOO. validation/scripts/stan_csv_fixture.sh
// runs it.
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta;
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
generated quantities {
  vector[J] log_lik;
  for (j in 1:J) log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
}
