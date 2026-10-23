// Gesmann and Morris' model 1 fitted to paid and INCURRED amounts instead
// of paid and outstanding: the reference for prospicio_reserving::bayes::
// Compartmental with `fitted: Fitted::PaidAndIncurred`
// (docs/design/reserving-compartmental.md, decision 12). Run by
// validation/scripts/reserving_compartmental_incurred_stan.py.
//
// The same reconstruction as compartmental_gm1.stan (see its header for the
// provenance and brms' conventions), with one change: the second response
// (delta = 0) is the incurred amount, whose mean is the premium times
// RLR (OS(t) + RRF PD(t)), the outstanding closed form plus the cumulative
// paid closed form. Priors, parameters and the generated quantities are
// the same; `test_mean` and `log_lik` of a delta = 0 cell are the incurred
// mean and its density.
functions {
  real outstanding(real t, real ker, real kp, real RLR) {
    return RLR * ker / (ker - kp) * (exp(-kp * t) - exp(-ker * t));
  }
  real paid(real t, real ker, real kp, real RLR, real RRF) {
    return RLR * RRF / (ker - kp) * (ker * (1 - exp(-kp * t)) - kp * (1 - exp(-ker * t)));
  }
  real incurred(real t, real ker, real kp, real RLR, real RRF) {
    return outstanding(t, ker, kp, RLR) + paid(t, ker, kp, RLR, RRF);
  }
}
data {
  int<lower=1> N;                          // observed cells, incurred and paid
  vector[N] Y;                             // amounts
  vector[N] t;                             // development time in years
  array[N] int<lower=0, upper=1> delta;    // 0: incurred, 1: paid
  int<lower=1> n_origin;                   // every origin of the square
  vector<lower=0>[n_origin] premium;
  int<lower=1, upper=n_origin> N_1;        // origins with a training cell
  array[n_origin] int<lower=0, upper=N_1> level;  // its level, 0 if none
  array[N] int<lower=1, upper=n_origin> origin;
  real<lower=0> t_last;                    // the square's last age, in years
  vector[n_origin] latest_paid;            // 0 for an origin with no cell
  int<lower=0> N_test;                     // the test diagonal's cells
  array[N_test] int<lower=1, upper=n_origin> test_origin;
  vector[N_test] test_t;
  array[N_test] int<lower=0, upper=1> test_delta;
}
parameters {
  real b_oRLR;
  real b_oRRF;
  real b_oker;
  real b_okp;
  vector<lower=0>[2] sd_1;
  matrix[2, N_1] z_1;
  cholesky_factor_corr[2] L_1;
  vector[2] b_sigma;                       // log sigma: outstanding, paid
}
transformed parameters {
  // brms' scale_r_cor: r_1 = (diag(sd_1) L_1 z_1)'.
  matrix[N_1, 2] r_1 = transpose(diag_pre_multiply(sd_1, L_1) * z_1);
}
model {
  real ker = 3 * exp(b_oker * 0.1);
  real kp = 1 * exp(b_okp * 0.1);
  vector[N] mu;
  vector[N] sigma;
  for (n in 1:N) {
    int j = level[origin[n]];
    real RLR = 0.7 * exp((b_oRLR + r_1[j, 1]) * 0.2);
    real RRF = 0.8 * exp((b_oRRF + r_1[j, 2]) * 0.1);
    if (delta[n] == 1) {
      mu[n] = premium[origin[n]] * paid(t[n], ker, kp, RLR, RRF);
    } else {
      mu[n] = premium[origin[n]] * incurred(t[n], ker, kp, RLR, RRF);
    }
    sigma[n] = exp(b_sigma[delta[n] + 1]);
  }
  target += normal_lpdf(Y | mu, sigma);
  target += normal_lpdf(b_oRLR | 0, 1);
  target += normal_lpdf(b_oRRF | 0, 1);
  target += normal_lpdf(b_oker | 0, 1);
  target += normal_lpdf(b_okp | 0, 1);
  target += student_t_lpdf(sd_1[1] | 10, 0, 0.2);
  target += student_t_lpdf(sd_1[2] | 10, 0, 0.1);
  target += std_normal_lpdf(to_vector(z_1));
  target += lkj_corr_cholesky_lpdf(L_1 | 1);
  target += student_t_lpdf(b_sigma | 1, 0, 1000);
}
generated quantities {
  real k_er = 3 * exp(b_oker * 0.1);
  real k_p = 1 * exp(b_okp * 0.1);
  real RLR_pop = 0.7 * exp(b_oRLR * 0.2);
  real RRF_pop = 0.8 * exp(b_oRRF * 0.1);
  real rho = multiply_lower_tri_self_transpose(L_1)[1, 2];
  real sigma_os = exp(b_sigma[1]);
  real sigma_pd = exp(b_sigma[2]);
  vector[n_origin] RLR;
  vector[n_origin] RRF;
  vector[n_origin] reserve;
  real total_reserve;
  vector[n_origin] ulr;                    // RLR RRF, the model's ultimate loss ratio
  vector[N_test] test_mean;
  vector[N] log_lik;
  {
    for (i in 1:n_origin) {
      vector[2] r;
      if (level[i] > 0) {
        r = to_vector(r_1[level[i]]);
      } else {
        r = diag_pre_multiply(sd_1, L_1) * to_vector(normal_rng(rep_vector(0, 2), 1));
      }
      RLR[i] = 0.7 * exp((b_oRLR + r[1]) * 0.2);
      RRF[i] = 0.8 * exp((b_oRRF + r[2]) * 0.1);
      ulr[i] = RLR[i] * RRF[i];
      reserve[i] = normal_rng(premium[i] * paid(t_last, k_er, k_p, RLR[i], RRF[i]), sigma_pd)
                   - latest_paid[i];
    }
    total_reserve = sum(reserve);
    for (n in 1:N_test) {
      int i = test_origin[n];
      if (test_delta[n] == 1) {
        test_mean[n] = premium[i] * paid(test_t[n], k_er, k_p, RLR[i], RRF[i]);
      } else {
        test_mean[n] = premium[i] * incurred(test_t[n], k_er, k_p, RLR[i], RRF[i]);
      }
    }
    for (n in 1:N) {
      int i = origin[n];
      real m;
      if (delta[n] == 1) {
        m = premium[i] * paid(t[n], k_er, k_p, RLR[i], RRF[i]);
      } else {
        m = premium[i] * incurred(t[n], k_er, k_p, RLR[i], RRF[i]);
      }
      log_lik[n] = normal_lpdf(Y[n] | m, exp(b_sigma[delta[n] + 1]));
    }
  }
}
