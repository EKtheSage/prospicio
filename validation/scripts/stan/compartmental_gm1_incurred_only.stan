// Gesmann and Morris' model 1 fitted to INCURRED amounts alone, with
// ordered rates: the reference for prospicio_reserving::bayes::Compartmental
// with `fitted: Fitted::IncurredOnly` and `rate_order: RateOrder::Ordered`
// (docs/design/reserving-compartmental.md, decision 12). Run by
// validation/scripts/reserving_compartmental_one_response_stan.py.
//
// The reconstruction of compartmental_gm1.stan (see its header for the
// provenance and brms' conventions) with three changes:
//
// * Only the incurred cells are observed, with mean premium times
//   RLR (OS(t) + RRF PD(t)) as in compartmental_gm1_incurred.stan, so there
//   is one sigma (the incurred's).
// * The rates are ordered as in compartmental_gm1_paid_only.stan.
// * The paid column gives only each origin's latest paid. A reserve is the
//   mean paid at the last age given the draw less the latest paid, with no
//   noise: no paid cell is fitted, so the model has no paid sigma.
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
  int<lower=1> N;                          // observed incurred cells
  vector[N] Y;                             // amounts
  vector[N] t;                             // development time in years
  int<lower=1> n_origin;                   // every origin of the square
  vector<lower=0>[n_origin] premium;
  int<lower=1, upper=n_origin> N_1;        // origins with a training cell
  array[n_origin] int<lower=0, upper=N_1> level;  // its level, 0 if none
  array[N] int<lower=1, upper=n_origin> origin;
  real<lower=0> t_last;                    // the square's last age, in years
  vector[n_origin] latest_paid;            // 0 for an origin with no cell
  int<lower=0> N_test;                     // the test diagonal's incurred cells
  array[N_test] int<lower=1, upper=n_origin> test_origin;
  vector[N_test] test_t;
}
parameters {
  real b_oRLR;
  real b_oRRF;
  real delta;                              // log(k_er - k_p)
  real b_okp;
  vector<lower=0>[2] sd_1;
  matrix[2, N_1] z_1;
  cholesky_factor_corr[2] L_1;
  real b_sigma;                            // log sigma of incurred
}
transformed parameters {
  matrix[N_1, 2] r_1 = transpose(diag_pre_multiply(sd_1, L_1) * z_1);
  real kp = 1 * exp(b_okp * 0.1);
  real ker = kp + exp(delta);
  real b_oker = log(ker / 3) / 0.1;
}
model {
  vector[N] mu;
  for (n in 1:N) {
    int j = level[origin[n]];
    real RLR = 0.7 * exp((b_oRLR + r_1[j, 1]) * 0.2);
    real RRF = 0.8 * exp((b_oRRF + r_1[j, 2]) * 0.1);
    mu[n] = premium[origin[n]] * incurred(t[n], ker, kp, RLR, RRF);
  }
  target += normal_lpdf(Y | mu, exp(b_sigma));
  target += normal_lpdf(b_oRLR | 0, 1);
  target += normal_lpdf(b_oRRF | 0, 1);
  target += normal_lpdf(b_oker | 0, 1);
  target += delta - log(ker);              // the Jacobian of delta -> b_oker
  target += normal_lpdf(b_okp | 0, 1);
  target += student_t_lpdf(sd_1[1] | 10, 0, 0.2);
  target += student_t_lpdf(sd_1[2] | 10, 0, 0.1);
  target += std_normal_lpdf(to_vector(z_1));
  target += lkj_corr_cholesky_lpdf(L_1 | 1);
  target += student_t_lpdf(b_sigma | 1, 0, 1000);
}
generated quantities {
  real k_er = ker;
  real k_p = kp;
  real b_ker = b_oker;
  real RLR_pop = 0.7 * exp(b_oRLR * 0.2);
  real RRF_pop = 0.8 * exp(b_oRRF * 0.1);
  real rho = multiply_lower_tri_self_transpose(L_1)[1, 2];
  real sigma_os = exp(b_sigma);
  vector[n_origin] RLR;
  vector[n_origin] RRF;
  vector[n_origin] reserve;
  real total_reserve;
  vector[n_origin] ulr;
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
      reserve[i] = premium[i] * paid(t_last, k_er, k_p, RLR[i], RRF[i]) - latest_paid[i];
    }
    total_reserve = sum(reserve);
    for (n in 1:N_test) {
      int i = test_origin[n];
      test_mean[n] = premium[i] * incurred(test_t[n], k_er, k_p, RLR[i], RRF[i]);
    }
    for (n in 1:N) {
      int i = origin[n];
      log_lik[n] = normal_lpdf(Y[n] | premium[i] * incurred(t[n], k_er, k_p, RLR[i], RRF[i]),
                               sigma_os);
    }
  }
}
