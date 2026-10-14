// Gesmann and Morris' model 1, the one-OS compartmental model with a normal
// likelihood on outstanding and cumulative paid amounts, the reference for
// prospicio_reserving::bayes::Compartmental with its defaults
// (docs/design/reserving-compartmental.md, decisions 2 and 7). Run by
// validation/scripts/reserving_compartmental_stan.py.
//
// Provenance: a reconstruction, written for this repository, of the Stan
// program that brms generates from the R code of Gesmann and Morris (2020),
// "Hierarchical Compartmental Reserving Models", CAS Research Paper, sec.
// 7.2.2 (the brms formula frml1 and priors of model 1, fitted there with
// brms 2.10.0 on rstan 2.19.2). The paper prints R code with the closed form
// as Stan function snippets, not a Stan program, and states no licence, so
// nothing here is copied: the model is theirs, the text is ours. brms is not
// installed; the reconstruction reproduces its conventions for that code:
//
// * nl = TRUE: each nonlinear parameter oRLR, oRRF, oker, okp has its own
//   intercept b_* with a normal(0, 1) prior and no centring;
// * RLR = 0.7 exp(0.2 oRLR), RRF = 0.8 exp(0.1 oRRF), ker = 3 exp(0.1 oker),
//   kp = exp(0.1 okp);
// * oRLR ~ 1 + (1 | ID | accident_year) and oRRF ~ 1 + (1 | ID | accident_year):
//   one correlated group term over the accident years with data (brms'
//   levels are the observed ones), non-centred as brms writes it,
//   r = (diag(sd) L z)' with z ~ std_normal() and L ~ lkj_corr_cholesky(1);
//   the sds are half Student-t (class sd, bounded at 0: student_t(10, 0, 0.2)
//   on sd(oRLR) and student_t(10, 0, 0.1) on sd(oRRF)), on the o scale;
// * sigma ~ 0 + deltaf with the log link: the sigma of each type is
//   exp(b_sigma), and the student_t(1, 0, 1000) priors (class b) are on
//   log sigma, so they add no Jacobian;
// * the mean of each cell is premium times the outstanding or paid closed
//   form at the cell's development year (the paper's Eq. 2, written as its
//   function snippets write it, with the division by ker - kp);
// * brms' normalising constants of the truncated sd priors are dropped
//   (they do not change the posterior).
//
// Generated quantities add, for each draw, what the paper computes with
// brms' predict(): an origin with no training cell (accident year 1997) gets
// a new effect from the population, r = diag(sd) L z with z ~ N(0, I) (the
// design note's decision 2; brms' default sample_new_levels instead reuses
// a random existing level's effect); the reserve of each origin is a normal
// draw of the cumulative paid at the square's last age less the latest
// observed paid (0 for an origin with no cell); and the predictive means of
// the test diagonal and the log likelihood of each cell (for PSIS-LOO).
functions {
  real outstanding(real t, real ker, real kp, real RLR) {
    return RLR * ker / (ker - kp) * (exp(-kp * t) - exp(-ker * t));
  }
  real paid(real t, real ker, real kp, real RLR, real RRF) {
    return RLR * RRF / (ker - kp) * (ker * (1 - exp(-kp * t)) - kp * (1 - exp(-ker * t)));
  }
}
data {
  int<lower=1> N;                          // observed cells, outstanding and paid
  vector[N] Y;                             // amounts
  vector[N] t;                             // development time in years
  array[N] int<lower=0, upper=1> delta;    // 0: outstanding, 1: paid
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
      mu[n] = premium[origin[n]] * outstanding(t[n], ker, kp, RLR);
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
        test_mean[n] = premium[i] * outstanding(test_t[n], k_er, k_p, RLR[i]);
      }
    }
    for (n in 1:N) {
      int i = origin[n];
      real m;
      if (delta[n] == 1) {
        m = premium[i] * paid(t[n], k_er, k_p, RLR[i], RRF[i]);
      } else {
        m = premium[i] * outstanding(t[n], k_er, k_p, RLR[i]);
      }
      log_lik[n] = normal_lpdf(Y[n] | m, exp(b_sigma[delta[n] + 1]));
    }
  }
}
