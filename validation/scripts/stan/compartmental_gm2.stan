// Gesmann and Morris' model 2, the one-OS compartmental model with a
// log-normal likelihood on outstanding and incremental paid loss ratios and
// random effects by accident year and by development year on all four
// parameters, the reference for prospicio_reserving::bayes::Compartmental
// with CompartmentalPriors::gesmann_morris_2() and the log-normal, loss-ratio
// and incremental settings (docs/design/reserving-compartmental.md,
// decisions 2 and 7). Run by
// validation/scripts/reserving_compartmental_gm2_stan.py.
//
// Provenance: a reconstruction, written for this repository, of the Stan
// program that brms generates from the R code of Gesmann and Morris (2020),
// "Hierarchical Compartmental Reserving Models", CAS Research Paper, sec.
// 7.2.3 (the brms formula frml2 and priors mypriors2 of model 2, fitted
// there with brms 2.10.0 on rstan 2.19.2). The paper prints R code with the
// closed form as Stan function snippets, not a Stan program, and states no
// licence, so nothing here is copied: the model is theirs, the text is ours.
// brms is not installed; the reconstruction reproduces its conventions for
// that code (the same as model 1's, compartmental_gm1.stan):
//
// * nl = TRUE: each nonlinear parameter oRLR, oRRF, oker, okp has its own
//   intercept b_* with a normal(0, 1) prior and no centring;
// * RLR = 0.7 exp(0.2 oRLR), RRF = 0.8 exp(0.1 oRRF), ker = 3 exp(0.1 oker),
//   kp = exp(0.1 okp), each o its intercept plus its group effects;
// * oRLR ~ 1 + (1 | ID | accident_year) + (1 | dev_year), and the same for
//   oRRF: one correlated term over the accident years (r_1, non-centred as
//   brms writes it, r = (diag(sd) L z)' with L ~ lkj_corr_cholesky(1)) and
//   an uncorrelated development-year term each (r_2, r_3);
//   oker ~ 1 + (1 | accident_year) + (1 | dev_year) and the same for okp:
//   uncorrelated terms (r_4 to r_7), r = sd z; brms' levels are the
//   observed ones (9 accident years, 9 development years);
// * the sd priors are half Student-t (class sd, bounded at 0), one per
//   nlpar and shared by its two groups: student_t(10, 0, 0.7) on oRLR's,
//   student_t(10, 0, 0.5) on oRRF's, student_t(10, 0, 0.3) on oker's and
//   okp's;
// * family lognormal with sigma ~ 0 + deltaf on the log link: the
//   normal(log(0.2), 0.2) priors (class b) are on log sigma, with no
//   Jacobian;
// * the response is the outstanding loss ratio or the incremental paid loss
//   ratio, each to the accident year's direct premium, and the median of
//   each cell is the paper's claimsprocess(dev_year, 1.0, ...): the
//   outstanding closed form, or the paid closed form at t less the same at
//   t - 1 with the cell's own parameters (t > 1), as its function snippet
//   writes it, with the division by ker - kp;
// * brms' normalising constants of the truncated sd priors are dropped
//   (they do not change the posterior).
//
// Generated quantities add, for each draw, what the paper computes with
// brms' predict(): an accident year with no training cell (1997) gets new
// effects from the population, and so does a development year with none
// (10), r = diag(sd) L z or sd z with z ~ N(0, 1) (the design note's
// decision 2; brms' default sample_new_levels instead reuses a random
// existing level's effect); the reserve of each origin is the sum of
// log-normal draws of its future incremental paid amounts to the square's
// last age; and the expected amounts of the test diagonal (outstanding and
// incremental paid) and the log likelihood of each cell (for PSIS-LOO).
functions {
  real os(real t, real ker, real kp, real RLR) {
    return RLR * ker / (ker - kp) * (exp(-kp * t) - exp(-ker * t));
  }
  real paid(real t, real ker, real kp, real RLR, real RRF) {
    return RLR * RRF / (ker - kp) * (ker * (1 - exp(-kp * t)) - kp * (1 - exp(-ker * t)));
  }
  // The paper's claimsprocess with devfreq 1: outstanding (delta 0) or
  // incremental paid (delta 1).
  real claimsprocess(real t, real ker, real kp, real RLR, real RRF, int delta) {
    if (delta == 0) {
      return os(t, ker, kp, RLR);
    }
    real out = paid(t, ker, kp, RLR, RRF);
    if (t > 1) {
      out -= paid(t - 1, ker, kp, RLR, RRF);
    }
    return out;
  }
}
data {
  int<lower=1> N;                          // observed cells, outstanding and paid
  vector<lower=0>[N] Y;                    // loss ratios
  vector[N] t;                             // development time in years
  array[N] int<lower=0, upper=1> delta;    // 0: outstanding, 1: incremental paid
  int<lower=1> n_origin;                   // every origin of the square
  int<lower=1> n_age;                      // every age of the square
  vector<lower=0>[n_origin] premium;
  int<lower=1, upper=n_origin> N_1;        // accident years with a training cell
  array[n_origin] int<lower=0, upper=N_1> level;  // its level, 0 if none
  int<lower=1, upper=n_age> N_2;           // development years with a training cell
  array[n_age] int<lower=0, upper=N_2> dev_level;  // its level, 0 if none
  array[N] int<lower=1, upper=n_origin> origin;
  array[N] int<lower=1, upper=n_age> age;
  array[n_origin] int<lower=0, upper=n_age> latest_age;  // last observed paid age, 0 if none
  int<lower=0> N_test;                     // the test diagonal's cells
  array[N_test] int<lower=1, upper=n_origin> test_origin;
  array[N_test] int<lower=1, upper=n_age> test_age;
  array[N_test] int<lower=0, upper=1> test_delta;
}
parameters {
  real b_oRLR;
  real b_oRRF;
  real b_oker;
  real b_okp;
  vector<lower=0>[2] sd_1;                 // accident year: oRLR, oRRF (correlated)
  matrix[2, N_1] z_1;
  cholesky_factor_corr[2] L_1;
  real<lower=0> sd_2;                      // development year: oRLR
  vector[N_2] z_2;
  real<lower=0> sd_3;                      // development year: oRRF
  vector[N_2] z_3;
  real<lower=0> sd_4;                      // accident year: oker
  vector[N_1] z_4;
  real<lower=0> sd_5;                      // development year: oker
  vector[N_2] z_5;
  real<lower=0> sd_6;                      // accident year: okp
  vector[N_1] z_6;
  real<lower=0> sd_7;                      // development year: okp
  vector[N_2] z_7;
  vector[2] b_sigma;                       // log sigma: outstanding, paid
}
transformed parameters {
  matrix[N_1, 2] r_1 = transpose(diag_pre_multiply(sd_1, L_1) * z_1);
  vector[N_2] r_2 = sd_2 * z_2;
  vector[N_2] r_3 = sd_3 * z_3;
  vector[N_1] r_4 = sd_4 * z_4;
  vector[N_2] r_5 = sd_5 * z_5;
  vector[N_1] r_6 = sd_6 * z_6;
  vector[N_2] r_7 = sd_7 * z_7;
}
model {
  vector[N] mu;
  vector[N] sigma;
  for (n in 1:N) {
    int j = level[origin[n]];
    int d = dev_level[age[n]];
    real ker = 3 * exp((b_oker + r_4[j] + r_5[d]) * 0.1);
    real kp = 1 * exp((b_okp + r_6[j] + r_7[d]) * 0.1);
    real RLR = 0.7 * exp((b_oRLR + r_1[j, 1] + r_2[d]) * 0.2);
    real RRF = 0.8 * exp((b_oRRF + r_1[j, 2] + r_3[d]) * 0.1);
    mu[n] = log(claimsprocess(t[n], ker, kp, RLR, RRF, delta[n]));
    sigma[n] = exp(b_sigma[delta[n] + 1]);
  }
  target += lognormal_lpdf(Y | mu, sigma);
  target += normal_lpdf(b_oRLR | 0, 1);
  target += normal_lpdf(b_oRRF | 0, 1);
  target += normal_lpdf(b_oker | 0, 1);
  target += normal_lpdf(b_okp | 0, 1);
  target += student_t_lpdf(sd_1[1] | 10, 0, 0.7);
  target += student_t_lpdf(sd_1[2] | 10, 0, 0.5);
  target += student_t_lpdf(sd_2 | 10, 0, 0.7);
  target += student_t_lpdf(sd_3 | 10, 0, 0.5);
  target += student_t_lpdf(sd_4 | 10, 0, 0.3);
  target += student_t_lpdf(sd_5 | 10, 0, 0.3);
  target += student_t_lpdf(sd_6 | 10, 0, 0.3);
  target += student_t_lpdf(sd_7 | 10, 0, 0.3);
  target += std_normal_lpdf(to_vector(z_1));
  target += std_normal_lpdf(z_2);
  target += std_normal_lpdf(z_3);
  target += std_normal_lpdf(z_4);
  target += std_normal_lpdf(z_5);
  target += std_normal_lpdf(z_6);
  target += std_normal_lpdf(z_7);
  target += lkj_corr_cholesky_lpdf(L_1 | 1);
  target += normal_lpdf(b_sigma | log(0.2), 0.2);
}
generated quantities {
  real k_er = 3 * exp(b_oker * 0.1);
  real k_p = 1 * exp(b_okp * 0.1);
  real RLR_pop = 0.7 * exp(b_oRLR * 0.2);
  real RRF_pop = 0.8 * exp(b_oRRF * 0.1);
  real rho = multiply_lower_tri_self_transpose(L_1)[1, 2];
  real sigma_os = exp(b_sigma[1]);
  real sigma_pd = exp(b_sigma[2]);
  // Each origin's values at a zero development effect.
  vector[n_origin] RLR;
  vector[n_origin] RRF;
  vector[n_origin] ker_o;
  vector[n_origin] kp_o;
  vector[n_origin] ulr;                    // RLR RRF at a zero development effect
  vector[n_origin] reserve;
  real total_reserve;
  vector[N_test] test_mean;
  vector[N] log_lik;
  {
    // Accident-year effects (oRLR, oRRF, oker, okp) of every origin and
    // development-year effects of every age, new ones from the population
    // for a level with no training cell.
    matrix[n_origin, 4] ay;
    matrix[n_age, 4] dy;
    for (i in 1:n_origin) {
      if (level[i] > 0) {
        ay[i, 1] = r_1[level[i], 1];
        ay[i, 2] = r_1[level[i], 2];
        ay[i, 3] = r_4[level[i]];
        ay[i, 4] = r_6[level[i]];
      } else {
        vector[2] r = diag_pre_multiply(sd_1, L_1) * to_vector(normal_rng(rep_vector(0, 2), 1));
        ay[i, 1] = r[1];
        ay[i, 2] = r[2];
        ay[i, 3] = sd_4 * normal_rng(0, 1);
        ay[i, 4] = sd_6 * normal_rng(0, 1);
      }
    }
    for (k in 1:n_age) {
      if (dev_level[k] > 0) {
        dy[k, 1] = r_2[dev_level[k]];
        dy[k, 2] = r_3[dev_level[k]];
        dy[k, 3] = r_5[dev_level[k]];
        dy[k, 4] = r_7[dev_level[k]];
      } else {
        dy[k, 1] = sd_2 * normal_rng(0, 1);
        dy[k, 2] = sd_3 * normal_rng(0, 1);
        dy[k, 3] = sd_5 * normal_rng(0, 1);
        dy[k, 4] = sd_7 * normal_rng(0, 1);
      }
    }
    for (i in 1:n_origin) {
      RLR[i] = 0.7 * exp((b_oRLR + ay[i, 1]) * 0.2);
      RRF[i] = 0.8 * exp((b_oRRF + ay[i, 2]) * 0.1);
      ker_o[i] = 3 * exp((b_oker + ay[i, 3]) * 0.1);
      kp_o[i] = 1 * exp((b_okp + ay[i, 4]) * 0.1);
      ulr[i] = RLR[i] * RRF[i];
      reserve[i] = 0;
      for (k in (latest_age[i] + 1):n_age) {
        real ker = 3 * exp((b_oker + ay[i, 3] + dy[k, 3]) * 0.1);
        real kp = 1 * exp((b_okp + ay[i, 4] + dy[k, 4]) * 0.1);
        real rlr = 0.7 * exp((b_oRLR + ay[i, 1] + dy[k, 1]) * 0.2);
        real rrf = 0.8 * exp((b_oRRF + ay[i, 2] + dy[k, 2]) * 0.1);
        reserve[i] += premium[i]
                      * lognormal_rng(log(claimsprocess(k, ker, kp, rlr, rrf, 1)), sigma_pd);
      }
    }
    total_reserve = sum(reserve);
    for (n in 1:N_test) {
      int i = test_origin[n];
      int k = test_age[n];
      real ker = 3 * exp((b_oker + ay[i, 3] + dy[k, 3]) * 0.1);
      real kp = 1 * exp((b_okp + ay[i, 4] + dy[k, 4]) * 0.1);
      real rlr = 0.7 * exp((b_oRLR + ay[i, 1] + dy[k, 1]) * 0.2);
      real rrf = 0.8 * exp((b_oRRF + ay[i, 2] + dy[k, 2]) * 0.1);
      test_mean[n] = premium[i] * claimsprocess(k, ker, kp, rlr, rrf, test_delta[n]);
    }
    for (n in 1:N) {
      int i = origin[n];
      int k = age[n];
      real ker = 3 * exp((b_oker + ay[i, 3] + dy[k, 3]) * 0.1);
      real kp = 1 * exp((b_okp + ay[i, 4] + dy[k, 4]) * 0.1);
      real rlr = 0.7 * exp((b_oRLR + ay[i, 1] + dy[k, 1]) * 0.2);
      real rrf = 0.8 * exp((b_oRRF + ay[i, 2] + dy[k, 2]) * 0.1);
      log_lik[n] = lognormal_lpdf(Y[n] | log(claimsprocess(t[n], ker, kp, rlr, rrf, delta[n])),
                                  exp(b_sigma[delta[n] + 1]));
    }
  }
}
