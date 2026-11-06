// The hierarchical Bornhuetter-Ferguson of docs/design/reserving-bayes.md
// (decision 30): the reference for prospicio_reserving::bayes::HierarchicalBf.
// Run by validation/scripts/reserving_hierarchical_bf_stan.py.
//
//   log L ~ Normal(log m_L - s^2 / 2, s)            s^2 = log(1 + c_L^2); flat when c_L = 0
//   tau ~ HalfNormal(tau_scale), omega ~ HalfNormal(omega_scale)
//   z_c ~ N(0, 1),  log s_c = tau z_c - tau^2 / 2
//   U_cw ~ Gamma(shape, mean L s_c P_cw),  alpha_cw = log U_cw
//   beta_d ~ N(0, development_sd^2), d = 2..m,  beta_1 = 0
//   delta_cd ~ N(0, 1),  beta_cd = beta_d + omega delta_cd
//   p_cd = exp(beta_cd) / sum_k exp(beta_ck)
//   X_cwd ~ quasi-ODP(mean U_cw p_cd, scale phi_c), phi_c data (a plug-in)
//
// The origins' gamma priors and the cells are the scaled Poisson quasi log
// likelihood of bayes_bf.stan, with the pattern normalised per program. The
// program effect is non-centred as in HierarchicalBf, so the two sample the
// same posterior; tau and omega are sampled on the natural scale here (Stan
// adds the log Jacobian) and on the log scale there.
data {
  int<lower=1> J;                       // programs
  int<lower=2> m;                       // development ages of the grid
  int<lower=1> N_o;                     // origins, all programs
  array[N_o] int<lower=1, upper=J> prog;
  vector<lower=0>[N_o] P;               // premium
  int<lower=1> N;                       // observed cells
  array[N] int<lower=1, upper=N_o> o;
  array[N] int<lower=1, upper=m> d;
  vector[N] x;                          // incremental values
  vector<lower=0>[J] phi;               // plugged-in scale of each program
  real<lower=0> apriori;                // m_L
  real<lower=0> level_cv;               // c_L; 0 is flat on log L
  real<lower=0> shape;                  // 1 / cv^2
  real<lower=0> development_sd;
  real<lower=0> tau_scale;
  real<lower=0> omega_scale;
  int<lower=0> N_fut;                   // future cells up to age m
  array[N_fut] int<lower=1, upper=N_o> fo;
  array[N_fut] int<lower=1, upper=m> fd;
}
parameters {
  real log_level;
  real<lower=0> tau;
  vector[J] z;
  vector[N_o] alpha;
  vector[m - 1] beta;                   // beta_2..beta_m
  real<lower=0> omega;
  matrix[J, m - 1] delta;
}
transformed parameters {
  vector[J] log_s = tau * z - 0.5 * tau ^ 2;
  matrix[J, m] log_share;
  for (c in 1:J) {
    row_vector[m] b = append_col(0, beta' + omega * delta[c]);
    log_share[c] = b - log_sum_exp(b);
  }
}
model {
  if (level_cv > 0) {
    real v = log1p(level_cv ^ 2);
    log_level ~ normal(log(apriori) - 0.5 * v, sqrt(v));
  }
  tau ~ normal(0, tau_scale);
  omega ~ normal(0, omega_scale);
  z ~ std_normal();
  beta ~ normal(0, development_sd);
  to_vector(delta) ~ std_normal();
  {
    // The gamma on U with the Jacobian of the log, less its value at the
    // mean, in v = alpha - log(L s P).
    vector[N_o] v = alpha - log(P) - log_level - log_s[prog];
    target += shape * sum(v - exp(v));
  }
  for (k in 1:N) {
    real eta = alpha[o[k]] + log_share[prog[o[k]], d[k]];
    target += (x[k] * eta - exp(eta)) / phi[prog[o[k]]];
  }
}
generated quantities {
  real level = exp(log_level);
  vector[J] ratio = level * exp(log_s);
  vector[N_o] reserve = rep_vector(0, N_o);
  vector[J] program_reserve = rep_vector(0, J);
  real total;
  for (k in 1:N_fut) {
    real mu = exp(alpha[fo[k]] + log_share[prog[fo[k]], fd[k]]);
    // Gamma with mean mu and variance phi mu: shape mu / phi, rate 1 / phi.
    real draw = gamma_rng(mu / phi[prog[fo[k]]], 1 / phi[prog[fo[k]]]);
    reserve[fo[k]] += draw;
  }
  for (w in 1:N_o) program_reserve[prog[w]] += reserve[w];
  total = sum(reserve);
}
