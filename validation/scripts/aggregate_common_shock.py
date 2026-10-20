"""Print aggregate 1.2.0's shared-mixing aggregate behind `shared_mixing_matches_aggregate`.

    python3.13 -m venv agg && agg/bin/pip install aggregate==1.2.0
    agg/bin/python validation/scripts/aggregate_common_shock.py

One `Aggregate` with two severity components, 10 gamma claims (mean 10,
cv 0.5) and 20 lognormal claims (mean 20, cv 1), under one gamma mixing
variable with cv 0.3: aggregate applies the mixed count once to the
claim-weighted mixture of the severities, so the components share the
mixing variable (`stats_df`'s `mixed` column; `independent` is the same
book with each component mixed on its own). A `Portfolio`'s units do not
share it: its density is the product of the units' transforms. The test
in `validation/tests/aggregate.rs` copies the mean, cv, skewness and the
quantiles below, on bs 0.5 and 2^14 points.
"""

import warnings

import numpy as np
from aggregate import build

warnings.filterwarnings("ignore")

a = build("agg Two [10 20] claims sev [gamma lognorm] [10 20] cv [0.5 1] mixed gamma 0.3",
          bs=0.5, log2=14)
xs, p = a.xs, a.agg_density
m = float((xs * p).sum())
v = float(((xs - m) ** 2 * p).sum())
print("mean", m, "cv", v ** 0.5 / m, "skew", float(((xs - m) ** 3 * p).sum()) / v ** 1.5)
cdf = np.cumsum(p)
for q in (0.5, 0.9, 0.99, 0.999):
    print("quantile", q, xs[np.searchsorted(cdf, q)])
print(a.stats_df.loc[[("agg", "cv"), ("agg", "skew")], ["mixed", "independent"]].to_string())
