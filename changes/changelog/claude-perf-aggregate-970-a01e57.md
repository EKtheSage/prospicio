### Aggregate and reinsurance

- The reinsurance tower's simulated paths (`Tower::apply`, `apply_aggregate`, `apply_annual`, `ceded_events`, `apply_term`, `apply_aggregate_term` and `apply_timeline`) run their years in parallel and allocate nothing per year: 17 to 38 times faster on 16 threads and 3.6 to 7.5 times on one, on 200,000 years. The results are the same to the last bit, on any number of threads, so seeded runs give the same numbers (#1053, part of #970).
