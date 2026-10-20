### Excel

- Link-ratio exclusions and the geometric average: the `average` argument of the chain-ladder, Mack, expected-loss and Mack bootstrap functions takes `"geometric"` or a row `kind, n_periods, drop_high, drop_low` (such as `{"volume",5,1,1}`, the latest five calendar periods with the high and low link ratio left out), and `PRO.ODP`, `PRO.ODP.SIMS` and `PRO.ODP.CDR` take the same row as a last argument (chainladder-python and R ChainLadder parity).
