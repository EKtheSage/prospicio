### Aggregate and reinsurance

- The Solvency II standard formula's counterparty default module for type 1
  exposures (Delegated Regulation (EU) 2015/35), as pure functions in
  `prospicio_aggregate::solvency2`: the probability of default by credit
  quality step (`probability_by_step`, Art 199(2)) and by solvency ratio
  (`probability_by_ratio`, Art 199(3), interpolated and clamped), the loss
  given default of a reinsurance arrangement (`reinsurance_lgd`, Art
  192(2), with the 50 % or 90 % share and the collateral factor as
  arguments), and `type1`, giving `V_inter`, `V_intra`, the standard
  deviation and `SCR_def,1` (Arts 200 and 201). In Python as
  `prospicio.reinsurance.solvency2_probability_by_step`,
  `solvency2_probability_by_ratio`, `solvency2_reinsurance_lgd` and
  `solvency2_type1` (returning `Solvency2Type1`), and in R as
  `solvency2_probability_by_step()`, `solvency2_probability_by_ratio()`,
  `solvency2_reinsurance_lgd()` and `solvency2_type1()` (#980, #1082).
