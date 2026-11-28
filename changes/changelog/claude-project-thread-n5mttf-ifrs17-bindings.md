### Reserving

- Python: `Ifrs17Group`, `PaaGroup` and `ReinsuranceHeld` in
  `prospicio.reserving` measure an IFRS 17 group under the general model,
  under the premium allocation approach, or a reinsurance contract held,
  from dicts of cash-flow columns, rolled forward period by period with
  each period's reconciliations as dicts of columns. Their state saves as
  versioned JSON (`to_json`, `from_json`) or by pickling, and a reloaded
  group rolls forward to the same numbers (#982).
