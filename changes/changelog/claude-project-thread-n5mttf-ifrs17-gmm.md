### Reserving

- `ifrs17::Group`: IFRS 17's general measurement model for a group of
  contracts. Recognise it from expected cash flows by future period, an RA
  and a locked-in discount curve, then roll it forward a period at a time:
  each `GroupPeriod` gives the CSM (accreted, adjusted for future service,
  released by coverage units), the loss component and its reversal, the
  reconciliations by measurement component and by remaining coverage, and
  insurance revenue, service expenses and finance expenses. Matches the
  IASB's Illustrative Examples 1, 2, 3A, 7 and 8 to the unit (Example 8's
  later years within one and a half, as it chains on rounded figures). Rust only for
  now; the Python and R bindings follow (#982).
