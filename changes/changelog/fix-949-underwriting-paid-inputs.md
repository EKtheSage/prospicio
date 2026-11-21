### Reserving

- `underwriting_paid_by_month` returns an error, instead of panicking, for an
  empty `written`, and also rejects `times` and `shares` of unequal length and
  a non-finite time. Valid inputs give the same results (#949, #1008).
