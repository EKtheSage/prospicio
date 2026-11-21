### Reserving

- Under `PaymentBasis::Incurred`, the `unpaid` draws of `discounted_reserves`
  record the basis, the paid pattern and the latest paid in their
  provenance, as the `discounted` draws did, and the `input_hash` of both
  is the fit's with the paid pattern and the latest paid hashed in. Before,
  two runs on different latest paid amounts gave different `unpaid` draws
  with the same record. The `input_hash` of an incurred-basis result
  changes once; the paid basis keeps the fit's (#958).
