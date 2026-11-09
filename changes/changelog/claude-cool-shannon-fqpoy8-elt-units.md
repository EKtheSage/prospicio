### Aggregate and reinsurance

- Event loss tables of several units that share events, `aggregate`
  2.1.0's `elt unit A [...] elt unit B [...]`: `UnitEventLossTables` with
  the portfolio's table rolled up by event and each unit's own table, and
  `EventLossTable::severity_moments`; Python `UnitEventLossTables` and
  `EventLossTable.severity_moments`, R `unit_event_loss_tables()` and
  `elt_severity_moments()` (#777, #819).
