### Aggregate and reinsurance

- A tower's ceded losses by event as an event set, for an occurrence retro
  on a reinsurer's own cessions (`Tower::ceded_events`): each event's
  ceded loss to the named layers after their annual terms, with its time
  and labels, adding to the layers' ceded totals in every year; several
  deals' sets join with `EventSet::merge`. Python `Tower.ceded_events`, R
  `ceded_events()` (#590, #661).
