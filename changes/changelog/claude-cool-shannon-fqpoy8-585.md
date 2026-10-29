### Aggregate and reinsurance

- A per-risk excess of loss with an event limit, inuring to a cat cover: `per_risk::per_risk_inuring(layer, event_limit, years, times, seed)` applies a per-risk layer to each risk's loss in each event, caps an event's recoveries at the event limit, uses up the layer's annual terms on them in order, and returns the cession and the events net of it as an `EventSet` for a tower of cat layers (`aggregate.md`, "A per-risk layer with an event limit"). Python `reinsurance.per_risk_inuring`, R `per_risk_inuring()` (#585, #705).
