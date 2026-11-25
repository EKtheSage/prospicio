### Pricing

- `retro::RetroPlan::loss_sensitive_premium(basic)` returns the plan's
  premium terms as the existing `LossSensitivePremium`, the tax multiplier
  folded in; it replaces `RetroPlan::premium`, added earlier in this
  release (`RetroPlan.premium` in Python). `balance` now refuses a plan
  that would balance only with a negative basic premium (#981, #1089).
