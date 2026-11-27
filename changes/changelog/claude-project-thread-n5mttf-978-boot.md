### Reserving

- ODP and Mack bootstrap fits save to JSON and read back exactly, their simulated reserves, patterns and cells included: `to_json`/`from_json` on `OdpBootstrapFit(s)` and `MackBootstrapFit(s)` in Rust, and `to_json`, `from_json` and `pickle` on Python's `OdpBootstrapFit` and `MackBootstrapFit` (#978).
