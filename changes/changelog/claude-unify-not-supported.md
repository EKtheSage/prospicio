### Breaking changes

- Rust: a setting a reserving method does not support is `prospicio_reserving::Error::NotSupported` in every method, no longer `Error::Bootstrap`, `Error::Bayes` or `Error::ClaimsDevelopment` (Merz and Wuthrich's closed form with another `alpha`, a tail, a fully selected sparse age or link ratio exclusions; the bootstraps with a sparse age or a one-year tail or pattern that conflicts with the re-reserving method's; `Synchronized` Bayesian segments; `Compartmental` with one response under free rates, its outstanding table under `Fitted::PaidOnly` and the backtest of a response it does not fit), and the messages read "{setting} is not supported by {method}: {reason}". `Compartmental`'s bound on the `k_p` prior under `Fitted::IncurredOnly` is `Error::InvalidSetting`. `Error` has the variant `NotYetBuilt`, so an exhaustive `match` needs an arm for it (#580).
- Python: these errors raise `prospicio.NotSupportedError`, a subclass of `ValueError`, and `Compartmental(priors, fitted="paid_only")` (or `"incurred_only"`) under free rates raises when it is built, not when it is fitted. R: they signal an error of class `prospicio_not_supported`, and the compartmental functions refuse those settings before reading the triangle (#580).

### Reserving

- One error for a setting a method does not support (#580): `Error::NotSupported` with `Error::is_not_supported()`, `Compartmental::check_supported()`, Python `prospicio.NotSupportedError` and the R condition class `prospicio_not_supported`, so a caller can catch the case in one place.
