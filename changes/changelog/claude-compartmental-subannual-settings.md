### Reserving

- `Compartmental` on quarterly or monthly data: model 2's development levels can be one for each development year instead of one for each age (`DevelopmentLevels::ByYear`; Python and R `development_levels="by_year"`), with one level for each age as the default and unchanged; a fit with cumulative paid on cells finer than a year warns, as a prompt to check the residuals; the help covers the rate priors, seasonality, what a sub-annual fit promises and earned exposure. Fixes the one-year view of a model with grouped development effects, which indexed them by age.
