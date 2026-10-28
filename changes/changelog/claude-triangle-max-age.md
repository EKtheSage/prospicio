### Reserving

- `Triangle::max_age(age)` keeps the development ages up to `age` months in
  every segment, measure and origin, as chainladder-python's
  `tri[tri.development <= age]`: any age at or past the first keeps the ages
  up to it, one past the last returns the triangle unchanged, and one before
  the first is an error (`Error::AgeBeforeFirst`). In Python
  (`Triangle.max_age`) and R (`max_age()`). The errors of
  `BayesBfNormalised`, `BayesOdp` and `BayesBf` for a falling last age now
  name it as the remedy (`reserving-bayes.md` decision 18, #667).
