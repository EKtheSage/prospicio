### Distributions and probability

- `linalg::inverse` factors the matrix once and solves `n` triangular systems (about 7 times faster at `n = 50`), `linalg::cholesky_inverse` and the eigenvalue routine allocate less, `rearrange` sums and sorts each column less often (about 2.5 times faster at `d = 10`, `n = 10 000`), `correlated_normals` draws without a temporary vector and `Sampled` sorts its draws on first use. The results are the same to the last bit, and seeded runs give the same numbers.
