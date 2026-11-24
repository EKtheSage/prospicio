---
type: Finding
title: faer against prospicio-math's linear algebra
description: faer 0.24.4 on one thread is no faster than prospicio-math::linalg below 16 by 16, 2 to 9 times faster for Cholesky, inverse and product from 32 by 32 to 512 by 512, and 8 to 126 times for the symmetric eigendecomposition (cyclic Jacobi here); it adds 48 crates and about a minute to a clean build, and no caller in the workspace factors a matrix large enough to gain
tags: [math, linalg, faer, performance, dependencies, environment]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-24T22:00:00Z }
sources:
  - id: linalg
    resource: ../crates/prospicio-math/src/linalg.rs
    title: prospicio_math::linalg (cholesky, cholesky_solve, solve, inverse, mat_mul, symmetric_eigen)
  - id: faer
    resource: https://crates.io/crates/faer
    title: faer 0.24.4 (MIT, rust-version 1.84), features std and linalg, default features off
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/974
    title: "#974: evaluate faer behind prospicio-math::linalg for medium and large dense systems"
  - id: gam
    resource: ../crates/prospicio-glm/src/gam.rs
    title: prospicio_glm::gam (the REML score's symmetric_eigenvalues, one smooth's penalty at a time)
---

# Finding

#974 asks whether `faer` should sit behind `prospicio_math::linalg`
[issue]. Measured on the cloud container (4 cores, one thread for both,
`faer::set_global_parallelism(Par::Seq)`, release build, best of five
batches), on a symmetric positive definite matrix `A'A + nI`. Times are
microseconds a call; the ratio is ours over faer's, so above 1 faer is
faster.

| n | Cholesky and solve | LU solve | Inverse | Product | Symmetric eigen |
|---|---|---|---|---|---|
| 4 | 0.19 / 0.16 = 1.2 | 0.24 / 0.59 = 0.4 | 0.41 / 0.71 = 0.6 | 0.10 / 0.06 = 1.8 | 1.3 / 1.3 = 1.0 |
| 8 | 0.42 / 0.47 = 0.9 | 0.63 / 0.81 = 0.8 | 1.8 / 2.6 = 0.7 | 0.47 / 0.09 = 5.2 | 8.8 / 5.0 = 1.7 |
| 16 | 1.1 / 1.0 = 1.1 | 1.5 / 2.0 = 0.7 | 5.3 / 3.6 = 1.5 | 1.6 / 0.33 = 4.9 | 52 / 17 = 3.1 |
| 32 | 5.7 / 3.2 = 1.8 | 7.5 / 5.1 = 1.5 | 34 / 15 = 2.3 | 10 / 1.4 = 7.4 | 432 / 55 = 7.8 |
| 64 | 35 / 10 = 3.4 | 28 / 20 = 1.4 | 224 / 47 = 4.8 | 72 / 9.8 = 7.4 | 6,719 / 241 = 28 |
| 128 | 226 / 55 = 4.1 | 217 / 132 = 1.6 | 2,151 / 394 = 5.5 | 626 / 75 = 8.4 | 80,003 / 1,385 = 58 |
| 256 | 1,779 / 247 = 7.2 | 2,057 / 625 = 3.3 | 13,993 / 1,918 = 7.3 | 3,969 / 570 = 7.0 | 828,550 / 6,570 = 126 |
| 512 | 15,370 / 1,667 = 9.2 | 13,082 / 4,310 = 3.0 | 194,433 / 12,506 = 15.5 | 45,560 / 4,134 = 11.0 | not run |

The two agree: solutions to `1e-17` in absolute terms on these
well-conditioned systems, eigenvalues to `7e-14` relative at 256.

faer's times exclude copying a row-major slice into its column-major
`Mat`. Done element by element (`Mat::from_fn`) the copy is as large as
the factorization up to `n = 32` and doubles the Cholesky at 256 and 512;
a wrapper would have to copy by column or hold `Mat`s.

## What it would cost

- **Dependencies**: 48 crates under `faer` with default features off and
  `std` and `linalg` on (`gemm` and `nano-gemm` kernels for four scalar
  types, `pulp`, `equator`, `dyn-stack`, `qd`, `raw-cpuid` and proc
  macros). All of them would be reviewed by
  `cargo deny`; faer itself is MIT.
- **Build time**: a clean release build of faer and its dependencies
  took 57 s wall and 2 minutes of CPU here. Every CI job that builds
  `prospicio-math` cold would pay it, and every crate depends on
  `prospicio-math`.
- **WASM**: `cargo check --target wasm32-unknown-unknown` passes with
  those features (46 crates).
- **MSRV**: 1.84 against the workspace's 1.92, so no change.
- **Bits**: faer chooses kernels by the CPU's features at run time, so
  results could differ between machines in the last bits, where
  `linalg`'s scalar loops give the same bits everywhere. The workspace's
  pins and parity tests would have to allow for that. Not measured.
- **`unsafe`**: faer has over a thousand `unsafe` occurrences in its
  source; `prospicio-math` has none.

## Where the workspace uses dense linear algebra

Every caller of `linalg` factors a matrix whose side is a number of
coefficients, of smooth basis functions, of units or of segments:

- GLMs, the GAM, the GLMM and their relatives factor the `p` by `p`
  information matrix once an iteration. Assembling `X'WX` over `n` rows
  costs about `n p^2 / 2` multiplications against `p^3 / 3` for the
  factorization, so the factorization is `2 p / (3 n)` of the assembly:
  under 1% for a freMTPL2-sized fit with 50 columns. #969 worked on the
  assembly for that reason.
- The GAM's REML score takes the eigenvalues of one smooth's penalty at a
  time, a matrix of 10 to 40 basis functions [gam], where the table has
  the two within a factor of 3 to 8 and both under half a millisecond.
- Copulas, correlation matrices and capital allocations factor a matrix
  of units or lines, in the tens.

None of these is in the range where faer is several times faster and the
factorization is also a visible share of the run.

## Recommendation

For the user to decide (#974):

1. **Do not adopt faer now.** It would add 48 crates and a minute of
   cold build to every crate for a gain no current caller would see.
   `docs/architecture.md` already says the linear algebra is
   `prospicio-math`'s own; its sentence that a BLAS or faer may come
   behind it can stay as it is.
2. **If the eigendecomposition becomes a cost** (a tensor smooth with
   hundreds of basis functions, a correlation matrix of hundreds of
   units), the first step is in-house: cyclic Jacobi is the slowest
   standard method at these sizes, and a tridiagonal reduction with the
   implicit QL iteration is the usual replacement. How much of faer's
   lead in the last column that would recover was not measured.
3. **Revisit with a benchmark of a real caller** when one works on
   matrices of 100 by 100 or more inside a loop. The harness below is the
   starting point.

## The harness

A project outside the workspace, so that nothing here depends on faer:

```toml
[dependencies]
prospicio-math = { path = "crates/prospicio-math" }
faer = { version = "0.24.4", default-features = false, features = ["std", "linalg"] }
```

```rust
use faer::prelude::*;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, Side};
use prospicio_math::linalg;
use std::hint::black_box;
use std::time::Instant;

fn spd(n: usize) -> Vec<f64> {
    // A' A + n I from a fixed pseudo-random A: symmetric positive definite.
    let mut s = 12345u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5
    };
    let a: Vec<f64> = (0..n * n).map(|_| next()).collect();
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut d = 0.0;
            for k in 0..n {
                d += a[k * n + i] * a[k * n + j];
            }
            m[i * n + j] = d + if i == j { n as f64 } else { 0.0 };
        }
    }
    m
}

fn time(mut f: impl FnMut() -> f64, budget_ms: u128) -> f64 {
    // Best of several batches, in microseconds a call.
    let mut reps = 1u32;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            black_box(f());
        }
        if t.elapsed().as_millis() >= budget_ms / 4 || reps > 1 << 22 {
            break;
        }
        reps *= 2;
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            black_box(f());
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e6 / reps as f64);
    }
    best
}

fn main() {
    faer::set_global_parallelism(faer::Par::Seq);
    println!("| n | op | own (us) | faer (us) | faer incl. copy (us) | ratio own/faer | max diff |");
    println!("|---|---|---|---|---|---|---|");
    for n in [4usize, 8, 16, 32, 64, 128, 256, 512] {
        let a = spd(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let to_mat = |v: &[f64]| Mat::<f64>::from_fn(n, n, |i, j| v[i * n + j]);
        let fa = to_mat(&a);
        let fb = Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
        let budget = 200;

        // Cholesky factor and solve.
        let own = time(|| { let l = linalg::cholesky(&a, n).unwrap(); linalg::cholesky_solve(&l, &b)[n / 2] }, budget);
        let fr = time(|| { let l = fa.llt(Side::Lower).unwrap(); l.solve(&fb)[(n / 2, 0)] }, budget);
        let frc = time(|| { let m = to_mat(&a); let l = m.llt(Side::Lower).unwrap(); l.solve(&fb)[(n / 2, 0)] }, budget);
        let x1 = { let l = linalg::cholesky(&a, n).unwrap(); linalg::cholesky_solve(&l, &b) };
        let x2 = fa.llt(Side::Lower).unwrap().solve(&fb);
        let d = (0..n).map(|i| (x1[i] - x2[(i, 0)]).abs()).fold(0.0, f64::max);
        println!("| {n} | cholesky + solve | {own:.2} | {fr:.2} | {frc:.2} | {:.2} | {d:.1e} |", own / fr);

        // LU solve.
        let own = time(|| linalg::solve(a.clone(), b.clone(), n).unwrap()[n / 2], budget);
        let fr = time(|| fa.partial_piv_lu().solve(&fb)[(n / 2, 0)], budget);
        let frc = time(|| to_mat(&a).partial_piv_lu().solve(&fb)[(n / 2, 0)], budget);
        let x1 = linalg::solve(a.clone(), b.clone(), n).unwrap();
        let x2 = fa.partial_piv_lu().solve(&fb);
        let d = (0..n).map(|i| (x1[i] - x2[(i, 0)]).abs()).fold(0.0, f64::max);
        println!("| {n} | LU solve | {own:.2} | {fr:.2} | {frc:.2} | {:.2} | {d:.1e} |", own / fr);

        // Inverse.
        let own = time(|| linalg::inverse(&a, n).unwrap()[n + 1], budget);
        let fr = time(|| fa.partial_piv_lu().inverse()[(1, 1)], budget);
        println!("| {n} | inverse | {own:.2} | {fr:.2} | | {:.2} | |", own / fr);

        // Product.
        let own = time(|| linalg::mat_mul(&a, &a, n)[n + 1], budget);
        let fr = time(|| (&fa * &fa)[(1, 1)], budget);
        println!("| {n} | product | {own:.2} | {fr:.2} | | {:.2} | |", own / fr);

        // Symmetric eigendecomposition.
        if n <= 256 {
            let own = time(|| linalg::symmetric_eigen(&a, n).0[0], budget);
            let fr = time(|| fa.self_adjoint_eigen(Side::Lower).unwrap().S().column_vector()[0], budget);
            let e1 = linalg::symmetric_eigen(&a, n).0;
            let ev = fa.self_adjoint_eigen(Side::Lower).unwrap();
            let s = ev.S().column_vector();
            let d = (0..n).map(|i| (e1[i] - s[i]).abs() / e1[i].abs()).fold(0.0, f64::max);
            println!("| {n} | symmetric eigen | {own:.2} | {fr:.2} | | {:.2} | {d:.1e} rel |", own / fr);
        }
    }
}
```
