//! Regular vine (R-vine) copulas: any vine whose trees satisfy the
//! proximity condition, given by an R-vine matrix in vinecopulib's
//! convention (#568). C- and D-vines ([`VineCopula`]) are special cases.
//!
//! The matrix is `d × d` with 1-based variable labels: the antidiagonal
//! `M[d − 1 − j][j]` holds the order, and above it column `j` row `t` holds
//! the variable that tree `t`'s edge `j` joins to `M[d − 1 − j][j]`, given
//! the variables above it in the column (`M[0..t][j]`). Entries below the
//! antidiagonal are 0, and so are the rows of a truncated vine's missing
//! trees. This is the matrix of vinecopulib, pyvinecopulib and
//! rvinecopulib (`Vinecop.matrix`, `rvine_matrix`); R's VineCopula package
//! stores the same vine flipped upside down and left to right.
//!
//! `pairs[t][j]` is tree `t`'s edge `j`, with first argument the diagonal
//! variable `M[d − 1 − j][j]` and second `M[t][j]`, both given the
//! conditioning set, as vinecopulib's `pair_copulas[t][j]`. The density
//! and the inverse Rosenblatt transform run the natural-order recursions of
//! Dißmann, Brechmann, Czado and Kurowicka (2013, algorithms 1 and 2).

use prospicio_core::{Error, Result, StreamRng};

use crate::copula::Copula;
use crate::vine::{PairCopula, VineCopula, VineKind, clamp01};

/// An R-vine copula in `d` dimensions.
///
/// ```
/// use prospicio_prob::rvine::RVineCopula;
/// use prospicio_prob::vine::{PairCopula, PairFamily};
///
/// // pyvinecopulib's D-vine on (1, 2, 3): edges 1-2, 2-3, then 1-3 | 2.
/// let m = vec![vec![2, 3, 3], vec![3, 2, 0], vec![1, 0, 0]];
/// let g = |r| PairCopula::new(PairFamily::Gaussian, r, 0).unwrap();
/// let v = RVineCopula::new(m, vec![vec![g(0.5), g(0.3)], vec![g(0.2)]]).unwrap();
/// let u = [0.2, 0.7, 0.4];
/// let want = g(0.5).log_density(0.2, 0.7)
///     + g(0.3).log_density(0.7, 0.4)
///     + g(0.2).log_density(g(0.5).h2(0.2, 0.7), g(0.3).h1(0.4, 0.7));
/// assert!((v.log_density(&u).unwrap() - want).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RVineCopula {
    /// The order, 0-based: `order[j]` is the antidiagonal of column `j`.
    order: Vec<usize>,
    /// `natural[t][j]`: the position in `order` of `M[t][j]`.
    natural: Vec<Vec<usize>>,
    /// `minimum[t][j]`: the least of `natural[0..=t][j]`.
    minimum: Vec<Vec<usize>>,
    pairs: Vec<Vec<PairCopula>>,
}

impl RVineCopula {
    /// The vine with R-vine matrix `matrix` (vinecopulib's convention,
    /// 1-based) and `pairs[t]` for each tree `t` kept, `d − 1 − t` pair
    /// copulas each. Fewer trees than `d − 1` truncate the vine; the
    /// matrix's rows for the missing trees are not read.
    pub fn new(matrix: Vec<Vec<usize>>, pairs: Vec<Vec<PairCopula>>) -> Result<Self> {
        let d = matrix.len();
        if d < 2 || matrix.iter().any(|r| r.len() != d) {
            return Err(Error::Data(format!(
                "an R-vine matrix is d × d with d at least 2; got {d} rows of lengths {:?}",
                matrix.iter().map(Vec::len).collect::<Vec<_>>()
            )));
        }
        let trunc = pairs.len();
        if trunc > d - 1 {
            return Err(Error::Data(format!(
                "{trunc} trees for a {d}-dimensional vine, which has {}",
                d - 1
            )));
        }
        for (t, tree) in pairs.iter().enumerate() {
            if tree.len() != d - 1 - t {
                return Err(Error::Data(format!(
                    "tree {t} of a {d}-dimensional vine needs {} pair copulas, got {}",
                    d - 1 - t,
                    tree.len()
                )));
            }
        }
        let order: Vec<usize> = (0..d).map(|j| matrix[d - 1 - j][j]).collect();
        let mut seen = order.clone();
        seen.sort_unstable();
        if seen.iter().enumerate().any(|(i, &v)| v != i + 1) {
            return Err(Error::Data(format!(
                "the antidiagonal must hold 1, ..., d once each; got {order:?}"
            )));
        }
        let order: Vec<usize> = order.iter().map(|v| v - 1).collect();
        let mut position = vec![0; d];
        for (j, &v) in order.iter().enumerate() {
            position[v] = j;
        }
        let mut natural = Vec::with_capacity(trunc);
        for (t, row) in matrix.iter().enumerate().take(trunc) {
            let mut r = Vec::with_capacity(d - 1 - t);
            for (j, &v) in row.iter().enumerate().take(d - 1 - t) {
                if v < 1 || v > d {
                    return Err(Error::Data(format!(
                        "entry ({t}, {j}) of the R-vine matrix is {v}; tree {t} needs labels 1 to {d}"
                    )));
                }
                r.push(position[v - 1]);
            }
            natural.push(r);
        }
        // Each column holds variables later in the order, once each.
        for j in 0..d - 1 {
            let col: Vec<usize> = (0..trunc.min(d - 1 - j)).map(|t| natural[t][j]).collect();
            let mut s = col.clone();
            s.sort_unstable();
            s.dedup();
            if s.len() != col.len() || col.iter().any(|&p| p <= j) {
                return Err(Error::Data(format!(
                    "column {j} of the R-vine matrix must hold distinct variables that come after \
                     its antidiagonal {} in the order",
                    order[j] + 1
                )));
            }
        }
        let mut minimum = natural.clone();
        for t in 1..trunc {
            let (below, rest) = minimum.split_at_mut(t);
            for (m, &under) in rest[0].iter_mut().zip(&below[t - 1]) {
                *m = (*m).min(under);
            }
        }
        // The proximity condition: edge (t, j)'s conditional distributions
        // come from two edges of tree t − 1.
        for t in 1..trunc {
            for j in 0..d - 1 - t {
                let m = minimum[t][j];
                let mut target: Vec<usize> = (0..t).map(|i| natural[i][j]).collect();
                target.push(natural[t][j]);
                let mut test: Vec<usize> = (0..t).map(|i| natural[i][m]).collect();
                test.push(m);
                target.sort_unstable();
                test.sort_unstable();
                if target != test {
                    return Err(Error::Data(format!(
                        "the R-vine matrix breaks the proximity condition at tree {t}, column {j}"
                    )));
                }
            }
        }
        Ok(Self {
            order,
            natural,
            minimum,
            pairs,
        })
    }

    /// The R-vine matrix, vinecopulib's convention (1-based, 0 where no
    /// tree is kept).
    pub fn matrix(&self) -> Vec<Vec<usize>> {
        let d = self.order.len();
        let mut m = vec![vec![0; d]; d];
        for (t, row) in self.natural.iter().enumerate() {
            for (j, &p) in row.iter().enumerate() {
                m[t][j] = self.order[p] + 1;
            }
        }
        for (j, &v) in self.order.iter().enumerate() {
            m[d - 1 - j][j] = v + 1;
        }
        m
    }

    /// `natural[t][j]`, the position in the order of `M[t][j]`, and
    /// `minimum[t][j]`, the least of `natural[0..=t][j]`.
    pub(crate) fn natural_order(&self) -> (&[Vec<usize>], &[Vec<usize>]) {
        (&self.natural, &self.minimum)
    }

    /// The order (the antidiagonal), 0-based.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// The pair copulas by tree and edge.
    pub fn pairs(&self) -> &[Vec<PairCopula>] {
        &self.pairs
    }

    /// The number of trees kept.
    pub fn trunc_level(&self) -> usize {
        self.pairs.len()
    }

    /// The variables tree `t`'s edge `j` joins, `(first, second)` 0-based,
    /// and its conditioning set.
    pub fn edge(&self, t: usize, j: usize) -> (usize, usize, Vec<usize>) {
        let given = (0..t).map(|i| self.order[self.natural[i][j]]).collect();
        (self.order[j], self.order[self.natural[t][j]], given)
    }

    /// The number of parameters: 0 for independence, 2 for the Student t,
    /// 1 for the others.
    pub fn n_parameters(&self) -> usize {
        self.pairs.iter().flatten().map(n_parameters).sum()
    }

    /// The log-density at `u` (one value per dimension, in `[0, 1]`).
    pub fn log_density(&self, u: &[f64]) -> Result<f64> {
        check_point(u, self.order.len())?;
        let d = self.order.len();
        let mut h2: Vec<f64> = self.order.iter().map(|&v| u[v]).collect();
        let mut h1 = vec![0.0; d];
        let mut ll = 0.0;
        for (t, tree) in self.pairs.iter().enumerate() {
            for (j, c) in tree.iter().enumerate() {
                let m = self.minimum[t][j];
                let a = h2[j];
                let b = if self.natural[t][j] == m {
                    h2[m]
                } else {
                    h1[m]
                };
                ll += c.log_density(a, b);
                // Columns at or after j + 1 are read later in this tree, so
                // writing column j in place is safe.
                h1[j] = c.h1(b, a);
                h2[j] = c.h2(a, b);
            }
        }
        Ok(ll)
    }

    /// The inverse Rosenblatt transform: the point whose conditional
    /// distributions are the uniforms `w`, given by variable as
    /// pyvinecopulib's `inverse_rosenblatt` (the last variable in the order
    /// is `w` of that variable; each earlier one is conditioned on those
    /// after it).
    pub fn inverse_rosenblatt(&self, w: &[f64]) -> Result<Vec<f64>> {
        let d = self.order.len();
        check_point(w, d)?;
        let trunc = self.pairs.len();
        if trunc == 0 {
            return Ok(w.to_vec());
        }
        let mut hinv2 = vec![vec![0.0; d]; trunc + 1];
        let mut h1 = vec![vec![0.0; d]; trunc + 1];
        for j in 0..d {
            hinv2[trunc.min(d - j - 1)][j] = w[self.order[j]];
        }
        h1[0][d - 1] = hinv2[0][d - 1];
        for var in (0..d - 1).rev() {
            for t in (0..=(trunc - 1).min(d - var - 2)).rev() {
                let c = self.pairs[t][var];
                let m = self.minimum[t][var];
                let a = hinv2[t + 1][var];
                let b = if self.natural[t][var] == m {
                    hinv2[t][m]
                } else {
                    h1[t][m]
                };
                let x = c.h2_inverse(a, b);
                hinv2[t][var] = x;
                h1[t + 1][var] = c.h1(b, x);
            }
        }
        let mut u = vec![0.0; d];
        for (j, &v) in self.order.iter().enumerate() {
            u[v] = hinv2[0][j];
        }
        Ok(u)
    }

    /// The vine of `trees`, each edge `(a, b, given, pair)` with `pair`'s
    /// first argument `a`, peeled into a matrix as vinecopulib does: each
    /// column takes as its antidiagonal the first leaf found in the highest
    /// tree still holding edges, then descends the trees.
    pub(crate) fn from_trees(d: usize, trees: Vec<Vec<TreeEdge>>) -> Result<Self> {
        let trunc = trees.len();
        if trunc == 0 {
            let mut m = vec![vec![0; d]; d];
            for j in 0..d {
                m[d - 1 - j][j] = j + 1;
            }
            return Self::new(m, Vec::new());
        }
        // Each edge's two endpoints in the previous tree's edges (or the
        // variables, in tree 0), and how many unpeeled edges touch each.
        let mut ends: Vec<Vec<(usize, usize)>> = Vec::with_capacity(trunc);
        let mut degrees: Vec<Vec<usize>> = Vec::with_capacity(trunc);
        for (t, tree) in trees.iter().enumerate() {
            if tree.len() != d - 1 - t {
                return Err(Error::Data(format!(
                    "tree {t} needs {} edges, got {}",
                    d - 1 - t,
                    tree.len()
                )));
            }
            let nodes = if t == 0 { d } else { d - t };
            let mut deg = vec![0; nodes];
            let mut e = Vec::with_capacity(tree.len());
            for edge in tree {
                let (n1, n2) = if t == 0 {
                    (edge.a, edge.b)
                } else {
                    let find = |x: usize| {
                        trees[t - 1].iter().position(|p| {
                            let mut all = p.all();
                            all.sort_unstable();
                            let mut want = edge.given.clone();
                            want.push(x);
                            want.sort_unstable();
                            all == want
                        })
                    };
                    match (find(edge.a), find(edge.b)) {
                        (Some(x), Some(y)) => (x, y),
                        _ => {
                            return Err(Error::Data(format!(
                                "tree {t}'s edge {} - {} breaks the proximity condition",
                                edge.a, edge.b
                            )));
                        }
                    }
                };
                deg[n1] += 1;
                deg[n2] += 1;
                e.push((n1, n2));
            }
            ends.push(e);
            degrees.push(deg);
        }
        let mut consumed: Vec<Vec<bool>> = trees.iter().map(|t| vec![false; t.len()]).collect();
        let mut order = vec![0; d];
        let mut struct_rows: Vec<Vec<usize>> = (0..trunc).map(|t| vec![0; d - 1 - t]).collect();
        let mut pairs: Vec<Vec<PairCopula>> = (0..trunc)
            .map(|t| vec![PairCopula::independence(); d - 1 - t])
            .collect();
        for col in 0..d - 1 {
            let t = trunc.min(d - 1 - col).max(1);
            let top = t - 1;
            let mut pick = None;
            for (k, edge) in trees[top].iter().enumerate() {
                if consumed[top][k] {
                    continue;
                }
                let (n1, n2) = ends[top][k];
                if degrees[top][n1] == 1 {
                    pick = Some((k, edge.a));
                    break;
                }
                if degrees[top][n2] == 1 {
                    pick = Some((k, edge.b));
                    break;
                }
            }
            let (k, diag) = pick.ok_or_else(|| Error::Data("no leaf to peel".into()))?;
            order[col] = diag;
            let mut place = |tree: usize, k: usize, consumed: &mut Vec<Vec<bool>>| {
                let edge = &trees[tree][k];
                struct_rows[tree][col] = if edge.a == diag { edge.b } else { edge.a };
                pairs[tree][col] = if edge.a == diag {
                    edge.pair
                } else {
                    edge.pair.transposed()
                };
                consumed[tree][k] = true;
                let (n1, n2) = ends[tree][k];
                degrees[tree][n1] -= 1;
                degrees[tree][n2] -= 1;
                edge.given.clone()
            };
            let mut given = place(top, k, &mut consumed);
            for lower in (0..top).rev() {
                given.push(diag);
                given.sort_unstable();
                let k = (0..trees[lower].len())
                    .find(|&k| {
                        let e = &trees[lower][k];
                        let mut all = e.all();
                        all.sort_unstable();
                        !consumed[lower][k] && (e.a == diag || e.b == diag) && all == given
                    })
                    .ok_or_else(|| Error::Data("the trees break the proximity condition".into()))?;
                given = place(lower, k, &mut consumed);
            }
        }
        order[d - 1] = struct_rows[0][d - 2];
        let mut m = vec![vec![0; d]; d];
        for (t, row) in struct_rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                m[t][j] = v + 1;
            }
        }
        for (j, &v) in order.iter().enumerate() {
            m[d - 1 - j][j] = v + 1;
        }
        Self::new(m, pairs)
    }
}

/// An edge of a vine tree, for [`RVineCopula::from_trees`].
#[derive(Debug, Clone)]
pub(crate) struct TreeEdge {
    pub a: usize,
    pub b: usize,
    pub given: Vec<usize>,
    pub pair: PairCopula,
}

impl TreeEdge {
    fn all(&self) -> Vec<usize> {
        let mut v = self.given.clone();
        v.push(self.a);
        v.push(self.b);
        v
    }
}

impl From<&VineCopula> for RVineCopula {
    /// The same C- or D-vine as an R-vine.
    fn from(v: &VineCopula) -> Self {
        let order = v.order();
        let d = order.len();
        let trees: Vec<Vec<TreeEdge>> = v
            .pairs()
            .iter()
            .enumerate()
            .map(|(t, tree)| {
                tree.iter()
                    .enumerate()
                    .map(|(e, &pair)| match v.kind() {
                        VineKind::Canonical => TreeEdge {
                            a: order[t],
                            b: order[t + 1 + e],
                            given: order[..t].to_vec(),
                            pair,
                        },
                        VineKind::Drawable => TreeEdge {
                            a: order[e],
                            b: order[e + t + 1],
                            given: order[e + 1..e + t + 1].to_vec(),
                            pair,
                        },
                    })
                    .collect()
            })
            .collect();
        Self::from_trees(d, trees).expect("a C- or D-vine is a valid R-vine")
    }
}

fn n_parameters(c: &PairCopula) -> usize {
    use crate::vine::PairFamily;
    match c.family() {
        PairFamily::Independence => 0,
        PairFamily::Student => 2,
        _ => 1,
    }
}

fn check_point(u: &[f64], d: usize) -> Result<()> {
    if u.len() != d {
        return Err(Error::Data(format!(
            "{} arguments for a {d}-dimensional copula",
            u.len()
        )));
    }
    if let Some(&bad) = u.iter().find(|x| !(0.0..=1.0).contains(*x)) {
        return Err(Error::InvalidParameter {
            name: "u",
            value: bad,
            reason: "must be in [0, 1]",
        });
    }
    Ok(())
}

impl Copula for RVineCopula {
    fn dim(&self) -> usize {
        self.order.len()
    }

    /// `d` uniforms from the stream, by variable, then the inverse
    /// Rosenblatt transform.
    fn sample(&self, rng: &mut StreamRng, u: &mut [f64]) {
        let w: Vec<f64> = (0..self.order.len()).map(|_| rng.next_open01()).collect();
        let x = self
            .inverse_rosenblatt(&w)
            .expect("uniforms of the vine's dimension");
        for (out, v) in u.iter_mut().zip(x) {
            *out = clamp01(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vine::PairFamily;

    fn pair(f: PairFamily, p: f64, r: u16) -> PairCopula {
        PairCopula::new(f, p, r).unwrap()
    }

    fn vines() -> Vec<VineCopula> {
        let t1 = vec![
            pair(PairFamily::Gumbel, 2.0, 0),
            pair(PairFamily::Clayton, 1.5, 90),
            PairCopula::student(0.4, 5.0).unwrap(),
        ];
        let t2 = vec![
            pair(PairFamily::Joe, 1.8, 270),
            pair(PairFamily::Frank, -3.0, 0),
        ];
        let t3 = vec![pair(PairFamily::Gaussian, 0.3, 0)];
        let mut out = Vec::new();
        for kind in [VineKind::Canonical, VineKind::Drawable] {
            for trees in [
                vec![t1.clone(), t2.clone(), t3.clone()],
                vec![t1.clone(), t2.clone()],
            ] {
                out.push(VineCopula::new(kind, vec![2, 0, 3, 1], trees).unwrap());
            }
        }
        out
    }

    #[test]
    fn c_and_d_vines_are_r_vines() {
        let pts = [
            [0.2, 0.7, 0.4, 0.9],
            [0.55, 0.1, 0.8, 0.3],
            [0.95, 0.9, 0.85, 0.99],
        ];
        for v in vines() {
            let r = RVineCopula::from(&v);
            assert_eq!(r.trunc_level(), v.pairs().len());
            for u in pts {
                let (a, b) = (v.log_density(&u).unwrap(), r.log_density(&u).unwrap());
                assert!((a - b).abs() < 1e-11, "{v:?}: {a} {b}");
            }
            // The round trip through the matrix.
            let again = RVineCopula::new(r.matrix(), r.pairs().to_vec()).unwrap();
            assert_eq!(again, r);
        }
    }

    #[test]
    fn the_inverse_rosenblatt_inverts_the_conditionals() {
        // The transform's output has density whose conditionals give w back:
        // check through the density, by a finite-difference Jacobian.
        for v in vines() {
            let r = RVineCopula::from(&v);
            let w = [0.3, 0.6, 0.2, 0.8];
            let x = r.inverse_rosenblatt(&w).unwrap();
            // ln c(x) = −ln |det ∂w/∂x|: perturb each w and read dx.
            let e = 1e-6;
            let mut jac = vec![vec![0.0; 4]; 4];
            for k in 0..4 {
                let (mut a, mut b) = (w, w);
                a[k] += e;
                b[k] -= e;
                let (xa, xb) = (
                    r.inverse_rosenblatt(&a).unwrap(),
                    r.inverse_rosenblatt(&b).unwrap(),
                );
                for i in 0..4 {
                    jac[i][k] = (xa[i] - xb[i]) / (2.0 * e);
                }
            }
            let det = det4(&jac);
            let want = -det.abs().ln();
            let got = r.log_density(&x).unwrap();
            assert!((got - want).abs() < 1e-5, "{v:?}: {got} {want}");
        }
    }

    fn det4(m: &[Vec<f64>]) -> f64 {
        // Gaussian elimination with partial pivoting.
        let mut a: Vec<Vec<f64>> = m.to_vec();
        let n = a.len();
        let mut det = 1.0;
        for c in 0..n {
            let p = (c..n)
                .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                .unwrap();
            if p != c {
                a.swap(p, c);
                det = -det;
            }
            det *= a[c][c];
            let pivot = a[c].clone();
            for row in a.iter_mut().skip(c + 1) {
                let f = row[c] / pivot[c];
                for (x, p) in row.iter_mut().zip(&pivot).skip(c) {
                    *x -= f * p;
                }
            }
        }
        det
    }

    #[test]
    fn matrices_are_checked() {
        let g = pair(PairFamily::Gaussian, 0.5, 0);
        let ok = vec![vec![2, 3, 3], vec![3, 2, 0], vec![1, 0, 0]];
        assert!(RVineCopula::new(ok.clone(), vec![vec![g, g], vec![g]]).is_ok());
        // A repeated antidiagonal.
        let bad = vec![vec![2, 3, 3], vec![3, 3, 0], vec![1, 0, 0]];
        assert!(RVineCopula::new(bad, vec![vec![g, g], vec![g]]).is_err());
        // A column entry before its antidiagonal in the order.
        let bad = vec![vec![2, 1, 3], vec![3, 2, 0], vec![1, 0, 0]];
        assert!(RVineCopula::new(bad, vec![vec![g, g]]).is_err());
        // Wrong tree sizes.
        assert!(RVineCopula::new(ok.clone(), vec![vec![g]]).is_err());
        assert!(RVineCopula::new(ok, vec![vec![g, g], vec![g], vec![g]]).is_err());
        // Proximity: tree 0 is 1-2, 2-4, 3-4, which cannot hold 1-3 | 2.
        let bad = vec![
            vec![2, 4, 4, 4],
            vec![3, 3, 3, 0],
            vec![4, 2, 0, 0],
            vec![1, 0, 0, 0],
        ];
        assert!(RVineCopula::new(bad, vec![vec![g, g, g], vec![g, g]]).is_err());
    }
}
