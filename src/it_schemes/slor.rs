use ndarray::Array2;
use crate::mesh::Node;
use crate::solver_core::calc_L_phi_ij;
use crate::it_schemes::IterativeScheme;

/// SLOR — Successive Line Over-Relaxation
///
/// This method is an extension of the Line Gauss-Seidel (LGS) scheme
/// with the addition of an over-relaxation factor `r`.
///
/// The formulation follows the delta form of the Laplace equation:
///
///     N C(i,j)^n + Lφ(i,j)^n = 0
///
/// where:
///     C(i,j)^n = Δφ(i,j)^n = φ(i,j)^(n+1) − φ(i,j)^n
///
/// Key ideas of SLOR:
///
/// • The discretization in the **y-direction is implicit**
///   → leads to a tridiagonal linear system for each vertical line.
///
/// • The discretization in the **x-direction is explicit**
///   → appears in the diagonal and in the right-hand side.
///
/// • A relaxation factor `r` is introduced:
///
///       r = 1.0  → Line Gauss-Seidel
///       r > 1.0  → Over-relaxation (faster convergence)
///       r < 1.0  → Under-relaxation (more stable)
///
/// • For each vertical line (fixed `i`), all corrections `C(i,j)`
///   are solved simultaneously using the Thomas algorithm.
///
/// The maximum correction magnitude is used as the convergence metric.
pub struct SLOR {
    /// Relaxation factor
    pub r: f64,
}

impl IterativeScheme for SLOR {

    /// Performs one full SLOR iteration over the domain.
    ///
    /// For each vertical line:
    /// 1) Assemble tridiagonal system in y for C(i,j)
    /// 2) Use full residual Lφ evaluated at iteration n
    /// 3) Solve the tridiagonal system
    /// 4) Update φ using the computed corrections
    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64 {

        let r = self.r;
        let (imax, jmax) = mesh.dim();
        let mut max_residual = 0.0;

        // Number of interior points in y (size of the tridiagonal system)
        let n = jmax - 2;

        // Correction vector for current vertical line
        let mut C = vec![0.0; n];

        // Tridiagonal coefficients
        let mut a = vec![0.0; n]; // sub-diagonal
        let mut b = vec![0.0; n]; // main diagonal
        let mut c = vec![0.0; n]; // super-diagonal
        let mut d = vec![0.0; n]; // right-hand side

        // ===== Vertical line sweep (characteristic of LGS/SLOR) =====
        for i in 1..imax-1 {

            // ----- Assemble linear system for this line -----
            for j in 1..jmax-1 {
                let jj = j - 1;

                // ===== Implicit operator in y applied to C =====
                // These coefficients come from δ̃yy C
                a[jj] = (1.0 / r) *
                    1.0 /
                    (((mesh[[i, j+1]].y - mesh[[i, j-1]].y) / 2.0)
                    * (mesh[[i, j]].y - mesh[[i, j-1]].y));

                c[jj] = (1.0 / r) *
                    1.0 /
                    (((mesh[[i, j+1]].y - mesh[[i, j-1]].y) / 2.0)
                    * (mesh[[i, j+1]].y - mesh[[i, j]].y));

                // ===== Explicit contribution in x included in diagonal =====
                b[jj] = (1.0 / r) * (
                    -2.0 /
                    ((mesh[[i+1, j]].x - mesh[[i-1, j]].x) / 2.0).powi(2)
                    - (a[jj] + c[jj])
                );

                // ===== Full residual Lφ evaluated at iteration n =====
                let L_phi_n_ij = calc_L_phi_ij(
                    mesh[[i, j]].x, mesh[[i, j]].y,
                    mesh[[i+1, j]].x, mesh[[i-1, j]].x,
                    mesh[[i, j+1]].y, mesh[[i, j-1]].y,
                    phi_n[[i, j]],
                    phi_n[[i+1, j]],
                    phi_n[[i-1, j]],
                    phi_n[[i, j-1]],
                    phi_n[[i, j+1]]
                );

                // ===== Right-hand side =====
                // Includes:
                // 1) Complete residual
                // 2) Gauss-Seidel contribution from previous line (i-1)
                d[jj] = -L_phi_n_ij
                    + C[jj] /
                    ((mesh[[i+1, j]].x - mesh[[i-1, j]].x) / 2.0).powi(2);
            }

            // ===== Solve tridiagonal system =====
            C = thomas(&a, &b, &c, &d);

            // ===== Update φ using delta formulation =====
            for j in 1..jmax-1 {
                let jj = j - 1;
                phi_n[[i, j]] += C[jj];

                // Track maximum correction
                let residual = C[jj].abs();
                if residual > max_residual {
                    max_residual = residual;
                }
            }
        }

        // ===== Final residual evaluation over domain =====
        for i in 1..imax-1 {
            for j in 1..jmax-1 {
                let residual = calc_L_phi_ij(
                    mesh[[i, j]].x,
                    mesh[[i, j]].y,
                    mesh[[i+1, j]].x,
                    mesh[[i-1, j]].x,
                    mesh[[i, j+1]].y,
                    mesh[[i, j-1]].y,
                    phi_n[[i, j]],
                    phi_n[[i+1, j]],
                    phi_n[[i-1, j]],
                    phi_n[[i, j-1]],
                    phi_n[[i, j+1]],
                ).abs();

                if residual > max_residual {
                    max_residual = residual;
                }
            }
        }

        max_residual
    }
}

/// Thomas algorithm for tridiagonal systems
///
/// Solves:
///
///     a_j x_{j-1} + b_j x_j + c_j x_{j+1} = d_j
///
/// Used to compute all corrections C(i,j) along a vertical line.
fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {

    let n = d.len();

    let mut c_star = vec![0.0; n];
    let mut d_star = vec![0.0; n];
    let mut x = vec![0.0; n];

    // ----- Forward sweep -----
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_star[i - 1];

        if denom.abs() < 1e-14 {
            panic!("Thomas breakdown at i={}", i);
        }

        c_star[i] = if i < n - 1 { c[i] / denom } else { 0.0 };
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom;
    }

    // ----- Back substitution -----
    x[n - 1] = d_star[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }

    x
}