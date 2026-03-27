use ndarray::Array2;
use crate::mesh::Node;
use crate::solver_core::calc_L_phi_ij;
use crate::it_schemes::IterativeScheme;

/// LineGaussSeidel
///
/// Implementation of the Line Gauss-Seidel (LGS) iterative method
/// for solving the Laplace equation using the delta formulation:
///
///     N C(i,j)^n + Lφ(i,j)^n = 0
///
/// where:
///     C(i,j)^n = Δφ(i,j)^n = φ(i,j)^(n+1) − φ(i,j)^n
///
/// In the LGS method:
/// - The discretization in the y-direction is treated implicitly;
/// - The discretization in the x-direction is treated explicitly;
/// - For each vertical line (fixed i), a tridiagonal system is solved
///   simultaneously for all corrections C(i,j).
pub struct LineGaussSeidel;

impl IterativeScheme for LineGaussSeidel {

    /// Performs one full Line Gauss-Seidel iteration over the domain.
    ///
    /// For each vertical line i:
    /// 1) Assemble a tridiagonal linear system in y for C(i,j);
    /// 2) Use the full residual Lφ evaluated at iteration n;
    /// 3) Solve the tridiagonal system (Thomas algorithm);
    /// 4) Update φ using the computed corrections.
    ///
    /// The maximum correction magnitude is returned as a convergence metric.
    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64 {

        let (imax, jmax) = mesh.dim();
        let mut max_residual = 0.0;

        // Number of interior points in y (size of tridiagonal system)
        let n = jmax - 2;

        // Correction vector along the current vertical line
        let mut C = vec![0.0; n];

        // Tridiagonal matrix coefficients
        let mut a = vec![0.0; n]; // sub-diagonal (j-1)
        let mut b = vec![0.0; n]; // main diagonal (j)
        let mut c = vec![0.0; n]; // super-diagonal (j+1)
        let mut d = vec![0.0; n]; // right-hand side

        // Sweep through vertical lines (LGS characteristic)
        for i in 1..imax-1 {

            // Assemble tridiagonal system for this vertical line
            for j in 1..jmax-1 {
                let jj = j - 1;

                // ----- Implicit contribution in y (δ̃yy operator applied to C) -----
                a[jj] = 1.0 /
                    (((mesh[[i, j+1]].y - mesh[[i, j-1]].y) / 2.0)
                    * (mesh[[i, j]].y - mesh[[i, j-1]].y));

                c[jj] = 1.0 /
                    (((mesh[[i, j+1]].y - mesh[[i, j-1]].y) / 2.0)
                    * (mesh[[i, j+1]].y - mesh[[i, j]].y));

                // ----- Explicit contribution in x included in the main diagonal -----
                b[jj] = -2.0 /
                    ((mesh[[i+1, j]].x - mesh[[i-1, j]].x) / 2.0).powi(2)
                    - (a[jj] + c[jj]);

                // ----- Compute full residual Lφ at iteration n -----
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

                // ----- Right-hand side of the linear system -----
                // Includes:
                // 1) The complete residual
                // 2) Gauss-Seidel contribution from the previous line (i-1)
                d[jj] = - L_phi_n_ij
                        - C[jj] /
                          ((mesh[[i+1, j]].x - mesh[[i-1, j]].x) / 2.0).powi(2);
            }

            // Solve tridiagonal system to obtain corrections C(i,j)
            C = thomas(&a, &b, &c, &d);

            // ----- Update solution using delta formulation -----
            for j in 1..jmax-1 {
                let jj = j - 1;
                phi_n[[i, j]] += C[jj];

                // Track maximum correction magnitude
                let residual = C[jj].abs();
                if residual > max_residual {
                    max_residual = residual;
                }
            }
        }

        // ----- Final residual evaluation over entire domain -----
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

/// Thomas algorithm
///
/// Solves a tridiagonal linear system:
///
///     a_j x_{j-1} + b_j x_j + c_j x_{j+1} = d_j
///
/// Used here to solve for all corrections C(i,j) along
/// a vertical line in the Line Gauss-Seidel method.
fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {

    let n = d.len();

    let mut c_star = vec![0.0; n];
    let mut d_star = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Forward sweep
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

    // Back substitution
    x[n - 1] = d_star[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }

    x
}