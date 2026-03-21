use ndarray::Array2;
use crate::mesh::Node;
use crate::tm_schemes::TimeMarchingScheme;
use crate::solver_core::calc_L_phi_ij;

/// Jacobi iterative method for solving the discrete Laplace equation.
///
/// This method updates the solution explicitly using values from the
/// previous iteration only. It is the simplest relaxation scheme and
/// serves as a baseline for more advanced methods.
///
/// # Mathematical Formulation
///
/// The update is based on:
///
/// φⁿ⁺¹ = φⁿ + Cⁿ
///
/// where the correction term is:
///
/// Cⁿ = - Lφⁿ / N
///
/// - Lφⁿ: discrete Laplacian (residual)
/// - N: diagonal approximation of the operator
///
/// # Characteristics
/// - Fully explicit method
/// - Uses only values from iteration n
/// - Easy to implement but slow convergence
/// - Highly parallelizable
///
/// # Notes
/// - Boundary nodes are not updated (handled separately)
/// - A temporary array is required to store φⁿ⁺¹
pub struct Jacobi;

impl TimeMarchingScheme for Jacobi {

    /// Performs one Jacobi iteration over the entire domain.
    ///
    /// # Arguments
    /// * `mesh` - Computational grid
    /// * `phi_n` - Potential field (updated in-place after iteration)
    ///
    /// # Returns
    /// * `f64` - Maximum residual in the domain (convergence metric)
    ///
    /// # Algorithm
    /// 1. Compute residual Lφ at each interior node
    /// 2. Compute correction term C = -Lφ / N
    /// 3. Update φⁿ⁺¹ using φⁿ (stored separately)
    /// 4. Replace φⁿ with φⁿ⁺¹ after full sweep
    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64 {

        let (imax, jmax) = mesh.dim();

        // Correction field
        let mut C_n = Array2::<f64>::zeros((imax, jmax));

        // Temporary storage for next iteration
        let mut phi_np1 = phi_n.clone();

        let mut max_residual = 0.0;

        // Loop over interior nodes
        for i in 1..imax-1 {
            for j in 1..jmax-1 {

                // Compute discrete Laplacian (residual)
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

                // Diagonal operator approximation
                let N_scheme = N_jacobi(mesh, i, j);

                // Compute correction
                C_n[[i, j]] = - L_phi_n_ij / N_scheme;

                // Update using only previous iteration values
                phi_np1[[i, j]] = phi_n[[i, j]] + C_n[[i, j]];

                // Track maximum residual
                if L_phi_n_ij.abs() > max_residual {
                    max_residual = L_phi_n_ij.abs();
                }
            }
        }

        // Update solution after full sweep
        phi_n.assign(&phi_np1);

        max_residual
    }
}

/// Computes the diagonal approximation of the discrete operator.
///
/// This corresponds to the central coefficient of the Laplacian
/// in finite difference form.
///
/// # Arguments
/// * `mesh` - Computational grid
/// * `i`, `j` - Node indices
///
/// # Returns
/// * `f64` - Diagonal coefficient N
///
/// # Notes
/// - Based on second-order central differences
/// - Accounts for non-uniform grid spacing
fn N_jacobi(mesh: &Array2<Node>, i: usize, j: usize) -> f64 {

    let dx = (mesh[[i+1, j]].x - mesh[[i-1, j]].x) / 2.0;
    let dy = (mesh[[i, j+1]].y - mesh[[i, j-1]].y) / 2.0;

    -2.0 / dx.powi(2) - 2.0 / dy.powi(2)
}