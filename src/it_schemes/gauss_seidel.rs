use ndarray::Array2;
use crate::mesh::Node;
use crate::it_schemes::IterativeScheme;
use crate::solver_core::calc_L_phi_ij;

/// Gauss-Seidel iterative method for solving the discrete Laplace equation.
///
/// This method improves upon Jacobi by updating the solution *in-place*,
/// meaning newly computed values are immediately used within the same iteration.
///
/// # Mathematical Formulation
///
/// φⁿ⁺¹ = φⁿ + Cⁿ
///
/// where:
///
/// Cⁿ = - Lφ / N
///
/// Unlike Jacobi:
/// - Updated values φⁿ⁺¹ are used as soon as they are available
/// - This effectively introduces implicit behavior along the sweep direction
///
/// # Characteristics
/// - Sequential (order-dependent)
/// - Faster convergence than Jacobi
/// - Low memory overhead (no temporary field required)
/// - Not trivially parallelizable
///
/// # Notes
/// - Boundary conditions are handled outside this function
/// - The update follows a lexicographic sweep (i → j)
/// - Convergence depends on sweep ordering
pub struct GaussSeidel;

impl IterativeScheme for GaussSeidel {

    /// Performs one Gauss-Seidel iteration over the domain.
    ///
    /// # Arguments
    /// * `mesh` - Computational grid
    /// * `phi_n` - Potential field (updated in-place)
    ///
    /// # Returns
    /// * `f64` - Maximum residual (used for convergence monitoring)
    ///
    /// # Algorithm
    /// 1. Loop over interior nodes
    /// 2. Compute residual Lφ using current (partially updated) field
    /// 3. Compute correction C = -Lφ / N
    /// 4. Immediately update φ at the current node
    ///
    /// # Key Difference from Jacobi
    /// - No temporary array is used
    /// - Updates propagate instantly through the domain
    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64 {

        let (imax, jmax) = mesh.dim();

        let mut max_residual = 0.0;

        // Sweep through interior nodes
        for i in 1..imax-1 {
            for j in 1..jmax-1 {

                // Compute residual using updated neighbors when available
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

                // Compute dx dy
                let dx = (mesh[[i+1, j]].x - mesh[[i-1, j]].x) / 2.0;
                let dy = (mesh[[i, j+1]].y - mesh[[i, j-1]].y) / 2.0;

                // Compute correction at poin i,j
                let N = -2.0 / dx.powi(2) - 2.0 / dy.powi(2); 

                // Compute correction
                let C_ij = - L_phi_n_ij / N ;

                // Immediate update (in-place)
                phi_n[[i, j]] += C_ij;

                // Track maximum residual
                if L_phi_n_ij.abs() > max_residual {
                    max_residual = L_phi_n_ij.abs();
                }
            }
        }

        max_residual
    }
}
