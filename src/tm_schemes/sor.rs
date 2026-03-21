use ndarray::Array2;
use crate::mesh::Node;
use crate::tm_schemes::TimeMarchingScheme;
use crate::solver_core::calc_L_phi_ij;

/// Successive Over-Relaxation (SOR) method.
///
/// This method extends Gauss-Seidel by introducing a relaxation factor (ω)
/// to accelerate convergence.
///
/// # Mathematical Formulation
///
/// The Gauss-Seidel correction is:
///
/// C_GS = - Lφ / N
///
/// SOR modifies this as:
///
/// φⁿ⁺¹ = φⁿ + ω C_GS
///
/// which is equivalent to:
///
/// φⁿ⁺¹ = φⁿ - ω (Lφ / N)
///
/// In this implementation, the relaxation is incorporated into the operator:
///
/// N_SOR = N / ω
///
/// so that:
///
/// C = - Lφ / N_SOR = - ω (Lφ / N)
///
/// # Relaxation Behavior
///
/// - ω = 1.0 → Gauss-Seidel
/// - 1 < ω < 2 → Over-relaxation (faster convergence)
/// - 0 < ω < 1 → Under-relaxation (more stable, slower)
///
/// # Characteristics
/// - Faster convergence than Gauss-Seidel (if ω is well chosen)
/// - Still sequential (depends on sweep order)
/// - Sensitive to choice of ω
///
/// # Notes
/// - Optimal ω depends on mesh and problem geometry
/// - Typical values: ω ≈ 1.5 – 1.9 for Laplace problems
pub struct SOR {
    /// Relaxation factor
    pub omega: f64,
}

impl TimeMarchingScheme for SOR {

    /// Performs one SOR iteration over the domain.
    ///
    /// # Arguments
    /// * `mesh` - Computational grid
    /// * `phi_n` - Potential field (updated in-place)
    ///
    /// # Returns
    /// * `f64` - Maximum residual in the domain
    ///
    /// # Algorithm
    /// 1. Loop over interior nodes
    /// 2. Compute residual Lφ
    /// 3. Apply relaxed correction
    /// 4. Update solution in-place
    ///
    /// # Key Idea
    /// The correction is amplified by ω, accelerating convergence.
    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64 {

        let (imax, jmax) = mesh.dim();

        // Optional: can be removed for performance
        let mut C_n = Array2::<f64>::zeros((imax, jmax));

        let mut max_residual = 0.0;

        for i in 1..imax-1 {
            for j in 1..jmax-1 {

                // Compute residual
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

                // Relaxed diagonal operator
                let N_scheme = N_sor(mesh, i, j, self.omega);

                // Compute correction (already includes ω)
                C_n[[i, j]] = - L_phi_n_ij / N_scheme;

                // In-place update
                phi_n[[i, j]] += C_n[[i, j]];

                // Track maximum residual
                if L_phi_n_ij.abs() > max_residual {
                    max_residual = L_phi_n_ij.abs();
                }
            }
        }

        max_residual
    }
}

/// Computes the relaxed diagonal operator for SOR.
///
/// # Arguments
/// * `mesh` - Computational grid
/// * `i`, `j` - Node indices
/// * `omega` - Relaxation factor
///
/// # Returns
/// * `f64` - Modified diagonal coefficient
///
/// # Notes
/// - Equivalent to scaling the Gauss-Seidel update by ω
/// - This formulation avoids explicitly multiplying the correction by ω
fn N_sor(mesh: &Array2<Node>, i: usize, j: usize, omega: f64) -> f64 {

    let dx = (mesh[[i+1, j]].x - mesh[[i-1, j]].x) / 2.0;
    let dy = (mesh[[i, j+1]].y - mesh[[i, j-1]].y) / 2.0;

    // Standard diagonal term scaled by 1/ω
    (-2.0 / dx.powi(2) - 2.0 / dy.powi(2)) * (1.0 / omega)
}