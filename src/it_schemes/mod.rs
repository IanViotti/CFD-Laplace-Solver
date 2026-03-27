pub mod jacobi;
pub mod gauss_seidel;
pub mod sor;
pub mod line_gauss_seidel;
pub mod slor;

use ndarray::Array2;
use crate::mesh::Node;

/// Trait defining a generic iterative scheme.
///
/// This trait provides a unified interface for all iterative methods
/// used to solve the discrete Laplace equation.
///
/// # Purpose
/// - Enables interchangeable numerical schemes (Jacobi, GS, SOR, Line-GS)
/// - Decouples solver logic from numerical method implementation
/// - Promotes extensibility and modular design
///
/// # Design Philosophy
/// Each scheme is responsible for:
/// - Computing the residual locally
/// - Updating the solution field
/// - Returning a convergence metric
///
/// The solver (`solver_core`) does not need to know:
/// - How the update is performed
/// - Whether the method is explicit, implicit, or semi-implicit
///
/// # Method Behavior
/// Implementations may differ in:
/// - Update strategy (explicit vs in-place)
/// - Use of neighboring values (old vs updated)
/// - Internal linear system solves (e.g., Line-GS)
///
/// # Arguments
/// * `mesh` - Computational grid (geometry and spacing)
/// * `phi_n` - Solution field (updated in-place)
///
/// # Returns
/// * `f64` - Maximum residual in the domain
///           (used for convergence monitoring)
///
/// # Notes
/// - Boundary conditions are handled outside this trait
/// - Only interior nodes are updated
/// - Residual definition is consistent across all schemes
///
/// # Implementations
/// - `Jacobi` → explicit, uses previous iteration only
/// - `GaussSeidel` → in-place, sequential updates
/// - `SOR` → relaxed Gauss-Seidel
/// - `LineGaussSeidel` → line-wise implicit solver
pub trait IterativeScheme {

    /// Performs one iteration (time-marching step).
    ///
    /// # Responsibilities
    /// - Compute local residuals
    /// - Update the solution field
    /// - Track and return convergence metric
    ///
    /// # Important
    /// - Must update `phi_n` in-place
    /// - Must not modify boundary conditions
    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64;
}