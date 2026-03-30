use ndarray::Array2;
use crate::{config, it_schemes};
use crate::mesh::Node;
use crate::solver_utils;


/// Solves the 2D Laplace equation for the velocity potential using
/// an iterative relaxation method.
///
/// This is the main driver of the CFD solver. It orchestrates:
/// - Initialization of the potential field
/// - Application of boundary conditions
/// - Iterative updates using a chosen time-marching (relaxation) scheme
/// - Residual monitoring and logging
///
/// # Arguments
/// * `mesh` - Structured computational grid containing node coordinates
/// * `config` - Simulation parameters (flow conditions, iteration count, geometry)
/// * `tm_scheme` - Time-marching scheme (e.g., Jacobi, Gauss-Seidel, SOR, Line-GS)
///
/// # Returns
/// * `Array2<f64>` - Final converged potential field φ
///
/// # Notes
/// - The solver performs a fixed number of iterations (`n_max`)
/// - Convergence is monitored via the maximum residual
/// - Boundary conditions are enforced at every iteration
pub fn solve(
    mesh: &Array2<Node>,
    config: &config::Config,
    it_scheme: &dyn it_schemes::IterativeScheme,
    jobname: &str,
) -> Array2<f64> {

    let n_max = config.n_max;

    // Initialize the potential field with freestream condition
    let mut phi_n = initialize_phi_field(mesh, config);

    // Residual logging utility
    let mut residual_writer =
        solver_utils::ResidualWriter::new(&format!("job_files/{}/solution_data/residual_history.csv", jobname));

    println!("\nStarting solver for {} iterations...\n", n_max);

    for iter in 1..=n_max {

        // Enforce boundary conditions before each iteration
        input_boundary_conditions(mesh, &mut phi_n, config);

        // Perform one iteration of the selected scheme
        let max_residual = it_scheme.step(mesh, &mut phi_n);

        // Log convergence history
        residual_writer.write(iter, max_residual);

        if iter % 100 == 0 || iter == 1 {
            println!(
                "Iteration: {}/{}, Max Residual Error: {:.6e}",
                iter, n_max, max_residual
            );
        }
    }

    phi_n
}


/// Initializes the potential field φ with the freestream solution.
///
/// The initial condition corresponds to a uniform flow:
/// φ = U∞ * x
///
/// # Arguments
/// * `mesh` - Computational grid
/// * `config` - Simulation parameters containing freestream velocity
///
/// # Returns
/// * `Array2<f64>` - Initialized potential field
///
/// # Notes
/// - This serves both as an initial guess and as the far-field boundary condition
fn initialize_phi_field(
    mesh: &Array2<Node>,
    config: &config::Config,
) -> Array2<f64> {

    let u_inf = config.u_inf;

    let (imax, jmax) = mesh.dim();
    let mut phi = Array2::<f64>::zeros((imax, jmax));

    for ((i, j), node) in mesh.indexed_iter() {
        phi[[i, j]] = u_inf * node.x;
    }

    phi
}


/// Applies boundary conditions to the potential field φ.
///
/// The following boundary conditions are enforced:
///
/// 1. **Far-field (outer boundaries):**
///    φ = U∞ * x
///
/// 2. **Symmetry condition (y = 0 outside airfoil):**
///    ∂φ/∂y = 0 → φ(i,0) = φ(i,1)
///
/// 3. **Airfoil surface (y = 0 over the chord):**
///    ∂φ/∂y = U∞ * dy/dx
///    Implemented using a finite difference approximation
///
/// # Arguments
/// * `mesh` - Computational grid
/// * `phi` - Potential field (modified in-place)
/// * `config` - Contains geometry and flow parameters
///
/// # Notes
/// - The airfoil is represented implicitly along the line j = 0
/// - The slope dy/dx corresponds to a biconvex airfoil:
///   y = 2t x(1 - x)
fn input_boundary_conditions(
    mesh: &Array2<Node>,
    phi: &mut Array2<f64>,
    config: &config::Config,
) {

    let (imax, jmax) = mesh.dim();

    let ile = config.ILE; // Leading edge index
    let ite = config.ITE; // Trailing edge index
    let u_inf = config.u_inf;
    let t = config.t;

    for i in 0..imax-1 {
        for j in 0..jmax-1 {

            if i == 0 || i == imax - 1 || j == jmax - 1 {
                // Far-field boundary condition
                phi[[i, j]] = u_inf * mesh[[i, j]].x;

            } else if j == 0 && (i < ile || i > ite) {
                // Symmetry condition (∂φ/∂y = 0)
                phi[[i, 0]] = phi[[i, 1]];

            } else if j == 0 && (i >= ile && i <= ite) {
                // Airfoil boundary condition (tangency condition)

                // dy/dx for y = 2t x(1-x)
                let dYdx = 2.0 * t - 4.0 * t * mesh[[i, j]].x;

                let phi_y = u_inf * dYdx;

                phi[[i, 0]] =
                    phi[[i, 1]]
                    - (mesh[[i, 1]].y - mesh[[i, 0]].y) * phi_y;
            }
        }
    }
}


/// Computes the discrete Laplace operator (residual) at a given grid point.
///
/// This corresponds to the finite-difference approximation of:
///
/// ∇²φ = φ_xx + φ_yy
///
/// using a non-uniform grid formulation.
///
/// # Arguments
/// * `x_i`, `y_j` - Coordinates of the current node
/// * `x_ip1`, `x_im1` - Neighboring x-coordinates
/// * `y_jp1`, `y_jm1` - Neighboring y-coordinates
/// * `phi_ij` - Value at (i,j)
/// * `phi_ip1j`, `phi_im1j` - Neighbors in x-direction
/// * `phi_ijp1`, `phi_ijm1` - Neighbors in y-direction
///
/// # Returns
/// * `f64` - Residual value Lφ(i,j)
///
/// # Notes
/// - This operator measures how well the discrete Laplace equation is satisfied
/// - It is used as a convergence metric in iterative schemes
pub fn calc_L_phi_ij(
    x_i: f64,
    y_j: f64,
    x_ip1: f64,
    x_im1: f64,
    y_jp1: f64,
    y_jm1: f64,
    phi_ij: f64,
    phi_ip1j: f64,
    phi_im1j: f64,
    phi_ijm1: f64,
    phi_ijp1: f64,
) -> f64 {

    let term_x =
        2.0 / (x_ip1 - x_im1) *
        ((phi_ip1j - phi_ij) / (x_ip1 - x_i)
       - (phi_ij - phi_im1j) / (x_i - x_im1));

    let term_y =
        2.0 / (y_jp1 - y_jm1) *
        ((phi_ijp1 - phi_ij) / (y_jp1 - y_j)
       - (phi_ij - phi_ijm1) / (y_j - y_jm1));

    term_x + term_y
}