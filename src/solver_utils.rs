use ndarray::Array2;
use ndarray::Array1;
use crate::config;
use crate::mesh::Node;
use std::fs::File;
use std::io::Write;


/// Structure representing the velocity vector at a grid point.
///
/// # Fields
/// * `u` - Velocity component in the x-direction
/// * `v` - Velocity component in the y-direction
#[derive(Clone, Copy, Debug)]
pub struct U {
    pub u: f64,
    pub v: f64,
}

/// Utility for writing residual history to a CSV file.
///
/// This is used to track convergence during iterative solution.
///
/// # Output format
/// CSV file with columns:
/// - iteration number
/// - maximum residual
pub struct ResidualWriter {
    file: File,
}

impl ResidualWriter {

    /// Creates a new residual writer and initializes the output file.
    ///
    /// # Arguments
    /// * `path` - Path to the output CSV file
    ///
    /// # Notes
    /// - Existing files are overwritten
    /// - A header row is written automatically
    pub fn new(path: &str) -> Self {
        let mut file = File::create(path).unwrap();
        writeln!(file, "iter,max").unwrap();

        Self { file }
    }

    /// Writes one iteration entry to the residual history file.
    ///
    /// # Arguments
    /// * `iter` - Iteration number
    /// * `max_residual` - Maximum residual at this iteration
    pub fn write(&mut self, iter: usize, max_residual: f64) {
        writeln!(self.file, "{},{}", iter, max_residual).unwrap();
    }
}

/// Computes the velocity field from the potential field using finite differences.
///
/// The velocity components are obtained as:
/// - u = ∂φ/∂x
/// - v = ∂φ/∂y
///
/// # Arguments
/// * `mesh` - Computational grid
/// * `phi` - Potential field
/// * `config` - Simulation parameters (freestream velocity)
///
/// # Returns
/// * `Array2<U>` - Velocity field at each grid node
///
/// # Notes
/// - Central differences are used in the interior
/// - One-sided approximations are used at boundaries
/// - Far-field boundaries are set to freestream values
pub fn calc_velocity_field(
    mesh: &Array2<Node>,
    phi: &Array2<f64>,
    config: config::Config,
) -> Array2<U> {

    let u_inf = config.u_inf;
    let (imax, jmax) = phi.dim();

    let mut velocity = Array2::<U>::from_elem(
        (imax, jmax),
        U { u: 0.0, v: 0.0 }
    );

    println!("\nCalculating velocity field and pressure coefficient...");

    for i in 0..imax {
        for j in 0..jmax {

            let u: f64;
            let v: f64;

            if i == 0 || i == imax - 1 || j == jmax - 1 {
                // Far-field boundary condition
                u = u_inf;
                v = 0.0;

            } else if j == 0 {
                // Symmetry / airfoil surface approximation
                u = (phi[[i+1, j]] - phi[[i-1, j]]) /
                    (mesh[[i+1, j]].x - mesh[[i-1, j]].x);

                v = (phi[[i, j+1]] - phi[[i, j]]) /
                    (mesh[[i, j+1]].y - mesh[[i, j]].y);

            } else {
                // Interior nodes (central differences)
                let dx = mesh[[i+1, j]].x - mesh[[i-1, j]].x;
                let dy = mesh[[i, j+1]].y - mesh[[i, j-1]].y;

                u = (phi[[i+1, j]] - phi[[i-1, j]]) / dx;
                v = (phi[[i, j+1]] - phi[[i, j-1]]) / dy;
            }

            velocity[[i, j]] = U { u, v };
        }
    }

    velocity
}

/// Computes the pressure coefficient field from the velocity field.
///
/// The pressure coefficient is defined as:
///
/// Cp = 1 - (U² / U∞²)
///
/// # Arguments
/// * `velocity` - Velocity field
/// * `config` - Simulation parameters (freestream velocity)
///
/// # Returns
/// * `Array2<f64>` - Pressure coefficient field
///
/// # Notes
/// - Assumes incompressible, irrotational flow
/// - Based on Bernoulli equation
pub fn calc_cp(
    velocity: &Array2<U>,
    config: config::Config
) -> Array2<f64> {

    let u_inf = config.u_inf;
    let mut cp = Array2::<f64>::zeros(velocity.dim());

    println!("\nCalculating pressure coefficient...");

    for ((i, j), velocity_ij) in velocity.indexed_iter() {
        cp[[i, j]] =
            1.0 - (velocity_ij.u.powi(2) + velocity_ij.v.powi(2))
            / (u_inf.powi(2));
    }

    cp
}

/// Saves the full solution (mesh + fields) to a CSV file.
///
/// # Output columns
/// x, y, φ, u, v, Cp
///
/// # Arguments
/// * `file_name` - Output file path
/// * `mesh` - Computational grid
/// * `phi` - Potential field
/// * `cp` - Pressure coefficient field
/// * `velocity_field` - Velocity field
///
/// # Notes
/// - Each row corresponds to one grid point
/// - Suitable for post-processing and visualization tools
pub fn save_solution(
    file_name: &str,
    mesh: &Array2<Node>,
    phi: &Array2<f64>,
    cp: &Array2<f64>,
    velocity_field: &Array2<U>,
) {

    let mut file = File::create(file_name).unwrap();

    writeln!(file, "x,y,phi,u,v,cp").unwrap();

    for ((i, j), phi) in phi.indexed_iter() {

        let node = &mesh[[i, j]];

        writeln!(
            file,
            "{},{},{},{},{},{}",
            node.x,
            node.y,
            phi,
            velocity_field[[i, j]].u,
            velocity_field[[i, j]].v,
            cp[[i, j]]
        ).unwrap();
    }

    println!("\nSolution saved to '{}'.\n", file_name);
}

/// Saves a scalar field matrix to a CSV file.
///
/// # Arguments
/// * `file_name` - Output file path
/// * `field` - Scalar field to be saved
///
/// # Notes
/// - Data is written row by row (j-direction)
/// - Useful for debugging or plotting in external tools
pub fn save_field_matrix(file_name: &str, field: &Array2<f64>) {

    let mut file = File::create(file_name).unwrap();

    let (imax, jmax) = field.dim();

    for j in 0..jmax {
        for i in 0..imax {
            write!(file, "{:.6}", field[[i, j]]).unwrap();

            if i < imax - 1 {
                write!(file, ",").unwrap();
            }
        }
        writeln!(file).unwrap();
    }

    println!("Field saved to '{}'", file_name);
}

/// Computes residual statistics from the residual field.
///
/// # Arguments
/// * `L_phi` - Residual field (discrete Laplacian)
///
/// # Returns
/// * `(max, mean)` - Maximum and mean absolute residual
///
/// # Notes
/// - Used for convergence monitoring
/// - Maximum residual is typically used as stopping criterion
pub fn compute_residual_error(L_phi: &Array2<f64>) -> (f64, f64) {

    let (imax, jmax) = L_phi.dim();

    let mut max_residual = 0.0;
    let mut sum_residual = 0.0;
    let mut count = 0;

    for i in 0..imax {
        for j in 0..jmax {

            let abs_residual = L_phi[[i, j]].abs();

            if abs_residual > max_residual {
                max_residual = abs_residual;
            }

            sum_residual += abs_residual;
            count += 1;
        }
    }

    let mean_residual = sum_residual / count as f64;

    (max_residual, mean_residual)
}

/// Computes the pressure coefficient distribution along the airfoil surface.
///
/// # Arguments
/// * `velocity` - Velocity field
/// * `config` - Contains airfoil index range and freestream velocity
///
/// # Returns
/// * `Array1<f64>` - Cp distribution along the airfoil chord
///
/// # Notes
/// - Evaluated at the midpoint between j=0 and j=1 (surface location)
/// - Velocity is averaged between these two points
/// - Corresponds to Cp(x/c) curve required in the project
pub fn airfoil_cp(
    velocity: &Array2<U>,
    config: &config::Config
) -> Array1<f64> {

    let ile = config.ILE;
    let ite = config.ITE;
    let u_inf = config.u_inf;

    let n_points = ite - ile + 1;
    let mut cp = Array1::<f64>::zeros(n_points);

    for (k, i) in (ile..=ite).enumerate() {

        let u = (velocity[[i, 1]].u + velocity[[i, 0]].u) / 2.0;
        let v = (velocity[[i, 1]].v + velocity[[i, 0]].v) / 2.0;

        let V2 = u.powi(2) + v.powi(2);

        cp[k] = 1.0 - V2 / (u_inf.powi(2));
    }

    cp
}

/// Saves the airfoil pressure coefficient distribution to a CSV file.
///
/// # Output columns
/// x, Cp
///
/// # Arguments
/// * `file_name` - Output file path
/// * `mesh` - Computational grid
/// * `cp` - Airfoil Cp distribution
/// * `config` - Contains airfoil index range
///
/// # Notes
/// - Uses j=0 line as airfoil surface
/// - Output is directly usable for Cp vs x/c plots
pub fn save_airfoil_cp(
    file_name: &str,
    mesh: &Array2<Node>,
    cp: &Array1<f64>,
    config: &config::Config
) {

    let mut file = File::create(file_name).unwrap();

    let ile = config.ILE;
    let ite = config.ITE;

    writeln!(file, "x,cp").unwrap();

    for (k, i) in (ile..=ite).enumerate() {

        let x = mesh[[i, 0]].x;

        writeln!(file, "{:.6},{:.6}", x, cp[k]).unwrap();
    }

    println!("Airfoil Cp saved to '{}'", file_name);
}

/// Thomas algorithm
///
/// Solves a tridiagonal linear system:
///
///     a_j x_{j-1} + b_j x_j + c_j x_{j+1} = d_j
///
/// Used here to solve for all corrections C(i,j) along
/// a vertical line in the Line Gauss-Seidel method.
pub fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {

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

// Initialize solver directory
pub fn init_solver_directory(jobname: &str) {
    std::fs::create_dir_all(&format!("job_files/{}/solution_data", jobname)).unwrap();
    std::fs::create_dir_all(&format!("job_files/{}/post_proc_result", jobname)).unwrap();
    std::fs::create_dir_all(&format!("job_files/{}/mesh", jobname)).unwrap();
}