mod mesh;
mod solver_core;
mod solver_utils;
mod config;
mod tm_schemes;

use crate::tm_schemes::{gauss_seidel::GaussSeidel, jacobi::Jacobi, sor::SOR};

/// Entry point of the CFD solver.
///
/// This program solves the 2D Laplace equation for the velocity potential
/// around a symmetric airfoil using a finite difference method.
///
/// The workflow includes:
/// 1. Defining simulation parameters
/// 2. Generating the computational mesh
/// 3. Solving the potential flow problem
/// 4. Post-processing (velocity and pressure coefficient)
/// 5. Saving results for visualization
///
/// # Workflow Overview
///
/// ```text
/// Config → Mesh → Solver → Velocity → Cp → Output
/// ```
///
/// # Notes
/// - The solver is modular and supports different iterative schemes
/// - Results are exported in CSV format for post-processing
/// - The airfoil is implicitly defined along the lower boundary (j = 0)
fn main() {

    // ===============================
    // Simulation Configuration
    // ===============================
    let config = config::Config {
        ILE: 10,   // Leading edge index
        ITE: 30,   // Trailing edge index
        IMAX: 41,  // Number of nodes in x-direction
        JMAX: 12,  // Number of nodes in y-direction
        XSF: 1.25, // Stretching factor in x-direction
        YSF: 1.25, // Stretching factor in y-direction
        u_inf: 1.0, // Freestream velocity (U∞)
        t: 0.05,   // Airfoil thickness parameter
        n_max: 1000, // Maximum number of solver iterations
    };

    // ===============================
    // Time-Marching Scheme Selection
    // ===============================
    let tm_scheme = SOR { omega: 1.8 };
    // Alternative schemes:
    // let tm_scheme = Jacobi;
    // let tm_scheme = GaussSeidel;

    let mesh_file_name = "job_files/mesh/mesh.csv";

    println!("\nStarting Laplace Solver");
    println!("Simulation configuration: {:#?}", config);

    // ===============================
    // Mesh Generation
    // ===============================
    let mesh = mesh::build_cartesian_mesh(config);

    // Save mesh for visualization/debugging
    mesh::save_mesh(mesh_file_name, &mesh);

    // ===============================
    // Solve Laplace Equation
    // ===============================
    let phi = solver_core::solve(&mesh, &config, &tm_scheme);

    // ===============================
    // Post-Processing
    // ===============================

    // Compute velocity field (u = ∂φ/∂x, v = ∂φ/∂y)
    let velocity_field =
        solver_utils::calc_velocity_field(&mesh, &phi, config);

    // Compute pressure coefficient (Bernoulli)
    let cp =
        solver_utils::calc_cp(&velocity_field, config);

    // ===============================
    // Output Results
    // ===============================

    // Save full solution (mesh + fields)
    solver_utils::save_solution(
        "job_files/solution_data/solution.csv",
        &mesh,
        &phi,
        &cp,
        &velocity_field,
    );

    // Save individual fields (useful for debugging/plotting)
    solver_utils::save_field_matrix(
        "job_files/solution_data/phi_matrix.csv",
        &phi
    );

    solver_utils::save_field_matrix(
        "job_files/solution_data/cp_matrix.csv",
        &cp
    );

    solver_utils::save_field_matrix(
        "job_files/solution_data/u_matrix.csv",
        &velocity_field.map(|v| v.u)
    );

    solver_utils::save_field_matrix(
        "job_files/solution_data/v_matrix.csv",
        &velocity_field.map(|v| v.v)
    );

    // ===============================
    // Airfoil Cp Distribution
    // ===============================
    let airfoil_cp =
        solver_utils::airfoil_cp(&velocity_field, &config);

    solver_utils::save_airfoil_cp(
        "job_files/solution_data/airfoil_cp.csv",
        &mesh,
        &airfoil_cp,
        &config
    );
}