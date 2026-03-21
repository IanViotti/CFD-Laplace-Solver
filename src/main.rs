mod mesh;
mod solver_core;
mod solver_utils;
mod config;
mod tm_schemes;

use crate::tm_schemes::{gauss_seidel::GaussSeidel, jacobi::Jacobi, sor::SOR};

fn main() {

    let config = config::Config {
        ILE: 10, // Leading Edge Index
        ITE: 30, // Trailing Edge Index
        IMAX: 41, // Max points in X direction
        JMAX: 12, // Max points in Y direction
        XSF: 1.25, // Stretching factor in x 
        YSF: 1.25, // Stretching factor in y
        u_inf: 1.0, // Free stream velocity
        t: 0.05, // Thickness of the airfoil
        n_max: 1000, // Max number of iterations
    };

    let tm_scheme = SOR { omega: 1.8}; // Successive Over-Relaxation with relaxation factor 

    let mesh_file_name = "job_files/mesh/mesh.csv";

    println!("\nStarting Laplace Solver");
    println!("with simulation configuration: {:#?}", config);

    // Build mesh
    let mesh = mesh::build_cartesian_mesh(config);

    // Save mesh to CSV
    mesh::save_mesh(mesh_file_name, &mesh);

    // Solve for phi field
    let phi = solver_core::solve(&mesh, &config, &tm_scheme);

    // Calculate velocity field
    let velocity_field = solver_utils::calc_velocity_field(&mesh, &phi, config);
    // Calculate pressure coefficient
    let cp = solver_utils::calc_cp(&velocity_field, config);

    // Save solution to CSV
    solver_utils::save_solution("job_files/solution_data/solution.csv", &mesh, &phi, &cp, &velocity_field);

    solver_utils::save_field_matrix("job_files/solution_data/phi_matrix.csv", &phi);
    solver_utils::save_field_matrix("job_files/solution_data/cp_matrix.csv", &cp);
    solver_utils::save_field_matrix("job_files/solution_data/u_matrix.csv", &velocity_field.map(|v| v.u));
    solver_utils::save_field_matrix("job_files/solution_data/v_matrix.csv", &velocity_field.map(|v| v.v));

    let airfoil_cp = solver_utils::airfoil_cp(&velocity_field, &config);
    
    solver_utils::save_airfoil_cp("job_files/solution_data/airfoil_cp.csv", &mesh, &airfoil_cp, &config);  

}