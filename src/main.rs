mod mesh;
mod solver_core;
mod solver_utils;
mod config;

fn main() {

    let config = config::Config {
        ILE: 10, // Leading Edge Index
        ITE: 30, // Trailing Edge Index
        IMAX: 41, // Max points in X direction
        JMAX: 12, // Max points in Y direction
        XSF: 1.25, // Stretching factor in x 
        YSF: 1.25, // Stretching factor in y
        u_inf: 1.0, // Free stream velocity
        t: 0.05, // thickness of the airfoil
        n_max: 1000, // Max number of iterations
        scheme: solver_utils::Scheme::Jacobi, // Numerical scheme to use
    };
    
    let file_name = "job_files/mesh.csv";

    println!("\nStarting Laplace Solver");
    println!("with simulation configuration: {:#?}", config);

    // Build mesh
    let mesh = mesh::build_cartesian_mesh(config);

    // Save mesh to CSV
    mesh::save_mesh(file_name, &mesh);

    // Solve for phi field
    let phi = solver_core::solve(&mesh, &config);

    // Calculate velocity field
    let velocity_field = solver_utils::calc_velocity_field(&mesh, &phi, config);
    // Calculate pressure coefficient
    let cp = solver_utils::calc_cp(&velocity_field, config);

    // Save solution to CSV
    solver_utils::save_solution("job_files/solution.csv", &mesh, &phi, &cp, &velocity_field);

}