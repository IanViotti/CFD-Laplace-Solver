use ndarray::Array2;
use crate::{config, tm_schemes};
use crate::mesh::Node;
use crate::solver_utils;


//  Main function to solve the Laplace equation using the finite difference method. 
// This function calls all the other relevant functions in this scope and returns the final phi matrix.
pub fn solve(mesh: &Array2<Node>,
            config: &config::Config,
            tm_scheme: &dyn tm_schemes::TimeMarchingScheme
             ) 
            -> Array2<f64> {

    let n_max = config.n_max;

    // Initialize the phi field with the initial conditions
    let mut phi_n = initialize_phi_field(mesh, config); 

    // Create a residual writer to log the residual history during the iterations
    let mut residual_writer = solver_utils::ResidualWriter::new("job_files/solution_data/residual_history.csv");

    // Solve for n_max iterations
    println!("\nStarting solver for {} iterations...\n", n_max);
    for iter in 1..=n_max {

        // Input boundary condition on phi array
        input_boundary_conditions(mesh, &mut phi_n, config);

        // Calculate the residual operator L_phi_n for the current phi field.
        //let L_phi_n = calc_residual_operator(mesh, &phi_n);

        // Compute the maximum and average residual error for the current iteration using the L_phi_n operator.
        //let (max_residual, avg_residual) = solver_utils::compute_residual_error(&L_phi_n);

        // Update the solution for n+1 using n and the residual operator L_phi_n.
        let max_residual = tm_scheme.step(mesh, &mut phi_n); 
        
        // Register the residual error for this iteration and print progress every 100 iterations
        residual_writer.write(iter, max_residual);
        if iter % 100 == 0 || iter == 1 {
            println!("Iteration: {}/{}, Max Residual Error: {:.6e}", iter, n_max, max_residual);
        }

        // Update time
        //phi_n = phi_np1;
    }
   phi_n 
}

// This function initializes the phi field with a given initial value. 
// It iterates over all nodes in the mesh and assigns the initial value to each node in the phi field.
fn initialize_phi_field(mesh: &Array2<Node>,
                            config: &config::Config,
                            ) -> Array2<f64> {

    let u_inf: f64 = config.u_inf;

    let (imax, jmax) = mesh.dim();
    let mut phi = Array2::<f64>::zeros((imax, jmax));

    for ((i, j), node) in mesh.indexed_iter() {
        phi[[i,j]] = u_inf * node.x;
    }

    phi
}


fn input_boundary_conditions(mesh: &Array2<Node>, 
                                phi: &mut Array2<f64>, 
                                config: &config::Config) {

    let (imax, jmax) = mesh.dim();

    let ile = config.ILE; // Leading Edge Index
    let ite = config.ITE; // Trailing Edge Index
    let u_inf = config.u_inf; // Free stream velocity
    let t = config.t; // thickness of the airfoil

    // Iterate over all nodes in the mesh
    for i in 0..imax-1 {
        for j in 0..jmax-1 {

            if i == 0 || i == imax - 1 || j == jmax - 1 {
                // Outter boundary condition
                phi[[i, j]] = u_inf * mesh[[i, j]].x; 
            }
            else if j == 0 && (i < ile || i > ite) {
                // Symmetry boundary condition
                phi[[i, 0]] = phi[[i, 1]]; 
            }
            else if j == 0 && (i >= ile && i <= ite) {
                // Airfoil surface boundary condition
                let phi_y_32 = u_inf * (2.0 * t - 4.0 * t * mesh[[i, j]].x);     
                phi[[i, 0]] = phi[[i, 1]] - (mesh[[i, 1]].y - mesh[[i, 0]].y) * phi_y_32;
            }
        }
    }
}

pub fn calc_L_phi_ij(x_i: f64, y_j: f64, x_ip1: f64, x_im1: f64, y_jp1: f64, y_jm1: f64, 
            phi_ij: f64, phi_ip1j: f64, phi_im1j: f64, phi_ijm1: f64, phi_ijp1: f64) -> f64
            {
                let L_phi_ij = 2.0 / (x_ip1 - x_im1) * ((phi_ip1j - phi_ij) / (x_ip1 - x_i) - (phi_ij - phi_im1j) / (x_i - x_im1)) +
                           2.0 / (y_jp1 - y_jm1) * ((phi_ijp1 - phi_ij) / (y_jp1 - y_j) - (phi_ij - phi_ijm1) / (y_j - y_jm1));

                return L_phi_ij;
            }