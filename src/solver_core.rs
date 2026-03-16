use ndarray::Array2;
use crate::config;
use crate::mesh::Node;
use crate::solver_utils;
use crate::solver_utils::Scheme;

pub fn solve(mesh: &Array2<Node>,
            config: &config::Config) 
            -> Array2<f64> {

    let n_max = config.n_max;

    // Initialize the phi field with the initial conditions
    let mut phi = initialize_phi_field(mesh, config); 

    // Create a residual writer to log the residual history during the iterations
    let mut residual_writer = solver_utils::ResidualWriter::new("job_files/residual_history.csv");

    // Solve for n_max iterations
    println!("\nStarting solver for {} iterations...\n", n_max);
    for iter in 1..=n_max {

        input_boundary_conditions(mesh, &mut phi, config);

        let L_phi_n = calc_residual_operator(mesh, &phi, config);

        let residual_error = solver_utils::compute_residual_error(&L_phi_n);

        residual_writer.write(iter, residual_error);

        phi = update_solution(mesh, &phi, &L_phi_n, config);

        if iter % 100 == 0 || iter == 1 {
            println!("Iteration: {}/{}, Residual Error: {:.6e}", iter, n_max, residual_error);
        }
    }
   phi 
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

// This function calculates the phi field at each interior node of the mesh using the finite difference approximation of the Laplacian operator. 
// It iterates over all interior nodes of the mesh and updates the phi value at each node based on
fn calc_residual_operator(mesh: &Array2<Node>, phi_n: &Array2<f64>, config: &config::Config) -> Array2<f64> {

    let (imax, jmax) = mesh.dim();
    let mut L_phi_n = Array2::<f64>::zeros((imax, jmax));

    // iterate only on interior nodes
    for i in 1..imax-2 {
        for j in 1..jmax-2 {

            L_phi_n[[i, j]] = calc_L_phi_ij(
                            mesh[[i, j]].x, mesh[[i, j]].y,
                            mesh[[i+1, j]].x, mesh[[i-1, j]].x,
                            mesh[[i, j+1]].y, mesh[[i, j-1]].y,
                            phi_n[[i, j]],
                            phi_n[[i+1, j]],
                            phi_n[[i-1, j]],
                            phi_n[[i, j-1]],
                            phi_n[[i, j+1]]
                        );
        }
    }

    L_phi_n
}

fn calc_L_phi_ij(x_i: f64, y_j: f64, x_ip1: f64, x_im1: f64, y_jp1: f64, y_jm1: f64, 
            phi_ij: f64, phi_ip1j: f64, phi_im1j: f64, phi_ijm1: f64, phi_ijp1: f64) -> f64
            {
                let L_phi_ij = 2.0 / (x_ip1 - x_im1) * ((phi_ip1j - phi_ij) / (x_ip1 - x_i) - (phi_ij - phi_im1j) / (x_i - x_im1)) +
                           2.0 / (y_jp1 - y_jm1) * ((phi_ijp1 - phi_ij) / (y_jp1 - y_j) - (phi_ij - phi_ijm1) / (y_j - y_jm1));

                return L_phi_ij;
            }

fn update_solution(mesh: &Array2<Node>, 
                    phi_n: &Array2<f64>, 
                    L_phi_n: &Array2<f64>, 
                    config: &config::Config) -> Array2<f64> {

    let scheme = config.scheme;
    let (imax, jmax) = mesh.dim();
    let mut phi_np1 = phi_n.clone();

    for i in 1..imax-1 {
        for j in 1..jmax-1 {
            let N_scheme = get_N_scheme(mesh, L_phi_n, i, j, &scheme);
            phi_np1[[i, j]] = phi_n[[i, j]] - L_phi_n[[i, j]] / N_scheme;
        }
    }
    phi_np1
}

fn get_N_scheme(mesh: &Array2<Node>, L_phi_n: &Array2<f64>, i: usize, j: usize, scheme: &Scheme) -> f64 {

    match scheme {
        Scheme::Jacobi => {
            -2.0 / ((mesh[[i+1, j]].x - mesh[[i-1, j]].x)/2.0).powi(2)
            -2.0 / ((mesh[[i, j+1]].y - mesh[[i, j-1]].y)/2.0).powi(2)
        }

        Scheme::GaussSeidel => {
            (L_phi_n[[i-1, j]] - 2.0) /  (mesh[[i, j]].x - mesh[[i-1, j]].x).powi(2) + 
            (L_phi_n[[i, j-1]] - 2.0) /  (mesh[[i, j]].y - mesh[[i, j-1]].y).powi(2)
        }
    }
}

