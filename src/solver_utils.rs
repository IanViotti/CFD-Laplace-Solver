use ndarray::Array2;
use crate::config;
use crate::mesh::Node;
use std::fs::File;
use std::io::Write;


#[derive(Clone, Copy, Debug)]
pub enum Scheme {
    Jacobi,
    GaussSeidel,
}

#[derive(Clone, Copy, Debug)]
pub struct U {
    pub u: f64,
    pub v: f64,
}

pub struct ResidualWriter {
    file: File,
}

impl ResidualWriter {

    pub fn new(path: &str) -> Self {
        let mut file = File::create(path).unwrap();
        writeln!(file, "iter,max,average").unwrap();

        Self { file }
    }

    pub fn write(&mut self, iter: usize, max_residual: f64, avg_residual: f64) {
        writeln!(self.file, "{},{},{}", iter, max_residual, avg_residual).unwrap();
    }
}

// Calculate the velocity field from the phi field using central differences
pub fn calc_velocity_field(
    mesh: &Array2<Node>,
    phi: &Array2<f64>,
    config: config::Config)
    -> Array2<U> {

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
                // Outter boundary condition
                u = u_inf; 
                v = 0.0;
            }
            else if j == 0 {
                // Symmetry boundary condition
                u = (phi[[i+1, j]] - phi[[i-1, j]]) / (mesh[[i+1, j]].x - mesh[[i-1, j]].x); 
                v = (phi[[i, j+1]] - phi[[i, j]]) / (mesh[[i, j+1]].y - mesh[[i, j]].y); // Checar derivada
            }
            else {
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

// Calculate the pressure coefficient from the velocity field
pub fn calc_cp(velocity: &Array2<U>,
            config: config::Config) -> Array2<f64> {

            let u_inf = config.u_inf;
            let mut cp = Array2::<f64>::zeros(velocity.dim());

            println!("\nCalculating pressure coefficient...");
            
            for ((i, j), velocity_ij) in velocity.indexed_iter() {
                cp[[i, j]] = 1.0 - (velocity_ij.u.powi(2) + velocity_ij.v.powi(2)) / (u_inf.powi(2));
            }
   cp 
}

// Save solution to CSV file
pub fn save_solution(file_name: &str, mesh: &Array2<Node>, phi: &Array2<f64>, cp: &Array2<f64>, velocity_field: &Array2<U>) {

    let mut file = File::create(file_name).unwrap();

    // CSV header
    writeln!(file, "x,y,phi,u,v,cp").unwrap();

    for ((i, j), phi) in phi.indexed_iter() {

        let node = &mesh[[i, j]];

        writeln!(
            file,
            "{},{},{},{},{},{}",
            node.x, node.y, phi, velocity_field[[i, j]].u, velocity_field[[i, j]].v, cp[[i, j]]
        ).unwrap();
    }

    println!("\nSolution saved to '{}'.\n", file_name);
}

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