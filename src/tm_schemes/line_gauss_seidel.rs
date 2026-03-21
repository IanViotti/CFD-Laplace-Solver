use ndarray::Array2;
use crate::mesh::Node;
use crate::tm_schemes::TimeMarchingScheme;
use crate::solver_core::calc_L_phi_ij;

pub struct LineGaussSeidel;

impl TimeMarchingScheme for LineGaussSeidel {

    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64 {

        
        let (imax, jmax) = mesh.dim();
        let mut C_n = Array2::<f64>::zeros((imax, jmax));

        let mut max_residual = 0.0;

        for i in 1..imax-1 {
            for j in 1..jmax-1 {
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
                let N_scheme = N_lgs(mesh, i, j);
                C_n[[i,j]] = - L_phi_n_ij / N_scheme;
                phi_n[[i,j]] = phi_n[[i,j]] + C_n[[i,j]];

                // Compute max residual
                if L_phi_n_ij.abs() > max_residual {
                    max_residual = L_phi_n_ij.abs();
                }
            }
        }


        return max_residual;
    }
} 

fn N_lgs(mesh: &Array2<Node>, i: usize, j: usize) -> f64 {
    -2.0 / ((mesh[[i+1, j]].x - mesh[[i-1, j]].x)/2.0).powi(2)
    -2.0 / ((mesh[[i, j+1]].y - mesh[[i, j-1]].y)/2.0).powi(2)
}


