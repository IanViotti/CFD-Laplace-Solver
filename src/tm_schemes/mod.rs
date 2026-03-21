pub mod jacobi;
pub mod gauss_seidel;
pub mod sor;
pub mod line_gauss_seidel;

use ndarray::Array2;
use crate::mesh::Node;

pub trait TimeMarchingScheme {
    fn step(
        &self,
        mesh: &Array2<Node>,
        phi_n: &mut Array2<f64>,
    ) -> f64;
}


