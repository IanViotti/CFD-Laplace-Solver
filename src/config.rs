use crate::tm_schemes::TimeMarchingScheme;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub ILE: usize, // Leading Edge Index
    pub ITE: usize, // Trailing Edge Index
    pub IMAX: usize, // Max points in X direction
    pub JMAX: usize, // Max points in Y direction
    pub XSF: f64, // Stretching factor in x 
    pub YSF: f64, // Stretching factor in y
    pub u_inf: f64, // Free stream velocity
    pub t: f64, // thickness of the airfoil
    pub n_max: usize, // Max number of iterations
}