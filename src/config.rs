/// Configuration structure containing all simulation parameters.
///
/// This struct centralizes the definition of:
/// - Mesh geometry
/// - Flow conditions
/// - Numerical parameters
///
/// It is passed throughout the solver to ensure consistency and flexibility.
///
/// # Fields
///
/// ## Geometry / Mesh Definition
/// * `ILE` - Index of the leading edge of the airfoil in the x-direction
/// * `ITE` - Index of the trailing edge of the airfoil in the x-direction
/// * `IMAX` - Total number of grid points in the x-direction
/// * `JMAX` - Total number of grid points in the y-direction
///
/// ## Mesh Stretching
/// * `XSF` - Geometric stretching factor in the x-direction
///           - Controls grid expansion upstream and downstream of the airfoil
///           - Values > 1 produce progressively larger cells away from the airfoil
///
/// * `YSF` - Geometric stretching factor in the y-direction
///           - Controls clustering of points near the airfoil surface
///           - Higher values increase stretching away from the wall
///
/// ## Flow Parameters
/// * `u_inf` - Freestream velocity magnitude (U∞)
///
/// ## Airfoil Geometry
/// * `t` - Maximum thickness parameter of the airfoil
///         - Used to define the surface slope (dy/dx)
///         - Typically corresponds to a symmetric biconvex airfoil
///
/// ## Numerical Parameters
/// * `n_max` - Maximum number of solver iterations
///
/// # Notes
/// - The airfoil is implicitly defined along the line j = 0
/// - The chord length is normalized to 1.0
/// - Indices ILE and ITE define the portion of the grid corresponding to the airfoil
/// - This struct is designed to be lightweight and easily copyable (`Copy` trait)
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub ILE: usize,
    pub ITE: usize,
    pub IMAX: usize,
    pub JMAX: usize,
    pub XSF: f64,
    pub YSF: f64,
    pub u_inf: f64,
    pub t: f64,
    pub n_max: usize,
    pub conv_criterion: f64,
}