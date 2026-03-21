use ndarray::Array2;
use std::fs::File;
use std::io::Write;
use crate::config::Config;

/// Represents a node in the structured computational grid.
///
/// # Fields
/// * `i` - Index in the streamwise (x) direction
/// * `j` - Index in the normal (y) direction
/// * `x` - Physical x-coordinate
/// * `y` - Physical y-coordinate
///
/// # Notes
/// - The mesh is structured and logically rectangular
/// - Physical spacing may be non-uniform due to stretching
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub i: usize,
    pub j: usize,
    pub x: f64,
    pub y: f64,
}

/// Builds a structured Cartesian mesh with geometric stretching.
///
/// The mesh is divided into three regions:
/// 1. **Over the airfoil (uniform in x)**
/// 2. **Upstream of the airfoil (stretched in x)**
/// 3. **Downstream of the airfoil (stretched in x)**
///
/// In the y-direction, geometric stretching is applied away from the airfoil.
///
/// # Arguments
/// * `mesh_info` - Configuration containing mesh parameters:
///   - IMAX, JMAX: grid dimensions
///   - ILE, ITE: airfoil index range
///   - XSF: x-direction stretching factor
///   - YSF: y-direction stretching factor
///
/// # Returns
/// * `Array2<Node>` - Structured mesh with coordinates
///
/// # Mesh Characteristics
///
/// ## X-direction:
/// - Uniform spacing over the airfoil
/// - Geometric expansion upstream and downstream
///
/// ## Y-direction:
/// - Symmetric clustering near the airfoil (j = 0, 1)
/// - Geometric stretching away from the surface
///
/// # Notes
/// - The airfoil lies along the line j = 0
/// - Two layers are initialized near the surface to define spacing
/// - Stretching is applied recursively
pub fn build_cartesian_mesh(mesh_info: Config) -> Array2<Node> {

    println!(
        "\nBuilding mesh with {} nodes.\n",
        mesh_info.IMAX * mesh_info.JMAX
    );

    // Uniform spacing over the airfoil chord
    let delta_x = 1.0 / (mesh_info.ITE - mesh_info.ILE) as f64;

    let baseline_node = Node { i: 0, j: 0, x: 0.0, y: 0.0 };

    let mut mesh =
        Array2::<Node>::from_elem((mesh_info.IMAX, mesh_info.JMAX), baseline_node);

    // ===============================
    // Region 1: Over the airfoil
    // ===============================
    for i in mesh_info.ILE..=mesh_info.ITE {
        for j in 0..mesh_info.JMAX {

            mesh[[i, j]].i = i;
            mesh[[i, j]].j = j;

            // --- Y coordinates (vertical stretching) ---
            if j == 0 {
                mesh[[i, j]].y = -delta_x / 2.0;
            } else if j == 1 {
                mesh[[i, j]].y = delta_x / 2.0;
            } else {
                mesh[[i, j]].y =
                    mesh[[i, j-1]].y +
                    (mesh[[i, j-1]].y - mesh[[i, j-2]].y) * mesh_info.YSF;
            }

            // --- X coordinates (uniform over airfoil) ---
            mesh[[i, j]].x =
                (i - mesh_info.ILE) as f64 * delta_x;
        }
    }

    // ===============================
    // Region 2: Upstream (forward)
    // ===============================
    for i in (0..mesh_info.ILE).rev() {
        for j in 0..mesh_info.JMAX {

            mesh[[i, j]].i = i;
            mesh[[i, j]].j = j;

            // --- Y coordinates (same stretching) ---
            if j == 0 {
                mesh[[i, j]].y = -delta_x / 2.0;
            } else if j == 1 {
                mesh[[i, j]].y = delta_x / 2.0;
            } else {
                mesh[[i, j]].y =
                    mesh[[i, j-1]].y +
                    (mesh[[i, j-1]].y - mesh[[i, j-2]].y) * mesh_info.YSF;
            }

            // --- X coordinates (geometric stretching upstream) ---
            mesh[[i, j]].x =
                mesh[[i+1, j]].x +
                (mesh[[i+1, j]].x - mesh[[i+2, j]].x) * mesh_info.XSF;
        }
    }

    // ===============================
    // Region 3: Downstream (aft)
    // ===============================
    for i in (mesh_info.ITE + 1)..mesh_info.IMAX {
        for j in 0..mesh_info.JMAX {

            mesh[[i, j]].i = i;
            mesh[[i, j]].j = j;

            // --- Y coordinates (same stretching) ---
            if j == 0 {
                mesh[[i, j]].y = -delta_x / 2.0;
            } else if j == 1 {
                mesh[[i, j]].y = delta_x / 2.0;
            } else {
                mesh[[i, j]].y =
                    mesh[[i, j-1]].y +
                    (mesh[[i, j-1]].y - mesh[[i, j-2]].y) * mesh_info.YSF;
            }

            // --- X coordinates (geometric stretching downstream) ---
            mesh[[i, j]].x =
                mesh[[i-1, j]].x +
                (mesh[[i-1, j]].x - mesh[[i-2, j]].x) * mesh_info.XSF;
        }
    }

    mesh
}

/// Saves the mesh to a CSV file for visualization or debugging.
///
/// # Output format
/// Columns:
/// id (implicit ordering), i, j, x, y
///
/// # Arguments
/// * `file_name` - Output file path
/// * `mesh` - Structured grid
///
/// # Notes
/// - Each row corresponds to one node
/// - Can be imported into visualization tools (e.g., Paraview, Tecplot)
pub fn save_mesh(file_name: &str, mesh: &Array2<Node>) {

    let mut file = File::create(file_name).unwrap();

    writeln!(file, "id,i,j,x,y").unwrap();

    for node in mesh.iter() {
        writeln!(
            file,
            "{},{},{},{}",
            node.i, node.j, node.x, node.y
        ).unwrap();
    }

    println!("\nMesh saved to '{}'.\n", file_name);
}