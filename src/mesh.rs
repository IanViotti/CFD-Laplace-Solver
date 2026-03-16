use ndarray::Array2;
use std::fs::File;
use std::io::Write;
use crate::config::Config;

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub i: usize,
    pub j: usize,
    pub x: f64,
    pub y: f64,
}



pub fn build_cartesian_mesh(mesh_info: Config) -> Array2<Node> {

    println!("\nBuilding mesh with {} nodes.\n", mesh_info.IMAX * mesh_info.JMAX);

    let delta_x = 1.0 / (mesh_info.ITE - mesh_info.ILE) as f64;  

    let baseline_node = Node { i: 0, j: 0, x: 0.0, y: 0.0 };

    let mut mesh = Array2::<Node>::from_elem((mesh_info.IMAX, mesh_info.JMAX), baseline_node);

    // Points over the airfoil
    for i in mesh_info.ILE..=mesh_info.ITE {
        for j in 0..mesh_info.JMAX{
            mesh[[i,j]].j = j;
            mesh[[i,j]].i = i;

            // Assign y coordinates
            if j == 0 {
                mesh[[i,j]].y = -delta_x / 2.0;
            }
            else if j == 1 {
                mesh[[i,j]].y = delta_x / 2.0;
            }
            else {
                mesh[[i,j]].y = mesh[[i,j-1]].y + (mesh[[i,j-1]].y - mesh[[i,j-2]].y) * mesh_info.YSF;
            }

            // Assign x coordinates
            mesh[[i,j]].x = (i - mesh_info.ILE) as f64 * delta_x;
        }
    }

    
    // Points fwd the airfoil
    for i in (0..mesh_info.ILE).rev() {
        for j in 0..mesh_info.JMAX{

            mesh[[i,j]].j = j;
            mesh[[i,j]].i = i;

            // Assign y coordinates
            if j == 0 {
                mesh[[i,j]].y = -delta_x / 2.0;
            }
            else if j == 1 {
                mesh[[i,j]].y = delta_x / 2.0;
            }
            else {
                mesh[[i,j]].y = mesh[[i,j-1]].y + (mesh[[i,j-1]].y - mesh[[i,j-2]].y) * mesh_info.YSF;
            }

            // Assign x coordinates
            mesh[[i,j]].x = mesh[[i+1,j]].x + (mesh[[i+1,j]].x - mesh[[i+2,j]].x) * mesh_info.XSF;

        }
    } 


    // Points aft of the airfoil
    for i in (mesh_info.ITE+1)..mesh_info.IMAX {
        for j in 0..mesh_info.JMAX{

            mesh[[i,j]].j = j;
            mesh[[i,j]].i = i;

            // Assign y coordinates
            if j == 0 {
                mesh[[i,j]].y = -delta_x / 2.0;
            }
            else if j == 1 {
                mesh[[i,j]].y = delta_x / 2.0;
            }
            else {
                mesh[[i,j]].y = mesh[[i,j-1]].y + (mesh[[i,j-1]].y - mesh[[i,j-2]].y) * mesh_info.YSF;
            }

            // Assign x coordinates
            mesh[[i,j]].x = mesh[[i-1,j]].x + (mesh[[i-1,j]].x - mesh[[i-2,j]].x) * mesh_info.XSF;

        }
    }


    return mesh;
}

pub fn save_mesh(file_name: &str, mesh: &Array2<Node>) {

    let mut file = File::create(file_name).unwrap();

    // CSV header
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