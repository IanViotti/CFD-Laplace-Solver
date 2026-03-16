import numpy as np
import matplotlib.pyplot as plt

def plot_mesh(csv_file):
    """
    Reads the mesh CSV and plots the node locations.
    """

    fig, ax = plt.subplots()

    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)

    x = data[:,2]
    y = data[:,3]

    ax.scatter(x,y,s=5)
    # Airfoil line (x=0 -> 1, y=0)
    ax.plot([0, 1], [0, 0], linewidth=4, color="black", alpha=0.7)

    plt.axis("equal")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mesh node locations")
    plt.savefig("job_files/mesh.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    plot_mesh("job_files/mesh.csv")