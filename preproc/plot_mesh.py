import numpy as np
import matplotlib.pyplot as plt

def plot_mesh(csv_file):
    """
    Reads the mesh CSV and plots the node locations.
    """

    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)

    x = data[:,2]
    y = data[:,3]

    plt.scatter(x,y,s=5)
    plt.axis("equal")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_mesh("job_files/mesh.csv")