import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_phi_contour(csv_file, levels=30):
    """
    Plot Cp contour using contourf from x,y coordinates in the CSV.
    """

    data = np.genfromtxt(csv_file, delimiter=",", names=True)

    x = data["x"]
    y = data["y"]
    phi = data["phi"]

    # determine grid dimensions
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    imax = len(unique_x)
    jmax = len(unique_y)

    # sort nodes by y then x to reconstruct grid ordering
    order = np.lexsort((x, y))

    x = x[order]
    y = y[order]
    phi = phi[order]

    # reshape arrays
    X = x.reshape(jmax, imax)
    Y = y.reshape(jmax, imax)
    PHI = phi.reshape(jmax, imax)

    fig, ax = plt.subplots()

    contour = ax.contourf(X, Y, PHI, levels=levels)
    plt.colorbar(contour, label="Phi")

    # bounding box
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        linewidth=1.5
    )

    # airfoil line
    ax.plot([0, 1], [0, 0], linewidth=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phi contour")
    ax.axis("equal")

    plt.savefig("job_files/phi_contour.png", dpi=500)

    plt.show()

def plot_U_vector(csv_file):
    """
    Reads the solution CSV and plots the vector field (phi_x, phi_y).

    Parameters
    ----------
    csv_file : str
        Path to the CSV solution file
    scale : float
        Scaling factor for quiver arrows
    """

    data = np.genfromtxt(csv_file, delimiter=",", names=True)

    x = data["x"]
    y = data["y"]
    u = data["u"]
    v = data["v"]

    fig, ax = plt.subplots()

    # Vector field
    ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale = 10, width=0.002, alpha=0.6)

    # Bounding box of the mesh domain
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        linewidth=1.5, color="black"
    )

    # Airfoil line (x=0 -> 1, y=0)
    ax.plot([0, 1], [0, 0], linewidth=4, color="black")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity vector field")
    ax.axis("equal")

    plt.savefig("job_files/velocity_vector_field.png", dpi=500)

    plt.show()


def plot_cp_contour(csv_file, levels=30):
    """
    Plot Cp contour using contourf from x,y coordinates in the CSV.
    """

    data = np.genfromtxt(csv_file, delimiter=",", names=True)

    x = data["x"]
    y = data["y"]
    cp = data["cp"]

    # determine grid dimensions
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    imax = len(unique_x)
    jmax = len(unique_y)

    # sort nodes by y then x to reconstruct grid ordering
    order = np.lexsort((x, y))

    x = x[order]
    y = y[order]
    cp = cp[order]

    # reshape arrays
    X = x.reshape(jmax, imax)
    Y = y.reshape(jmax, imax)
    CP = cp.reshape(jmax, imax)

    fig, ax = plt.subplots()

    contour = ax.contourf(X, Y, CP, levels=levels)
    plt.colorbar(contour, label="Cp")

    # bounding box
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        linewidth=1.5
    )

    # airfoil line
    ax.plot([0, 1], [0, 0], linewidth=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Cp contour")
    ax.axis("equal")

    plt.savefig("job_files/airfoil_cp_contour.png", dpi=500)

    plt.show()

if __name__ == "__main__":
    plot_phi_contour("job_files/solution.csv")
    plot_U_vector("job_files/solution.csv")
    plot_cp_contour("job_files/solution.csv")
