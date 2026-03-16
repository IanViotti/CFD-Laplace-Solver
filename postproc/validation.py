import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_cp(solution_file):

    # Reference data from PDF page 8
    x_ref = np.array([
        0.00000,0.05000,0.10000,0.15000,0.20000,
        0.25000,0.30000,0.35000,0.40000,0.45000,
        0.50000,0.55000,0.60000,0.65000,0.70000,
        0.75000,0.80000,0.85000,0.90000,0.95000,
        1.00000
    ])

    cp_ref = -np.array([
        -0.11456,0.00111,0.04022,0.06819,0.08709,
        0.10500,0.11706,0.12593,0.13203,0.13561,
        0.13673,0.13561,0.13204,0.12594,0.11706,
        0.10501,0.08710,0.06819,0.04023,0.00111,
        -0.11456
    ])

    # Read solution
    df = pd.read_csv(solution_file)

    # Region of the airfoil
    airfoil = df[(df["x"] >= 0.0) & (df["x"] <= 1.0)]

    # Select nodes above the airfoil
    airfoil_upper = airfoil[airfoil["y"] > 0]

    # Get the first row above the surface (j = 1)
    y_surface = airfoil_upper["y"].min()
    surface = airfoil_upper[np.isclose(airfoil_upper["y"], y_surface)]

    # Sort by x
    surface = surface.sort_values("x")

    x_num = surface["x"].values
    cp_num = surface["cp"].values

    # Compute error
    rms = np.sqrt(np.mean((cp_num - cp_ref)**2))
    print(f"RMS Cp error = {rms:.6f}")

    # Plot comparison
    plt.figure(figsize=(7,5))
    plt.plot(x_num, cp_num, 'o-', label="CFD")
    plt.plot(x_ref, cp_ref, 's--', label="Reference")
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title("Cp comparison (upper surface)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return rms


if __name__ == "__main__":
    compare_cp("job_files/solution.csv")