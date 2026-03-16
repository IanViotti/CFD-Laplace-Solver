import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_residual_log10(csv_file):
    # Read CSV
    df = pd.read_csv(csv_file)

    iteration = df["iteration"].values
    residual = df["residual"].values

    # Compute log10
    log_residual = np.log10(residual)

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(iteration, log_residual)
    plt.xlabel("Iteration")
    plt.ylabel("log10(Residual)")
    plt.title("Solver Convergence")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    plot_residual_log10("job_files/residual_history.csv")