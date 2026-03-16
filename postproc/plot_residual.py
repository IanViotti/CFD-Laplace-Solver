import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_residual_log10(csv_file):

    # Read CSV
    df = pd.read_csv(csv_file)

    iteration = df["iter"].values
    max_residual = df["max"].values
    avg_residual = df["average"].values

    # Compute log10
    log_max = np.log10(max_residual)
    log_avg = np.log10(avg_residual)

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(iteration, log_max, label="log10(max residual)")
    plt.plot(iteration, log_avg, label="log10(avg residual)")

    plt.xlabel("Iteration")
    plt.ylabel("log10(Residual)")
    plt.title("Solver Convergence")
    plt.grid(True)
    plt.legend()

    plt.savefig("job_files/residual_history.png", dpi=500)

    plt.show()


if __name__ == "__main__":
    plot_residual_log10("job_files/residual_history.csv")