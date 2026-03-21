import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_residual_log10(csv_file, result_file):

    # Read CSV
    df = pd.read_csv(csv_file)

    iteration = df["iter"].values
    avg_residual = df["max"].values

    # Compute log10
    log_avg = np.log10(avg_residual)

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(iteration, log_avg, label="log10(avg residual)")

    plt.xlabel("Iteration")
    plt.ylabel("log10(Residual)")
    plt.title("Solver Convergence")
    plt.grid(True)
    plt.legend()

    plt.savefig(result_file, dpi=500)

    plt.show()


if __name__ == "__main__":
    plot_residual_log10("job_files/solution_data/residual_history.csv", 
                        result_file="job_files/post_proc_results/SOR_residual_history.png")