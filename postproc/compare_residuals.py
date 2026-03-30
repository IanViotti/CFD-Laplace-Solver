import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_residual_log10(jobs, result_file):
    
    # Plot
    plt.figure(figsize=(7,5))

    for job in jobs:
        csv_file = f"job_files/{job}/solution_data/residual_history.csv"
        # Read CSV
        df = pd.read_csv(csv_file)

        iteration = df["iter"].values
        avg_residual = df["max"].values

        # Compute log10
        log_avg = np.log10(avg_residual)

        
        plt.plot(iteration, log_avg, label=job)

    plt.xlabel("Iteration")
    plt.ylabel("log10(Residual)")
    plt.title("Solver Convergence")
    plt.grid(True)
    plt.legend()

    plt.savefig(result_file, dpi=500)

    plt.show()


if __name__ == "__main__":
    jobs = ["jacobi_t05", "gs_t05", "sor_r1.8_t05", "lgs_t05", "slor_r1.8_t05"]
    plot_residual_log10(jobs, 
                        result_file="job_files/general/residual_comparison.png")