import networkx as nx
import numpy as np
import timeit
import matplotlib.pyplot as plt
from weighted_config.config_models import ScipyOptimizeConfigurationModelGenerator
from weighted_config.utils import GraphGenerator
import os


def compare_methods(graph_sizes, methods, num_trials=3):
    results = {method: {"time": [], "error": []} for method in methods}

    for size in graph_sizes:
        print(f"Processing graph size {size}...")
        G = GraphGenerator().good_graph(size)

        for method in methods:
            times = []
            errors = []
            for _ in range(num_trials):
                start_time = timeit.default_timer()
                graph_generator = ScipyOptimizeConfigurationModelGenerator(
                    G,
                    method=method,
                    maxiter=1000,
                )
                graph_generator.generate()
                run_time = timeit.default_timer() - start_time
                error = (
                    graph_generator.result_.fun
                )  # Get the optimization error from the results_
                times.append(run_time)
                errors.append(error)

            # Calculate the average time and error over num_trials
            avg_time = np.mean(times)
            avg_error = np.mean(errors)
            results[method]["time"].append(avg_time)
            results[method]["error"].append(avg_error)

    return results


def plot_results(results, graph_sizes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot run time
    for method, data in results.items():
        ax1.plot(graph_sizes, data["time"], label=method, marker="o")
    ax1.set_xlabel("Graph Size")
    ax1.set_ylabel("Average Run Time (seconds)")
    ax1.legend()
    ax1.set_title("Run Time Comparison of Methods")

    # Plot optimization error
    for method, data in results.items():
        ax2.plot(graph_sizes, data["error"], label=method, marker="o")
    ax2.set_xlabel("Graph Size")
    ax2.set_ylabel("Average Optimization Error")
    ax2.legend()
    ax2.set_title("Optimization Error Comparison of Methods")

    fig.tight_layout()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fig_name = "compare_optim_methods.png"
    fig_path = os.path.join(script_dir, fig_name)
    fig.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    graph_sizes = [
        10,
        30,
        60,
    ]
    methods = [
        "L-BFGS-B",
        "TNC",
        "trust-constr",
    ]
    results = compare_methods(graph_sizes, methods, num_trials=2)
    plot_results(results, graph_sizes)
