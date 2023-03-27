import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from weighted_config.config_models import ScipyOptimizeConfigurationModelGenerator
from weighted_config.utils import GraphGenerator
import timeit
import os


graph_size = 50
num_trials = 3
methods = ["L-BFGS-B"]

ftol_values = [1e-3, 1e-4, 1e-5]
gtol_values = [1e-3, 1e-4, 1e-5]
maxiter_values = [100, 500, 1000]

results = {
    "ftol": {val: {"time": [], "error": []} for val in ftol_values},
    "gtol": {val: {"time": [], "error": []} for val in gtol_values},
    "maxiter": {val: {"time": [], "error": []} for val in maxiter_values},
}

G = GraphGenerator().good_graph(graph_size)

# Iterate through each parameter and its values
for param, values in results.items():
    for value in values:
        times = []
        errors = []
        for _ in range(num_trials):
            options = {"ftol": 1e-4, "gtol": 1e-4, "maxiter": 500}
            options[param] = value
            generator = ScipyOptimizeConfigurationModelGenerator(
                G, method="L-BFGS-B", optimize_options=options
            )
            start_time = timeit.default_timer()
            generated_graph = generator.generate()
            run_time = timeit.default_timer() - start_time
            error = generator.result_.fun

            times.append(run_time)
            errors.append(error)

        # Calculate the average time and error over num_trials
        avg_time = np.mean(times)
        avg_error = np.mean(errors)
        results[param][value]["time"] = avg_time
        results[param][value]["error"] = avg_error


# Plotting function
def plot_parameter_performance(results):
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for i, (param, values) in enumerate(results.items()):
        # Plot run time
        for value, data in values.items():
            axes[i, 0].plot(value, data["time"], label=f"{param} = {value}", marker="o")
        axes[i, 0].set_xlabel(param)
        axes[i, 0].set_ylabel("Average Run Time (seconds)")
        axes[i, 0].legend()
        axes[i, 0].set_title(f"Run Time Comparison for {param}")

        # Plot optimization error
        for value, data in values.items():
            axes[i, 1].plot(
                value, data["error"], label=f"{param} = {value}", marker="o"
            )
        axes[i, 1].set_xlabel(param)
        axes[i, 1].set_ylabel("Average Optimization Error")
        axes[i, 1].legend()

    fig.tight_layout()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fig_name = "performance_comparison.png"
    fig_path = os.path.join(script_dir, fig_name)
    fig.savefig(fig_path, dpi=300)


# Call the plotting function
plot_parameter_performance(results)
