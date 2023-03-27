import numpy as np
import networkx as nx
from scipy.optimize import minimize
from .graph import FullyConnectedGraph
from numba import njit
from typing import Optional


class ConfigurationModelGenerator:
    """
    Abstract class for generating configuration models for a given graph.

    Parameters
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.
    :param weight_bound_upper: float
        The upper bound on edge weights.
    :param weight_bound_lower: float
        The lower bound on edge weights.

    Attributes
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.
    :param input_graph_edges: list
        The edges of the input graph.
    :param input_strength_distribution: np.array
        The strength distribution of the input graph.
    :param weight_bound_upper: float
        The upper bound on edge weights.
    :param weight_bound_lower: float
        The lower bound on edge weights.

    Methods
    -------
    :param generate: Generate a configuration model for the input graph.

    """

    def __init__(
        self,
        graph: nx.Graph,
        weight_bound_upper: Optional[float] = 1,
        weight_bound_lower: Optional[float] = 0.00001,
    ):
        self.graph = FullyConnectedGraph(graph)
        self.weight_bound_upper = weight_bound_upper
        self.weight_bound_lower = weight_bound_lower

    def generate(self) -> nx.Graph:
        raise NotImplementedError("This is an abstract method.")


class ScipyOptimizeConfigurationModelGenerator(ConfigurationModelGenerator):
    """
    Generate a configuration model for a given graph using scipy.optimize.minimize.

    Parameters
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.
    :param weight_bound_upper: float
        The upper bound on edge weights.
    :param weight_bound_lower: float
        The lower bound on edge weights.
    :param method: str
        The method to use for optimization.
    :param maxiter: int
        The maximum number of iterations to perform.
    :param optimize_options: dict
        Additional options to pass to scipy.optimize.minimize.

    Attributes
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.
    :param weight_bound_upper: float
        The upper bound on edge weights.
    :param weight_bound_lower: float
        The lower bound on edge weights.
    :param method: str
        The method to use for optimization. See scipy.optimize.minimize for more details.
    :param maxiter: int
        The maximum number of iterations to perform.
    :param optimize_options: dict
        Additional options to pass to scipy.optimize.minimize.
    :param result_: scipy.optimize.OptimizeResult
        The result of the optimization.

    Methods
    -------
    :method generate: nx.Graph
        Generate a configuration model for the input graph.



    Examples
    --------
    >>> import networkx as nx
    >>> from weighted_config.config_models import ScipyOptimizeConfigurationModelGenerator
    >>> from weighted_config.utils import GraphGenerator
    >>> input_graph = GraphGenerator().good_graph(60)
    >>> generator = ScipyOptimizeConfigurationModelGenerator(input_graph, maxiter=10000)
    >>> generated_graph = generator.generate()
    >>> optimization_result = generator.result_
    """

    def __init__(
        self,
        graph: nx.Graph,
        weight_bound_upper: Optional[float] = 1,
        weight_bound_lower: Optional[float] = 0.00001,
        method: str = "L-BFGS-B",
        optimize_options: Optional[dict] = None,
        maxiter: Optional[int] = 10000,
    ):
        super().__init__(
            graph,
            weight_bound_upper=weight_bound_upper,
            weight_bound_lower=weight_bound_lower,
        )
        self.method = method
        self.optimize_options = optimize_options or {}
        self.maxiter = (
            maxiter
            if "maxiter" not in self.optimize_options
            else self.optimize_options["maxiter"]
        )
        if "maxiter" not in self.optimize_options and self.maxiter is not None:
            self.optimize_options["maxiter"] = self.maxiter

    def generate(self) -> nx.Graph:
        num_nodes = self.graph.number_of_nodes
        num_edges = len(self.graph.weights)
        original_strengths = self.graph.strength_distribution_arr
        initial_edge_weights = np.random.random(size=num_edges)
        bounds = [(self.weight_bound_lower, self.weight_bound_upper)] * num_edges

        result = minimize(
            fun=_strength_distribution_difference_wrapper,
            x0=initial_edge_weights,
            args=(original_strengths, num_nodes),
            bounds=bounds,
            method=self.method,
            options=self.optimize_options,
        )
        self.result_ = result
        new_graph = FullyConnectedGraph.from_weights(result.x, num_nodes=num_nodes)
        new_graph.update_nodes(self.graph.graph)
        return new_graph.graph


@njit
def _strength_distribution_difference_numba(
    edge_weights: np.ndarray, original_strengths: np.ndarray, num_nodes: int
) -> float:
    """
    Calculate the absolute difference between the strength distributions of the original graph and
    the generated graph represented by the given edge_weights.

    :param edge_weights: A NumPy array containing the weights of the generated graph's edges.
    :param original_strengths: A NumPy array containing the strengths of the original graph's nodes.
    :param num_nodes: An integer representing the number of nodes in the graph.
    :return: A float representing the absolute difference between the strength distributions.
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    triu_indices = np.triu_indices(num_nodes, 1)
    for i, (row, col) in enumerate(zip(*triu_indices)):
        adjacency_matrix[row, col] = edge_weights[i]
    adjacency_matrix += adjacency_matrix.T
    generated_strengths = np.sum(adjacency_matrix, axis=1) - np.diag(adjacency_matrix)
    absolute_difference = np.sum(np.abs(original_strengths - generated_strengths))
    return absolute_difference


def _strength_distribution_difference_wrapper(
    edge_weights: np.ndarray, original_strengths: np.ndarray, num_nodes: int
) -> float:
    """
    A wrapper function for _strength_distribution_difference_numba to be compatible with SciPy's minimize function.

    :param edge_weights: A NumPy array containing the weights of the generated graph's edges.
    :param original_strengths: A NumPy array containing the strengths of the original graph's nodes.
    :param num_nodes: An integer representing the number of nodes in the graph.
    :return: A float representing the absolute difference between the strength distributions.
    """
    return _strength_distribution_difference_numba(
        edge_weights, original_strengths, num_nodes
    )
