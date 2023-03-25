import numpy as np
import networkx as nx
from scipy.optimize import minimize
from .graph import FullyConnectedGraph


class ConfigurationModelGenerator:
    """
    Abstract class for generating configuration models for a given graph.

    Parameters
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.

    Attributes
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.
    :param input_graph_edges: list
        The edges of the input graph.
    :param input_strength_distribution: np.array
        The strength distribution of the input graph.

    Methods
    -------
    :method generate: nx.Graph
        Generate a configuration model for the input graph.
    :method gen_graph_from_weights: nx.Graph
        Generate a graph from a list of edge weights.
    :method objective_function: float
        The objective function to minimize.

    """

    def __init__(self, graph: nx.Graph):
        self.graph = FullyConnectedGraph(graph)
        self.input_graph_edges = self.graph.graph.edges()
        self.input_strength_distribution = self.strength_distribution(self.graph.graph)

    @staticmethod
    def strength_distribution(graph: nx.Graph):
        degrees = sorted(graph.degree(weight="weight"), key=lambda x: x[0])
        return np.array([strength for (_, strength) in degrees])

    def generate(self):
        raise NotImplementedError("This is an abstract method.")

    def gen_graph_from_weights(self, weights):
        edgelist = [
            (n1, n2, weight)
            for (n1, n2), weight in zip(self.input_graph_edges, weights)
        ]
        return FullyConnectedGraph.from_edgelist(edgelist).graph

    def objective_function(self, weights):
        generated_graph = self.gen_graph_from_weights(weights)
        generated_strengths = self.strength_distribution(generated_graph)
        input_strengths = self.input_strength_distribution
        return np.sum(np.absolute(generated_strengths - input_strengths))


class ScipyOptimizeConfigurationModelGenerator(ConfigurationModelGenerator):
    """
    Use scipy.optimize.minimize to find the edge weights that minimize the difference
    between the strength distribution of the input graph and the generated graph.

    Parameters
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.
    :param method: str
        The method to use for scipy.optimize.minimize. See the documentation for
        scipy.optimize.minimize for more information.

    Attributes
    ----------
    :param graph: nx.Graph
        The graph to generate a configuration model for.
    :param method: str
        The method to use for scipy.optimize.minimize. See the documentation for
        scipy.optimize.minimize for more information.
    :param input_graph_edges: list
        The edges of the input graph.
    :param input_strength_distribution: np.array
        The strength distribution of the input graph.

    Methods
    -------
    :method generate: nx.Graph
        Generate a configuration model for the input graph.
    :method gen_graph_from_weights: nx.Graph
        Generate a graph from a list of edge weights.
    :method objective_function: float
        The objective function to minimize.

    Examples
    --------
    >>> from weighted_config.config_models import ScipyOptimizeConfigurationModelGenerator

    >>> generator = ScipyOptimizeConfigurationModelGenerator(G_original)
    >>> G_configuration = generator.generate()
    >>>
    >>> # Same strength distribution
    >>> print(G_original.degree(weight="weight"))
    >>> print(G_configuration.degree(weight="weight"))
    >>>
    >>> # Different edges
    >>> print(list(G_original.edges(data="weight"))[:5])
    >>> print(list(G_configuration.edges(data="weight"))[:5])

    """

    def __init__(self, graph: nx.Graph, method: str = "L-BFGS-B"):
        super().__init__(graph)
        self.method = method

    def generate(self):
        N = self.graph.graph.number_of_nodes()
        num_edges = N * (N - 1) // 2
        initial_weights = np.random.random(size=num_edges)

        result = minimize(self.objective_function, initial_weights, method=self.method)
        optimized_weights = result.x
        return self.gen_graph_from_weights(optimized_weights)
