import networkx as nx
from typing import Any
import numpy as np


class FullyConnectedGraph:
    """
    A fully connected graph with weights on all edges. Must have 'weight' attribute.
    """

    def __init__(self, input_graph: nx.Graph):
        """

        :param input_graph: A networkx graph with 'weight' attribute for all edges.
        """
        if not self.is_fully_connected(input_graph):
            raise ValueError("The input graph is not fully connected.")
        if not self.has_weights(input_graph):
            raise ValueError("The input graph does not weights at all edges.")
        self.graph = input_graph.copy()

    @property
    def number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(self.graph, weight="weight")

    @property
    def weights(self) -> np.ndarray:
        return self.adjacency_matrix[np.triu_indices(self.number_of_nodes, 1)]

    @property
    def edgelist(self) -> list[tuple[Any, Any, float]]:
        return list(self.graph.edges(data="weight"))

    @property
    def strength_distribution_arr(self) -> np.ndarray:
        return np.array([d for _, d in self.graph.degree(weight="weight")])

    @property
    def strength_distribution(self) -> dict[Any, float]:
        return dict(self.graph.degree(weight="weight"))

    def update_nodes(self, G: nx.Graph) -> None:
        """
        Update the nodes of the graph to match the nodes of G.
        :param G: nx.Graph
        """
        self.graph = nx.relabel_nodes(self.graph, dict(zip(self.graph.nodes, G.nodes)))

    @staticmethod
    def is_fully_connected(graph: nx.Graph) -> bool:
        N = graph.number_of_nodes()
        E = graph.number_of_edges()
        return E == N * (N - 1) // 2

    @staticmethod
    def has_weights(graph: nx.Graph) -> bool:
        edges = graph.edges(data="weight")
        return all(weight is not None for _, _, weight in edges)

    @classmethod
    def from_edgelist(
        cls, edgelist: list[tuple[Any, Any, float]]
    ) -> "FullyConnectedGraph":
        graph = nx.Graph()
        graph.add_weighted_edges_from(edgelist)
        return cls(graph)

    @classmethod
    def from_weights(cls, weights: np.ndarray, num_nodes: int) -> "FullyConnectedGraph":
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        adjacency_matrix[np.triu_indices(num_nodes, 1)] = weights
        adjacency_matrix += adjacency_matrix.T
        np.fill_diagonal(adjacency_matrix, 0)
        return cls.from_adjacency_matrix(adjacency_matrix)

    @classmethod
    def from_adjacency_matrix(
        cls, adjacency_matrix: np.ndarray
    ) -> "FullyConnectedGraph":
        graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph)
        return cls(graph)
