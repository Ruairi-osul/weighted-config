import networkx as nx
from itertools import combinations
from typing import Any


class FullyConnectedGraph:
    def __init__(self, input_graph: nx.Graph):
        if not self.is_fully_connected(input_graph):
            raise ValueError("The input graph is not fully connected.")
        if not self.has_weights(input_graph):
            raise ValueError("The input graph does not weights at all edges.")
        self.graph = input_graph.copy()

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
