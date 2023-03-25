import networkx as nx
import numpy as np


class GraphGenerator:
    def good_graph(self, n_nodes: int) -> nx.Graph:
        graph = nx.complete_graph(n_nodes)
        num_edges = graph.number_of_edges()
        weights = np.random.uniform(0, 1, num_edges)
        for (u, v), weight in zip(graph.edges(), weights):
            graph[u][v]["weight"] = weight
        return graph

    def not_fully_connected(self, n_nodes: int) -> nx.Graph:
        graph = nx.complete_graph(n_nodes)
        num_edges = graph.number_of_edges()
        weights = np.random.uniform(0, 1, num_edges)
        for (u, v), weight in zip(graph.edges(), weights):
            graph[u][v]["weight"] = weight
        graph.remove_edge(0, 1)
        return graph

    def no_weights(self, n_nodes: int) -> nx.Graph:
        graph = nx.complete_graph(n_nodes)
        return graph

    def wrong_weights_attribute(self, n_nodes: int) -> nx.Graph:
        graph = nx.complete_graph(n_nodes)
        num_edges = graph.number_of_edges()
        weights = np.random.uniform(0, 1, num_edges)
        for (u, v), weight in zip(graph.edges(), weights):
            graph[u][v]["foo"] = weight
        return graph

    def wrong_weights_type(self, n_nodes: int) -> nx.Graph:
        graph = nx.complete_graph(n_nodes)
        num_edges = graph.number_of_edges()
        weights = np.random.uniform(0, 1, num_edges)
        for (u, v), weight in zip(graph.edges(), weights):
            graph[u][v]["weight"] = str(weight)
        return graph

    def missing_subset_of_weights(self, n_nodes: int) -> nx.Graph:
        graph = nx.complete_graph(n_nodes)
        num_edges = graph.number_of_edges()
        weights = np.random.uniform(0, 1, num_edges)
        for (u, v), weight in zip(graph.edges(), weights):
            graph[u][v]["weight"] = weight
        del graph[0][1]["weight"]
        return graph
