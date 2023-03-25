import networkx as nx
import numpy as np
import pytest
from itertools import combinations
from weighted_config.graph import FullyConnectedGraph


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


class TestFullyConnectedGraph:
    graph_generator = GraphGenerator()
    fcg = FullyConnectedGraph

    def test_is_fully_connected(self):
        # Test that the fully connected graph is recognized as such
        good_graph = self.graph_generator.good_graph(10)
        assert FullyConnectedGraph.is_fully_connected(good_graph) == True

        # Test that an incomplete graph is recognized as not fully connected
        not_connected_graph = self.graph_generator.not_fully_connected(10)
        assert FullyConnectedGraph.is_fully_connected(not_connected_graph) == False

    def test_has_weights(self):
        good_graph = self.graph_generator.good_graph(10)
        # Test that a fully connected graph with edge weights is recognized as such
        assert FullyConnectedGraph.has_weights(good_graph) == True

        # Test that a fully connected graph without edge weights is recognized as such
        no_weight_graph = self.graph_generator.no_weights(10)
        assert FullyConnectedGraph.has_weights(no_weight_graph) == False

        # Test that a fully connected graph with missing edge weights is recognized as such
        missing_weight_graph = self.graph_generator.missing_subset_of_weights(10)
        assert FullyConnectedGraph.has_weights(missing_weight_graph) == False

        # Test that a fully connected graph with weights at the wrong attribute is recognized as such
        wrong_attr_graph = self.graph_generator.missing_subset_of_weights(10)
        assert FullyConnectedGraph.has_weights(wrong_attr_graph) == False

        # Test that a fully connected graph with weights of the wrong type is recognized as such
        wrong_weight_type_graph = self.graph_generator.missing_subset_of_weights(10)
        assert FullyConnectedGraph.has_weights(wrong_weight_type_graph) == False
