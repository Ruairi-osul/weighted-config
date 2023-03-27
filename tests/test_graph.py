import numpy as np
import networkx as nx
from weighted_config.graph import FullyConnectedGraph


# Helper functions for generating different types of graphs
def create_fully_connected_weighted_graph(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            graph.add_edge(i, j, weight=np.random.random())
    return graph


def create_fully_connected_no_weights(N):
    graph = nx.complete_graph(N)
    return graph


def create_fully_connected_partial_weights(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    graph = create_fully_connected_weighted_graph(N, seed)
    graph.edges[(0, 1)]["weight"] = None
    return graph


class TestFullyConnectedGraph:
    def test_is_fully_connected(self):
        good_graph = create_fully_connected_weighted_graph(10)
        assert FullyConnectedGraph.is_fully_connected(good_graph) == True

        not_connected_graph = nx.path_graph(10)
        assert FullyConnectedGraph.is_fully_connected(not_connected_graph) == False

    def test_has_weights(self):
        good_graph = create_fully_connected_weighted_graph(10)
        assert FullyConnectedGraph.has_weights(good_graph) == True

        no_weight_graph = create_fully_connected_no_weights(10)
        assert FullyConnectedGraph.has_weights(no_weight_graph) == False

        missing_weight_graph = create_fully_connected_partial_weights(10)
        assert FullyConnectedGraph.has_weights(missing_weight_graph) == False

    def test_from_edgelist(self):
        edgelist = [(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]
        fcg = FullyConnectedGraph.from_edgelist(edgelist)
        assert fcg.graph.number_of_nodes() == 3
        assert fcg.graph.number_of_edges() == 3
        assert fcg.graph[0][1]["weight"] == 1.0
        assert fcg.graph[0][2]["weight"] == 2.0
        assert fcg.graph[1][2]["weight"] == 3.0

    def test_from_weights(self):
        weights = np.array([1.0, 2.0, 3.0])
        fcg = FullyConnectedGraph.from_weights(weights, num_nodes=3)
        assert fcg.graph.number_of_nodes() == 3
        assert fcg.graph.number_of_edges() == 3
        assert fcg.graph[0][1]["weight"] == 1.0
        assert fcg.graph[0][2]["weight"] == 2.0
        assert fcg.graph[1][2]["weight"] == 3.0

    def test_from_adjacency_matrix(self):
        adjacency_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        fcg = FullyConnectedGraph.from_adjacency_matrix(adjacency_matrix)
        assert fcg.graph.number_of_nodes() == 3
        assert fcg.graph.number_of_edges() == 3
        assert fcg.graph[0][1]["weight"] == 1.0
        assert fcg.graph[0][2]["weight"] == 2.0
        assert fcg.graph[1][2]["weight"] == 3.0

    def test_update_nodes(self):
        good_graph = create_fully_connected_weighted_graph(3)
        fcg = FullyConnectedGraph(good_graph)
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C"])
        fcg.update_nodes(G)
        assert set(fcg.graph.nodes) == {"A", "B", "C"}
        assert fcg.graph.number_of_edges() == 3

    def test_number_of_nodes(self):
        good_graph = create_fully_connected_weighted_graph(4)
        fcg = FullyConnectedGraph(good_graph)
        assert fcg.number_of_nodes == 4

    def test_from_adjacency_matrix(self):
        np.random.seed(42)
        adjacency_matrix = np.random.rand(10, 10)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
        np.fill_diagonal(adjacency_matrix, 0)

        fcg = FullyConnectedGraph.from_adjacency_matrix(adjacency_matrix)
        assert fcg.graph.number_of_nodes() == 10
        assert fcg.graph.number_of_edges() == 45

        output_adjacency_matrix = fcg.adjacency_matrix
        assert np.allclose(
            adjacency_matrix, output_adjacency_matrix, rtol=1e-5, atol=1e-5
        )

    def test_from_weights(self):
        np.random.seed(42)
        weights = np.random.rand(
            45
        )  # For a 10-node graph, there are 10*(10-1)//2 = 45 weights
        fcg = FullyConnectedGraph.from_weights(weights, num_nodes=10)
        assert fcg.graph.number_of_nodes() == 10
        assert fcg.graph.number_of_edges() == 45

        adjacency_matrix = fcg.adjacency_matrix
        triu_weights = adjacency_matrix[np.triu_indices(fcg.number_of_nodes, 1)]
        assert np.allclose(weights, triu_weights, rtol=1e-5, atol=1e-5)

    def test_construct_graph_from_weights(self):
        good_graph = create_fully_connected_weighted_graph(5, seed=42)
        fcg = FullyConnectedGraph(good_graph)
        weights = fcg.weights

        new_fcg = FullyConnectedGraph.from_weights(weights, num_nodes=5)
        assert nx.is_isomorphic(
            fcg.graph,
            new_fcg.graph,
            edge_match=lambda e1, e2: e1["weight"] == e2["weight"],
        )

    def test_edgelist(self):
        good_graph = create_fully_connected_weighted_graph(4, seed=42)
        fcg = FullyConnectedGraph(good_graph)
        edgelist = fcg.edgelist
        assert len(edgelist) == 6

        def edge_in_edgelist(edge, weight_tol=1e-3):
            return any(
                e1 == edge[0]
                and e2 == edge[1]
                and np.isclose(w, edge[2], rtol=weight_tol, atol=weight_tol)
                for e1, e2, w in edgelist
            )

        assert edge_in_edgelist((0, 1, good_graph[0][1]["weight"]))
        assert edge_in_edgelist((2, 3, good_graph[2][3]["weight"]))

    def test_strength_distribution_arr(self):
        good_graph = create_fully_connected_weighted_graph(4, seed=42)
        fcg = FullyConnectedGraph(good_graph)
        strength_distribution_arr = fcg.strength_distribution_arr
        assert strength_distribution_arr.shape == (4,)
        assert np.allclose(
            strength_distribution_arr,
            np.array([d for _, d in good_graph.degree(weight="weight")]),
        )

    def test_strength_distribution(self):
        good_graph = create_fully_connected_weighted_graph(4, seed=42)
        fcg = FullyConnectedGraph(good_graph)
        strength_distribution = fcg.strength_distribution
        assert len(strength_distribution) == 4
        assert strength_distribution == dict(good_graph.degree(weight="weight"))
