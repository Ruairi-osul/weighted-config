import numpy as np
from weighted_config.config_models import ScipyOptimizeConfigurationModelGenerator
from weighted_config.utils import GraphGenerator
from weighted_config.graph import FullyConnectedGraph
import networkx as nx


class TestScipyConfigModel:
    def test_generator_initialization(self):
        input_graph = GraphGenerator().good_graph(60)
        generator = ScipyOptimizeConfigurationModelGenerator(input_graph, maxiter=10000)

        assert isinstance(generator.graph, FullyConnectedGraph)
        assert generator.maxiter == 10000

    def test_generated_graph_properties(self):
        input_graph = GraphGenerator().good_graph(60)
        generator = ScipyOptimizeConfigurationModelGenerator(input_graph, maxiter=10000)
        generated_graph = generator.generate()

        assert isinstance(generated_graph, nx.Graph)
        assert input_graph.number_of_nodes() == generated_graph.number_of_nodes()
        assert input_graph.number_of_edges() == generated_graph.number_of_edges()

    def test_generated_graph_edge_weights(self):
        input_graph = GraphGenerator().good_graph(60)
        generator = ScipyOptimizeConfigurationModelGenerator(input_graph, maxiter=10000)
        generated_graph = generator.generate()

        for u, v in generated_graph.edges():
            weight = generated_graph[u][v]["weight"]
            assert weight >= generator.weight_bound_lower
            assert weight <= generator.weight_bound_upper

    def test_optimize_result_properties(self):
        input_graph = GraphGenerator().good_graph(60)
        generator = ScipyOptimizeConfigurationModelGenerator(input_graph, maxiter=10000)
        generated_graph = generator.generate()
        optimization_result = generator.result_

        assert hasattr(optimization_result, "fun")
        assert hasattr(optimization_result, "x")
        assert len(optimization_result.x) == input_graph.number_of_edges()

    def test_generate(self):
        input_graph = GraphGenerator().good_graph(30)
        input_fsg = FullyConnectedGraph(input_graph)
        input_dist = input_fsg.strength_distribution_arr

        generator = ScipyOptimizeConfigurationModelGenerator(
            input_graph,
            maxiter=10000,
            weight_bound_upper=1,
        )
        generated_graph = generator.generate()
        generated_fsg = FullyConnectedGraph(generated_graph)

        generated_dis = generated_fsg.strength_distribution_arr

        assert np.allclose(input_dist, generated_dis, rtol=5e-1)

        input_edge_idx = input_graph.edges()
        input_edge_weights = [input_graph[u][v]["weight"] for u, v in input_edge_idx]
        generated_edge_weights = [
            generated_graph[u][v]["weight"] for u, v in input_edge_idx
        ]
        assert ~np.allclose(input_edge_weights, generated_edge_weights)
