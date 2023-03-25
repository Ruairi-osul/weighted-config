import numpy as np
from weighted_config.config_models import ScipyOptimizeConfigurationModelGenerator
from weighted_config.utils import GraphGenerator


class TestScipyConfigModel:
    def test_generate(self):
        good_graph = GraphGenerator().good_graph(10)
        generator = ScipyOptimizeConfigurationModelGenerator(good_graph)

        generated_graph = generator.generate()

        input_dist = generator.input_strength_distribution
        generated_dis = generator.strength_distribution(generated_graph)

        assert np.allclose(input_dist, generated_dis)

        input_edge_idx = generator.input_graph_edges
        input_edge_weights = [good_graph[u][v]["weight"] for u, v in input_edge_idx]
        generated_edge_weights = [
            generated_graph[u][v]["weight"] for u, v in input_edge_idx
        ]
        assert ~np.allclose(input_edge_weights, generated_edge_weights)
