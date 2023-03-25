from weighted_config.graph import FullyConnectedGraph
from weighted_config.utils import GraphGenerator


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
