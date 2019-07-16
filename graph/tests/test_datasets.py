import unittest

import torch_geometric as tg

from graph.datasets.snap import Amazon, SnapNetwork, Epinions


class TestDatasets(unittest.TestCase):

    def test_load_amazon(self):
        dataset = Amazon("/tmp/Amazon")
        data = dataset[0]

        self.assertIsInstance(dataset, tg.data.Dataset)
        self.assertIsInstance(dataset, SnapNetwork)
        self.assertIsInstance(data, tg.data.Data)
        self.assertEqual(len(dataset), 1)

        self.assertEqual(data.num_edge_features, 0)
        self.assertEqual(data.num_edges, 925872)
        self.assertEqual(data.num_node_features, 1)
        self.assertEqual(data.num_nodes, 334863)

    def test_load_epinions(self):
        dataset = Epinions("/tmp/Epinions")
        data = dataset[0]

        self.assertIsInstance(dataset, tg.data.Dataset)
        self.assertIsInstance(dataset, SnapNetwork)
        self.assertIsInstance(data, tg.data.Data)
        self.assertEqual(len(dataset), 1)

        self.assertEqual(data.num_edge_features, 0)
        self.assertEqual(data.num_edges, 508837)
        self.assertEqual(data.num_node_features, 1)
        self.assertEqual(data.num_nodes, 75879)


if __name__ == '__main__':
    unittest.main()
