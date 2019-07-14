import unittest

import torch_geometric as tg

from graph.datasets.amazon import Amazon


class TestDatasets(unittest.TestCase):

    def test_load_amazon(self):
        dataset = Amazon("/tmp")
        data = dataset[0]

        self.assertIsInstance(dataset, tg.data.Dataset)
        self.assertIsInstance(data, tg.data.Data)
        self.assertEqual(len(dataset), 1)

        self.assertEqual(data.num_edge_features, 0)
        self.assertEqual(data.num_edges, 925872)
        self.assertEqual(data.num_node_features, 1)
        self.assertEqual(data.num_nodes, 334863)



if __name__ == '__main__':
    unittest.main()
