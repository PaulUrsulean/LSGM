import sys
import unittest
from os.path import dirname, abspath

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

from graph.torch_lsh import LSHDecoder
from graph.torch_lsh import CosineSimilarity


class TestCosineSimilarity(unittest.TestCase):

    def test_pairwise_sim(self):
        n_nodes, latent_dim = 100, 16
        embeddings = torch.randn((n_nodes, latent_dim))
        sim = CosineSimilarity(0, 0)

        # Test Shape
        pairwise_sim = sim.pairwise_sim(embeddings)
        self.assertEqual(pairwise_sim.size(), torch.Size([n_nodes, n_nodes]))

        # Compare actual values
        sim_ground_truth = torch.empty(size=(n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                sim_ground_truth[i, j] = F.cosine_similarity(embeddings[i], embeddings[j], dim=0)
        self.assertTrue(torch.allclose(pairwise_sim, sim_ground_truth, atol=1e-7))

    def test_signature_matrix_shape(self):
        N, D = 100, 16
        bands, rows = 12, 6

        X = torch.randn((N, D))
        sim = CosineSimilarity(bands, rows)
        signature_matrix = sim.signature(X)

        self.assertEqual(list(signature_matrix.size()), [bands, rows, N, ])


class TestLSHDecoder(unittest.TestCase):

    def test_returns_correct_items(self):
        emb = np.array([[0.0, 1.0],
                        [0.0, -1.0],
                        [1.0, 0.0],
                        [0.05, 0.9]
                        ], dtype=np.float32)
        emb_tensor = torch.tensor(emb)

        dec = LSHDecoder(bands=2, rows=10, verbose=False)
        adj = dec(emb_tensor)
        indices = adj.coalesce().indices().t().detach().numpy()

        self.assertTrue(len(indices) == 2)
        self.assertTrue([0, 3] in indices and [3, 0] in indices)

    def test_does_not_fail_on_large_input(self):
        emb = np.random.normal(size=(100000, 128)).astype(np.float32)
        emb_tensor = torch.tensor(emb)

        dec = LSHDecoder(bands=2, rows=32, verbose=True)
        dec(emb_tensor)


if __name__ == '__main__':
    unittest.main()
