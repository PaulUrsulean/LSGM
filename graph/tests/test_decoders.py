import unittest

import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean

from graph.modules import CosineSimDecoder, EuclideanDistanceDecoder


def create_example_embeddings(n_nodes, normalize=False):
    """ Creates random node embeddings for a fully connected graph
    :param n_nodes: Number of nodes in the graph
    :param normalize: If embeddings should be scaled to unit norm
    :return: tuple with embeddings as ndarray, tensor and edge indices """
    emb_numpy = np.random.random((n_nodes, 128))
    if normalize:
        emb_numpy = emb_numpy / np.linalg.norm(emb_numpy, axis=1)[:, np.newaxis]

    emb_tensor = torch.tensor(emb_numpy, dtype=torch.float32)

    # Make graph fully connected (with self loops)
    edge_indices = sum([[i] * n_nodes for i in range(n_nodes)], [])
    edge_indices = torch.tensor([edge_indices, list(range(n_nodes)) * n_nodes], dtype=torch.long)
    return emb_numpy, emb_tensor, edge_indices


class TestCosineSimDecoder(unittest.TestCase):

    def test_calculates_correct_single_similarity(self):
        """
        Tests if the forward method returns the correct cosine distance
        """
        # Generate two random embeddings and their connection index
        emb_numpy, emb_tensor, edge_indices = create_example_embeddings(2)

        # Instantiate decoder and calculate cosine similarity
        dec = CosineSimDecoder()
        assert edge_indices.t()[1].detach().numpy().tolist() == [0, 1]
        sim = dec.forward(emb_tensor, edge_indices, sigmoid=False)

        # Calculate distance in scipy and convert to similartiy, cos_sim = 1.0 - cos_dist
        dist_scipy = 1.0 - cosine(emb_numpy[0], emb_numpy[1])

        # Compare only connection from node 0 to 1 (edge index 1)
        self.assertAlmostEqual(dist_scipy, sim[1].item(), places=5)

    def test_calculates_correct_pairwise_similarity(self):
        """
        Tests if the forward_all method returns the correct cosine distance for all nodes *pairwise*
        """
        # Generate multiple random embeddings
        n_nodes = 100
        emb_numpy, emb_tensor, edge_indices = create_example_embeddings(n_nodes)

        # Instantiate decoder and calculate pairwise cosine similarities
        dec = CosineSimDecoder()
        sim = dec.forward_all(emb_tensor, edge_indices, sigmoid=False).detach().numpy()

        # Calculate distances in scipy and convert to similartiy
        sim_matrix = np.empty((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                # cos_sim = 1.0 - cos_dist
                sim_matrix[i, j] = 1.0 - cosine(emb_numpy[i], emb_numpy[j])

        self.assertTrue(np.allclose(sim_matrix, sim))


class TestEuclideanDistanceDecoder(unittest.TestCase):

    def test_calculates_correct_single_similarity(self):
        """
        Tests if the forward method returns the correct euclidian distance
        """
        # Generate two random embeddings and their connection index
        emb_numpy, emb_tensor, edge_indices = create_example_embeddings(2, normalize=True)

        # Instantiate decoder and calculate cosine similarity
        dec = EuclideanDistanceDecoder()
        assert edge_indices.t()[1].detach().numpy().tolist() == [0, 1]
        sim = dec.forward(emb_tensor, edge_indices, sigmoid=False)

        # Calculate distance in scipy and convert to similartiy, cos_sim = 1.0 - cos_dist
        dist_scipy = 1.0 - euclidean(emb_numpy[0], emb_numpy[1])

        # Compare only connection from node 0 to 1 (edge index 1)
        self.assertAlmostEqual(dist_scipy, sim[1].item(), places=5)

    def test_calculates_correct_pairwise_similarity(self):
        """
        Tests if the forward_all method returns the correct euclidian distance for all nodes *pairwise*
        """
        # Generate multiple random embeddings
        N = 100
        emb_numpy, emb_tensor, edge_indices = create_example_embeddings(N, normalize=True)

        # Instantiate decoder and calculate pairwise similarities
        dec = EuclideanDistanceDecoder()
        sim = dec.forward_all(emb_tensor, edge_indices, sigmoid=False, normalize=False).detach().numpy()

        # Calculate distances in scipy and convert to similartiy
        sim_matrix = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                sim_matrix[i, j] = 1.0 - euclidean(emb_numpy[i], emb_numpy[j])

        self.assertTrue(np.allclose(sim_matrix, sim))


if __name__ == '__main__':
    unittest.main()
