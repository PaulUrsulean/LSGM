import unittest

import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean

from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from lsh import *

def create_example_embeddings(n_nodes, embed_dim=128, normalize=False):
    """ Creates random node embeddings for a fully connected graph
    :param n_nodes: Number of nodes in the graph
    :param normalize: If embeddings should be scaled to unit norm
    :return: tuple with embeddings as ndarray, tensor and edge indices """
    emb_numpy = np.random.random((n_nodes, embed_dim))
    if normalize:
        emb_numpy = emb_numpy / np.linalg.norm(emb_numpy, axis=1)[:, np.newaxis]

    emb_tensor = torch.tensor(emb_numpy, dtype=torch.float32)

    # Make graph fully connected (with self loops)
    edge_indices = sum([[i] * n_nodes for i in range(n_nodes)], [])
    edge_indices = torch.tensor([edge_indices, list(range(n_nodes)) * n_nodes], dtype=torch.long)
    return emb_numpy, emb_tensor, edge_indices


class TestLSH(unittest.TestCase):

    def test_lsh_cosine(self):
        """
        Tests if the lsh correctly estimates cosine distance
        """
        emb_numpy, emb_tensor, edge_indices = create_example_embeddings(100)
        
        distances = []
        n_eligible = 0
        
        for i in range(len(emb_numpy)):
            for j in range(i+1, len(emb_numpy)):
                distances.append(cosine(emb_numpy[i], emb_numpy[j]))
                                
        test_dist = (max(distances) + min(distances)) / 2
        
        n_eligible = (np.array(distances) < test_dist).sum()
        
        closest_pairs, _ = LSH(emb_numpy, d=test_dist, dist_func = 'cosine', r=8, b=64)
        
        print("Cosine sim pairs generated: {} from {} total, with {} distance".format(len(closest_pairs), n_eligible, test_dist))
        
        for v1, v2, d in closest_pairs:
            self.assertLessEqual(cosine(emb_numpy[v1], emb_numpy[v2]), test_dist)
            
    def test_lsh_euclidean_dist(self):
        """
        Tests if the lsh correctly estimates euclidean distance
        """ 
        emb_numpy, emb_tensor, edge_indices = create_example_embeddings(100)
        
        distances = []
        n_eligible = 0
        
        for i in range(len(emb_numpy)):
            for j in range(i+1, len(emb_numpy)):
                distances.append(euclidean(emb_numpy[i], emb_numpy[j]))
                                
        test_dist = (max(distances) + min(distances)) / 2
        
        n_eligible = (np.array(distances) < test_dist).sum()
        
        closest_pairs, _ = LSH(emb_numpy, d=test_dist, dist_func = 'euclidean')
        
        print("Euclidean sim pairs generated: {} from {} total, with {} distance".format(len(closest_pairs), n_eligible, test_dist))
        
        for v1, v2, d in closest_pairs:
            self.assertLessEqual(euclidean(emb_numpy[v1], emb_numpy[v2]), test_dist)
            
    def test_lsh_dot_product(self):
        """
        Tests if the lsh correctly estimates dot product
        """ 
        emb_numpy, emb_tensor, edge_indices = create_example_embeddings(100)
        
        distances = []
        n_eligible = 0
        
        for i in range(len(emb_numpy)):
            for j in range(i+1, len(emb_numpy)):
                distances.append(emb_numpy[i] @ emb_numpy[j])
                                
        test_dist = (max(distances) + min(distances)) / 2
        
        n_eligible = (np.array(distances) < test_dist).sum()
        
        closest_pairs, _ = LSH(emb_numpy, d=test_dist, dist_func = 'dot', b=32, r=8)
        
        print("Dot prod sim pairs generated: {} from {} total, with {} distance".format(len(closest_pairs), n_eligible, test_dist))
        
        for v1, v2, d in closest_pairs:
            self.assertLessEqual(euclidean(emb_numpy[v1], emb_numpy[v2]), test_dist)


if __name__ == '__main__':
    unittest.main()
