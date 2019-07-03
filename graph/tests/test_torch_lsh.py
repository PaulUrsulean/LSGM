import unittest
import torch
import numpy as np
from os.path import dirname, abspath
import sys
from itertools import combinations

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))


from graph.torch_lsh import LSHDecoder
from graph.torch_lsh import CosineSimilarity


class TestCosineSimilarity(unittest.TestCase):


	def test_signature_matrix_shape(self):
		N, D = 100, 16
		bands, rows = 12, 6

		X = torch.randn((N, D))
		sim = CosineSimilarity(bands, rows)
		signature_matrix = sim.signature(X)

		self.assertEqual(list(signature_matrix.size()), [N, bands, rows])




class TestLSHDecoder(unittest.TestCase):

	def test_indices_to_connections(self):
		dec = LSHDecoder()
		indices = [1, 7, 10]
		connections = combinations(indices, 2)

		true_connections = [(1, 7), (7, 1), (1, 10), (10, 1), (7, 10), (10, 7)]

		self.assertEqual(set(connections), set(true_connections))


	def test_returns_correct_items(self):

		emb = np.array([[0.0, 1.0],
						[0.0, -1.0],
						[1.0, 0.0],
						[0.05, 0.9]
						], dtype=np.float32)
		emb = np.random.normal(size=(100000, 128)).astype(np.float32)
		emb_tensor = torch.tensor(emb)

		dec = LSHDecoder(bands=20, rows=5, verbose=True)
		adj = dec(emb_tensor)
		indices = adj.coalesce().indices().t().detach().numpy()  
		#print(indices)

		self.assertTrue(len(indices) == 2)
		self.assertTrue([0, 3] in indices and [3, 0] in indices)

	def test_true(self):

  		self.assertEqual(True, False)



if __name__ == '__main__':
    unittest.main()