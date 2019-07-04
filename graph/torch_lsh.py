from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations, combinations_with_replacement

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm


class LSHDistanceMetric(ABC):

    @abstractmethod
    def sim(v1, v2):
        pass


class CosineSimilarity(LSHDistanceMetric):

    def __init__(self, bands: int, rows: int):
        super(CosineSimilarity, self).__init__()
        self.bands = bands
        self.rows = rows

    def sim(self, v1, v2):
        return F.cosine_similarity(v1, v2, dim=0)

    def signature(self, X: torch.Tensor):
        """

        :return: signature matrix with shape (n_samples, bands, rows)
        """
        device = X.device
        dims = X.size(1)
        m = MultivariateNormal(torch.zeros(dims), torch.eye(dims))
        #random_planes = torch.randn((self.bands * self.rows, dims)).to(device)
        random_planes = m.sample_n(self.bands * self.rows).to(device)
        projections = torch.mm(random_planes, X.t())
        sigature_matrix = (projections >= 0) * 2 - 1  # +1 if >0 else -1
        return sigature_matrix.reshape(X.size(0), self.bands, self.rows)


class LSHDecoder(torch.nn.Module):

    def __init__(self,
                 sim_thresh: float = 0.5,
                 bands: int = 16,
                 rows: int = 8,
                 metric: LSHDistanceMetric = CosineSimilarity,
                 verbose: bool = False,
                 assure_correctness=True):

        super(LSHDecoder, self).__init__()
        self.sim_thresh = sim_thresh
        self.bands = bands
        self.rows = rows
        self.verbose = verbose
        self.sim_metric = metric(bands, rows)
        self.assure_correctness = assure_correctness

    def indices_to_connections(self, duplicates_list, Z=None):
        edges = []
        values = []
        for i_a in range(len(duplicates_list)):
            for i_b in range(i_a + 1, len(duplicates_list)):
                if Z is not None:
                    sim = self.sim_metric.sim(Z[duplicates_list[i_a]], Z[duplicates_list[i_b]])
                    if sim >= self.sim_thresh:
                        edges.append((duplicates_list[i_a], duplicates_list[i_b]))
                        values += [sim]
                else:
                    edges.append((duplicates_list[i_a], duplicates_list[i_b]))
                    values += [1.]
        return edges

    def pairs_from_signature(self, signature_matrix, Z):
        # Signature matrix should have shape [nodes, bands, rows]
        assert (list(signature_matrix.size()) == [int(signature_matrix.size(0)), self.bands, self.rows])

        pairs_values = []
        pairs_indices = set()
        # Hash for each node all rows for each band
        # Set (i, j) in a sparse matrix to 1 if they collide

        # For now move signature matrix to cpu and convert to numpy -> Do in pytorch and implement own hashing function to speed it up!
        sig_mat = signature_matrix.detach().cpu().numpy()

        if self.verbose:
            progress_bar = tqdm(total=self.bands * signature_matrix.size(0))
            progress_bar.set_description("Hashing values in signature matrix")
            curr_iter = 0
        for band in range(self.bands):
            hashtable = defaultdict(list)

            for elem_i in range(signature_matrix.size(0)):
                hash_key = hash(sig_mat[elem_i, band, :].data.tobytes())
                hashtable[hash_key].append(elem_i)
                if self.verbose:
                    progress_bar.update()
                    curr_iter = curr_iter + 1

            for (hash_val, duplicates) in hashtable.items():
                if len(duplicates) <= 1:
                    continue
                pairwise_indices = combinations(duplicates, 2)  # TODO: Check true distance
                for a, b in pairwise_indices:
                    if self.assure_correctness:
                        dist = self.sim_metric.sim(Z[a], Z[b])
                        if dist > self.sim_thresh:
                            pairs_indices.add((a, b))
                            pairs_indices.add((b, a))
                    else:
                        pairs_indices.add((a, b))
                        pairs_indices.add((b, a))


        if self.verbose:
            progress_bar.close()

        pairs_indices = np.asarray(list(pairs_indices)).T
        # pairs_indices, pairs_indices_unique = np.unique(pairs_indices, return_index=True)
        pairs = torch.sparse.FloatTensor(torch.LongTensor(pairs_indices),
                                         torch.ones(pairs_indices.shape[1]),
                                         torch.Size([int(signature_matrix.size(0)), int(signature_matrix.size(0))]))

        return pairs

    def forward(self, Z):
        signature_matrix = self.sim_metric.signature(Z)
        pairs_matrix = self.pairs_from_signature(signature_matrix, Z)
        return pairs_matrix
