import os.path as path
import sys
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


class LSHDistanceMetric(ABC):

    @abstractmethod
    def sim(v1, v2):
        pass

    @abstractmethod
    def pairwise_sim(embeddings):
        pass

    @abstractmethod
    def dist(v1, v2):
        pass

    @abstractmethod
    def signature(X):
        pass


class CosineSimilarity(LSHDistanceMetric):

    def __init__(self, bands: int, rows: int):
        super(CosineSimilarity, self).__init__()
        self.bands = bands
        self.rows = rows

    def sim(self, v1, v2):
        return F.cosine_similarity(v1, v2, dim=0)

    def pairwise_sim(self, embeddings):
        embeddings = embeddings / torch.norm(embeddings, dim=1)[:, None]
        return torch.mm(embeddings, embeddings.t())

    def dist(self, v1, v2):
        return 1 - self.sim(v1, v2)

    def signature(self, X: torch.Tensor):
        """
        :return: signature matrix with shape (bands, rows, n_samples)
        """
        device = X.device
        N, D = X.shape

        distribution = MultivariateNormal(torch.zeros(D), torch.eye(D))
        random_planes = distribution.sample((self.bands * self.rows,)).to(device)

        # signature_matrix is (b*r) x N
        signature_matrix = (torch.mm(random_planes, X.t()) >= 0).int() * 2 - 1

        return signature_matrix.reshape(self.bands, self.rows, N)


class DotProductSimilarity(LSHDistanceMetric):

    def __init__(self, bands: int, rows: int):
        super(DotProductSimilarity, self).__init__()
        self.bands = bands
        self.rows = rows

    def sim(self, v1, v2):
        return v1.dot(v2)

    def dist(self, v1, v2):
        raise NotImplementedError()

    def signature(self, X: torch.Tensor):
        """
        :return: signature matrix with shape (n_samples, bands, rows)
        """
        device = X.device
        N, D = X.shape

        augment_col = X.norm(dim=1)
        augmented = X / augment_col.max()
        augment_col = torch.sqrt(1 - augmented.norm(dim=1))[:, None]

        augmented = np.cat((augmented, augment_col), dim=1)

        distribution = MultivariateNormal(torch.zeros(D), torch.eye(D))
        random_planes = distribution.sample((self.bands * self.rows,)).to(device)

        signature_matrix = (torch.mm(random_planes, augmented.t()) >= 0).int() * 2 - 1
        return signature_matrix.reshape(self.bands, self.rows, N)


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
        self.sim_metric_str = 'cosine' if metric is CosineSimilarity else 'dot'
        self.sim_metric = metric(bands, rows)
        self.assure_correctness = assure_correctness

    def recover_duplicates(self, signature_matrix, embeddings):

        N, D = embeddings.shape
        assert list(signature_matrix.shape) == [self.bands, self.rows, N]

        # Don't need to use .cpu().numpy() if we just want to access the data
        # https://discuss.pytorch.org/t/get-value-out-of-torch-cuda-float-tensor/2539/4
        signature = signature_matrix.detach()

        bands_loop = tqdm(range(self.bands), desc="Hashing values in signature matrix") if self.verbose else range(
            self.bands)

        pairs_indices = torch.LongTensor().to(signature_matrix.device)
        pairs_similarites = torch.FloatTensor().to(signature_matrix.device)

        print(f"Size of Signature: {signature.element_size() * signature.nelement() / 10 ** 6}")

        for band in bands_loop:
            hashtable = defaultdict(list)

            elems_loop = tqdm(range(N), desc=f"Hashing nodes to buckets") if self.verbose else range(N)

            for i in elems_loop:
                # Only bring one band column to the CPU at a time to reduce memory overhead
                key = hash(signature[band, :, i].cpu().numpy().data.tobytes())
                hashtable[key].append(i)

            dupes_loop = tqdm(hashtable.items(),
                              desc=f"Checking elements in same bucket") if self.verbose else hashtable.items()

            for _, duplicates in dupes_loop:
                if len(duplicates) < 2:
                    continue

                duplicates_embeddings = embeddings[duplicates]
                duplicates = torch.Tensor(duplicates)

                pairwise_sim = self.sim_metric.pairwise_sim(duplicates_embeddings)

                # Remove self loops manually
                diagonal_indices = np.diag_indices(duplicates.size(0))
                pairwise_sim[diagonal_indices[0], diagonal_indices[1]] = -np.inf

                # Calculate connections with high enough similarity
                pairs = pairwise_sim >= (self.sim_thresh if self.assure_correctness else -np.inf)

                # Nonzero values are now wanted connections, add to existing ones
                nonzero_indices = pairs.nonzero()

                # Add distances to array. TODO: Potentially just add 1 and use LongTensor instead of FloatTensor
                pairs_similarites = torch.cat((pairs_similarites,
                                               pairwise_sim[nonzero_indices[:, 0], nonzero_indices[:, 1]]), dim=0)

                # Convert local indices (only ones in bucket) to global node indices
                nonzero_indices[:, 0] = duplicates[nonzero_indices[:, 0]]
                nonzero_indices[:, 1] = duplicates[nonzero_indices[:, 1]]
                pairs_indices = torch.cat((pairs_indices, nonzero_indices), dim=0)

        return torch.sparse.FloatTensor(
            indices=pairs_indices.t(),
            values=pairs_similarites,
            size=torch.Size([N, N])
        )

    def forward(self, Z):
        signature_matrix = self.sim_metric.signature(Z)
        pairs_matrix = self.recover_duplicates(signature_matrix, Z)

        return pairs_matrix
