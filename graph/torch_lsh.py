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

    @abstractmethod
    def dist(v1, v2):
        pass


class CosineSimilarity(LSHDistanceMetric):

    def __init__(self, bands: int, rows: int):
        super(CosineSimilarity, self).__init__()
        self.bands = bands
        self.rows = rows

    def sim(self, v1, v2):
        return F.cosine_similarity(v1, v2, dim=0)
    
    def dist(self, v1, v2):
        return 1 - self.sim(v1, v2)

    def signature(self, X: torch.Tensor):
        """
        :return: signature matrix with shape (n_samples, bands, rows)
        """
        device = X.device
        N, D = X.shape
        
        distribution = MultivariateNormal(torch.zeros(D), torch.eye(D))
        random_planes = distribution.sample_n(self.bands * self.rows).to(device)
        
        # Projections is (b*r) x N
        signature_matrix = (torch.mm(random_planes, X.t()) >= 0).int() * 2 - 1
        
#         # unit8 underflows to 255 in when applying -1
#         projections = torch.mm(random_planes, X.t())
#         sigature_matrix = (projections >= 0).int() * 2 - 1  # +1 if >0 else -1
        
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
        self.sim_metric = metric(bands, rows)
        self.assure_correctness = assure_correctness
        
    def recover_duplicates(self, signature_matrix, embeddings):
        
        N, D = embeddings.shape
        pairs_indices = set()
        assert list(signature_matrix.shape) == [self.bands, self.rows, N]
        
        # Don't need to use .cpu().numpy() if we just want to access the data
        # https://discuss.pytorch.org/t/get-value-out-of-torch-cuda-float-tensor/2539/4
        signature = signature_matrix.detach()
        
        if self.verbose:
            progress_bar = tqdm(total=self.bands * N)
            progress_bar.set_description("Hashing values in signature matrix")
        
        for band in range(self.bands):
            hashtable = defaultdict(list)
            
            for i in range(N):
                # Only bring one band column to the CPU at a time to reduce memory overhead
                key = hash(signature[band, :, i].cpu().numpy().data.tobytes())
                hashtable[key].append(i)
                
                if self.verbose:
                    progress_bar.update()

            for _, duplicates in hashtable.items():
                if len(duplicates) < 2:
                    continue
                    
                for i, j in combinations(duplicates, 2):
                    if self.assure_correctness:
                        
                        similarity = self.sim_metric.sim(Z[i], Z[j])
                        
                        # TODO: Also return distances
                        if similarity > self.sim_thresh:
                            pairs_indices.add((i, j))
                            pairs_indices.add((j, i))
                    else:
                        pairs_indices.add((i, j))
                        pairs_indices.add((j, i))
     
        if self.verbose:
            progress_bar.close()
            
        pairs_indices = np.asarray(list(pairs_indices)).T
        
        return torch.sparse.FloatTensor(torch.LongTensor(pairs_indices),
                                         torch.ones(pairs_indices.shape[1]),
                                         torch.Size([N, N]))

    def forward(self, Z):
        signature_matrix = self.sim_metric.signature(Z)
        pairs_matrix = self.recover_duplicates(signature_matrix, Z)

        return pairs_matrix
