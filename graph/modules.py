import sys
from os.path import dirname, abspath

import scipy.sparse as sparse
from scipy.special import expit
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.autoencoder import InnerProductDecoder

sys.path.append(dirname(abspath(__file__)))
from graph.lsh import *


def _norm_batch(batch):
    """
    Normalizes all rows in 2D matrix to have norm 1
    :param batch:
    :return:
    """
    assert len(batch.size()) >= 2, "Needs tensor with dimensions >= 2"
    return batch / batch.norm(dim=1)[:, None]


class CosineSimDecoder(torch.nn.Module):
    """
    Calculates pairwise similarity of embeddings using the cosine similarity.
    cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
    Values lie between -1 and 1
    """

    def forward(self, z, edge_index, sigmoid=True):
        """
        Calculates the cosine similarity only for two nodes' connection
        :param z: Latent space Z, tensor of shape [n_nodes, n_embed_dim]
        :param edge_index: Calculate similarity only for this connection/ connected 2 nodes
        :param sigmoid: Whether or not to apply a sigmoid for the result
        :return: Scalar similarity score
        """
        a, b = z[edge_index[0]], z[edge_index[1]]
        value = F.cosine_similarity(a, b, dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        """
        Calculates the cosine similarity only for two nodes' connection
        :param z: Latent space Z, tensor of shape [n_nodes, n_embed_dim]
        :param edge_index: Calculate similarity only for this connection/ connected 2 nodes
        :param sigmoid: Whether or not to apply a sigmoid for the result
        :return: Scalar similarity score
        """
        z = _norm_batch(z)
        values = torch.mm(z, z.t())
        return torch.sigmoid(values) if sigmoid else values


class CosineSimHashDecoder(CosineSimDecoder):
    def forward_all(self, z, sigmoid=True, d=0.5):
        n_nodes = z.shape[0]

        pairs, _ = LSH(z.detach().cpu(), d=d, b=8, r=32)

        v1s, v2s, ds = zip(*(list(pairs)))

        diag_indices = np.diag_indices(n_nodes)[0]

        v1s = np.concatenate((v1s, diag_indices))
        v2s = np.concatenate((v2s, diag_indices))
        ds = np.concatenate((ds, np.zeros(n_nodes)))

        if not sigmoid:
            # Zeros get converted to 1s in the transformation below
            ds = np.zeros_like(ds)

        return torch.sparse.FloatTensor(torch.LongTensor([v1s, v2s]), 1 - torch.FloatTensor(ds) / 2,
                                        torch.Size([n_nodes, n_nodes]))


#         # DOK type sparse matrix has efficient changing of sparse structure
#         adjacency = sparse.dok_matrix(sparse.identity(len(z)))

#         for v1, v2, d in pairs:
#             adjacency[v1, v2] = adjacency[v2, v1] = 1 - d/2 if sigmoid else 1


class EuclideanDistanceDecoder(torch.nn.Module):
    """
        Calculates pairwise similarity of embeddings with the L2 norm.
        sim(u, v) = sqrt(sum((u - v)**2))
        Values lie between 0 and infinity
    """

    def forward(self, z, edge_index, sigmoid=True):
        """
        Calculates the Euclidean distance only for two nodes' connection
        :param z: Latent space Z, tensor of shape [n_nodes, n_embed_dim]
        :param edge_index: Calculate similarity only for this connection/ connected 2 nodes
        :param sigmoid: Whether or not to apply a sigmoid for the result
        :return: Scalar similarity score
        """
        a, b = z[edge_index[0]], z[edge_index[1]]
        a, b = _norm_batch(a), _norm_batch(b)

        # Note: Changed to 10 since sigmoid(10)=1 and that's what we want for 0 dist.
        distance = F.pairwise_distance(a, b, p=2)
        value = 1.0 - distance  # Euclidean distance in range (0, 2) for normalized vectors
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        # Calculates all pairwise similarities.
        # Note that using torch.nn.functional.pdist and transforming resulting matrix
        # to right shape might be faster
        z = _norm_batch(z)
        values = 1.0 - torch.norm(z[:, None] - z, dim=2, p=2)
        return torch.sigmoid(values) if sigmoid else values


class EuclideanDistanceHashDecoder(EuclideanDistanceDecoder):
    def forward_all(self, z, sigmoid=True, d=0.5, normalize=True, debug=False):

        n_nodes = z.shape[0]

        if normalize:
            z = _norm_batch(z)

        z = z.detach().cpu()

        offset = 1.0 if normalize else 10.0

        # Sample to get an idea of the distance to filter for

        sample_ix = np.random.choice(np.arange(len(z)), replace=False, size=min(10, len(z)))
        assert len(sample_ix) >= 2, "Not enough nodes to sample"
        sample = z[sample_ix]

        sample_distances = []

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                sample_distances.append(np.linalg.norm(sample[i] - sample[j]))

        # This is done in order to tackle the problem of unnormalized distance measures
        dist = min(sample_distances) + d * (max(sample_distances) - min(sample_distances))

        pairs, _ = LSH(z, d=dist, dist_func='euclidean', b=4, r=64)

        v1s, v2s, ds = zip(*(list(pairs)))

        diag_indices = np.diag_indices(n_nodes)[0]

        v1s = np.concatenate((v1s, diag_indices))
        v2s = np.concatenate((v2s, diag_indices))

        ds = expit(offset - np.array(ds)) if sigmoid else np.ones_like(ds)
        ds = np.concatenate((ds, np.ones(n_nodes)))

        return torch.sparse.FloatTensor(torch.LongTensor([v1s, v2s]), torch.FloatTensor(ds),
                                        torch.Size([n_nodes, n_nodes])) if not debug else (
            torch.sparse.FloatTensor(torch.LongTensor([v1s, v2s]), torch.FloatTensor(ds),
                                     torch.Size([n_nodes, n_nodes])),
            dist)


#         # DOK type sparse matrix has efficient changing of sparse structure
#         adjacency = sparse.dok_matrix(sparse.identity(len(z)))

#         # DOK matrices can be given a batch of updates via _update
#         sparse_indices = dict()

#         # Have to iterate twice in order to be able to return sigmoid-like values
#         # since Euclidean distance is in the range [0, inf)
#         max_dist = max(pairs, key=itemgetter(2))[2]

#         for v1, v2, d in pairs:
#             sparse_indices[(v1, v2)] = sparse_indices[(v2, v1)] = expit(offset - d) if sigmoid else 1

#         adjacency._update(sparse_indices)

#         return DOK_to_torch(adjacency) if not debug else (DOK_to_torch(adjacency), dist)


class InnerProductHashDecoder(InnerProductDecoder):
    def forward_all(self, z, sigmoid=True, d=0.25, debug=False, normalize=False):

        n_nodes = z.shape[0]

        if normalize:
            z = _norm_batch(z)

        z = z.detach().cpu()

        sample_ix = np.random.choice(np.arange(len(z)), replace=False, size=min(10, len(z)))
        assert len(sample_ix) >= 2, "Not enough nodes to sample"
        sample = z[sample_ix]

        sample_distances = []

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                sample_distances.append(sample[i] @ sample[j])

        # This is done in order to tackle the problem of unnormalized distance measures
        dist = (min(sample_distances) + d * (max(sample_distances) - min(sample_distances)))

        pairs, _ = LSH(z, d=dist, dist_func='dot', b=8, r=48)

        v1s, v2s, ds = zip(*(list(pairs)))

        diag_indices = np.diag_indices(n_nodes)[0]

        v1s = np.concatenate((v1s, diag_indices))
        v2s = np.concatenate((v2s, diag_indices))

        if sigmoid:
            ds = expit(ds)
        ds = np.concatenate((ds, np.ones(n_nodes)))

        return torch.sparse.FloatTensor(torch.LongTensor([v1s, v2s]), torch.FloatTensor(ds),
                                        torch.Size([n_nodes, n_nodes])) if not debug else (
            torch.sparse.FloatTensor(torch.LongTensor([v1s, v2s]), torch.FloatTensor(ds),
                                     torch.Size([n_nodes, n_nodes])),
            dist)


#         # DOK type sparse matrix has efficient changing of sparse structure
#         adjacency = sparse.dok_matrix(sparse.identity(len(z)))

#         # DOK matrices can be given a batch of updates via _update
#         sparse_indices = dict()

#         for v1, v2, d in pairs:
#             sparse_indices[(v1, v2)] = sparse_indices[(v2, v1)] = expit(d) if sigmoid else d

#         adjacency._update(sparse_indices)

#         return DOK_to_torch(adjacency) if not debug else (DOK_to_torch(adjacency), dist)


class GAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class VGAEEncoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(VGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(
            2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


def DOK_to_torch(X):
    assert type(X) == sparse.dok.dok_matrix
    coo = X.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def create_decoder(name, use_lsh=False):
    """
    Returns desired decoder version and LSH-Version if required.
    Available choices are ['dot', 'cosine', 'lsh']
    :return: torch.nn.Module object with implemented forward and forward_all method
    """
    if name == "dot":
        return InnerProductDecoder() if not use_lsh else InnerProductHashDecoder()
    elif name == "cosine":
        return CosineSimDecoder() if not use_lsh else CosineSimHashDecoder()
    elif name == "l2":
        return EuclideanDistanceDecoder() if not use_lsh else EuclideanDistanceHashDecoder()
    else:
        raise NotImplementedError(
            f"Decoder with name {name} and {'' if use_lsh else 'non-'}LSH version not implemented.")


def create_encoder(name, num_features, latent_dim):
    if name == 'gae':
        return GAEEncoder(num_features, latent_dim)
    else:
        return VGAEEncoder(num_features, latent_dim)
