import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


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

    def forward_all(self, z, edge_index, sigmoid=True):
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
    def forward_all(self, z, edge_index, sigmoid=True):
        # pairs = lsh...
        # matrix = pairs to matrix
        raise NotImplementedError()


class EuclideanDistanceDecoder(torch.nn.Module):
    """
        Calculates pairwise similarity of embeddings with the L2 norm.
        sim(u, v) = sqrt(sum((u - v)**2))
        Values lie between 0 and infinity
    """

    def forward(self, z, edge_index, sigmoid=True, normalize=True):
        """
        Calculates the cosine similarity only for two nodes' connection
        :param z: Latent space Z, tensor of shape [n_nodes, n_embed_dim]
        :param edge_index: Calculate similarity only for this connection/ connected 2 nodes
        :param sigmoid: Whether or not to apply a sigmoid for the result
        :return: Scalar similarity score
        """
        a, b = z[edge_index[0]], z[edge_index[1]]
        if normalize:
            a, b = _norm_batch(a), _norm_batch(b)

        # Distance is in range [0, 2] as values are normalized
        distance = F.pairwise_distance(a, b, p=2)
        value = 1.0 - distance
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, edge_index, sigmoid=True, normalize=True):
        if normalize:
            z = _norm_batch(z)

        # Calculates all pairwise similarities.
        # Note that using torch.nn.functional.pdist and transforming resulting matrix
        # to right shape might be faster
        values = 1.0 - torch.norm(z[:, None] - z, dim=2, p=2)
        return torch.sigmoid(values) if sigmoid else values


class Encoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels, args):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
