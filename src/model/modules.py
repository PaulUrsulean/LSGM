import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoders
# MLP Encoder First


def node2edge(m, adj_rec, adj_send):
    """
    Calculates edge embeddings
    :param m: Tensor with shape (N_OBJ, N_HIDDEN)
    :param adj_rec:
    :param adj_send:
    :return:
    """
    outgoing = torch.matmul(adj_send, m)
    incoming = torch.matmul(adj_rec, m)
    return torch.cat((outgoing, incoming), dim=2)


def edge2node(m, adj_rec, adj_send):
    """
    Performs accumulation of message passing by summing over connected edges for each node
    :param x: tensor with shape (N_OBJ, N_HIDDEN)
    :param adj_rec:
    :param adj_send:
    :return:
    """
    outgoing = torch.matmul(adj_rec.t(), m)
    return outgoing


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, keep_prob=1.):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout = nn.Dropout(1.0 - keep_prob)
        self.batchnorm = nn.BatchNorm1d(output_size)

        self.linear_input_hidden = nn.Linear(input_size, hidden_size)
        self.linear_hidden_out = nn.Linear(hidden_size, output_size)

        # TODO Weights initialization

    def shape_invariant_batchnorm(self, input):
        """
        As we use a 1d-batchnorm but want to allow for arbitarary tensor shapes (e.g. running for multiple timesteps and
        objects in one pass), we have to scale the tensor before passing it into the operation
        :return: Tensor with same shape but batchnorm applied
        """
        x = input.view(input.size(0) * input.size(1),
                       -1)  # [N_TIMESTEPS, N_OBJ, N_FEAT] -> [N_TIMESTEPS * N_OBJ, N_FEAT]
        x = self.batchnorm(x)
        x = x.view(input.size(0), input.size(1), -1)  # -> [N_TIMESTEPS, N_OBJ, N_FEAT]
        return x

    def forward(self, input):
        x = self.linear_input_hidden(input)
        x = F.elu(x)
        x = self.linear_hidden_out(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.shape_invariant_batchnorm(x)
        return x


class MLPEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, keep_prob=1., n_layers=2):
        super(MLPEncoder, self).__init__()

        self.mlp1 = MLP(input_size, hidden_size, hidden_size)
        self.mlp2 = MLP(hidden_size * 2, hidden_size, hidden_size)
        self.mlp3 = MLP(hidden_size, hidden_size, hidden_size)
        self.mlp4 = MLP(hidden_size * 2, hidden_size, hidden_size)
        self.mlp_out = nn.Linear(hidden_size, output_size)

    def forward(self, input, adj_rec, adj_send):
        """
        Implementation from (Kipf et al, 2018)
        :param input: Tensor of shape TODO
        :return:
        """

        # Transform to column-stacking o ftimesteps and features
        x = input.view(input.size(0), input.size(1), -1)

        x = self.mlp1(x)
        x = node2edge(x, adj_rec, adj_send)
        x = self.mlp2(x)
        x_skip = x

        x = edge2node(x, adj_rec, adj_send)
        x = self.mlp3(x)
        # x = node2edge(x, adj_rec, adj_send)

        x = torch.cat((x, x_skip), dim=2)
        x = self.mlp4(x)
        return self.mlp_out(x)
