from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def edges_to_adj(edges, n_atoms):
    n_samples = edges.size(0)
    indices = get_offdiag_indices(n_atoms)
    n_edge_types = edges.size(-1)

    graphs = np.empty((n_samples, n_edge_types, n_atoms, n_atoms))

    for sample in range(n_samples):
        for edge_type in range(n_edge_types):
            graph = edges[sample, :, edge_type]
            fully_connected = torch.zeros(n_atoms * n_atoms)
            fully_connected[indices] = graph
            adjacency_matrix = fully_connected.view(n_atoms, n_atoms).detach().numpy()
            graphs[sample, edge_type, :, :] = adjacency_matrix

    return graphs


def id_2_loc(i):
    """
    Remove when real function is provided
    :param i:
    :return:
    """
    lat = np.random.rand() * 8 + 36
    lon = np.random.rand() * 8 - 3.9
    return (lon, lat)




def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices
