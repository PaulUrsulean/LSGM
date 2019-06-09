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


def plot_interactions(station_ids: List, latent_graph: np.ndarray, map):
    """
    Given station ids and latent graph plot edges in different colors
    """
    fig = plt.figure(figsize=(8, 8))

    locations = [id_2_loc(i) for i in station_ids]
    pixel_coords = [map(*coords) for coords in locations]

    map.shadedrelief()
    map.drawcountries()
    # m.bluemarble()
    # m.etopo()

    # Plot Locations
    for i, (x, y) in enumerate(pixel_coords):
        plt.plot(x, y, 'ok', markersize=5, color='white')
        plt.text(x, y, str(i), fontsize=12, color='white');

    # Draw Latent Graph
    n_atoms = len(station_ids)
    n_edge_types = latent_graph.shape[-1]

    colors = plt.get_cmap('Set1')

    for i in range(n_atoms):
        for j in range(n_atoms):
            for edge_type in range(n_edge_types):
                if latent_graph[i, j, edge_type] > 0.5:
                    # Draw Line
                    # x = pixel_coords[i]
                    # y = pixel_coords[j]
                    x = locations[i]
                    y = locations[j]
                    # plt.plot(x, y, color=colors(edge_type), lw=5)
                    map.drawgreatcircle(x[0], x[1], y[0], y[1], color=colors(edge_type))
    plt.figure()


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices
