from typing import List

from matplotlib.axis import Axis
from mpl_toolkits.basemap import Basemap
from numpy.core.multiarray import ndarray
from matplotlib.pyplot import get_cmap

def plot_interactions(locations: List[str],
                      latent_graph: ndarray,
                      map: Basemap,
                      ax: Axis,
                      skip_first: bool = False):
    """
    Given station ids and latent graph plot edges in different colors
    """

    # Transform lan/lot into region-specific values
    pixel_coords = [map(*coords) for coords in locations]

    # Draw contours and borders
    map.shadedrelief()
    map.drawcountries()
    # m.bluemarble()
    # m.etopo()


    # Plot Locations of weather stations
    for i, (x, y) in enumerate(pixel_coords):
        ax.plot(x, y, 'ok', markersize=10, color='yellow')
        ax.text(x + 10, y + 10, "Station " + str(i), fontsize=20, color='yellow');

    # Infer number of edge types and atoms from latent graph
    n_atoms = latent_graph.shape[-1]
    n_edge_types = latent_graph.shape[0]

    color_map = get_cmap('Set1')

    for i in range(n_atoms):
        for j in range(n_atoms):
            for edge_type in range(n_edge_types):
                if latent_graph[edge_type, i, j] > 0.5:

                    if skip_first and edge_type == 0:
                        continue

                    # Draw line between points
                    x = locations[i]
                    y = locations[j]
                    map.drawgreatcircle(x[0], x[1], y[0], y[1],
                                        color=color_map(edge_type - 1),
                                        label=str(edge_type))
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    return ax
