import math
from os import path as osp

import numpy as np
import torch
from torch_geometric import transforms as T
from torch_geometric.datasets import CoraFull, Coauthor, Planetoid, Reddit
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from graph.datasets.amazon import Amazon


def sparse_precision_recall(data, sparse_matrix):
    print("Compute Sparse-Precision-Recall")
    all_edges = extract_all_edges(data)

    pred = sparse_matrix.coalesce().indices().t().detach().cpu().numpy()
    pred = set(zip(pred[:, 0], pred[:, 1]))
    true = set(zip(all_edges[:, 0], all_edges[:, 1]))

    print(f"Sparse Precision-Recall: {len(pred)} edges detected by LSH out of {len(all_edges)} in total.")
    return evaluate_edges(pred, true)


def dense_precision_recall(data, dense_matrix, min_sim, distance_measure):
    print("Compute Dense-Precision-Recall")

    all_edges = extract_all_edges(data)
    pred = (dense_matrix.detach().cpu().numpy() > min_sim).nonzero()

    pred = set(zip(pred[0], pred[1]))
    true = set(zip(all_edges[:, 0], all_edges[:, 1]))

    print(f"Dense Precision-Recall: {len(pred)} edges detected out of {len(all_edges)} in total.")
    return evaluate_edges(pred, true)


def sparse_v_dense_precision_recall(dense_matrix, sparse_matrix, min_sim):
    """
    Compares Sparse-Adjacency matrix (LSH-Version) to the Dense-Adjacency matrix (non-LSH-version), which serves as GT.
    :param dense_matrix:
    :param sparse_matrix:
    :param min_sim_percentile:
    :return:
    """

    dense_pred = (dense_matrix.detach().cpu().numpy() > min_sim).nonzero()
    dense_pred = set(zip(dense_pred[0], dense_pred[1]))

    sparse_pred = sparse_matrix.coalesce().indices().t().detach().cpu().numpy()
    sparse_pred = set(zip(sparse_pred[:, 0], sparse_pred[:, 1]))

    print(f"LSH found {len(sparse_pred)} edges out of {len(dense_pred)} edges that the naive version predicted.")
    return evaluate_edges(sparse_pred, dense_pred)


def evaluate_edges(pred, true):
    sum = 0.0

    for conn in tqdm(pred, desc="Checking precision"):
        if conn in true:
            sum += 1.0

    precision = (sum / len(pred)) if len(pred) != 0 else 0

    sum = 0.0

    for conn in tqdm(true, desc="Checking recall"):
        if conn in pred:
            sum += 1.0

    recall = (sum / len(true)) if len(pred) != 0 else 0
    return precision, recall


def extract_all_edges(data):
    return torch.cat((data.val_pos_edge_index,
                      data.test_pos_edge_index,
                      data.train_pos_edge_index), 1).t().detach().cpu().numpy()


def sample_percentile(q, matrix_or_embeddings, dist_measure=None, sigmoid=False, sample_size=20000):
    """
    :param q: The percentile to look for the corresponding value in the pairs. In [0, 1]
    :param matrix_or_embeddings: As the name suggests, this param can either be the dense (N, N) adjacency matrix with values already computed, or the (N, D) matrix of embeddings.
    :param dist_measure: If given the matrix of embeddings, the distances must be computed directly in this function
    :param sigmoid: Whether to sigmoid computed distances. Only valid if embeddings are given.
    """

    assert q <= 1.0 and q >= 0.0, "Invalid value for q"
    assert isinstance(matrix_or_embeddings, torch.Tensor), "matrix_or_embeddings is not a torch Tensor."

    N_1, N_2 = matrix_or_embeddings.shape
    sample_size = min(sample_size, int(N_1 / 10))
    sample_a = np.random.choice(np.arange(N_1), size=sample_size, replace=False)
    sample_b = np.random.choice(np.arange(N_1), size=sample_size, replace=False)

    # Matrix case
    if N_1 == N_2:
        sample_distances = matrix_or_embeddings[sample_a, sample_b].detach()

    # Embeddings case
    else:
        assert N_1 > N_2, "Dimensions of embeddings bigger than n_nodes, something might be wrong."
        assert dist_measure in ['cosine', 'dot'], "dist_measure must be set as 'cosine' or 'dot'"

        sample_a, sample_b = matrix_or_embeddings[sample_a].detach(), matrix_or_embeddings[sample_b].detach()

        # If cosine just normalize vectors
        if dist_measure == 'cosine':
            sample_a /= torch.norm(sample_a, dim=1)[:, None]
            sample_b /= torch.norm(sample_b, dim=1)[:, None]

        sample_distances = torch.mm(sample_a, sample_b.t())

        if sigmoid:
            sample_distances = torch.sigmoid(sample_distances)

    return np.percentile(sample_distances.cpu().numpy(), q * 100)


def load_data(dataset_name):
    """
    Loads required data set and normalizes features.
    Implemented data sets are any of type Planetoid and Reddit.
    :param dataset_name: Name of data set
    :return: Tuple of dataset and extracted graph
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)

    if dataset_name == 'cora_full':
        dataset = CoraFull(path, T.NormalizeFeatures())
    elif dataset_name.lower() == 'coauthor':
        dataset = Coauthor(path, 'Physics', T.NormalizeFeatures())
    elif dataset_name.lower() == 'reddit':
        dataset = Reddit(path, T.NormalizeFeatures())
    elif dataset_name.lower() == 'amazon':
        dataset = Amazon(path)
    else:
        dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())


    print(f"Loading data set {dataset_name} from: ", path)
    data = dataset[0]  # Extract graph
    return dataset, data