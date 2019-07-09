import torch
from tqdm import tqdm
import numpy as np


def sparse_precision_recall(data, pos_pred_indices):
    print("Compute Sparse-Precision-Recall")
    all_edges = extract_all_edges(data)
    
    pred = pos_pred_indices.coalesce().indices().t().detach().cpu().numpy()
    pred = set(zip(pred[:, 0], pred[:, 1]))
    true = set(zip(all_edges[:, 0], all_edges[:, 1]))

    print(f"Sparse Precision-Recall: {len(pred)} edges detected by LSH out of {len(all_edges)} in total.")
    return evaluate_edges(pred, true)


def dense_precision_recall(data, pos_pred_indices, min_sim):
    print("Compute Dense-Precision-Recall")
    all_edges = extract_all_edges(data)
    pred = (pos_pred_indices.detach().cpu().numpy() > min_sim).nonzero()
    
    pred = set(zip(pred[0], pred[1]))
    true = set(zip(all_edges[:, 0], all_edges[:, 1]))
    
    print(f"Dense Precision-Recall: {len(pred)} edges detected out of {len(all_edges)} in total.")
    return evaluate_edges(pred, true)


def sparse_v_dense_precision_recall(dense_matrix, sparse_matrix, sim_threshold):
    """
    Compares Sparse-Adjacency matrix (LSH-Version) to the Dense-Adjacency matrix (non-LSH-version), which serves as GT.
    :param dense_matrix:
    :param sparse_matrix:
    :param sim_threshold:
    :return:
    """
    dense_pred = (dense_matrix.detach().cpu().numpy() > sim_threshold).nonzero()
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
                
    precision = sum / len(pred) if len(pred) != 0 else 0
    
    sum = 0.0
    
    for conn in tqdm(true, desc="Checking recall"):
        if conn in pred:
            sum += 1.0        

    recall = sum / len(true) if len(pred) != 0 else 0
    return precision, recall


def extract_all_edges(data):
    return torch.cat((data.val_pos_edge_index,
                      data.test_pos_edge_index,
                      data.train_pos_edge_index), 1).t().detach().cpu().numpy()
