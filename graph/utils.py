import torch
from tqdm import tqdm
import numpy as np


def sparse_precision_recall(data, pos_pred_indices):
    print("Compute Sparse-Precision-Recall")
    all_edges = extract_all_edges(data)
    pred = sorted(pos_pred_indices.coalesce().indices().t().detach().cpu().numpy().tolist())
    true = sorted(all_edges)
    print("Pred:", pred[:20])
    print("True:", true[:20])
    print(f"{len(pred)} edges detected by LSH out of {len(all_edges)} in total.")
    return evaluate_edges(pred, true)

def dense_precision_recall(data, pos_pred_indices, min_sim=0.73):
    print("Compute Dense-Precision-Recall")
    print("Similarity Threshold: ", min_sim)
    all_edges = extract_all_edges(data)
    pred = pos_pred_indices.detach().cpu().numpy() > min_sim

    pred = pred.nonzero()
    pred = np.array([pred[0].tolist(), pred[1].tolist()]).T.tolist()
    true = all_edges
    print(f"{len(pred)} edges detected out of {len(all_edges)} in total.")
    return evaluate_edges(pred, true)

def sparse_v_dense_precision_recall(dense_matrix, sparse_matrix, sim_threshold):
    dense_pred = (dense_matrix.detach().cpu().numpy() > sim_threshold).nonzero()
    dense_pred = np.array([dense_pred[0].tolist(), dense_pred[1].tolist()]).T.tolist()
    
    sparse_pred = sorted(sparse_matrix.coalesce().indices().t().detach().cpu().numpy().tolist())
    print(f"LSH found {len(sparse_pred)} edges out of {len(dense_pred)} edges that the naive version predicted.")
    
    return evaluate_edges(sparse_pred, dense_pred)

def evaluate_edges(pred, true):
    sum = 0.0
    progress_bar = tqdm(total = len(pred) + len(true))
    progress_bar.set_description("Checking precision and recall")
    
    for conn in pred:
        if conn in true:
            sum += 1.0
        progress_bar.update()
                
    precision = sum / len(pred) if len(pred) != 0 else 0
    
    sum = 0.0
    
    for conn in true:
        if conn in pred:
            sum += 1.0
        progress_bar.update()
        
    progress_bar.close()

    recall = sum / len(true) if len(pred) != 0 else 0
    return precision, recall

def extract_all_edges(data):
    return torch.cat((data.val_pos_edge_index,
                      data.test_pos_edge_index,
                      data.train_pos_edge_index), 1).t().detach().cpu().numpy().tolist()
