import torch
import numpy as np


def sparse_precision_recall(data, pos_pred_indices):
    all_edges = extract_all_edges(data)
    pred = sorted(pos_pred_indices.coalesce().indices().t().detach().cpu().numpy().tolist())
    true = sorted(all_edges)
    print("Pred:", pred[:20])
    print("True:", true[:20])
    print(f"{len(pred)} edges detected by LSH out of {len(all_edges)} in total.")
    precision, recall = evaluate_edges(pred, true)
    return precision, recall


def evaluate_edges(pred, true):
    sum = 0.0
    for conn in pred:
        if conn in true:
            sum += 1.0
    precision = sum / len(pred)
    sum = 0.0
    for conn in true:
        if conn in pred:
            sum += 1.0
    recall = sum / len(true)
    return precision, recall


def dense_precision_recall(data, pos_pred_indices, min_sim=0.73):
    all_edges = extract_all_edges(data)
    pred = pos_pred_indices.detach().cpu().numpy() > min_sim

    pred = pred.nonzero()
    pred = np.array([pred[0].tolist(), pred[1].tolist()]).T.tolist()
    true = all_edges
    print(f"{len(pred)} edges detected out of {len(all_edges)} in total.")
    precision, recall = evaluate_edges(pred, true)
    return precision, recall


def extract_all_edges(data):
    return torch.cat((data.val_pos_edge_index,
                      data.test_pos_edge_index,
                      data.train_pos_edge_index), 1).t().detach().cpu().numpy().tolist()
