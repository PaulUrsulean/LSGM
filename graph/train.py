import argparse
import os.path as osp
import time

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CoraFull
from torch_geometric.nn import GAE, VGAE

import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from graph.early_stopping import EarlyStopping
from graph.modules import *


def load_data(dataset_name):
    """ Loads required data set and normalizes features.
    Implemented data sets are any of type Planetoid and Reddit.
    :param dataset_name: Name of data set
    :return: Tuple of dataset and extracted graph
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)

    if dataset_name == 'cora_full':
        dataset = CoraFull(path, T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())

    print(f"Loading data set {dataset_name} from: ", path)
    data = dataset[0]  # Extract graph
    return dataset, data


def run_experiment(args):
    """
    Performing experiment
    :param args:
    :return:
    """
    dataset, data = load_data(args.dataset)  # Todo: change specification of data set maybe without args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define Model
    encoder = create_encoder(args.model, dataset.num_features, args.latent_dim).to(device)
    decoder = create_decoder(args.decoder, args.lsh).to(device)

    if args.model == 'GAE':
        model = GAE(encoder=encoder, decoder=decoder).to(device)
    else:
        model = VGAE(encoder=encoder, decoder=decoder).to(device)

    # Split edges of a torch_geometric.data.Data object into pos negative train/val/test edges
    # default ratios of positive edges: val_ratio=0.05, test_ratio=0.1
    data.train_mask = data.val_mask = data.test_mask = data.y = None  # TODO See if necessary or why
    data = model.split_edges(data)
    node_features, train_pos_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train_epoch(epoch):
        """
        Performing training over a single epoch and optimize over loss
        :return: log - loss of training loss
        """
        # Todo: Add logging of results

        model.train()
        optimizer.zero_grad()
        # Compute latent embedding Z
        latent_embeddings = model.encode(node_features, train_pos_edge_index)

        # Calculate loss and
        loss = model.recon_loss(latent_embeddings, train_pos_edge_index)
        if args.model in ['VGAE']:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()

        t = time.time()
        k = model.decoder.forward_all(latent_embeddings, sigmoid=True)
        print(f"Computing full graph took {time.time() - t} milliseconds.")

        # Compute gradients
        loss.backward()
        # Perform optimization step
        optimizer.step()

        # print("Train-Epoch: {} Loss: {}".format(epoch, loss))

        # ToDo: Add logging via Tensorboard
        log = {
            'loss': loss
        }

        return log

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            # compute latent var
            z = model.encode(node_features, train_pos_edge_index)

        # model.test return - AUC, AP
        return model.test(z, pos_edge_index, neg_edge_index)

    # Training routine
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)
    logs = []

    for epoch in range(1, args.epochs):
        log = train_epoch(epoch)
        logs.append(log)

        # Validation metrics
        val_auc, val_ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
        print('Validation-Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, val_auc, val_ap))

        # Stop training if validation scores have not improved
        early_stopping(val_ap, model)
        if early_stopping.early_stop:
            print("Applying early-stopping")
            break

    # Training is finished, calculate test metrics
    test_auc, test_ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Test Results: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, test_auc, test_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    # Dataset
    parser.add_argument('--dataset', type=str, default='PubMed', help="Data Set Name")

    # Training
    parser.add_argument('--epochs', type=int, default=500, help="Number of Epochs in Training")
    parser.add_argument('--lr', type=int, default=0.001, help="Learning Rate")
    # Early Stopping
    parser.add_argument('--use_early_stopping', default="True")
    parser.add_argument('--early_stopping_patience', type=int, default=100)

    # Model Specific
    parser.add_argument('--model', type=str, default='VGAE', help="Specify Model Type", choices=['gae', 'vgae'])
    parser.add_argument('--latent-dim', type=int, default=16, help="Size of latent embedding.")
    parser.add_argument('--lsh', action='store_true', default=False, help="Use Local-Sensitivity-Hashing")
    parser.add_argument('--decoder', type=str, default='dot', help="Specify Decoder Type")

    args = parser.parse_args()
    run_experiment(args)
