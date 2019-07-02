import argparse
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE

from modules import Encoder
from early_stopping import EarlyStopping


def create_encoder(num_features, channels, args):
    """
    To-do: Add description
    :param num_features:
    :param channels:
    :param args:
    :return:
    """
    encoder = Encoder(num_features, channels, args)
    return encoder


def create_decoder(args):
    """
    Creates Decoder part of Model e.g. Inner-Product;cosine; euclidean-dist
    :return:
    """
    # Todo: incorporate additional decoders
    if args.decoder == "IP":
        """
        Inner-Product Decoder: 
        return None
        """
        decoder = None

    if args.decoder == "":
        pass

    if args.decoder == "":
        pass

    return decoder


def load_data(args):
    """
    Handling loading and preprocessing of data
    :param args:
    :return:
    """
    # ToDo: Add other datasets e.g. reddit
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
    # Initialize Data Set
    dataset = Planetoid(path, args.dataset, T.NormalizeFeatures())
    # Create data object
    data = dataset[0]
    print("Loading Data Set from: ", path)

    return dataset, data


def run_experiment(args):
    """
    Performing experiment
    :param args:
    :return:
    """

    """
    Loading Data set and data object
    # Todo: change specification of data set maybe without args
    """
    dataset, data = load_data(args)
    channels = 16
    """
    Device specification
    # Todo: change specification of gpu assignment
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Todo: check for available cuda
    # assert (torch.cuda.is_available() is False, "Cuda not available - running on gpu")
    """
    Creating model components
    # To do: add components in specific function
    """
    encoder = create_encoder(dataset.num_features, channels, args)
    decoder = create_decoder(args)
    """
    Creating model: GAE or VGAE (by default)
    # Todo: change logic
    """
    model = kwargs[args.model](encoder=encoder, decoder=decoder).to(dev)
    print("Creating Model", model)
    """
    Data loading logic
    """
    # TODO See if necessary or why
    data.train_mask = data.val_mask = data.test_mask = data.y = None

    """
    Generate data set splits
    """
    # splits edges of a torch_geometric.data.Data object into pos negative train/val/test edges
    # default ratios of positive edges: val_ratio=0.05, test_ratio=0.1
    data = model.split_edges(data)

    # Train data set
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train_epoch(epoch):
        """
        Performing training over a single epoch and optimize over loss
        :return: log - loss of training loss
        """
        # Todo: Add logging of results

        model.train()
        optimizer.zero_grad()
        # Compute latent variable
        z = model.encode(x, train_pos_edge_index)
        """
        For VGAE
        Loss = loss_reconstructin + loss_kl
        """
        loss = model.recon_loss(z, train_pos_edge_index)
        if args.model in ['VGAE']:
            # VGAE.kl_loss:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()

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
            z = model.encode(x, train_pos_edge_index)

        # model.test return - AUC, AP
        return model.test(z, pos_edge_index, neg_edge_index)

    """
    Train Logic
    # Todo: customable parameters
    """
    # Initialize early-stopping object
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)
    # Create list for logging metrices
    logs = []

    for epoch in range(1, args.epochs):
        """
        Training of model
        """
        log = train_epoch(epoch)
        logs.append(log)

        """
        Validation of model
        """
        # Evaluate model on val
        val_auc, val_ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
        print('Validation-Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, val_auc, val_ap))

        """
        Early stopping
        """
        early_stopping(val_ap, model)

        if early_stopping.early_stop:
            print("Apply early-stopping")
            break

        #if early_stopping.early_stop:



    """
    Testing the model
    """
    test_auc, test_ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Test-Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, test_auc, test_ap))

    """ TODO: Do based on validation set
        Early Stopping Logic
        
        best_train_loss = inf

        did_improve = log["loss"] <= best_train_loss

        if did_improve:
            best_train_loss = log["loss"]
            not_improved_count = 0

            # To do: Add saving best model
        else:
            not_improved_count += 1

        if args.use_early_stopping and not_improved_count > args.early_stopping_patience:
            print("Model did not improve")
            break
    """


if __name__ == '__main__':
    """
    Parse Configurations
    """
    torch.manual_seed(5)

    # Todo: change parameter parsing
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='PubMed', help="Data Set Name")

    # Training
    parser.add_argument('--epochs', type=int, default=500, help="Number of Epochs in Training")
    parser.add_argument('--lr', type=int, default=0.001, help="Learning Rate")
    # Early Stopping
    parser.add_argument('--use_early_stopping', default="True")
    parser.add_argument('--early_stopping_patience', type=int, default=100)

    # Model Specific
    parser.add_argument('--model', type=str, default='VGAE', help="Specify Model Type")
    parser.add_argument('--lsh', action='store_true', default=False, help="Use Local-Sensitivity-Hashing")
    parser.add_argument('--decoder', type=str, default='IP', help="Specify Decoder Type")

    args = parser.parse_args()

    assert args.model in ['GAE', 'VGAE']
    assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
    kwargs = {'GAE': GAE, 'VGAE': VGAE}

    run_experiment(args)
