import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.nn.models.autoencoder import negative_sampling
from tqdm import tqdm

from graph.modules import CosineSimDecoder, VGAEEncoder
from graph.utils import load_data


class EmbeddingEncoder(nn.Module):
    def __init__(self, emb_dim, out_channels, n_nodes):
        super(EmbeddingEncoder, self).__init__()

        self.embedding = Embedding(num_embeddings=n_nodes, embedding_dim=emb_dim)
        self.conv1 = GCNConv(emb_dim, 2 * out_channels, cached=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logvar = GCNConv(
            2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        emb = self.embedding(x)
        x = F.relu(self.conv1(emb, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


def train_val_test_split(data):
    return data


def test(model, pos_edge_index, node_features, num_nodes):
    model.eval()
    with torch.no_grad():
        # compute latent var
        z = model.encode(node_features, pos_edge_index)
    # model.test return - AUC, AP
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=num_nodes)
    return model.test(z, pos_edge_index, neg_edge_index)


def train_model(dataset, data, epochs, learning_rate, device):
    # Define Model
    # encoder = EmbeddingEncoder(emb_dim=16, out_channels=16, n_nodes=dataset.num_nodes).to(device)
    encoder = VGAEEncoder(data.num_features, 16, cached=False)
    decoder = CosineSimDecoder().to(device)

    model = VGAE(encoder=encoder, decoder=decoder).to(device)

    node_features, train_pos_edge_index = data.x.to(device), data.edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data_loader = NeighborSampler(data, size=[25], num_hops=1, batch_size=20000,
                                  shuffle=False, add_self_loops=False)

    model.train()

    for epoch in tqdm(range(epochs)):
        for data_flow in tqdm(data_loader(data.train_mask)):
            optimizer.zero_grad()

            data_flow = data_flow.to(device)
            block = data_flow[0]
            embeddings = model.encode(node_features, block.edge_index)  # TODO Avoid computation of all node features!

            loss = model.recon_loss(embeddings, block.edge_index)
            loss = loss + (1 / data.num_nodes) * model.kl_loss()

            # Compute gradients
            loss.backward()
            # Perform optimization step
            optimizer.step()

            print(f"Loss: {loss.item()}")
            val_auc, val_ap = test(model, block.edge_index, embeddings, data.num_nodes)
            print('Validation-Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, val_auc, val_ap))

    return model


def run_experiment(seed: int, epochs: int, learning_rate: float, gpu_id=1):
    device = torch.device(f'cuda:{gpu_id}' if (torch.cuda.is_available() and gpu_id > 0) else 'cpu')

    # Load Amazon Data Set
    dataset, data = load_data('Reddit')
    # data = train_val_test_split(data)

    model = train_model(dataset, data, epochs=epochs, learning_rate=learning_rate, device=device)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--gpu', type=int, default=1, help="ID of GPU to use, 0 for none")

    # Training
    parser.add_argument('--epochs', type=int, default=500, help="Number of Epochs in Training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")

    args = parser.parse_args()

    run_experiment(seed=args.seed,
                   epochs=args.epochs,
                   learning_rate=args.lr,
                   gpu_id=args.gpu)
