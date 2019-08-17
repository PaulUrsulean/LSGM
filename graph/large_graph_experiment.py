import argparse

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Embedding
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.nn.models.autoencoder import negative_sampling
from tqdm import tqdm

from graph.datasets.snap import AmazonCoPurchase
from graph.modules import CosineSimDecoder


class EmbeddingEncoder(nn.Module):
    def __init__(self, emb_dim, out_channels, n_nodes):
        super(EmbeddingEncoder, self).__init__()

        self.embedding = Embedding(num_embeddings=n_nodes, embedding_dim=emb_dim)
        self.conv1 = GCNConv(emb_dim, 2 * out_channels, cached=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logvar = GCNConv(
            2 * out_channels, out_channels, cached=False)

    def forward(self, x: Tensor, edge_index):
        x = x.to(torch.int64)
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


def train_model_and_save_embeddings(dataset, data, epochs, learning_rate, device):
    # Define Model
    encoder = EmbeddingEncoder(emb_dim=200, out_channels=64, n_nodes=dataset.num_nodes).to(device)

    decoder = CosineSimDecoder().to(device)

    model = VGAE(encoder=encoder, decoder=decoder).to(device)

    node_features, train_pos_edge_index = data.x.to(device), data.edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # data.edge_index = data.edge_index.long()

    assert data.edge_index.max().item() < dataset.num_nodes

    data_loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=10000,
                                  shuffle=False, add_self_loops=False)

    model.train()

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for data_flow in tqdm(data_loader()):
            optimizer.zero_grad()

            data_flow = data_flow.to(device)
            block = data_flow[0]
            embeddings = model.encode(node_features[block.n_id],
                                      block.edge_index)  # TODO Avoid computation of all node features!

            loss = model.recon_loss(embeddings, block.edge_index)
            loss = loss + (1 / len(block.n_id)) * model.kl_loss()

            epoch_loss += loss.item()

            # Compute gradients
            loss.backward()
            # Perform optimization step
            optimizer.step()

        z = model.encode(node_features, train_pos_edge_index)

        torch.save(z.cpu(), "large_emb.pt")

        print(f"Loss after epoch {epoch} / {epochs}: {epoch_loss}")

    return model


def run(seed: int, epochs: int, learning_rate: float, gpu_id=1):
    device = torch.device(f'cuda:{gpu_id}' if (torch.cuda.is_available() and gpu_id > 0) else 'cpu')

    # Load Amazon Data Set
    dataset = AmazonCoPurchase("../data/amazon_co")
    data = dataset[0]
    # data = train_val_test_split(data)

    model = train_model_and_save_embeddings(dataset, data, epochs=epochs, learning_rate=learning_rate, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--gpu', type=int, default=1, help="ID of GPU to use, 0 for none")

    # Training
    parser.add_argument('--epochs', type=int, default=500, help="Number of Epochs in Training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")

    args = parser.parse_args()

    run(seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.lr,
        gpu_id=args.gpu)
