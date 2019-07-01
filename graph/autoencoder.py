"""
TAKEN FROM https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE, VGAE, InnerProductDecoder

from graph.modules import CosineSimDecoder, EuclideanDistanceDecoder

torch.manual_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
parser.add_argument('--decoder', type=str, default='inner')
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()
assert args.model in ['GAE', 'VGAE']
assert args.decoder in ['inner', 'cosine', 'euclid']
assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
kwargs = {'GAE': GAE, 'VGAE': VGAE}
decoders = {'inner': InnerProductDecoder, 'cosine': CosineSimDecoder, 'euclid': EuclideanDistanceDecoder}

path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
dataset = Planetoid(path, args.dataset, T.NormalizeFeatures())
data = dataset[0]


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


channels = 16
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = decoders[args.decoder]()
print(f"Using {decoder._get_name()}")
model = kwargs[args.model](Encoder(dataset.num_features, channels), decoder=decoder).to(dev)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = model.split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


auc_list = []
for epoch in range(1, 401):
    train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    auc_list.append(auc)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

print(f"Last AUC: {auc_list[-1]}")
