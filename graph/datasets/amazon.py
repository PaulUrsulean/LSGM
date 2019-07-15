import os

import torch
import numpy as np
from torch import Tensor
from torch_geometric.data import InMemoryDataset, download_url, Data, Dataset


# TODO: Make InMemoryDataset if possible
class Amazon(Dataset):
    url = "http://snap.stanford.edu./data/bigdata/communities/"

    def __init__(self, root, transform=None, pre_transform=None):
        self.n_nodes = 334863
        super(Amazon, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        # TODO: add "com-amazon.all.dedup.cmty.txt.gz", "com-amazon.top5000.cmty.txt.gz" for community detection tasks
        return ["com-amazon.ungraph.txt.gz"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for file in self.raw_file_names:
            download_url(self.url + file, self.raw_dir)

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        edges = Tensor(np.loadtxt(self.raw_paths[0], dtype=np.long).T).long()
        self.n_nodes = torch.max(edges)
        node_features = torch.arange(self.n_nodes)
        data = Data(x=node_features, edge_index=edges)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, self.processed_paths[0])

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data
