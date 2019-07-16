import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import download_url, Data, Dataset


class SnapNetwork(ABC, Dataset):
    """
    Wrapper for data sets on http://snap.stanford.edu./data/index.html that contain networks with ground truth community information.
    """

    def __init__(self, root):
        super(SnapNetwork, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

        self.num_nodes = self.get_num_nodes()
        self.num_communities = self.get_num_communities()
        self.base_url = self.get_base_url()

    @abstractmethod
    def get_num_nodes(self):
        pass

    # @abstractmethod
    def get_num_communities(self):
        pass

    def get_base_url(self):
        return "http://snap.stanford.edu./data/bigdata/communities/"

    @abstractmethod
    def get_raw_file_names(self):
        pass

    @property
    def raw_file_names(self):
        # TODO: add "com-amazon.all.dedup.cmty.txt.gz", "com-amazon.top5000.cmty.txt.gz" for community detection tasks
        return self.get_raw_file_names()

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for file in self.raw_file_names:
            download_url(self.get_base_url() + file, self.raw_dir)

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        edges = Tensor(np.loadtxt(self.raw_paths[0], dtype=np.long).T).long()
        self.n_nodes = self.get_num_nodes()
        node_features = torch.arange(self.n_nodes)
        data = Data(x=node_features, edge_index=edges)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, self.processed_paths[0])

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data


class Slashdot(SnapNetwork):
    def get_raw_file_names(self):
        return ["soc-Slashdot0902.txt.gz"]

    def get_num_nodes(self):
        return 82168


class Amazon(SnapNetwork):

    def get_num_communities(self):
        return 75149

    def get_num_nodes(self):
        return 334863

    def get_raw_file_names(self):
        return [
            "com-amazon.ungraph.txt.gz",
            "com-amazon.all.dedup.cmty.txt.gz",
            "com-amazon.top5000.cmty.txt.gz"
        ]


class Epinions(SnapNetwork):

    def get_base_url(self):
        return "http://snap.stanford.edu./data/"

    def get_num_nodes(self):
        return 75879

    def get_raw_file_names(self):
        return [
            "soc-Epinions1.txt.gz"
        ]
