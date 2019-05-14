import unittest

import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.optim import lr_scheduler
import numpy as np

from src.model.graph_operations import gen_fully_connected
from src.model.modules import MLPEncoder, RNNDecoder
from src.trainer import Trainer
from src.model import losses

from sklearn.metrics import mean_squared_error

from src.trainer.trainer import TrainConfig


class MyTestCase(unittest.TestCase):

    def test_run_epoch(self):
        n_examples = 99
        n_atoms = 5
        n_steps = 10
        n_feat = 7
        n_hid = 20
        n_edges = 3

        encoder = MLPEncoder(n_steps * n_feat, n_hid, n_edges)
        decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        data_loaders = dict(
            train_loader= data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat))),
            valid_loader = data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat))),
            test_loader = data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat)))
        )

        opt = torch.optim.Adam(params=list(encoder.parameters()) + list(decoder.parameters()))
        scheduler = lr_scheduler.StepLR(opt, 200)

        trainer = Trainer(encoder=encoder,
                          decoder=decoder,
                          nll_loss=losses.nll_gaussian(.1),  # TODO
                          kl_loss=losses.kl_categorical_uniform(n_atoms, n_edges),
                          data_loaders=data_loaders,
                          metrics=[],
                          optimizer=opt,
                          lr_scheduler=scheduler,
                          config=TrainConfig())
        for i in range(5):
            log = trainer._train_epoch(1)
            print(log)

    def test_overfit_epoch(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
