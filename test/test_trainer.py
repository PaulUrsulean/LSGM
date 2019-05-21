import unittest

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset

from src.config import generate_config
from src.model.modules import MLPEncoder, RNNDecoder, MLPDecoder, CNNEncoder
from src.trainer import Trainer


class MyTestCase(unittest.TestCase):

    def test_run_epoch(self):
        n_examples = 1
        n_atoms = 5
        n_steps = 100
        n_feat = 7
        n_hid = 20
        n_edges = 3
        n_timesteps = 10

        data_loaders = dict(
            train_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat))),
            valid_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat))),
            test_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat)))
        )

        config = generate_config(n_atoms=n_atoms,
                                 n_edges=n_edges,
                                 epochs=2,
                                 use_early_stopping=True,
                                 early_stopping_patience=1,
                                 gpu_id=None,
                                 timesteps=n_timesteps,
                                 log_dir='/tmp'
                                 )

        encoder = MLPEncoder(config['timesteps'] * n_feat, n_hid, n_edges)
        decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        trainer = Trainer(encoder=encoder,
                          decoder=decoder,
                          data_loaders=data_loaders,
                          config=config)
        trainer.train()
        trainer.test()
        # No errors thrown

    def test_evaluation(self):
        n_examples = 30
        n_atoms = 5
        n_steps = 30
        n_feat = 7
        n_hid = 20
        n_timesteps = 6
        n_edges = 3

        data_loaders = dict(
            train_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_feat))),
            valid_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_feat))),
            test_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, 30, n_feat)))
        )

        config = generate_config(n_atoms=n_atoms,
                                 n_edges=n_edges,
                                 epochs=2,
                                 use_early_stopping=True,
                                 early_stopping_patience=1,
                                 gpu_id=None,
                                 timesteps=n_timesteps,
                                 log_dir='/tmp'
                                 )

        encoder = MLPEncoder(config['timesteps'] * n_feat, n_hid, n_edges)
        decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        trainer = Trainer(encoder=encoder,
                          decoder=decoder,
                          data_loaders=data_loaders,
                          config=config)
        trainer.test()
        # No errors thrown

    def test_overfit_epoch(self):
        n_examples = 1
        n_atoms = 3
        n_steps = 50
        n_feat = 5
        n_hid = 100
        n_edges = 2
        n_epochs = 500
        n_timesteps = 10

        data_loaders = dict(
            train_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_feat))),
            valid_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_feat))),
            test_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat)))
        )

        config = generate_config(n_atoms=n_atoms,
                                 n_edges=n_edges,
                                 epochs=n_epochs,
                                 use_early_stopping=False,
                                 early_stopping_patience=2,
                                 gpu_id=None,
                                 log_dir='/tmp',
                                 timesteps=n_timesteps
                                 )

        encoder = MLPEncoder(config['timesteps'] * n_feat, n_hid, n_edges)
        # encoder = CNNEncoder(n_feat, n_hid, n_edges)
        decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        trainer = Trainer(encoder=encoder,
                          decoder=decoder,
                          data_loaders=data_loaders,
                          config=config)

        history = trainer.train()
        last_log = history[-1]

        # Assert training loss smaller than validation loss
        self.assertLess(last_log['loss'], last_log['val_loss'])
        # Assert validation mse loss increased in second half of training
        self.assertGreater(last_log['val_mse_loss'], history[n_epochs // 2]['val_mse_loss'])


if __name__ == '__main__':
    unittest.main()
