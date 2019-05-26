import unittest

import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset

from src.config import generate_config
from src.model import Model
from src.model.modules import MLPEncoder, RNNDecoder


class MyTestCase(unittest.TestCase):

    def test_run_epoch(self):
        n_examples = 1
        n_atoms = 5
        n_steps = 100
        n_feat = 7
        n_hid = 60
        n_edges = 3
        n_timesteps = 10

        data_loaders = dict(
            train_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat))),
            valid_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat))),
            test_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat)))
        )

        config = generate_config(n_atoms=n_atoms,
                                 n_edges=n_edges
                                 )
        config['data']['timesteps'] = n_timesteps
        config['training']['epochs'] = 2
        config['training']['use_early_stopping'] = True
        config['training']['early_stopping_patience'] = 1
        config['training']['gpu_id'] = None
        config['logging']['log_dir'] = '/tmp'

        encoder = MLPEncoder(n_timesteps * n_feat, n_hid, n_edges)
        decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        trainer = Model(encoder=encoder,
                        decoder=decoder,
                        data_loaders=data_loaders,
                        config=config)
        trainer.train()
        trainer.test()
        # No errors thrown

    def test_overfit_epoch(self):
        n_examples = 1
        n_atoms = 2
        n_steps = 20
        n_feat = 1
        n_hid = 20
        n_edges = 1
        n_epochs = 300
        n_timesteps = 10

        data_loaders = dict(
            train_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_feat))),
            valid_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_timesteps, n_feat))),
            test_loader=data.DataLoader(TensorDataset(torch.rand(n_examples, n_atoms, n_steps, n_feat)))
        )

        config = generate_config(n_atoms=n_atoms,
                                 n_edges=n_edges
                                 )

        config['data']['timesteps'] = n_timesteps
        config['model']['prediction_steps'] = 1
        config['training']['epochs'] = n_epochs
        config['training']['use_early_stopping'] = False
        config['training']['gpu_id'] = None
        config['logging']['log_dir'] = '/tmp'

        encoder = MLPEncoder(n_timesteps * n_feat, n_hid, n_edges)
        decoder = RNNDecoder(n_in_node=n_feat, edge_types=n_edges, n_hid=n_hid)

        trainer = Model(encoder=encoder,
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
