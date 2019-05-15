import unittest

import torch
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset

from src.model import losses
from src.model.modules import MLPEncoder, RNNDecoder
from src.trainer import Trainer


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



        config = dict(
            edge_types=n_edges,
            n_atoms=n_atoms,

            epochs=30,

            use_early_stopping=True,
            early_stopping_patience=1,
            early_stopping_mode='min', # in ["min", "max"]
            early_stopping_metric='val_loss',

            gpu_id=None,  # or None
            log_dir='./logs_test',

            timesteps=1,  # In forecast
            prediction_steps=2,  #

            temp=2,
            hard=True,
            burn_in=False,

            beta=1,

            log_step=1,

            logger_config = ".",

            pred_steps=1
        )

        trainer = Trainer(encoder=encoder,
                          decoder=decoder,
                          data_loaders=data_loaders,
                          metrics=[losses.nll_gaussian(.1)], # What other metrics + allow full VAE loss
                          config=config)
        trainer.train()

    def test_overfit_epoch(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
