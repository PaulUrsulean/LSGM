import unittest

import torch
import numpy as np
import src.model.losses as losses
from torch import Tensor


class TestLoss(unittest.TestCase):

    def __init__(self, args):
        super().__init__(args)

        self.predictions = Tensor([[1.0], [0.0]])
        self.targets = Tensor([[1.0], [1.0]])
        self.edge_probs = Tensor([1.0, 0.5])
        self.n_atoms = 2
        self.n_edge_types = 2
        self.prediction_variance = 5e-5

    def test_loss_uniform_prior(self):
        loss, nll, kl = losses.vae_loss(predictions=self.predictions,
                               targets=self.targets,
                               edge_probs=self.edge_probs,
                               n_atoms=self.n_atoms,
                               n_edge_types=self.n_edge_types,
                               prediction_variance=self.prediction_variance,
                               log_prior=None)

        correct_loss = losses.nll_gaussian(self.predictions, self.targets, self.prediction_variance) \
                       + losses.kl_categorical_uniform(self.edge_probs, self.n_atoms, self.n_edge_types)
        self.assertEqual(loss, correct_loss)

    def test_loss_manual_prior(self):
        prior = np.log([.4, .6])
        loss, nll, kl = losses.vae_loss(predictions=self.predictions,
                               targets=self.targets,
                               edge_probs=self.edge_probs,
                               n_atoms=self.n_atoms,
                               n_edge_types=self.n_edge_types,
                               prediction_variance=self.prediction_variance,
                               log_prior=prior)

        correct_loss = losses.nll_gaussian(self.predictions, self.targets, self.prediction_variance) \
                       + losses.kl_categorical(self.edge_probs, log_prior=Tensor(prior), num_atoms=self.n_atoms)
        self.assertEqual(loss, correct_loss)


if __name__ == '__main__':
    unittest.main()
