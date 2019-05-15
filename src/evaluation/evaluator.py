# Heavily inspired by https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.model import losses
from src.model.graph_operations import gen_fully_connected
from src.model.modules import RNNDecoder


class Evaluator:
    def __init__(self, encoder, decoder,
                 data_loader, config,
                 metrics=[],
                 nll_loss=None,
                 kl_loss=None
                 ):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.nll_loss = nll_loss
        self.kl_loss = kl_loss

        if self.nll_loss is None:
            self.nll_loss = losses.nll_gaussian(.1)

        if self.kl_loss is None:
            self.kl_loss = losses.kl_categorical_uniform(config['n_atoms'], config['edge_types'])

        self.data_loader = data_loader
        self.metrics = metrics

        self.device = torch.device('cpu') if config['gpu_id'] is None else torch.device(f'cuda:{config["gpu_id"]}')

        self.data_loader = data_loader

        self.rel_rec, self.rel_send = None, None

        # setup_logging(config['log_dir'], config['logger_config']) TODO
        self.logger = logging.getLogger('tester')

    def _eval_metrics(self, output, target):
        """
        From https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
        :param output:
        :param target:
        :return:
        """
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def test(self):
        """
        Calculate test loss
        :return:
        """
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_id, (data) in enumerate(self.data_loader):
                data = data[0].to(self.device)

                self.rel_rec, self.rel_send = gen_fully_connected(data.size(1))

                logits = self.encoder(data, self.rel_rec, self.rel_send)
                edges = F.gumbel_softmax(logits, tau=self.config['temp'], hard=self.config['hard'])
                prob = F.softmax(edges)

                if isinstance(self.decoder, RNNDecoder):
                    output = self.decoder(data, edges,
                                          pred_steps=self.config['pred_steps'],
                                          burn_in=self.config['burn_in'],
                                          burn_in_steps=self.config['timesteps'] - self.config['prediction_steps'])
                else:
                    raise NotImplementedError()

                ground_truth = data[:, :, 1:, :]

                loss = self.nll_loss(output, ground_truth) \
                       + self.config['beta'] * self.kl_loss(prob)

                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, ground_truth)

        return {
            'test_loss': total_loss / len(self.data_loader),
            'test_metrics': (total_metrics / len(self.data_loader)).tolist()
        }

    def _extract_latent_graphs(self):
        """
        Calculate test loss
        :return:
        """
        self.encoder.eval()

        all_edges = []
        n_atoms = self.config['n_atoms']
        edge_types = self.config['edge_types']
        sum_edges = Variable(torch.zeros(n_atoms, n_atoms, edge_types))

        with torch.no_grad():
            for batch_id, (data) in enumerate(self.data_loader):
                data = data[0].to(self.device)

                self.rel_rec, self.rel_send = gen_fully_connected(data.size(1))

                logits = self.encoder(data, self.rel_rec, self.rel_send)
                edges = F.gumbel_softmax(logits, tau=self.config['temp'], hard=True)

                # Convert from n x (n-1) matrix to n x n matrix
                full_graph = torch.zeros(len(data), n_atoms, n_atoms, edge_types)
                edges = edges.view(-1, n_atoms, n_atoms - 1, edge_types)
                for i in range(n_atoms):
                    for j in range(n_atoms):
                        if i == j:
                            continue
                        elif i > j:
                            full_graph[:, i, j, :] = edges[:, i, j, :]
                        else:
                            full_graph[:, i, j, :] = edges[:, i, j - 1, :]


                for i, e in enumerate(edges):
                    all_edges.append(e.numpy())
                    sum_edges += full_graph[i]

        return all_edges, (sum_edges / len(self.data_loader)).numpy()
