# Heavily inspired by https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from src.model.graph_operations import gumbel, gen_fully_connected
from src.model.modules import RNNDecoder


class TrainConfig:
    def __init__(self):
        self.gpu_id = None
        self.temp = 0.1  # TODO
        self.hard = True
        self.timesteps = 100
        self.prediction_steps = 1
        self.beta = 1.0
        self.log_step = 10


class Trainer:

    def __init__(self, encoder, decoder, nll_loss,
                 kl_loss,
                 data_loaders, metrics, optimizer, lr_scheduler, config: TrainConfig):
        self.encoder = encoder
        self.decoder = decoder
        self.nll_loss = nll_loss
        self.kl_loss = kl_loss
        self.data_loader = data_loaders
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.config = config
        self.device = torch.device('cpu') if config.gpu_id is None else torch.device(f'cuda:{config.gpu_id}')

        self.train_loader = data_loaders['train_loader']
        self.valid_loader = data_loaders['valid_loader']
        self.test_loader = data_loaders['test_loader']

        self.do_validation = True

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
        #        self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    # TODO: encoder.cuda() et.

    def _train_epoch(self, epoch):
        """
        Metrics part partly taken from https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
        :param epoch:
        :return:
        """
        self.encoder.train()
        self.decoder.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_id, batch in enumerate(self.train_loader):

            batch = batch[0].to(self.device)
            batch = Variable(batch)

            rel_rec, rel_send = gen_fully_connected(batch.size(1))

            # TODO: rel-rec etc. not named, move to gpu

            self.optimizer.zero_grad()

            logits = self.encoder(batch, rel_rec, rel_send)  # TODO Assumes fully connected
            edges = F.gumbel_softmax(logits, tau=self.config.temp, hard=self.config.hard)
            prob = F.softmax(edges)  # my_softmax(edges, -1)#TODO

            if isinstance(self.decoder, RNNDecoder):
                output = self.decoder(batch, edges, 1,  # TODO 100?
                                      burn_in=False,
                                      burn_in_steps=self.config.timesteps - self.config.prediction_steps)
            else:
                raise NotImplementedError()

            ground_truth = batch[:, :, 1:, :]

            loss = self.nll_loss(output, ground_truth) \
                   + self.config.beta * self.kl_loss(prob)  # TODO why args.var in impl?

            loss.backward()
            self.optimizer.step()

            #    self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_id)
            #    self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, ground_truth)

            # if batch_id % self.config.log_step == 0:
            #    self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            #        epoch,
            #        batch_id * self.data_loader.batch_size,
            #        self.data_loader.n_samples,
            #        100.0 * batch_id / len(self.data_loader),
            #        loss.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        # if self.do_validation:
        #    val_log = self._val_loss(epoch)
        # log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log
