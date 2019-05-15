# Heavily inspired by https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
import logging
from math import inf

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.logger import WriterTensorboardX, setup_logging
from src.model import losses
from src.model.graph_operations import gen_fully_connected
from src.model.modules import RNNDecoder


def generate_config(n_edges, n_atoms, *args, **kwargs):
    config = dict(
        edge_types=n_edges,
        n_atoms=n_atoms,

        epochs=30,

        use_early_stopping=True,
        early_stopping_patience=1,
        early_stopping_mode='min',  # in ["min", "max"]
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

        logger_config=".",

        pred_steps=1
    )

    # Override with manually set values
    for key, value in kwargs.items():
        config[key] = value
    return config


class Trainer:

    def __init__(self, encoder, decoder,
                 data_loaders,
                 config, nll_loss=None,
                 metrics=[],
                 kl_loss=None):
        self.encoder = encoder
        self.decoder = decoder
        self.nll_loss = nll_loss
        self.kl_loss = kl_loss
        self.data_loader = data_loaders
        self.metrics = metrics

        self.optimizer = torch.optim.Adam(params=list(encoder.parameters()) + list(decoder.parameters()))
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 200)  # TODO Hyperparameter

        if self.nll_loss is None:
            self.nll_loss = losses.nll_gaussian(.1)

        if self.kl_loss is None:
            self.kl_loss = losses.kl_categorical_uniform(config['n_atoms'], config['edge_types'])

        self.config = config
        self.device = torch.device('cpu') if config['gpu_id'] is None else torch.device(f'cuda:{config["gpu_id"]}')

        self.train_loader = data_loaders['train_loader']
        self.valid_loader = data_loaders['valid_loader']
        self.test_loader = data_loaders['test_loader']

        self.do_validation = True

        self.rel_rec, self.rel_send = None, None

        setup_logging(config['log_dir'], config['logger_config'])
        self.logger = logging.getLogger('trainer')
        self.writer = WriterTensorboardX(config['log_dir'], self.logger, True)

        assert (config['early_stopping_mode'] in ['min', 'max'])
        self.mnt_best = inf if config['early_stopping_mode'] == 'min' else -inf

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
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
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

            self.rel_rec, self.rel_send = gen_fully_connected(batch.size(1))

            # TODO: rel-rec etc. not named, move to gpu

            self.optimizer.zero_grad()

            logits = self.encoder(batch, self.rel_rec, self.rel_send)  # TODO Assumes fully connected
            edges = F.gumbel_softmax(logits, tau=self.config['temp'], hard=self.config['hard'])
            prob = F.softmax(edges)  # my_softmax(edges, -1)#TODO

            if isinstance(self.decoder, RNNDecoder):
                output = self.decoder(batch, edges,
                                      pred_steps=self.config['pred_steps'],  # TODO 100?
                                      burn_in=self.config['burn_in'],
                                      burn_in_steps=self.config["timesteps"] - self.config["prediction_steps"])
            else:
                raise NotImplementedError()

            ground_truth = batch[:, :, 1:, :]

            loss = self.nll_loss(output, ground_truth) \
                   + self.config['beta'] * self.kl_loss(prob)  # TODO why args.var in impl?

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_loader) + batch_id)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, ground_truth)

            if batch_id % self.config['log_step'] == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_id * self.train_loader.batch_size,
                    len(self.train_loader.dataset),
                    100.0 * batch_id / len(self.train_loader),
                    loss.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        val_log = self._val_loss(epoch)
        log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _val_loss(self, epoch):
        """
        Calculate validation loss
        :param epoch:
        :return:
        """
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_id, (data) in enumerate(self.valid_loader):
                data = data[0].to(self.device)

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
                total_val_metrics += self._eval_metrics(output, ground_truth)

                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_id, 'valid')
                self.writer.add_scalar('loss', loss.item())

        return {
            'val_loss': total_loss / len(self.valid_loader),
            'val_metrics': (total_val_metrics / len(self.valid_loader)).tolist()
        }

    def train(self):
        """
        Full training logic
        (Taken from https://github.com/victoresque/pytorch-template and modified)
        """
        for epoch in range(0, self.config['epochs']):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.config['use_early_stopping']:
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.config['early_stopping_mode'] == 'min' and log[
                        self.config['early_stopping_metric']] <= self.mnt_best) or \
                               (self.config['early_stopping_mode'] == 'max' and log[
                                   self.config['early_stopping_metric']] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(
                        self.config['early_stopping_metric']))
                    self.config['use_early_stopping'] = False
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.config['early_stopping_metric']]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.config['early_stopping_patience']:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(not_improved_count))
                    break
