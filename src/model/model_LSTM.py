import os
import json
import time
import logging
from math import inf
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import clip_grad_value_

from pathlib import Path
from src.logger import WriterTensorboardX, setup_logging
from src. model.utils import my_softmax, nll, kl
from src.model import losses
from src.model.utils import load_models
from src.model.utils import load_lstm_models
from src.model.modules_LSTM import RNN


class RecurrentBaseline:

    def __init__(self, rnn,
                 data_loaders,
                 config):

        self.parse_config(config)
        self.config = config

        # Data Loaders
        self.data_loader = data_loaders
        self.train_loader = data_loaders['train_loader']
        self.valid_loader = data_loaders['valid_loader']
        self.test_loader = data_loaders['test_loader']

        # For LSTM loss=nll and kl=none
        # self.metrics = [F.mse_loss, nll, kl]
        self.metrics = [F.mse_loss]

        self.optimizer = torch.optim.Adam(lr=self.learning_rate,
                                          betas=self.learning_betas,
                                          params=list(rnn.parameters()))

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=self.scheduler_stepsize,
                                                            gamma=self.scheduler_gamma)

        self.device = torch.device('cpu') if self.gpu_id is None else torch.device(f'cuda:{self.gpu_id}')

        # Move model to GPU
        self.rnn = rnn.to(self.device)

        self.do_validation = True

        # Create unique folder name for current training run and store there
        self.setup_save_dir(config)

        # Logging config
        setup_logging(self.log_path, self.logger_config_path)
        self.logger = logging.getLogger('trainer')
        self.writer = WriterTensorboardX(self.log_path, self.logger, True)

        # Early stopping behaviour
        assert (self.early_stopping_mode in ['min', 'max'])
        self.mnt_best = inf if self.early_stopping_mode == 'min' else -inf

    def setup_save_dir(self, config):
        save_dir = Path(config['logging']['log_dir'])
        exp_folder_name = time.asctime().replace(' ', '_').replace(':', '_') + str(hash(self))[:5]
        exp_folder_path = save_dir / exp_folder_name
        os.makedirs(exp_folder_path)
        self.log_path = exp_folder_path
        self.models_log_path = exp_folder_path / "models"
        self.save_dict(self.config, 'config.json', exp_folder_path)
        if self.do_save_models:
            os.makedirs(self.models_log_path)

    def parse_config(self, config):
        # dataset specific
        dataset = config['data']['name']

        n_features = config['data'][dataset]['dims']
        self.n_atoms = config['data'][dataset]['atoms']
        self.timesteps = config['data']['timesteps']

        #globals
        self.random_seed = config['globals']['seed']
        self.log_prior = config['globals']['prior']  # TODO
        self.add_const = config['globals']['add_const']
        self.eps = config['globals']['eps']

        #logging
        self.logger_config_path = config['logging']['logger_config']
        self.do_save_models = config['logging']['store_models']
        self.log_step = config['logging']['log_step']

        #Training specific
        self.clip_value = config['training']['grad_clip_value']
        self.early_stopping_mode = config['training']['early_stopping_mode']
        self.use_early_stopping = config['training']['use_early_stopping']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.early_stopping_metric = config['training']['early_stopping_metric']
        self.learning_rate = config['training']['optimizer']['learning_rate']
        self.learning_betas = config['training']['optimizer']['betas']
        self.scheduler_stepsize = config['training']['scheduler']['stepsize']
        self.scheduler_gamma = config['training']['scheduler']['gamma']
        self.gpu_id = config["training"]["gpu_id"]
        self.epochs = config['training']['epochs']

        # Model specific
        self.prediction_steps = config['model']['prediction_steps']
        self.burn_in = config['model']['burn_in']
        self.temp = config['model']['temp']
        self.sample_hard = config['model']['hard']
        self.n_edge_types = config['model']['n_edge_types']
        self.dynamic_graph = config['model']['dynamic_graph']
        self.prediction_var = config['model']['prediction_variance']
        self.hidden_dim = config['model']['hidden_dim']
        self.n_layers = config['model']['num_layers']

        # Loss specific
        self.loss_beta = config['loss']['beta']

    def save_dict(self, dict, name, save_dir):
        # Save config to file
        with open(save_dir / name, 'w') as f:
            json.dump(dict, f, indent=4)

    def _eval_metrics(self, output, target, nll=None, kl=None, log=True):
        """
        From https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
        :param nll:
        :param kl:
        :param output:
        :param target:
        :return:
        """
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            if metric.__name__ == "nll":
                acc_metrics[i] += nll
            elif metric.__name__ == 'kl':
                acc_metrics[i] += kl
            else:
                acc_metrics[i] += metric(output, target)
            if log:
                self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def save_models(self, epoch):

        rnn_path = self.models_log_path / f'rnn_epoch{epoch}.pt'
        torch.save(self.rnn.state_dict(), rnn_path)

    def _train_epoch(self, epoch):
        """
        Metrics part partly taken from https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
        :param epoch:
        :return:
        """
        # Train logic for single epoch
        self.rnn.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        # training for single batch
        for batch_id, batch in enumerate(self.train_loader):

            # for weather add if - loop :
            batch = batch[0].to(self.device)
            batch = Variable(batch)

            if batch.size(2) > self.timesteps:
                # In case more time steps  are available, clip to avoid errors with dimensions
                batch = batch[:, :, :self.timesteps, :]  # TODO Test when removed

            # set gradients to zero
            self.optimizer.zero_grad()

            output = self.rnn(inputs=batch,
                              prediction_steps= 100,         # self.prediction_steps,
                              burn_in=True,                     # for training burn_in = True
                              burn_in_steps=self.timesteps - self.prediction_steps
                              )

            ground_truth = batch[:, :, 1:, :]

            loss = losses.nll_gaussian(preds=output,
                                       target=ground_truth,
                                       variance=self.prediction_var,
                                       add_const=self.add_const       # add_const=False (default)
                                       )

            loss.backward()

            if self.clip_value is not None:
                clip_grad_value_(self.rnn.parameters(), self.clip_value)

            self.optimizer.step()

            # Tensorboard writer
            self.writer.set_step(epoch*len(self.train_loader)+batch_id) #momentane batch-nr
            self.writer.add_scalar('loss', loss.item())


            total_loss += loss.item()
            # Dummy replace afterwards kl=0
            total_metrics += self._eval_metrics(output, ground_truth)

            if batch_id % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
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

    def test(self):

        if self.do_save_models:
            try:
                self.rnn = load_lstm_models(self.rnn,
                                            config={
                                                'training': {
                                                    'load_path': os.path.join(self.log_path,
                                                                              "config_LSTM.json")}})

                self.rnn=self.rnn.to(self.device)


            except FileNotFoundError:
                self.logger.debug("No models stored yet, test with models in memory.")

        self.rnn.eval()

        test_loss = 0.0
        tot_mse = 0.0
        total_test_metrics = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_id, (data) in enumerate(self.test_loader):

                data = data[0].to(self.device)

                assert (data.size(2) - self.timesteps >= self.timesteps)

                data_rnn = data[:, :, :self.timesteps, :].contiguous()

                output = self.rnn(inputs=data_rnn,
                                  prediction_steps= 1 # self.prediction_steps,
                                  )

                ground_truth = data_rnn[:, :, 1:, :]

                loss = losses.nll_gaussian(preds=output,
                                           target=ground_truth,
                                           variance=self.prediction_var,
                                           add_const=self.add_const
                                           )

                test_loss += loss.item()
                total_test_metrics += self._eval_metrics(output, ground_truth, log=False)


                output = output[:, :, self.timesteps:self.timesteps + 20, :]
                target = data[:, :, self.timesteps + 1:self.timesteps + 21, :]

                # For plotting purposes
                output = self.rnn(data, 100, burn_in=True,
                               burn_in_steps=self.timesteps)

                output = output[:, :, self.timesteps:self.timesteps + 20, :]
                ground_truth = data[:, :, self.timesteps + 1:self.timesteps + 21, :]


                mse = ((ground_truth - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
                tot_mse += mse.data.cpu().numpy()

        res = {
            'test_loss': test_loss / len(self.test_loader),
            'test_full_loss': list(float(f) for f in (tot_mse / len(self.test_loader))),
            'test_metrics': (total_test_metrics / len(self.test_loader)).tolist()
        }

        # Tidy up
        log = {}
        for key, value in res.items():
            if key == 'test_metrics':
                log.update({'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
            else:
                log[key] = value

        self.logger.debug(res)
        self.save_dict(log, 'test.json', self.log_path)

        return log


    def _val_loss(self, epoch):
        """
        Calculate validation loss
        :param epoch:
        :return:
        """
        self.rnn.eval()

        total_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_id, (data) in enumerate(self.valid_loader):
                # add for weather if-loop
                data = data[0].to(self.device)

                if data.size(2) > self.timesteps:
                    # In case more timestep are available, clip to avaid errors with dimensions
                    data = data[:, :, :self.timesteps, :]

                output = self.rnn(inputs=data,
                                  prediction_steps=1         # self.prediction_steps,  see LSTM-baseline.py --> hard coded to 1
                                  # burn_in=self.burn_in, # For validation burn_in=false ; leave on default
                                  # burn_in_steps=self.timesteps - self.prediction_steps # burn_in_steps = 1 --> left on default
                                  )
                ground_truth= data[:, :, 1:, :]

                loss = losses.nll_gaussian(preds=output,
                                           target=ground_truth,
                                           variance=self.prediction_var,
                                           add_const=self.add_const
                                           )

                # Tensorboard
                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_id, 'val')
                self.writer.add_scalar('loss', loss.item())

                total_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, ground_truth)

        return {
            'val_loss': total_loss / len(self.valid_loader),
            'val_metrics': (total_val_metrics / len(self.valid_loader)).tolist()
        }

    def train(self):
        """
        Full training logic
        (Taken from https://github.com/victoresque/pytorch-template and modified)
        """
        self.rnn.train()

        logs = []

        for epoch in range(0, self.epochs):

            # return dict
            # results: {train_log: {total_loss, metrics: {mse_loss, nll, kl=None}},
            #               , val_log: {val_loss, val_metrics{F.mse_loss, nll, kl=None}}}
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            logs.append(log)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.use_early_stopping:
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.early_stopping_mode == 'min' and log[
                        self.early_stopping_metric] <= self.mnt_best) or \
                               (self.early_stopping_mode == 'max' and log[
                                   self.early_stopping_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(
                        self.early_stopping_metric))
                    self.use_early_stopping = False
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.early_stopping_metric]
                    not_improved_count = 0
                    best = True
                    if self.do_save_models:
                        self.save_models(epoch)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stopping_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(not_improved_count))
                    break
            else:
                if self.do_save_models and epoch % self.log_step == 0:
                    self.save_models(epoch)

        return logs
