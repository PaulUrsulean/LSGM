# Heavily inspired by https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
import json
import logging
import os
import time
from math import inf
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.logger import WriterTensorboardX, setup_logging
from src.model import losses
from src.model.modules import RNNDecoder
from src.model.utils import gen_fully_connected, my_softmax, nll, kl, load_models


class Model:

    def __init__(self, encoder, decoder,
                 data_loaders,
                 config):

        self.parse_config(config)
        self.config = config

        # Data Loaders
        self.data_loader = data_loaders
        self.train_loader = data_loaders['train_loader']
        self.valid_loader = data_loaders['valid_loader']
        self.test_loader = data_loaders['test_loader']

        self.metrics = [F.mse_loss, nll, kl]

        self.optimizer = torch.optim.Adam(lr=self.learning_rate,
                                          betas=self.learning_betas,
                                          params=list(encoder.parameters()) + list(decoder.parameters()))

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=self.scheduler_stepsize,
                                                            gamma=self.scheduler_gamma)

        self.device = torch.device('cpu') if self.gpu_id is None else torch.device(f'cuda:{self.gpu_id}')

        # Move models to gpu
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.do_validation = True

        # Create unique foldername for current training run and save all there
        self.setup_save_dir(config)

        # Logging config
        setup_logging(self.log_path, self.logger_config_path)
        self.logger = logging.getLogger('trainer')
        self.writer = WriterTensorboardX(self.log_path, self.logger, True)

        # Early stopping behaviour
        assert (self.early_stopping_mode in ['min', 'max'])
        self.mnt_best = inf if self.early_stopping_mode == 'min' else -inf

        # Parse Config and set model attributes
        self.rel_rec, self.rel_send = gen_fully_connected(self.n_atoms
                                                          , device=self.device)

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
        dataset = config['data']['name']
        n_features = config['data'][dataset]['dims']
        self.n_atoms = config['data'][dataset]['atoms']

        self.random_seed = config['globals']['seed']
        self.logger_config_path = config['logging']['logger_config']
        self.early_stopping_mode = config['training']['early_stopping_mode']
        self.use_early_stopping = config['training']['use_early_stopping']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.early_stopping_metric = config['training']['early_stopping_metric']
        self.learning_rate = config['training']['optimizer']['learning_rate']
        self.learning_betas = config['training']['optimizer']['betas']
        self.scheduler_stepsize = config['training']['scheduler']['stepsize']
        self.scheduler_gamma = config['training']['scheduler']['gamma']
        self.gpu_id = config["training"]["gpu_id"]
        self.do_save_models = config['logging']['store_models']
        self.timesteps = config['data']['timesteps']
        self.prediction_steps = config['model']['prediction_steps']
        self.burn_in = config['model']['burn_in']
        self.temp = config['model']['temp']
        self.sample_hard = config['model']['hard']

        self.n_edge_types = config['model']['n_edge_types']

        self.log_prior = config['globals']['prior']  # TODO
        self.add_const = config['globals']['add_const']
        self.eps = config['globals']['eps']
        self.loss_beta = config['loss']['beta']
        self.prediction_var = config['model']['decoder']['prediction_variance']

        self.epochs = config['training']['epochs']
        self.dynamic_graph = config['model']['dynamic_graph']

        self.log_step = config['logging']['log_step']

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
        encoder_path = self.models_log_path / f'encoder_epoch{epoch}.pt'
        decoder_path = self.models_log_path / f'decoder_epoch{epoch}.pt'

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

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

            if batch.size(2) > self.timesteps:
                # In case more timesteps are available, clip to avaid errors with dimensions
                batch = batch[:, :, :self.timesteps, :]

            self.rel_rec, self.rel_send = gen_fully_connected(self.n_atoms
                                                              , device=self.device)

            self.optimizer.zero_grad()

            logits = self.encoder(batch, self.rel_rec, self.rel_send)
            edges = gumbel_softmax(logits, tau=self.temp, hard=self.sample_hard)
            prob = my_softmax(logits, -1)

            if isinstance(self.decoder, RNNDecoder):
                output = self.decoder(batch, edges,
                                      pred_steps=self.prediction_steps,  # In implementation hard-coded 100
                                      burn_in=True,
                                      burn_in_steps=self.timesteps - self.prediction_steps)
            else:
                output = self.decoder(batch,
                                      rel_type=edges,
                                      rel_rec=self.rel_rec,
                                      rel_send=self.rel_send,
                                      pred_steps=self.prediction_steps
                                      )

            ground_truth = batch[:, :, 1:, :]

            loss, nll, kl = losses.vae_loss(predictions=output,
                                            targets=ground_truth,
                                            edge_probs=prob,
                                            n_atoms=self.n_atoms,
                                            n_edge_types=self.n_edge_types,
                                            log_prior=self.log_prior,
                                            add_const=self.add_const,
                                            eps=self.eps,
                                            beta=self.loss_beta,
                                            prediction_variance=self.config['model']['decoder']['prediction_variance'])
            if torch.isnan(loss).any().__bool__():
                self.logger.warn("Loss NAN")
                self.logger.warn(nll.item())
            loss.backward()
            self.optimizer.step()

            # Tensorboard writer
            self.writer.set_step(epoch * len(self.train_loader) + batch_id)
            self.writer.add_scalar('loss', loss.item())

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, ground_truth, nll=nll, kl=kl)

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
                self.encoder, self.decoder = load_models(self.encoder,
                                                         self.decoder,
                                                         config={
                                                             'training': {
                                                                 'load_path': os.path.join(self.log_path, "config.json")}})
                self.encoder = self.encoder.to(self.device)
                self.decoder = self.decoder.to(self.device)
            except FileNotFoundError:
                self.logger.debug("No models stored yet, test with models in memory.")

        self.encoder.eval()
        self.decoder.eval()

        test_loss = 0.0
        tot_mse = 0.0
        total_test_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_id, (data) in enumerate(self.test_loader):
                data = data[0].to(self.device)
                batch = Variable(data)

                assert (data.size(2) - self.timesteps >= self.timesteps)

                data_encoder = data[:, :, :self.timesteps, :].contiguous()
                data_decoder = data[:, :, -self.timesteps:, :].contiguous()

                logits = self.encoder(data_encoder, self.rel_rec, self.rel_send)
                edges = F.gumbel_softmax(logits, tau=self.temp, hard=True)
                prob = my_softmax(logits, -1)

                output = self.decoder(data_decoder,
                                      rel_type=edges,
                                      rel_rec=self.rel_rec,
                                      rel_send=self.rel_send,
                                      pred_steps=1)

                ground_truth = data_decoder[:, :, 1:, :]

                loss, nll, kl = losses.vae_loss(predictions=output,
                                                targets=ground_truth,
                                                edge_probs=prob,
                                                n_atoms=self.n_atoms,
                                                n_edge_types=self.n_edge_types,
                                                log_prior=self.log_prior,
                                                add_const=self.add_const,
                                                eps=self.eps,
                                                beta=self.loss_beta,
                                                prediction_variance=self.prediction_var)

                test_loss += loss.item()
                total_test_metrics += self._eval_metrics(output, ground_truth, nll=nll, kl=kl, log=False)

                # For plotting purposes
                if isinstance(self.decoder, RNNDecoder):
                    if self.dynamic_graph:
                        # Only makes sense when time-series is long enough
                        output = self.decoder(data, edges, self.rel_rec, self.rel_send, 100,
                                              burn_in=True, burn_in_steps=self.timesteps,
                                              dynamic_graph=True, encoder=self.encoder,
                                              temp=self.temp)
                    else:
                        output = self.decoder(data, edges, self.rel_rec, self.rel_send, 100,
                                              burn_in=True, burn_in_steps=self.timesteps)
                    output = output[:, :, self.timesteps:self.timesteps + 21, :]
                    target = data[:, :, self.timesteps:self.timesteps + 21, :]
                    # TODO: In paper, why? Why second one negative
                    # output = output[:, :, args.timesteps:, :]
                    # target = data[:, :, -args.timesteps:, :]
                    #
                else:
                    data_plot = data[:, :, self.timesteps:self.timesteps + 21,
                                :].contiguous()
                    output = self.decoder(data_plot, edges, self.rel_rec, self.rel_send,
                                          20)  # 20 in paper imp
                    target = data_plot[:, :, 1:, :]

                mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
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
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_id, (data) in enumerate(self.valid_loader):
                data = data[0].to(self.device)
                batch = Variable(data)

                if data.size(2) > self.timesteps:
                    # In case more timesteps are available, clip to avaid errors with dimensions
                    data = data[:, :, :self.timesteps, :]

                logits = self.encoder(data, self.rel_rec, self.rel_send)
                edges = F.gumbel_softmax(logits, tau=self.temp, hard=True)
                prob = my_softmax(logits, -1)

                # validation output uses teacher forcing
                output = self.decoder(data, edges, self.rel_rec, self.rel_send, 1)

                ground_truth = data[:, :, 1:, :]

                loss, nll, kl = losses.vae_loss(predictions=output,
                                                targets=ground_truth,
                                                edge_probs=prob,
                                                n_atoms=self.n_atoms,
                                                n_edge_types=self.n_edge_types,
                                                log_prior=self.log_prior,
                                                add_const=self.add_const,
                                                eps=self.eps,
                                                beta=self.loss_beta,
                                                prediction_variance=self.prediction_var)

                # Tensorboard
                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_id, 'val')
                self.writer.add_scalar('loss', loss.item())

                total_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, ground_truth, nll=nll, kl=kl)

        return {
            'val_loss': total_loss / len(self.valid_loader),
            'val_metrics': (total_val_metrics / len(self.valid_loader)).tolist()
        }

    def train(self):
        """
        Full training logic
        (Taken from https://github.com/victoresque/pytorch-template and modified)
        """
        self.encoder.train()
        self.decoder.train()

        logs = []

        for epoch in range(0, self.epochs):
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

    """
        def _extract_latent_graphs(self):
            self.encoder.eval()
    
            all_edges = []
            n_atoms = self.n_atoms
            edge_types = self.config['edge_types']
            sum_edges = Variable(torch.zeros(n_atoms, n_atoms, edge_types))
    
            with torch.no_grad():
                for batch_id, (data) in enumerate(self.data_loader):
                    data = data[0].to(self.device)
    
                    self.rel_rec, self.rel_send = gen_fully_connected(data.size(1))
    
                    logits = self.encoder(data, self.rel_rec, self.rel_send)
                    edges = F.gumbel_softmax(logits, tau=self.temp, hard=True)
    
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
    
    """
