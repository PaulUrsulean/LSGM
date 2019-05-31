import os
import json
import time
import logging

import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from src.logger import WriterTensorboardX, setup_logging
from src. model.utils import my_softmax, nll, kl
from src.model import losses

def check_nan_info_rnn(module, input, output):
    print(f"Input to encoder contained {torch.isnan(input[0].data).sum().__int__()} NaN values.")
    print(f"Output of encoder contained {torch.isnan(output.data).sum().__int__()} NaN values.")
    print(f"Input to encoder contained {torch.isnan(input[0].data).nonzero()} NaN values.")
    print(f"Output of encoder contained {torch.isnan(output.data).nonzero()} NaN values.")

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

        self.metrics = [F.mse_loss, nll, kl]

        self.optimizer = torch.optim.Adam(lr=self.learning_rate,
                                          betas=self.learning_betas,
                                          params=list(rnn.parameters()))

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=self.scheduler_stepsize,
                                                            gamma=self.scheduler_gamma)

        self.device = torch.device('cpu') if self.gpu_id is None else torch.device(f'cuda:{self.gpu_id}')

        # Move model to GPU
        self.rnn= rnn.to(self.device)
        print(self.rnn)

        # Check for Nan
        # rnn.register_forward_hook(check_nan_info_rnn)

        self.do_validation = True

        # Create unique folder name for current training run and save all there
        self.setup_save_dir(config)

        # Logging config
        setup_logging(self.log_path, self.logger_config_path)
        self.logger = logging.getLogger('trainer')
        self.writer = WriterTensorboardX(self.log_path, self.logger, True)

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

        #dataset specific
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

    def _train_epoch(self, epoch):
        """
        Metrics part partly taken from https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py
        :param epoch:
        :return:
        """
        # Train logic for single epoch
        self.rnn.train()
        total_loss=0
        total_metrics = np.zeros(len(self.metrics))

        # training for single batch
        for batch_id, batch in enumerate(self.train_loader):
            # push batch to device
            batch=batch[0].to(self.device)
            assert (torch.isnan(batch).any().__bool__() is False)

            if batch.size(2) > self.timesteps:
                # In case more time steps  are available, clip to avoid errors with dimensions
                batch = batch[:, :, :self.timesteps, :]  # TODO Test when removed

            # set gradients to zero
            self.optimizer.zero_grad()

            output = self.rnn(inputs=batch,
                              prediction_steps=self.prediction_steps,
                              burn_in=self.burn_in,
                              burn_in_steps=self.timesteps - self.prediction_steps
                              )

            #print("after logits", logits)
                #n_in=batch,
                 #             n_hid=self.hidden_dim,
                              #n_out= ,
                   #           n_atoms=self.n_atoms,
                  #            n_layers= self.n_layers,
                    #          do_prob=0

            ground_truth= batch[: , : , 1: , :]

            loss = losses.nll_gaussian(preds=output,
                                       target=ground_truth,
                                       variance=self.prediction_var,
                                       add_const=self.add_const
                                       )
            if torch.isnan(loss.data).any().__bool__():
                print(f"NLL NaN: {torch.isnan(nll.data).any().__bool__()}")
                self.logger.debug(loss.item())
                rnn_weights=torch.cat([param.view(-1) for param in self.rnn.parameters()])
                self.logger.debug("Any rnn weights NaN")
                self.logger.debug(torch.isnan(rnn_weights.clone().cpu().any().__bool__()))

            mse= F.mse_loss(output, ground_truth)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step(epoch*len(self.train_loader)+batch_id)
            self.writer.add_scalar('loss', loss.item())
            self.writer.add_scalar('mse', mse.item())
            self.writer.add_scalar('learning_rate', self.lr_scheduler.get_lr()[-1])

            total_loss+=loss.item()
            #total_metrics+=self._eval_metrics(output)

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
            #val_log = s
            # mse = ((ground_truth - output)**2).mean(dim=0).mean(dim=0).mean(dim=-1)

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
                data = data[0].to(self.device)

                if data.size(2) > self.timesteps:
                    # In case more timesteps are available, clip to avaid errors with dimensions
                    data = data[:, :, :self.timesteps, :]

                output = self.rnn(inputs=data,
                                  prediction_steps=self.prediction_steps,
                                  burn_in=self.burn_in,
                                  burn_in_steps=self.timesteps - self.prediction_steps
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
                #total_val_metrics += self._eval_metrics(output, ground_truth, nll=nll, kl=kl)

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

        logs=[]

        for epoch in range (0, self.epochs):

            results = self._train_epoch(epoch)
            print("done with epoch result")